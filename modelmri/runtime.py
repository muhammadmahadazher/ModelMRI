"""Model loading and streaming generation.

One ModelRuntime owns the currently loaded model + tokenizer. Generation
runs in a worker thread and yields text pieces through a
TextIteratorStreamer, so callers (REST or WebSocket) can stream tokens
as they are produced.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class ModelStatus:
    loaded: bool
    hf_id: str | None = None
    device: str | None = None
    dtype: str | None = None
    n_params: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class ModelRuntime:
    """Owns the loaded model; thread-safe load, streaming generate."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.model: AutoModelForCausalLM | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.hf_id: str | None = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def status(self) -> ModelStatus:
        if not self.loaded:
            return ModelStatus(loaded=False, device=self.device)
        return ModelStatus(
            loaded=True,
            hf_id=self.hf_id,
            device=self.device,
            dtype=str(next(self.model.parameters()).dtype).removeprefix("torch."),
            n_params=sum(p.numel() for p in self.model.parameters()),
        )

    def load(self, hf_id: str = DEFAULT_MODEL) -> ModelStatus:
        """Load a HuggingFace causal LM. Blocking — call from a worker thread."""
        with self._lock:
            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            tokenizer = AutoTokenizer.from_pretrained(hf_id)
            model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype)
            model.to(self.device)
            model.eval()
            self.tokenizer, self.model, self.hf_id = tokenizer, model, hf_id
            return self.status()

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Yield generated text pieces. Blocking iterator — consume off the event loop."""
        if not self.loaded:
            raise RuntimeError("No model loaded. POST /api/model/load first.")

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs: dict = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs["do_sample"] = False

        worker = threading.Thread(
            target=self.model.generate, kwargs=gen_kwargs, daemon=True
        )
        worker.start()
        yield from streamer
