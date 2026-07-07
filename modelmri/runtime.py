"""Model loading, streaming generation, and attention capture.

One ModelRuntime owns the currently loaded model + tokenizer. Generation
runs in a worker thread and yields text pieces through a
TextIteratorStreamer. After a generation completes, the full token
sequence is retained so attention maps can be computed on demand.

Attention capture requires eager attention: SDPA / flash attention never
materializes the attention matrix, so models are loaded with
attn_implementation="eager".
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
    """Owns the loaded model; thread-safe load, streaming generate, attention."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.model: AutoModelForCausalLM | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.hf_id: str | None = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Last completed generation (prompt + output), for attention capture.
        self.last_ids: torch.Tensor | None = None
        self._attn: list[torch.Tensor] | None = None  # per layer: [H, S, S] fp16
        self._attn_tokens: list[str] | None = None

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
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                torch_dtype=dtype,
                attn_implementation="eager",  # required to materialize attention
            )
            model.to(self.device)
            model.eval()
            self.tokenizer, self.model, self.hf_id = tokenizer, model, hf_id
            self.last_ids = None
            self._attn = None
            self._attn_tokens = None
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

        result: dict = {}

        def _generate() -> None:
            result["ids"] = self.model.generate(**gen_kwargs)

        worker = threading.Thread(target=_generate, daemon=True)
        worker.start()
        yield from streamer
        worker.join(timeout=30)

        ids = result.get("ids")
        if ids is not None:
            self.last_ids = ids[0].detach().to("cpu")
            self._attn = None  # invalidate cache; recomputed on demand
            self._attn_tokens = None

    # ---------------- attention ----------------

    def attention_meta(self) -> dict:
        """Shape info for the last generation's attention, without computing it."""
        if not self.loaded or self.last_ids is None:
            return {"available": False}
        cfg = self.model.config
        return {
            "available": True,
            "n_layers": cfg.num_hidden_layers,
            "n_heads": cfg.num_attention_heads,
            "n_tokens": int(self.last_ids.shape[0]),
        }

    def attention(self, layer: int, head: int) -> dict:
        """Token strings + [S, S] attention matrix for one layer/head.

        Computed with a single full forward pass over the last generated
        sequence; all layers are cached so switching heads is instant.
        """
        if not self.loaded or self.last_ids is None:
            raise RuntimeError("Generate something first, then inspect attention.")

        with self._lock:
            if self._attn is None:
                with torch.no_grad():
                    out = self.model(
                        self.last_ids.unsqueeze(0).to(self.device),
                        output_attentions=True,
                    )
                self._attn = [
                    a[0].detach().to(torch.float16).cpu() for a in out.attentions
                ]
                self._attn_tokens = [
                    self.tokenizer.decode([tid]) for tid in self.last_ids.tolist()
                ]

        n_layers, n_heads = len(self._attn), self._attn[0].shape[0]
        if not (0 <= layer < n_layers and 0 <= head < n_heads):
            raise ValueError(f"layer must be in [0,{n_layers}), head in [0,{n_heads})")

        matrix = self._attn[layer][head].to(torch.float32)
        return {
            "layer": layer,
            "head": head,
            "tokens": self._attn_tokens,
            "matrix": [[round(v, 4) for v in row] for row in matrix.tolist()],
        }
