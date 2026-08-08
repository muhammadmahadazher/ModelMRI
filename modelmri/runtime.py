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

from . import devices, ollama, progress
from .saes import SAEHandle, SAEStatus

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _hub_error_message(hf_id: str, err: Exception) -> str:
    """Turn a HuggingFace hub failure into one actionable sentence."""
    text = str(err)
    if "gated repo" in text or "restricted" in text or "403" in text:
        return (
            f"'{hf_id}' is a gated model: accept its license at "
            f"https://huggingface.co/{hf_id} while signed in, then run "
            f"`huggingface-cli login` so ModelMRI can download it. "
            f"Ungated alternatives: Qwen/Qwen3-0.6B, Qwen/Qwen2.5-0.5B-Instruct, gpt2."
        )
    if "not a local folder" in text or "Repository Not Found" in text or "404" in text:
        return (
            f"'{hf_id}' was not found on the HuggingFace Hub. Check the id "
            f"(it is case-sensitive and looks like 'owner/name')."
        )
    if "offline" in text.lower() or "connection" in text.lower():
        return (
            f"Could not reach the HuggingFace Hub to fetch '{hf_id}'. "
            f"Check your connection, or pick a model already cached locally."
        )
    return f"Could not load '{hf_id}': {text.splitlines()[0]}"


def _tree_bytes(root) -> int:
    try:
        return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    except OSError:
        return 0


def local_hf_models() -> list[dict]:
    """Models already in the HuggingFace cache (offline-usable)."""
    import os
    from pathlib import Path

    hub = (
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    )
    out: list[dict] = []
    if not hub.is_dir():
        return out
    for d in sorted(hub.glob("models--*")):
        parts = d.name.removeprefix("models--").split("--")
        # max, not sum -- exactly the trap progress.py documents. On Windows
        # (and anywhere symlinks are unavailable) snapshots/ holds full copies
        # of the blobs, so adding the two trees reports every model at roughly
        # double its real size, and the picker then sorts by that.
        size = max(_tree_bytes(d / "blobs"), _tree_bytes(d / "snapshots"))
        out.append({"id": "/".join(parts), "size_gb": round(size / 1e9, 2)})
    return out


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
        self.backend: str = "hf"  # "hf" (full introspection) | "ollama" (text only)
        # GPU when one is usable (NVIDIA / AMD ROCm / Intel / Apple), else CPU
        self.accel = devices.detect()
        self.device = self.accel.torch_device
        # Bumped on every load. Everything derived from a generation carries
        # the epoch it was produced under, because a model swap that lands
        # mid-generation would otherwise leave one model's token ids to be
        # interpreted by another model's weights -- which does not crash, it
        # just quietly reports numbers about nothing.
        self.epoch = 0
        # Last completed generation (prompt + output), for attention capture.
        self.last_ids: torch.Tensor | None = None
        self.last_ids_epoch = -1
        self._attn: list[torch.Tensor] | None = None  # per layer: [H, S, S] fp16
        self._attn_tokens: list[str] | None = None
        # SAE state
        self.sae: SAEHandle | None = None
        self._feats: torch.Tensor | None = None  # [S, d_sae] fp16, last generation
        self._steer: tuple[int, float] | None = None  # (feature_id, scale)

    @property
    def loaded(self) -> bool:
        return self.model is not None or bool(self.backend == "ollama" and self.hf_id)

    def accelerator(self) -> dict:
        """What we're running on, and why (surfaced in the UI)."""
        return self.accel.to_dict()

    def status(self) -> ModelStatus:
        if self.backend == "ollama" and self.hf_id:
            return ModelStatus(loaded=True, hf_id=self.hf_id, device="ollama")
        if self.model is None:
            return ModelStatus(loaded=False, device=self.device)
        return ModelStatus(
            loaded=True,
            hf_id=self.hf_id,
            device=self.device,
            dtype=str(next(self.model.parameters()).dtype).removeprefix("torch."),
            n_params=sum(p.numel() for p in self.model.parameters()),
        )

    def load(self, hf_id: str = DEFAULT_MODEL, source: str = "hf") -> ModelStatus:
        """Load a model. source="hf" (full introspection) or "ollama" (text only).

        Blocking — call from a worker thread.
        """
        if source not in ("hf", "ollama"):
            raise ValueError(f"unknown source {source!r} (use 'hf' or 'ollama')")

        if source == "ollama":
            st = ollama.status()
            if not st["up"]:
                raise RuntimeError(
                    "Ollama is not running at 127.0.0.1:11434 — start it, or "
                    "install from ollama.com"
                )
            if hf_id not in st["models"]:
                raise ValueError(
                    f"'{hf_id}' is not installed in Ollama. Installed: "
                    f"{', '.join(st['models']) or 'none'} — run `ollama pull {hf_id}`"
                )
            with self._lock:
                self.epoch += 1
                self.model = None
                self.tokenizer = None
                self.backend = "ollama"
                self.hf_id = hf_id
                self.last_ids = None
                self._attn = None
                self._attn_tokens = None
                self.sae = None
                self._feats = None
                self._steer = None
            return self.status()

        with self._lock:
            dtype = devices.torch_dtype(self.accel)
            progress.TRACKER.start(hf_id)
            try:
                tokenizer = AutoTokenizer.from_pretrained(hf_id)
                progress.TRACKER.stage("weights")
                model = AutoModelForCausalLM.from_pretrained(
                    hf_id,
                    torch_dtype=dtype,
                    attn_implementation="eager",  # materialises attention
                )
            except OSError as err:
                # Gated repos, typos and private models all land here with a
                # multi-screen traceback. Say what to actually do instead.
                message = _hub_error_message(hf_id, err)
                progress.TRACKER.finish(error=message)
                raise ValueError(message) from err
            except BaseException as err:
                progress.TRACKER.finish(error=f"{type(err).__name__}: {err}")
                raise
            progress.TRACKER.stage("device", f"moving to {self.accel.name}")
            try:
                model.to(self.device)
            except Exception as err:
                # Out of VRAM, or a driver that says yes then fails: keep the
                # tool usable instead of dying on the user's first click.
                if self.accel.kind == "cpu":
                    progress.TRACKER.finish(error=str(err))
                    raise
                self.accel = devices.detect(prefer="cpu")
                self.accel.reason = f"fell back to CPU: {type(err).__name__}: {err}"
                self.device = self.accel.torch_device
                progress.TRACKER.stage("device", "GPU rejected the model, using CPU")
                model = model.to(torch.float32).to(self.device)
            model.eval()
            progress.TRACKER.finish()
            self.epoch += 1
            self.backend = "hf"
            self.tokenizer, self.model, self.hf_id = tokenizer, model, hf_id
            self.last_ids = None
            self._attn = None
            self._attn_tokens = None
            self.sae = None
            self._feats = None
            self._steer = None
            return self.status()

    def _block(self, layer: int) -> torch.nn.Module:
        """The decoder block whose *input* is the residual stream at `layer`."""
        root = self.model
        if hasattr(root, "transformer") and hasattr(root.transformer, "h"):
            return root.transformer.h[layer]  # GPT-2 family
        if hasattr(root, "model") and hasattr(root.model, "layers"):
            return root.model.layers[layer]  # Llama/Qwen/Gemma family
        raise RuntimeError(f"Don't know how to find block {layer} in {type(root)}")

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        commit: bool = True,
    ) -> Iterator[str]:
        """Yield generated text pieces. Blocking iterator — consume off the event loop.

        commit=False runs the model without touching the analysis target. The
        steering A/B needs this: it fires two short completions to compare, and
        committing those would silently rebase last_ids onto a 24-token
        sequence while the panels are still showing a 260-token one. Nothing
        errors; the heat map just starts describing a different generation
        than the token strip above it.
        """
        if not self.loaded:
            raise RuntimeError("No model loaded. POST /api/model/load first.")
        epoch = self.epoch

        if self.backend == "ollama":
            yield from ollama.stream_generate(
                self.hf_id,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            return

        if getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt  # base models (e.g. GPT-2) have no chat template
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        # timeout: if the generate worker ever stalls or dies, the streamer
        # raises instead of blocking its consumer forever (a hang observed
        # once in the wild is a hang that must be made impossible)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=180.0
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

        steer_handle = None
        if self._steer is not None and self.sae is not None:
            fid, scale = self._steer
            direction = self.sae.steering_vector(fid).to(self.device)
            block = self._block(self.sae.layer)

            def _steer_hook(module, args):  # noqa: ANN001 - torch hook signature
                hidden = args[0]
                return (hidden + scale * direction.to(hidden.dtype),) + args[1:]

            steer_handle = block.register_forward_pre_hook(_steer_hook)

        try:
            worker = threading.Thread(target=_generate, daemon=True)
            worker.start()
            yield from streamer
            worker.join(timeout=30)
        finally:
            if steer_handle is not None:
                steer_handle.remove()

        ids = result.get("ids")
        if ids is None or not commit:
            return
        if self.epoch != epoch:
            # A load completed while this generation was streaming. These ids
            # belong to a model that is no longer here; committing them would
            # point the attention view at the wrong weights.
            return
        self.last_ids = ids[0].detach().to("cpu")
        self.last_ids_epoch = epoch
        self._attn = None  # invalidate caches; recomputed on demand
        self._attn_tokens = None
        self._feats = None

    # ---------------- attention ----------------

    def attention_meta(self) -> dict:
        """Shape info for the last generation's attention, without computing it."""
        if self.backend == "ollama":
            return {"available": False, "reason": "internals unavailable via Ollama"}
        if not self.loaded or self.last_ids is None:
            return {"available": False}
        if self.last_ids_epoch != self.epoch:
            return {"available": False, "reason": "model changed since that generation"}
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
        if self.last_ids_epoch != self.epoch:
            raise RuntimeError(
                "That generation was produced by a different model. Generate again."
            )

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

    # ---------------- SAE features ----------------

    def sae_status(self) -> SAEStatus:
        if self.sae is None:
            return SAEStatus(loaded=False)
        return self.sae.status()

    def load_sae(self, repo: str, hook: str) -> SAEStatus:
        """Load an SAE and validate it against the current model. Blocking."""
        if self.backend == "ollama":
            raise RuntimeError(
                "SAE features need model internals — unavailable via Ollama. "
                "Load a HuggingFace model instead."
            )
        if not self.loaded:
            raise RuntimeError("Load a model first.")
        sae = SAEHandle.load(repo, hook)
        d_model = self.model.config.hidden_size
        if sae.d_in != d_model:
            raise ValueError(
                f"SAE d_in={sae.d_in} does not match model hidden_size={d_model} "
                f"({self.hf_id}). This SAE was trained on a different model."
            )
        n_layers = self.model.config.num_hidden_layers
        if not 0 <= sae.layer < n_layers:
            raise ValueError(f"SAE layer {sae.layer} out of range [0,{n_layers})")
        self._block(sae.layer)  # raises early if architecture unsupported
        self.sae = sae
        self._feats = None
        self._steer = None
        return sae.status()

    def _compute_features(self) -> torch.Tensor:
        """[S, d_sae] feature activations for the last generation (cached)."""
        if self.sae is None:
            raise RuntimeError("No SAE loaded. POST /api/sae/load first.")
        if self.last_ids is None:
            raise RuntimeError("Generate something first.")
        if self.last_ids_epoch != self.epoch:
            raise RuntimeError(
                "That generation was produced by a different model. Generate again."
            )
        if self._feats is None:
            captured: list[torch.Tensor] = []

            def _capture(module, args):  # noqa: ANN001 - torch hook signature
                captured.append(args[0].detach())

            handle = self._block(self.sae.layer).register_forward_pre_hook(_capture)
            try:
                with torch.no_grad():
                    self.model(self.last_ids.unsqueeze(0).to(self.device))
            finally:
                handle.remove()
            resid = captured[0][0].to("cpu")  # [S, d_in]
            self._feats = self.sae.encode(resid).to(torch.float16)
        return self._feats

    def features_summary(self, top_k: int = 8) -> dict:
        """Per-token top-K firing features for the last generation."""
        feats = self._compute_features().float()  # [S, d_sae]
        tokens = [self.tokenizer.decode([tid]) for tid in self.last_ids.tolist()]
        acts, ids = feats.topk(top_k, dim=-1)
        return {
            "tokens": tokens,
            "top": [
                [
                    [int(fid), round(float(act), 3)]
                    for fid, act in zip(id_row, act_row)
                    if act > 0
                ]
                for id_row, act_row in zip(ids.tolist(), acts.tolist())
            ],
        }

    def feature_detail(self, feature_id: int) -> dict:
        """One feature's activation across the last generation's tokens."""
        feats = self._compute_features().float()
        if not 0 <= feature_id < feats.shape[1]:
            raise ValueError(f"feature_id must be in [0,{feats.shape[1]})")
        col = feats[:, feature_id]
        return {
            "feature_id": feature_id,
            "activations": [round(v, 3) for v in col.tolist()],
            "max": round(float(col.max()), 3),
            "argmax": int(col.argmax()),
        }

    def set_steering(self, feature_id: int | None, scale: float = 0.0) -> dict:
        """Set (or clear, with feature_id=None) single-feature steering."""
        if feature_id is None:
            self._steer = None
        else:
            if self.sae is None:
                raise RuntimeError("No SAE loaded.")
            if not 0 <= feature_id < self.sae.d_sae:
                raise ValueError(f"feature_id must be in [0,{self.sae.d_sae})")
            self._steer = (feature_id, float(scale))
        return self.steering_status()

    def steering_status(self) -> dict:
        if self._steer is None:
            return {"active": False}
        fid, scale = self._steer
        return {"active": True, "feature_id": fid, "scale": scale}
