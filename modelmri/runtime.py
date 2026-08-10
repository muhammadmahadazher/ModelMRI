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

import os
import re
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from . import ablate, capacity, devices, paths, ollama, progress, session
from .saes import SAEHandle, SAEStatus


def _require_causal_lm(hf_id: str) -> None:
    """Refuse a repo the playground cannot run, and say what it is.

    The picker filters these out, but an id can also be typed, and the failure
    mode without this is a multi-screen HuggingFace traceback about
    sentencepiece for a model that has no tokenizer because it is a diffusion
    model. Reading the config first costs a few kilobytes.
    """
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(hf_id)
    except Exception:
        return  # let the real loader produce the real error

    archs = list(getattr(cfg, "architectures", None) or [])
    if any(a.endswith(("ForCausalLM", "LMHeadModel")) for a in archs):
        return
    if not archs:
        return  # unknown shape: don't block on a guess

    kind = archs[0]
    raise ValueError(
        f"{hf_id} is a {kind}, which is not a causal language model. The "
        "playground generates text; this repo cannot do that. Robot policies "
        "belong in the robot panel, and sparse autoencoders in the features "
        "panel."
    )


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

    hub = paths.hf_hub_cache()
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


class LoadCancelled(RuntimeError):
    """The user stopped a load. Not an error in the code, and not silent."""


def download_size(hf_id: str) -> int:
    """Bytes this repo will actually pull, from its own published metadata.

    0 means unknown, never "small" -- GGUF and pickle repos publish nothing
    to go on, and treating unknown as zero is how a guard lets through the
    one download it existed to stop.
    """
    try:
        from . import hub

        info = hub._api(f"/models/{hf_id}", hub.token(), timeout=8)
        return hub.weight_bytes(info)
    except Exception:
        return 0


def _free_space() -> tuple[Path, int]:
    """Free bytes on the volume the HuggingFace cache lives on. Kept as a
    seam so tests can state the disk situation instead of depending on the
    developer's own drive."""
    return capacity.free_space(paths.hf_hub_cache())


def _preflight(hf_id: str, accel, confirm: bool) -> None:
    """Refuse a download that cannot work, before a byte moves.

    The rule lives in `capacity`, shared with the Ollama pull path so the
    two cannot drift into disagreeing about what is too big.
    """
    _, free = _free_space()
    capacity.guard(
        download_size(hf_id),
        paths.hf_hub_cache(),
        label=hf_id,
        vram_gb=getattr(accel, "vram_gb", None),
        accel_name=getattr(accel, "name", ""),
        confirm=confirm,
        free_override=free,
    )


def _repo_dir(hf_id: str) -> Path:
    """The cache directory for a repo id, with the id neutralised.

    `hf_id` arrives from an HTTP body. Replacing only "/" left backslashes
    and dots intact, so on Windows `a\\..\\..\\x` walked straight out of the
    cache — and the one caller that follows this deletes files. Everything
    that is not a plain repo-id character becomes a dash; the result can only
    ever name a directory inside the cache.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "--", hf_id.replace("/", "--"))
    safe = safe.replace("..", "--").strip(". ")
    return paths.hf_hub_cache() / f"models--{safe or 'unnamed'}"


def _clean_partials(hf_id: str) -> int:
    """Delete the half-written blobs a cancelled download left. Bytes freed."""
    repo = _repo_dir(hf_id)
    freed = 0
    try:
        for stub in repo.rglob("*.incomplete"):
            try:
                freed += stub.stat().st_size
                stub.unlink()
            except OSError:
                pass
    except OSError:
        pass
    return freed


# The child that does the downloading. Run as `python -c`, so terminating it
# terminates the transfer -- which is the whole point. huggingface_hub offers
# no way to interrupt a download in-process, so the download does not happen
# in our process.
_PREFETCH = """
import sys
from huggingface_hub import HfApi, snapshot_download
repo = sys.argv[1]

# Weight formats nothing in this stack can load. Each is a complete
# duplicate of the PyTorch weights, and skipping them is free.
ignore = [
    "*.h5", "*.msgpack", "*.tflite", "*.onnx", "*.onnx_data", "*.gguf",
    "*.ot",
    "onnx/*", "coreml/*", "openvino/*", "tflite/*",
]

# `pytorch_model.bin` is byte-for-byte redundant when safetensors is present
# -- transformers loads the safetensors and never opens the .bin -- so
# fetching both doubles the transfer for nothing. Measured on gpt2: 523 MB
# of safetensors, an identical 523 MB .bin, and a 523 MB rust_model.ot, for
# 1.7 GB downloaded where 523 MB was needed.
#
# Only dropped when a root-level .safetensors actually exists to load
# instead. A repo whose weights live only in .bin still gets its .bin, and
# an adapter's stray safetensors in a subdirectory does not count.
try:
    root = [f for f in HfApi().list_repo_files(repo) if "/" not in f]
    if any(f.endswith(".safetensors") for f in root):
        ignore += ["*.bin", "*.pth"]
except Exception:
    pass  # No listing, no optimisation -- fetch everything rather than guess.

snapshot_download(repo, ignore_patterns=ignore)
"""


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
        self.last_prompt: str = ""
        self.last_n_prompt_tokens: int = 0
        # One entry per intervention: "live", "steered", "ablate:L.H".
        # Comparing two runs means holding two, and they must all be dropped
        # together — a stale "live" beside a fresh "steered" would render a
        # difference between two different generations.
        self._attn_variants: dict[str, list[torch.Tensor]] = {}
        self._attn_tokens: list[str] | None = None
        # An opened `.mri`. When set, the attention methods serve it instead of
        # the model, so every panel reads a shared session through the same
        # calls it uses for a live one. Loading a model clears it -- you asked
        # for live weights, you should not silently keep getting a recording.
        self.replay: session.Session | None = None
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

    def load(
        self,
        hf_id: str = DEFAULT_MODEL,
        source: str = "hf",
        confirm: bool = False,
    ) -> ModelStatus:
        """Load a model. source="hf" (full introspection) or "ollama" (text only).

        `confirm=True` overrides the size guard — the user has been told the
        numbers and chosen anyway.

        Blocking — call from a worker thread.
        """
        if source not in ("hf", "ollama"):
            raise ValueError(f"unknown source {source!r} (use 'hf' or 'ollama')")

        if source == "ollama":
            st = ollama.status()
            if not st["up"]:
                raise RuntimeError(
                    f"Ollama is not running at {st.get('host') or ollama.default_host()}"
                    " — start it, or "
                    "install from ollama.com. Set OLLAMA_HOST if it listens "
                    "somewhere else."
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
                self.replay = None
                self.last_ids = None
                self._attn_variants.clear()
                self._attn_tokens = None
                self.sae = None
                self._feats = None
                self._steer = None
            return self.status()

        with self._lock:
            dtype = devices.torch_dtype(self.accel)
            progress.TRACKER.start(hf_id)
            try:
                # Check what this repo *is* before spending minutes on it.
                # AutoModelForCausalLM on a diffusion or segmentation repo
                # fails deep inside the tokenizer with "You need to have
                # sentencepiece or tiktoken installed", which sends people
                # installing packages that were never the problem.
                _require_causal_lm(hf_id)
                # Refuse the impossible before a byte moves.
                _preflight(hf_id, self.accel, confirm)
                tokenizer = AutoTokenizer.from_pretrained(hf_id)
                progress.TRACKER.stage("weights")
                # Fetch in a child process first, so Stop works. Best effort:
                # any failure falls through to from_pretrained downloading it.
                try:
                    self._prefetch_weights(hf_id)
                except LoadCancelled:
                    raise
                except Exception:
                    pass
                if progress.TRACKER.cancelled.is_set():
                    raise LoadCancelled("Load stopped before the weights loaded.")
                model = AutoModelForCausalLM.from_pretrained(
                    hf_id,
                    torch_dtype=dtype,
                    attn_implementation="eager",  # materialises attention
                )
            except LoadCancelled as err:
                progress.TRACKER.finish(error=str(err))
                raise
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
                try:
                    model = model.to(torch.float32).to(self.device)
                except Exception as cpu_err:
                    # float32 on CPU needs roughly twice the VRAM figure that
                    # just failed, so this is the *likely* path for a big
                    # model, not the exotic one. Uncaught, it escaped before
                    # TRACKER.finish() ran: the progress meter stayed "active"
                    # for the rest of the session and its watcher thread
                    # polled the disk forever.
                    progress.TRACKER.finish(
                        error=f"{type(err).__name__} on {self.accel.name}, then "
                        f"{type(cpu_err).__name__} on CPU: not enough memory "
                        f"for this model"
                    )
                    raise RuntimeError(
                        f"'{hf_id}' does not fit: {type(err).__name__} on GPU, "
                        f"then {type(cpu_err).__name__} on CPU. Try a smaller model."
                    ) from cpu_err
            model.eval()
            progress.TRACKER.finish()
            self.epoch += 1
            self.backend = "hf"
            self.tokenizer, self.model, self.hf_id = tokenizer, model, hf_id
            self.replay = None
            self.last_ids = None
            self._attn_variants.clear()
            self._attn_tokens = None
            self.sae = None
            self._feats = None
            self._steer = None
            return self.status()

    def _prefetch_weights(self, hf_id: str) -> None:
        """Download the repo in a child process, so Stop can actually stop it.

        `from_pretrained` downloads inside our own process, and there is no
        supported way to interrupt it: the thread is blocked in a socket
        read, and Python cannot kill a thread. That is why a 1.5 TB download
        could only be stopped by killing the server.

        A child process can be terminated. Once it has finished, the
        subsequent `from_pretrained` finds every file in the cache and does
        no network I/O at all, so this costs nothing in the normal case.

        Failures here are deliberately not fatal: if the child cannot run for
        any reason, we fall through and let `from_pretrained` download the
        old way. A broken optimisation must not break loading.
        """
        # DEVNULL on BOTH streams, and it has to stay that way.
        #
        # `stderr=PIPE` here deadlocked a load that had already finished
        # downloading: huggingface_hub writes tqdm progress bars to stderr,
        # nothing in this process was draining the pipe, and the child
        # blocked forever once the ~64 KB buffer filled. The UI sat at
        # "551 MB / 551 MB · 234s · reading from local cache" indefinitely.
        # If you ever want the child's output, drain it on a thread -- do not
        # simply open the pipe.
        env = {**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
        proc = subprocess.Popen(
            [sys.executable, "-c", _PREFETCH, hf_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            # Windows: put it in its own group so terminate() does not also
            # signal the server it was spawned from.
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            if sys.platform == "win32"
            else 0,
        )
        try:
            while proc.poll() is None:
                if progress.TRACKER.cancelled.wait(0.4):
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    freed = _clean_partials(hf_id)
                    raise LoadCancelled(
                        f"Download stopped. Removed {freed / 1e6:,.0f} MB of "
                        f"partial files; anything already complete was kept."
                    )
        finally:
            if proc.poll() is None:  # an exception on our side, not the child's
                proc.terminate()

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

        # One installer, shared with the attention capture. Two copies of
        # "what steering does" would eventually disagree, and the comparison
        # would then be between a real run and an approximation of one.
        steer_handle = None
        if self._steer is not None and self.sae is not None:
            steer_handle = self._steer_handle()

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
        # Producing your own generation is an unambiguous request to look at
        # it. Leaving a shared session open would mean the token strip shows
        # what you just generated while the heat map below it still describes
        # somebody else's run -- a discrepancy nothing on screen explains.
        self.replay = None
        self.last_ids = ids[0].detach().to("cpu")
        self.last_ids_epoch = epoch
        # Kept for export. The ids alone cannot say where the prompt ended,
        # and a shared session that cannot show what was asked is a heat map
        # with no question attached.
        self.last_prompt = prompt
        self.last_n_prompt_tokens = int(inputs["input_ids"].shape[1])
        self._attn_variants.clear()  # recomputed on demand
        self._attn_tokens = None
        self._feats = None

    # ---------------- attention ----------------

    def attention_meta(self) -> dict:
        """Shape info for the last generation's attention, without computing it."""
        if self.replay is not None:
            return self.replay.attention_meta()
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

    def _capture(self, variant: str = "live") -> list[torch.Tensor]:
        """Every layer's attention for the last generation, under `variant`.

        The whole comparison feature rests on this being the SAME token
        sequence every time. It is a second forward pass over `last_ids`, not
        a second generation — so the two sides cannot disagree about what
        token 5 is, and a cell-by-cell difference means something.

        Generating twice would look equivalent and would not be: with
        temperature above zero the sampled tokens diverge, and even at
        greedy a different chat template inserts a different number of
        leading tokens. Subtracting misaligned sequences produces a smooth,
        plausible, entirely fictitious picture.
        """
        cached = self._attn_variants.get(variant)
        if cached is not None:
            return cached

        ids = self.last_ids.unsqueeze(0).to(self.device)
        handles: list[Any] = []
        try:
            if variant.startswith("ablate:"):
                _, _, spec = variant.partition(":")
                try:
                    a_layer, a_head = (int(x) for x in spec.split("."))
                except ValueError as err:
                    raise ValueError(
                        f"cannot read {variant!r} — expected ablate:LAYER.HEAD"
                    ) from err
                block = self._block(a_layer)
                n_heads = self.model.config.num_attention_heads
                head_dim = ablate.head_geometry(block, n_heads)
                if not 0 <= a_head < n_heads:
                    raise ValueError(f"head must be in [0,{n_heads})")
                handles.append(
                    ablate.out_projection(block).register_forward_pre_hook(
                        ablate._cut(a_head, head_dim, "zero")
                    )
                )
            elif variant == "steered":
                if self._steer is None or self.sae is None:
                    raise RuntimeError(
                        "Nothing is being steered, so there is no steered run "
                        "to compare against. Set a feature and a scale first."
                    )
                handles.append(self._steer_handle())
            elif variant != "live":
                raise ValueError(f"unknown variant {variant!r}")

            with torch.no_grad():
                out = self.model(ids, output_attentions=True)
        finally:
            for handle in handles:
                handle.remove()

        captured = [a[0].detach().to(torch.float16).cpu() for a in out.attentions]
        self._attn_variants[variant] = captured
        if self._attn_tokens is None:
            self._attn_tokens = [
                self.tokenizer.decode([tid]) for tid in self.last_ids.tolist()
            ]
        return captured

    def _ready_for_attention(self) -> None:
        if not self.loaded or self.last_ids is None:
            raise RuntimeError("Generate something first, then inspect attention.")
        if self.last_ids_epoch != self.epoch:
            raise RuntimeError(
                "That generation was produced by a different model. Generate again."
            )

    def attention(self, layer: int, head: int, variant: str = "live") -> dict:
        """Token strings + [S, S] attention matrix for one layer/head.

        One full forward pass over the last generated sequence; all layers
        are cached so switching heads is instant. `variant` selects which run
        to look at — see `_capture`.
        """
        if self.replay is not None:
            return self.replay.attention_slice(layer, head)
        self._ready_for_attention()

        with self._lock:
            captured = self._capture(variant)

        n_layers, n_heads = len(captured), captured[0].shape[0]
        if not (0 <= layer < n_layers and 0 <= head < n_heads):
            raise ValueError(f"layer must be in [0,{n_layers}), head in [0,{n_heads})")

        matrix = captured[layer][head].to(torch.float32)
        return {
            "layer": layer,
            "head": head,
            "variant": variant,
            "tokens": self._attn_tokens,
            "matrix": [[round(v, 4) for v in row] for row in matrix.tolist()],
        }

    def attention_diff(
        self, layer: int, head: int, a: str = "live", b: str = "steered"
    ) -> dict:
        """`a` minus `b`, cell by cell, over one token sequence.

        Both sides are forward passes over the same `last_ids`, so index i is
        the same token in both by construction. That is the only arrangement
        in which subtracting two attention matrices means anything — see the
        note in `_capture`, and `compare_replay` for the case where the other
        side comes from a file and the alignment has to be checked instead of
        guaranteed.
        """
        if self.replay is not None:
            raise RuntimeError(
                "You are viewing a recording. Close it to compare two runs of "
                "your own model."
            )
        self._ready_for_attention()

        with self._lock:
            left, right = self._capture(a), self._capture(b)

        n_layers, n_heads = len(left), left[0].shape[0]
        if not (0 <= layer < n_layers and 0 <= head < n_heads):
            raise ValueError(f"layer must be in [0,{n_layers}), head in [0,{n_heads})")

        delta = left[layer][head].to(torch.float32) - right[layer][head].to(
            torch.float32
        )
        rows = [[round(v, 4) for v in row] for row in delta.tolist()]
        peak = float(delta.abs().max()) if delta.numel() else 0.0

        # An all-zero difference is a result, and one specific case produces
        # it every time for a reason worth stating rather than leaving the
        # user to conclude the intervention did nothing.
        note = ""
        if peak == 0.0 and b.startswith("ablate:"):
            cut_layer = int(b.partition(":")[2].split(".")[0])
            if layer <= cut_layer:
                note = (
                    f"Exactly zero, and it has to be: ablation removes a "
                    f"head's OUTPUT, while layer {layer}'s attention weights "
                    f"are computed from its input. Removing a head at layer "
                    f"{cut_layer} can only change attention at layer "
                    f"{cut_layer + 1} and above. Look downstream."
                )
        elif peak == 0.0:
            note = "The two runs produced identical attention here."

        return {
            "layer": layer,
            "head": head,
            "a": a,
            "b": b,
            "tokens": self._attn_tokens,
            "matrix": rows,
            "max_abs": round(peak, 4),
            "moved": int((delta.abs() > 0.01).sum()),
            "cells": int(delta.numel()),
            "note": note,
        }

    def _steer_handle(self):
        """Install the steering hook and return its handle.

        Lifted out of `generate_stream` so the attention capture can install
        exactly the same intervention. Two implementations of "what steering
        does" would eventually disagree, and the comparison would be between
        a real run and an approximation of one.
        """
        fid, scale = self._steer
        direction = self.sae.steering_vector(fid).to(self.device)
        block = self._block(self.sae.layer)

        if self.sae.point == "resid_post":

            def _post(module, args, output):  # noqa: ANN001
                tup = isinstance(output, tuple)
                hidden = output[0] if tup else output
                moved = hidden + scale * direction.to(hidden.dtype)
                return ((moved,) + tuple(output[1:])) if tup else moved

            return block.register_forward_hook(_post)

        def _pre(module, args):  # noqa: ANN001
            hidden = args[0]
            return (hidden + scale * direction.to(hidden.dtype),) + args[1:]

        return block.register_forward_pre_hook(_pre)

    def ablate_heads(self, layer: int | None = None, baseline: str = "zero") -> dict:
        """Rank heads by how far removing one moves the next-token answer.

        `layer=None` sweeps every layer, which is n_layers x n_heads + 2
        forward passes: 146 for gpt2, 450 for Qwen3-0.6B. That count is the
        portable part of the cost.

        What a pass costs is not. On one RTX 4060 the same model measured
        between 12 and 71 ms/pass across sessions, so no figure in seconds
        belongs in this docstring — the panel measures a layer on the user's
        machine and extrapolates. Back to back the rate is steady (1.0-1.1x
        over six runs) and the extrapolation holds to within 2.5%, but the
        FIRST ranking after a load pays CUDA warm-up and runs several times
        slower (Qwen3: 3.05 s, then 0.80, 0.78), which is why the panel keeps
        the fastest rate it has seen rather than the latest.

        Hence: one layer by default, the whole model only when told.

        The measurement itself lives in `ablate`, along with the four things
        that make the number honest.
        """
        if self.replay is not None:
            raise RuntimeError(
                "This is a recording. Ranking heads means running the model, "
                "and a `.mri` does not carry one."
            )
        if self.backend == "ollama":
            raise RuntimeError(
                "Ollama serves text only — there is no forward pass to "
                "intervene in. Load the model through HuggingFace."
            )
        if not self.loaded or self.last_ids is None:
            raise RuntimeError("Generate something first, then rank its heads.")
        if self.last_ids_epoch != self.epoch:
            raise RuntimeError(
                "That generation was produced by a different model. Generate again."
            )

        cfg = self.model.config
        n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
        if layer is not None and not 0 <= layer < n_layers:
            raise ValueError(f"layer must be in [0,{n_layers})")
        layers = list(range(n_layers)) if layer is None else [layer]

        with self._lock:
            # Attribute at the last prompt token: its next-token distribution
            # is the model's answer to the question, before any of its own
            # output feeds back in.
            size = int(self.last_ids.shape[0])
            position = max(0, min(self.last_n_prompt_tokens - 1, size - 1))
            try:
                return ablate.rank_heads(
                    self.model,
                    self._block,
                    self.last_ids.unsqueeze(0).to(self.device),
                    position=position,
                    layers=layers,
                    n_heads=n_heads,
                    baseline=baseline,
                    decode=lambda t: self.tokenizer.decode([t]),
                )
            except ablate.AblationError as err:
                # Not a crash: a shape this code cannot read honestly.
                raise RuntimeError(str(err)) from err

    # ---------------- sessions (.mri) ----------------

    # One byte per attention value before gzip. 24 MB of them is already a
    # large thing to attach to a message; past that we export the cross
    # through the view you are on instead of the full cube, and say so.
    _FULL_EXPORT_BUDGET = 24_000_000

    def export_session(self, layer: int = 0, head: int = 0, note: str = "") -> bytes:
        """Serialise the current analysis to a `.mri` someone else can open."""
        if self.replay is not None:
            raise RuntimeError(
                "You are viewing a shared session. Close it, then generate "
                "something of your own to export."
            )
        if self.backend == "ollama":
            raise RuntimeError(
                "Ollama serves text only — there are no internals to export. "
                "Load the model through HuggingFace to capture a session."
            )
        # Populates the attention cache if this is the first look, and raises
        # the same guidance the panel would if there is nothing to capture.
        self.attention(layer, head)

        captured = self._attn_variants["live"]
        n_layers, n_heads = len(captured), int(captured[0].shape[0])
        size = int(self.last_ids.shape[0])
        full = n_layers * n_heads * size * size <= self._FULL_EXPORT_BUDGET
        if full:
            wanted = [(li, hi) for li in range(n_layers) for hi in range(n_heads)]
            scope = "every layer and head"
        else:
            wanted = [(li, head) for li in range(n_layers)]
            wanted += [(layer, hi) for hi in range(n_heads) if hi != head]
            scope = (
                f"every layer at head {head}, and every head at layer {layer} — "
                f"the full cube would have been "
                f"{n_layers * n_heads * size * size / 1e6:.0f} MB before compression"
            )

        tokens = self._attn_tokens or []
        cut = min(self.last_n_prompt_tokens, len(tokens))
        return session.build(
            model_id=self.hf_id,
            device=self.device,
            dtype=str(next(self.model.parameters()).dtype).removeprefix("torch."),
            n_params=sum(p.numel() for p in self.model.parameters()),
            tokens=tokens,
            prompt=self.last_prompt,
            generation="".join(tokens[cut:]),
            attention={(li, hi): captured[li][hi] for li, hi in wanted},
            n_layers=n_layers,
            n_heads=n_heads,
            note=note,
            scope=scope,
        )

    def open_session(self, data: bytes) -> dict:
        """Open a `.mri`. Replaces any session already open; leaves the model."""
        parsed = session.parse(data)
        self.replay = parsed
        return self.session_info()

    def close_session(self) -> dict:
        self.replay = None
        return self.session_info()

    def session_info(self) -> dict:
        """What the UI needs to say whether you are looking at a recording."""
        if self.replay is None:
            return {"open": False}
        return {
            "open": True,
            "meta": self.replay.meta,
            "prompt": self.replay.prompt,
            "generation": self.replay.generation,
            "n_tokens": len(self.replay.tokens),
            "n_slices": len(self.replay.attention),
            "slices": sorted(self.replay.attention),
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

            block = self._block(self.sae.layer)
            if self.sae.point == "resid_post":
                # resid_post is the block's OUTPUT. Hooking the input here fed
                # the SAE the stream from the wrong side of the block, which
                # does not error -- it just yields features for activations the
                # SAE never saw in training.
                def _capture(module, args, output):  # noqa: ANN001
                    hidden = output[0] if isinstance(output, tuple) else output
                    captured.append(hidden.detach())

                handle = block.register_forward_hook(_capture)
            else:

                def _capture(module, args):  # noqa: ANN001
                    captured.append(args[0].detach())

                handle = block.register_forward_pre_hook(_capture)
            epoch = self.epoch
            ids = self.last_ids
            try:
                with torch.no_grad():
                    self.model(ids.unsqueeze(0).to(self.device))
            finally:
                handle.remove()
            if self.epoch != epoch:
                # A load completed during the forward pass -- seconds, on CPU
                # for a 0.5B model. Caching this would file one model's
                # features under another model's generation.
                raise RuntimeError(
                    "The model changed while features were computing. Generate again."
                )
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
