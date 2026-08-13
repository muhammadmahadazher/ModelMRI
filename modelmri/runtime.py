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

import gc
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from . import (
    ablate,
    attribute,
    capacity,
    corpus,
    devices,
    feature_ablate,
    nullmodel,
    ollama,
    patch,
    paths,
    progress,
    receipts,
    session,
    telemetry,
)
from .errors import BadRequest, Refusal
from .saes import SAEHandle, SAEStatus

# One logger for the package, so a failure that the API answers generically
# still leaves a traceback in the terminal the user is already looking at.
log = logging.getLogger("modelmri")


def _load_failed(err: BaseException) -> str:
    """What the progress meter is allowed to say when a load breaks.

    The exception's CLASS, never its message. The progress snapshot is served
    verbatim by /api/model/progress -- `Snapshot.to_dict` is `asdict(self)` --
    and the load meter polls it once a second, so anything written here is a
    response body. Pasting the exception's text there put a torch message,
    with an absolute path and a site-packages frame in it, into a 200: past
    the 500 arm that returns a fixed sentence precisely to stop that, because
    the leak was not on the route anybody had hardened.

    Measured before the fix, from `POST /api/model/load` failing and the very
    next `GET /api/model/progress`:

        "error": "RuntimeError: CUDA out of memory. Tried to allocate 2.00
                  GiB. Loading C:/Users/<name>/.../model.safetensors
                  ... site-packages/torch/nn/modules/module.py, line 1518"

    The class name is kept because it is genuinely useful and carries nothing
    about the machine. Callers log the real exception, since dropping the text
    without recording it would trade a leak for an erasure, and an erasure is
    the worse bug -- the same rule `_internal` in server.py follows, and the
    one the float32-on-CPU arm below already states in its own comment.
    """
    return (
        f"{type(err).__name__} — the load failed. The full error is in the "
        "terminal running `modelmri serve`."
    )


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
    raise BadRequest(
        f"{hf_id} is a {kind}, which is not a causal language model. The "
        "playground generates text; this repo cannot do that. Robot policies "
        "belong in the robot panel, and sparse autoencoders in the features "
        "panel."
    )


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# How long a second load request waits for the first to let go. Long enough
# that a double-clicked button still serialises, short enough that a load
# which has stopped returning cannot silently swallow every request after it.
LOAD_QUEUE_WAIT_S = 2.0


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
    # The fallback deliberately does not paste the hub's own words. This
    # string is published to the browser, and the first line of an OSError
    # from huggingface_hub routinely carries the local cache path — which is
    # the machine's, not the reader's. The three branches above are the
    # failures this function can name; this one it cannot, so it says that
    # and points at the terminal, where the caller logs the real exception.
    return (
        f"Could not load '{hf_id}', and this is not one of the failures "
        f"ModelMRI knows how to explain. The full error is in the terminal "
        f"running `modelmri serve`."
    )


def local_hf_models() -> list[dict]:
    """Models already in the HuggingFace cache (offline-usable)."""

    hub = paths.hf_hub_cache()
    out: list[dict] = []
    if not hub.is_dir():
        return out
    for d in sorted(hub.glob("models--*")):
        parts = d.name.removeprefix("models--").split("--")
        # One counting rule, shared with the load meter, so the picker and the
        # progress bar cannot disagree about how big the same model is. It
        # takes the max of blobs/ and snapshots/ rather than the sum, and
        # ignores the subfolder copies of the weights that repos ship for
        # other runtimes -- both traps are documented in progress.py, and the
        # second one had this listing reporting Llama-3.2-1B at 4.96 GB
        # against the 2.48 GB it actually occupies.
        size = progress._bytes_on_disk("/".join(parts))
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
                # The blob is already gone (a previous sweep, or the download
                # child still shutting down), or Windows refuses the unlink
                # because the terminated child has not released the handle
                # yet. Continuing is right: `freed` accumulates, so one stuck
                # blob makes the "Removed N MB" the user reads an
                # understatement rather than aborting the sweep and leaving
                # the rest of the partials on disk.
                pass
    except OSError:
        # `rglob` itself, and it is NOT dead code even though it reads that
        # way on a modern interpreter. On CPython 3.12+ the recursive walk
        # swallows every OSError internally (3.13 via glob._GlobberBase,
        # 3.12 via Path.walk with on_error=None), so nothing escapes. On
        # 3.10/3.11 — and pyproject sets requires-python = ">=3.10" —
        # _RecursiveWildcardSelector catches only PermissionError, so any
        # other scandir failure on a SUBdirectory escapes: a Google Drive or
        # OneDrive placeholder (WinError 1920), or a directory deleted
        # mid-walk. Both are ordinary here. Cleanup after a cancelled
        # download must not itself raise, so the count stays at whatever was
        # removed before the walk stopped.
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


def _user_span(tokenizer: Any, prompt: str, text: str) -> tuple[int, int] | None:
    """Half-open token span of the user's own words inside the templated text.

    `None` is not a fallback to "treat everything as typed" — it is the state
    "we could not locate your words", and the caller has to say so. Getting it
    wrong in the permissive direction is the expensive error. Measured here on
    Qwen3-0.6B, bf16/cuda/eager, prompt "The capital of France is", attributing
    at the last prompt token: of the tokens the panel ranks, the two largest are
    the template's own 'assistant' at 2.02161 nats and '<|im_start|>' at
    0.322664, while the five words the user typed score between 3.12788e-05 and
    7.9083e-05 — 25,563x below the template's best. Labelling the scaffolding as
    the user's writing does not merely add rows, it puts the whole top of a list
    titled "your words" on text the user never wrote.

    Two things this refuses rather than guesses. A slow tokenizer has no offset
    mapping at all, so there is nothing to map through. And a prompt that occurs
    more than once in the templated text is genuinely ambiguous: a chat template
    contains the words "user", "assistant" and "model", so a prompt of exactly
    one of those matches the scaffolding as well as the content, and `index`
    would confidently return the first hit, which is the template's.

    The tokens a fast tokenizer adds for itself report a zero-width `(0, 0)`
    offset, and there is deliberately no separate rule for them: the overlap
    test is half-open at both ends, so `b <= start` already excludes `(0, 0)`
    from a span starting at character 0 — which is gpt2's span, and the case the
    rule would have been written for. An explicit `b <= a: continue` was here
    and was removed after a mutation test showed it could not change any
    answer: it only ever fires on an index between two overlapping ones, and a
    middle index moves neither end of a range.

    Verified against all three, prompt "The capital of France is" through the
    same chat template `generate_stream` applies: gpt2 (0, 5) over the whole
    5-token sequence, Qwen3-0.6B (3, 8) of 13, gemma-3-270m-it (5, 10) of 15 —
    each covering exactly ['The', ' capital', ' of', ' France', ' is'], and
    gemma's two leading `<bos>` outside it.
    """
    if not getattr(tokenizer, "is_fast", False):
        return None
    if not prompt or text.count(prompt) != 1:
        return None
    start = text.index(prompt)
    stop = start + len(prompt)

    try:
        offsets = tokenizer(
            [text], return_offsets_mapping=True, add_special_tokens=True
        )["offset_mapping"][0]
    except Exception:
        # An unknown span is a state this feature carries; a generation that
        # died because the offsets could not be computed is not.
        return None

    lo = hi = None
    for i, pair in enumerate(offsets):
        a, b = int(pair[0]), int(pair[1])
        # Half-open on both sides: a token ending exactly where the prompt
        # begins does not overlap it, which is also what excludes the (0, 0)
        # offsets of added specials from a span that starts at character 0.
        if b <= start or a >= stop:
            continue
        if lo is None:
            lo = i
        hi = i + 1
    if lo is None or hi is None:
        return None
    return (lo, hi)


@dataclass
class ModelStatus:
    loaded: bool
    hf_id: str | None = None
    device: str | None = None
    dtype: str | None = None
    n_params: int | None = None
    # Does this model expect a conversation, or is it a raw text continuer?
    # None means unknown, which is NOT the same as False -- False is the
    # positive claim "this is a base model" and the UI renders it as one.
    #
    # The same signal `generate_stream` already branches on: a tokenizer with
    # a chat template was instruction-tuned, one without was not. It is worth
    # publishing because the difference explains most "why is it answering
    # nonsense" — gpt2 continues your sentence, it does not answer your
    # question, and a UI that does not say so invites the reader to conclude
    # the tool is broken when the tool is working exactly as intended.
    instruct: bool | None = None
    # The load plan, when this model was built from a GGUF: file size against
    # resident size, the expansion between them, and the standing caveat that
    # every measurement below describes the quantised weights. None otherwise,
    # never {} -- an empty dict reads as "from a GGUF, nothing to say".
    gguf: dict | None = None

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
        # Set when the loaded model came from a GGUF. NOT a third backend: a
        # dequantised GGUF is an ordinary torch module and every panel works on
        # it unchanged, so gating fifteen call sites on a new backend value
        # would be inventing a difference that does not exist. What IS
        # different is what the numbers describe -- the quantised weights, not
        # the original -- and that is provenance, so it rides here and appears
        # in status(). None on every other load.
        self.gguf: dict | None = None
        # GPU when one is usable (NVIDIA / AMD ROCm / Intel / Apple), else CPU
        self.accel = devices.detect()
        self.device = self.accel.torch_device
        # Bumped on every load. Everything derived from a generation carries
        # the epoch it was produced under, because a model swap that lands
        # mid-generation would otherwise leave one model's token ids to be
        # interpreted by another model's weights -- which does not crash, it
        # just quietly reports numbers about nothing.
        self.epoch = 0
        self._last_patch: dict = {}
        # One receipt per measurement that has run, keyed by the operation, so
        # `export_session` can put the setup of every number in the `.mri`
        # beside the number. Each carries the epoch it was taken under and is
        # filtered on that at export -- the same rule `_last_patch` keeps, and
        # for the same reason: a receipt describing the model that WAS loaded,
        # riding along with an export of the one that is, would be a lie told
        # in the one field a reader is meant to be able to trust.
        self._receipts: dict[str, dict] = {}
        # Base vs instruction-tuned for an Ollama model, from Ollama itself.
        # None until asked, and None again on any HF load.
        self._ollama_instruct: bool | None = None
        # Last completed generation (prompt + output), for attention capture.
        self.last_ids: torch.Tensor | None = None
        self.last_ids_epoch = -1
        self.last_prompt: str = ""
        self.last_n_prompt_tokens: int = 0
        # Where the user's own words sit inside the templated prompt, as a
        # half-open token span. Additive and allowed to be absent: None means
        # "could not be located", never "all of it". Token attribution shows the
        # two groups apart because the template's tokens can outscore the user's
        # by 25,563x (measured on Qwen3-0.6B, see `_user_span`), so a panel that
        # cannot tell them apart has to say that rather than pick one.
        self.last_user_span: tuple[int, int] | None = None
        # What the last generation cost, including what watching it cost.
        # None until something has been generated -- a telemetry bar with
        # zeros in it is a claim about a run that never happened.
        self.last_telemetry: telemetry.Telemetry | None = None
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
            # NOT unconditionally True. Ollama publishes base tags —
            # `llama3.2:1b-text-fp16`, `qwen2.5:0.5b-base`, `gemma2:2b-text-*`
            # all exist — and claiming instruction-tuning for them silences
            # the one caveat that explains why they answer strangely, which is
            # the caveat's whole reason for existing.
            #
            # None, not False: unknown is a third state. False is a positive
            # claim ("this is a base model") that the UI renders as such.
            return ModelStatus(
                loaded=True,
                hf_id=self.hf_id,
                device="ollama",
                instruct=self._ollama_instruct,
            )
        if self.model is None:
            return ModelStatus(loaded=False, device=self.device)
        return ModelStatus(
            loaded=True,
            hf_id=self.hf_id,
            device=self.device,
            dtype=str(next(self.model.parameters()).dtype).removeprefix("torch."),
            n_params=sum(p.numel() for p in self.model.parameters()),
            instruct=bool(getattr(self.tokenizer, "chat_template", None)),
            gguf=self.gguf,
        )

    def unload(self) -> dict:
        """Drop the model and give the memory back.

        There was no way to do this. Custom models had `unload`; the model
        actually holding your VRAM did not, so the only way to free a 2.5 GB
        checkpoint was to kill the server — on the machines where that matters
        most, the ones where the next model will not fit beside this one.

        Everything a load sets is cleared, because a half-unloaded runtime is
        worse than either state: the SAE is bound to a model that is gone, the
        retained attention describes a sequence nothing can reproduce, and the
        steering hook would install into nothing.
        """
        with self._load_slot("unload"):
            was = self.hf_id
            freed = self._accel_bytes()

            self.epoch += 1
            self.model = None
            self.tokenizer = None
            self.hf_id = None
            self.backend = None
            self.gguf = None
            self.replay = None
            self.last_ids = None
            self.last_user_span = None
            self._attn_variants.clear()
            self._attn_tokens = None
            self.sae = None
            self._feats = None
            self._steer = None

            gc.collect()
            self._empty_accel_cache()
            now = self._accel_bytes()

            return {
                "unloaded": bool(was),
                "was": was,
                # What actually came back, not what should have. An allocator
                # that keeps its arena is a real outcome and the reader should
                # see it rather than a promise.
                "freed_bytes": max(0, (freed or 0) - (now or 0)),
                "accelerator_bytes_in_use": now,
                "status": asdict(self.status()),
            }

    def _accel_bytes(self) -> int | None:
        """Bytes this process has on the accelerator, or None if unknowable."""
        try:
            if self.accel.kind in ("cuda", "rocm") and torch.cuda.is_available():
                return int(torch.cuda.memory_allocated())
            if self.accel.kind == "xpu" and hasattr(torch, "xpu"):
                return int(torch.xpu.memory_allocated())
        except Exception:
            return None
        return None

    def _empty_accel_cache(self) -> None:
        """Hand the allocator's cache back to the driver."""
        try:
            if self.accel.kind in ("cuda", "rocm") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif self.accel.kind == "xpu" and hasattr(torch, "xpu"):
                torch.xpu.empty_cache()
            elif self.accel.kind == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
        except Exception as err:
            # Best effort by definition — the memory is already dereferenced,
            # and an allocator that will not release its arena is not a reason
            # to fail the unload the caller asked for. Recorded rather than
            # swallowed, because "I pressed Unload and nvidia-smi did not move"
            # is a question somebody will ask, and the answer is in here.
            log.info("could not empty the %s cache: %s", self.accel.kind, err)

    def _in_flight(self) -> str:
        """A sentence describing the load already running, or "" if none is.

        Reads the progress tracker rather than the lock, because the lock can
        only answer "held" and the useful answer is *what* is holding it and
        for how long.
        """
        snap = progress.TRACKER.snapshot()
        if not snap.active:
            return ""
        return (
            f"'{snap.hf_id}' has been loading for {snap.elapsed_s:.0f}s "
            f"({snap.stage}: {snap.detail})"
        )

    @contextmanager
    def _load_slot(self, hf_id: str) -> Iterator[None]:
        """The load lock, with a ceiling on how long a caller waits for it.

        A load holds this for as long as it takes, and one that stops
        returning held it forever: the next request blocked in `with
        self._lock` with no timeout, no message and no way out. That is what
        "no model is actually loading" looks like from the outside -- and
        because the browser labelled the meter with the model it had just
        picked rather than the one the server was loading, the wedged load's
        byte counts appeared under the queued model's name.

        Refusing is the honest answer: one model is loading at a time by
        design, so a second request is a 409, not a queue slot.
        """
        if not self._lock.acquire(timeout=LOAD_QUEUE_WAIT_S):
            busy = self._in_flight() or "another load is running"
            raise Refusal(
                f"Cannot load '{hf_id}' yet: {busy}. Stop it first, or wait "
                f"for it to finish."
            )
        try:
            yield
        finally:
            self._lock.release()

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
            raise BadRequest(f"unknown source {source!r} (use 'hf' or 'ollama')")

        if source == "ollama":
            st = ollama.status()
            if not st["up"]:
                raise Refusal(
                    f"Ollama is not running at {st.get('host') or ollama.default_host()}"
                    " — start it, or "
                    "install from ollama.com. Set OLLAMA_HOST if it listens "
                    "somewhere else."
                )
            if hf_id not in st["models"]:
                raise BadRequest(
                    f"'{hf_id}' is not installed in Ollama. Installed: "
                    f"{', '.join(st['models']) or 'none'} — run `ollama pull {hf_id}`"
                )
            with self._lock:
                self.epoch += 1
                self.model = None
                self.tokenizer = None
                self.backend = "ollama"
                self.gguf = None
                self.hf_id = hf_id
                # Asked once per load, from Ollama's own metadata. None when
                # it cannot be determined — never assumed.
                self._ollama_instruct = ollama.is_instruct(hf_id)
                self.replay = None
                self.last_ids = None
                self.last_user_span = None
                self._attn_variants.clear()
                self._attn_tokens = None
                self.sae = None
                self._feats = None
                self._steer = None
            return self.status()

        with self._load_slot(hf_id):
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
                except Exception as err:  # see the comment
                    # The child could not be started or could not be waited
                    # on: no usable sys.executable, a box that refuses process
                    # creation, a denied CREATE_NEW_PROCESS_GROUP. Falling
                    # through is right — from_pretrained downloads the old
                    # way and the only thing lost is the ability to Stop.
                    #
                    # Deliberately NOT narrowed. OSError is the honest guess
                    # and it is a guess; this sits on the model-load critical
                    # path, where one missed type turns a working slow
                    # fallback into a failed load. Logged instead, because
                    # the only other symptom is "Stop does not stop the
                    # download", which nobody reports as a bug in here.
                    log.warning(
                        "prefetch child unusable for %s (%s: %s); the download "
                        "will run in-process and Stop will not interrupt it",
                        hf_id,
                        type(err).__name__,
                        err,
                    )
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
                #
                # Logged as well as answered: `_hub_error_message` names the
                # three failures it recognises and refuses to paste the hub's
                # own text for the fourth, so the terminal is the only place
                # the real exception survives.
                message = _hub_error_message(hf_id, err)
                log.warning("hub load of %s failed", hf_id, exc_info=err)
                progress.TRACKER.finish(error=message)
                raise BadRequest(message) from err
            except BaseException as err:
                # The OSError arm above is careful to publish only an authored
                # sentence. This one pasted the exception's own text into the
                # same snapshot, which is the browser-facing one.
                log.exception("load of %s failed", hf_id, exc_info=err)
                progress.TRACKER.finish(error=_load_failed(err))
                raise
            progress.TRACKER.stage("device", f"moving to {self.accel.name}")
            try:
                model.to(self.device)
            except Exception as err:
                # Out of VRAM, or a driver that says yes then fails: keep the
                # tool usable instead of dying on the user's first click.
                #
                # Two sinks here and BOTH are served verbatim: the snapshot by
                # /api/model/progress, and `accel.reason` by /api/accelerator
                # -- devices.py calls that field "shown in the UI" and means
                # it. So the class, never the text, in either.
                log.warning(
                    "moving %s onto %s failed", hf_id, self.device, exc_info=err
                )
                if self.accel.kind == "cpu":
                    progress.TRACKER.finish(error=_load_failed(err))
                    raise
                rejected = self.accel.name
                self.accel = devices.detect(prefer="cpu")
                self.accel.reason = (
                    f"fell back to CPU: {rejected} rejected this model "
                    f"({type(err).__name__})"
                )
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
                    # A Refusal, not a crash: both attempts are named, only
                    # their exception CLASSES are interpolated (never their
                    # text), and the sentence ends with what to do. "Out of
                    # memory" is a capacity answer, and the tool is still
                    # usable afterwards.
                    raise Refusal(
                        f"'{hf_id}' does not fit: {type(err).__name__} on GPU, "
                        f"then {type(cpu_err).__name__} on CPU. Try a smaller model."
                    ) from cpu_err
            model.eval()
            progress.TRACKER.finish()
            self.epoch += 1
            self.backend = "hf"
            # Cleared, not left: loading an HF model after a GGUF one would
            # otherwise leave the previous file's provenance attached to it,
            # and the UI would caption a full-precision model as quantised.
            self.gguf = None
            self.tokenizer, self.model, self.hf_id = tokenizer, model, hf_id
            self.replay = None
            self.last_ids = None
            self.last_user_span = None
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

    def patch_trace(self, clean: str, corrupt: str) -> dict:
        """Causal trace between two prompts. See patch.py for what it measures.

        Holds the lock for the whole trace because it hangs hooks on every
        block and runs the model hundreds of times: a generation interleaved
        with that would read a spliced residual stream and report it as the
        model's own behaviour.
        """
        with self._lock:
            if self.replay is not None:
                # A recording that CARRIES a trace can serve it. The refusal
                # below is still right for one that does not -- patching means
                # running the model again with an activation replaced, and a
                # `.mri` holds activations rather than weights -- but refusing
                # a file that already holds the answer was the format failing
                # to be worth sending.
                recorded = self.replay.patch
                if recorded.get("grids"):
                    return {**recorded, "recorded": True}
                raise Refusal(
                    "This is a recording, and it does not carry a patching "
                    "trace. Patching means running the model again with an "
                    "activation replaced, and a `.mri` holds activations "
                    "rather than weights — there is nothing here to re-run. "
                    "Whoever exported it can run the trace and share it again."
                )
            if self.backend == "ollama":
                raise Refusal(
                    "Ollama serves text, not activations, so there is no "
                    "residual stream to move between two runs. Load this model "
                    "through HuggingFace to trace it."
                )
            if self.model is None:
                raise Refusal("No model loaded — pick one first.")
            # The config's count, not a guess: `_block` raises a Refusal for
            # an architecture it cannot walk, and asking it for layer 0 first
            # turns "unsupported layout" into that message rather than into an
            # IndexError from a range built on the wrong number.
            n_layers = int(self.model.config.num_hidden_layers)
            blocks = [self._block(i) for i in range(n_layers)]
            try:
                result = patch.trace(
                    self.model,
                    self.tokenizer,
                    blocks,
                    clean,
                    corrupt,
                    device=self.device,
                )
                # Kept so a `.mri` can carry it. Tagged with the epoch, which
                # moves on every load, unload and generation: a trace measured
                # against a different model or a different run must not ride
                # along with an export and be read as belonging to it.
                # Both prompts are hashed into the receipt, not just the clean
                # one: a patching result is a statement about a PAIR, and the
                # single `prompt_sha256` every other receipt carries would
                # describe half of what was measured.
                result["receipt"] = self.receipt(
                    "patch_trace",
                    prompt=clean,
                    clean_sha256=receipts.digest(clean),
                    corrupt_sha256=receipts.digest(corrupt),
                )
                self._last_patch = {
                    **result,
                    "clean": clean,
                    "corrupt": corrupt,
                    "epoch": self.epoch,
                }
                return result
            except patch.PatchError as err:
                # A pair this measurement cannot be taken on, not a failure of
                # the code. Every one of these names what to change.
                raise BadRequest(
                    str(err)
                ) from err  # leak-ok: authored, see test_no_machine_leaks

    def _block(self, layer: int) -> torch.nn.Module:
        """The decoder block whose *input* is the residual stream at `layer`."""
        root = self.model
        if hasattr(root, "transformer") and hasattr(root.transformer, "h"):
            return root.transformer.h[layer]  # GPT-2 family
        if hasattr(root, "model") and hasattr(root.model, "layers"):
            return root.model.layers[layer]  # Llama/Qwen/Gemma family
        # A refusal, not an internal error, and the same one lens.py gives for
        # a missing final norm: this architecture is not one of the layouts
        # ModelMRI knows how to walk. It is user-reachable — POST
        # /api/sae/load and the attention panel both arrive here on an exotic
        # model — so "something inside ModelMRI failed" would be false about a
        # limitation ModelMRI knows it has.
        #
        # The message used to be `Don't know how to find block {layer} in
        # {type(root)}`, which printed a Python class repr at a reader. Names
        # the layouts instead, the way lens.py does.
        raise Refusal(
            f"could not find this model's decoder blocks, so there is no layer "
            f"{layer} to read. Supported layouts: transformer.h (the GPT-2 "
            f"family) and model.layers (Llama, Qwen, Gemma). If this "
            f"architecture keeps its blocks somewhere else, open an issue "
            f"with the model id and it becomes one line here."
        )

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
            raise Refusal("No model loaded. POST /api/model/load first.")
        epoch = self.epoch

        if self.backend == "ollama":
            # No translating wrap. There used to be an `except RuntimeError:
            # raise Refusal(str(err))` here, justified by "ollama.py has not
            # adopted Refusal yet" — and by the time it was read, ollama.py's
            # only two raises were `_relayed` and `_unreachable`, both of
            # which return a Refusal already. So the wrap was not translating
            # anything; it was relabelling every RuntimeError from underneath
            # as a deliberate no. Measured: an internal
            # RuntimeError("CUDA out of memory ... <absolute path>") came back
            # from /api/model/prompt as 409 with that path in the body, on the
            # one handler whose own comment says that must not happen.
            #
            # ollama.py's Refusals propagate untouched; anything else reaches
            # the 500 arm, which is where it belongs.
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

        # Timed around the loop rather than inside `generate`, because the
        # boundary that separates prompt processing from decode is the arrival
        # of the FIRST streamed token and there is nowhere else to observe it.
        run = telemetry.Run(self.accel.kind)
        try:
            worker = threading.Thread(target=_generate, daemon=True)
            worker.start()
            with run:
                for chunk in streamer:
                    run.token()
                    yield chunk
            worker.join(timeout=30)
        finally:
            if steer_handle is not None:
                steer_handle.remove()

        cfg = getattr(self.model, "config", None)
        n_prompt = int(inputs["input_ids"].shape[1])
        # From `generate`'s own output ids, NOT from counting stream chunks.
        # A TextIteratorStreamer yields one chunk per token plus a final flush
        # from `TextStreamer.end()`, so the chunk count is always one too many —
        # measured: 8 real tokens reported as 9, and at max_new_tokens=1 the
        # inflated count divided by a near-zero decode window produced 308
        # tok/s on a machine doing 31. None when the worker did not deliver
        # ids, which reports the count as approximate rather than inventing it.
        produced = result.get("ids")
        generated = int(produced.shape[1]) - n_prompt if produced is not None else None
        self.last_telemetry = run.finish(
            prompt_tokens=n_prompt,
            generated_tokens=generated,
            n_layers=int(getattr(cfg, "num_hidden_layers", 0) or 0),
            n_heads=int(getattr(cfg, "num_attention_heads", 0) or 0),
            dtype_bytes=2 if self.accel.dtype in ("float16", "bfloat16") else 4,
            device=self.accel.torch_device,
            dtype=self.accel.dtype,
            context=telemetry.context_limit(self.model, self.tokenizer),
        )

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
        # Derived from the same `text` that was actually tokenised above, so
        # the indices refer to these ids and no others. A span that does not
        # fit inside the prompt is a claim about a different tokenisation, and
        # it is discarded rather than believed -- the same rule session.py
        # applies to `n_prompt` arriving from a file.
        span = _user_span(self.tokenizer, prompt, text)
        if span is not None and not (
            0 <= span[0] < span[1] <= self.last_n_prompt_tokens
        ):
            span = None
        self.last_user_span = span
        self._attn_variants.clear()  # recomputed on demand
        self._attn_tokens = None
        self._feats = None
        # A receipt describes a measurement taken against a PARTICULAR
        # generation -- its prompt hash and token count are two of its fields
        # -- so a new generation invalidates every one of them. The epoch
        # cannot carry this on its own: the epoch moves on load and unload and
        # deliberately NOT on generation, so a receipt from the previous
        # prompt would still match and would be exported beside these tokens.
        self._receipts.clear()
        # The patch trace has exactly the same problem, and had it before this
        # commit. `_patch_for_export` guards on the epoch and its docstring
        # says the guard is there so that "a trace measured on an earlier
        # prompt ... would not be written into the file beside a different
        # run's tokens" -- but the epoch does not move on generation, so that
        # guard never fired for the case it describes. MEASURED, not argued:
        # patching "The Eiffel Tower is in the city of", then generating
        # "Bananas are yellow because", produced a `.mri` whose tokens and
        # attention were the bananas and whose patch section was the Eiffel
        # Tower, with nothing downstream able to tell. `adopt_step` clears it
        # on the same rebase for the same reason; the generate path was the
        # one rebase that did not.
        self._last_patch = {}

    # ---------------- attention ----------------

    def attention_meta(self) -> dict:
        """Shape info for the last generation's attention, without computing it."""
        if self.replay is not None:
            return self.replay.attention_meta()
        if self.backend == "ollama":
            return {"available": False, "reason": "internals unavailable via Ollama"}
        # Two states, not one. Every other branch here carries a `reason`; this
        # one carried nothing, so "you have not loaded a model" and "you have a
        # model but have not generated anything" arrived at the panel as the
        # same empty answer — and the panel cannot tell them apart either,
        # because there is nothing in the payload to tell them apart WITH.
        #
        # They want opposite things from the reader. One is "pick a model", the
        # other is "press the button you are already looking at".
        if not self.loaded:
            return {"available": False, "reason": "no model loaded"}
        if self.last_ids is None:
            return {
                "available": False,
                "reason": "generate something first — attention is read off a real run",
            }
        if self.last_ids_epoch != self.epoch:
            return {"available": False, "reason": "model changed since that generation"}
        cfg = self.model.config
        # Not every causal LM has these. Pure state-space and RNN models
        # (Mamba, RWKV) have no attention heads to report, and reading the key
        # unguarded turned "this architecture has no attention" into an
        # AttributeError and a 500 — the panel said the tool was broken when
        # the honest answer is that there is nothing here to show.
        n_layers = getattr(cfg, "num_hidden_layers", None)
        n_heads = getattr(cfg, "num_attention_heads", None)
        if n_layers is None or n_heads is None:
            return {
                "available": False,
                "reason": (
                    "this architecture publishes no attention layers or heads, "
                    "so there is no attention to show — state-space and RNN "
                    "models reach here"
                ),
            }
        return {
            "available": True,
            "n_layers": n_layers,
            "n_heads": n_heads,
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
                    raise BadRequest(
                        f"cannot read {variant!r} — expected ablate:LAYER.HEAD"
                    ) from err
                block = self._block(a_layer)
                n_heads = self.model.config.num_attention_heads
                # AblationError is a refusal — its own docstring says "we
                # cannot take this measurement, and we say why rather than
                # guess" — but it is still a plain RuntimeError, and this is
                # the one path that reaches it with no wrap. `ablate_heads`
                # and `attribute_tokens` both translate; this did not, and
                # relied on the server catching RuntimeError. Once that arm
                # became `except Refusal`, an unsupported block here would
                # have turned into a 500. Delete the wrap when
                # ablate.AblationError subclasses Refusal.
                try:
                    head_dim = ablate.head_geometry(block, n_heads)
                    if not 0 <= a_head < n_heads:
                        raise BadRequest(f"head must be in [0,{n_heads})")
                    handles.append(
                        ablate.out_projection(block).register_forward_pre_hook(
                            ablate._cut(a_head, head_dim, "zero")
                        )
                    )
                except ablate.AblationError as err:
                    raise Refusal(
                        str(err)
                    ) from err  # leak-ok: authored, see test_no_machine_leaks
            elif variant == "steered":
                if self._steer is None or self.sae is None:
                    raise Refusal(
                        "Nothing is being steered, so there is no steered run "
                        "to compare against. Set a feature and a scale first."
                    )
                handles.append(self._steer_handle())
            elif variant != "live":
                raise BadRequest(f"unknown variant {variant!r}")

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
            raise Refusal("Generate something first, then inspect attention.")
        if self.last_ids_epoch != self.epoch:
            raise Refusal(
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
            raise BadRequest(f"layer must be in [0,{n_layers}), head in [0,{n_heads})")

        matrix = captured[layer][head].to(torch.float32)
        return {
            "layer": layer,
            "head": head,
            "variant": variant,
            "tokens": self._attn_tokens,
            # Where the prompt ends and the model's own output starts.
            #
            # Without it the strip is one undifferentiated row of chips, and
            # the panel has no meaningful token to rest on. The last prompt
            # token is the right one: its row answers "what did the model look
            # at to decide its first word", and on a long generation it is
            # still on screen, where the final token — 24,000px to the right
            # inside a scroll container — is not.
            "n_prompt": int(self.last_n_prompt_tokens or 0),
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
            raise Refusal(
                "You are viewing a recording. Close it to compare two runs of "
                "your own model."
            )
        self._ready_for_attention()

        with self._lock:
            left, right = self._capture(a), self._capture(b)

        n_layers, n_heads = len(left), left[0].shape[0]
        if not (0 <= layer < n_layers and 0 <= head < n_heads):
            raise BadRequest(f"layer must be in [0,{n_layers}), head in [0,{n_heads})")

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

            def _post(module, args, output):
                tup = isinstance(output, tuple)
                hidden = output[0] if tup else output
                moved = hidden + scale * direction.to(hidden.dtype)
                return ((moved,) + tuple(output[1:])) if tup else moved

            return block.register_forward_hook(_post)

        def _pre(module, args):
            hidden = args[0]
            return (hidden + scale * direction.to(hidden.dtype),) + args[1:]

        return block.register_forward_pre_hook(_pre)

    def _require_live_generation(self, nothing_yet: str) -> None:
        """There is a generation, and it belongs to the model now loaded.

        CALL THIS WITH `self._lock` HELD. That is the whole reason it exists:
        both intervention methods used to take these two checks before
        acquiring the lock, and `load` holds the same lock across the epoch
        bump and the model swap. A load that lands in the window between the
        check and the acquisition then hands the intervention one model's
        token ids and another model's weights, and it returns a confident
        ranking rather than the refusal the identical call gets one moment
        later. Nothing downstream can notice that: the ids are the right
        length, the KLs are finite, and the layer and head numbers exist in
        both models.
        """
        if not self.loaded or self.last_ids is None:
            raise Refusal(nothing_yet)
        if self.last_ids_epoch != self.epoch:
            raise Refusal(
                "That generation was produced by a different model. Generate again."
            )

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
            raise Refusal(
                "This is a recording. Ranking heads means running the model, "
                "and a `.mri` does not carry one."
            )
        if self.backend == "ollama":
            raise Refusal(
                "Ollama serves text only — there is no forward pass to "
                "intervene in. Load the model through HuggingFace."
            )
        with self._lock:
            self._require_live_generation(
                "Generate something first, then rank its heads."
            )
            cfg = self.model.config
            n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
            if layer is not None and not 0 <= layer < n_layers:
                raise BadRequest(f"layer must be in [0,{n_layers})")
            layers = list(range(n_layers)) if layer is None else [layer]

            # Attribute at the last prompt token: its next-token distribution
            # is the model's answer to the question, before any of its own
            # output feeds back in.
            size = int(self.last_ids.shape[0])
            position = max(0, min(self.last_n_prompt_tokens - 1, size - 1))
            ids = self.last_ids.unsqueeze(0).to(self.device)

            extra: dict = {}
            if baseline == "resample":
                extra = self._resample_donors(ids, layers)

            try:
                ranked = ablate.rank_heads(
                    self.model,
                    self._block,
                    ids,
                    position=position,
                    layers=layers,
                    n_heads=n_heads,
                    baseline=baseline,
                    decode=lambda t: self.tokenizer.decode([t]),
                    **extra,
                )
                # The corpus is part of a resample measurement -- `corpus.py`
                # says so at length -- so its label rides in the receipt, and
                # only for the baseline that actually drew from it.
                ranked["receipt"] = self.receipt(
                    "ablate_heads",
                    layer=layer,
                    baseline=baseline,
                    position=position,
                    corpus=extra.get("corpus"),
                )
                return ranked
            except ablate.AblationError as err:
                # Not a crash: a shape this code cannot read honestly.
                raise Refusal(
                    str(err)
                ) from err  # leak-ok: authored, see test_no_machine_leaks

    def receipt(
        self,
        op: str,
        *,
        seed: int | None = None,
        prompt: str | None = None,
        **request,
    ) -> dict:
        """Stamp what produced a number, file it, and hand it back.

        Returns the receipt as a plain dict so a caller can attach it to its
        own result without importing the dataclass.

        A FAILING RECEIPT NEVER FAILS A MEASUREMENT. The provenance is worth
        a great deal and it is still worth strictly less than the number it
        describes -- a tokenizer that will not serialise, or a cache directory
        that has been evicted mid-run, must not turn a completed ablation
        sweep into a 500. The individual fields already answer None with a
        reason for the cases that can be anticipated; this catches the ones
        that cannot, and records the failure in the receipt itself rather than
        returning an empty dict that reads as "no provenance was collected".
        """
        try:
            stamped = receipts.stamp(
                self, op, request=request, seed=seed, prompt=prompt
            ).to_dict()
        except Exception as err:  # pragma: no cover - defensive
            log.warning("receipt for %s could not be taken", op, exc_info=err)
            stamped = {
                "op": op,
                "request": {},
                "could_not_stamp": (
                    f"the setup of this measurement could not be read back "
                    f"({type(err).__name__}), so this number travels without one"
                ),
            }
        self._receipts[op] = {**stamped, "epoch": self.epoch}
        return stamped

    def _resample_donors(self, ids, layers: list[int]) -> dict:
        """Capture `RESAMPLE_DRAWS` donor activations, or refuse saying why.

        Costs one forward pass per draw, on top of the sweep itself. Called
        with `self._lock` already held by `ablate_heads`.
        """
        sentences, label = corpus.load()
        size = int(ids.shape[-1])
        donor_ids = corpus.donor_ids(
            self.tokenizer,
            sentences,
            at_least=size,
            want=ablate.RESAMPLE_DRAWS,
            device=self.device,
        )
        donors = [
            ablate.capture_projection_inputs(self.model, self._block, d, layers)
            for d in donor_ids
        ]
        return {"donors": donors, "corpus": label}

    def estimate_ablation(
        self, layer: int | None = None, baseline: str = "zero"
    ) -> dict:
        """What would this sweep cost here? One probe pass, then arithmetic.

        Exists because `baseline="resample"` is `RESAMPLE_DRAWS` times the work
        of the other two — 98 passes against 14 for one gpt2 layer — and the
        panel should be able to say so before the user waits for it rather
        than after.
        """
        if self.replay is not None:
            raise Refusal(
                "This is a recording. Estimating a sweep means running the "
                "model, and a `.mri` does not carry one."
            )
        if self.backend == "ollama":
            raise Refusal(
                "Ollama serves text only — there is no forward pass to "
                "intervene in. Load the model through HuggingFace."
            )
        with self._lock:
            self._require_live_generation("Generate something first.")
            cfg = self.model.config
            n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
            if layer is not None and not 0 <= layer < n_layers:
                raise BadRequest(f"layer must be in [0,{n_layers})")
            layers = list(range(n_layers)) if layer is None else [layer]

            size = int(self.last_ids.shape[0])
            position = max(0, min(self.last_n_prompt_tokens - 1, size - 1))
            out = ablate.estimate_cost(
                self.model,
                self._block,
                self.last_ids.unsqueeze(0).to(self.device),
                position=position,
                layers=layers,
                n_heads=n_heads,
                baseline=baseline,
                device_kind=self.accel.kind,
            )
            if baseline == "resample":
                # The draws multiply the sweep, and the donor captures are one
                # extra pass each. Both are real cost and neither is in the
                # single-baseline projection.
                out["estimate"]["passes"] = (
                    len(layers) * n_heads * ablate.RESAMPLE_DRAWS
                    + ablate.RESAMPLE_DRAWS
                    + 2
                )
                probe_seconds = out["probe"]["seconds"]
                out["estimate"]["seconds"] = round(
                    probe_seconds * out["estimate"]["passes"], 2
                )
                out["estimate"]["notes"] = list(out["estimate"].get("notes", [])) + [
                    f"{ablate.RESAMPLE_DRAWS} draws per head, plus one capture "
                    "pass per draw"
                ]
            return out

    def adopt_step(self, step: dict) -> dict:
        """Point every mechanistic panel at a generation an agent already made.

        This is the join nothing else in the category can build, and the reason
        is structural rather than clever: LangSmith, Langfuse, Phoenix,
        Braintrust, Weave, Opik and Laminar all stop at the API boundary and
        none of them ever holds the weights. ModelMRI has the recorder and the
        model in one process, so a recorded step that ran on this machine can
        be re-established as "the last generation" and every existing panel —
        attention, lens, ablation, patching, SAE — works on it unchanged.

        `step` is a row from `TraceStore.get_trace`. What makes it adoptable is
        `meta.input_ids`: the exact ids the recorder saw. They are checked
        against this tokenizer rather than trusted, because adopting ids the
        loaded model would not produce points every downstream panel at a
        sequence the model never saw — and none of those panels would notice.

        There is deliberately **no substitute-model path**. Replaying a hosted
        model's prompt through whatever happens to be loaded, however loudly
        labelled, is a machine for confident wrong conclusions.
        """
        meta = step.get("meta") or {}
        recorded = meta.get("input_ids")
        if not recorded:
            raise Refusal(
                "this step was not produced by a model on this machine, so "
                "there are no weights here to look inside. Steps recorded "
                "through instrument_transformers() or instrument_ollama() "
                "carry the token ids that make this possible; a hosted API "
                "call cannot."
            )

        wanted = str(meta.get("model") or "")
        if self.replay is not None:
            raise Refusal(
                "This is a recording. Adopting a step means running a model, "
                "and a `.mri` does not carry one."
            )
        if self.backend == "ollama":
            raise Refusal(
                "Ollama serves text only — there is no forward pass for the "
                "panels to read. Load this model through HuggingFace to adopt "
                "the step."
            )

        with self._lock:
            if self.model is None:
                raise Refusal(
                    f"no model is loaded. This step ran on {wanted or 'a local model'}"
                    " — load it first, then adopt the step."
                )
            if wanted and self.hf_id and wanted != self.hf_id:
                raise Refusal(
                    f"this step was produced by {wanted} and {self.hf_id} is "
                    "loaded. Load the model that made it — reading one model's "
                    "token ids through another model's weights produces numbers "
                    "about nothing, and no panel here would show that it had."
                )

            prompt = str(meta.get("prompt") or step.get("input") or "")
            retokenised = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
            recorded_ids = [int(t) for t in recorded]

            # The prompt's ids, not the whole recorded sequence: the recording
            # holds prompt + generation, and re-tokenising the prompt can only
            # reproduce the prompt half.
            # `or` treated a recorded 0 as absent, so an empty prompt fell
            # back to len(retokenised) — also 0 — and the id-verification guard
            # below degenerated to `[] != []` and never fired. The step then
            # adopted with n_prompt_tokens 0, and every panel's
            # `max(0, min(n_prompt - 1, size - 1))` collapsed to position 0
            # while the response claimed the ids had been verified.
            recorded_n = meta.get("n_prompt_tokens")
            n_prompt = len(retokenised) if recorded_n is None else int(recorded_n)
            if n_prompt <= 0:
                raise Refusal(
                    "this step recorded a prompt of zero tokens, so there is "
                    "nothing to verify the recorded ids against and no position "
                    "for the panels to attribute at. Re-record it with a "
                    "non-empty prompt."
                )
            if [int(t) for t in retokenised.tolist()] != recorded_ids[:n_prompt]:
                raise Refusal(
                    f"re-tokenising this step's prompt gives {len(retokenised)} "
                    f"ids and the recorder captured {n_prompt}, and they do not "
                    "match. A tokenizer or transformers upgrade between the "
                    "recording and now is the usual cause. Refusing rather than "
                    "adopting near-identical ids, which would point every panel "
                    "at a sequence the model never saw."
                )

            ids = torch.tensor(recorded_ids, dtype=torch.long)
            self.last_ids = ids
            self.last_prompt = prompt
            self.last_n_prompt_tokens = n_prompt
            self.last_user_span = None
            self.last_ids_epoch = self.epoch
            # Everything derived from the PREVIOUS generation has to go, or a
            # stale attention capture would be rendered against these tokens.
            # Same discipline the load path uses.
            self._attn_variants = {}
            self._attn_tokens = None
            self._last_patch = {}
            # `_feats` too. Every other rebase path clears it; this one did
            # not, and `_compute_features` guards its cache on
            # `last_ids_epoch == epoch` — which adopt satisfies — so the
            # PREVIOUS generation's [S, d_sae] activations were returned
            # against the adopted tokens. Reproduced: 6 adopted tokens against
            # a cached (2, 16), with features_summary publishing 6 tokens and
            # 2 rows of activations belonging to a different sequence.
            self._feats = None
            self.last_telemetry = None

        return {
            "adopted": True,
            "model": wanted or self.hf_id,
            "step_id": step.get("id"),
            "kind": step.get("kind"),
            "n_tokens": len(recorded_ids),
            "n_prompt_tokens": n_prompt,
            "prompt": prompt,
            "generation": self.tokenizer.decode(recorded_ids[n_prompt:]),
            "means": (
                "Every panel is now reading the generation this agent step "
                "actually made. Nothing was re-run — these are the recorded "
                "token ids, verified against this tokenizer."
            ),
        }

    def control_ranking(
        self, layer: int | None = None, baseline: str = "zero", seed: int = 0
    ) -> dict:
        """The same ranking on an untrained twin, and how far the two agree.

        Runs `ablate.rank_heads` twice over the same token ids — once on the
        loaded model, once on the same architecture with random weights — and
        reports the rank correlation between them. Both go through the same
        function, deliberately: a second implementation of the measurement
        could differ from the one being checked, and then agreement would mean
        nothing in either direction.
        """
        if self.replay is not None:
            raise Refusal(
                "This is a recording. A control means running a second model, "
                "and a `.mri` does not carry one."
            )
        if self.backend == "ollama":
            raise Refusal(
                "Ollama serves text only — there is no forward pass to "
                "intervene in. Load the model through HuggingFace."
            )

        # The real ranking first, and through the public method, so it obeys
        # every gate (live generation, layer range, baseline validation) rather
        # than duplicating them here.
        real = self.ablate_heads(layer, baseline)

        with self._lock:
            cfg = self.model.config
            n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
            layers = list(range(n_layers)) if layer is None else [layer]
            ids = self.last_ids.unsqueeze(0).to(self.device)
            size = int(self.last_ids.shape[0])
            position = max(0, min(self.last_n_prompt_tokens - 1, size - 1))

            twin = nullmodel.build_twin(
                cfg, seed=seed, dtype=self.model.dtype, device=self.device
            )
            try:
                extra: dict = {}
                if baseline == "resample":
                    # Donors must come from the TWIN's own activations. A
                    # trained model's activations spliced into an untrained one
                    # would be a third experiment, not a control.
                    sentences, label = corpus.load()
                    donor_ids = corpus.donor_ids(
                        self.tokenizer,
                        sentences,
                        at_least=size,
                        want=ablate.RESAMPLE_DRAWS,
                        device=self.device,
                    )
                    extra = {
                        "donors": [
                            ablate.capture_projection_inputs(
                                twin, lambda i: self._block_of(twin, i), d, layers
                            )
                            for d in donor_ids
                        ],
                        "corpus": label,
                    }

                control = ablate.rank_heads(
                    twin,
                    lambda i: self._block_of(twin, i),
                    ids,
                    position=position,
                    layers=layers,
                    n_heads=n_heads,
                    baseline=baseline,
                    decode=lambda t: self.tokenizer.decode([t]),
                    **extra,
                )
            except ablate.AblationError as err:
                raise Refusal(
                    str(err)
                ) from err  # leak-ok: authored, see test_no_machine_leaks
            finally:
                nullmodel.teardown(twin)

        agreement = ablate.compare_baselines(
            {"model": real["ranked"], "untrained": control["ranked"]}, top=5
        )
        pair = agreement["pairs"][0] if agreement["pairs"] else {}
        return {
            "seed": seed,
            "baseline": baseline,
            "model": real,
            "untrained": control,
            "spearman": pair.get("spearman"),
            "top_k": pair.get("top_k", 0),
            "top_k_shared": pair.get("top_k_shared", 0),
            "verdict": nullmodel.verdict(
                pair.get("spearman"),
                top_k_shared=pair.get("top_k_shared", 0),
                top_k=pair.get("top_k", 0),
            ),
        }

    def _block_of(self, model, index: int):
        """The transformer block at `index` on an arbitrary model.

        `self._block` is bound to the loaded model. The twin has the same
        architecture, so the same lookup rules apply — but they have to be
        applied to it rather than to `self.model`.
        """
        for path in (
            "model.layers",
            "transformer.h",
            "gpt_neox.layers",
            "model.decoder.layers",
        ):
            node = model
            for part in path.split("."):
                node = getattr(node, part, None)
                if node is None:
                    break
            if node is not None:
                return node[index]
        raise ablate.AblationError(
            "cannot find the transformer blocks on the untrained twin, so "
            "there is no control to compare against."
        )

    def compare_baselines(self, layer: int | None = None) -> dict:
        """Run every baseline on the same layer and report how much they differ.

        The panel has always shown one baseline at a time with nothing saying
        the others existed. Measured on gpt2 layer 0 the three disagree badly —
        Spearman 0.34 to 0.47, and the top five share only two or three heads —
        so which one is selected has been quietly deciding the answer.
        """
        rankings = {}
        for name in ablate.BASELINES:
            rankings[name] = self.ablate_heads(layer, name)["ranked"]
        out = ablate.compare_baselines(rankings, top=5)
        out["rankings"] = rankings
        out["receipt"] = self.receipt(
            "compare_baselines", layer=layer, baselines=list(ablate.BASELINES)
        )
        return out

    def attribute_tokens(self, position: int | None = None) -> dict:
        """Rank the prompt's own tokens by how far masking one moves the answer.

        The companion question to `ablate_heads`: that one asks which part of
        the machinery mattered, this one asks which part of the input did. The
        measurement is `attribute.rank_tokens`, along with the reasons index 0
        and the attribution position are excluded and why the scores do not add
        up. Everything here is the part that needs the live tokenizer.

        Cost is `tested tokens + 7` forward passes: six inside the ranking plus
        one here, to read the model's own answer at `position`. Measured through
        the endpoint on this machine: 10 on gpt2 with "The capital of France is"
        (3 tested), 20 on gemma-3-270m-it (13 tested), 23 on Qwen3-0.6B
        attributing at token 17 (16 tested). The count is the portable part. The
        seconds are not, and on one RTX 4060 they did not even transfer between
        those three: warm and back to back, 0.12-0.14 s, 0.84-0.92 s and
        1.00-1.04 s, or roughly 13, 44 and 44 ms a pass. The first call after a
        load pays CUDA warm-up on top — 0.40 s for the same gpt2 work. So
        `passes` and `elapsed_s` both come back and the caller derives a rate on
        its own machine rather than trusting one from mine.

        `position` defaults to the last prompt token — the same expression
        `ablate_heads` uses, and for the same reason: that distribution is the
        model's answer to the question, before any of its own output feeds back
        in.

        **The scores move if you generated more, and the amount is not small.**
        Nothing after `position` can reach it through a causal mask, so this
        ought to be exactly invariant, and in bfloat16 it is not: the pass runs
        over the whole retained sequence and a longer one reduces in a different
        order. Measured on gemma-3-270m-it, same prompt, same position 14, same
        13 candidates, only the generation length differing — '\\n' at index 11
        scored 9.33529 after one generated token and 9.57509 after six, and
        index 0 went 0.88465 -> 0.98639, an 11% move. Rows are comparable to
        each other inside one response. They are not comparable across two
        generations of different lengths, and no rounding hides that.
        """
        if self.replay is not None:
            raise Refusal(
                "This is a recording. Attributing tokens means masking one and "
                "running the model again, and a `.mri` does not carry one."
            )
        if self.backend == "ollama":
            raise Refusal(
                "Ollama serves text only — there is no forward pass to mask a "
                "token out of. Load the model through HuggingFace."
            )
        with self._lock:
            # Inside the lock, all of it. `load` holds this same lock across
            # the epoch bump and the model swap, so a check taken outside it
            # is a check against a state that can be gone by the time the
            # first forward pass runs. Scripted with two toy models: the
            # epoch check passed, the call blocked on the lock, `load` bumped
            # the epoch 1 -> 2 and swapped the weights, and attribution then
            # returned a full ranking of model A's token ids under model B —
            # while the identical call one moment later refused. A ranking
            # attributed to the wrong model is the one output here that
            # nothing downstream can catch.
            self._require_live_generation(
                "Generate something first, then ask which of its tokens mattered."
            )
            size = int(self.last_ids.shape[0])
            if position is None:
                position = max(0, min(self.last_n_prompt_tokens - 1, size - 1))
            if not 0 <= position < size:
                raise BadRequest(
                    f"position must be in [0,{size}) — that generation is "
                    f"{size} tokens long."
                )

            ids = self.last_ids.unsqueeze(0).to(self.device)
            # Deliberately not the wider detector this feature was first
            # specified with, `all_special_ids | additional_special_tokens |
            # ^<\|?.+\|?>$`. The loose regex is the problem: measured here, it
            # claims 6581 ids on gemma-3-270m-it against 10 for
            # `control_token_ids`, and among the extras are '<div>' 224, '<b>'
            # 200, '<html>' 217, '<table>' 168 and '<li>' 223 — ordinary
            # content in that vocabulary. The refusal below would then fire on
            # a real answer and call it formatting, which is the one failure
            # this whole check exists to prevent. The narrower set still
            # catches Qwen3's '<think>' 151667, and only its chat-template arm
            # does.
            control = attribute.control_token_ids(self.tokenizer)
            started = time.perf_counter()

            # One pass of our own before the ranking, because the refusal below
            # has to quote a probability that was measured rather than a
            # constant carried over from another model on another day. This is
            # the plain `model(ids)` call every other reader of this model
            # makes; rank_tokens re-runs it and refuses if an explicit mask and
            # position_ids move the answer at all, so the p quoted here and the
            # `p_top_before` in the rows are the same number to within 1e-6.
            with torch.no_grad():
                answer = ablate.distribution(self.model(ids).logits[0, position])
            top = int(answer.argmax())
            if top in control:
                raise Refusal(
                    f"the model's next token here is "
                    f"{self.tokenizer.decode([top])!r} at "
                    f"p={float(answer[top]):.4f}, which is a formatting "
                    "decision, not an answer — asking which of your words "
                    "caused it would rank noise. Pick a token further along "
                    "the strip, where the model is saying something."
                )

            span = self.last_user_span
            n_prompt = int(self.last_n_prompt_tokens or 0)
            try:
                out = attribute.rank_tokens(
                    self.model,
                    ids,
                    position=position,
                    typed_span=span,
                    # Without this every token past the prompt falls outside
                    # `span` and comes back labelled "template" — the model's
                    # own output, filed under a chat template, on models that
                    # have none. Measured on Qwen3-0.6B at position 17: the top
                    # row is index 15 'Okay' at 9.11195, the model's word.
                    n_prompt=n_prompt,
                    control_ids=control,
                    decode=lambda t: self.tokenizer.decode([t]),
                )
            except attribute.AttributionError as err:
                # Not a crash: a measurement this code cannot take honestly.
                raise Refusal(
                    str(err)
                ) from err  # leak-ok: authored, see test_no_machine_leaks

        # The ranking timed itself; this adds the answer-reading pass, so
        # passes and elapsed_s still describe the same piece of work and a rate
        # derived from them is right.
        out["passes"] += 1
        out["elapsed_s"] = round(time.perf_counter() - started, 2)
        out["passes_note"] = (
            f"{out['passes']} forward passes: one to read the model's own "
            f"answer at this position, then a base, a repeat of it for the "
            f"noise floor, a plain model(ids) that gates on agreeing with the "
            f"base, one that reverses position_ids and gates on the answer "
            f"MOVING, "
            f"{out['n_tested']} masked tokens, index 0, one joint mask, "
            f"and one check that masking really empties the column. The count "
            f"transfers between machines; the seconds do not."
        )
        if span is None:
            note = (
                "Cannot locate your words inside the templated prompt; every "
                "token below is shown in one group. That is an unknown, not a "
                "claim that all of them are yours."
            )
        elif span == (0, self.last_n_prompt_tokens):
            # gpt2's case: no chat template, so the prompt is the whole prompt.
            # Saying "the rest is the template" here would invent one.
            note = (
                f"Tokens {span[0]}-{span[1] - 1} are the words you typed, and "
                "that is the entire prompt — this model has no chat template "
                f"wrapped around it. Anything from token {n_prompt} on is the "
                "model's own output rather than yours, and is listed as that."
            )
        else:
            note = (
                f"Tokens {span[0]}-{span[1] - 1} are the words you typed; "
                f"everything else below token {n_prompt} is the chat template "
                f"and everything from {n_prompt} on is the model's own output. "
                "Three lists rather than one because on Qwen3-0.6B the "
                "template's 'assistant' scores 2.02161 nats here against "
                "7.9083e-05 for the strongest word the user typed, so one "
                "list would be a list about the template."
            )
        out["span_note"] = note
        out["receipt"] = self.receipt("attribute_tokens", position=position)
        return out

    def rank_features(
        self,
        position: int | None = None,
        scope: str = "position",
        top_k: int = 64,
    ) -> dict:
        """Rank SAE features by how far removing one moves the answer.

        The third member of the family. `ablate_heads` asks which piece of the
        machinery mattered, `attribute_tokens` asks which piece of the input
        did, and this asks which piece of the SAE's decomposition did — the
        question the features panel appears to answer today and does not: it
        ranks by raw activation, and activation is what fired, not what
        mattered. The measurement, the one defensible intervention and every
        reason the numbers are shaped the way they are live in
        `feature_ablate`; everything here is the part that needs the live
        model, the live SAE and the lock.

        `scope="position"` (the default) puts on trial the features firing at
        the attributed token; `scope="prompt"` removes each feature wherever it
        fires at or before it, which is a different and larger question —
        measured on gpt2 at blocks.8.hook_resid_pre, 4 of that ranking's top-8
        fire ONLY at earlier tokens and reach the prediction through attention,
        so the current panel cannot show them at all.

        `top_k` trims the ROWS IN THE RESPONSE and nothing else. A row it drops
        was tested and scored; `truncated` in the payload means something
        different and worse — a candidate never measured. `sum_of_singles` and
        `joint_kl` stay over every scored row, so trimming the response cannot
        move them.

        `position` defaults to the last prompt token, the same expression
        `ablate_heads` and `attribute_tokens` use and for the same reason: that
        distribution is the model's answer to the question, before any of its
        own output feeds back in.

        Cost is `2 x features tested + 6` forward passes. Two per row, not one:
        the feature's own edit and a random direction of the same norm at the
        same tokens, because part of a score is the SIZE of the edit — a
        Gaussian direction at the top feature's norm costs about 0.09 nats
        against that feature's own 0.417, and 9 of the 43 rows on the reference
        prompt do not clear their own control. Measured through this module on
        this machine, gpt2 in float32 on CPU — which the dtype gate below makes
        the only configuration this answers in — "The Eiffel Tower is located
        in the city of" attributed at position 10: 92 passes and 10.09 s at
        position scope (43 candidates), 518 passes and 49.44 s at prompt scope
        (494 candidates, 256 tested). The pass count is the portable part; the
        seconds are this CPU's. `passes` and `elapsed_s` both come back so a
        caller derives a rate on its own machine, the same contract the other
        two rankings carry.

        **It refuses anything but float32, and that is the one refusal here
        that is not obvious.** `feature_ablate` checks its floor by writing the
        stream back unchanged, which is bit-exact in every dtype and scores 0.0
        in every dtype — so nothing inside the measurement can notice that in
        bfloat16 a 1-ulp change to the stream is worth ~0.01-0.03 nats on its
        own. Measured on gpt2 here, an edit whose true effect is 4.9e-07 nats
        reads 0.02836 in bfloat16 and outranks a feature with 100x its
        activation. The long version, with the numbers and with what float16
        does differently, is on the check itself.

        The other refusals are split across two files on purpose. A recording,
        Ollama, nothing generated yet, a generation from a previous model, no
        SAE and the dtype are all states this object can see, so they are
        refused here. That the SAE does not reconstruct the stream it is
        attached to is only knowable after encoding real activations, so
        `feature_ablate` refuses it, in one message quoting the FVU it
        measured — duplicating that check here would put the same sentence in
        two places and let them drift.
        """
        if self.replay is not None:
            raise Refusal(
                "This is a recording. Ranking features means subtracting one "
                "from the residual stream and running the model again, and a "
                "`.mri` does not carry one."
            )
        if self.backend == "ollama":
            raise Refusal(
                "Ollama serves text only — there is no residual stream to "
                "subtract a feature's direction from. Load the model through "
                "HuggingFace."
            )
        with self._lock:
            # Inside the lock, all of it, for the reason `_require_live_generation`
            # gives at length: `load` holds this same lock across the epoch bump
            # and the model swap, so a check taken outside it is a check against
            # a state that can be gone before the first forward pass runs. This
            # path is worse than the other two if it slips, not better — the
            # ranking would be one model's features under another model's
            # weights, and the SAE below is validated against the model that was
            # loaded when it was loaded.
            self._require_live_generation(
                "Generate something first, then ask which of its features mattered."
            )
            if self.sae is None:
                raise Refusal(
                    "No SAE loaded, so there are no features to remove. POST "
                    "/api/sae/load first — this ranks the same decomposition "
                    "the features panel plots, by causal effect instead of by "
                    "activation."
                )
            # THE DTYPE GATE, AND IT IS NOT PEDANTRY — MEASURED ON THIS BOX.
            #
            # `feature_ablate` verifies its floor by writing the captured
            # stream back UNCHANGED, which is bit-exact in every dtype and
            # scores 0.0 in every dtype. That check cannot see the failure
            # below, which is why this one exists.
            #
            # Same ids, same SAE, same hook, same position; only the model's
            # dtype differs. gpt2, cuda, eager, jbloom/GPT2-Small-SAEs-Reformatted
            # @ blocks.8.hook_resid_pre calibrated centered+b_dec, "The Eiffel
            # Tower is located in the city of Paris", position 10:
            #
            #   feature 9420  activation 0.0003  fp32 -3e-08   bf16 0.00593
            #   feature 16664 activation 0.0131  fp32 -1e-08   bf16 0.01170
            #   feature 3841  activation 0.051   fp32  4.9e-07 bf16 0.02836
            #
            # Those are edits with no causal effect at all — fp32 puts them at
            # the resolution, where they belong — reading in bfloat16 as some
            # of the LARGER scores in the table. 3841 outranks feature 21062,
            # whose activation is 5.49 and whose real effect is 0.0044. The
            # pedestal is the model's own arithmetic: a 1-ulp change to the
            # stream propagates through the remaining blocks in bfloat16 and
            # comes out worth ~0.01-0.03 nats, so it is added to every edit and
            # subtracted from none. Top-8 agreement between the two dtypes is
            # 3 of 8; the singles sum to 0.66446 in fp32 against 1.38186 in
            # bf16, which inverts the payload's own sentence about how they add
            # up (bf16: 1.38 against a joint 2.11, and at prompt scope 5.29
            # against 5.59, i.e. it looks additive when it is not); and the
            # candidate set itself moves, 43 features firing in fp32 against 44
            # in bf16 and 42 in fp16.
            #
            # float16 is better and still not admissible: its top-8 matched
            # fp32's exactly, but its pedestal is ~2e-4 (3841 reads 0.00040
            # against 4.9e-07) and roughly thirty of the 43 real scores are
            # below that, so the tail is still ordered by rounding. What would
            # make either dtype work is a resolution MEASURED per dtype the way
            # the float32 one was, replacing feature_ablate.RESOLUTION_KL —
            # measuring it on one model and one prompt, as here, is not that.
            #
            # A Refusal, not a BadRequest: the request is fine, and what makes
            # the measurement impossible is which model is resident.
            dtype = next(self.model.parameters()).dtype
            if dtype != torch.float32:
                name = str(dtype).removeprefix("torch.")
                raise Refusal(
                    f"this model is loaded in {name}, and a feature ranking "
                    f"taken in {name} is a ranking of rounding error below its "
                    "top two or three rows. Measured here on gpt2 at "
                    "blocks.8.hook_resid_pre, the same 43 features scored in "
                    "both dtypes: feature 3841, activation 0.051, moves the "
                    "answer 4.9e-07 nats in float32 and 0.02836 in bfloat16 — "
                    "an edit with no causal effect reading as one of the "
                    "larger scores in the table, above a feature with 100x its "
                    "activation. Writing the stream back unchanged is still "
                    "bit-exact and still scores 0.0, so the noise floor cannot "
                    "catch this. It works in float32, which ModelMRI selects "
                    "for CPU and never for a GPU: start the server with the "
                    "GPU hidden (`MODELMRI_DEVICE=cpu modelmri serve`, or in "
                    "PowerShell `$env:MODELMRI_DEVICE='cpu'`) and load the "
                    "model again — verified on this machine to select "
                    "cpu/float32."
                )

            if top_k < 1:
                raise BadRequest(f"top_k must be at least 1, got {top_k}.")

            size = int(self.last_ids.shape[0])
            if position is None:
                position = max(0, min(self.last_n_prompt_tokens - 1, size - 1))
            if not 0 <= position < size:
                raise BadRequest(
                    f"position must be in [0,{size}) — that generation is "
                    f"{size} tokens long."
                )

            try:
                out = feature_ablate.rank_features(
                    self.model,
                    self._block(self.sae.layer),
                    self.last_ids.unsqueeze(0).to(self.device),
                    self.sae,
                    position=position,
                    scope=scope,
                    decode=lambda t: self.tokenizer.decode([t]),
                )
            except feature_ablate.FeatureAblationError as err:
                # Not a crash: a measurement this code cannot take honestly —
                # an SAE that does not reconstruct, an edit that does not land
                # where the capture came from, a position where nothing fires.
                # `except Exception` here would swallow a CUDA out-of-memory
                # into a 409, which is the bug errors.py exists to stop.
                raise Refusal(
                    str(err)
                ) from err  # leak-ok: authored, see test_no_machine_leaks

        # Outside the lock: arithmetic on a dict, and no model touched.
        ranked = out["ranked"]
        n_scored = len(ranked)
        n_below = sum(1 for row in ranked if row["below_resolution"])
        # Counted in THIS run rather than quoted from the reference one. A KL
        # cannot be negative, so each of these is float32 summation over the
        # vocabulary showing itself, and it is the evidence for why the line a
        # panel greys out on is `resolution_kl` and not `noise_floor_kl` —
        # measured through this endpoint on gpt2/CPU/float32, the floor is
        # exactly 0.0 while 4 of 43 rows came back below it.
        n_negative = sum(1 for row in ranked if row["kl"] < 0)
        out["ranked"] = ranked[:top_k]
        out["n_returned"] = len(out["ranked"])
        out["n_scored"] = n_scored
        out["n_below_resolution"] = n_below
        out["n_negative_kl"] = n_negative
        out["rows_note"] = (
            f"{len(out['ranked'])} of {n_scored} scored rows are in this "
            f"response, the highest-scoring ones. A row left out here WAS "
            f"tested and scored, which is not what `truncated` means — that is "
            f"a candidate never measured at all. sum_of_singles and joint_kl "
            f"are over all {n_scored} scored rows and do not move when this "
            f"list is trimmed. {n_below} of {n_scored} scored below "
            f"resolution_kl ({out['resolution_kl']:g} nats), where the number "
            f"is arithmetic rather than a measurement, and {n_negative} came "
            f"back NEGATIVE, which a KL cannot be — that is float32 summation "
            f"over the vocabulary, and it is why the line to grey out on is "
            f"the resolution and not noise_floor_kl."
        )
        # The SAE is as much a part of this measurement as the model is: the
        # same prompt through a different SAE ranks different features, so a
        # receipt that named only the model would describe half the setup.
        sae = self.sae
        out["receipt"] = self.receipt(
            "rank_features",
            position=position,
            scope=scope,
            top_k=top_k,
            sae_repo=getattr(sae, "repo", None) if sae else None,
            sae_hook=getattr(sae, "hook", None) if sae else None,
        )
        return out

    # ---------------- sessions (.mri) ----------------

    # One byte per attention value before gzip. 24 MB of them is already a
    # large thing to attach to a message; past that we export the cross
    # through the view you are on instead of the full cube, and say so.
    _FULL_EXPORT_BUDGET = 24_000_000

    def export_session(self, layer: int = 0, head: int = 0, note: str = "") -> bytes:
        """Serialise the current analysis to a `.mri` someone else can open."""
        if self.replay is not None:
            raise Refusal(
                "You are viewing a shared session. Close it, then generate "
                "something of your own to export."
            )
        if self.backend == "ollama":
            raise Refusal(
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
        # THE NAME, NOT THE PATH. `hf_id` is a Hub id for a Hub model and an
        # absolute filesystem path for one loaded from a local folder
        # (discover.py sets `id=str(d)`). A `.mri` is the one artefact in this
        # project designed to LEAVE the machine — people attach them to issues
        # — so publishing the raw id shipped `C:\Users\<their real name>\...`
        # to whoever they sent it to, and `modelmri open` then printed it on
        # the recipient's terminal. The no-machine-leaks test never loaded a
        # folder model, so nothing caught it.
        shared_id = self.hf_id or ""
        try:
            if shared_id and Path(shared_id).exists():
                shared_id = Path(shared_id).name
        except OSError:
            # A malformed path is not a reason to fail an export; the name is
            # metadata, and an unreadable one is better dropped than leaked.
            shared_id = Path(shared_id).name if shared_id else ""

        return session.build(
            model_id=shared_id,
            device=self.device,
            dtype=str(next(self.model.parameters()).dtype).removeprefix("torch."),
            n_params=sum(p.numel() for p in self.model.parameters()),
            tokens=tokens,
            prompt=self.last_prompt,
            generation="".join(tokens[cut:]),
            attention={(li, hi): captured[li][hi] for li, hi in wanted},
            n_prompt=cut,
            n_layers=n_layers,
            n_heads=n_heads,
            note=note,
            scope=scope,
            # The causal result, when one was measured against THIS run. A
            # `.mri` carried attention and the logit lens, so the one finding
            # in this tool that is causal rather than correlational was the
            # one you could not send anybody.
            patch=self._patch_for_export(),
            receipts=self._receipts_for_export(),
        )

    def _receipts_for_export(self) -> list[dict]:
        """Every measurement's setup, for the ones this export describes.

        Filtered on the epoch for the same reason `_patch_for_export` is: a
        receipt naming the model that WAS loaded, written into a file about
        the one that is, would be a falsehood in the single field the format
        exists to make trustworthy. A receipt from a superseded epoch is
        dropped rather than relabelled -- it describes a real measurement, it
        just does not describe THIS one.
        """
        return [
            {k: v for k, v in receipt.items() if k != "epoch"}
            for receipt in self._receipts.values()
            if receipt.get("epoch") == self.epoch
        ]

    def _patch_for_export(self) -> dict:
        """The last patch trace, if it belongs to the state being exported.

        The epoch moves on every load, unload and generation. Without that
        check a trace measured on an earlier prompt -- or on a model since
        swapped out -- would be written into the file beside a different
        run's tokens, and nothing downstream could tell.
        """
        last = self._last_patch
        if not last or last.get("epoch") != self.epoch:
            return {}
        return {k: v for k, v in last.items() if k != "epoch"}

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
            # So the panel can offer the recorded trace instead of a button
            # that can only refuse.
            "patch": {
                "available": self.replay.has_patch(),
                "clean": self.replay.patch.get("clean", ""),
                "corrupt": self.replay.patch.get("corrupt", ""),
            },
        }

    # ---------------- SAE features ----------------

    def sae_status(self) -> SAEStatus:
        if self.sae is None:
            return SAEStatus(loaded=False)
        return self.sae.status()

    # ------------------------------------------------------------------ GGUF

    def plan_gguf(self, path: str, *, dtype: str | None = None) -> dict:
        """What loading this GGUF would cost, without loading it.

        Cheap on purpose: it reads the header, which is a few hundred
        kilobytes of a multi-gigabyte file, and returns both memory figures.
        The point is that "this will not fit" is answerable before the
        download rather than twenty minutes into it.
        """
        from . import gguf_load

        report = gguf_load.plan(
            path,
            dtype=dtype or self.accel.dtype,
            device_kind=self.accel.kind,
        ).to_dict()
        # Whether this file IS the model currently held. Added here rather than
        # in gguf_load, which knows nothing about sessions.
        #
        # Without it the panel tells you a model you are already running will
        # not fit — and it is not wrong: its own weights are in the free-memory
        # figure it just read, so loading a second copy really would fail. It
        # is still the wrong sentence to show, because the question the reader
        # is asking has already been answered by the thing in front of them.
        current = (self.gguf or {}).get("plan", {}).get("path")
        report["already_loaded"] = bool(
            current and Path(current) == Path(report["path"])
        )
        return report

    def load_gguf(
        self, path: str, *, dtype: str | None = None, confirm: bool = False
    ) -> ModelStatus:
        """Load a GGUF as a full torch module — every panel, not just chat.

        The Ollama backend already runs GGUF files, at their real bit width and
        fast, and cannot show you a single attention head. This is the other
        trade: transformers dequantises the weights, which costs about three
        times the file on disk, and in exchange the model is an ordinary module
        that the lens, the ablation sweep and the patching grid all work on.

        Which of those you want is a real choice, so the cost is named before
        the load rather than discovered during it.

        Blocking — call from a worker thread.
        """
        from . import gguf_load

        want = dtype or self.accel.dtype
        with self._load_slot(f"gguf {Path(path).name}"):
            # start_external, NOT start. `start` spawns a watcher that polls
            # the HuggingFace cache for a directory named after its argument
            # and calls HfApi().model_info() on it -- so a local filename went
            # out as a live hub lookup for a repo called
            # "loading Qwen3-0.6B-Q4_K_M.gguf", once per load, and the bar then
            # sat at 0 bytes watching a directory that cannot exist. Nothing
            # here downloads: the file is already on disk and the cost is CPU.
            progress.TRACKER.start_external(
                Path(path).name, stage="dequantise", detail="reading the header"
            )
            try:
                loaded = gguf_load.load(
                    path,
                    dtype=want,
                    device=self.device,
                    device_kind=self.accel.kind,
                    confirm=confirm,
                    on_stage=progress.TRACKER.stage,
                )
            except BaseException as err:
                # finish() before re-raising, or the progress meter stays
                # "active" for the rest of the session and its watcher thread
                # polls forever. Same bug the HF path already fixed.
                progress.TRACKER.finish(error=_load_failed(err))
                raise
            progress.TRACKER.finish()

            report = loaded.to_dict()
            self.epoch += 1
            self.backend = "hf"
            self.gguf = report
            self.model = loaded.model
            self.tokenizer = loaded.tokenizer
            # The file name, not a hub id. Naming it after the repo would let
            # a .mri recorded from a Q4_K_M file claim to come from the
            # full-precision model of the same name -- which is exactly the
            # substitution this module exists to prevent.
            self.hf_id = Path(loaded.plan.path).name
            self.replay = None
            self.last_ids = None
            self.last_user_span = None
            self._attn_variants.clear()
            self._attn_tokens = None
            self.sae = None
            self._feats = None
            self._steer = None
            self._ollama_instruct = None
            return self.status()

    def compare_quantisation(
        self, quantised: str, original: str, prompt: str, *, want_attention: bool = True
    ) -> dict:
        """What quantisation cost this model's behaviour, on one prompt.

        UNLOADS whatever is currently held first. Two models never sit in
        memory at once — and the currently-loaded one is a third, so it has to
        go too. On the 8 GB card this was written on, keeping it would mean the
        comparison only runs for models small enough to fit three times, which
        excludes every model anyone would want to compare.

        Blocking — call from a worker thread.
        """
        from . import behavdiff

        with self._load_slot("quantisation comparison"):
            # Before the slot does anything else: the caller's model is dead
            # weight for this measurement and its memory is what makes the
            # measurement possible.
            if self.model is not None:
                log.info("unloading %s to make room for the comparison", self.hf_id)
                self.epoch += 1
                self.model = None
                self.tokenizer = None
                self.hf_id = None
                self.backend = None
                self.gguf = None
                self.replay = None
                self.last_ids = None
                self.last_user_span = None
                self._attn_variants.clear()
                self._attn_tokens = None
                self.sae = None
                self._feats = None
                self._steer = None
                gc.collect()
                self._empty_accel_cache()

            progress.TRACKER.start_external(
                "quantisation comparison", stage="load", detail="first model"
            )
            try:
                result = behavdiff.compare_behaviour(
                    quantised,
                    original,
                    prompt,
                    dtype=self.accel.dtype,
                    device=self.device,
                    device_kind=self.accel.kind,
                    want_attention=want_attention,
                    on_stage=progress.TRACKER.stage,
                )
            except BaseException as err:
                progress.TRACKER.finish(error=_load_failed(err))
                raise
            progress.TRACKER.finish()
            return result.to_dict()

    def load_sae(self, repo: str, hook: str) -> SAEStatus:
        """Load an SAE and validate it against the current model. Blocking."""
        if self.backend == "ollama":
            raise Refusal(
                "SAE features need model internals — unavailable via Ollama. "
                "Load a HuggingFace model instead."
            )
        if not self.loaded:
            raise Refusal("Load a model first.")
        sae = SAEHandle.load(repo, hook)
        d_model = self.model.config.hidden_size
        if sae.d_in != d_model:
            raise BadRequest(
                f"SAE d_in={sae.d_in} does not match model hidden_size={d_model} "
                f"({self.hf_id}). This SAE was trained on a different model."
            )
        n_layers = self.model.config.num_hidden_layers
        if not 0 <= sae.layer < n_layers:
            raise BadRequest(f"SAE layer {sae.layer} out of range [0,{n_layers})")
        self._block(sae.layer)  # raises early if architecture unsupported
        self.sae = sae
        self._feats = None
        self._steer = None
        return sae.status()

    def _compute_features(self) -> torch.Tensor:
        """[S, d_sae] feature activations for the last generation (cached)."""
        if self.sae is None:
            raise Refusal("No SAE loaded. POST /api/sae/load first.")
        if self.last_ids is None:
            raise Refusal("Generate something first.")
        if self.last_ids_epoch != self.epoch:
            raise Refusal(
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
                def _capture(module, args, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    captured.append(hidden.detach())

                handle = block.register_forward_hook(_capture)
            else:

                def _capture(module, args):
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
                raise Refusal(
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
                    # `strict`: topk returns values and indices of the same
                    # shape, so a length mismatch here would mean the pairing
                    # of feature id to activation had already gone wrong, and
                    # zip's default is to truncate and publish it anyway.
                    for fid, act in zip(id_row, act_row, strict=True)
                    if act > 0
                ]
                for id_row, act_row in zip(ids.tolist(), acts.tolist(), strict=True)
            ],
        }

    def feature_detail(self, feature_id: int) -> dict:
        """One feature's activation across the last generation's tokens."""
        feats = self._compute_features().float()
        if not 0 <= feature_id < feats.shape[1]:
            raise BadRequest(f"feature_id must be in [0,{feats.shape[1]})")
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
                raise Refusal("No SAE loaded.")
            if not 0 <= feature_id < self.sae.d_sae:
                raise BadRequest(f"feature_id must be in [0,{self.sae.d_sae})")
            self._steer = (feature_id, float(scale))
        return self.steering_status()

    def steering_status(self) -> dict:
        if self._steer is None:
            return {"active": False}
        fid, scale = self._steer
        return {"active": True, "feature_id": fid, "scale": scale}
