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
from dataclasses import asdict, dataclass, field
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
    grammar,
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

# How long the weight prefetch may make NO progress before it is stopped.
#
# The stall bound rather than a plain deadline, because bytes arriving means
# the download is working however long it takes — a 40 GB checkpoint on a slow
# line must not be killed for being big. Reset on every byte.
PREFETCH_STALL_SECONDS = 120.0

# And a ceiling, for a child that reports progress it is not making. Generous:
# this only has to be shorter than "forever", which is what it replaced.
PREFETCH_MAX_SECONDS = 3600.0


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

    Two questions are asked, in this order, because the architecture STRING is
    the weaker answer and it used to be the only one. `load()` calls
    `AutoModelForCausalLM.from_pretrained` a few lines below, so the authority
    on "can the playground run this" is what that class actually builds, not
    how the checkpoint spells its own class name. Every multimodal Gemma spells
    it `...ForConditionalGeneration`, and `AutoModelForCausalLM` maps every one
    of them straight back to that same class. Measured on transformers 5.14.1,
    this guard refused google/gemma-4-E4B-it, gemma-4-E2B-it, gemma-4-12B-it,
    both gemma-4 w4a16 QAT builds and gemma-3-4b-it -- models the very next
    line loads without complaint. Two code paths were answering one question
    differently. The single Gemma 4 build that did get through,
    gemma-4-E4B-it-qat-mobile-transformers, got through by accident: its config
    publishes no `architectures` key at all, so it fell into the "unknown
    shape" arm rather than being judged.

    The mapping is consulted for the DECLARED class, never as a bare "is this
    model_type in the mapping" test. Those are different questions and the
    loose one is wrong: bert-base-uncased declares `BertForMaskedLM` while the
    mapping holds `BertLMHeadModel` for it, so `AutoModelForCausalLM` will
    cheerfully build an encoder with an untrained LM head and generate fluent
    nonsense from it. Checked against the repos above plus vit, SmolVLM2 and
    bert: the strict form unblocks the Gemmas and loosens nothing else.
    """
    from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(hf_id)
    except Exception:
        return  # let the real loader produce the real error

    archs = list(getattr(cfg, "architectures", None) or [])
    if any(a.endswith(("ForCausalLM", "LMHeadModel")) for a in archs):
        return
    if not archs:
        return  # unknown shape: don't block on a guess

    try:
        mapped = MODEL_FOR_CAUSAL_LM_MAPPING.get(type(cfg), None)
    except Exception:
        # `_LazyAutoMapping.get` imports the modeling module for this one
        # model_type, so it fails on whatever an import fails on: a
        # transformers built without that model, or an optional dependency the
        # module needs at import time. None is the honest answer -- we could
        # not establish that the loader builds this -- and refusing on that is
        # the safe direction, because the suffix test above has already
        # declined to vouch for it. A guard whose whole job is to replace a
        # traceback must not raise one of its own.
        mapped = None
    if mapped is not None and mapped.__name__ in archs:
        return

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
            f"Ungated alternatives: Qwen/Qwen3-0.6B, Qwen/Qwen2.5-0.5B-Instruct, "
            f"Qwen/Qwen3-1.7B."
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
    """The user stopped a load. Not an error in the code, and not silent.

    Carries `sentence` for the reason `Refusal` and `BadRequest` do, written
    out in `errors.py`: the route publishes this text to a browser, and
    `str()` on an exception is whatever the raiser happened to pass —
    including, on a subclass somebody adds later, a tuple repr or a library's
    own words. `sentence` is set once, here, from the authored argument, so
    what reaches the reader is provably something a person wrote. CodeQL
    raised `str(err)` on the image sibling as py/stack-trace-exposure, and it
    is right about the shape even where today's raisers happen to be careful.
    """

    def __init__(self, *args):
        super().__init__(*args)
        #: Empty for `LoadCancelled()` with no argument — the caller is then
        #: saying nothing, and inventing a sentence for them would be worse.
        self.sentence = str(args[0]) if args else ""


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


# `move_to_device`, `resident_bytes` and the publish interval live in
# `device_move.py` and are re-exported here. They moved because
# `gguf_load` needs `move_to_device` and importing it from this module was
# the ONE back-edge that made the package's import graph cyclic -- CodeQL
# reported `py/cyclic-import` on seven deferred imports, all of which were
# fine at load time and none of which was the cause. The function never
# needed anything from this module, so a leaf module is where it belonged.
#
# Re-exported rather than relocated-and-rewritten because
# `runtime.move_to_device` is the name eight tests and two call sites
# already use, and moving a function is not a reason to move its callers.
# The redundant `as` is the re-export spelling, not a typo: PEP 484's
# explicit form, which says "this name is part of THIS module's surface"
# to every tool that reads it. A bare F401 suppression said it to ruff
# alone -- and a per-line ruff suppression is invisible to CodeQL, which
# answered with `py/unused-import` on the very next scan.
from .device_move import DEVICE_PUBLISH_EVERY_S as DEVICE_PUBLISH_EVERY_S  # noqa: E402
from .device_move import move_to_device as move_to_device  # noqa: E402
from .device_move import resident_bytes as resident_bytes  # noqa: E402


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
#
# `original/` is Meta's own checkpoint format, shipped beside the transformers
# one -- and it belongs in THIS list rather than under the conditional below,
# where only `*.pth` reached it. That condition needs a repo file listing, and
# its documented failure mode is to fetch everything rather than guess: one
# unlucky listing and a Llama repo pulls its whole second copy. Measured in
# this machine's cache, which holds both:
# `meta-llama/Llama-3.2-1B-Instruct/model.safetensors` at 2,471,645,608 bytes
# and `original/consolidated.00.pth` at 2,471,677,246 -- 4.9 GB on disk for a
# 2.5 GB model.
#
# It also settles a disagreement between the fetcher and the meter.
# `progress._default_keep` counts top-level files only, naming "Meta's
# original/" as a variant `from_pretrained` never reads, so the bar was
# already drawing a 2.5 GB denominator over a fetch that could pull 4.9 GB.
# Both sides now describe the same set of files.
ignore = [
    "*.h5", "*.msgpack", "*.tflite", "*.onnx", "*.onnx_data", "*.gguf",
    "*.ot",
    "onnx/*", "coreml/*", "openvino/*", "tflite/*", "original/*",
]

# `pytorch_model.bin` is byte-for-byte redundant when safetensors is present
# -- transformers loads the safetensors and never opens the .bin -- so
# fetching both doubles the transfer for nothing. A repo that also ships a
# rust_model.ot pulls a third identical copy, for several times the bytes
# that were needed.
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
    from a span starting at character 0 — which is the span a base model with no
    chat template gets, and the case the
    rule would have been written for. An explicit `b <= a: continue` was here
    and was removed after a mutation test showed it could not change any
    answer: it only ever fires on an index between two overlapping ones, and a
    middle index moves neither end of a range.

    Verified against both, prompt "The capital of France is" through the
    same chat template `generate_stream` applies: Qwen3-0.6B (3, 8) of 13,
    gemma-3-270m-it (5, 10) of 15 —
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
    # nonsense" — a base model continues your sentence, it does not answer your
    # question, and a UI that does not say so invites the reader to conclude
    # the tool is broken when the tool is working exactly as intended.
    instruct: bool | None = None
    # The load plan, when this model was built from a GGUF: file size against
    # resident size, the expansion between them, and the standing caveat that
    # every measurement below describes the quantised weights. None otherwise,
    # never {} -- an empty dict reads as "from a GGUF, nothing to say".
    gguf: dict | None = None
    # How many transformer blocks this model has. A property of the LOADED
    # MODEL, not of a run -- which is why it lives here and not on
    # /api/attention/meta, where it needs a generation to have happened first.
    # The probe and patchscope panels both need to offer a layer before there
    # is anything to generate from, and without this their layer pickers were
    # empty until the user generated something they did not need.
    #
    # None for backends with no blocks to count (ollama serves text only).
    n_layers: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DirectionSteer:
    """A saved contrastive direction, installed on the live model.

    THE SECOND ARM OF A TAGGED UNION. `_steer` holds the SAE-feature form and
    has since steering existed; this holds the form that needs no SAE, which
    is almost every model. They are mutually exclusive on purpose — two live
    interventions at once would each be reported without the other, and the
    A/B beside them would name one of the two.

    Kept as its own object rather than squeezed into `_steer`'s tuple because
    `_capture`'s cache key is built from whatever is here, and a key that does
    not carry everything the answer depends on is exactly the defect
    `tests/test_steered_attention_cache.py` exists to pin.

    `vector` is UNIT NORM and already on the model's device — `steer_vectors.load`
    returns a CPU float32 tensor and does no device work, unlike
    `SAEHandle.steering_vector`, so the move happens once here rather than
    inside a forward hook.

    `residual_norm` is what the stream measured at `layer` when this was
    applied, or None when there was no generation to measure it on. None is
    an unknown, never a zero: a relative strength of 0.0 says the push is
    negligible, which is the one thing an absent measurement must not say.

    `measured_on` is the `last_ids` tensor that norm was taken on, held so
    `strength_report` can tell whether it still describes the generation in
    front of the reader. THE IDENTITY, NOT A COPY OR A COUNTER: every path
    that sets `ModelRuntime.last_ids` builds a fresh tensor (`generate_stream`
    detaches one, `adopt` constructs one), so `is` answers exactly the
    question being asked and cannot be forgotten by a future assignment site
    the way a serial number could.
    """

    name: str
    layer: int
    strength: float
    vector: Any
    origin_model: str
    residual_norm: float | None
    #: How the norm above was taken, or "" when it was not taken at all.
    measured: str
    #: Why it was not taken, or "" when it was.
    unmeasured: str
    warnings: list[str] = field(default_factory=list)
    #: The generation `residual_norm` was measured on, or None when it was not
    #: measured at all.
    measured_on: Any = None

    def strength_report(self, current_ids: Any) -> dict:
        """The published strength, and where its denominator came from.

        `current_ids` is REQUIRED and has no default. A default of None would
        read as "no generation" against a `measured_on` that holds one, so a
        caller who simply forgot the argument would be told the number is old
        — quietly, and in a sentence that sounds authored. There is one caller
        and it has `last_ids` in its hand; making it say so costs nothing.

        THE NORM IS MEASURED ONCE, AT APPLY TIME, and the direction outlives
        the generation it was applied during on purpose — generating under it
        is the point of applying it. So the number stays true and the SENTENCE
        beside it expires: `residual_norm_at` writes "the current generation
        ... just now", and this is re-read and re-rendered on every generation.

        A real measurement wearing a fresh prompt's provenance is worse than
        no measurement, because nothing on screen says to distrust it. So the
        number is kept — it was genuinely taken, on this model, at this layer
        — and the sentence is replaced by one that names which generation it
        belongs to and what to do about it. Re-measuring here instead would
        put a forward pass behind every status poll, which is precisely what
        `residual_norm_at` refuses to do.
        """
        from . import steer_vectors

        measured = self.measured
        if self.residual_norm is not None and current_ids is not self.measured_on:
            measured = (
                f"the mean L2 norm of the residual stream entering layer "
                f"{self.layer}, measured on this machine at the last token of "
                f"the generation that was in front of you when this direction "
                f"was applied. A newer generation has replaced that one, so "
                f"this describes the earlier prompt rather than the one on "
                f"screen now — re-apply the direction to measure it against "
                f"this generation."
            )
        return {
            "alpha": self.strength,
            "relative": steer_vectors.relative_strength(
                self.strength, self.residual_norm
            ),
            "residual_norm": self.residual_norm,
            "layer": self.layer,
            "measured": measured,
            "unmeasured": self.unmeasured,
        }


def text_config(cfg):
    """The sub-config describing the LANGUAGE tower.

    A text-only config describes it directly. A multimodal one (Gemma 3 and 4,
    PaliGemma, Qwen-VL) has no shape of its own at the top level and nests one
    config per modality, because "how many layers" has three answers for those
    models and the tool is asking about the text one.

    Returns `cfg` unchanged when there is nothing nested, so every caller can
    use it unconditionally rather than each deciding for itself -- which is
    how the fourteen call sites for this ended up disagreeing.
    """
    if getattr(cfg, "num_hidden_layers", None) is not None:
        return cfg
    inner = getattr(cfg, "text_config", None)
    return cfg if inner is None else inner


def decoder_blocks(root):
    """The ModuleList of decoder blocks, or None if this layout is unknown.

    Ordered most specific first: a multimodal model has BOTH
    `model.language_model.layers` and `model.vision_tower.encoder.layers`, and
    picking the wrong one draws attention maps for the image encoder while the
    panel says they are the text model's.
    """
    for path in (
        ("transformer", "h"),  # GPT-2 family
        ("model", "language_model", "layers"),  # Gemma 3/4, PaliGemma
        ("language_model", "model", "layers"),
        ("model", "layers"),  # Llama, Qwen, Mistral, Gemma 1/2
    ):
        node = root
        for name in path:
            node = getattr(node, name, None)
            if node is None:
                break
        else:
            if hasattr(node, "__len__") and len(node):
                return node
    return None


def _recorded_receipt(replay, op: str) -> dict | None:
    """The receipt for one recorded measurement, out of the file's own list.

    `session.build` writes a run's receipts as ONE list, and every export
    helper strips the copy nested inside its section -- `_patch_graph_for_export`
    says why: "a second copy nested inside a section would be a second answer
    to the question of what produced these numbers." That is the right call,
    and the other half of it was never built. Nothing served the list: it
    appears nowhere in `server.py` and nowhere in the viewer, so `ReceiptLine`
    -- whose own docstring says it "renders receipts from a `.mri` a stranger
    sent" -- rendered nothing on every recorded measurement, and the file's
    provenance was carried the whole way and shown to no one.

    Matched on `op`, which is the field a receipt uses to say what it is a
    receipt FOR. The LAST one wins: a session that ran the same measurement
    twice describes the numbers it kept, which are the second run's.

    `None` rather than an empty dict when there is none. The panel renders
    nothing for a missing receipt and a sentence for a present one, and an
    empty dict would be a receipt claiming a run with no model, no revision
    and no dtype -- provenance that says nothing, which is worse than an
    absence that says so.
    """
    for receipt in reversed(getattr(replay, "receipts", None) or []):
        if isinstance(receipt, dict) and receipt.get("op") == op:
            return receipt
    return None


def _recorded_patch(recorded: dict) -> dict:
    """A stored patching section in the shape a LIVE trace answers in.

    The panel has one renderer, and it reads `data.clean.answer.text`,
    `data.corrupt.tokens[...]` and `data.components`. The file stores the
    prompts flat -- `clean` and `corrupt` are strings there, which is what the
    CLI and `session_info` read -- so something has to put the two shapes back
    together, and doing it here means the panel never learns that a recording
    is different from a run.

    Absent parts stay absent rather than becoming empty strings: a file
    written before the strip was carried has no token labels, and inventing
    blank ones would draw a grid whose columns are labelled with nothing.
    """
    tokens = recorded.get("tokens") or {}
    answers = recorded.get("answers") or {}
    return {
        **{
            k: v
            for k, v in recorded.items()
            if k not in ("clean", "corrupt", "tokens", "answers")
        },
        "clean": {
            "prompt": recorded.get("clean") or "",
            "tokens": tokens.get("clean") or [],
            "answer": answers.get("clean") or {},
        },
        "corrupt": {
            "prompt": recorded.get("corrupt") or "",
            "tokens": tokens.get("corrupt") or [],
            "answer": answers.get("corrupt") or {},
        },
        "components": recorded.get("components") or sorted(recorded.get("grids") or {}),
    }


class ModelRuntime:
    """Owns the loaded model; thread-safe load, streaming generate, attention."""

    # DECLARED ON THE CLASS, and that is deliberate rather than tidy. The
    # second arm of the steering union is read by `_capture` and
    # `generate_stream`, and `tests/test_steered_attention_cache.py` builds a
    # runtime with `ModelRuntime.__new__` and six hand-set attributes — a
    # weightless fixture that runs in milliseconds and is the reason the
    # cache-key regression stays caught. An instance-only attribute would make
    # every read in that fixture an AttributeError, and the fix would have
    # been to teach the fixture about a field it has nothing to do with.
    _steer_dir: DirectionSteer | None = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # NOT the lock. Generation streams, so it decodes on a daemon thread
        # and holds no lock -- this says whether that thread is running
        # forward passes, which is what any hook-installing measurement has to
        # know. See `decoding`.
        self._decoding = threading.Event()
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
        # The PATCHING graph, kept separate from `_last_patch` because it
        # is a separate `.mri` section and a separate measurement.
        self._last_patch_graph: dict = {}
        self._last_ranking: dict = {}
        self._last_lens: dict = {}
        # Head type labels. Tagged with the epoch and NOT cleared on
        # generation, unlike everything else here -- deliberately. These are
        # measured on random sequences of the module's own making and say
        # nothing about the current prompt, so a new generation does not
        # invalidate them. A model change does, and the epoch moves on load
        # and unload and not on generation, which is exactly that rule.
        self._last_types: dict = {}
        # The last grounding run. Epoch-scoped like the head labels and for
        # the same reason: it is measured on ITS OWN document and question,
        # not on the current generation, so a new generation does not
        # invalidate it -- but a model swap does, and the epoch moves on load
        # and unload and deliberately not on generation.
        self._last_ground: dict = {}
        # The last finetune-vs-base comparison. NOT epoch-scoped, and that is
        # the one piece of state here that is not: a diff names its own two
        # models and is a claim about neither the model loaded here nor the
        # prompt in front of it. Loading a third model does not invalidate it.
        self._last_model_diff: dict = {}
        # A trained translator, when one has been fitted or loaded for THIS
        # model. Cleared on every load: a lens fitted to one model reads
        # another one's residual stream as confident nonsense.
        self._tuned: dict = {}
        self._tuned_info: dict = {}
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
        # The grammar enforcer's vocabulary table for the CURRENT model, built
        # once. `grammar.tokenizer_data` walks the whole vocabulary -- 50,257
        # tokens on gpt2, two decodes each -- and rebuilding it per request
        # would cost more than the generation it is measuring. Keyed on
        # `epoch`, which is the counter that moves on load and unload and
        # deliberately not on generation, so a model swap invalidates it and
        # nothing else does.
        self._grammar_data = None
        self._grammar_data_epoch = -1
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
        # The union's other arm: a saved contrastive direction, which needs no
        # SAE. See `DirectionSteer` for why it is a second slot rather than a
        # widened tuple.
        self._steer_dir: DirectionSteer | None = None

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
            # TWO getattrs, not one. `self.model.config` is not a given: a
            # TorchScript module, an adapter-loaded network and the stub the
            # load tests use all have parameters and no config, and reaching
            # through it raised AttributeError from inside `status()` -- which
            # is the one method every route calls, so a missing block count
            # took the whole session endpoint down rather than reporting an
            # unknown. Caught by the load test, not by review.
            #
            # `or None` on the outside: a missing count is "unknown", and 0
            # would be the positive claim "this model has no blocks", which
            # the UI renders as an empty dropdown.
            n_layers=(
                int(
                    getattr(
                        text_config(getattr(self.model, "config", None)),
                        "num_hidden_layers",
                        0,
                    )
                    or 0
                )
                or None
            ),
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
            self._steer_dir = None

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

    def decoding(self) -> bool:
        """Is a generation worker running forward passes right now?

        `self._lock` cannot answer this. Generation streams, so it runs
        `model.generate` on a daemon thread and holds no lock for the duration
        -- which is correct for streaming and wrong for anything that installs
        forward hooks on the shared blocks, because those hooks catch whatever
        pass happens next. See `patch.trace`, where a concurrent decode step
        produced a one-token cache and an IndexError.
        """
        return self._decoding.is_set()

    def _refuse_if_decoding(self, what: str) -> None:
        """Refuse a hook-installing measurement while a generation decodes.

        Named cause, named next step, like every other refusal here. The
        alternative is not "it works anyway": it is a measurement that
        silently reads somebody else's forward pass.
        """
        if self._decoding.is_set():
            raise Refusal(
                f"A generation is still running, and {what} reads activations "
                f"by installing hooks on the model -- it would capture the "
                f"generation's forward passes instead of this measurement's. "
                f"Wait for the run to finish, then ask again."
            )

    def _in_flight(self, snap: progress.Snapshot) -> str:
        """A sentence describing the load already running, or "" if none is.

        Reads the progress tracker rather than the lock, because the lock can
        only answer "held" and the useful answer is *what* is holding it and
        for how long.

        TAKES the snapshot rather than reading one. Both halves of a refusal
        are built from it, and taking one each let the two halves describe two
        different instants — a load could finish between them, so the sentence
        named a model that was no longer loading and then advised about a phase
        it was no longer in.
        """
        if not snap.active:
            return ""
        return (
            f"'{snap.hf_id}' has been loading for {snap.elapsed_s:.0f}s "
            f"({snap.stage}: {snap.detail})"
        )

    def _way_out(self, snap: progress.Snapshot) -> str:
        """What the reader can actually DO about the load in the way.

        STAGE-DEPENDENT, because Stop is not always an escape and saying it is
        would be a promise the code does not keep. `load` checks
        `TRACKER.cancelled` once, immediately before
        `AutoModelForCausalLM.from_pretrained` — so a stop lands while the
        weights are still arriving, and does nothing at all once they have.
        `from_pretrained` and `model.to(device)` are single calls into
        transformers and torch; neither takes a cancellation token, and a
        10 GB checkpoint on a smaller card sits in exactly that phase for
        minutes.

        The first version of this sentence said "Press Stop — this is the one
        control that works while a load is running", which is true for half of
        a load's life and false for the half somebody is most likely to be
        staring at.

        `snap` is the caller's, shared with `_in_flight` — see there.
        """
        if not snap.active:
            # `active` FIRST, before any field is read as evidence. Every
            # branch below infers a phase from `stage` and the byte counts,
            # and on an inactive tracker those are the dataclass defaults:
            # "" and 0. `bytes_total == 0` is documented as "unknown" in
            # progress.py, so a blank snapshot fell straight through to the
            # terminal branch and told the reader the weights had finished
            # arriving, the load was inside transformers, and restarting the
            # server was the way out — when nothing had ever loaded and no
            # Hub request had been made.
            #
            # Not a narrow race. `unload()` takes this slot and never starts
            # the tracker, so `active` is False for its whole body — epoch
            # bump, teardown, `gc.collect()`, cache clear — which scales with
            # the model being freed, i.e. exactly when a second click lands.
            # Measured over 90s of alternating load/unload requests: 38
            # rounds, 10 of them given this false diagnosis.
            #
            # So: say what is actually known (the slot is held, nothing is
            # reporting) and give the one step that fits either way.
            return (
                "Nothing is reporting progress, though, so there is no live "
                "load behind it — usually the tail of an unload, which keeps "
                "the slot while it frees the memory and never publishes to "
                "the meter. There is nothing to Stop; try again in a few "
                "seconds."
            )
        if snap.stage == "resolving":
            return (
                "Press Stop on the progress bar to end it — nothing has "
                "started downloading yet."
            )
        # THE BYTES, not the stage. `stage` stays "weights" from the first
        # request through to the end of `from_pretrained`, so it cannot tell
        # "still arriving" from "arrived, now loading" — and those are exactly
        # the two halves that differ in whether Stop does anything. An earlier
        # version of this branched on the stage and told a reader to press
        # Stop in the phase where Stop is ignored.
        if snap.bytes_total > 0 and snap.bytes_done < snap.bytes_total:
            return (
                "Press Stop on the progress bar to end it — the weights are "
                "still arriving, which is the phase a stop can interrupt."
            )
        return (
            "The weights have finished arriving and this load is now inside "
            "transformers, which offers no way to interrupt it — Stop will "
            "not end this phase. It will finish or fail on its own; if it has "
            "been far longer than the model's size warrants, the model is "
            "probably larger than this machine can hold and restarting the "
            "server is the way out."
        )

    @contextmanager
    def _load_slot(self, what: str) -> Iterator[None]:
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
            # `what` IS THE ACTION, not a model id. It used to be called
            # `hf_id` and interpolated as "Cannot load '{hf_id}'", which is
            # true for one of the four callers and gibberish for the rest:
            # pressing Unload while a load ran answered "Cannot load 'unload'
            # yet", and the quantisation comparison answered "Cannot load
            # 'quantisation comparison' yet".
            #
            # Unload matters most here, because it is the button somebody
            # reaches for when a load is taking too long — and it is held
            # behind the very lock they are trying to get out from under. It
            # cannot simply skip the lock: freeing the model a load is halfway
            # through writing is worse than waiting. So the refusal names the
            # button that DOES work.
            #
            # ONE snapshot for both halves, so the sentence describes one
            # instant. The fallback says only what the lock proved — that the
            # slot is held. It used to say "another load is already running",
            # which is a claim about the tracker, and the tracker is precisely
            # what has nothing to report on this branch: `_in_flight` returns
            # "" only when `active` is False, so that fallback asserted a
            # running load exactly when there was none to name.
            snap = progress.TRACKER.snapshot()
            busy = self._in_flight(snap) or "the load slot is already held"
            raise Refusal(f"Cannot {what} yet: {busy}. {self._way_out(snap)}")
        try:
            yield
        finally:
            self._lock.release()

    def load(
        self,
        hf_id: str = DEFAULT_MODEL,
        source: str = "hf",
        confirm: bool = False,
        device: str = "",
    ) -> ModelStatus:
        """Load a model. source="hf" (full introspection) or "ollama" (text only).

        `confirm=True` overrides the size guard — the user has been told the
        numbers and chosen anyway.

        `device` names where to put it — "cuda:1", "cpu", or "" to keep doing
        what this has always done and let `devices.detect()` choose. Empty is
        the default on purpose: a machine with one GPU behaves exactly as
        before, and nobody has to learn a new argument to get the old
        behaviour.

        It is READ BACK rather than assumed. `devices.detect(prefer=...)`
        returns the device it could actually honour, which is not always the
        one asked for — "cuda:3" on a two-card box comes back as CPU with a
        reason saying so. Storing the request instead of the result is how a
        panel ends up reporting a card the model is not on.

        Blocking — call from a worker thread.
        """
        # Re-resolved on EVERY load, including the ones that name nothing.
        # An earlier version only touched `self.accel` when a device was
        # named, so a single deliberate CPU load stuck: the next load with no
        # device went to the CPU too, on a machine with a working GPU, and the
        # only symptom was everything being slower for the rest of the
        # session. "Let the tool choose" has to mean choosing, not remembering
        # somebody else's choice.
        chosen = devices.detect(prefer=device or "auto")
        self.accel = chosen
        self.device = chosen.torch_device
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
                self._steer_dir = None
                self._tuned = {}
                self._tuned_info = {}
                self._last_types = {}
                self._last_ground = {}
            return self.status()

        with self._load_slot(f"load {hf_id!r}"):
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
                progress.TRACKER.finish(error=str(err), cancelled=True)
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
                move_to_device(model, self.device)
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
                    # Reported like the first attempt, and for a stronger
                    # reason: this one is twice the bytes at float32 and runs
                    # after the reader has already watched an attempt fail.
                    # Leaving the counters where the GPU move abandoned them
                    # would have frozen the bar part-way through a retry that
                    # is moving a different, larger model.
                    model = model.to(torch.float32)
                    move_to_device(model, self.device)
                except Exception as cpu_err:
                    # float32 on CPU needs roughly twice the VRAM figure that
                    # just failed, so this is the *likely* path for a big
                    # model, not the exotic one. Uncaught, it escaped before
                    # TRACKER.finish() ran: the progress meter stayed "active"
                    # for the rest of the session and its watcher thread
                    # polled the disk forever.
                    # MEMORY IS ONE CAUSE, not the only one, and both sinks
                    # asserted it unconditionally under two bare `except`
                    # arms. Reachable with no hardware fault at all: a repo
                    # leaving any parameter on `meta` raises
                    # `NotImplementedError: Cannot copy out of meta tensor` on
                    # the GPU move, and the CPU retry raises the same — so the
                    # reader was told to try a smaller model, and the 1.7B ->
                    # 0.5B retry failed identically.
                    #
                    # Read by class NAME: `torch.OutOfMemoryError` moved
                    # between versions and this must not import torch to
                    # decide how to word a sentence.
                    out_of_memory = "OutOfMemoryError" in (
                        type(err).__name__,
                        type(cpu_err).__name__,
                    )
                    # BOTH exceptions logged. Only `err` was, so the CPU
                    # failure — the one that decided the outcome — left no
                    # trace anywhere.
                    log.warning(
                        "loading %s failed on %s (%s) and on CPU (%s)",
                        hf_id,
                        self.accel.name,
                        type(err).__name__,
                        type(cpu_err).__name__,
                        exc_info=cpu_err,
                    )
                    if out_of_memory:
                        finish_note = (
                            f"{type(err).__name__} on {self.accel.name}, then "
                            f"{type(cpu_err).__name__} on CPU: not enough "
                            f"memory for this model"
                        )
                        said = (
                            f"'{hf_id}' does not fit: {type(err).__name__} on "
                            f"GPU, then {type(cpu_err).__name__} on CPU. Try a "
                            f"smaller model."
                        )
                    else:
                        finish_note = (
                            f"{type(err).__name__} on {self.accel.name}, then "
                            f"{type(cpu_err).__name__} on CPU"
                        )
                        said = (
                            f"'{hf_id}' could not be placed on "
                            f"{self.accel.name} ({type(err).__name__}), and "
                            f"the CPU fallback failed too "
                            f"({type(cpu_err).__name__}). This is about how "
                            f"the checkpoint loads rather than its size, so a "
                            f"smaller model of the same kind will most likely "
                            f"fail the same way."
                        )
                    progress.TRACKER.finish(error=finish_note)
                    # A Refusal, not a crash: both attempts are named, only
                    # their exception CLASSES are interpolated (never their
                    # text), and the sentence ends with what to do.
                    raise Refusal(said) from cpu_err
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
            self._steer_dir = None
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
        # BOUNDED. This loop had no deadline at all: it polled every 0.4s and
        # waited for the child to exit, forever. `proc.poll()` stays None for a
        # child stuck inside a network read that never returns, so a stalled
        # connection to the Hub hung the load — and, because this runs under
        # `asyncio.to_thread`, hung the request behind it too.
        #
        # MEASURED on this branch's CI: two macOS jobs sat here for 150 and 360
        # minutes and were killed by the runner having produced nothing. The
        # faulthandler dump named the frame — `_prefetch_weights` in
        # `threading.Event.wait` — under `test_a_failed_load_does_not_publish_
        # the_exception`, a test that asks for a model it EXPECTS to be
        # refused. The child went looking for it on the network and never came
        # back.
        #
        # Two bounds rather than one. The stall bound is the honest one: bytes
        # arriving means the download is working however long it takes, and a
        # 40 GB checkpoint on a slow line must not be killed for being big.
        # The ceiling is the backstop for a child that reports progress it is
        # not making.
        #
        # Expiry is NOT an error. This whole method is an optimisation whose
        # documented fallback is "let `from_pretrained` download the old way",
        # so a stalled prefetch returns and the load continues — where the Hub
        # library's own timeouts apply and a real failure raises a real
        # exception instead of waiting. It is logged rather than swallowed:
        # the load that follows will be slower and the terminal says why.
        started = last_change = time.monotonic()
        last_bytes = -1
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

                now = time.monotonic()
                try:
                    arrived = int(progress.TRACKER.snapshot().bytes_done)
                except Exception:
                    # The tracker is a convenience here, not the mechanism. If
                    # it cannot be read, the ceiling below still applies.
                    arrived = last_bytes
                if arrived != last_bytes:
                    last_bytes, last_change = arrived, now

                stalled_for = now - last_change
                if (
                    stalled_for > PREFETCH_STALL_SECONDS
                    or now - started > PREFETCH_MAX_SECONDS
                ):
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    from .capacity import _one_line

                    log.warning(
                        "the prefetch for %s made no progress for %.0fs "
                        "(%.0fs total) and was stopped; the load continues "
                        "through from_pretrained, which is slower",
                        _one_line(hf_id),
                        stalled_for,
                        now - started,
                    )
                    return
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
                    return {
                        **_recorded_patch(recorded),
                        "recorded": True,
                        "receipt": _recorded_receipt(self.replay, "patch_trace"),
                    }
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
            # Before anything installs a hook. `patch.trace` reads the clean
            # run's activations off the shared blocks, and a generation
            # decoding on another thread fills those hooks with its own
            # one-token passes.
            self._refuse_if_decoding("a patching trace")
            # The config's count, not a guess: `_block` raises a Refusal for
            # an architecture it cannot walk, and asking it for layer 0 first
            # turns "unsupported layout" into that message rather than into an
            # IndexError from a range built on the wrong number.
            n_layers = int(text_config(self.model.config).num_hidden_layers)
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
                # THE PROMPTS FLATTENED, AND THE REST KEPT. `result` carries
                # `clean`/`corrupt` as {prompt, tokens, answer}; overwriting
                # them with the bare prompt strings -- which is all the reader
                # preserved -- threw away the token strip that labels the
                # grid's columns and the two answers the panel puts above it.
                # A recipient got the grid and no way to read it: `PatchPanel`
                # asks for `data.corrupt.tokens` and got a string, and its
                # shape guard told them to restart a server they do not have.
                #
                # So the strings stay where the reader and the CLI expect
                # them, and the parts they cannot hold travel beside.
                clean_side = result.get("clean") or {}
                corrupt_side = result.get("corrupt") or {}
                self._last_patch = {
                    **result,
                    "clean": clean,
                    "corrupt": corrupt,
                    "tokens": {
                        "clean": clean_side.get("tokens") or [],
                        "corrupt": corrupt_side.get("tokens") or [],
                    },
                    "answers": {
                        "clean": clean_side.get("answer") or {},
                        "corrupt": corrupt_side.get("answer") or {},
                    },
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
        blocks = decoder_blocks(self.model)
        if blocks is not None:
            # BOUNDS, here rather than at the routes. This is the only place
            # that knows how many blocks the resident model actually has, and
            # the layer arrives from several directions — a path parameter, a
            # JSON field, and `variant=ablate:N` parsed out of a query string,
            # which is the one that had no check at all.
            #
            # MEASURED: GET /api/attention?variant=ablate:9999.0 answered 500
            # with `IndexError: index 9999 is out of range` from inside
            # torch.nn.ModuleList — an error about a container, at somebody
            # who typed a number into a URL.
            #
            # Negative indices are refused rather than quietly wrapping.
            # Python would read -1 as the LAST block, and a reader who meant
            # "layer -1" as "unset" would silently measure the top of the
            # network and be told nothing about it.
            if not isinstance(layer, int) or isinstance(layer, bool):
                raise BadRequest(
                    f"a layer index has to be a whole number, and this asked "
                    f"for {layer!r}."
                )
            if not 0 <= layer < len(blocks):
                raise BadRequest(
                    f"layer {layer} is outside this model, which has "
                    f"{len(blocks)} of them — 0 to {len(blocks) - 1}."
                )
            return blocks[layer]
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
            f"family), model.layers (Llama, Qwen, Gemma 1/2) and "
            f"model.language_model.layers (the multimodal Gemma 3/4 and "
            f"PaliGemma shape). If this "
            f"architecture keeps its blocks somewhere else, open an issue "
            f"with the model id and it becomes one line here."
        )

    def count_tokens(self, text: str) -> int | None:
        """How many tokens `text` is on this backend, or None if it cannot say.

        None is an answer, not a failure. Ollama runs the tokenizer inside its
        own process and `/api/generate`'s stream carries text and nothing
        else, so there is no local tokenizer to ask — and a character count
        dressed up as a token count is a fabricated measurement, which is the
        one thing this tool must not render. The generation trace then carries
        the timing and the text without a token count, which is true.
        """
        tok = self.tokenizer
        if tok is None:
            return None
        try:
            return len(tok.encode(text))
        except Exception:
            # A tokenizer refusing a string is not worth losing anything
            # over: this only ever annotates a recording.
            log.exception("could not count tokens")
            return None

    def _no_grammar_on_ollama(self) -> Refusal:
        return Refusal(
            "Constrained decoding needs the logits of every step, and the "
            "Ollama backend hands this server finished text rather than a "
            "forward pass there is anything to mask. Load this model through "
            "the 'hf' source to enforce a schema, or drop 'response_format' — "
            "this refuses rather than answering a request for structured "
            "output with unconstrained text and saying nothing about it."
        )

    def _tokenizer_data(self):
        """The grammar enforcer's vocabulary table for the loaded model.

        Built once per model, not once per request — see `_grammar_data` in
        `__init__` for what the walk costs. Read and written only from the
        thread that is about to generate; the epoch check is what makes a
        stale table impossible rather than unlikely.
        """
        if self._grammar_data is None or self._grammar_data_epoch != self.epoch:
            self._grammar_data = grammar.tokenizer_data(self.tokenizer)
            self._grammar_data_epoch = self.epoch
        return self._grammar_data

    def mask_recorder(self, schema: dict, temperature: float = 0.0):
        """A `MaskRecorder` enforcing `schema` against the loaded model.

        Separate from `generate_stream` on purpose, and the separation is the
        fix. Every way constrained decoding can say no — no model, the wrong
        backend, an absent optional extra, a schema that cannot be compiled —
        has to be findable BEFORE a caller decides to stream. A Refusal raised
        inside an SSE generator arrives as 200 with a zero-byte body, which an
        OpenAI client reads as a successful empty completion; the same comment
        sits over the `loaded` check in `server._v1_complete` for the same
        measured reason.

        Blocking on first use for a given model: it builds the vocabulary
        table. Cached after that.
        """
        if not self.loaded:
            raise Refusal("No model loaded. POST /api/model/load first.")
        if self.backend == "ollama":
            raise self._no_grammar_on_ollama()
        # Before the vocabulary walk, so an unenforceable schema costs nothing.
        grammar.validate_schema(schema)
        return grammar.MaskRecorder(
            self.tokenizer,
            schema,
            data=self._tokenizer_data(),
            temperature=temperature,
        )

    def _install_grammar(self, gen_kwargs: dict, recorder) -> None:
        """Make the recorder the LAST thing that touches the logits.

        `generate` merges custom processors first and then appends its own
        sampling warpers after them, and says so in a `# TODO (joao): find a
        strategy to specify the order of the processors` right above the
        merge. There is no way to ask for another position, so the only way to
        be last is for nothing else to be there. Three things follow, and all
        three are load-bearing rather than tidy:

        * `top_k` DEFAULTS TO 50 in transformers even when nobody asked for
          it, and a checkpoint's own `generation_config.json` can set `top_p`
          or `typical_p`. Left alone they would truncate the distribution
          after the mask had already been measured against the untruncated
          one, so the receipt would describe a step that did not happen.
        * temperature moves to the recorder (see `grammar.MaskRecorder`), so
          HF's own is neutralised here or it would be applied twice.
        * beams and extra return sequences are forced off: the recorder masks
          and reads row 0, so any other row would come back unmasked and
          unrecorded — several sequences, one of them enforced.

        Which knobs EXIST is asked of the installed transformers rather than
        listed from memory: `top_h` is 5.x-only and `min_p` arrived in 4.45,
        and passing a keyword this version has never heard of is an error on
        the generation, not a warning.
        """
        from transformers import LogitsProcessorList, StoppingCriteriaList

        gen_kwargs["logits_processor"] = LogitsProcessorList([recorder])
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [grammar.ChoiceTap(recorder)]
        )
        gen_kwargs["num_beams"] = 1
        gen_kwargs["num_return_sequences"] = 1

        # Warpers only run under sampling; greedy decoding has none to
        # neutralise, and setting `temperature` beside `do_sample=False` earns
        # a warning from transformers for no gain.
        if not gen_kwargs.get("do_sample"):
            return

        neutral = {
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": None,
            "top_h": None,
            "typical_p": 1.0,
            "epsilon_cutoff": 0.0,
            "eta_cutoff": 0.0,
        }
        cfg = getattr(self.model, "generation_config", None)
        if cfg is None:
            from transformers import GenerationConfig

            cfg = GenerationConfig()
        for name, value in neutral.items():
            if hasattr(cfg, name):
                gen_kwargs[name] = value

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        commit: bool = True,
        *,
        recorder=None,
    ) -> Iterator[str]:
        """Yield generated text pieces. Blocking iterator — consume off the event loop.

        commit=False runs the model without touching the analysis target. The
        steering A/B needs this: it fires two short completions to compare, and
        committing those would silently rebase last_ids onto a 24-token
        sequence while the panels are still showing a 260-token one. Nothing
        errors; the heat map just starts describing a different generation
        than the token strip above it.

        `recorder` is a `grammar.MaskRecorder`, from `mask_recorder` above. It
        both enforces the schema and records what the mask cost, and it must
        see every step in order — which is exactly how a logits processor is
        called, and why nothing here offers a cheaper hook.
        """
        if not self.loaded:
            raise Refusal("No model loaded. POST /api/model/load first.")
        epoch = self.epoch

        if self.backend == "ollama":
            # `mask_recorder` refuses this already, and this is the second
            # gate rather than a duplicate one: `generate_stream` is public,
            # and a caller that built a recorder against an HF model and then
            # reached a runtime that had since been pointed at Ollama would
            # otherwise get unconstrained text under a schema.
            if recorder is not None:
                raise self._no_grammar_on_ollama()
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
            text = prompt  # base models have no chat template
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

        if recorder is not None:
            self._install_grammar(gen_kwargs, recorder)

        result: dict = {}

        def _generate() -> None:
            # HELD FOR EXACTLY AS LONG AS FORWARD PASSES ARE HAPPENING.
            # Set here rather than around the generator below, because
            # `generate_stream` is a generator: a consumer that stops pulling
            # leaves it suspended at `yield`, its `finally` unrun and
            # `worker.join` never reached -- while this thread keeps decoding.
            # The flag has to describe the THREAD, so the thread owns it.
            self._decoding.set()
            try:
                result["ids"] = self.model.generate(**gen_kwargs)
            finally:
                self._decoding.clear()

        # One installer, shared with the attention capture. Two copies of
        # "what steering does" would eventually disagree, and the comparison
        # would then be between a real run and an approximation of one.
        steer_handle = None
        if self._steer_dir is not None or (
            self._steer is not None and self.sae is not None
        ):
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
            n_layers=int(getattr(text_config(cfg), "num_hidden_layers", 0) or 0),
            n_heads=int(getattr(text_config(cfg), "num_attention_heads", 0) or 0),
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
        self._last_patch_graph = {}
        self._last_ranking = {}
        self._last_lens = {}
        # The generation itself gets a receipt, and it is the one every other
        # receipt depends on: each of them names a prompt, and this says how
        # that prompt was answered. `temperature` is the field `verify` cannot
        # work without -- a generation that differs on re-run says nothing
        # about the model if it was sampled, and the `.mri` recorded no
        # sampling configuration at all before this.
        self.receipt(
            "generate",
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            # Named rather than left to be inferred from `temperature > 0`,
            # which is the rule TODAY and is exactly the sort of thing that
            # changes without the reader of a two-year-old file being told.
            greedy=temperature <= 0,
            n_prompt_tokens=self.last_n_prompt_tokens,
            n_generated_tokens=int(self.last_ids.shape[0]) - self.last_n_prompt_tokens,
        )

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
        n_layers = getattr(text_config(cfg), "num_hidden_layers", None)
        n_heads = getattr(text_config(cfg), "num_attention_heads", None)
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
        # A CACHE KEY HAS TO CARRY EVERYTHING THE ANSWER DEPENDS ON.
        # `ablate:L.H` already does -- a different ablation is a different
        # variant string, so it cannot collide. "steered" did not.
        # `set_steering` writes `self._steer` and clears nothing, and every
        # place that clears `_attn_variants` is about the MODEL changing:
        # load, unload, generate, adopt_step, GGUF. So a map measured under
        # feature 100 at scale 5 went on being served, labelled "steered",
        # after the feature was changed to 200 at scale -3 -- and
        # `/api/attention/diff?a=live&b=steered` reported a movement for an
        # intervention that was never run.
        key = variant
        if variant == "steered":
            # BEFORE the lookup, not in the miss path below it. Sitting after
            # the cache made this refusal unreachable in exactly the case it
            # exists for: once a steered map had been cached, switching
            # steering OFF could no longer reach it, and the route kept
            # serving a steered map for a model that was no longer steered.
            #
            # Two arms, two keys, and the direction's key carries name, layer
            # AND strength for the same reason the feature's carries id and
            # scale: anything the answer depends on that is not in the key is
            # a stale map served under a fresh label.
            if self._steer_dir is not None:
                steer = self._steer_dir
                key = f"steered:dir:{steer.name}:{steer.layer}:{steer.strength!r}"
            elif self._steer is None or self.sae is None:
                raise Refusal(
                    "Nothing is being steered, so there is no steered run "
                    "to compare against. Set a feature and a scale, or apply "
                    "a saved direction, first."
                )
            else:
                fid, scale = self._steer
                key = f"steered:{fid}:{scale!r}"

        cached = self._attn_variants.get(key)
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
                n_heads = text_config(self.model.config).num_attention_heads
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
                # The refusal that used to live here moved above the cache
                # lookup, where it can still fire once a map has been cached.
                handles.append(self._steer_handle())
            elif variant != "live":
                raise BadRequest(f"unknown variant {variant!r}")

            with torch.no_grad():
                out = self.model(ids, output_attentions=True)
        finally:
            for handle in handles:
                handle.remove()

        captured = [a[0].detach().to(torch.float16).cpu() for a in out.attentions]
        self._attn_variants[key] = captured
        if self._attn_tokens is None:
            self._attn_tokens = [
                self.tokenizer.decode([tid]) for tid in self.last_ids.tolist()
            ]
        return captured

    def _ready_for_attention(self) -> None:
        # Two states, not one — the same split `attention_meta` makes, for the
        # reason written out above it: they want opposite things from the
        # reader. One is "pick a model", the other is "press the button you
        # are already looking at".
        #
        # Collapsed into `not self.loaded or self.last_ids is None`, a reader
        # with nothing loaded was told to generate something — and following
        # that instruction gave a second refusal, POST /api/model/prompt ->
        # 409 "no model loaded". A next step the reader could not take.
        # Measured on /api/attention, /api/attention/baselines and
        # /api/session/export, which all reach this.
        #
        # `self.loaded`, not `self.model is None`: an Ollama model IS loaded
        # and has no `self.model`, and telling that reader to pick a model
        # would be the same mistake pointed the other way.
        if not self.loaded:
            raise Refusal("No model loaded — pick one first.")
        if self.last_ids is None:
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

        Both arms of the union install through here for that same reason —
        one place that knows what steering does, whichever kind is live.
        """
        if self._steer_dir is not None:
            # A PRE-HOOK, ALWAYS, and at the block whose input is the stream
            # the direction was fitted in. `steer_vectors._last_token_states`
            # captures at `register_forward_pre_hook` on block L, so "layer L"
            # in a saved direction means the residual stream ENTERING L —
            # applying it anywhere else would push a vector through a basis it
            # was not measured in and report the saved layer for it.
            #
            # Constant alpha, on every position, matching the feature arm
            # exactly. The relative figure is a REPORT (see
            # `DirectionSteer.strength_report`), not a rescaling: rescaling
            # per layer would make two runs at "the same strength"
            # incomparable, which is the opposite of what the honesty rule
            # asks for.
            steer = self._steer_dir
            block = self._block(steer.layer)
            direction, strength = steer.vector, steer.strength

            def _direction_pre(module, args):
                hidden = args[0]
                return (hidden + strength * direction.to(hidden.dtype),) + args[1:]

            return block.register_forward_pre_hook(_direction_pre)

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

        `nothing_yet` is the no-generation sentence ONLY. "No model loaded"
        gets its own, for the reason `attention_meta` sets out: the two states
        want opposite things from the reader — "pick a model" against "press
        the button you are already looking at". While they shared one
        sentence, /api/attention/ablate, /api/attention/attribute and
        /api/features/ablate answered an empty runtime with "Generate
        something first", and POST /api/model/prompt answered that with 409
        "no model loaded" — a next step the reader could not take.
        """
        if not self.loaded:
            # `self.loaded`, not `self.model is None`: an Ollama model is
            # loaded and has no `self.model`. Every caller here refuses Ollama
            # before reaching this, but the check should be right on its own.
            raise Refusal("No model loaded — pick one first.")
        if self.last_ids is None:
            raise Refusal(nothing_yet)
        if self.last_ids_epoch != self.epoch:
            raise Refusal(
                "That generation was produced by a different model. Generate again."
            )

    def ablate_heads(self, layer: int | None = None, baseline: str = "zero") -> dict:
        """Rank heads by how far removing one moves the next-token answer.

        `layer=None` sweeps every layer, which is n_layers x n_heads + 2
        forward passes: 450 for Qwen3-0.6B. That count is the
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
            # A recording that CARRIES a ranking can serve it, the same way a
            # recorded patch trace is served. The blanket refusal below was
            # right when the format held nothing here; now that a `.mri`
            # carries the ranking, refusing a file that already holds the
            # answer is the format failing rather than the reader asking for
            # too much -- which is the lesson `patch_trace` records.
            recorded = getattr(self.replay, "ranking", None) or {}
            if recorded.get("ranked"):
                return {
                    **recorded,
                    "recorded": True,
                    # The receipt out of the file's own list. Every section is
                    # exported WITHOUT its own copy, deliberately -- see
                    # `_recorded_receipt` -- and nothing ever read the list they
                    # were consolidated into.
                    "receipt": _recorded_receipt(self.replay, "ablate_heads"),
                }
            raise Refusal(
                "This is a recording, and it does not carry a head ranking. "
                "Ranking heads means running the model once per head, and a "
                "`.mri` holds activations rather than weights — there is "
                "nothing here to re-run. Whoever exported it can rank the "
                "heads and share it again."
            )
        if self.backend == "ollama":
            raise Refusal(
                "Ollama serves text only — there is no forward pass to "
                "intervene in. Load the model through HuggingFace."
            )
        with self._lock:
            # Before anything installs a forward hook. The hooks go on the
            # shared block modules, so a generation decoding on its own
            # thread fills them with one-token passes -- see
            # `_refuse_if_decoding` and `patch.trace`.
            self._refuse_if_decoding("a head ablation")
            self._require_live_generation(
                "Generate something first, then rank its heads."
            )
            cfg = self.model.config
            n_layers, n_heads = (
                text_config(cfg).num_hidden_layers,
                text_config(cfg).num_attention_heads,
            )
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
                # Kept so a `.mri` can carry it, on the same terms as
                # `_last_patch`: tagged with the epoch and dropped when the run
                # it describes stops being the current one. Until this, a file
                # recorded THAT a ranking had run and carried none of it, so
                # `verify` could only report it as the one measurement in the
                # file it was unable to check.
                self._last_ranking = {**ranked, "layer": layer, "epoch": self.epoch}
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
        of the other two — one layer goes from n_heads + 2 passes to
        n_heads * RESAMPLE_DRAWS + 2 — and the
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
            # Before anything installs a forward hook. The hooks go on the
            # shared block modules, so a generation decoding on its own
            # thread fills them with one-token passes -- see
            # `_refuse_if_decoding` and `patch.trace`.
            self._refuse_if_decoding("an ablation cost estimate")
            self._require_live_generation("Generate something first.")
            cfg = self.model.config
            n_layers, n_heads = (
                text_config(cfg).num_hidden_layers,
                text_config(cfg).num_attention_heads,
            )
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
            self._last_patch_graph = {}
            self._last_ranking = {}
            self._last_lens = {}
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
            # Before anything installs a forward hook. The hooks go on the

            # shared block modules, so a generation decoding on its own

            # thread fills them with one-token passes -- see

            # `_refuse_if_decoding` and `patch.trace`.

            self._refuse_if_decoding("a control ranking")
            cfg = self.model.config
            n_layers, n_heads = (
                text_config(cfg).num_hidden_layers,
                text_config(cfg).num_attention_heads,
            )
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
        the others existed. The three can disagree badly — weak rank
        correlation across a layer, and a top five sharing only two or three
        heads — so which one is selected has been quietly deciding the answer.
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

    def train_tuned_lens(
        self,
        texts: list[str],
        *,
        corpus_label: str = "",
        steps: int = 250,
        on_progress=None,
    ) -> dict:
        """Fit a per-layer translator on text you provide, and keep it.

        Cached on disk under (model, dtype, corpus hash, token count), because
        every one of those four changes the lens. A second call with the same
        corpus loads rather than retrains.
        """
        from . import tuned_lens as tl

        if self.replay is not None:
            raise Refusal(
                "This is a recording. Training a lens means running the "
                "model, and a `.mri` does not carry one."
            )
        if self.backend == "ollama":
            raise Refusal(
                "Ollama serves text only — there is no residual stream here "
                "to fit a translator to."
            )
        if self.model is None:
            raise Refusal("No model loaded — pick one first.")

        # A padless tokenizer cannot batch, and base-model tokenizers commonly
        # ship without a pad token.
        # Set on the tokenizer we already hold rather than reloading it.
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = str(next(self.model.parameters()).dtype).removeprefix("torch.")
        sha = tl.corpus_hash(texts)
        n_tokens = sum(len(self.tokenizer(t)["input_ids"]) for t in texts)
        cached = tl.cache_path(self.hf_id or "", dtype, sha, n_tokens)

        if cached.is_file():
            info, state = tl.load(cached, model_id=self.hf_id or "", dtype=dtype)
            self._tuned, self._tuned_info = state, info
            return {**info, "cached": True, "path": cached.name}

        info, state = tl.train(
            self.model,
            self.tokenizer,
            texts,
            corpus_label=corpus_label or f"{len(texts)} sequences",
            steps=steps,
            on_progress=on_progress,
        )
        info.model_id = self.hf_id or ""
        tl.save(info, state, cached)
        self._tuned, self._tuned_info = state, info.to_dict()
        return {**info.to_dict(), "cached": False, "path": cached.name}

    def tuned_lens_status(self) -> dict:
        """Whether a translator is loaded, and what it was fitted to."""
        return {
            "trained": bool(self._tuned),
            **({"info": self._tuned_info} if self._tuned else {}),
        }

    def feature_evidence(
        self,
        texts: list[str],
        *,
        feature_id: int | None = None,
        corpus_label: str = "",
        top_k: int = 10,
    ) -> dict:
        """What a feature fires on in YOUR corpus, and what it promotes.

        Two readouts plus a pointer to the third: `feature_ablate` already
        measures what removing it does, and a claim that survives all three is
        worth something a claim resting on one is not.
        """
        from . import feature_corpus as fc

        with self._lock:
            if self.replay is not None:
                raise Refusal(
                    "This is a recording. Sweeping a corpus means running the "
                    "model, and a `.mri` does not carry one."
                )
            if self.model is None:
                raise Refusal("No model loaded — pick one first.")
            if self.sae is None:
                raise Refusal(
                    "No SAE loaded, so there are no features to show evidence "
                    "for. Load one against this model first."
                )

            layer = int(getattr(self.sae, "layer", 0) or 0)
            block = self._block(layer)
            stats, per_feature = fc.sweep(
                self.model,
                block,
                self.tokenizer,
                self.sae,
                texts,
                device=self.device,
                layer=layer,
                corpus_label=corpus_label,
            )
            saved = fc.save(
                stats,
                per_feature,
                model=self.hf_id or "",
                sae=getattr(self.sae, "repo", ""),
            )
            out = {
                "corpus": stats.to_dict(),
                "saved_rows": saved,
                "top_by_firing_rate": [
                    {"feature": f, "n_fired": n, "max_activation": round(a, 5)}
                    for f, (n, a) in sorted(
                        per_feature.items(), key=lambda kv: -kv[1][0]
                    )[:20]
                ],
                "receipt": self.receipt(
                    "feature_evidence",
                    n_sequences=stats.n_sequences,
                    n_tokens=stats.n_tokens,
                    corpus_sha256=stats.corpus_sha256,
                    sae_repo=getattr(self.sae, "repo", None),
                    layer=layer,
                ),
            }
            if feature_id is not None:
                out["evidence"] = fc.evidence(
                    self.model,
                    block,
                    self.tokenizer,
                    self.sae,
                    texts,
                    feature_id,
                    device=self.device,
                    top_k=top_k,
                )
                out["logit_weights"] = fc.logit_weights(
                    self.model, self.tokenizer, self.sae, feature_id
                )
            return out

    # ------------------------------------------------- how good is this SAE?
    #
    # `SAECalibration` already answers that in ACTIVATION space — FVU and L0,
    # measured on real activations from the model the SAE is attached to. The
    # three methods below answer it in OUTPUT space, which is a different
    # question with a different answer: the directions carrying the residual
    # stream's variance are not the directions the next token depends on, so an
    # SAE can post an excellent FVU and still cost the model most of its
    # predictive loss. `saes.ce_recovered` carries the argument in full.

    def _tokenized_for_ce(self, texts) -> list[torch.Tensor]:
        """One `[1, S]` id tensor per text, IN ORDER, on the model's device.

        Two things here are load-bearing and neither is obvious.

        NOTHING IS DROPPED. `saes._sequence_for_ce` refuses a sequence shorter
        than two tokens by its POSITIONAL INDEX -- "sequence 2 is 1 token(s)
        long" -- and that index is only useful while it still counts the lines
        the reader typed. `head_corpus` skips an empty one (`if
        ids.shape[-1] == 0: continue`), which is right for a sweep that reports
        a total and wrong here: every index after the skip would name a
        different line than the sentence claims.

        A LIST, not a generator. `ce_recovered` reads `sequences` TWICE --
        mean-ablation needs a vector averaged over the whole corpus, which is
        not known until every sequence has been read once -- and a consumed
        iterator makes the second sweep silently empty.

        On `self.device` because `ce_recovered` never moves them: it derives
        the device and dtype from the stream it CAPTURED, and moves only the
        tensor it writes back.
        """
        return [
            self.tokenizer(str(t), return_tensors="pt")["input_ids"].to(self.device)
            for t in texts
        ]

    def sae_fidelity(
        self,
        texts: list[str],
        *,
        floor: str,
        corpus_label: str = "",
        max_sequences: int | None = None,
        confirm: bool = False,
    ) -> dict:
        """How much of the model's predictive loss survives this SAE.

        `floor` is `mean_ablate` or `zero_ablate`, by name, WITH NO DEFAULT.
        The same reconstruction scores differently against the two, so a house
        answer here would be this file deciding something the reader has to be
        told; `saes.ce_recovered` makes the same argument at length and refuses
        an unnamed floor before anything runs.

        No dtype gate, and that is deliberate -- the one place this parts
        company with `rank_features`, which refuses anything but float32.
        `feature_ablate` ranks the causal effects of a decomposition, so a
        decomposition of nothing ranks arbitrary directions; here a broken SAE
        has a real answer, a CE-recovered at or below zero, and refusing to
        print it would hide exactly the finding this measurement exists to
        make.
        """
        from . import saes

        with self._lock:
            if self.replay is not None:
                raise Refusal(
                    "This is a recording. Scoring an SAE's reconstruction "
                    "means running the model three times over your corpus, "
                    "and a `.mri` does not carry one."
                )
            if self.backend == "ollama":
                raise Refusal(
                    "Ollama serves text only — there is no residual stream "
                    "here to reconstruct, and no per-token loss to compare. "
                    "Load the model through HuggingFace."
                )
            if self.model is None:
                raise Refusal("No model loaded — pick one first.")
            if self.sae is None:
                raise Refusal(
                    "No SAE loaded, so there is no reconstruction to score. "
                    "Load one against this model first — this measures what "
                    "its reconstruction costs the model's own predictions, "
                    "which the FVU beside it cannot say."
                )
            # Before anything installs a forward hook. The hooks go on the
            # shared block modules, so a generation decoding on its own thread
            # fills all three loss accumulators with one-token passes -- see
            # `_refuse_if_decoding` and `patch.trace`.
            self._refuse_if_decoding("a CE-recovered measurement")

            sequences = self._tokenized_for_ce(texts)
            # Priced against what will actually be SCORED. `max_sequences`
            # caps the corpus inside `ce_recovered`, so pricing the whole list
            # would refuse a capped run for a cost it was never going to pay --
            # and the cap is the cheaper alternative the refusal names, so it
            # has to be true.
            scored = (
                len(sequences)
                if max_sequences is None
                else min(len(sequences), max(1, int(max_sequences)))
            )
            saes.confirm_ce_recovered(scored, confirm=confirm)

            layer = int(getattr(self.sae, "layer", 0) or 0)
            block = self._block(layer)
            got = saes.ce_recovered(
                self.model,
                block,
                sequences,
                self.sae,
                floor=floor,
                corpus_label=corpus_label,
                max_sequences=max_sequences,
            )
            out = got.to_dict()
            # The SAE is as much a part of this measurement as the model is,
            # so it is named in the receipt beside it -- the same reason
            # `rank_features` puts the repo and the layer in its own.
            out["receipt"] = self.receipt(
                "sae_fidelity",
                floor=got.floor,
                corpus=got.corpus_label,
                corpus_sha256=got.corpus_sha256,
                n_sequences=got.n_sequences,
                n_tokens=got.n_tokens,
                sae_repo=got.repo,
                sae_hook=got.hook,
                layer=got.layer,
            )
            return out

    def sae_fidelity_passes(self, n_sequences: int) -> dict:
        """What that will cost in forward passes, before any is spent.

        Free, exact and portable, and it needs no model — which is what lets a
        panel quote the price while the reader is still typing the corpus.
        `sae_fidelity_cost` is the other half and is not free.
        """
        from . import saes

        return saes.ce_recovered_price(n_sequences)

    def sae_fidelity_cost(
        self, texts: list[str], *, max_sequences: int | None = None
    ) -> dict:
        """The same pass count, plus what a pass costs ON THIS MACHINE.

        SPENDS THREE REAL FORWARD PASSES of its own — a warm-up, a capture and
        the probe — which is why it is a separate call from
        `sae_fidelity_passes` rather than the same one with more fields. A
        panel that asked for this on mount or on every keystroke would be
        running the model to find out whether to run the model.

        The probe is taken on the corpus's OWN first sequence rather than on
        whatever was last generated, because representative here means the
        LENGTH: a pass over 64 tokens does not price a pass over 512, and the
        length that matters is the one about to be swept.
        """
        from . import saes

        with self._lock:
            if self.replay is not None:
                raise Refusal(
                    "This is a recording. Pricing this run means running the "
                    "model once to time it, and a `.mri` does not carry one."
                )
            if self.backend == "ollama":
                raise Refusal(
                    "Ollama serves text only — there is no forward pass here "
                    "to time. Load the model through HuggingFace."
                )
            if self.model is None:
                raise Refusal("No model loaded — pick one first.")
            if self.sae is None:
                raise Refusal(
                    "No SAE loaded, so there is no reconstruction to price. "
                    "Load one against this model first."
                )
            self._refuse_if_decoding("a CE-recovered cost estimate")

            if not texts:
                raise BadRequest(
                    "there is nothing here to price — CE-recovered needs at "
                    "least one sequence to be a loss on."
                )
            # ONLY THE FIRST SEQUENCE IS TOKENIZED, because only the first is
            # probed. `_tokenized_for_ce` puts every tensor it builds on the
            # model's device, so tokenizing the rest would move a twenty
            # thousand line corpus onto the accelerator to answer a question
            # about one pass over one line of it -- and a preflight that can
            # OOM ahead of the measurement it exists to price is not one. The
            # count below still comes from the whole corpus, because that is
            # the number the caller is about to spend.
            probed = self._tokenized_for_ce(texts[:1])[0]
            scored = (
                len(texts)
                if max_sequences is None
                else min(len(texts), max(1, int(max_sequences)))
            )
            layer = int(getattr(self.sae, "layer", 0) or 0)
            block = self._block(layer)
            return saes.estimate_ce_recovered_cost(
                self.model,
                block,
                probed,
                self.sae,
                n_sequences=scored,
                device_kind=self.accel.kind,
            )

    def diff_models(
        self,
        model_a: str,
        model_b: str,
        prompts: list[str],
        *,
        include_heads: bool = False,
        include_tokens: bool = False,
    ) -> dict:
        """What a finetune changed, over a prompt set rather than one prompt.

        Loads each side ONCE in sequence and never holds both: 8 GB will not
        fit two of the models worth comparing. The model currently loaded HERE
        is untouched -- this runs its own pair, and unloading the session's
        model to make room is the caller's decision rather than a side effect
        of asking a question.
        """
        from . import model_diff

        out = model_diff.compare(
            model_diff.loader(
                dtype=self.accel.dtype,
                device=self.accel.torch_device,
                device_kind=self.accel.kind,
            ),
            model_a,
            model_b,
            prompts,
            include_heads=include_heads,
            include_tokens=include_tokens,
        ).to_dict()
        # Stored WITHOUT an epoch. See `_last_model_diff`: this is a claim
        # about two named models, and what is loaded here is irrelevant to it.
        self._last_model_diff = out
        out["receipt"] = self.receipt(
            "model_diff",
            model_a=model_a,
            model_b=model_b,
            n_prompts=out.get("n_prompts"),
            include_heads=include_heads,
            include_tokens=include_tokens,
        )
        return out

    def ground_answer(
        self,
        document: str,
        question: str,
        *,
        max_chunks: int = 0,
    ) -> dict:
        """Did the answer come from the document, or from the weights?

        The one question every local RAG interface leaves unanswered. They all
        show which chunks were RETRIEVED; none of them shows whether the answer
        depended on them, and a retriever that pulled the right paragraph next
        to a model that ignored it looks identical to a working system in
        every one of those UIs.

        NOTHING IS DOWNLOADED and nothing is indexed. The document is text you
        hand it, chunking is by blank line and heading, and every chunk goes
        into the prompt — retrieval is somebody else's job.

        Runs its own prompt and does NOT commit it: grounding is about a
        document-plus-question of its own, and committing would leave every
        other panel describing a prompt the user never asked to analyse.
        """
        from . import ground as ground_mod

        with self._lock:
            if self.replay is not None:
                # A recording cannot MEASURE grounding -- masking a passage
                # out needs the model -- but it can carry what was measured,
                # and "the answer came from the weights, not from the document
                # I gave it" is exactly the finding somebody wants to show a
                # colleague. Serve the recorded one; refuse only when the file
                # does not have it.
                recorded = getattr(self.replay, "ground", None) or {}
                if recorded.get("chunks"):
                    return {
                        **self._recorded_ground(recorded),
                        "recorded": True,
                        "receipt": _recorded_receipt(self.replay, "ground"),
                    }
                raise Refusal(
                    "This is a recording, and it does not carry a grounding "
                    "result. Masking a passage out of the model's attention "
                    "means running the model, and a `.mri` holds activations "
                    "rather than weights. Whoever exported it can ground an "
                    "answer and share it again."
                )
            if self.backend == "ollama":
                raise Refusal(
                    "Ollama serves text only — there is no attention mask to "
                    "take a passage out of. Load the model through "
                    "HuggingFace."
                )
            if self.model is None:
                raise Refusal("No model loaded — pick one first.")

            chunks = ground_mod.split(document)
            out = ground_mod.measure(
                self.model,
                self.tokenizer,
                chunks,
                question,
                device=self.device,
                max_chunks=max_chunks or ground_mod.MAX_CHUNKS,
            ).to_dict()

        # Kept for the export, stamped with the epoch it belongs to. Without
        # the epoch a grounding measured on one model would be written into a
        # `.mri` beside a different model's attention.
        self._last_ground = {**out, "epoch": self.epoch}
        out["receipt"] = self.receipt(
            "ground",
            n_chunks=out["n_chunks"],
            n_prompt_tokens=out["n_prompt_tokens"],
            # The question travels; the document does not. A receipt is meant
            # to be shareable and a grounded document is usually the private
            # half of the pair — its length and chunk count say what was
            # measured without carrying the text itself.
            document_chars=len(document),
            question=question,
        )
        return out

    def patchscope(
        self,
        source_prompt: str,
        *,
        source_layer: int,
        source_position: int = -1,
        target_prompt: str = "",
        target_layer: int | None = None,
        target_position: int = -1,
        max_new_tokens: int = 12,
        draws: int = 1,
    ) -> dict:
        """Ask the model to describe a hidden state, with two controls.

        Its own surface, not a column beside the logit lens. The lens reads a
        state through the unembedding and reports tokens; this hands the state
        to the model inside a different prompt and reports a SENTENCE. Rendered
        side by side they would read as two measurements of one thing, and
        they are not.
        """
        from . import patch

        with self._lock:
            if self.replay is not None:
                raise Refusal(
                    "This is a recording. A patchscope means running the model "
                    "on a second prompt, and a `.mri` does not carry one."
                )
            if self.backend == "ollama":
                raise Refusal(
                    "Ollama serves text only — there is no residual stream "
                    "here to splice."
                )
            if self.model is None:
                raise Refusal("No model loaded — pick one first.")

            # A patchscope makes THREE decodes and compares them, so a bad
            # token budget is spent three times before anything raises.
            # MEASURED: `max_new_tokens: -1` did not answer for 180 seconds
            # and then returned a 500 — transformers' own
            # "`max_new_tokens` must be greater than 0" arrived after the
            # setup for a run that could never produce a sentence.
            if max_new_tokens < 1:
                raise BadRequest(
                    f"a patchscope has to be allowed at least one token to "
                    f"answer in, and this asked for {max_new_tokens}. The "
                    f"whole reading is the SENTENCE the model produces when "
                    f"handed the state; with no tokens there is nothing to "
                    f"read."
                )
            if draws < 1:
                raise BadRequest(
                    f"a patchscope needs at least one draw, and this asked for {draws}."
                )

            def decode(prompt: str) -> str:
                # GREEDY, and commit=False. Three decodes are being compared,
                # so sampling would put a second source of difference between
                # them -- and committing would rebase `last_ids` onto the
                # target prompt, leaving every other panel describing a run
                # the user never asked for.
                # `generate_stream` takes no lock of its own, so calling it
                # while holding this one serialises rather than deadlocking.
                return "".join(
                    self.generate_stream(
                        prompt, max_new_tokens, temperature=0.0, commit=False
                    )
                )

            try:
                out = patch.patchscope(
                    self.model,
                    self.tokenizer,
                    [
                        self._block(i)
                        for i in range(text_config(self.model.config).num_hidden_layers)
                    ],
                    decode,
                    source_prompt,
                    device=self.device,
                    source_layer=source_layer,
                    source_position=source_position,
                    target_prompt=target_prompt or patch.DEFAULT_TARGET,
                    target_layer=target_layer,
                    target_position=target_position,
                    draws=draws,
                )
            except patch.PatchError as err:
                raise BadRequest(
                    str(err)
                ) from err  # leak-ok: authored, see test_no_machine_leaks
            out["receipt"] = self.receipt(
                "patchscope",
                prompt=source_prompt,
                source_layer=source_layer,
                target_layer=out["target"]["layer"],
                target_sha256=receipts.digest(out["target"]["prompt"]),
            )
            return out

    def path_trace(
        self, clean: str, corrupt: str, *, layer: int, position: int
    ) -> dict:
        """Which earlier component wrote what makes this receiver matter.

        The follow-up to a bright cell in the patching grid: that grid says
        WHERE the answer is carried, and patching a residual stream restores
        everything that ever wrote into it at once. This splits that apart.
        """
        from . import patch

        with self._lock:
            # Before anything installs a forward hook. The hooks go on the

            # shared block modules, so a generation decoding on its own

            # thread fills them with one-token passes -- see

            # `_refuse_if_decoding` and `patch.trace`.

            self._refuse_if_decoding("a path trace")
            if self.replay is not None:
                raise Refusal(
                    "This is a recording. Path patching means running the "
                    "model with one component's contribution swapped, and a "
                    "`.mri` holds activations rather than weights."
                )
            if self.backend == "ollama":
                raise Refusal(
                    "Ollama serves text only — there is no residual stream "
                    "here to patch into."
                )
            if self.model is None:
                raise Refusal("No model loaded — pick one first.")

            n_layers = int(text_config(self.model.config).num_hidden_layers)
            blocks = [self._block(i) for i in range(n_layers)]
            try:
                out = patch.path_trace(
                    self.model,
                    self.tokenizer,
                    blocks,
                    clean,
                    corrupt,
                    device=self.device,
                    receiver_layer=layer,
                    receiver_position=position,
                )
            except patch.PatchError as err:
                # A pair or a receiver this measurement cannot be taken on,
                # not a failure of the code. Every one names what to change.
                raise BadRequest(
                    str(err)
                ) from err  # leak-ok: authored, see test_no_machine_leaks
            out["receipt"] = self.receipt(
                "path_trace",
                clean_sha256=receipts.digest(clean),
                corrupt_sha256=receipts.digest(corrupt),
                receiver_layer=layer,
                receiver_position=position,
            )
            return out

    def patch_graph(
        self,
        clean: str,
        corrupt: str,
        *,
        depth: int = 0,
        max_receivers: int = 0,
    ) -> dict:
        """A patching graph: what wrote the thing that wrote the answer.

        The node grid says WHERE the answer is carried and `path_trace` says
        what wrote into one receiver. This asks that second question again of
        the senders that survived their controls, which is the question a
        circuit view is actually opened for.

        A PATCHING graph, never an attribution graph: circuit-tracer's are
        built from transcoders and this is a different object from a different
        measurement. `circuit.py` reads one of theirs; this computes one of
        ours, and the payload says so in both places.
        """
        from . import patch
        from . import patch_graph as graph_mod

        # A recording that CARRIES a graph serves it, for the same reason
        # `patch_trace` serves a recorded trace: this one cost roughly 1,500
        # forward passes to build, the recipient has no weights to rebuild it
        # with, and refusing a file that already holds the answer is the format
        # failing to be worth sending.
        if self.replay is not None:
            recorded = self.replay.patch_graph
            if recorded.get("edges"):
                return {
                    **recorded,
                    "recorded": True,
                    # The receipt out of the file's own list. Every section is
                    # exported WITHOUT its own copy, deliberately -- see
                    # `_recorded_receipt` -- and nothing ever read the list they
                    # were consolidated into.
                    "receipt": _recorded_receipt(self.replay, "patch_graph"),
                }
            raise Refusal(
                "This is a recording, and it does not carry a patching graph. "
                "Building one means running the model again with activations "
                "replaced, hundreds of times, and a `.mri` holds activations "
                "rather than weights — there is nothing here to re-run. "
                "Whoever exported it can build the graph and share it again."
            )

        depth = int(depth or graph_mod.DEFAULT_DEPTH)
        max_receivers = int(max_receivers or graph_mod.DEFAULT_MAX_RECEIVERS)

        # The node grid FIRST, and its own flagged sites are the seeds. A
        # threshold invented here would be a second opinion about which cells
        # matter, disagreeing with the grid the reader is looking at.
        grid = self.patch_trace(clean, corrupt)
        sites = list(grid.get("sites") or [])

        n_layers = int(text_config(self.model.config).num_hidden_layers)
        blocks = [self._block(i) for i in range(n_layers)]

        def trace_fn(layer: int, position: int) -> dict:
            with self._lock:
                # Before anything installs a forward hook. The hooks go on the
                # shared block modules, so a generation decoding on its own
                # thread fills them with one-token passes -- see
                # `_refuse_if_decoding` and `patch.trace`.
                self._refuse_if_decoding("a patching graph")
                return patch.path_trace(
                    self.model,
                    self.tokenizer,
                    blocks,
                    clean,
                    corrupt,
                    device=self.device,
                    receiver_layer=layer,
                    receiver_position=position,
                )

        try:
            built = graph_mod.build(
                trace_fn,
                sites,
                depth=depth,
                max_receivers=max_receivers,
                clean=clean,
                corrupt=corrupt,
                answer=str((grid.get("clean") or {}).get("answer") or ""),
            )
        except patch.PatchError as err:
            raise BadRequest(str(err)) from err  # leak-ok: authored

        out = built.to_dict()
        # The grid's own passes count too -- it was run to get the seeds, and
        # a cost that omits the step it depended on is not the cost.
        out["passes"] = int(out.get("passes") or 0) + int(grid.get("passes") or 0)
        out["receipt"] = self.receipt(
            "patch_graph",
            clean_sha256=receipts.digest(clean),
            corrupt_sha256=receipts.digest(corrupt),
            depth=depth,
            max_receivers=max_receivers,
        )
        # Stamped with the epoch for the same reason the patch trace is: a
        # graph measured on an earlier prompt, or on a model since swapped out,
        # must not be written into a `.mri` beside a different run's tokens.
        self._last_patch_graph = dict(out, epoch=self.epoch)
        return out

    def probe_layers(
        self,
        examples: list[dict],
        *,
        n_permutations: int = 0,
        save_as: str = "",
    ) -> dict:
        """Fit a linear probe at every layer, with its null and majority line.

        `examples` is `[{"text": ..., "label": 0|1}, ...]`. Captured at the
        same pre-hook point `steer_vectors` and `patch` use, so a direction
        fitted here lives in the space those already measure and can be pushed
        straight back through the steering harness.
        """
        from . import probe as probe_mod
        from . import steer_vectors

        with self._lock:
            if self.replay is not None:
                raise Refusal(
                    "This is a recording. Fitting a probe means running the "
                    "model on your examples, and a `.mri` does not carry one."
                )
            if self.backend == "ollama":
                raise Refusal(
                    "Ollama serves text only — there is no residual stream "
                    "here to fit a probe to."
                )
            if self.model is None:
                raise Refusal("No model loaded — pick one first.")

            texts, labels = [], []
            for row in examples or []:
                if not isinstance(row, dict):
                    raise BadRequest("each example is {text, label}")
                text = row.get("text")
                label = row.get("label")
                if not isinstance(text, str) or not text.strip():
                    raise BadRequest("an example has no text")
                if not isinstance(label, int) or isinstance(label, bool):
                    raise BadRequest(
                        f"the label for {text[:40]!r} is not an integer. "
                        "A probe separates two classes; label them 0 and 1."
                    )
                texts.append(text)
                labels.append(label)

            # REFUSED BEFORE A FORWARD PASS, and before torch sees an empty
            # list. `POST /api/probe {"examples": []}` reached
            # `torch.stack([])` and answered 500 with "stack expects a
            # non-empty TensorList" — an error about a tensor library, at
            # somebody who clicked Train with the box empty. The route's own
            # per-row checks were thorough and never ran, because there were
            # no rows to check.
            if not texts:
                raise BadRequest(
                    "a probe is trained on YOUR labelled examples and none "
                    "were sent. Give it at least one text for each class — "
                    "label them 0 and 1 — and it will report where in the "
                    "network those two become separable."
                )
            # And a set that cannot separate anything. `probe_mod.sweep` would
            # fit a classifier on one class and report an accuracy of 1.0,
            # which is a number that means nothing and looks like a result.
            if len(set(labels)) < 2:
                only = sorted(set(labels))[0]
                raise BadRequest(
                    f"every example here is labelled {only}, so there is no "
                    f"second class to separate it from. A probe measures "
                    f"where two classes become distinguishable; with one "
                    f"class it would report perfect accuracy and mean "
                    f"nothing by it."
                )

            n_layers = int(text_config(self.model.config).num_hidden_layers)
            layers = list(range(n_layers))
            ids_list = [
                self.tokenizer(t, return_tensors="pt")["input_ids"].to(self.device)
                for t in texts
            ]
            states = steer_vectors._last_token_states(
                self.model, self._block, ids_list, layers
            )

            report = probe_mod.sweep(
                states,
                labels,
                n_permutations=n_permutations or probe_mod.N_PERMUTATIONS,
            )
            out = report.to_dict()
            out["receipt"] = self.receipt(
                "probe_layers",
                n_examples=len(texts),
                n_permutations=out["n_permutations"],
            )

            if save_as:
                best = report.best_layer
                if best is None:
                    # Refused, not saved with a caveat attached. A direction
                    # fitted at a layer whose probe never beat noise is a
                    # direction fitted to noise, and the steering store is
                    # where it would be picked up later with none of this
                    # context attached to it.
                    raise Refusal(
                        "no layer read this concept above its own permutation "
                        "null, so there is no direction here worth saving. A "
                        "vector fitted where the probe did not beat shuffled "
                        "labels is fitted to noise, and the store is the one "
                        "place it would later be used without any of this "
                        "beside it."
                    )
                direction = probe_mod.direction_at(states, labels, best)
                out["saved"] = steer_vectors.save(
                    save_as,
                    direction,
                    {
                        "model": self.hf_id or "",
                        "layer": best,
                        "hidden_size": int(direction.shape[0]),
                        "method": "probe",
                        "dtype": str(next(self.model.parameters()).dtype).removeprefix(
                            "torch."
                        ),
                        "accuracy": out["layers"][best]["accuracy"],
                        "null_high": out["layers"][best]["null_high"],
                        "majority": out["majority"],
                        # WRITTEN, because `steer_vectors.load` checks
                        # `payload.get("beats_null") is False` and a missing
                        # key is not False — so for every direction this
                        # product could produce, the failed-null warning was
                        # unreachable code. The refusal above means this is
                        # always True here; writing it anyway is what makes
                        # the check honest rather than vacuous, and a future
                        # writer that saves a weaker direction inherits a
                        # reader for it.
                        "beats_null": True,
                        # So the steering panel can report a strength relative
                        # to something for a probe-fitted direction too. Same
                        # definition as `fit_direction`'s: the mean L2 norm of
                        # the last-token states this layer was fitted from.
                        "residual_norm": round(
                            float(states[best].norm(dim=-1).mean()), 3
                        ),
                        "note": (
                            "fitted by a layer-sweep probe. READABLE IS NOT "
                            "USED: this direction was linearly decodable, "
                            "which is not evidence the model reads it."
                        ),
                    },
                )
            return out

    def head_types(
        self, seq_len: int = 24, n_sequences: int = 6, seed: int = 0
    ) -> dict:
        """Label heads by behaviour, each gated on a null measured here.

        Independent of any generation: it runs its own random sequences, so it
        needs a model and nothing else. That is also why the result survives a
        new prompt while every other measurement here does not.
        """
        from . import head_types as ht

        with self._lock:
            if self.replay is not None:
                recorded = getattr(self.replay, "head_types", None) or {}
                if recorded.get("labels"):
                    return {
                        **recorded,
                        "recorded": True,
                        # The receipt out of the file's own list. Every section is
                        # exported WITHOUT its own copy, deliberately -- see
                        # `_recorded_receipt` -- and nothing ever read the list they
                        # were consolidated into.
                        "receipt": _recorded_receipt(self.replay, "head_types"),
                    }
                raise Refusal(
                    "This is a recording, and it does not carry head type "
                    "labels. Labelling heads means running the model on new "
                    "random sequences, and a `.mri` holds activations rather "
                    "than weights. Whoever exported it can label the heads "
                    "and share it again."
                )
            if self.backend == "ollama":
                raise Refusal(
                    "Ollama serves text only — the attention these labels are "
                    "read from never leaves its process."
                )
            if self.model is None:
                raise Refusal("No model loaded — pick one first.")

            out = ht.label_heads(
                self.model,
                self.tokenizer,
                seq_len=seq_len,
                n_sequences=n_sequences,
                seed=seed,
            ).to_dict()
            out["receipt"] = self.receipt(
                "head_types", seq_len=seq_len, n_sequences=n_sequences, seed=seed
            )
            self._last_types = {**out, "epoch": self.epoch}
            return out

    def _weights_only(self, what: str):
        """The three states in which a weight-space readout has no weights.

        Shared by the OV readouts below, because they are the only measurements
        here that need a MODEL and nothing else — no generation, no prompt, not
        even a forward pass. A recording carries activations rather than
        weights and Ollama never hands the weights out at all, so both are
        refused for a reason a reader can act on rather than for the generic
        "generate something first" that fits every other panel.
        """
        if self.replay is not None:
            raise Refusal(
                f"This is a recording, and {what} is read off the model's "
                f"WEIGHTS. A `.mri` carries activations — what the model did — "
                f"rather than the weights that did it. Load the model itself "
                f"and this answers without a prompt."
            )
        if self.backend == "ollama":
            raise Refusal(
                f"Ollama serves text and never hands out the weights, so "
                f"{what} cannot be read from a model it is holding. The same "
                f"checkpoint pulled from the Hub answers this."
            )
        if self.model is None:
            raise Refusal("No model loaded — pick one first.")

        cfg = text_config(self.model.config)
        n_heads = getattr(cfg, "num_attention_heads", None)
        if n_heads is None:
            raise Refusal(
                f"this architecture publishes no attention heads, so there is "
                f"no {what} to read — state-space and RNN models reach here."
            )
        return int(n_heads)

    def head_ov_vocabulary(
        self, layer: int, head: int, token: str | int, top_k: int = 10
    ) -> dict:
        """What one head writes into the stream when it attends to one token.

        Exact, and needs no prompt: it is `W_U @ W_O[h] @ W_V[kv(h)] @ e`, a
        product of weights. That is what makes it worth having beside
        `ablate_heads`, which answers the same question about THIS generation
        and gives a different answer for the next one.
        """
        from . import ov_circuits

        with self._lock:
            n_heads = self._weights_only("what a head writes")
            block = self._block(layer)
            ids = (
                [int(token)]
                if isinstance(token, int) and not isinstance(token, bool)
                else self.tokenizer.encode(str(token), add_special_tokens=False)
            )
            if not ids:
                raise BadRequest(
                    f"{token!r} encodes to no tokens at all, so there is no "
                    f"embedding to push through the head. Try a word rather "
                    f"than whitespace, or send a token id."
                )
            # The FIRST token of a multi-token word, said out loud below rather
            # than silently. A head reads one token at a time, so "what does it
            # do with `unbelievable`" has no single answer when the tokenizer
            # splits it into three.
            out = ov_circuits.ov_vocabulary(
                self.model,
                self.tokenizer,
                block,
                head,
                n_heads=n_heads,
                source_token_id=ids[0],
                top_k=top_k,
            )
            out["layer"] = layer
            out["source_token_count"] = len(ids)
            if len(ids) > 1:
                out["means"] += (
                    f" The text you sent is {len(ids)} tokens and this is the "
                    f"first of them — a head reads one token at a time, so "
                    f"there is no single answer for the whole word."
                )
            out["receipt"] = self.receipt(
                "head_ov", layer=layer, head=head, token=str(token), top_k=top_k
            )
            return out

    def head_evidence(
        self,
        texts: list[str],
        *,
        layer: int,
        head: int,
        corpus_label: str,
        read_attention: bool = True,
        top_k: int = 10,
    ) -> dict:
        """Where this head wrote in YOUR text, and what it pushes toward.

        Needs a model and text, and NOT a generation — which is why it does not
        go through `_require_live_generation`. The corpus is the caller's, so
        this measures what the head does on the text they care about rather
        than on whatever happened to be in the prompt box.
        """
        from . import head_corpus

        with self._lock:
            n_heads = self._weights_only("what a head writes")
            # `_block` bounds the layer against this model's real depth, which
            # is the only place that knows it — see its own comment about
            # `variant=ablate:9999` answering 500 from inside a ModuleList.
            self._block(layer)
            out = head_corpus.evidence(
                self.model,
                self.tokenizer,
                self._block,
                [str(t) for t in texts],
                layer=layer,
                head=head,
                n_heads=n_heads,
                corpus_label=corpus_label,
                read_attention=read_attention,
                top_k=top_k,
            )
            out["receipt"] = self.receipt(
                "head_evidence",
                layer=layer,
                head=head,
                corpus=corpus_label,
                n_sequences=len(texts),
                read_attention=read_attention,
            )
            return out

    def head_evidence_cost(self, n_sequences: int, *, read_attention: bool) -> dict:
        """What that will cost, in forward passes, before any is spent."""
        from . import head_corpus

        return head_corpus.sweep_cost(n_sequences, read_attention=read_attention)

    def head_ov_spectrum(
        self, layer: int, head: int, n_samples: int = 0, seed: int = 0
    ) -> dict:
        """The eigenvalue readout of one head's OV circuit, over a named sample.

        `n_samples=0` means "the module's own default", which it reports —
        rather than this route inventing a second default that could drift
        from it.
        """
        from . import ov_circuits

        with self._lock:
            n_heads = self._weights_only("a head's OV circuit")
            block = self._block(layer)
            out = ov_circuits.ov_spectrum(
                self.model,
                block,
                head,
                n_heads=n_heads,
                n_samples=n_samples or ov_circuits.SPECTRUM_SAMPLE,
                seed=seed,
            )
            out["layer"] = layer
            out["receipt"] = self.receipt(
                "head_ov_spectrum",
                layer=layer,
                head=head,
                n_samples=out["n_sampled"],
                seed=seed,
            )
            return out

    def direct_attribution(self, position: int | None = None, top_k: int = 40) -> dict:
        """How many logits each head and MLP put behind the predicted token.

        A different question from `ablate_heads`, and they disagree: a head can
        contribute nothing directly and still decide the answer by feeding a
        later head. The response says so rather than leaving a near-zero bar to
        be read as "this head does not matter".
        """
        from . import dla

        with self._lock:
            # Before anything installs a forward hook. The hooks go on the

            # shared block modules, so a generation decoding on its own

            # thread fills them with one-token passes -- see

            # `_refuse_if_decoding` and `patch.trace`.

            self._refuse_if_decoding("direct logit attribution")
            if self.replay is not None:
                raise Refusal(
                    "This is a recording. Direct attribution means running the "
                    "model, and a `.mri` does not carry one."
                )
            if self.backend == "ollama":
                raise Refusal(
                    "Ollama serves text only — there are no components here to "
                    "attribute a logit across."
                )
            if self.model is None:
                raise Refusal("No model loaded — pick one first.")
            if self.last_ids is None:
                raise Refusal("Generate something first — attribution reads that run.")

            # The last PROMPT token by default, for the same reason
            # `ablate_heads` uses it: that is where the model answers the
            # question, before any of its own output feeds back in.
            size = int(self.last_ids.shape[0])
            at = (
                max(0, min(self.last_n_prompt_tokens - 1, size - 1))
                if position is None
                else max(0, min(int(position), size - 1))
            )
            out = dla.attribute(
                self.model, self.tokenizer, self.last_ids, position=at, top_k=top_k
            ).to_dict()
            out["receipt"] = self.receipt(
                "direct_attribution", position=at, top_k=top_k
            )
            return out

    def logit_lens(self, top_k: int = 5, kind: str = "plain") -> dict:
        """What the model was about to say at every layer, not just the last.

        A ModelRuntime method rather than a call `server.py` makes into
        `lens.py` directly, and that is the point of moving it. Every other
        measurement here opens with `if self.replay is not None`; the lens did
        not, because it was computed outside the runtime, so the replay guard
        had to be duplicated at the route -- and a duplicated guard is one that
        gets forgotten on the next route that needs it. `server.py` records
        exactly how that failed: with a recording open AND a model loaded, the
        lens happily reported the LIVE model's layers inside a session every
        other panel was drawing from the file.

        Now it is one guard, in the same place as the others, and a recording
        that carries a lens can serve it.
        """
        with self._lock:
            if self.replay is not None:
                # getattr, not attribute access: `replay` is a Session in
                # every real path, but a 500 is the wrong answer for "this is
                # a recording" under ANY replay object. The refusal below is
                # the correct response either way, and reaching it must not
                # depend on the shape of what was opened.
                recorded = getattr(self.replay, "lens", None) or []
                if recorded:
                    return {
                        "layers": recorded,
                        **(getattr(self.replay, "lens_info", None) or {}),
                        "recorded": True,
                        # And the receipt, like every other recorded
                        # measurement. See `_recorded_receipt`.
                        "receipt": _recorded_receipt(self.replay, "logit_lens"),
                    }
                raise Refusal(
                    "This is a recording, and it does not carry a logit lens. "
                    "The lens means running the model, and a `.mri` holds "
                    "activations rather than weights. Whoever exported it can "
                    "read the lens and share it again."
                )
            if self.backend == "ollama":
                raise Refusal(
                    "Ollama serves text only — the layers never leave its process."
                )
            if self.model is None:
                raise Refusal("No model loaded — pick one first.")
            if self.last_ids is None:
                raise Refusal("Generate something first — the lens reads that run.")

            from .lens import logit_lens as _logit_lens

            if kind not in ("plain", "tuned", "both"):
                raise BadRequest(
                    f"unknown lens kind {kind!r} — use plain, tuned or both"
                )
            if kind in ("tuned", "both") and not self._tuned:
                raise Refusal(
                    "No tuned lens has been trained for this model. Train one "
                    "on your own text first — nothing is fetched, because a "
                    "downloaded lens would break the offline promise the rest "
                    "of this package keeps."
                )

            out = _logit_lens(
                self.model, self.tokenizer, self.last_ids.unsqueeze(0), top_k
            )
            if kind in ("tuned", "both"):
                from . import tuned_lens as tl

                tuned = tl.read(
                    self.model,
                    self.tokenizer,
                    self.last_ids.unsqueeze(0),
                    self._tuned,
                    top_k,
                )
                # BESIDE, never instead. `layers` stays the plain reading on
                # every kind, so a caller that ignores `tuned` gets the
                # untranslated rows rather than translated ones it did not ask
                # for -- and the panel renders two columns from one payload.
                out["tuned"] = tuned["layers"]
                out["tuned_info"] = self._tuned_info
            out["kind"] = kind
            out["receipt"] = self.receipt("logit_lens", top_k=top_k, kind=kind)
            # Kept for export on the same terms as the ranking and the patch
            # trace: tagged with the epoch, dropped when the run it describes
            # stops being the current one.
            self._last_lens = {**out, "epoch": self.epoch}
            return out

    def token_anchors(self, position: int | None = None, **kw) -> dict:
        """The smallest set of the prompt's own tokens that HOLDS the answer.

        The opposite question from `attribute_tokens`, and the pair is the
        point: that one masks a token out and measures what breaks — necessity.
        This one keeps a few and perturbs everything else, and asks whether the
        prediction survives — sufficiency. A token can be necessary and not
        sufficient, and both rankings are real; conflating them is the mistake
        `anchors.py` was written against.
        """
        from . import anchors as anchors_mod
        from . import attribute

        with self._lock:
            self._require_live_generation(
                "Generate something first, then ask which of its tokens hold "
                "the answer on their own."
            )
            size = int(self.last_ids.shape[0])
            if position is None:
                position = max(0, min(self.last_n_prompt_tokens - 1, size - 1))
            if not 0 <= position < size:
                raise BadRequest(
                    f"position must be in [0,{size}) — that generation is "
                    f"{size} tokens long."
                )
            control = attribute.control_token_ids(self.tokenizer)
            pool, perturbation = anchors_mod.donor_pool(
                self.tokenizer, control_ids=control
            )
            out = anchors_mod.find_anchor(
                self.model,
                self.last_ids.unsqueeze(0).to(self.device),
                position=position,
                pool=pool,
                perturbation=perturbation,
                control_ids=control,
                typed_span=self.last_user_span,
                n_prompt=int(self.last_n_prompt_tokens or 0),
                decode=lambda t: self.tokenizer.decode([t]),
                **kw,
            )
            out["receipt"] = self.receipt("anchors", position=position, **kw)
            return out

    def resolve_target_token(self, text: str) -> int:
        """One token id for `text`, or a sentence saying why there is not one.

        A counterfactual moves the NEXT TOKEN, so the target has to be a single
        token. "Rome" is one on most tokenizers and "Colosseum" is not, and the
        difference is a property of the tokenizer rather than of the words, so
        it is measured here rather than assumed either way.

        Leading whitespace matters and is not added silently: mid-sentence, the
        model predicts " Rome" with a space and `Rome` without one is a
        different id that it may never emit. Both are tried, the one that
        tokenizes to a single id wins, and if both do the spaced form wins
        because that is the continuation position this runs at.
        """
        if not isinstance(text, str) or not text.strip():
            raise BadRequest(
                "name the token the model should be steered toward, as text."
            )
        candidates = [text if text.startswith(" ") else " " + text, text.strip()]
        tried = []
        for form in candidates:
            ids = self.tokenizer(form, add_special_tokens=False).input_ids
            tried.append((form, len(ids)))
            if len(ids) == 1:
                return int(ids[0])
        pieces = [
            self.tokenizer.decode([i])
            for i in self.tokenizer(candidates[0], add_special_tokens=False).input_ids
        ]
        shapes = ", ".join(f"{f!r} -> {n} tokens" for f, n in tried)
        raise BadRequest(
            f"{text!r} is not a single token for this tokenizer ({shapes}), and "
            "a counterfactual moves exactly one next-token prediction. It cuts "
            f"into {pieces}. Name one of those pieces, or a shorter word that "
            "means the same thing — this is a property of the tokenizer, not "
            "of the model."
        )

    def token_counterfactual(
        self,
        position: int | None = None,
        *,
        target: str | None = None,
        target_token_id: int | None = None,
        **kw,
    ) -> dict:
        """The smallest edit to the prompt that makes the model say something else.

        The third question about a prompt, beside `attribute_tokens`
        (necessity — what breaks when this word is removed) and `token_anchors`
        (sufficiency — do these words hold the answer alone). This one is
        directional: what do I write INSTEAD to get the answer I name.

        Its output doubles as the corrupt half of a patching pair. Every number
        `patch_graph` publishes is a difference between a clean prompt and a
        corrupt one, and until now that corrupt prompt was typed by hand.
        """
        from . import attribute
        from . import counterfactual as cf_mod

        with self._lock:
            self._require_live_generation(
                "Generate something first, then ask what would make it say "
                "something else."
            )
            size = int(self.last_ids.shape[0])
            if position is None:
                position = max(0, min(self.last_n_prompt_tokens - 1, size - 1))
            if not 0 <= position < size:
                raise BadRequest(
                    f"position must be in [0,{size}) — that generation is "
                    f"{size} tokens long."
                )
            if (target is None) == (target_token_id is None):
                raise BadRequest(
                    "name the target once: either `target` as text or "
                    "`target_token_id` as an id. Passing both leaves it "
                    "ambiguous which one the answer is about, and passing "
                    "neither leaves nothing to steer toward."
                )
            resolved = (
                int(target_token_id)
                if target_token_id is not None
                else self.resolve_target_token(target)
            )
            control = attribute.control_token_ids(self.tokenizer)
            pool, perturbation = cf_mod.donor_pool(self.tokenizer, control_ids=control)
            out = cf_mod.find_counterfactual(
                self.model,
                self.last_ids.unsqueeze(0).to(self.device),
                position=position,
                target_token_id=resolved,
                pool=pool,
                perturbation=perturbation,
                control_ids=control,
                typed_span=self.last_user_span,
                n_prompt=int(self.last_n_prompt_tokens or 0),
                decode=lambda t: self.tokenizer.decode([t]),
                **kw,
            )
            out["target_named"] = target
            out["edited_text"] = self.tokenizer.decode(out["edited_ids"])
            out["receipt"] = self.receipt(
                "counterfactual", position=position, target=resolved, **kw
            )
            return out

    def counterfactual_cost(
        self, position: int | None = None, *, max_edits: int = 3, **kw
    ) -> dict:
        """What a counterfactual search will cost, before it costs it."""
        from . import counterfactual as cf_mod

        with self._lock:
            self._require_live_generation(
                "Generate something first, then price a counterfactual search over it."
            )
            size = int(self.last_ids.shape[0])
            if position is None:
                position = max(0, min(self.last_n_prompt_tokens - 1, size - 1))
            if not 0 <= position < size:
                raise BadRequest(
                    f"position must be in [0,{size}) — that generation is "
                    f"{size} tokens long."
                )
            # The editable window is what the cost depends on, and it is the
            # same window the search itself uses: index 0 is an attention sink
            # and `position` is its own query.
            n_positions = max(0, position - 1)
            if n_positions < 1:
                raise BadRequest(
                    f"there is nothing editable before position {position}, so "
                    "there is no search to price."
                )
            return cf_mod.estimate_cost(
                n_positions=n_positions, max_edits=max_edits, **kw
            )

    def token_gradients(self, position: int | None = None, **kw) -> dict:
        """Integrated gradients over the input embeddings, with the gap named.

        NOT a causal measurement, and the module says so at length: a gradient
        answers what the output was sensitive to in the limit of an
        infinitesimal nudge, where `attribute_tokens` answers what happens when
        a token is actually removed. Both are here because they disagree, and
        the completeness gap is what says whether this one converged at all.
        """
        from . import gradients as grad_mod

        with self._lock:
            self._require_live_generation(
                "Generate something first, then attribute it to its inputs."
            )
            size = int(self.last_ids.shape[0])
            if position is None:
                position = max(0, min(self.last_n_prompt_tokens - 1, size - 1))
            if not 0 <= position < size:
                raise BadRequest(
                    f"position must be in [0,{size}) — that generation is "
                    f"{size} tokens long."
                )
            out = grad_mod.integrated_gradients(
                self.model,
                self.tokenizer,
                self.last_ids.to(self.device),
                position=position,
                **kw,
            ).to_dict()
            out["receipt"] = self.receipt("gradients", position=position, **kw)
            return out

    def patch_screen(self, clean: str, corrupt: str, **kw) -> dict:
        """Rank patching sites cheaply, then check the screen against the exact
        grid on the few it shortlisted.

        A SCREEN, and the payload says so in its own field names: the numbers
        are a first-order approximation and the module measures its own
        agreement with the thing it approximates rather than asking to be
        trusted.
        """
        from . import patch_screen as screen_mod

        with self._lock:
            if self.replay is not None:
                raise Refusal(
                    "This is a recording. Screening patch sites means running "
                    "the model forward and backward, and a `.mri` carries "
                    "activations rather than the weights that produced them."
                )
            if self.backend == "ollama":
                raise Refusal(
                    "Ollama serves text only — there is no gradient to take. "
                    "Load the same checkpoint through HuggingFace."
                )
            if self.model is None:
                raise Refusal("No model loaded — pick one first.")
            blocks = decoder_blocks(self.model)
            if blocks is None:
                raise Refusal(
                    "this architecture publishes no decoder blocks, so there "
                    "are no patching sites to screen."
                )
            out = screen_mod.screen(
                self.model,
                self.tokenizer,
                list(blocks),
                clean,
                corrupt,
                device=self.device,
                **kw,
            )
            out["receipt"] = self.receipt("patch_screen", **kw)
            return out

    def neuron_evidence(
        self, texts: list[str], neuron: int, *, layer: int, top_k: int = 10
    ) -> dict:
        """What one MLP neuron fires on, for a model that has no SAE.

        A neuron browser is not a worse SAE, it is a blunter instrument —
        neurons are polysemantic, which is the entire reason SAEs exist — and
        the module says so where a reader will see it rather than in a
        footnote.
        """
        from . import neurons as neuron_mod

        with self._lock:
            self._weights_only("what a neuron fires on")
            block = self._block(layer)
            out = neuron_mod.evidence(
                self.model,
                block,
                self.tokenizer,
                [str(t) for t in texts],
                neuron,
                device=self.device,
                top_k=top_k,
            )
            out["receipt"] = self.receipt(
                "neuron_evidence", layer=layer, neuron=neuron, n_sequences=len(texts)
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
        the endpoint on this machine: 20 on gemma-3-270m-it (13 tested), 23 on
        Qwen3-0.6B attributing at token 17 (16 tested). The count is the
        portable part. The seconds are not: on one RTX 4060, warm and back to
        back, those two ran 0.84-0.92 s and
        1.00-1.04 s. The first call after a
        load pays CUDA warm-up on top of that. So
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
            # A base model's case: no chat template, so the prompt is the whole
            # prompt.
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
        features in that ranking's top-8 can fire ONLY at earlier tokens and
        reach the prediction through attention,
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
        Gaussian direction at the top feature's norm can cost a sizeable
        fraction of that feature's own score, and rows that fail to clear their
        own control are routine rather than rare. Prompt scope tests several
        times as many candidates as position scope and costs proportionally
        more. The pass count is the portable part; the
        seconds are the machine's. `passes` and `elapsed_s` both come back so a
        caller derives a rate on its own machine, the same contract the other
        two rankings carry.

        **It refuses anything but float32, and that is the one refusal here
        that is not obvious.** `feature_ablate` checks its floor by writing the
        stream back unchanged, which is bit-exact in every dtype and scores 0.0
        in every dtype — so nothing inside the measurement can notice that in
        bfloat16 a 1-ulp change to the stream is worth ~0.01-0.03 nats on its
        own. Measured here, an edit whose true effect is 4.9e-07 nats
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
            # Before anything installs a forward hook. The hooks go on the
            # shared block modules, so a generation decoding on its own
            # thread fills them with one-token passes -- see
            # `_refuse_if_decoding` and `patch.trace`.
            self._refuse_if_decoding("a feature ranking")
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
            # dtype differs. cuda, eager, jbloom/GPT2-Small-SAEs-Reformatted
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
                    "top two or three rows. Measured here at "
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
        # measured through this endpoint on CPU/float32, the floor is
        # exactly 0.0 while rows still come back below it.
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

    def export_session(
        self,
        layer: int = 0,
        head: int = 0,
        note: str = "",
        trace: dict | None = None,
        step_ref: str = "",
    ) -> bytes:
        """Serialise the current analysis to a `.mri` someone else can open.

        With `trace`, the agent run ships alongside the mechanistic snapshot —
        the recipient clicks the failing tool call and lands in the attention
        view of the generation that produced the bad argument, on a machine
        with no GPU and no account. `session.build` redacts both halves before
        anything is written; see `bundle.py`.
        """
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
        # A run with no analysis behind it is still worth sending: a portable
        # timeline that opens with nothing installed is something no hosted
        # platform offers either. So when the caller explicitly asked to
        # bundle a trace, "nothing has been generated yet" stops being a
        # refusal and becomes a smaller file.
        #
        # Only when they asked. Without `trace` this is the analysis export,
        # and an empty one would be a file with nothing in it.
        # No local `from . import session` here: `session` is already imported
        # at module scope, and importing it inside this function would make
        # the name local to the WHOLE function — so the `session.build(...)`
        # at the bottom would raise UnboundLocalError on every export that
        # does not take this branch. Which is most of them.
        if trace is not None and not self._attn_variants.get("live"):
            return session.build(
                model_id="",
                device=self.device,
                dtype="",
                n_params=None,
                tokens=[],
                prompt="",
                generation="",
                attention={},
                n_layers=0,
                n_heads=0,
                n_prompt=0,
                note=note,
                scope="the agent run only — no generation was loaded when this "
                "was exported, so there is no attention behind it",
                trace=trace,
                step_ref=step_ref,
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

        lens_rows, lens_info = self._lens_for_export()
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
            # The graph walked backwards from those same sites, when one was
            # built. A SEPARATE argument from `graph`, which is somebody
            # else's transcoder attribution graph — two different objects from
            # two different measurements, and `session.py` keeps them apart on
            # purpose.
            patch_graph=self._patch_graph_for_export(),
            ranking=self._ranking_for_export(),
            head_types=self._types_for_export(),
            ground=self._ground_for_export(),
            model_diff=self._model_diff_for_export(),
            # The agent run this analysis belongs to, when the caller named
            # one. Additive: without it the file is exactly what it was.
            trace=trace,
            step_ref=step_ref,
            lens=lens_rows,
            lens_info=lens_info,
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

    def _lens_for_export(self) -> tuple[list, dict]:
        """(rows, scalars) for the logit lens, if it belongs to this run.

        Split in two because `lens` is an existing key with a declared list
        type -- the rows the panel draws -- and the scalars that come with it
        (`final`, `settled_at`, `reliability`) do not fit in a list. They ride
        in an additive `lens_info` rather than changing the shape of a key
        that already exists, so the format version does not move.
        """
        last = self._last_lens
        if not last or last.get("epoch") != self.epoch:
            return [], {}
        rows = last.get("layers") or []
        info = {
            k: last[k]
            for k in ("final", "settled_at", "n_layers", "reliability")
            if k in last
        }
        return rows, info

    def _types_for_export(self) -> dict:
        """Head labels, if they describe the model being exported.

        Epoch-filtered like the ranking, but for a different reason: these
        survive a new generation on purpose and die on a model swap. Labels
        measured on one model written beside another model's attention would
        name heads that do not exist.
        """
        last = self._last_types
        if not last or last.get("epoch") != self.epoch:
            return {}
        return {k: v for k, v in last.items() if k not in ("epoch", "receipt")}

    @staticmethod
    def _recorded_ground(recorded: dict) -> dict:
        """A recorded grounding, with its sentence rebuilt from its numbers.

        `means` is NOT carried in the file. Rebuilding it here means the prose
        and the fields can never disagree: a file hand-edited to flip
        `ungrounded` gets a summary that says so, instead of a stored sentence
        describing the numbers it used to have.

        The rebuild goes through the same dataclass the live path uses, so
        there is one implementation of what these numbers mean rather than a
        second one for replay that drifts.
        """
        from . import ground as ground_mod

        rows = [
            ground_mod.Score(
                index=int(c.get("index", 0)),
                preview=str(c.get("preview") or ""),
                n_tokens=int(c.get("n_tokens") or 0),
                dependence=float(c.get("dependence") or 0.0),
                attention=c.get("attention"),
                depended_on=bool(c.get("depended_on")),
                looked_not_used=c.get("looked_not_used"),
            )
            for c in recorded.get("chunks") or []
        ]
        report = ground_mod.Grounding(
            question=str(recorded.get("question") or ""),
            answer=str(recorded.get("answer") or ""),
            answer_p=float(recorded.get("answer_p") or 0.0),
            position=int(recorded.get("position") or 0),
            chunks=rows,
            n_chunks=int(recorded.get("n_chunks") or len(rows)),
            n_prompt_tokens=int(recorded.get("n_prompt_tokens") or 0),
            noise_floor=float(recorded.get("noise_floor") or 0.0),
            joint=float(recorded.get("joint") or 0.0),
            attention_share=recorded.get("attention_share"),
            attention_available=bool(recorded.get("attention_available")),
            attention_note=str(recorded.get("attention_note") or ""),
            floor_degenerate=bool(recorded.get("floor_degenerate")),
            passes=int(recorded.get("passes") or 0),
            seconds=float(recorded.get("seconds") or 0.0),
            ungrounded=bool(recorded.get("ungrounded")),
        )
        return report.to_dict()

    def _model_diff_for_export(self) -> dict:
        """The last comparison, if there was one.

        NO EPOCH CHECK, alone among the export helpers here. Every other
        section describes the model in this file and dies when the model
        changes; a diff names its own two models and survives, because
        unloading a model does not make "these two checkpoints differ at layer
        4" untrue.

        `means` and `receipt` are dropped for the same reason the grounding
        export drops them: the sentence is regenerated from the numbers and
        the receipts list already holds the receipt once.
        """
        last = self._last_model_diff
        if not last:
            return {}
        return {k: v for k, v in last.items() if k not in ("receipt", "means")}

    def _ground_for_export(self) -> dict:
        """The last grounding, if it describes the model being exported.

        Same epoch rule as the head labels, and the same reasoning: grounding
        is measured on its own document and question rather than on the
        current generation, so it survives a new one -- and dies on a model
        swap, because "the answer came from the weights" is a claim about
        WHICH weights.

        `means` and `receipt` are dropped. The sentence is regenerated from
        the numbers by `Grounding.means()` on the way back out, so carrying a
        copy is a second thing to keep in step with the fields it describes,
        and the receipts list already holds the receipt once.
        """
        last = self._last_ground
        if not last or last.get("epoch") != self.epoch:
            return {}
        return {k: v for k, v in last.items() if k not in ("epoch", "receipt", "means")}

    def _ranking_for_export(self) -> dict:
        """The last head ranking, if it belongs to the state being exported.

        Same epoch rule as `_patch_for_export`, and the same reason: a ranking
        measured against an earlier prompt written beside these tokens would
        be a claim about which heads carry an answer the file does not show.
        """
        last = self._last_ranking
        if not last or last.get("epoch") != self.epoch:
            return {}
        # `receipt` is dropped: the receipts list carries it once already, and
        # a second copy inside the section is a second thing to keep in step.
        return {k: v for k, v in last.items() if k not in ("epoch", "receipt")}

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

    def _patch_graph_for_export(self) -> dict:
        """The last patching graph, if it belongs to the state being exported.

        Same epoch rule and the same reason as `_patch_for_export`. The
        `receipt` is dropped: `session.build` writes the run's receipts as one
        list, and a second copy nested inside a section would be a second
        answer to the question of what produced these numbers.
        """
        last = self._last_patch_graph
        if not last or last.get("epoch") != self.epoch:
            return {}
        return {k: v for k, v in last.items() if k not in ("epoch", "receipt")}

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
            # Same reason again, and the last of the four. `runtime.logit_lens`
            # has answered out of a recording since the section landed, and
            # nothing ever told a panel one was there -- so `FeaturesPanel`,
            # which owns the lens, stayed behind its `!replay` gate and the
            # trajectory in the file was displayed by nothing in either build.
            "lens": {
                "available": self.replay.has_lens(),
                # ROWS, not layers. A lens trajectory usually carries one more
                # row than the model has layers -- the embedding read before
                # any block runs -- and calling it `n_layers` here would make
                # the panel's own count disagree with the model's.
                "n_rows": len(self.replay.lens or []),
                # What the model actually said, so the panel can put the
                # trajectory under its own answer. Absent in a file that
                # carried the rows and not the scalars.
                "final": (self.replay.lens_info or {}).get("final"),
            },
            # Same reason as `patch`: the grounding panel needs a document and
            # a question, and a recording carries neither the document nor a
            # model to re-run it on. Telling the panel it HAS a result lets it
            # show the finding instead of a form whose only button refuses.
            "ground": {
                "available": self.replay.has_ground(),
                "question": self.replay.ground.get("question", ""),
            },
            # Same reason again. The graph cost 1,500 forward passes to build
            # and the recipient has no model to rebuild it with, so the panel
            # needs to know the recording HAS one — otherwise it offers a
            # button whose only outcome is a refusal.
            "patch_graph": {
                "available": self.replay.has_patch_graph(),
                "n_nodes": len(self.replay.patch_graph.get("nodes", [])),
                "n_edges": len(self.replay.patch_graph.get("edges", [])),
                # ITS OWN PROMPTS. `_patch_graph` has preserved these all
                # along and nothing read them: the panel was handed the PATCH
                # section's pair instead, so a file carrying a graph and no
                # patch trace prefilled its two boxes with the hardcoded demo
                # prompts -- a graph shown above the prompts it was not
                # measured on.
                "clean": self.replay.patch_graph.get("clean") or "",
                "corrupt": self.replay.patch_graph.get("corrupt") or "",
            },
            # THE HEAD RANKING, AND THE LABELS BEHIND IT. `ablate_heads` and
            # `head_types` have both answered out of a recording since those
            # sections landed, and `viewer.ts` implements the labels route
            # with a refusal written for recipients -- but nothing told the
            # panel either section was there.
            #
            # `AttentionPanel` gates the Rank-heads button on `!replay`, with
            # a comment that is true of MEASURING one ("needs a forward pass
            # per head") and not of SHOWING one already in the file. And the
            # only caller of the labels sits inside `{ranked && …}`, so the
            # labels were locked behind a button a recording could never
            # press: two sections shipped in the file and readable by nothing.
            "ranking": {
                "available": self.replay.has_ranking(),
                # What the ranking is ABOUT. A ranking names the token whose
                # logit it watched, and a table of heads with no target is a
                # list of numbers about nothing in particular.
                "target_token": self.replay.ranking.get("target_token"),
                "n_heads": len(self.replay.ranking.get("ranked", [])),
            },
            "head_types": {
                "available": self.replay.has_head_types(),
                "n_labels": len(self.replay.head_types.get("labels", [])),
            },
            # And the fourth, which was missing while its three siblings were
            # here. `session.py` parses an agent run out of a `.mri` and
            # `mcp_server` already reports whether one is present; the web UI
            # did not, so a bundle built around a failing step opened to an
            # agents panel reading "0 recordings" with the run sitting inside
            # the file. `step_ref` is the step the bundle was built around —
            # the reason it was sent — and it is carried here so the panel can
            # open on it rather than on step one.
            "trace": {
                "available": self.replay.has_trace(),
                # The run's own id, so a panel showing one carried run can
                # tell that the file underneath it has been swapped for
                # another rather than merely re-read.
                "id": self.replay.trace.get("id", ""),
                "name": self.replay.trace.get("name", ""),
                "n_steps": len(self.replay.trace.get("steps", [])),
                # What the SENDER's run held, which is not what this file
                # holds when the section was capped on the way in.
                "n_steps_total": self.replay.trace.get(
                    "n_steps_total", len(self.replay.trace.get("steps", []))
                ),
                "truncated": self.replay.trace.get("truncated", 0),
                "step_ref": self.replay.trace.get("step_ref") or None,
            },
        }

    # ---------------- SAE features ----------------

    def sae_status(self) -> SAEStatus:
        if self.sae is None:
            return SAEStatus(loaded=False)
        return self.sae.status()

    def sae_releases(self, repo: str, layer: int | None = None) -> dict:
        """What this repo publishes, so a picker can offer it rather than guess.

        A Gemma Scope release is addressed by (layer, dictionary width, average
        L0), and those are choices with consequences — four times the features
        at width_65k, an order of magnitude of sparsity between average_l0 22
        and 294. A panel that cannot show the alternatives cannot let anyone
        make the choice, so this hands over the whole index.

        `listed` is the honest half. False means the Hub could not be read, and
        `layers` is then None rather than {} — "unknown" and "this repo
        publishes nothing" are different answers and a picker must not render
        the first as the second.
        """
        from . import saes

        index = saes.release_index(repo)
        if index is None:
            return {
                "repo": repo,
                "listed": False,
                "layers": None,
                "note": (
                    "The list of published releases lives on the HuggingFace "
                    "Hub and could not be read just now. Naming a width and an "
                    "average L0 still loads one without it."
                ),
            }
        if layer is not None:
            index = {layer: index[layer]} if layer in index else {}
        return {
            "repo": repo,
            "listed": True,
            # JSON has no integer keys; a picker reading "20" back as a layer
            # index is doing string->int either way, and being explicit here
            # beats a dict whose keys change type on the wire.
            "layers": {str(n): index[n] for n in sorted(index)},
            "note": (
                "Widths and average-L0 values as published. Defaulting either "
                "one is reported in the loaded SAE's release.chosen_by."
            ),
        }

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
        with self._load_slot(f"load the GGUF {Path(path).name!r}"):
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
            self._steer_dir = None
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

        with self._load_slot("run the quantisation comparison"):
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
                self._steer_dir = None
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

    def load_sae(
        self,
        repo: str,
        hook: str,
        *,
        width: str | None = None,
        average_l0: int | None = None,
    ) -> SAEStatus:
        """Load an SAE and validate it against the current model. Blocking.

        `width` and `average_l0` address a Gemma Scope release, which is
        published per (layer, dictionary width, average L0) rather than per
        hook point. Both default to None, meaning "choose by the rule and say
        which rule" — the answer comes back in `status().release.chosen_by`,
        so a defaulted coordinate is never a silent one.
        """
        if self.backend == "ollama":
            raise Refusal(
                "SAE features need model internals — unavailable via Ollama. "
                "Load a HuggingFace model instead."
            )
        if not self.loaded:
            raise Refusal("Load a model first.")
        sae = SAEHandle.load(repo, hook, width=width, average_l0=average_l0)
        d_model = self.model.config.hidden_size
        if sae.d_in != d_model:
            raise BadRequest(
                f"SAE d_in={sae.d_in} does not match model hidden_size={d_model} "
                f"({self.hf_id}). This SAE was trained on a different model."
            )
        n_layers = text_config(self.model.config).num_hidden_layers
        if not 0 <= sae.layer < n_layers:
            raise BadRequest(f"SAE layer {sae.layer} out of range [0,{n_layers})")
        self._block(sae.layer)  # raises early if architecture unsupported
        self.sae = sae
        self._feats = None
        self._steer = None
        self._steer_dir = None
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
            # THE HOOK BELOW BELONGS TO THE BLOCK, NOT TO THIS CALL. Any
            # forward pass through the model while it is installed fills
            # `captured` too, and `generate_stream` runs `model.generate` on a
            # daemon thread that holds no lock -- so a stream still yielding
            # tokens puts its decode steps in here. The eight other
            # hook-installing measurements already refuse; this one is worse
            # than those if it does not, because its answer is CACHED.
            self._refuse_if_decoding("reading SAE features")
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
            # AND A BACKSTOP, because refusing is a race and this is not.
            # A COUNT, not a shape: this method's own pass calls the hook
            # exactly once, so a second entry means a foreign pass ran through
            # it and `captured[0]` may be that one rather than this one. The
            # shape check catches the other order -- a decode step is exactly
            # ONE token, so `captured[0]` comes back [1, 1, d] and would cache
            # a [1, d_sae] matrix beside a token list of length S, pairing
            # every token in the panel with somebody else's row.
            #
            # Caching is what makes this the worst of the family: the poisoned
            # matrix is served to every later features request until the next
            # generation, so one unlucky overlap misreads a panel indefinitely.
            n_tokens = int(self.last_ids.shape[0])
            if len(captured) != 1 or captured[0].shape[-2] != n_tokens:
                got = (
                    "x".join(str(d) for d in captured[0].shape) if captured else "none"
                )
                raise Refusal(
                    f"Another forward pass ran through this model while the "
                    f"features were being read, so what came back ({got}, "
                    f"{len(captured)} pass(es)) is not this prompt's "
                    f"{n_tokens} tokens. Nothing was cached. Wait for the run "
                    f"in flight to finish, then ask again."
                )
            resid = captured[0][0].to("cpu")  # [S, d_in]
            self._feats = self.sae.encode(resid).to(torch.float16)
        return self._feats

    def features_summary(self, top_k: int = 8) -> dict:
        """Per-token top-K firing features for the last generation."""
        feats = self._compute_features().float()  # [S, d_sae]
        # `/api/features/summary` takes `top_k` as a bare query integer, and
        # `topk` raises a torch error for anything above this SAE's width --
        # which reaches the caller as a 500 about an index, rather than as the
        # 422 it is. The bound is the SAE's own `d_sae`, read here rather than
        # written down: two SAEs for the same model routinely differ in width.
        width = int(feats.shape[-1])
        if not 1 <= top_k <= width:
            raise BadRequest(
                f"top_k must be in [1,{width}]. This SAE has {width} features, "
                f"and asking for more than it has is not a wider answer."
            )
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
        """Set (or clear, with feature_id=None) single-feature steering.

        SIGNATURE AND BEHAVIOUR UNCHANGED. The only addition is that both
        arms of the union now drop the other one — setting a feature replaces
        a direction and `feature_id=None` clears whichever was live.
        `FeaturesPanel.onSteerTest` calls this with None three times per A/B,
        including in its catch arm, on the promise that it "always leaves the
        model clean"; a direction it could not see would survive that and the
        next generation would come back steered with nothing on screen saying
        so.

        The None arm still short-circuits BEFORE the `self.sae is None`
        refusal, and that ordering is load-bearing —
        `tests/test_smoke.py::test_steer_clear_is_ok_without_sae` pins it.
        """
        if feature_id is None:
            self._steer = None
            self._steer_dir = None
        else:
            if self.sae is None:
                raise Refusal("No SAE loaded.")
            if not 0 <= feature_id < self.sae.d_sae:
                raise BadRequest(f"feature_id must be in [0,{self.sae.d_sae})")
            self._steer = (feature_id, float(scale))
            self._steer_dir = None
        return self.steering_status()

    def clear_steering(self) -> dict:
        """Take off whichever intervention is installed. Named, so `DELETE
        /api/steer` reads as what it does rather than as a POST of a null."""
        self._steer = None
        self._steer_dir = None
        return self.steering_status()

    def set_steering_direction(self, name: str, strength: float = 0.0) -> dict:
        """Install a SAVED direction from the vector store, or refuse by name.

        The second arm, and the one that works on the models nobody trains an
        SAE for. Everything that makes it safe happens in `steer_vectors.load`
        — which refuses a width this model's residual stream cannot be, naming
        both checkpoints — and everything that makes it honest happens in
        `residual_norm_at`, which measures the stream this coefficient is
        being added to so the strength can be reported relative to it.

        WARNINGS ARE CARRIED, NOT SWALLOWED. `load` warns rather than refuses
        for a direction fitted on a different model of the same width, because
        lifting one onto a finetune is a real experiment — but it must never
        be silent, so the sentences ride on the status and the panel prints
        them on the card.
        """
        from . import steer_vectors

        with self._lock:
            if self.replay is not None:
                raise Refusal(
                    "This is a recording. Steering means running the model "
                    "with a vector added to its residual stream, and a `.mri` "
                    "holds activations rather than weights."
                )
            if self.backend == "ollama":
                raise Refusal(
                    "Ollama serves text only — there is no residual stream "
                    "here to add a direction to."
                )
            if self.model is None:
                raise Refusal("No model loaded — pick one first.")

            hidden_size = int(text_config(self.model.config).hidden_size)
            vector, payload, warnings = steer_vectors.load(
                name, hidden_size=hidden_size, model=self.hf_id or ""
            )
            layer = payload.get("layer")
            if not isinstance(layer, int) or isinstance(layer, bool):
                raise Refusal(
                    f"{name!r} does not record which layer it was fitted at, "
                    "so there is nowhere to put it. A direction without its "
                    "layer is a vector without a basis."
                )
            # Bounds and architecture, in the module that owns both. Raises
            # BadRequest naming this model's layer count.
            self._block(layer)

            norm = float(vector.norm())
            if norm == 0.0:
                raise Refusal(
                    f"{name!r} is all zeros, so there is no direction in it "
                    "to push along."
                )
            # Unit norm at install, so `strength` is in the residual stream's
            # own units and `relative_strength` is the quotient it says it is.
            # Everything that writes into this store already normalises; doing
            # it again costs one division and closes the gap for a file that
            # arrived some other way.
            unit = (vector / norm).to(self.device)

            residual_norm, measured, unmeasured = self.residual_norm_at(layer)
            self._steer = None
            self._steer_dir = DirectionSteer(
                name=str(payload.get("name") or name),
                layer=layer,
                strength=float(strength),
                vector=unit,
                origin_model=str(payload.get("model") or ""),
                residual_norm=residual_norm,
                measured=measured,
                unmeasured=unmeasured,
                warnings=list(warnings),
                # The generation the norm above describes. None when there was
                # nothing to measure on, so an absent measurement is never
                # reported as a stale one.
                measured_on=self.last_ids if residual_norm is not None else None,
            )
            # `receipt` files it under its own op as well as returning it, so
            # `export_session` carries how the strength beside a steered run
            # was arrived at rather than only the number.
            self.receipt(
                "steer_direction",
                name=name,
                layer=layer,
                strength=float(strength),
                relative_strength=steer_vectors.relative_strength(
                    float(strength), residual_norm
                ),
                residual_norm=residual_norm,
                residual_norm_basis=measured or unmeasured,
                fitted_on=str(payload.get("model") or ""),
            )
            return self.steering_status()

    def residual_norm_at(self, layer: int) -> tuple[float | None, str, str]:
        """`(norm, how it was measured, why it was not)` at one layer.

        CALL THIS WITH `self._lock` HELD, for the reason
        `_require_live_generation` gives in capitals: it runs a forward pass,
        and a norm measured outside the lock can describe a model that has
        since been swapped out.

        ONE DEFINITION, shared with the fit. `steer_vectors.fit_direction`
        records `torch.cat([pos, neg]).norm(dim=-1).mean()` — the mean L2 norm
        of the last-token residual vectors entering that layer — and this is
        the same statistic taken on the current prompt instead of on the
        contrast pairs. Two definitions of "the stream's own norm" is exactly
        the failure `_steer_handle`'s docstring warns about for interventions,
        and it would be worse here because both numbers would look right.

        Costs ONE forward pass, which is why it happens at apply time and not
        on every status poll. Returns `None` and a sentence when there is no
        generation to measure on — an unknown, never a zero.
        """
        from . import steer_vectors

        if self.last_ids is None:
            return (
                None,
                "",
                "nothing has been generated yet, so there is no prompt to "
                "measure this model's residual norm on — the strength below "
                "is the raw coefficient with nothing to compare it against.",
            )
        if self.last_ids_epoch != self.epoch:
            return (
                None,
                "",
                "the last generation was produced by a different model, so "
                "its residual norm would not describe the stream this "
                "direction is being added to. Generate again to measure it.",
            )

        states = steer_vectors._last_token_states(
            self.model,
            self._block,
            [self.last_ids.unsqueeze(0).to(self.device)],
            [layer],
        )
        norm = float(states[layer].norm(dim=-1).mean())
        return (
            round(norm, 3),
            f"the mean L2 norm of the residual stream entering layer {layer} "
            f"at the last token of the current generation, measured on this "
            f"machine just now — the same statistic `fit_direction` records "
            f"at fit time",
            "",
        )

    def steering_status(self) -> dict:
        """What is installed, if anything.

        ADDITIVE ONLY. `{"active": False}` still has exactly one key, because
        `demo.ts` mirrors that shape offline with nothing type-checking it
        against this, and `test_smoke.py` and `e2e_check.py` both read
        `active` straight off the top. The feature arm keeps `feature_id` and
        `scale` where they were and gains `kind` and `layer` beside them.

        The relative strength is published for the DIRECTION arm only. It
        costs a forward pass to measure and it is measured once, at apply
        time; making the SAE-feature route pay for one would change what an
        existing call costs, which is not a thing to do quietly.
        """
        if self._steer_dir is not None:
            steer = self._steer_dir
            return {
                "active": True,
                "kind": "direction",
                "name": steer.name,
                "layer": steer.layer,
                "scale": steer.strength,
                "fitted_on": steer.origin_model,
                "warnings": list(steer.warnings),
                "strength": steer.strength_report(self.last_ids),
            }
        if self._steer is None:
            return {"active": False}
        fid, scale = self._steer
        return {
            "active": True,
            "kind": "feature",
            "feature_id": fid,
            "scale": scale,
            "layer": self.sae.layer if self.sae is not None else None,
        }

    def direction_catalogue(self) -> dict:
        """Every saved direction, judged against the model loaded right now.

        NO LOCK, deliberately. `load` holds `self._lock` across a whole model
        load, and a catalogue that blocked for the length of a download would
        be a list of files behind a progress bar. What it does instead is
        publish WHAT it judged against — `model` and `hidden_size` are in the
        payload beside the rows, so a reader can see which model each
        `compatible` was decided against rather than trusting that it was the
        one on screen.

        `compatible` is a THREE-state field. `False` is the positive claim
        "this cannot be applied here", and it comes with the exact sentence
        `steer_vectors.load` would refuse with. `None` is "nothing is loaded,
        so there is nothing to be compatible with" — which is not the same
        claim and must not render as a red cross.
        """
        from . import steer_vectors

        rows = steer_vectors.catalogue()
        hidden_size: int | None = None
        current = ""
        # BOUND ONCE. Without the lock, a load or an unload can land between
        # the guard and the config read, and `self.model` re-read on the
        # second line would be the None the first line just ruled out —
        # `AttributeError` and a 500 on the one route whose whole contract is
        # to answer with a list rather than a refusal. The local makes the two
        # statements describe the same model or no model at all, which is the
        # atomicity this needs and the only kind it can have without blocking.
        model = self.model
        if model is not None:
            hidden_size = int(text_config(model.config).hidden_size)
            current = self.hf_id or ""

        for row in rows:
            row["compatible"] = None
            row["mismatch"] = ""
            row["warnings"] = []
            if row.get("unreadable"):
                row["compatible"] = False
                row["mismatch"] = (
                    f"{row.get('name', 'this file')!r} is not a direction "
                    "this version can read, so nothing here knows what model "
                    "it belongs to."
                )
                continue
            if hidden_size is None:
                continue
            dims = row.get("hidden_size")
            if not isinstance(dims, int) or isinstance(dims, bool):
                row["compatible"] = False
                row["mismatch"] = (
                    "this direction does not record its own width, so it "
                    "cannot be checked against a model before being applied."
                )
                continue
            if dims != hidden_size:
                row["compatible"] = False
                row["mismatch"] = (
                    f"{row.get('name')!r} is a {dims}-dimensional direction "
                    f"and {current or 'this model'}'s residual stream is "
                    f"{hidden_size}. It was fitted on "
                    f"{row.get('model') or 'another model'}. Refusing rather "
                    "than reshaping it into something that would steer, "
                    "plausibly, at random."
                )
                continue
            row["compatible"] = True
            origin = row.get("model") or ""
            if current and origin and origin != current:
                row["warnings"].append(
                    f"this direction was fitted on {origin} and you are "
                    f"steering {current}. The hidden sizes match, but equal "
                    "size is not equal basis — the result may be confident "
                    "and meaningless."
                )
            if row.get("beats_null") is False:
                row["warnings"].append(
                    "this direction did not beat its own label-shuffled null "
                    "when it was fitted, so it was never evidence of anything."
                )

        return {
            "directions": rows,
            "model": current or None,
            "hidden_size": hidden_size,
            "means": (
                "Directions fitted from contrast pairs, or exported from a "
                "layer sweep by the probe panel. Each is only meaningful "
                "against the model it was fitted on: `compatible` is judged "
                "against the model named here, and `null` means nothing is "
                "loaded to judge against."
            ),
        }

    def fit_steering_direction(
        self,
        positive_texts: list[str],
        negative_texts: list[str],
        *,
        layers: list[int] | None = None,
        method: str = "caa",
        save_as: str = "",
        confirm: bool = False,
        estimate_only: bool = False,
    ) -> dict:
        """Fit a direction from contrast pairs, with its null at every layer.

        The reader-side twin of `probe_layers`: same capture point, same
        refusal order, same "what this cost" receipt. What is different is the
        estimator — `steer_vectors.fit_direction` scores on held-out pairs
        against label-shuffled refits, and the whole per-layer table comes back
        rather than a verdict, because the honest numbers ARE the product.

        TWO CALLS, ONE ROUTE. `estimate_only=True` spends one warm-up and one
        probe pass to say what the rest would cost ON THIS MACHINE and returns
        without fitting; the panel renders that, and the confirm runs it for
        real. Quoting somebody else's milliseconds is what `budget.py` exists
        to stop.
        """
        from . import budget, steer_vectors

        with self._lock:
            self._refuse_if_decoding("fitting a steering direction")
            if self.replay is not None:
                raise Refusal(
                    "This is a recording. Fitting a direction means running "
                    "the model on your contrast pairs, and a `.mri` does not "
                    "carry one."
                )
            if self.backend == "ollama":
                raise Refusal(
                    "Ollama serves text only — there is no residual stream "
                    "here to fit a direction in."
                )
            if self.model is None:
                raise Refusal("No model loaded — pick one first.")
            if method not in steer_vectors.METHODS:
                raise BadRequest(
                    f"unknown method {method!r} — use one of "
                    f"{', '.join(steer_vectors.METHODS)}"
                )

            positive = [
                t for t in (positive_texts or []) if isinstance(t, str) and t.strip()
            ]
            negative = [
                t for t in (negative_texts or []) if isinstance(t, str) and t.strip()
            ]

            # EVERY CHECK BEFORE A SINGLE FORWARD PASS. `probe_layers` learned
            # this the expensive way: its per-row validation was thorough and
            # never ran, because an empty list reached `torch.stack([])` and
            # answered 500 with a sentence about a tensor library. A sweep is
            # worse — the refusals below would arrive after 2n passes.
            if not positive or not negative:
                raise BadRequest(
                    "a direction is fitted from YOUR contrast pairs and one "
                    "of the two sets is empty. Give it matched lists — the "
                    "same sentence written two ways, one line each — and it "
                    "will report where in the network they come apart."
                )
            if len(positive) != len(negative):
                raise BadRequest(
                    f"{len(positive)} positive prompts against "
                    f"{len(negative)} negative ones — contrastive pairs must "
                    "be matched, because the direction is fitted from their "
                    "differences."
                )
            if len(positive) < steer_vectors.MIN_PAIRS:
                raise Refusal(
                    f"{len(positive)} pairs is not enough to fit a direction "
                    f"that can be checked. This needs at least "
                    f"{steer_vectors.MIN_PAIRS}, because half are held out "
                    "for scoring and a direction scored on its own fitting "
                    "set separates it by construction."
                )

            n_layers = int(text_config(self.model.config).num_hidden_layers)
            chosen = list(range(n_layers)) if not layers else [int(x) for x in layers]
            for layer in chosen:
                self._block(layer)  # bounds and architecture, in words

            ids_list = [
                self.tokenizer(t, return_tensors="pt")["input_ids"].to(self.device)
                for t in positive + negative
            ]
            passes = len(ids_list)

            # ONE REAL ITERATION of the loop being projected, hooks and all —
            # `budget.probe_pass`'s docstring is emphatic that a probe doing
            # less work than the body is wrong in the direction that approves
            # a run which then falls over. The warm-up first, because the
            # first pass after a load measured 3-4x the steady rate.
            def one_prompt() -> None:
                steer_vectors._last_token_states(
                    self.model, self._block, ids_list[:1], chosen
                )

            with torch.no_grad():
                self.model(ids_list[0])
            probe = budget.probe_pass(one_prompt, self.accel.kind)
            estimate = budget.project(probe, passes)

            out: dict = {
                "ran": False,
                "method": method,
                "n_pairs": len(positive),
                "passes": passes,
                "layers": [],
                "best_layer": None,
                "survived": 0,
                "estimate": estimate.to_dict(),
                "probe": probe.to_dict(),
                "means": (
                    "One forward pass per prompt, twice over: the two sets "
                    "are captured separately and every layer is fitted from "
                    "the same capture."
                ),
            }
            # THE QUOTE COMES BACK BEFORE THE GUARD RUNS. `budget.check`
            # raises `TooCostly`, and a price that refuses to be quoted
            # because it is high answers "how much would this cost?" with a
            # 409 — in exactly the case the panel asked for a number. The
            # estimate carries its own `verdict`, so a caller pricing a sweep
            # can see "refuse" and offer the override; `estimate_ablation`,
            # the other route of this shape, likewise leaves the check to the
            # run path. Pricing is not permission.
            if estimate_only:
                return out

            budget.check(
                estimate,
                label=f"fitting a direction over {len(chosen)} layers",
                confirm=confirm,
            )

            report, vectors = steer_vectors.sweep(
                self.model,
                self._block,
                ids_list[: len(positive)],
                ids_list[len(positive) :],
                chosen,
                method=method,
            )
            out.update(report)
            out["ran"] = True
            out["receipt"] = self.receipt(
                "fit_direction",
                method=method,
                n_pairs=len(positive),
                layers=len(chosen),
                passes=passes,
                positive_sha256=receipts.digest("\n".join(positive)),
                negative_sha256=receipts.digest("\n".join(negative)),
            )

            if save_as:
                best = report["best_layer"]
                if best is None:
                    # The same rule `probe_layers` keeps at its own save arm:
                    # the store is the one place this direction would later be
                    # picked up with none of the table beside it.
                    raise Refusal(
                        "no layer beat its own label-shuffled null, so there "
                        "is no direction here worth saving. A vector fitted "
                        "where the estimator produces the same separation "
                        "from shuffled labels is fitted to noise, and the "
                        "store is the one place it would later be used "
                        "without any of this beside it."
                    )
                row = next(r for r in report["layers"] if r["layer"] == best)
                out["saved"] = steer_vectors.save(
                    save_as,
                    vectors[best],
                    {
                        "model": self.hf_id or "",
                        "layer": best,
                        "hidden_size": int(vectors[best].shape[0]),
                        "method": method,
                        "dtype": str(next(self.model.parameters()).dtype).removeprefix(
                            "torch."
                        ),
                        "beats_null": row["beats_null"],
                        "p_value": row["p_value"],
                        "effect": row["effect"],
                        "null_max": row["null_max"],
                        "residual_norm": row["residual_norm"],
                        "n_pairs": row["n_pairs"],
                        "note": (
                            "fitted from contrast pairs by "
                            f"steer_vectors.{method}, scored on the held-out "
                            "half against label-shuffled refits."
                        ),
                    },
                )
            return out
