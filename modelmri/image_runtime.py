"""Hold one diffusers pipeline, and be honest about what holding it costs.

`imaging.py` says what a checkpoint IS. `image_attention`, `image_steps` and
`vision_attr` measure things about a pipeline somebody has already loaded.
Nothing owned the loading — so all of that was library code with tests and no
way to reach it from the UI.

This is that owner, and it is deliberately the same shape as `VLAHandle`: one
object, one lock, `load` / `unload` / `status`, blocking calls the server runs
off the event loop. A second lifecycle invented from scratch would be a second
set of bugs.

## What makes this different from the text runtime

A diffusion pipeline is several models — a denoiser, a VAE, one or two text
encoders — and `from_pretrained` pulls all of them. On the 8 GB card this
project targets that is the whole card, so:

**It refuses before it downloads.** `capacity.guard` gets the real byte count
read from the checkpoint's own safetensors headers, not an estimate from the
parameter count, and not after a twenty-minute download.

**It refuses to hold two.** A text model and a pipeline resident together is
the same two-processes-on-one-card problem `policy.check_capacity` exists for,
except here they are in ONE process and the OOM is immediate.

**It scans before it loads.** A diffusers pipeline is a directory of
checkpoints, and `from_pretrained` will happily unpickle a `.bin`. That window
is exactly what `weights_scan` was written for, so it runs here rather than
being something the user is trusted to remember.

## What it will not do

It does not generate images for their own sake. Every entry point exists to
support a measurement, and the pipeline's own final decode is skipped wherever
a measurement does not need it — the point is what happened inside, not the
picture.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import imaging
from .errors import BadRequest, Refusal

log = logging.getLogger(__name__)

# What one pipeline may cost before the guard wants a confirmation. Diffusion
# pipelines cluster around 2-7 GB in fp16; a 12 GB one is SDXL-plus-refiner
# territory and worth a sentence rather than a silent twenty-minute download.
LARGE_PIPELINE_BYTES = 12_000_000_000

# Components whose weights count toward what will be resident. `safety_checker`
# is excluded deliberately -- it is loaded as None by most modern pipelines and
# counting it would over-quote a number people plan around.
WEIGHTED_COMPONENTS = ("unet", "transformer", "vae", "text_encoder", "text_encoder_2")

# Weight file extensions, and `.bin` is in the set because REAL pipelines use
# it. Counting only safetensors read the ordinary cached
# `stabilityai/stable-diffusion-x4-upscaler` as **0.00 GB** when it is 1.7 GB
# of `.bin` — so `capacity.guard` saw "the source published nothing to go on",
# correctly allowed it through as unknown, and the refusal that exists to
# prevent an OOM would never have fired on a real model.
#
# That these are pickles is exactly why `_scan` runs before the load rather
# than after it.
WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".pt", ".pth", ".ckpt"})


class NotLoaded(Refusal):
    """No pipeline is held, and the message says what to load."""


@dataclass
class ImageStatus:
    """What is held, or why nothing is."""

    loaded: bool = False
    repo: str = ""
    family: str = ""
    architecture: str = ""
    device: str = ""
    dtype: str = ""
    # From `imaging.detect`, so a panel asks rather than infers.
    capabilities: list = field(default_factory=list)
    cross_attention_dim: int | None = None
    image_size: int | None = None
    components: dict = field(default_factory=dict)
    bytes_resident: int = 0
    load_seconds: float | None = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "loaded": self.loaded,
            "repo": self.repo,
            "family": self.family,
            "architecture": self.architecture,
            "device": self.device,
            "dtype": self.dtype,
            "capabilities": list(self.capabilities),
            "cross_attention_dim": self.cross_attention_dim,
            "image_size": self.image_size,
            "components": dict(self.components),
            "bytes_resident": self.bytes_resident,
            "load_seconds": self.load_seconds,
            "reason": self.reason,
            "means": self.means(),
        }

    def means(self) -> str:
        if not self.loaded:
            return (
                f"No image model is held in this process, so nothing here can "
                f"say what one attends to or when it commits. {self.reason}"
            ).strip()

        cross = (
            f" It attends to prompt tokens through a "
            f"{self.cross_attention_dim}-wide cross-attention."
            if self.cross_attention_dim
            else (
                " It is UNCONDITIONAL — no cross-attention to a prompt — so "
                "there are no word-to-pixel maps here to draw."
                if self.cross_attention_dim == 0
                else ""
            )
        )
        return (
            f"{self.repo} is held on {self.device or 'an unnamed device'} as "
            f"{self.dtype or 'an unstated dtype'}, "
            f"{self.bytes_resident / 1e9:,.1f} GB of weights.{cross} What can "
            f"be measured on it: {', '.join(self.capabilities) or 'nothing'}."
        )


class ImageHandle:
    """One pipeline at a time, with the lock that makes that true.

    One at a time is not a simplification. Two resident pipelines on an 8 GB
    card is an OOM in the middle of somebody's measurement, and the lock is
    what stops a second `load` racing a running capture.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.pipe = None
        self.status_ = ImageStatus(
            reason="Nothing has been loaded yet. Point it at a cached "
            "diffusers pipeline, or pull one."
        )

    # ------------------------------------------------------------- reading

    def status(self) -> ImageStatus:
        return self.status_

    def require(self):
        """The pipeline, or a refusal naming what to do."""
        if self.pipe is None:
            raise NotLoaded(
                f"No image model is loaded. {self.status_.reason} Load one "
                f"first — every measurement here runs the real pipeline."
            )
        return self.pipe

    # ------------------------------------------------------------- loading

    def load(
        self,
        repo: str,
        *,
        device: str = "",
        dtype: str = "",
        confirm: bool = False,
        already_held_bytes: int = 0,
    ) -> ImageStatus:
        """Bring a pipeline up. Blocking — the server runs this in a thread.

        The order is the point, and every step of it refuses before the next
        one costs anything:

          1. what IS this           `imaging.detect`, reading JSON only
          2. is anything in it live `weights_scan`, reading opcodes only
          3. will it fit           `capacity.guard`, from real byte counts
          4. only then, load
        """
        if not repo or not str(repo).strip():
            raise BadRequest(
                "no model was named. There is no default worth guessing: the "
                "checkpoint decides which panels apply."
            )
        repo = str(repo).strip()

        from . import imaging

        with self._lock:
            # Configs only. The family decides both whether this is loadable
            # at all and WHICH loader opens it, and both answers are in a few
            # kilobytes of JSON — so neither costs a download.
            local = _resolve_configs(repo)
            found = imaging.detect(local)
            if not found.known:
                # Refused before a weight is downloaded or scanned. A panel
                # drawn for the wrong family is a picture of something that
                # does not exist.
                raise Refusal(found.means())

            # Only now do the weights move.
            local = _resolve(repo)
            self._scan(local)
            resident = _weights_bytes(local)
            _guard(
                resident,
                local,
                confirm=confirm,
                already_held_bytes=already_held_bytes,
            )

            pipe, chosen_device, chosen_dtype, seconds = _load_pipeline(
                local, family=found.family, device=device, dtype=dtype
            )
            self.pipe = pipe
            self.status_ = ImageStatus(
                loaded=True,
                repo=repo,
                family=found.family,
                architecture=found.architecture,
                device=chosen_device,
                dtype=chosen_dtype,
                capabilities=list(found.capabilities),
                cross_attention_dim=found.cross_attention_dim,
                image_size=found.image_size,
                components=dict(found.components),
                bytes_resident=resident,
                load_seconds=round(seconds, 2),
            )
            return self.status_

    def unload(self) -> ImageStatus:
        """Drop it and hand the memory back, not merely forget it.

        `del` plus an allocator flush. Rebinding to None leaves the weights
        allocated until the next collection, and the next thing the user does
        is usually load something else.
        """
        with self._lock:
            had = self.status_.repo
            self.pipe = None
            _free()
            self.status_ = ImageStatus(
                reason=(
                    f"{had} was unloaded and its memory handed back."
                    if had
                    else "Nothing has been loaded yet."
                )
            )
            return self.status_

    # ------------------------------------------------------------ internals

    def _scan(self, local: Path) -> None:
        """Refuse a pipeline carrying something that executes on load.

        `from_pretrained` will unpickle a `.bin` without asking, and a
        diffusers pipeline is a directory of them. This window is precisely
        what `weights_scan` exists for, so it runs here rather than being
        something the user is trusted to remember.
        """
        from . import weights_scan

        for report in weights_scan.scan_dir(local, limit=400):
            if report.dangerous:
                raise weights_scan.Unsafe(report.means())


# The JSON a family can be named from. Kilobytes, and every one of them is
# read by `imaging.detect` — `model_index.json` for a pipeline,
# `config.json` for a single transformers checkpoint, `*/config.json` for a
# pipeline's components.
_CONFIG_PATTERNS = ["*.json", "*/*.json"]


def _resolve_configs(repo: str) -> Path:
    """Enough of `repo` to say WHAT it is, and nothing that weighs anything.

    The order in `load` claims that every step refuses before the next one
    costs anything, and for a while step zero quietly broke that promise:
    `_resolve` downloaded the entire repository, and only then did
    `imaging.detect` get a chance to say the family was one nothing here can
    open. Asking this tool to read `facebook/sam3` spent **fifteen minutes**
    pulling eight files and then raised a `diffusers` `OSError` about a
    missing `model_index.json` — a refusal that was knowable from a 4 KB
    config before a single weight moved.

    So the configs come down first. On a second call `snapshot_download`
    serves both from the same cache entry, so the JSON is not fetched twice
    and the weights are not fetched at all if the family is refused.
    """
    return _snapshot(repo, _CONFIG_PATTERNS)


def _resolve(repo: str) -> Path:
    """A local directory for `repo`, downloading only if it is not cached.

    A path that exists is used as-is, so somebody can point at a pipeline that
    never came from the Hub — the same rule `custom.py` follows.
    """
    # Weights and configs. No `.ckpt`/`.pt` mirrors of the same tensors, which
    # are usually duplicates and always pickles. `.bin` included, because most
    # published pipelines still ship it and excluding it downloaded a
    # directory of configs with no weights in it — `from_pretrained` then
    # failed with a confusing message about a missing file rather than the
    # honest one.
    #
    # It is a pickle, and that is not waved through: `_scan` walks every one
    # before anything loads, which is the whole reason that step is in the
    # sequence.
    return _snapshot(
        repo,
        [
            *_CONFIG_PATTERNS,
            "*.txt",
            "*.safetensors",
            "*.bin",
            "*/*.txt",
            "*/*.safetensors",
            "*/*.bin",
        ],
    )


def _snapshot(repo: str, allow: list) -> Path:
    """One cache entry, fetched to whatever depth the caller asked for.

    A path that exists is used as-is, so somebody can point at a checkpoint
    that never came from the Hub — the same rule `custom.py` follows.
    """
    candidate = Path(repo).expanduser()
    if candidate.is_dir():
        return candidate

    from huggingface_hub import snapshot_download

    from . import paths

    return Path(
        snapshot_download(
            repo_id=repo,
            cache_dir=str(paths.hf_hub_cache()),
            allow_patterns=list(allow),
        )
    )


def _weights_bytes(local: Path) -> int:
    """What will be resident, from the checkpoint's own headers.

    Read rather than estimated from a parameter count, and summed over the
    components that actually hold weights — `safety_checker` is excluded
    because most modern pipelines load it as None and counting it would
    over-quote a number people plan around.
    """
    total = 0
    for name in WEIGHTED_COMPONENTS:
        folder = local / name
        if not folder.is_dir():
            continue
        for weights in folder.iterdir():
            if not weights.is_file() or weights.suffix.lower() not in WEIGHT_SUFFIXES:
                continue
            try:
                total += weights.stat().st_size
            except OSError:
                continue
    if total:
        return total
    # A single-file pipeline, or a layout this does not know. Falling back to
    # the whole directory OVER-quotes rather than under-quotes, which is the
    # right direction for a number that gates a refusal.
    return sum(
        f.stat().st_size
        for f in local.rglob("*")
        if f.is_file() and f.suffix.lower() in WEIGHT_SUFFIXES
    )


def _guard(
    resident: int, target: Path, *, confirm: bool, already_held_bytes: int
) -> None:
    """Refuse before the download, not after it."""
    from . import capacity, devices

    if already_held_bytes > 0 and not confirm:
        raise capacity.TooBig(
            f"this process is already holding {already_held_bytes / 1e9:,.1f} "
            f"GB of weights, and a pipeline adds {resident / 1e9:,.1f} GB on "
            f"top of it. Unlike a model that can be offloaded, both of these "
            f"are wanted resident at once — unload the other one first.",
            overridable=True,
        )

    accel = devices.detect()
    capacity.guard(
        resident,
        target,
        label="this image pipeline",
        vram_gb=getattr(accel, "vram_gb", None),
        accel_name=getattr(accel, "name", ""),
        confirm=confirm,
    )


# Which loader opens which family, most specific first.
#
# `imaging.detect` already names every family this tool claims to read, and
# for a while `_load_pipeline` only knew ONE of them. Every ViT, CLIP,
# detector, segmenter and VLM went through `DiffusionPipeline.from_pretrained`
# and came back as a raw `diffusers` OSError about a missing
# `model_index.json` — a sentence about a file the user never heard of, for a
# checkpoint that is not a pipeline and was never going to have one.
#
# The fallback to `AutoModel` is the point of the tuples rather than a single
# name: a checkpoint for a family whose task head transformers does not
# expose still loads as a bare backbone, which is enough for `weights_scan`,
# the weight table, and patch attention. What is NOT done is silently
# pretending it loaded as the head — `_load_transformers` reports which class
# actually opened it, and `imaging` decides capabilities from the family, so
# a bare backbone never claims a head's measurements.
_TRANSFORMERS_LOADERS = {
    imaging.VIT: ("AutoModelForImageClassification", "AutoModel"),
    imaging.CLIP: ("AutoModel",),
    imaging.DETECTION: ("AutoModelForObjectDetection", "AutoModel"),
    imaging.SEGMENTATION: ("AutoModelForSemanticSegmentation", "AutoModel"),
    imaging.VLM: ("AutoModelForVision2Seq", "AutoModel"),
}

_DIFFUSION_FAMILIES = frozenset({imaging.UNET_DIFFUSION, imaging.DIT_DIFFUSION})


def _load_pipeline(local: Path, *, family: str, device: str, dtype: str):
    """Open the checkpoint with the loader its FAMILY actually needs."""
    import torch

    from . import devices

    accel = devices.detect()
    want_device = device or getattr(accel, "torch_device", "cpu")
    # The accelerator's own preferred dtype, not a hardcoded fp16. `devices`
    # already knows bf16 from fp16 from fp32 per backend, and picking fp16 on
    # a card without it is how you get a black image and no error.
    want_dtype = dtype or getattr(accel, "dtype", "float32")
    torch_dtype = getattr(torch, want_dtype, torch.float32)

    t0 = time.time()
    if family in _DIFFUSION_FAMILIES:
        model = _load_diffusion(local, torch_dtype)
    elif family in _TRANSFORMERS_LOADERS:
        model = _load_transformers(local, family, torch_dtype)
    else:
        # Unreachable through `load`, which refuses an unknown family before
        # it gets here. Stated anyway: a family added to `imaging` and not to
        # this table must say so rather than fall through to whichever loader
        # happens to be written first.
        raise Refusal(
            f"`{family}` is a family this tool can identify but has no loader "
            f"for yet, so there is nothing honest to open it with. It was "
            f"named rather than guessed at, and nothing was loaded."
        )

    model = model.to(want_device)
    # Inference only. A pipeline left in train mode still builds a graph, the
    # memory that costs is memory the measurement wanted, and `vision_attr`
    # refuses a training-mode model outright because dropout makes the same
    # input give a different answer every pass.
    if hasattr(model, "set_progress_bar_config"):
        model.set_progress_bar_config(disable=True)
    if hasattr(model, "eval"):
        model.eval()
    return model, str(want_device), str(want_dtype), time.time() - t0


def _load_diffusion(local: Path, torch_dtype):
    try:
        from diffusers import DiffusionPipeline
    except ImportError:
        raise Refusal(
            "Reading a diffusion pipeline needs the `diffusers` package, "
            "which is not installed. `pip install 'modelmri[image]'` adds it "
            "— it is optional because most people open a language model and "
            "should not pay for a dependency they will never import."
        ) from None

    return DiffusionPipeline.from_pretrained(
        str(local),
        torch_dtype=torch_dtype,
        # NEVER downloaded silently. A pipeline that needs code from the Hub
        # is a pipeline that runs somebody else's Python, and that decision
        # does not belong to a checkbox nobody read.
        trust_remote_code=False,
        safety_checker=None,
        requires_safety_checker=False,
    )


def _load_transformers(local: Path, family: str, torch_dtype):
    """A single transformers checkpoint, through the first class that opens it.

    Each candidate is tried in turn and the LAST failure is what gets
    reported. Reporting the first would name `AutoModelForObjectDetection`
    for a checkpoint whose only real problem is a corrupt weight file, which
    sends somebody looking for a head that was never the issue.
    """
    import transformers

    # transformers 5 renamed `torch_dtype` to `dtype` and warns on every load
    # that still uses the old name; 4.x only knows the old one. Both
    # `from_pretrained`s take `**kwargs`, so the signature cannot be asked and
    # the installed version is the only thing that answers. Read, not assumed:
    # pinning either name breaks on half the versions this supports.
    dtype_kw = "torch_dtype"
    try:
        if int(str(transformers.__version__).split(".")[0]) >= 5:
            dtype_kw = "dtype"
    except (AttributeError, ValueError):
        # A build with no parseable version. The older keyword is the safer
        # guess because 5 still accepts it, warning; 4 rejects the new one.
        pass

    last = None
    for name in _TRANSFORMERS_LOADERS[family]:
        auto = getattr(transformers, name, None)
        if auto is None:
            # This transformers build does not ship that class. Not an error:
            # the next candidate is there precisely for this.
            continue
        try:
            return auto.from_pretrained(
                str(local),
                # Same rule as the diffusion path, for the same reason.
                trust_remote_code=False,
                **{dtype_kw: torch_dtype},
            )
        except Exception as err:
            last = err

    # The exception TYPE, never its text. `from_pretrained` puts absolute
    # paths from this machine into its messages, and a refusal is something a
    # user pastes into an issue.
    why = (
        type(last).__name__
        if last is not None
        else "the installed transformers ships none of the classes for it"
    )
    raise Refusal(
        f"This is {imaging.label(family)}, and none of the loaders for that "
        f"family could open it: {why}. The checkpoint was identified from its "
        f"config before anything was loaded, so this is about the weights "
        f"rather than about what it is."
    ) from last


def _free() -> None:
    """Hand memory back to the allocator, on whichever backend this is."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as err:
        # Best-effort by nature: the collection above already did the part
        # that matters, and an allocator that will not flush is not a reason
        # to fail an unload. Logged rather than swallowed, because an
        # empty_cache that keeps failing is the first sign of a wedged
        # accelerator and the unload will look fine while memory does not
        # come back.
        log.debug("could not flush the allocator cache (%s)", type(err).__name__)
