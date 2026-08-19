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
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from . import fmt, imaging
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
    #: {capability: why it cannot be measured on this checkpoint}. Empty when
    #: everything the family offers is actually available here.
    unavailable: dict = field(default_factory=dict)
    cross_attention_dim: int | None = None
    image_size: int | None = None
    #: What steers this checkpoint — "text", "class", "none", or "" when it
    #: could not be read. A class-conditioned model takes a NUMBER from a
    #: fixed list and has no prompt at all, so a panel showing it a prompt box
    #: is asking a question it cannot be asked.
    conditioning: str = ""
    n_classes: int | None = None
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
            "unavailable": dict(self.unavailable),
            "cross_attention_dim": self.cross_attention_dim,
            "image_size": self.image_size,
            "conditioning": self.conditioning,
            "n_classes": self.n_classes,
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

        # WHAT IT TAKES, first, because it decides what the reader can even
        # ask. A class-conditioned checkpoint has no prompt: the panel used to
        # show one, and the click failed on a pipeline whose `__call__` has no
        # `prompt` parameter at all.
        # A MEASURED ZERO FIRST. `cross_attention_dim == 0` is the denoiser's
        # own config saying it attends to nothing — a stronger and different
        # claim from `None`, which is "the config did not say". The first
        # version of this ordering put the component-derived `conditioning`
        # ahead of it, so a pipeline listing a text encoder beside a
        # zero-width denoiser was described as taking a prompt. It does not,
        # and the test that caught it exists because drawing word maps for one
        # would be inventing them.
        if self.cross_attention_dim == 0:
            steer = (
                " It is UNCONDITIONAL — no cross-attention to a prompt — so "
                "there are no word-to-pixel maps here to draw."
            )
        elif self.conditioning == "class":
            steer = (
                f" It is CLASS-CONDITIONED on {self.n_classes or 'a fixed set of'} "
                f"labels — you give it a class number, not a prompt, so there "
                f"are no words here for a picture to have looked at."
            )
        elif self.conditioning == "none":
            steer = (
                " It is UNCONDITIONAL — nothing steers it, so there is neither "
                "a prompt nor a class to vary."
            )
        elif self.cross_attention_dim:
            steer = (
                f" It attends to prompt tokens through a "
                f"{self.cross_attention_dim}-wide cross-attention."
            )
        elif self.conditioning == "text":
            steer = (
                " It takes a text prompt; the denoiser's config does not state "
                "a cross-attention width, so how wide it is is unknown rather "
                "than absent."
            )
        else:
            steer = ""

        can = ", ".join(self.capabilities) or "nothing"
        cannot = ""
        if self.unavailable:
            # NAMED, not silently omitted. A control that is simply absent
            # leaves the reader wondering whether they missed it; this says
            # which measurement, and the reason travels with it so they can
            # tell whether another checkpoint would answer.
            cannot = (
                f" NOT available on this checkpoint: "
                f"{', '.join(sorted(self.unavailable))} — see `unavailable` "
                f"for why each one cannot be taken here."
            )
        # 0 IS NOT A SIZE HERE, it is a failed measurement — no weight file
        # could be read. The panel's own pill already says "resident weights
        # could not be sized" for exactly this case, and this sentence sat
        # one line under it claiming "0.0 GB of weights": two statements about
        # one quantity, on one screen, disagreeing about whether it is known.
        weight = (
            # `fmt.bytes_si`, not `/1e9`. Zero means "no weight file could be
            # read"; a small nonzero one is a real size that
            # `{n / 1e9:,.1f} GB` rounds away — measured on a 4 MB pipeline
            # that reported "0.0 GB of weights" from this very sentence, one
            # edit after the zero case was fixed and this one was not.
            f"{fmt.bytes_si(self.bytes_resident)} of weights"
            if self.bytes_resident > 0
            else "weights whose size could not be read"
        )
        return (
            f"{self.repo} is held on {self.device or 'an unnamed device'} as "
            f"{self.dtype or 'an unstated dtype'}, {weight}.{steer} What can "
            f"be measured on it: {can}.{cannot}"
        )


def _measurable(pipe, offered: tuple) -> tuple[list, dict]:
    """Which of `offered` this LOADED pipeline can actually support.

    Returns `(kept, withheld)` where `withheld` maps each dropped capability
    to the sentence explaining it.

    `imaging` reads the checkpoint's JSON and can therefore say that a
    pipeline with no text encoder has no words to attend to. It cannot say
    whether the intermediate latents are reachable, because that is a fact
    about the pipeline CLASS: diffusers exposes them only through
    `callback_on_step_end`, and a pipeline whose `__call__` does not take it
    has nothing to film.

    MEASURED on facebook/DiT-XL-2-256: `DiTPipeline.__call__` takes
    `class_labels`, `guidance_scale`, `generator`, `num_inference_steps`,
    `output_type` and `return_dict` — no `callback_on_step_end` — and
    `Transformer2DModel` has no `set_attn_processor`. All four advertised
    measurements refused at the click, after the reader had configured them.

    Every removal keeps its reason. "This cannot be measured" is a worse
    answer than the same thing with the sentence that says whether a different
    checkpoint would work.
    """
    import inspect

    kept, withheld = [], {}
    try:
        params = inspect.signature(type(pipe).__call__).parameters
    except (TypeError, ValueError):
        # An exotic callable this cannot introspect. Nothing is withheld on
        # that basis: a capability removed because we could not look is a
        # guess, and the run itself refuses honestly if it turns out to be
        # unsupported.
        params = None

    denoiser = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)

    for cap in offered:
        if cap in ("step_commit", "latent_trace") and params is not None:
            if "callback_on_step_end" not in params:
                withheld[cap] = (
                    f"`{type(pipe).__name__}.__call__` does not accept "
                    f"`callback_on_step_end`, which is the only place "
                    f"diffusers exposes an intermediate latent. The run would "
                    f"produce its final image and nothing in between, so there "
                    f"is nothing here to measure between the steps."
                )
                continue
        if cap in ("cross_attention", "token_knockout") and denoiser is not None:
            if not hasattr(denoiser, "set_attn_processor"):
                withheld[cap] = (
                    f"`{type(denoiser).__name__}` does not expose "
                    f"`set_attn_processor`, which is how the attention "
                    f"probabilities are captured where they are computed. "
                    f"Reconstructing them from hidden states afterwards would "
                    f"be a different quantity from the one the model used."
                )
                continue
        kept.append(cap)
    return kept, withheld


class ImageHandle:
    """One pipeline at a time, with the lock that makes that true.

    One at a time is not a simplification. Two resident pipelines on an 8 GB
    card is an OOM in the middle of somebody's measurement, and the lock is
    what stops a second `load` racing a running capture.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.pipe = None
        # The checkpoint's OWN preprocessor, and `None` when it did not ship
        # one. It is not a convenience: a ViT is trained on a specific size
        # and a specific per-channel normalisation, and an image resized and
        # scaled by anything else produces logits that are a fact about the
        # wrong tensor. Every occlusion score would then be noise wearing the
        # model's name, so the absence is reported and refused rather than
        # papered over with a plausible default.
        self.processor = None
        # WHY there is no processor, in the project's own words. Empty when
        # there is one. Kept because "there is none" and "it needs a package
        # you do not have" send a reader to two different places.
        self.processor_reason = ""
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

    def require_processor(self):
        """The checkpoint's preprocessor, or a refusal saying why there is none.

        Separate from `require` because the two failures need different
        actions. "Nothing is loaded" is fixed by loading something; "this
        checkpoint published no preprocessor" is not fixed by loading it
        again, and a measurement that guessed the normalisation would return
        confident numbers about a tensor the model was never trained on.
        """
        self.require()
        if self.processor is None:
            raise NotLoaded(
                f"`{self.status_.repo}` cannot be preprocessed here: "
                f"{self.processor_reason or 'no preprocessor was found'}. "
                f"Guessing a size and a normalisation produces logits about a "
                f"tensor this model was never trained on — confident numbers "
                f"that are not about your image — so it is refused instead."
            )
        return self.processor

    # ------------------------------------------------------------- loading

    def load(
        self,
        repo: str,
        *,
        device: str = "",
        dtype: str = "",
        confirm: bool = False,
        already_held_bytes: int = 0,
        local_ok: bool = True,
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

        # ONE list, resolved once and handed to the tracker, the killable
        # child and the in-process fetch alike. Three callers deriving it
        # separately is three chances for the bar to be counting different
        # files from the ones arriving.
        allow = (
            _allow_for(repo)
            if not (local_ok and Path(repo).expanduser().is_dir())
            else list(_WEIGHT_PATTERNS)
        )
        with self._lock, _tracked(repo, allow) as tracking:
            tracking.stage("identify", "reading the checkpoint's own JSON")
            # Configs only. The family decides both whether this is loadable
            # at all and WHICH loader opens it, and both answers are in a few
            # kilobytes of JSON — so neither costs a download.
            local = _resolve_configs(repo, local_ok=local_ok)
            found = imaging.detect(local)
            if not found.known:
                # Refused before a weight is downloaded or scanned. A panel
                # drawn for the wrong family is a picture of something that
                # does not exist.
                raise Refusal(found.means())

            _stop_if_asked("before the weights were fetched")
            tracking.stage("weights", "fetching whatever is not cached")
            # Only now do the weights move. In a child process, so the Stop
            # button works during the one step long enough for anyone to want
            # it — see `_prefetch`. `_resolve` still runs afterwards and is
            # what decides the load; by then the cache is usually full and it
            # touches the network not at all.
            _prefetch_weights(repo, allow, local_ok=local_ok)
            local = _resolve(repo, allow, local_ok=local_ok)
            _stop_if_asked("before the weights were scanned")
            tracking.stage("scan", "reading opcodes — loading nothing")
            self._scan(local)
            resident = _weights_bytes(local)
            _guard(
                resident,
                local,
                confirm=confirm,
                already_held_bytes=already_held_bytes,
            )

            _stop_if_asked("before the pipeline was opened")
            tracking.stage(
                "open", "opening the pipeline — this step cannot be interrupted"
            )
            pipe, chosen_device, chosen_dtype, seconds = _load_pipeline(
                local, family=found.family, device=device, dtype=dtype
            )
            self.pipe = pipe
            # Best-effort and NEVER fatal: a diffusion pipeline has no image
            # processor and does not want one, so its absence must not fail a
            # load. What must not happen is a measurement that needs it
            # proceeding without it — that is `require_processor`'s job.
            self.processor, self.processor_reason = _load_processor(local)
            # CHECKED AGAINST THE OBJECT, not guessed from the family. See
            # `_measurable` for the four measurements this caught being
            # advertised on a checkpoint that supports none of them.
            measurable, withheld = _measurable(pipe, tuple(found.capabilities))
            tracking.stage("ready")
            tracking.finish()
            self.status_ = ImageStatus(
                loaded=True,
                repo=repo,
                family=found.family,
                architecture=found.architecture,
                device=chosen_device,
                dtype=chosen_dtype,
                capabilities=measurable,
                # WHAT CANNOT BE MEASURED, AND WHY. A panel that simply omits
                # a control leaves the reader wondering whether they missed
                # it; one that says "not on this checkpoint, because …" tells
                # them whether another would work.
                unavailable=withheld,
                cross_attention_dim=found.cross_attention_dim,
                image_size=found.image_size,
                conditioning=found.conditioning,
                n_classes=found.n_classes,
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
            # Dropped with the model it belongs to. A processor left behind
            # would be the previous checkpoint's normalisation applied to the
            # next one's input — the exact wrong-tensor failure
            # `require_processor` exists to prevent, arriving through the back
            # door and looking like a working measurement.
            self.processor = None
            self.processor_reason = ""
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

#: Configs plus the weight formats this stack can actually open. ONE list,
#: because the killable child and the in-process fetch have to ask for exactly
#: the same files — see `_prefetch_weights`.
_WEIGHT_PATTERNS = [
    *_CONFIG_PATTERNS,
    "*.txt",
    "*.safetensors",
    "*.bin",
    "*/*.txt",
    "*/*.safetensors",
    "*/*.bin",
]


def _resolve_configs(repo: str, *, local_ok: bool = True) -> Path:
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
    return _snapshot(repo, _CONFIG_PATTERNS, local_ok=local_ok)


def _one_copy(names: list[str]) -> list[str]:
    """One copy of each component's weights, from a repo's full file list.

    Grouped by DIRECTORY, which is what a diffusers component is. Grouping by
    filename stem does not work: transformers writes `pytorch_model.bin` and
    `model.safetensors` for the same tensors, so the two copies share no stem
    at all. A shard set (`model-00001-of-00002.safetensors`) sits in one
    directory and is kept whole.
    """
    from collections import defaultdict

    weights: dict[str, list[str]] = defaultdict(list)
    other: list[str] = []
    for name in names:
        if name.endswith((".safetensors", ".bin", ".pth")):
            weights[name.rsplit("/", 1)[0] if "/" in name else ""].append(name)
        else:
            other.append(name)

    keep: list[str] = []
    for group in weights.values():
        # Format first. `use_safetensors` defaults to preferring them, so a
        # directory holding both had its pickle downloaded and never read.
        safe = [n for n in group if n.endswith(".safetensors")]
        chosen = safe or group
        # Then precision. `_load_pipeline` never asks for a variant, so an
        # fp16 twin cannot be opened by this tool — but if it is the ONLY
        # thing published for that component, dropping it would leave the
        # component with no weights at all.
        plain = [n for n in chosen if ".fp16." not in n]
        keep.extend(plain or chosen)
    return sorted(keep + other)


def _allow_for(repo: str) -> list[str]:
    """Exactly what to fetch for `repo`, or the full pattern set if unknown.

    The return value is handed to BOTH the downloader and the progress
    tracker, which is what keeps the bar's denominator equal to the bytes
    that are actually going to move.
    """
    from fnmatch import fnmatchcase

    try:
        from huggingface_hub import HfApi

        # `model_info`, not `list_repo_files`, for one reason: the latter takes
        # no timeout, and this call sits in front of the download on the load's
        # critical path. An unbounded listing here would reintroduce, one line
        # earlier, exactly the hang `ETAG_TIMEOUT_S` exists to stop.
        #
        # `files_metadata=False` — the sizes are not wanted here. The tracker
        # asks for them separately, and requesting them makes the Hub do more
        # work for an answer this function throws away.
        info = HfApi().model_info(repo, files_metadata=False, timeout=ETAG_TIMEOUT_S)
        names = [f.rfilename for f in (info.siblings or [])]
    except Exception as err:
        # No listing, no de-duplication — fetch everything rather than guess.
        # Too much is a slow load; too little is a load that dies on a missing
        # file, and those are not the same cost.
        log.info(
            "could not list %s (%s); fetching every published weight format",
            repo,
            type(err).__name__,
        )
        return list(_WEIGHT_PATTERNS)
    matched = [n for n in names if any(fnmatchcase(n, pat) for pat in _WEIGHT_PATTERNS)]
    return _one_copy(matched) or list(_WEIGHT_PATTERNS)


def _prefetch_weights(repo: str, allow: list, *, local_ok: bool = True) -> None:
    """Killable pre-download of exactly what `_resolve` will ask for.

    Skipped for a directory that already exists, on the same test `_snapshot`
    uses — a local checkpoint has nothing to download, and spawning a child to
    discover that would add a process to every load from disk.

    The allow-list is READ FROM `_resolve` rather than copied. Two lists that
    are meant to be identical and are written twice are two lists that drift,
    and the failure mode is silent: the child fetches one set, `_resolve` then
    quietly downloads the difference in-process, un-killably, and Stop goes
    back to doing nothing for exactly the files it was needed for.
    """
    if local_ok and Path(repo).expanduser().is_dir():
        return
    _prefetch(repo, allow)


def _resolve(repo: str, allow: list | None = None, *, local_ok: bool = True) -> Path:
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
    return _snapshot(repo, allow or _WEIGHT_PATTERNS, local_ok=local_ok)


#: How long a Hub METADATA check may take before the Hub is treated as
#: unreachable. Deliberately short, and deliberately separate from the
#: download budget: the check is a few hundred bytes and the fetch is
#: gigabytes, so a single ceiling for both would either hang on the check or
#: abort the fetch.
ETAG_TIMEOUT_S = 8.0


class ImageLoadCancelled(RuntimeError):
    """The load was stopped on request. NOT a failure — somebody asked.

    A RuntimeError rather than a Refusal because nothing was refused: the
    request was fine and the answer is that it did not finish. The route
    answers 200 with a plain sentence, the way the text side does, so the
    panel does not paint a red error over something the reader did on purpose.
    """


@contextmanager
def _tracked(repo: str, allow: list):
    """Publish this load's progress, and mark it finished however it ends.

    A context manager on the same `with` line as the handle's lock, which is
    the reason the load body below is untouched. Every exit finishes the
    tracker — a refusal is a FINISHED load, not a running one, and leaving it
    active would leave the panel spinning on a job that already answered.
    That is the exact failure this exists to remove, so it must not be
    reintroduced by an unhandled path.
    """
    from . import progress

    # The SAME list `_resolve` hands `snapshot_download`. Not a copy of it and
    # not a rule that approximates it: a diffusion pipeline keeps every weight
    # in a subfolder, and the tracker's own shape rule counted only the root
    # `model_index.json` — 584 bytes against 1.3 GB, drawn as a full bar.
    progress.IMAGE_LOADS.start(repo, tuple(allow))
    try:
        yield progress.IMAGE_LOADS
    except ImageLoadCancelled as err:
        progress.IMAGE_LOADS.finish(error=str(err), cancelled=True)
        raise
    except (Refusal, BadRequest) as err:
        progress.IMAGE_LOADS.finish(error=err.sentence)
        raise
    except BaseException as err:
        # BaseException, deliberately. A KeyboardInterrupt or a worker being
        # torn down must not leave the tracker reporting an active load
        # forever — the panel would poll a job nothing is running.
        progress.IMAGE_LOADS.finish(error=f"{type(err).__name__} while loading")
        raise


def _stop_if_asked(when: str) -> None:
    """Raise if a stop was asked for. Called BETWEEN stages only."""
    from . import progress

    if progress.IMAGE_LOADS.cancelled.is_set():
        raise ImageLoadCancelled(f"Load stopped {when}.")


#: Run in a CHILD interpreter so the download can be killed. Takes the repo,
#: the cache directory and the allow-list as JSON, because an allow-list built
#: by string-formatting into source is an injection waiting for a repo name
#: with a quote in it.
_PREFETCH = """
import json, sys
from huggingface_hub import snapshot_download
repo, cache, allow = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
snapshot_download(repo_id=repo, cache_dir=cache, allow_patterns=allow)
"""


def _prefetch(repo: str, allow: list) -> None:
    """Fetch `repo` in a killable child, so Stop can stop it mid-download.

    Returns normally whether the child succeeded or failed — the caller's
    `_snapshot` is what actually decides the load, and it will either find a
    full cache (fast) or download the rest itself (correct). The ONLY outcome
    this raises for is a stop the user asked for.
    """
    import json
    import os
    import subprocess
    import sys

    from . import paths, progress, runtime

    # Progress bars to a pipe nobody drains deadlocked this exact pattern on
    # the text side once already: hub writes tqdm to stderr, the ~64 KB buffer
    # fills, the child blocks forever. Both streams go to DEVNULL and must
    # stay there.
    env = {**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _PREFETCH,
                repo,
                str(paths.hf_hub_cache()),
                json.dumps(list(allow)),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            # Windows: its own group, so terminate() does not also signal the
            # server that spawned it.
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            if sys.platform == "win32"
            else 0,
        )
    except Exception as err:
        # No usable interpreter, a box that refuses process creation, a denied
        # process group. Said out loud rather than swallowed, because the only
        # other symptom is "Stop does not stop the download" — which nobody
        # reports as a bug in here.
        log.warning(
            "prefetch child unusable for %s (%s: %s); the download will run "
            "in-process and Stop will not interrupt it",
            repo,
            type(err).__name__,
            err,
        )
        return

    try:
        while proc.poll() is None:
            if progress.IMAGE_LOADS.cancelled.wait(0.4):
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                freed = runtime._clean_partials(repo)
                raise ImageLoadCancelled(
                    f"Download stopped. Removed {freed / 1e6:,.0f} MB of "
                    f"partial files; anything already complete was kept."
                )
    finally:
        if proc.poll() is None:  # an exception on our side, not the child's
            proc.terminate()


def _snapshot(repo: str, allow: list, *, local_ok: bool = True) -> Path:
    """One cache entry, fetched to whatever depth the caller asked for.

    A path that exists is used as-is, so somebody can point at a checkpoint
    that never came from the Hub — the same rule `custom.py` follows.

    `local_ok=False` switches that off, and the server passes it for any
    request that did not come from the person at the keyboard. The reason is
    that a Hub id and a directory name overlap: `models` is a valid repo id
    AND a directory that may sit in the server's working directory, so a shape
    test alone cannot separate them. Without this a remote caller naming
    `models` had the server's own `./models` opened for them.

    `expanduser()` therefore only ever runs for a local caller, which is the
    only context in which reading HOME/USERPROFILE is that caller's business.
    """
    if local_ok:
        candidate = Path(repo).expanduser()
        if candidate.is_dir():
            return candidate

    from huggingface_hub import snapshot_download

    from . import paths

    try:
        return Path(
            snapshot_download(
                repo_id=repo,
                cache_dir=str(paths.hf_hub_cache()),
                allow_patterns=list(allow),
                # A CEILING on the metadata check, which is the part that
                # hangs. `snapshot_download` revalidates against the Hub even
                # when every file is already cached, and with no timeout a
                # stalled HEAD blocks forever — measured: a load of a fully
                # cached 6.3 GB pipeline that never returned, holding the
                # handle lock so every later load queued behind it with no
                # ceiling of its own. A check that cannot answer in seconds is
                # not going to; the download it guards keeps its own budget.
                etag_timeout=ETAG_TIMEOUT_S,
            )
        )
    except Exception as err:
        # Already complete here? Then open it. The revalidation failing is a
        # fact about the NETWORK, and declining to use files that are whole on
        # this disk because a HEAD request timed out would be the tool
        # refusing a job it can do offline.
        try:
            cached = Path(
                snapshot_download(
                    repo_id=repo,
                    cache_dir=str(paths.hf_hub_cache()),
                    allow_patterns=list(allow),
                    local_files_only=True,
                )
            )
        except Exception:
            cached = None
        if cached is not None:
            # SAID, not silent. "Served from the cache without revalidating"
            # is a different claim from "revalidated", and the difference is
            # exactly staleness — which is the reader's to weigh, not this
            # function's to hide.
            log.warning(
                "could not reach the Hub for %s (%s); using the complete copy "
                "already in the cache, unrevalidated",
                repo,
                type(err).__name__,
            )
            return cached

        # A name that is not on the Hub is a fact about the REQUEST, and the
        # server was answering it with 500 "Something inside ModelMRI failed
        # rather than refusing" — which is this project telling on itself: an
        # unhandled exception where a refusal belonged.
        #
        # The class only, never the text. `snapshot_download` puts the full
        # URL, the cache directory and an authentication paragraph into its
        # message, and the cache directory is a path on this machine.
        raise Refusal(_hub_refusal(repo, err)) from err


def _hub_refusal(repo: str, err: Exception) -> str:
    """Why a download did not happen, in this project's words rather than the
    Hub client's.

    The three cases a reader acts on differently: the name is wrong, the repo
    needs credentials, or the network is not there. Anything else names the
    exception class, which says what KIND of failure it was without quoting a
    library.
    """
    name = type(err).__name__
    if "RepositoryNotFound" in name or "EntryNotFound" in name:
        return (
            f"`{repo}` is not a repository on the Hub, and it is not a "
            f"directory on this machine either. Nothing was downloaded."
        )
    if "GatedRepo" in name:
        return (
            f"`{repo}` is gated on the Hub — its owner requires you to accept "
            f"terms and be authenticated before it can be downloaded. Nothing "
            f"here holds credentials for you."
        )
    if "RevisionNotFound" in name:
        return f"`{repo}` exists on the Hub but the revision asked for does not."
    if "LocalEntryNotFound" in name or "ConnectionError" in name or "Offline" in name:
        return (
            f"`{repo}` is not in this machine's cache and the Hub could not be "
            f"reached to fetch it ({name}). Nothing was downloaded, and this is "
            f"about the network rather than about the model."
        )
    return (
        f"`{repo}` could not be fetched from the Hub ({name}). Nothing was downloaded."
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


# The classes that can carry an image preprocessor, most specific first.
# `AutoProcessor` is here because a multimodal checkpoint publishes a
# COMPOSITE processor — a tokenizer and an image processor in one object —
# and `AutoImageProcessor` does not load it. Reaching only for the latter
# read `facebook/sam3`, which ships `processor_config.json`, as having no
# preprocessor at all.
_PROCESSOR_CLASSES = ("AutoImageProcessor", "AutoProcessor")


def _load_processor(local: Path) -> tuple:
    """The checkpoint's own image preprocessor, and WHY there is none if so.

    Returns `(processor, reason)`. Never raises. A diffusers pipeline ships
    no preprocessor and does not want one, so a failure here is the ordinary
    case for half the families this loads — making it fatal would refuse
    every diffusion model over a file it was never supposed to have.

    ## The reason is the whole point of the second return value

    The first version returned a bare `None`, and the refusal built on it
    said "`facebook/sam3` did not publish an image preprocessor". That was
    FALSE. sam3 publishes one; loading it raised `ImportError` because
    torchvision is not installed on this machine. A broad `except` had
    collapsed a fact about the machine into a claim about the checkpoint,
    and it sent a reader to look for a missing file that is right there.

    A missing optional dependency is fixable in one command. A checkpoint
    that genuinely has no preprocessor is not fixable at all. Reporting the
    second when it is the first is the more expensive of the two mistakes.
    """
    try:
        import transformers
    except ImportError:
        return None, (
            "the `transformers` package is not installed, and it is what "
            "reads a checkpoint's preprocessor"
        )

    missing_dependency = ""
    last = None
    for name in _PROCESSOR_CLASSES:
        auto = getattr(transformers, name, None)
        if auto is None:
            continue
        try:
            found = auto.from_pretrained(
                str(local),
                # Same rule as every other loader here.
                trust_remote_code=False,
            )
        except ImportError as err:
            # A LIBRARY this machine does not have, not a checkpoint that
            # lacks a file. transformers raises this for torchvision-backed
            # fast processors, and the package name is the actionable half.
            missing_dependency = _missing_package(err) or "a library"
            continue
        except Exception as err:
            last = err
            continue
        # A composite processor holds the image half as an attribute. The
        # image half is what turns a picture into the tensor the model was
        # trained on, so that is what travels — not the tokenizer beside it.
        inner = getattr(found, "image_processor", None)
        return (inner if inner is not None else found), ""

    if missing_dependency:
        return None, (
            f"reading its preprocessor needs `{missing_dependency}`, which is "
            f"not installed here. The checkpoint published one — this is a "
            f"missing package on this machine, not a missing file in the model"
        )
    if last is not None:
        # The TYPE only. `from_pretrained` puts absolute paths from this
        # machine into its messages.
        return None, (
            f"its preprocessor could not be read ({type(last).__name__}), so "
            f"nothing here knows what size or normalisation it expects"
        )
    return None, "it did not publish an image preprocessor"


def _missing_package(err: ImportError) -> str:
    """Which package an ImportError is about, without relaying its message.

    `err.name` is set when the import failed on a module rather than on a
    name inside one, and a module name is bounded — it cannot carry a path.
    transformers raises a plain `ImportError` with prose for its optional
    backends, so the prose is SCANNED for a known package name rather than
    relayed: the text says "requires the Torchvision library but it was not
    found in your environment", and quoting that verbatim is how library
    text reaches a browser.
    """
    if getattr(err, "name", None):
        return str(err.name).split(".")[0]
    said = str(err).lower()
    for package in ("torchvision", "torchaudio", "pillow", "timm", "av"):
        if package in said:
            return package
    return ""


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
