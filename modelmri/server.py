"""FastAPI application: REST for control, WebSocket for token streams."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from . import __version__, behavdiff, custom, gguf_read, image_attention, otel, paths
from . import image_steps as _steps_defaults
from . import openai_api as openai_api_mod
from .custom import AdapterError, CustomHandle

# Every browser-facing handler below answers `str(err)` for these types, and
# CodeQL flags each one as py/stack-trace-exposure. Those alerts are dismissed
# as false positives, which is a claim, so here is the claim and where it is
# checked.
#
# NOT suppressed with inline `# codeql[...]` comments: that syntax is a CodeQL
# CLI convention and GitHub code scanning does not honour it for Python. I
# tried it, and the only effect was to put all 69 handlers into the diff and
# turn seven alerts into sixty-nine. Dismissal through the code-scanning API
# is the mechanism that exists, and unlike a query filter it leaves the rule
# armed for handlers written later.
#
# CodeQL is right in general: `str(err)` on an arbitrary exception publishes a
# library's internal text, and this repo has already shipped six of those (see
# the 0.10 changelog, "SIX PLACES PUBLISHED A LIBRARY'S OWN TEXT"). What makes
# it safe HERE is that these types are only ever constructed with authored
# sentences — never with a caught exception's text, except at four sites in
# custom.py which are marked `leak-ok` because the text there is the reader's
# own adapter code.
#
# That invariant is enforced, not asserted:
# `tests/test_no_machine_leaks.py::test_every_published_exception_that_embeds_
# another_is_marked` walks every raise site in the package, and a companion
# test walks the four internal error types that runtime.py re-raises verbatim.
# Both are mutation-checked. Break the invariant and the suppression stops
# being honest and the tests say so.
from .errors import BadRequest, Refusal
from .image_runtime import ImageLoadCancelled
from .runtime import DEFAULT_MODEL, ModelRuntime, _load_failed
from .traces import TraceStore, record_generation
from .vla import DEFAULT_VLA_REPO as VLA_DEFAULT_REPO
from .vla import VLAHandle

log = logging.getLogger("modelmri")

# THE THREE ANSWERS, AND THE ORDER THEY ARE WRITTEN IN.
#
# Every handler below that can fail ends in the same three arms:
#
#     except Refusal as err:     409, str(err)   — we decided not to answer
#     except BadRequest as err:  422, str(err)   — the request is wrong
#     except Exception as err:   500, _INTERNAL  — something broke
#
# The point of the split is that the first two messages were written for a
# reader and the third was not. Before it, a CUDA out-of-memory and "this is a
# recording, and a .mri does not carry a model" were both 409s carrying the
# exception's own text — so a full GPU was reported as a conflict, and torch's
# text, which can name directories on this machine, went to the browser.
#
# Do not put `str(err)` on the 500 arm. That is the entire change.


# Local-first: the person who opened this page also started the process, so
# the honest place to send them is the terminal they already have. A message
# that says only "internal error" would be true and useless.
_INTERNAL = (
    "Something inside ModelMRI failed rather than refusing. The full error "
    "is in the terminal running `modelmri serve`."
)


# Reading a LeRobot dataset needs two optional packages, and "it is not
# installed" is a sentence with a one-line fix — so it is a refusal rather than
# a 500. The module name comes from `err.name`, a field ImportError publishes
# for exactly this, and not from `str(err)`: the message is ours, the name is a
# lookup key, and nothing else from the exception is republished.
def _body_object(body: object):
    """`None` when the body is an object, or the 422 that says it is not.

    RETURNS the response rather than raising it, and that is deliberate.
    This check has to run before each handler's own `try`, because that
    try is what defaults a MISSING body to `{}` — a refusal raised inside
    it would be swallowed by the block meant to report it.

    Eleven handlers read `await request.json()` and index the result.
    Seven of them already refuse a body that is not JSON at all; none of
    them refused a body that is valid JSON and not an object, so a bare
    array reached `.get` and raised AttributeError into the generic 500.
    """
    if isinstance(body, dict):
        return None
    return JSONResponse(
        {
            "error": (
                f"This endpoint takes a JSON object, and the request body "
                f"was a {type(body).__name__}. There is nothing in it to "
                f"read the measurement's arguments from."
            )
        },
        status_code=422,
    )


def _whole(body: dict, field: str, default: int) -> int:
    """One integer field, or a refusal naming the field and what it held.

    `int(body.get(field, 0))` raised `ValueError: invalid literal for int()
    with base 10: 'abc'` straight into a 500 — a sentence about CPython rather
    than about the request. Several of these routes already answer a NUMERIC
    bad value with an authored refusal, so this only makes the two paths
    agree.

    A missing key and an explicit `null` both take the default, which is the
    behaviour `body.get(f) or default` already had and is what lets a panel
    omit a control it is not showing.
    """
    raw = body.get(field)
    if raw is None or raw == "":
        return default
    # Bools FIRST: `isinstance(True, int)` is True, so a JSON `true` would
    # otherwise pass straight through as the number 1.
    if isinstance(raw, bool):
        raise BadRequest(
            f"`{field}` must be a whole number, and this request sent "
            f"{str(raw).lower()}."
        )
    if isinstance(raw, int):
        return raw
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError) as err:
        raise BadRequest(
            f"`{field}` must be a whole number, and this request sent {raw!r}."
        ) from err


def _real(body: dict, field: str, default: float) -> float:
    """One float field, on the same rule as `_whole`."""
    raw = body.get(field)
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        raise BadRequest(
            f"`{field}` must be a number, and this request sent {str(raw).lower()}."
        )
    if isinstance(raw, (int, float)):
        return float(raw)
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError) as err:
        raise BadRequest(
            f"`{field}` must be a number, and this request sent {raw!r}."
        ) from err


#: Ceilings for the two image preflights. Not arbitrary: a denoising run of
#: more than this is minutes per step on any consumer card, and a knockout of
#: more than this many words is one arm per word — both are numbers a caller
#: reaches by typo rather than by intent, and quoting a confident price for
#: them is how a cost route stops being worth reading.
MAX_DENOISE_STEPS = 1000
MAX_KNOCKOUT_WORDS = 200


def _bounded(value: int, field: str, *, low: int, high: int) -> int:
    """One integer inside its range, or a refusal naming the range.

    A preflight exists so the number arrives before anything is spent. One
    that answers "-40 denoising passes" — measured — has inverted that: the
    figure is what the reader decides on, and a confident impossible one is
    worse than none.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise BadRequest(f"`{field}` must be a whole number, not {value!r}.")
    if not low <= value <= high:
        raise BadRequest(
            f"`{field}` must be between {low} and {high:,}, and this asked for "
            f"{value:,}. Nothing here will price a run it would refuse."
        )
    return value


def _missing_reader_dep(err: ImportError) -> JSONResponse:
    """409 for a missing pyarrow / av, with the real ImportError in the log."""
    log.warning("LeRobot reader dependency missing", exc_info=err)
    what = f" ({err.name} is missing)" if getattr(err, "name", None) else ""
    return JSONResponse(
        {
            "error": "Reading a LeRobot dataset needs pyarrow for the metadata "
            f"and av for the video{what}. Install them with "
            "`pip install modelmri[vla]`."
        },
        status_code=409,
    )


def _internal(err: BaseException, where: str) -> JSONResponse:
    """The 500 arm: generic to the browser, complete to the log.

    Logging is not decoration here. The handlers this replaced returned the
    exception's text, which at least meant the failure was visible somewhere;
    dropping the text without recording it would trade a leak for an erasure,
    and an erasure is the worse bug.
    """
    log.exception("unhandled error in %s", where, exc_info=err)
    return JSONResponse({"error": _INTERNAL}, status_code=500)


# WHERE `_unmigrated` WENT, BECAUSE IT WAS THE THING THIS PASS EXISTED TO KILL.
#
# There used to be a helper here that caught a bare RuntimeError or ValueError
# on twelve routes and answered 409/422 with `str(err)`. Its justification was
# that the modules behind those routes still raised plain exceptions for their
# deliberate refusals, so relaying the text kept their answers working until
# each one adopted Refusal.
#
# The justification stopped being true and nobody noticed. Counted at the time
# it came out: hub.py 3 taxonomy raises and 0 plain, ollama.py 4 and 0,
# lens.py 2 and 0, traces.py 4 and 0, vla.py 9 and 0, vla_data.py 9 and 1,
# saes.py 2 and 1 — and both of those remaining plain raises carry a comment
# in their own file saying they belong on the 500 path. So the arms were no
# longer transitional. They were catch-alls, and what they actually caught was
# breakage: measured, a torch-shaped `RuntimeError("CUDA out of memory ...
# <absolute path>")` came back as 409 with that path in the body on
# /api/hub/models, /api/ollama/pull, /api/vla/load, /api/vla/attention,
# /api/vla/analyse, and 422 on /api/sae/load and /api/traces/import.
#
# It also logged at `log.debug`, and this logger has no handler and an
# effective level of WARNING under what `modelmri serve` installs — so those
# failures were leaked to the browser AND erased from the terminal at the same
# time, which is both halves of the bug `_internal` was written to avoid.
#
# The one module that genuinely still raised a plain exception was session.py.
# It now raises `SessionError(BadRequest)`, which was always the honest
# classification, so it needs no arm either.


class Body(BaseModel):
    """A request body that REFUSES a key it does not know.

    Pydantic ignores unknown fields by default, and the consequence is not
    cosmetic. Measured against the running server:

        POST /api/model/load {"model_id": "Qwen/Qwen3-1.7B"}
        -> 200 {"loaded": true, "hf_id": "Qwen/Qwen2.5-0.5B-Instruct", ...}

    The field is `hf_id`. A caller who wrote `model_id` was told the load
    succeeded, and a DIFFERENT model was loaded — every panel then measuring a
    model nobody asked for, with nothing anywhere saying the key was dropped.

    The dangerous version of this is not the load. It is a sweep or a probe
    whose parameter name was misspelled: the run uses the default, finishes,
    and reports numbers labelled as though the parameter had been applied. A
    silently wrong measurement presented as a right one is the single failure
    this project is built to avoid, and `extra="ignore"` manufactures it from
    one typo.

    422 with the offending key named is the honest answer. It is also the
    cheaper one to debug: the alternative is noticing, later, that the answer
    was about something else.
    """

    model_config = ConfigDict(extra="forbid")


class LoadRequest(Body):
    # `min_length`, which `ImageLoadRequest.repo` has carried for months and
    # this did not. An explicit `{"hf_id": ""}` went all the way to
    # transformers and came back "Could not load '', and this is not one of
    # the failures ModelMRI knows how to explain" — the sentence reserved for
    # failures the tool genuinely does not understand, about the one input it
    # understands perfectly. The default still applies when the key is absent,
    # which is how the client asks for it: `api.ts` omits `hf_id` rather than
    # sending an empty one.
    hf_id: str = Field(default=DEFAULT_MODEL, min_length=1, max_length=400)
    source: str = "hf"  # "hf" | "ollama"
    # The user saw the size warning and chose to proceed anyway.
    confirm: bool = False
    # Where to put it: "cuda:1", "cpu", or "" to let the tool choose as it
    # always has. Empty by default so a machine with one GPU behaves exactly
    # as before — the choice is offered, not imposed.
    device: str = ""


class GgufLoad(Body):
    # `min_length=1`, and it is not decoration. `Path("").resolve()` is the
    # process's own working directory, which exists and is inside an allowed
    # root, so an empty field did not fail as an empty field — it named the
    # server's source tree. MEASURED: `{"path": ""}` answered 409 "<the repo
    # this server was started in> is not a file", a directory the request
    # never mentioned, with no next step a reader could take.
    path: str = Field(min_length=1, max_length=4096)
    # None means "whatever this accelerator prefers". Named explicitly rather
    # than defaulted to float32 here, because the dtype is half of the memory
    # figure and a silent default would make the preflight describe a load
    # nobody asked for.
    dtype: str | None = None
    # Overrides a tight fit, and nothing else.
    confirm: bool = False


class QuantCompare(Body):
    # Both bounded, for the reason spelled out on `GgufLoad.path` — and this
    # route is where the empty string was worst. `original` resolved to the
    # cwd, the cwd IS an allowed root, and the empty field was handed to the
    # loader as "the full-precision checkpoint": a run documented as expensive
    # proceeding against the server's own source tree, not refused at all.
    quantised: str = Field(min_length=1, max_length=4096)
    original: str = Field(min_length=1, max_length=4096)
    prompt: str = "The capital of France is"
    # Off makes the run cheaper when only the token-level answer is wanted.
    attention: bool = True


class SAELoadRequest(Body):
    """Which SAE to load. Empty means "the one for the model that is loaded".

    NOT defaulted to a repo id. A constant here is one model's release
    answering for every model: `{}` used to ask for the gpt2 SAE whatever was
    resident, and against anything else that is a d_in mismatch rather than a
    load. The registry knows which release belongs to which model; this asks
    it.
    """

    repo: str = ""
    hook: str = ""
    #: Gemma Scope is published per (layer, dictionary width, average L0).
    #: `runtime.load_sae` has always taken these; the request model did not,
    #: so the coordinates could not be named through the API at all. None
    #: still means "choose by the rule and say which rule", reported back in
    #: `release.chosen_by`.
    width: str | None = None
    average_l0: int | None = None


class SteerRequest(Body):
    feature_id: int | None = None  # None clears steering
    scale: float = Field(default=0.0, ge=-100.0, le=100.0)


class HubSignInRequest(Body):
    token: str = Field(min_length=1, max_length=400)


class OllamaPullRequest(Body):
    name: str = Field(min_length=1, max_length=200)
    # The user saw the size warning and chose to proceed. Never a default,
    # and never enough to override a disk that has no room.
    confirm: bool = False


class VLALoadRequest(Body):
    # One constant, in vla.py. Two copies of a default is two things to
    # forget to change.
    repo: str = VLA_DEFAULT_REPO


class VLAAnalyseRequest(Body):
    episode: int = Field(default=0, ge=0)
    t: int = Field(default=0, ge=0)


class VLADatasetRequest(Body):
    repo_id: str = Field(min_length=1, max_length=200)


class ScanRequest(Body):
    """A checkpoint or a directory of them.

    `limit` bounds a directory walk, and what it drops is reported rather than
    silently omitted — a scan that stopped at 200 files reads as "200 files,
    all fine".
    """

    path: str = Field(min_length=1, max_length=4096)
    limit: int = Field(default=200, ge=1, le=5000)


class ImageLoadRequest(Body):
    """A cached pipeline directory, or a Hub id.

    No default. The checkpoint decides which panels apply, so guessing one
    would silently decide what the user is looking at.
    """

    repo: str = Field(min_length=1, max_length=400)
    device: str = Field(default="", max_length=32)
    dtype: str = Field(default="", max_length=32)
    confirm: bool = False


class ImageRunRequest(Body):
    """One prompt through the pipeline, with the seed that makes it repeat.

    `seed` is optional and `None` is NOT 0. A diffusion run without a fixed
    seed is one draw from a distribution, and every comparison downstream —
    knockout especially — is meaningless without the same seed on both arms.
    """

    prompt: str = Field(min_length=1, max_length=4000)
    steps: int = Field(default=20, ge=1, le=200)
    seed: int | None = Field(default=None, ge=0, lt=2**31)


class ImageKnockoutRequest(ImageRunRequest):
    """Which words to remove, one at a time.

    `seed` is REQUIRED here rather than optional: every arm has to run at the
    identical seed or the difference between two images is the sampler rather
    than the word.
    """

    # The bound lives in `image_attention` and is published on the image
    # status, so the panel can disclose it before the click rather than after.
    words: list[str] = Field(
        default_factory=list, max_length=image_attention.MAX_KNOCKOUT_WORDS
    )
    seed: int = Field(default=0, ge=0, lt=2**31)


class TraceToDatasetRequest(Body):
    """Recorded runs, turned into a set of cases you can re-run.

    No expected answers anywhere in this request, deliberately. The row is
    evidence that a run happened; deciding what the right answer was is a
    judgement, and one invented here would be indistinguishable from one the
    reader made.
    """

    trace_ids: list[str] = Field(default_factory=list, max_length=500)
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    only_errors: bool = Field(default=False)


class ExperimentCompareRequest(Body):
    """Two runs of one dataset, case by case.

    `higher_is_better` has NO default and is required. There is no way to tell
    from a metric's name which direction is good — KL divergence is better
    lower, faithfulness is better higher — and a wrong guess inverts every
    conclusion while producing output that looks entirely reasonable. Pydantic
    rejecting the request is the right place for that to be caught.
    """

    before: str = Field(min_length=1)
    after: str = Field(min_length=1)
    metric: str = Field(min_length=1, max_length=120)
    higher_is_better: bool
    # The dataset the two runs share, so the comparison can quote what each
    # case actually asked. Optional, and its absence is REPORTED rather than
    # filled in: `references: null` means nothing looked, `0` means it looked
    # and there were none.
    dataset: str | None = Field(default=None)
    floor: float | None = Field(default=None)
    top_k: int = Field(default=12, ge=1, le=200)


class ScorerRunRequest(Body):
    """One scorer, one output, and whatever that scorer needs to compare to.

    `reference` is `None` rather than "" for the scorers that need nothing to
    compare against — `json_valid` asks only whether the text parses — and the
    two are different: an empty expected string is a real expectation.
    """

    name: str = Field(min_length=1, max_length=64)
    output: str = Field(default="", max_length=200_000)
    reference: object | None = Field(default=None)
    # Per-scorer knobs: tolerances, normalisation, ignore_extra_keys. Bounded
    # so a request cannot carry an arbitrary object graph into a metric.
    options: dict | None = Field(default=None)


class TrajectoryCompareRequest(Body):
    """What was supposed to happen, against what did.

    Both sides are lists of recorded-step dicts, or of bare names, or a
    mixture — `trajectory` normalises them and never opens a store itself, so
    the caller does the fetching and this stays testable without one.
    """

    reference: list = Field(default_factory=list, max_length=2000)
    candidate: list = Field(default_factory=list, max_length=2000)


class ImageAttributionRequest(Body):
    """One picture, covered up a window at a time.

    The image travels IN the request as a data URL, never as a path: a path
    in a body names a file on the server's disk, which is somebody else's
    machine as often as it is yours, and a browser cannot produce one for a
    file the user picked anyway.

    `target` is `None` for "whatever the model itself says", which is the
    ordinary question. Naming a class asks a different one — auditing a label
    you supplied rather than attributing the model's own answer — and the
    result says which of the two it was.
    """

    image: str = Field(min_length=1)
    patch: int = Field(default=16, ge=1, le=512)
    # `None` means "the patch size", i.e. non-overlapping windows. Not
    # defaulted to a number here, because `vision_attr` refuses a stride wider
    # than the patch and it should be the one deciding that.
    stride: int | None = Field(default=None, ge=1, le=512)
    fill: str = Field(default="grey")
    target: int | None = Field(default=None, ge=0)
    batch: int = Field(default=32, ge=1, le=64)


class AdapterRequest(Body):
    """A LoRA on this machine, by path.

    A PATH rather than an upload: an adapter is tens of megabytes and already
    on the disk of whoever is asking. Unlike the image routes, this reads a
    file the caller names — which is the same trust boundary
    `/api/weights/scan` already sits behind, and it is refused from anywhere
    but this machine for the same reason.
    """

    path: str = Field(min_length=1)
    top: int = Field(default=40, ge=1, le=500)


class CVPredictRequest(Body):
    """One picture, and what the model says about it.

    The image travels IN the request as a data URL for the same reason
    `ImageAttributionRequest` does: a path in a body names a file on the
    server's disk, which is somebody else's machine as often as it is yours.
    """

    image: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    # A detector's boxes and a segmenter's masks come with scores, and the
    # threshold decides what is reported as found. Named rather than fixed:
    # the right cut for a crowded street scene is not the right cut for a
    # single object on a plain ground.
    mask_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class CVReadoutRequest(Body):
    """Just the picture.

    This route took `CVPredictRequest`, which declares `top_k` and
    `mask_threshold` — and `layer_readout` reads neither: it returns per-layer
    maps, not class predictions, so there is no ranking to cut and no mask to
    threshold. Both were accepted and silently discarded, so a caller tuning
    them watched nothing change and had no way to learn why.

    Its own model, so a request naming either is refused by name. `Body`
    forbids unknown keys, which is what makes that refusal say which field it
    is rather than shrugging.
    """

    image: str = Field(min_length=1)


class CVAttributeRequest(Body):
    """Occlusion attribution over a prediction this model actually made.

    `target` names WHICH answer to attribute. `None` is "whatever the model
    itself said", which is the ordinary question -- naming one asks a
    different question, auditing a label you supplied, and the result says
    which of the two it was. `region` narrows a segmenter's mask; `query`
    picks a detector's box.
    """

    image: str = Field(min_length=1)
    target: int | None = Field(default=None, ge=0)
    query: int | None = Field(default=None, ge=0)
    region: tuple[int, int, int, int] | None = None
    patch: int = Field(default=16, ge=1, le=512)
    stride: int | None = Field(default=None, ge=1, le=512)
    fill: str = Field(default="grey")
    batch: int = Field(default=32, ge=1, le=64)


class StepTraceRequest(Body):
    """One run, keeping every step's latent, to find where it committed.

    Deliberately NOT the filmstrip. This decodes nothing -- it measures how far
    the latent moved per step, which is a property of the denoiser rather than
    of the VAE that would draw it. The two answer different questions and the
    responses say so.
    """

    prompt: str = Field(min_length=1)
    steps: int = Field(default=20, ge=1, le=200)
    seed: int | None = None
    # 0 means "use the module's own default". Not a magic number here: the
    # threshold decides which step gets called the commit point, and this file
    # restating a value `image_steps` owns is two places to change it.
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class FilmstripRequest(Body):
    """Watch the picture form, decoding only the steps you name.

    `every` and `at` are two ways to choose the same thing and both are
    reported back, because a strip of 8 frames from a 50-step run must never
    be mistakable for a 50-step recording. Decoding every step would not fit
    beside the pipeline in 8 GB anyway.
    """

    prompt: str = Field(min_length=1)
    steps: int = Field(default=20, ge=1, le=200)
    seed: int | None = None
    every: int | None = Field(default=None, ge=1, le=200)
    at: list[int] | None = None
    include_final: bool = True
    # The LONGEST SIDE of each emitted frame, not a pixel count. Bounds and
    # default imported from the module that enforces them: a second copy here
    # is a second opinion, and the first version of this accepted only values
    # `image_steps` refuses.
    frame_pixels: int = Field(
        default=_steps_defaults.DEFAULT_FRAME_PIXELS,
        ge=_steps_defaults.MIN_FRAME_PIXELS,
        le=_steps_defaults.MAX_FRAME_PIXELS,
    )


class VLAFrameRequest(Body):
    """One frame, plus the seed that makes the answer reproducible.

    `seed` is optional and `None` is NOT 0. None means "do not fix the
    sampler", which is a different request and a different claim about the
    result — most of these policies sample, so an unseeded run is one draw
    from a distribution rather than the policy's answer.
    """

    episode: int = Field(default=0, ge=0)
    t: int = Field(default=0, ge=0)
    seed: int | None = Field(default=None, ge=0, lt=2**31)


class VLACompareRequest(Body):
    """A whole episode, strided.

    `stride=0` means "choose one that fits the work budget" rather than
    "measure every frame" — an unstrided 200-frame episode is 200 forward
    passes, and `vla_actions.plan_frames` reports whatever it picks.
    """

    episode: int = Field(default=0, ge=0)
    stride: int = Field(default=0, ge=0, le=1000)
    seed: int | None = Field(default=None, ge=0, lt=2**31)


class CustomLoadRequest(Body):
    path: str = Field(min_length=1, max_length=4096)


class CustomRootRequest(Body):
    # A folder to also look in, for a model that does not live where the
    # server was started. Added to the allowed roots rather than bypassing
    # them — see custom.add_root.
    path: str = Field(min_length=1, max_length=4096)


class CustomRunRequest(Body):
    # None means "use the adapter's example_input()"; the panel sends the
    # shape it showed you, so nothing ever runs on a shape you didn't see.
    shape: list[int] | None = Field(default=None, max_length=8)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class PatchGraphRequest(Body):
    """A pair, and how far back to walk from what the grid flagged."""

    clean: str
    corrupt: str
    # 0 means "the module's default". Named rather than defaulted here so the
    # two do not drift apart the way a duplicated constant always does.
    depth: int = 0
    max_receivers: int = 0


class PatchRequest(Body):
    # Two prompts, not one, and the pair is the unit of meaning: neither is
    # usable without the other, so they arrive together rather than as a
    # prompt plus a query parameter.
    clean: str
    corrupt: str


class PromptRequest(Body):
    prompt: str
    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    commit: bool = True


class RubricScoreRequest(Body):
    """A rubric to score every recorded run against.

    `rules` is REQUIRED and named. It used to arrive as a bare `dict`, and
    the route then called `parse(body.get("rules", body))` — so a body without
    a `rules` key was handed to the parser whole, and `rubric.parse` turned any
    dict lacking that key into `[]`, which is a legal empty rubric.

    Measured: `{"rulez": [...]}` — one transposed letter — answered 200 with
    `counts: {}` and "0 rule(s) against 111 recorded run(s). No run matched any
    rule.", where the correct spelling reported 66 of them matching. A
    confident all-clear, produced by a typo, on a route whose whole job is
    telling you which runs went wrong. `{"Rules": ...}` and `{}` did the same.

    This is verbatim the failure `Body`'s own docstring says the class exists
    to prevent; the route simply was not using it.
    """

    rules: list | dict


class RubricSaveRequest(Body):
    """The same rubric, under a name, kept on this machine."""

    name: str = Field(min_length=1)
    rules: list | dict


class JudgeRequest(Body):
    """A rubric put to the loaded model, and the text to put it about.

    Shared with `/api/judge/plan`, which prices this exact request — two
    models would let the priced call and the run drift apart on the very
    parameter the price is a function of.

    Both routes read these with `.get` off a bare dict before, so a misspelled
    key reached a default silently: `{"n_paraphrase": 5}` answered 200 with the
    default four-prompt plan, where the correct spelling answered 422 naming
    the cap.
    """

    text: str = ""
    rubric: str = ""
    n_paraphrases: int = Field(default=0, ge=0)


class _Recording:
    """One generation, timed as the app performs it, filed as a trace after.

    Both generation paths — POST /api/model/prompt and the /ws/generate
    socket — stream pieces out of `runtime.generate_stream` and both end in
    exactly two ways, a completion or a message saying why not. This holds the
    clock and the pieces so each of them is three lines rather than a second
    copy of the arithmetic.

    ONLY COMMITTED RUNS. `commit=False` is the steering A/B firing two throwaway
    completions to compare against each other; the panel is a record of runs
    you made, not of the tool's internal probes, and four rows per A/B would
    bury the run they were comparing.
    """

    def __init__(self, runtime: ModelRuntime, prompt: str) -> None:
        self._runtime = runtime
        self.prompt = prompt
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._t0 = time.perf_counter()
        self.pieces = 0
        self._text: list[str] = []
        # Read NOW, not at filing time. A load can complete while this is
        # still streaming — the same race `generate_stream` guards its commit
        # against — and the trace has to name the model that actually ran.
        self.model = str(runtime.hf_id or "")
        self.backend = str(runtime.backend or "")

    def piece(self, text: str) -> None:
        self.pieces += 1
        self._text.append(text)

    def file(self, store: TraceStore, failure: str | None = None) -> None:
        """Store what happened. Silent on success, harmless on failure.

        `failure` is the SENTENCE the reader was given, never `str(err)` —
        see the three arms at each call site, and record_generation's
        docstring for why a trace in particular must not carry torch's words.
        """
        record_generation(
            store,
            model=self.model,
            backend=self.backend,
            prompt=self.prompt,
            output="".join(self._text),
            started_at=self.started_at,
            duration_ms=int((time.perf_counter() - self._t0) * 1000),
            # THE BEST COUNT EACH BACKEND CAN HONESTLY GIVE, AND NO MORE.
            #
            # In: the local tokenizer, or None on Ollama, which tokenises in
            # another process and streams text back with no counts in it.
            # Out: the streamed pieces — which is the same number the
            # playground puts on screen for this run ("257 tok · 14.12s"), so
            # the two views of one generation cannot disagree, and it works on
            # every backend without asking anything of them.
            tokens_in=self._runtime.count_tokens(self.prompt),
            tokens_out=self.pieces,
            failure=failure,
        )


def _is_loopback(host: str) -> bool:
    """Is this client on the same machine?

    Parsed rather than compared against a list of three spellings. Loopback is
    all of 127.0.0.0/8 and ::1, and a dual-stack listener reports an IPv4 peer
    as the IPv4-mapped `::ffff:127.0.0.1` — so `host in ("127.0.0.1", "::1")`
    refuses `127.0.0.5` and the mapped form, both of which are this machine.
    `ipaddress` already knows the ranges; a hand-written list is a smaller
    version of the same knowledge that drifts.

    A host that is not an IP at all is NOT loopback. That covers `localhost`,
    which is a name a resolver decides the meaning of rather than an address,
    and it is why the tests hand the client an explicit address: a request
    whose origin cannot be established is not one to widen a boundary for.
    """
    import ipaddress

    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _not_from_this_machine(
    request, doing: str, because: str = ""
) -> JSONResponse | None:
    """A refusal when the request did not come from the person at the keyboard.

    Two checks, because loopback alone does not settle it.

    A page on ANY website can POST to `localhost`, and the request arrives from
    127.0.0.1 like every other. A JSON body already forces a preflight that
    fails without CORS headers -- there are none -- but relying on that is
    relying on a side effect, so an `Origin` from anywhere else is refused
    explicitly: a defence you can read rather than one you have to derive.

    And a client on the network is not the user either. `serve` defaults to
    loopback, but `--host` takes anything, and on a server bound to 0.0.0.0
    every handler here is reachable by whoever can route to the port.

    Used by everything that turns a string from a request body into a path on
    THIS filesystem. `custom.allowed_roots` says why in one line -- "a local
    tool that will import any path on the filesystem on request is a nastier
    primitive than it looks" -- and that reasoning is not specific to
    adapters: reading a corpus, a document or a prompt set is the same
    primitive with the result coming back as text instead of code.
    """
    origin = request.headers.get("origin") or ""
    if origin:
        from urllib.parse import urlparse

        host = (urlparse(origin).hostname or "").lower()
        if host not in ("127.0.0.1", "::1", "localhost"):
            return JSONResponse(
                {
                    "error": (
                        f"That request came from another site. {doing} is "
                        f"something only this machine's own page may do."
                    )
                },
                status_code=403,
            )

    if not _is_loopback(request.client.host if request.client else ""):
        # `because` rather than one flattened sentence for everybody: what to
        # do INSTEAD differs by caller, and a refusal that does not say is
        # half a refusal.
        tail = because or (
            "Naming a path is naming a file on the SERVER's disk, and that is "
            "not something a request over the network gets to do — send the "
            "text itself instead, or run ModelMRI where the file lives."
        )
        return JSONResponse(
            {"error": f"{doing} is only possible from this machine. {tail}"},
            status_code=403,
        )
    return None


def _label_names(model) -> list | None:
    """The head's own class names, or `None` when it publishes none.

    `id2label` is a dict keyed by index, and transformers hands it back with
    STRING keys after a JSON round-trip — so the order is restored by sorting
    on the INTEGER, not by trusting insertion order and not by sorting the
    keys as text, which puts "10" immediately after "1". A list built in the
    wrong order puts every name against the wrong class and looks entirely
    reasonable doing it.

    `None` rather than a list of "class 0", "class 1": `vision_attr` drops
    names that do not match the head's width rather than applying them to the
    wrong classes, and invented names would defeat that check.

    Module level rather than a closure so it can be tested directly. The first
    version was a closure, and the test written for it could only re-implement
    the sort — a test that passes whether or not the code is right.
    """
    table = getattr(getattr(model, "config", None), "id2label", None)
    if not isinstance(table, dict) or not table:
        return None
    try:
        # `key=int`, not `key=lambda k: int(k)` — the lambda was a wrapper
        # around a callable that already does exactly that, and it hid the
        # one thing this line is about: the sort is on the INTEGER.
        return [str(table[k]) for k in sorted(table, key=int)]
    except (TypeError, ValueError):
        # Keys that are not indices at all. Unknown, said as unknown.
        return None


def _from_this_machine(request) -> bool:
    """The same question `_not_from_this_machine` answers, as a bool.

    Split out because two callers need the ANSWER rather than the response:
    one to decide whether a local directory may be resolved at all, and the
    refusal builder itself. Answering it twice in two places is how the two
    drift apart.
    """
    return _not_from_this_machine(request, "this") is None


def _scan_summary(
    reports, dangerous, unscanned, *, n_total: int | None = None, readable: bool = True
) -> str:
    """The scanner's own summary sentence, kept at the name callers import.

    The sentence itself moved to `weights_scan.summary` when `cli.py` was
    found carrying a second, staler copy of it. This wrapper stays because
    the route and its tests reach for it here, and because the import is
    deferred: `weights_scan` walks pickle opcodes and is not needed to build
    the app.
    """
    from . import weights_scan

    return weights_scan.summary(
        reports, dangerous, unscanned, n_total=n_total, readable=readable
    )


def create_app(
    trace_db: str | None = None, dataset_repo: str = "lerobot/pusht"
) -> FastAPI:
    app = FastAPI(title="ModelMRI", version=__version__)

    # THE BACKSTOP, BECAUSE HALF THE ROUTES NEVER HAD AN ARM.
    #
    # Counted: 56 routes, 28 of them with no `except` at all. Several of those
    # talk to a network daemon or walk a filesystem and will realistically
    # fail — /api/ollama, /api/models/discovered, /api/vla/datasets,
    # /api/custom/candidates, /api/paths. Starlette answered those with
    # `text/plain` "Internal Server Error", so the reader got a bare string
    # instead of the `{"error": ...}` every other route and the frontend's
    # `explain()` expect, and was never pointed at the terminal.
    #
    # Nothing was leaked and nothing was erased — uvicorn logs the traceback —
    # but a three-answer contract that covers half the surface is not a
    # contract. One handler is the fix; 28 more copies of the three arms is
    # not. Registering it does mean Starlette calls this instead of letting
    # the exception reach uvicorn's own logger, which is why it goes through
    # `_internal` and its `log.exception` rather than returning a bare
    # response.
    #
    # It sits UNDER every per-route arm: those return normally, so nothing
    # propagates this far unless a route has no arm for it. FastAPI's own
    # HTTPException and RequestValidationError handlers are registered
    # separately and still win.
    @app.exception_handler(Exception)
    async def unhandled(request: Request, err: Exception) -> JSONResponse:
        return _internal(err, request.url.path)

    runtime = ModelRuntime()
    app.state.runtime = runtime
    # The store is created here rather than at import: this is the process
    # that downloads, and it is the last moment before anything can.
    paths.ensure_models_home()

    from .image_runtime import ImageHandle

    # One pipeline per process, for the reason `ImageHandle` records:
    # two resident on an 8 GB card is an OOM in the middle of somebody's
    # measurement.
    app.state.image = ImageHandle()
    app.state.vla = VLAHandle()
    # `.mri` files minted by `/v1` for `{"modelmri": {"mri": true}}`, held by
    # id so a client can fetch one it was handed. Bounded and in memory: see
    # `openai_api.MriStore` for why it is neither unbounded nor on disk.
    app.state.mri_store = openai_api_mod.MriStore()
    app.state.vla_reader = None
    app.state.custom = CustomHandle()
    if trace_db:
        db_path = str(trace_db)
    else:
        # Platform data dir, but keep using an existing ~/.modelmri database
        # rather than starting an empty one beside it and losing the history.
        db_path = str(paths.trace_db_path())
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    traces = TraceStore(db_path)
    app.state.traces = traces

    async def _file(rec: _Recording | None, failure: str | None = None) -> None:
        """Store a finished generation, off the event loop. Never raises.

        `to_thread` because the store serialises every caller behind one lock
        and one connection, so a write that happens to queue behind a
        `GET /api/traces` would otherwise stall the socket still feeding
        tokens to the browser.
        """
        if rec is not None:
            await asyncio.to_thread(rec.file, traces, failure)

    # `modelmri open somebody.mri` hands the file over here, so the page is
    # already showing the analysis when the browser tab opens. Failing to
    # read it is not fatal: the server starting with nothing open beats the
    # server not starting.
    #
    # It was also silent, and the premise for that — "the CLI already parsed
    # and reported on it" — holds for `modelmri open` and not for MODELMRI_OPEN
    # set by hand. Then the tab opened empty with no message anywhere. One log
    # line, because a shrug the reader cannot see is indistinguishable from a
    # bug.
    #
    # The four types are the ones measured escaping this call, not a guess:
    # read_bytes gives OSError (moved, unreadable) and ValueError ("embedded
    # null byte" for a malformed variable); session.parse promises SessionError
    # (a ValueError) but a document whose "meta" is a list escapes as
    # TypeError("list object is not a mapping") and a deeply nested one as
    # RecursionError. Those last two are a defect in session.parse's contract —
    # when it holds them, this tuple shrinks to (OSError, ValueError).
    if pending := os.environ.get("MODELMRI_OPEN"):
        try:
            runtime.open_session(Path(pending).read_bytes())
        except (OSError, ValueError, TypeError, RecursionError) as err:
            log.warning(
                "MODELMRI_OPEN could not be opened (%s: %s); starting with no "
                "session open",
                type(err).__name__,
                err,
            )

    # Serve the built React app when present (frontend/ builds into static/app);
    # fall back to the legacy single-file playground otherwise.
    static = files("modelmri") / "static"
    app_index = static / "app" / "index.html"
    if app_index.is_file():
        app.mount("/app", StaticFiles(directory=str(static / "app")), name="app")

    @app.get("/")
    def index() -> HTMLResponse:
        page = app_index if app_index.is_file() else static / "index.html"
        # no-cache: each deploy replaces the hashed asset files, so a cached
        # index.html would reference deleted bundles and half-break the page
        return HTMLResponse(
            page.read_text("utf-8"),
            headers={"Cache-Control": "no-cache, must-revalidate"},
        )

    @app.get("/api/session")
    def session() -> dict:
        """What this process is holding — ALL of it.

        It used to answer with `runtime.status()` alone, which is the text
        model, and the header at the top of the page reads this route. So a
        3.3 GB diffusion pipeline could be resident, with every control in its
        panel live, under a badge reading "no model loaded" — the header and
        the panel answering one question two ways.

        The three handles are separate on purpose (a pipeline is several
        models and the server refuses to hold one beside a text model without
        being asked twice), but "is anything loaded" is one question and this
        is where it is asked.

        Additive: `model` is unchanged, so every existing reader — including
        `RunsOn` in six panels — is untouched.
        """
        image = app.state.image.status()
        vla = app.state.vla.status()
        return {
            "app": "modelmri",
            "version": __version__,
            "model": runtime.status().to_dict(),
            # Trimmed to what a header needs. The full status of each is one
            # request away on its own route, and duplicating it here would be
            # two places to keep in step.
            "image": {
                "loaded": bool(image.loaded),
                "repo": image.repo,
                "device": image.device,
                "family": image.family,
            },
            "vla": {
                "loaded": bool(vla.loaded),
                "repo": vla.repo,
                "device": vla.device,
            },
        }

    @app.post("/api/model/load")
    async def load_model(req: LoadRequest):
        from .capacity import TooBig
        from .runtime import LoadCancelled

        try:
            status = await asyncio.to_thread(
                runtime.load, req.hf_id, req.source, req.confirm, req.device
            )
            return status.to_dict()
        except LoadCancelled as err:
            # Not a failure: the user asked. 200 with a plain answer, so the
            # UI does not paint a red error over something it did on purpose.
            # Stays first: LoadCancelled is a RuntimeError, so it survives
            # today only because nothing broader is written above it, and
            # anyone who ever widens an arm here to RuntimeError turns Stop
            # into a red 409.
            return JSONResponse({"cancelled": True, "message": err.sentence})
        except TooBig as err:
            # capacity.py's own refusal, raised by _preflight before a byte
            # moves. Still a plain ValueError there, and this arm answers the
            # same 422 the pull path does — the two must not drift, which is
            # why capacity.guard is shared between them in the first place.
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/model/load")

    @app.get("/api/devices")
    async def list_devices() -> dict:
        """Every device on this machine, and which one a load goes to by default.

        `/api/session` reports the ONE device in use. This reports the ones
        that exist, which is a different question and the only one that makes
        choosing possible — a second card kept free for something else, a card
        another process is already filling, or a deliberate run on CPU to
        compare against.

        `free_bytes` is `null` where the backend has no way to report it
        (Apple's unified memory, Intel XPU, and system RAM on every platform).
        Null is UNKNOWN; rendering it as 0 would say the machine is out of
        memory when nobody asked.
        """
        from . import devices as devices_mod

        rows = await asyncio.to_thread(devices_mod.available)
        usable = [r for r in rows if r["kind"] != "cpu"]
        default = next((r["id"] for r in rows if r["is_default"]), "cpu")
        return {
            "devices": rows,
            "default": default,
            "means": (
                f"{len(rows)} device(s) here — {len(usable)} accelerator(s) "
                f"and the CPU. A load with no device named goes to {default}, "
                f"which is what has always happened."
            ),
        }

    @app.post("/api/model/unload")
    async def unload_model():
        """Drop the model and hand the memory back.

        There was no way to do this, which mattered most on the machines where
        it mattered most: an 8 GB card holding a 2.5 GB checkpoint has room for
        the next model only if the last one leaves, and the only way to make it
        leave was killing the server.

        Reports what was actually freed rather than what should have been —
        `freed_bytes` is the difference in allocator-reported bytes across the
        call, and an allocator that keeps its arena is a real outcome the
        reader should see.
        """
        try:
            return await asyncio.to_thread(runtime.unload)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except Exception as err:
            return _internal(err, "/api/model/unload")

    @app.post("/api/model/cancel")
    def cancel_load() -> dict:
        """Stop an in-flight load. The reason this exists is in runtime.py."""
        from .progress import TRACKER

        return {"stopping": TRACKER.request_cancel()}

    @app.get("/api/model/progress")
    def load_progress() -> dict:
        """Polled while a load runs — a minutes-long wait needs a heartbeat."""
        from .progress import TRACKER

        return TRACKER.snapshot().to_dict()

    @app.get("/api/pull/progress")
    def pull_progress() -> dict:
        """A download in flight, by name. Separate from the load meter.

        Two slots because the two jobs overlap: the picker starts a pull, and
        the page behind it can load something else while that pull runs. One
        slot meant whichever started last owned the numbers, and the other's
        name was left attached to them.
        """
        from .progress import PULLS

        return PULLS.snapshot().to_dict()

    @app.get("/api/paths")
    def where() -> dict:
        """Every directory this program reads or writes.

        A tool that puts gigabytes on your disk should be able to say where,
        without you reading its source. Nothing here creates a directory —
        asking is not writing.
        """
        return paths.describe()

    @app.get("/api/accelerator")
    def accelerator() -> dict:
        return runtime.accelerator()

    # ---------------- HuggingFace account + model browsing ----------------

    @app.get("/api/hub/auth")
    def hub_auth() -> dict:
        from . import hub

        return hub.whoami().to_dict()

    @app.post("/api/hub/signin")
    def hub_signin(req: HubSignInRequest):
        from . import hub

        try:
            return hub.sign_in(req.token).to_dict()
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/hub/signin")

    @app.post("/api/hub/signout")
    def hub_signout() -> dict:
        from . import hub

        return hub.sign_out().to_dict()

    @app.get("/api/hub/models")
    async def hub_models(q: str = "", limit: int = 24):
        from . import hub

        try:
            if not q.strip():
                return await asyncio.to_thread(hub.suggested)
            return await asyncio.to_thread(hub.search, q, limit)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/hub/models")

    @app.get("/api/ollama/resolve")
    async def ollama_resolve(name: str) -> dict:
        """Look up any Ollama model by name, with its size and a verdict.

        Ollama has no search API, so the picker offers a name box instead of
        a result list — which covers strictly more: namespaced models, any
        tag, and anything published since.
        """
        from . import capacity as _capacity
        from . import ollama as _ollama

        found = await asyncio.to_thread(_ollama.resolve, name)
        if not found["found"]:
            return {**found, "ok": False, "overridable": False, "warning": ""}

        target = _capacity.ollama_models_dir()
        _, measured = _capacity.free_space(target)
        # `free_space` returns 0 for a volume it could not read, which
        # `capacity.guard` understands and correctly skips its refusal on.
        # NO LONGER a conversion. `capacity.free_space` returns `None` for a
        # volume it could not measure, so 0 arriving here means the disk is
        # genuinely full — and `measured or None` turned exactly that into
        # `{"free_bytes": null, "ok": true}`, a green "it fits" on a disk with
        # zero bytes free. The two states are distinguished at the source now.
        free = measured
        try:
            _capacity.guard(
                found["bytes"],
                target,
                label=found["name"],
                vram_gb=runtime.accel.vram_gb,
                accel_name=runtime.accel.name,
            )
        except _capacity.TooBig as err:
            return {
                **found,
                "free_bytes": free,
                "ok": False,
                "overridable": err.overridable,
                # `err.sentence`, not `str(err)`. `TooBig` carries an authored
                # sentence for exactly this slot — see errors.py — and these
                # two were the last places in this file still publishing
                # whatever `str()` returned. CodeQL flagged the same shape
                # elsewhere; these were not flagged and were wrong anyway.
                "warning": err.sentence,
            }
        return {
            **found,
            "free_bytes": free,
            "ok": True,
            "overridable": False,
            "warning": "",
        }

    @app.get("/api/ollama/size")
    async def ollama_size(name: str) -> dict:
        """What a pull would cost, and whether this machine can take it.

        The picker asks before offering the button, so the number is on
        screen before the click rather than discovered halfway through.
        """
        from . import capacity as _capacity
        from . import ollama as _ollama

        need = await asyncio.to_thread(_ollama.manifest_size, name)
        target = _capacity.ollama_models_dir()
        _, measured = _capacity.free_space(target)
        # The SAME conversion the next four lines already do for `free_bytes`,
        # applied to the field the comment was actually written about.
        # `manifest_size` documents its own 0 as "the registry cannot answer —
        # treated as unknown by the guard, never as small", and the guard does
        # honour that. The payload did not: `bytes: 0, ok: true` for a name
        # that does not exist reads as "this model is free and it fits".
        unknown_size = need <= 0
        # `free_space` returns 0 for a volume it could not read, which
        # `capacity.guard` understands and correctly skips its refusal on.
        # NO LONGER a conversion. `capacity.free_space` returns `None` for a
        # volume it could not measure, so 0 arriving here means the disk is
        # genuinely full — and `measured or None` turned exactly that into
        # `{"free_bytes": null, "ok": true}`, a green "it fits" on a disk with
        # zero bytes free. The two states are distinguished at the source now.
        free = measured
        try:
            _capacity.guard(
                need,
                target,
                label=name,
                vram_gb=runtime.accel.vram_gb,
                accel_name=runtime.accel.name,
            )
        except _capacity.TooBig as err:
            return {
                "name": name,
                "bytes": need,
                "free_bytes": free,
                "ok": False,
                "overridable": err.overridable,
                # `err.sentence`, not `str(err)`. `TooBig` carries an authored
                # sentence for exactly this slot — see errors.py — and these
                # two were the last places in this file still publishing
                # whatever `str()` returned. CodeQL flagged the same shape
                # elsewhere; these were not flagged and were wrong anyway.
                "warning": err.sentence,
            }
        return {
            "name": name,
            "bytes": None if unknown_size else need,
            "free_bytes": free,
            # `ok` stays true: nothing here refused it, and turning an unknown
            # size into a refusal would block a legitimate pull on the
            # registry's silence. What changes is that the silence is SAID.
            # The picker already renders `warning` beside the button, so this
            # reaches the person about to click it.
            "ok": True,
            "overridable": False,
            "warning": (
                f"the registry published no size for `{name}`, so whether it "
                f"fits this machine is UNKNOWN rather than yes. Either the "
                f"name is wrong or that model does not publish a manifest — "
                f"nothing here will guess a number for it."
                if unknown_size
                else ""
            ),
        }

    @app.post("/api/ollama/pull")
    async def ollama_pull(req: OllamaPullRequest):
        from . import capacity as _capacity
        from . import ollama as _ollama

        # Enforced here, not in the browser. A check the client performs is a
        # check the client can skip, and this one guards someone's disk.
        need = await asyncio.to_thread(_ollama.manifest_size, req.name)
        try:
            _capacity.guard(
                need,
                _capacity.ollama_models_dir(),
                label=req.name,
                vram_gb=runtime.accel.vram_gb,
                accel_name=runtime.accel.name,
                confirm=req.confirm,
            )
        except _capacity.TooBig as err:
            return JSONResponse(
                {"error": str(err), "overridable": err.overridable}, status_code=422
            )

        def run() -> dict:
            # THE UPDATES USED TO BE THROWN AWAY. This loop existed already
            # and did `last = update` and nothing else, so a nine gigabyte
            # pull sat on "Pulling…" with no bytes, no percentage and no end
            # in sight until it finished -- while the daemon was streaming
            # exact `completed`/`total` counts the whole time. The data was
            # there; nobody published it.
            #
            # It goes into the same tracker the HuggingFace loads use, so
            # there is one meter for both rather than a second one that can
            # disagree with the first.
            from .progress import PULLS

            PULLS.start_external(
                req.name, stage="weights", detail="pulling from the Ollama registry"
            )
            try:
                last = {}
                for update in _ollama.pull(req.name):
                    last = update
                    PULLS.publish(
                        bytes_done=update.get("bytes_done"),
                        bytes_total=update.get("bytes_total"),
                        detail=update.get("status") or None,
                    )
                PULLS.finish()
                return {"pulled": req.name, "last": last}
            except BaseException as err:
                # The class, never the text -- the snapshot is served verbatim
                # by /api/model/progress. Same rule as the load path.
                PULLS.finish(error=_load_failed(err))
                raise

        try:
            return await asyncio.to_thread(run)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/ollama/pull")

    @app.get("/api/models/local")
    def models_local() -> list[dict]:
        from .runtime import local_hf_models

        return local_hf_models()

    @app.get("/api/models/discovered")
    async def models_discovered():
        """Everything already on disk: the HF cache plus any model folder or
        .gguf found under the working directory. Walking a synced drive is
        slow enough to belong off the event loop."""
        from .discover import discover

        return await asyncio.to_thread(discover)

    @app.get("/api/ollama")
    def ollama_status() -> dict:
        from . import ollama as _ollama

        return _ollama.status()

    @app.get("/api/ollama/suggested")
    async def ollama_suggested() -> list[dict]:
        """Somewhere to start on the Ollama tab, sized against this GPU.

        The HuggingFace tab opens on curated picks annotated with whether they
        fit; the Ollama tab opened on an empty box and a blinking cursor. Same
        panel, same job, two different first impressions.

        Names are curated, sizes are not: each is resolved live against the
        registry, so a republished tag cannot leave a stale number on screen.
        """
        from . import ollama as _ollama

        vram = getattr(runtime.accel, "vram_gb", None)
        return await asyncio.to_thread(_ollama.suggested, vram)

    @app.post("/api/model/prompt")
    async def prompt(req: PromptRequest):
        if not runtime.loaded:
            # The sentence ten other sites already use, not the lowercase
            # fragment this had. "no model loaded" is the MACHINE-READABLE
            # status reason — deliberately lowercase, pinned by a test — and
            # putting it in a human-facing refusal slot published a field name
            # as advice. It is also the route the OTHER refusals point at
            # ("Generate something first"), so a reader following one landed
            # on a second refusal that told them nothing and offered no step.
            return JSONResponse(
                {"error": "No model loaded — pick one first."}, status_code=409
            )

        # A generation you asked for in this app is a run, and the agents
        # panel is where runs go. See `_Recording` for why `commit` gates it.
        rec = _Recording(runtime, req.prompt) if req.commit else None

        def run() -> str:
            out = []
            for piece in runtime.generate_stream(
                req.prompt, req.max_new_tokens, req.temperature, req.commit
            ):
                out.append(piece)
                if rec is not None:
                    rec.piece(piece)
            return "".join(out)

        try:
            generation = await asyncio.to_thread(run)
        except Refusal as err:
            # Ollama quitting mid-session, Ollama refusing the prompt, no
            # model loaded. `runtime.generate_stream` translates ollama.py's
            # plain RuntimeErrors into Refusals at the call, which is what
            # lets this handler tell them apart from the next arm.
            await _file(rec, str(err))
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            await _file(rec, str(err))
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            # CUDA out of memory, a streamer timeout, an architecture
            # transformers cannot run eagerly. These were 409s carrying
            # "{type}: {err}" — a full GPU reported as a conflict, in torch's
            # words, which can name paths on this machine.
            #
            # The trace gets the sentence the reader gets, for the same
            # reason: torch's text can name paths on this machine, and a
            # trace is a document people export and attach to issues.
            await _file(rec, _INTERNAL)
            return _internal(err, "/api/model/prompt")
        await _file(rec)
        return {"generation": generation}

    @app.get("/api/attention/meta")
    def attention_meta() -> dict:
        return runtime.attention_meta()

    @app.get("/api/sae")
    def sae_status() -> dict:
        return asdict(runtime.sae_status())

    def _sae_for_current() -> tuple[str, str]:
        """The release registered for whatever model is loaded.

        Raises rather than falling back to a default: "this model has no SAE
        anyone has published" is a real and common answer -- SAEs are trained
        per model and public ones exist for a handful -- and a fallback would
        turn it into a confusing dimension mismatch three calls later.
        """
        from . import sae_registry

        current = runtime.hf_id if runtime.backend == "hf" else None
        usable = [e for e in sae_registry.for_model(current) if e["supported"]]
        if not usable:
            listed = sae_registry.for_model(current)
            extra = (
                f" {listed[0]['repo']} is registered for it but this build "
                f"cannot open it: {listed[0]['note']}"
                if listed
                else ""
            )
            raise Refusal(
                f"No sparse autoencoder is registered for "
                f"{current or 'the loaded model'}. They are trained per model "
                f"and public ones exist for only a handful.{extra} The logit "
                f"lens works on every model and needs nothing extra."
            )
        return usable[0]["repo"], usable[0]["default_hook"]

    @app.post("/api/sae/load")
    async def sae_load(req: SAELoadRequest):
        try:
            repo, hook = req.repo, req.hook
            if not repo or not hook:
                chosen_repo, chosen_hook = _sae_for_current()
                repo = repo or chosen_repo
                hook = hook or chosen_hook
            status = await asyncio.to_thread(
                runtime.load_sae,
                repo,
                hook,
                width=req.width,
                average_l0=req.average_l0,
            )
            return asdict(status)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/sae/load")

    @app.get("/api/features/summary")
    async def features_summary(top_k: int = 8):
        try:
            return await asyncio.to_thread(runtime.features_summary, top_k)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/features/summary")

    # BEFORE `/api/features/{feature_id}`, AND THAT IS LOAD-BEARING.
    #
    # Starlette matches routes in registration order and `{feature_id}` has no
    # converter in the path, so its regex accepts any single segment. Declared
    # after it, this route is unreachable: `/api/features/ablate` matches the
    # detail route, FastAPI tries to parse "ablate" as an int, and the endpoint
    # answers 422 "Input should be a valid integer" for a request that was
    # perfectly well formed. Grouping it with the other ablate routes further
    # down reads better and does not work. The two refusal tests in
    # test_smoke.py pin this by asserting 409, which a shadowed route cannot
    # return.
    @app.get("/api/features/ablate")
    async def ablate_features(
        position: int | None = None, scope: str = "position", top_k: int = 64
    ):
        """Rank SAE features by how far removing one moves the next-token answer.

        The features panel ranks by activation, which is what fired rather than
        what mattered. This is the same question `/api/attention/ablate` asks of
        heads and `/api/attention/attribute` asks of the prompt's tokens, and
        the answers differ: measured at blocks.8.hook_resid_pre with
        "The Eiffel Tower is located in the city of", the top-8 by activation
        and the top-8 by ablation KL at the attributed token share 6 of 8, and
        at `scope=prompt` they share 0 of 8.

        `scope=position` (default) tries the features firing at `position` —
        43 candidates on that prompt, 92 passes. `scope=prompt` tries each
        feature wherever it fires at or before it — 494 candidates, capped at
        256 tested, 518 passes — and it is the one that finds features the
        panel cannot show: measured here its top-8 is [5856, 11149, 19941,
        1066, 2194, 7703, 20110, 2319], and 19941, 1066, 7703, 20110 and 2319
        fire only at tokens 1, 4, {2,3}, 3 and 2.

        Two passes per tested feature, not one: every row is paired with a
        random direction of the same norm at the same tokens (`control_kl`),
        because a score is partly the size of the edit that produced it.

        Prompt scope tests several times as many candidates as position scope
        and takes several times as long for it.

        `top_k` trims the returned rows only. A row it drops was tested and
        scored; `truncated: true` means something else and worse — a candidate
        that was never measured. `sum_of_singles` and `joint_kl` cover every
        scored row either way.

        `passes` and `elapsed_s` are both returned so a caller can derive a
        rate on ITS machine, the same contract the other two ranking routes
        carry. Seconds measured here do not transfer.

        Read `residual_means` before quoting a score. The baseline follows the
        scope, because the edits do: substituting the SAE's reconstruction with
        no feature removed costs 0.0775 nats at the attributed token alone and
        0.2212 over the eleven tokens a prompt-scope ranking edits. Only 2 of
        43 scores clear the first and 1 of 256 clears the second.

        Read `removal_check` too. The edit lands exactly — the stream the model
        received differs from the intended one by 0.0 in float32 — but that is
        not the same as the feature having left the SAE's reading of the
        stream, which is per row in `encoder_residual` and fails on 38 of 43
        rows here because encoder and decoder directions are not dual.

        **float32 only, and on a GPU that means it refuses.** ModelMRI loads
        bfloat16 on an NVIDIA GPU, and in bfloat16 an edit with no causal
        effect scores 0.028 nats here while the noise floor still reads 0.0 —
        the ranking below its top rows is rounding error. `runtime.rank_features`
        carries the measurement and the refusal says how to get float32.

        409 when there is nothing to measure or nothing that would mean
        anything: a recording is open, the model is served by Ollama, nothing
        has been generated, the generation belongs to a previous model, no SAE
        is loaded, the model is not float32, or the SAE does not reconstruct
        the stream it is attached to. 422 when `position`, `scope` or `top_k`
        is outside what this can answer.
        """
        try:
            return await asyncio.to_thread(
                runtime.rank_features, position, scope, top_k
            )
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            # A full GPU is not a conflict. `rank_features` translates only
            # FeatureAblationError, so a torch OOM during 262 forward passes —
            # the realistic failure on this route — arrives here and is logged
            # rather than published in torch's own words.
            return _internal(err, "/api/features/ablate")

    @app.get("/api/features/{feature_id}")
    async def feature_detail(feature_id: int):
        try:
            return await asyncio.to_thread(runtime.feature_detail, feature_id)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/features/{feature_id}")

    @app.post("/api/steer")
    def steer(req: SteerRequest):
        try:
            return runtime.set_steering(req.feature_id, req.scale)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/steer")

    @app.get("/api/steer")
    def steer_status() -> dict:
        return runtime.steering_status()

    # ---------------- VLA (robot policy) ----------------

    def _reader():
        """Lazily open the cached LeRobot dataset (kept on app.state)."""
        from .vla_data import LeRobotV3Reader

        if getattr(app.state, "vla_reader", None) is None:
            chosen = getattr(app.state, "vla_dataset", dataset_repo)
            app.state.vla_reader = LeRobotV3Reader.discover(repo_id=chosen)
        return app.state.vla_reader

    # ------------------------------------------------------ SAEs and the lens

    @app.get("/api/sae/available")
    def sae_available() -> dict:
        """Which SAEs fit the model that is loaded, and what else exists.

        An empty `matching` is the common, honest answer: sparse autoencoders
        are trained per model, and public ones exist for only a handful. The
        panel says so, and names the ones it knows, rather than looking broken.
        """
        from . import sae_registry

        current = runtime.hf_id if runtime.backend == "hf" else None
        matching = sae_registry.for_model(current)
        return {
            "model": current,
            "matching": matching,
            "usable": [m for m in matching if m["supported"]],
            "catalogue": sae_registry.catalogue(),
        }

    @app.get("/api/lens")
    async def lens(top_k: int = 5, kind: str = "plain"):
        """Logit lens — what the model would have said at each layer.

        The fallback for every model with no SAE, which is most of them.

        `kind=plain|tuned|both`. `layers` is ALWAYS the plain reading, on
        every kind; a tuned one arrives beside it in `tuned` rather than in
        its place. A translator fitted to minimise disagreement with the final
        distribution will reduce disagreement with the final distribution, so
        a caller that got translated rows where it expected plain ones would
        have no way to tell the model from the fit.
        """
        # The replay guard, the Ollama arm and the "generate something first"
        # arm all live in `ModelRuntime.logit_lens` now, beside every other
        # measurement's. They were duplicated here because the lens was
        # computed outside the runtime, and the comment this replaces recorded
        # how that failed: with a recording open AND a model loaded, the lens
        # reported the LIVE model's layers inside a session every other panel
        # was drawing from the file. One guard, one place.
        try:
            return await asyncio.to_thread(runtime.logit_lens, top_k, kind)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/lens")

    @app.post("/api/features/evidence")
    async def feature_evidence(request: Request):
        """What a feature fires on in YOUR corpus, and what it promotes.

        NOTHING IS DOWNLOADED — the corpus is a local file or a list of
        strings you supply. The corpus name, its token count and the fraction
        of features that never fired travel with every number, because a top
        activation is a top activation IN THIS TEXT.
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "this request body is not JSON"}, 422)
        if (not_an_object := _body_object(body)) is not None:
            return not_an_object

        from . import feature_corpus as fc

        texts = body.get("texts")
        label = str(body.get("label") or "")
        try:
            # A path in the body names a file on the SERVER's disk. See
            # `_not_from_this_machine`: loopback alone does not settle it.
            if body.get("file"):
                refusal = _not_from_this_machine(
                    request, "Reading a corpus off this machine's disk"
                )
                if refusal is not None:
                    return refusal
            if body.get("file"):
                texts, label = fc.load_corpus(str(body["file"]))
            if not isinstance(texts, list) or not texts:
                return JSONResponse(
                    {
                        "error": "a feature sweep needs text. Pass `texts` (a "
                        "list of strings) or `file` (a .txt or .jsonl). "
                        "Nothing is downloaded."
                    },
                    status_code=422,
                )
            # Through the same guard as every other field. This one read
            # `int(feature)` off a LOCAL rather than off `body`, so the sweep
            # that fixed the others went straight past it and
            # `{"feature": "x"}` still answered 500.
            feature_id = (
                _whole(body, "feature", -1) if body.get("feature") is not None else None
            )
            return await asyncio.to_thread(
                runtime.feature_evidence,
                [str(t) for t in texts],
                feature_id=feature_id,
                corpus_label=label,
                top_k=_whole(body, "top_k", 10),
            )
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/features/evidence")

    @app.post("/api/diff/models")
    async def diff_models(request: Request):
        """What a finetune changed, over a PROMPT SET rather than one prompt.

        Loads each side ONCE, in sequence — 8 GB will not hold both, and the
        models worth comparing are exactly the ones near that limit. Every
        number comes back as a median over the prompt set with its middle
        half, because a diff measured on one prompt is a sample.

        Refuses a pair whose per-layer table would line up the wrong things —
        different layer counts, hidden sizes or vocabularies — and names both
        sides when it does.
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "this request body is not JSON"}, 422)
        if (not_an_object := _body_object(body)) is not None:
            return not_an_object

        prompts = body.get("prompts")
        # A path in the body names a file on the SERVER's disk. See
        # `_not_from_this_machine`: loopback alone does not settle it.
        if body.get("file"):
            refusal = _not_from_this_machine(
                request, "Reading a prompt set off this machine's disk"
            )
            if refusal is not None:
                return refusal
        if body.get("file"):
            from . import feature_corpus as fc

            try:
                prompts, _ = fc.load_corpus(str(body["file"]))
            except BadRequest as err:
                return JSONResponse({"error": err.sentence}, status_code=422)
        if not isinstance(prompts, list):
            return JSONResponse(
                {
                    "error": "this needs a prompt SET. Pass `prompts` (a list "
                    "of strings) or `file` (a local .txt or .jsonl) — the "
                    "whole output is a spread across them, and one prompt "
                    "cannot produce one."
                },
                status_code=422,
            )

        # NAMED BEFORE THEY ARE COMPARED. Both sides went through
        # `str(body.get("a") or "")`, so an absent side became `""` and
        # `model_diff` then found `"" == ""` and refused with "both sides are
        # the same model, so every difference would be zero by construction" —
        # for a request that named no model at all. Measured: a perfectly good
        # six-prompt body with no `a` and no `b` got that sentence, and naming
        # only ONE side reached `transformers` and came back a 500.
        sides = {name: str(body.get(name) or "").strip() for name in ("a", "b")}
        if missing := [name for name, value in sides.items() if not value]:
            return JSONResponse(
                {
                    "error": (
                        f"this compares two models and "
                        f"{' and '.join(f'`{m}`' for m in missing)} "
                        f"{'was' if len(missing) == 1 else 'were'} not given. "
                        f"Pass both as HuggingFace ids or local paths."
                    )
                },
                status_code=422,
            )

        try:
            # Through the runtime rather than calling `model_diff.compare`
            # directly: that is what keeps the result available to
            # `export_session`, and it is where the accelerator settings and
            # the receipt already live.
            return await asyncio.to_thread(
                runtime.diff_models,
                sides["a"],
                sides["b"],
                [str(p) for p in prompts],
                # OFF by default. The head half costs n_layers x n_heads
                # forward passes per prompt PER SIDE -- 5,412 on a 1.7B with
                # six prompts -- which is two
                # orders of magnitude more than everything else in this
                # comparison, so it is opted into rather than out of.
                include_heads=bool(body.get("include_heads")),
                # Far cheaper than the head half — 248 passes for a 24-token
                # prompt over four — but still opted into, because a
                # 500-token prompt is not.
                include_tokens=bool(body.get("include_tokens")),
            )
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/diff/models")

    @app.post("/api/custom/ablate")
    async def custom_ablate(request: Request):
        """What matters in the network YOU trained.

        The custom panel maps one forward pass and says nothing causal. This
        replaces each module output with its mean over your own samples, or
        occludes each input region, and reports how far the answer moved —
        every strong site against a null of the same shape.

        Refuses rather than defaults: an adapter that does not declare TASK
        gets no metric picked for it, and one with no sample_inputs() gets no
        mean invented for it.
        """
        try:
            body = await request.json()
        except Exception:
            # A 422 NAMING THE BODY, the way eight other routes in this file
            # already answer the same bytes. `body = {}` swallowed it, so the
            # request continued on defaults and the reader got whatever
            # sentence the NEXT check produced: `POST /api/custom/ablate` with
            # non-JSON answered "no custom model is loaded" — true, and about
            # something else entirely — and `POST /api/vla/sweep` answered 200
            # after running a full default sweep off a body nobody had read.
            #
            # The worse-formed body already got the better answer: `[1,2,3]`
            # parses, so it reached `_body_object` below and was refused
            # properly. Only bytes that are not JSON at all were let through.
            return JSONResponse({"error": "this request body is not JSON"}, 422)
        if (not_an_object := _body_object(body)) is not None:
            return not_an_object
        try:
            return await asyncio.to_thread(
                app.state.custom.ablate,
                str(body.get("kind") or "layers"),
                grid=_whole(body, "grid", 0),
            )
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/custom/ablate")

    @app.post("/api/ground")
    async def ground_answer(request: Request):
        """Did the answer come from the document, or from the weights?

        The question every local RAG interface leaves unanswered. NOTHING IS
        DOWNLOADED and nothing is indexed: the document is text the caller
        hands over or a local file, and every passage goes into the prompt.

        Two numbers per passage, side by side, because the interesting case is
        the one where they disagree — attention on a passage with no causal
        dependence on it is what an answer coming from the weights looks like
        from outside.
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "this request body is not JSON"}, 422)
        if (not_an_object := _body_object(body)) is not None:
            return not_an_object

        document = body.get("document")
        # A path in the body names a file on the SERVER's disk. See
        # `_not_from_this_machine`: loopback alone does not settle it.
        if body.get("file"):
            refusal = _not_from_this_machine(
                request, "Reading a document off this machine's disk"
            )
            if refusal is not None:
                return refusal
        if body.get("file"):
            from . import feature_corpus as fc

            try:
                texts, _ = fc.load_corpus(str(body["file"]))
            except BadRequest as err:
                return JSONResponse({"error": err.sentence}, status_code=422)
            # Blank line between lines, because that is exactly what
            # `ground.split` treats as a passage boundary — a corpus file's
            # one-per-line convention would otherwise arrive as a single
            # undifferentiated passage.
            document = "\n\n".join(texts)
        if not isinstance(document, str) or not document.strip():
            return JSONResponse(
                {
                    "error": "grounding needs a document. Pass `document` (the "
                    "text) or `file` (a local .txt or .jsonl). Nothing is "
                    "downloaded and nothing is indexed."
                },
                status_code=422,
            )

        try:
            return await asyncio.to_thread(
                runtime.ground_answer,
                document,
                str(body.get("question") or ""),
                max_chunks=_whole(body, "max_chunks", 0),
            )
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/ground")

    @app.post("/api/patchscope")
    async def patchscope(request: Request):
        """Ask the model to describe a hidden state, with two controls.

        Its own surface, not a column beside the logit lens — the lens reports
        tokens read through the unembedding, this reports a SENTENCE the model
        produced, and side by side they would read as two measurements of one
        thing.

        The target prompt is returned with every response because it is part
        of the result: two decodes taken under different targets are not
        comparable, and the method's known failure is a target that describes
        anything fluently.
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "this request body is not JSON"}, 422)
        if (not_an_object := _body_object(body)) is not None:
            return not_an_object

        try:
            return await asyncio.to_thread(
                runtime.patchscope,
                str(body.get("prompt") or ""),
                source_layer=_whole(body, "layer", 0),
                source_position=_whole(body, "position", -1),
                target_prompt=str(body.get("target") or ""),
                target_layer=(
                    int(body["target_layer"])
                    if body.get("target_layer") is not None
                    else None
                ),
                max_new_tokens=_whole(body, "max_new_tokens", 12),
            )
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/patchscope")

    @app.post("/api/patch/path")
    async def path_trace(request: Request):
        """Which component wrote what makes a patching site matter.

        The node grid says WHERE; this says WHAT PUT IT THERE. Same recovery
        fraction, same eight same-norm control draws, same shifted-position
        control — so an edge number and a node number can be read together.
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "this request body is not JSON"}, 422)
        if (not_an_object := _body_object(body)) is not None:
            return not_an_object

        # A DEFAULT THAT IS ALWAYS REFUSED IS NOT A DEFAULT. Both of these fell
        # back to -1, and `patch.path_trace` rejects -1 for each — so calling
        # this route without naming a receiver produced "layer -1 is outside
        # this model's 24", which reads as though the caller sent -1. They did
        # not; they left it out. Two different mistakes deserve two different
        # sentences, and only one of them is about a range.
        #
        # There is no honest default here. A path trace asks what wrote into
        # ONE receiver, and picking that receiver for somebody would be this
        # tool choosing which cell of the patching grid they meant.
        missing = [f for f in ("layer", "position") if body.get(f) is None]
        if missing:
            return JSONResponse(
                {
                    "error": (
                        f"a path trace needs a receiver to trace INTO, and "
                        f"{' and '.join('`' + f + '`' for f in missing)} "
                        f"{'were' if len(missing) > 1 else 'was'} not sent. "
                        f"Pick a bright cell from the patching grid: its layer "
                        f"is the receiver layer and its token is the receiver "
                        f"position."
                    )
                },
                status_code=422,
            )

        try:
            return await asyncio.to_thread(
                runtime.path_trace,
                str(body.get("clean") or ""),
                str(body.get("corrupt") or ""),
                layer=_whole(body, "layer", -1),
                position=_whole(body, "position", -1),
            )
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/patch/path")

    @app.post("/api/probe")
    async def probe_layers(request: Request):
        """Fit a linear probe at every layer, with its null and majority line.

        A layer inside the permutation null is reported as inside it. That is
        the feature: "we have probes" is not worth shipping, "we show you when
        your curve is inside the null" is.
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "this request body is not JSON"}, 422)
        if (not_an_object := _body_object(body)) is not None:
            return not_an_object

        try:
            return await asyncio.to_thread(
                runtime.probe_layers,
                body.get("examples") or [],
                n_permutations=_whole(body, "n_permutations", 0),
                save_as=str(body.get("save_as") or ""),
            )
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/probe")

    @app.get("/api/attention/types")
    async def head_types(seq_len: int = 24, n_sequences: int = 6, seed: int = 0):
        """Behavioural labels for every head, each gated on a measured null.

        A label here must NEVER be read as explaining the ablation ranking:
        the ranking measures what breaks when a head is removed, this measures
        a positional habit on random repeated tokens, and a head can be
        labelled and irrelevant or unlabelled and load-bearing.
        """
        try:
            return await asyncio.to_thread(
                runtime.head_types, seq_len, n_sequences, seed
            )
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/attention/types")

    @app.get("/api/attention/ov")
    async def head_ov(layer: int = 0, head: int = 0, token: str = "", top_k: int = 10):
        """What one head writes into the stream when it attends to one token.

        THE ONLY MEASUREMENT HERE THAT NEEDS NO PROMPT. Every other attention
        route answers a question about the current generation and gives a
        different answer for the next one; this is a product of weights, so it
        is the same every time and it is about the head rather than about the
        run. That is the whole reason it sits beside the ranking instead of
        inside it.
        """
        if not token.strip():
            return JSONResponse(
                {
                    "error": (
                        "Name a token for the head to read. This answers what "
                        "head H writes when it attends to a particular token, "
                        "so there is no answer without one — try a word the "
                        "prompt you are looking at contains."
                    )
                },
                status_code=422,
            )
        try:
            return await asyncio.to_thread(
                runtime.head_ov_vocabulary, layer, head, token, top_k
            )
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/attention/ov")

    @app.get("/api/attention/ov/spectrum")
    async def head_ov_spectrum(
        layer: int = 0, head: int = 0, n_samples: int = 0, seed: int = 0
    ):
        """The eigenvalue readout of one head's OV circuit, over a named sample.

        `n_samples=0` asks for the module's own default rather than a second
        one written here that could drift from it; whatever is used comes back
        in `n_sampled`.

        Read the sentence, not the fraction: the full circuit is
        vocabulary-by-vocabulary and cannot be formed, so this is measured over
        a SAMPLE and carries its size, its seed and how much of the spectrum
        sits off the real line.
        """
        try:
            return await asyncio.to_thread(
                runtime.head_ov_spectrum, layer, head, n_samples, seed
            )
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/attention/ov/spectrum")

    @app.get("/api/attention/direct")
    async def direct_attribution(position: int | None = None, top_k: int = 40):
        """Direct logit attribution, beside the ablation ranking.

        Sited here rather than in its own panel because the two answer related
        questions and disagree, and a reader needs to see that: the ranking
        says what breaks when a head is removed, this says what a head put
        into the logit directly, and a head can be near zero here and still
        carry the answer through a later head.
        """
        try:
            return await asyncio.to_thread(runtime.direct_attribution, position, top_k)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/attention/direct")

    @app.get("/api/lens/tuned")
    def tuned_lens_status() -> dict:
        """Whether a translator has been fitted for the loaded model."""
        return runtime.tuned_lens_status()

    @app.post("/api/lens/tune")
    async def train_tuned_lens(request: Request):
        """Fit a per-layer translator on text you supply.

        NOTHING IS FETCHED. Pretrained lenses exist on the Hub and pulling one
        would break the offline promise the rest of this package keeps, so the
        corpus comes from the caller: a list of strings, a local file, or the
        prompts already in this machine's trace store.
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "this request body is not JSON"}, 422)
        if (not_an_object := _body_object(body)) is not None:
            return not_an_object

        texts = body.get("texts")
        # A path in the body names a file on the SERVER's disk. See
        # `_not_from_this_machine`: loopback alone does not settle it.
        if body.get("file"):
            refusal = _not_from_this_machine(
                request, "Reading a corpus off this machine's disk"
            )
            if refusal is not None:
                return refusal
        source = body.get("file")
        label = str(body.get("label") or "")
        # Read INSIDE a try that converts a refusal. The first pass put
        # these one line above the handler's own try, so the authored
        # sentence was raised and then swallowed by the generic 500 —
        # a guard that is unreachable is not a guard.
        try:
            steps = _whole(body, "steps", 250)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)

        try:
            if source:
                from . import sweep as sweep_mod

                # The same reader `modelmri sweep` uses, so a corpus file that
                # works for one works for the other rather than being two
                # nearly-identical formats.
                texts = sweep_mod.load_prompts(source)
                label = label or Path(str(source)).name
            if not isinstance(texts, list) or not texts:
                return JSONResponse(
                    {
                        "error": "a tuned lens needs text to fit to. Pass "
                        "`texts` (a list of strings) or `file` (a .txt or "
                        ".jsonl). Nothing is downloaded."
                    },
                    status_code=422,
                )
            out = await asyncio.to_thread(
                runtime.train_tuned_lens,
                [str(t) for t in texts],
                corpus_label=label,
                steps=steps,
            )
            return out
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/lens/tune")

    # ---------------------------------------------------------- custom models
    #
    # A model you trained yourself. Every route here is inert until called:
    # candidates are found by reading text, never by importing, and loading
    # happens only for a path you named. See modelmri/custom.py.

    @app.get("/api/custom")
    def custom_status() -> dict:
        from . import custom as custom_mod

        return {
            **app.state.custom.status().to_dict(),
            "roots": [str(r) for r in custom_mod.allowed_roots()],
        }

    @app.get("/api/custom/candidates")
    async def custom_candidates() -> dict:
        from . import custom as custom_mod

        adapters, scripts = await asyncio.to_thread(
            lambda: (custom_mod.find_adapters(), custom_mod.find_torchscript())
        )
        # The caps, REPORTED. Both walks stop at 40, and a panel whose whole
        # job is "here is what is on your disk" was showing 40 of 45 with
        # nothing to say five were missing.
        n_adapters = getattr(adapters, "n_total", len(adapters))
        n_scripts = getattr(scripts, "n_total", len(scripts))
        return {
            "adapters": adapters,
            "torchscript": scripts,
            "roots": [str(r) for r in custom_mod.allowed_roots()],
            "n_adapters_found": n_adapters,
            "n_torchscript_found": n_scripts,
            "truncated": n_adapters > len(adapters) or n_scripts > len(scripts),
        }

    @app.post("/api/custom/load")
    async def custom_load(req: CustomLoadRequest):
        try:
            status = await asyncio.to_thread(app.state.custom.load, req.path)
        except AdapterError as err:
            # THE ONE EXEMPTION FROM THE GENERIC 500, AND IT IS EARNED WHERE
            # IT IS TRUE RATHER THAN CLAIMED FOR A WHOLE HANDLER.
            #
            # `custom.load` imports and runs a Python file the user wrote and
            # pointed at. Their adapter raising ModuleNotFoundError is a fact
            # about their file, and answering "check the terminal" would hide
            # the one line that fixes it while blaming us for their import.
            # So that text is published — but it is published because
            # custom.py caught it AT THE CALL INTO THEIR CODE and put it in an
            # AdapterError: `_import_adapter` around `exec_module`,
            # `load_from_adapter` around `load()` and `example_input()`,
            # `inspect` around the forward pass. Each of those knows the code
            # that raised was theirs.
            #
            # There used to be an `except Exception` below this arm doing the
            # same echo for anything at all, and it could not know that.
            # Measured: an OSError from custom.py's own path resolution came
            # back as 422 `"OSError: [Errno 13] Permission denied: '<abs
            # path>'"` with no log line — ModelMRI's failure, in ModelMRI's
            # words, filed as the user's malformed request. Those reach
            # `_internal` now.
            #
            # Pinned by
            # test_smoke.py::test_custom_load_never_500s_on_a_users_broken_adapter,
            # which asserts "ModuleNotFoundError" reaches the browser at 422 —
            # still true, through AdapterError.
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/custom/load")
        return status.to_dict()

    @app.post("/api/custom/run")
    async def custom_run(req: CustomRunRequest):
        try:
            return await asyncio.to_thread(app.state.custom.run, req.shape, req.seed)
        except AdapterError as err:
            # Their forward pass raised, and `custom.inspect` says so by
            # catching around the model call itself. See the note on
            # custom_load for why the echo lives there and not here: `run`
            # also allocates the example tensor and installs the hooks, and
            # both of those are ModelMRI's — a CUDA OOM while building
            # `torch.randn(*shape)` is not a fact about the user's file.
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/custom/run")

    @app.post("/api/custom/scan")
    async def custom_scan(req: CustomRootRequest, request: Request):
        """Also look in this folder, then scan.

        The scan was limited to the directory the server was launched in,
        which is the wrong question to ask somebody whose model lives on
        another drive: their answer is "it is over there" and the tool's was
        "restart me somewhere else".

        The folder joins the allowed roots for this run only. It does not
        bypass them — `custom._resolve` still refuses anything outside the
        list — so the boundary moves once, deliberately, when a person asks.
        """
        # LOOPBACK ONLY, and this is the reason the whole thing is not a
        # security hole. Widening the roots turns "import from the directory I
        # was started in" into "import from a directory somebody named", and
        # the adapter loader IMPORTS what it loads — so on a server bound to
        # 0.0.0.0 this would be remote code execution with two requests. The
        # person at the keyboard may move the boundary; a stranger on the
        # network may not.
        # And a page in YOUR browser is not you either. Loopback alone does not
        # settle it: a tab open on some other site can POST to localhost, and
        # the request arrives from 127.0.0.1 like any other. A JSON body
        # already forces a preflight that fails without CORS headers — there
        # are none — but relying on that is relying on a side effect. An Origin
        # from anywhere else is refused explicitly, which is a defence you can
        # read rather than one you have to derive.
        refusal = _not_from_this_machine(
            request,
            "Choosing a folder to scan",
            "ModelMRI imports what it loads, so widening where it may import "
            "from is not something a request over the network gets to do — "
            "start the server where your model lives, or set "
            "MODELMRI_MODELS_DIR.",
        )
        if refusal is not None:
            return refusal
        try:
            root = custom.add_root(req.path)
            adapters, scripts = await asyncio.to_thread(
                lambda: (custom.find_adapters(), custom.find_torchscript())
            )
            # THE SAME FIELDS `/api/custom/candidates` returns, because this
            # is the same walk with the same cap. The panel renders whichever
            # of the two answered last, so a shape that differs between them
            # loses the truncation notice depending on which button was
            # pressed.
            n_adapters = getattr(adapters, "n_total", len(adapters))
            n_scripts = getattr(scripts, "n_total", len(scripts))
            return {
                "added": str(root),
                "adapters": adapters,
                "torchscript": scripts,
                "roots": [str(r) for r in custom.allowed_roots()],
                "n_adapters_found": n_adapters,
                "n_torchscript_found": n_scripts,
                "truncated": n_adapters > len(adapters) or n_scripts > len(scripts),
            }
        except AdapterError as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/custom/scan")

    @app.post("/api/custom/unload")
    def custom_unload() -> dict:
        return app.state.custom.unload().to_dict()

    # ---------------- what the weights themselves look like ----------------
    #
    # Netron and TensorBoard's Debugger V2 both do a version of this and
    # neither does it on a model that is LIVE: Netron reads a file, the
    # Debugger needs a TensorFlow run instrumented in advance. These read the
    # module already sitting in this process's memory.

    # ---------------- image models (ROADMAP v1.0, Theme A) ----------------
    #
    # Every route needs a pipeline held by `app.state.image`, so every one can
    # refuse with the same sentence when nothing is loaded. What each of them
    # is ALLOWED to answer comes from `imaging.detect` via the handle's
    # `capabilities` — a panel asks rather than infers, and a family this does
    # not know offers an empty list rather than everything.

    @app.get("/api/image")
    def image_status() -> dict:
        """What is held, or why nothing is. Never raises."""
        return app.state.image.status().to_dict()

    @app.get("/api/image/available")
    async def image_available() -> dict:
        """Every image model already on this disk. Downloads nothing.

        The same rule the model picker follows: say what is here before asking
        anybody to type a name.
        """
        from . import imaging

        found = await asyncio.to_thread(imaging.scan_cache)
        # `scan_cache` stops at its own limit, so a flat count is a claim that
        # this is everything. MEASURED against a cache of 205 checkpoints: 200
        # rows and "200 image model(s) cached on this machine", with five of
        # the reader's own models absent and nothing saying so.
        capped = len(found) >= imaging.SCAN_CACHE_LIMIT
        tail = (
            f" That is as many as this walk reads in one pass "
            f"({imaging.SCAN_CACHE_LIMIT}), so there may be more here that "
            f"are not listed — type a name to load one it did not reach."
            if capped
            else ""
        )
        return {
            "models": [m.to_dict() for m in found],
            "known": sum(1 for m in found if m.known),
            "truncated": capped,
            "scan_limit": imaging.SCAN_CACHE_LIMIT,
            "means": (
                f"{len(found)} image model(s) cached on this machine, "
                f"{sum(1 for m in found if m.known)} of which this can open. "
                f"Nothing was downloaded to answer this.{tail}"
            ),
        }

    @app.get("/api/image/tasks")
    def image_tasks() -> dict:
        """Which kinds of image model can be searched for, and what each offers.

        A picker asks this rather than hardcoding a list, so a task added to
        `image_catalog.TASKS` appears without a second edit here.
        """
        from . import image_catalog

        return {
            "tasks": image_catalog.tasks(),
            "default": image_catalog.DEFAULT_TASK,
            "means": (
                "Each of these is a task the Hub publishes and this tool can "
                "open. A task says what a model DOES — the architecture, and "
                "so which panels apply, is read from the checkpoint's own "
                "config when it loads."
            ),
        }

    @app.get("/api/image/search")
    async def image_search(q: str = "", task: str = "", limit: int = 24) -> dict:
        """Image models on the Hub, with what they weigh and whether they are here.

        The image counterpart to `/api/hub/models`. Downloads nothing: it reads
        a listing, and the only local thing it touches is the cache index, to
        mark the rows that are already on this disk.
        """
        from . import image_catalog

        try:
            rows = await asyncio.to_thread(image_catalog.search, q, task, limit)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=503)

        sized = [r for r in rows if r["size_bytes"]]
        here = sum(1 for r in rows if r["cached"])
        half = sum(1 for r in rows if r.get("partial"))
        # Every cap that shaped this list is named. A truncation nobody is
        # told about reads as "this is all there is".
        cut = []
        # `!=`, not `>`. A limit is clamped in BOTH directions — `?limit=-1`
        # was honestly recorded as `asked=-1, used=1`, and then went unsaid,
        # because `-1 > 1` is False. A clamp the reader is not told about
        # reads as "this is all there is" whichever way it moved.
        if rows.limit_asked != rows.limit_used:
            cut.append(
                f"{rows.limit_asked} were asked for and this list was built "
                f"from {rows.limit_used} — the Hub is queried for at most "
                f"{image_catalog.MAX_RESULTS} and never for fewer than one"
            )
        if rows.cache_capped:
            cut.append(
                f"this machine's cache was walked to its {image_catalog.imaging.SCAN_CACHE_LIMIT}"
                f"-entry limit, so a model past that point shows as not here"
            )
        # A task whose Hub call failed during an all-tasks search is a cap
        # like any other: nine of ten searched is a different answer from ten,
        # and a thin result then reads as "there is not much" rather than "one
        # source did not answer".
        if (
            rows.tasks_searched is not None
            and rows.tasks_total is not None
            and rows.tasks_searched < rows.tasks_total
        ):
            cut.append(
                f"{rows.tasks_total - rows.tasks_searched} of "
                f"{rows.tasks_total} image tasks could not be reached, so "
                f"anything published only under those is missing here"
            )
        return {
            "models": rows,
            # "" means every task, and says so rather than reporting one tag
            # that was never the filter. It used to answer `text-to-image` for
            # a search that had not been narrowed to anything.
            "task": task or "",
            "tasks_searched": rows.tasks_searched,
            "tasks_total": rows.tasks_total,
            "limit_asked": rows.limit_asked,
            "limit_used": rows.limit_used,
            "cache_capped": rows.cache_capped,
            "means": (
                f"{len(rows)} model(s) from the Hub, {here} already on this "
                f"machine and {half} with a cache entry but no weights in it. "
                f"{len(rows) - len(sized)} publish no size metadata, which is "
                f"UNKNOWN rather than small. Nothing was downloaded."
                + (
                    " " + ". ".join(c[:1].upper() + c[1:] for c in cut) + "."
                    if cut
                    else ""
                )
            ),
        }

    @app.post("/api/image/adapter")
    async def image_adapter(req: AdapterRequest, request: Request):
        """What a LoRA changes, read off the adapter rather than guessed.

        A weight diff, deliberately, and the cheap half of the question. The
        text side answers "what did a finetune change" by RUNNING both models
        over a prompt set, which for diffusion means two multi-gigabyte
        pipelines resident at once — and the common case is an 80 MB LoRA that
        states its own targets. `/api/image/filmstrip` is where you go to see
        what it does to a picture; these are different questions.

        Every norm here is a MAGNITUDE. The response says so, because a large
        move in a layer the sampler barely exercises can matter less than a
        small one in a layer it leans on.
        """
        from . import adapter_diff

        # A path in the body names a file on the SERVER's disk, which is
        # somebody else's machine as often as it is yours. Same trust boundary
        # as the corpus reader and `/api/weights/scan`: loopback alone does not
        # settle it, because any page on any site can POST to localhost.
        refusal = _not_from_this_machine(
            request, "Reading an adapter off this machine's disk"
        )
        if refusal is not None:
            return refusal

        # The base model is passed ONLY when one is already resident. The
        # relative norm needs real weights, and loading a pipeline to produce a
        # denominator would turn an 80 MB read into a 10 GB one — the response
        # reports `relative: null` instead, which is the honest answer to
        # "compared with what?".
        base = None
        try:
            if app.state.image.status().loaded:
                base = app.state.image.require()
        except (Refusal, BadRequest):
            base = None

        try:
            report = await asyncio.to_thread(
                adapter_diff.read, req.path, base=base, top=req.top
            )
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": err.sentence}, status_code=code)
        except Exception as err:
            return _internal(err, "/api/image/adapter")

        return report.to_dict()

    @app.get("/api/image/discovered")
    async def image_discovered() -> dict:
        """Image models in ordinary folders — the running directory included.

        `/api/image/local` reads the Hub cache. This walks the same roots the
        text picker walks, so a checkpoint cloned into the working directory
        turns up here rather than nowhere. The roots are returned: "nothing
        found" without "and here is where I looked" is not an answer.
        """
        from . import image_catalog

        out = await asyncio.to_thread(image_catalog.discovered)
        rows, roots = out["models"], out["roots"]
        cap = (
            " The walk hit its budget, so this list is what was reached "
            "rather than everything there is."
            if out["truncated"]
            else ""
        )
        return {
            **out,
            "means": (
                f"{len(rows)} image model(s) found in {len(roots)} "
                f"director{'y' if len(roots) == 1 else 'ies'} outside the Hub "
                f"cache.{cap}"
            ),
        }

    @app.get("/api/image/local")
    async def image_local() -> dict:
        """What is on this disk, what it weighs, and whether it finished.

        `/api/image/available` answers what each cached model IS.
        This answers what it COSTS, read off the files rather than the Hub —
        including the case a browse list cannot show: a cache entry holding
        configs and no weights, which is an interrupted download and not a
        model that is ready.
        """
        from . import fmt, image_catalog, imaging

        rows = await asyncio.to_thread(image_catalog.local)
        whole = [r for r in rows if r["complete"] is True]
        # `is None` is a THIRD state, not a falsy one. An entry nobody could
        # size is not an interrupted download, and the sentence below used to
        # call it one — sending a reader to re-fetch a model that may be
        # sitting there complete behind a permissions error.
        unsized = [r for r in rows if r["complete"] is None]
        held = sum(r["size_bytes"] or 0 for r in whole)
        partial = len(rows) - len(whole) - len(unsized)
        # Each clause appears only when it has something to report. "0 hold
        # configs and no weights" is a sentence about nothing, and a summary
        # made of those is one nobody reads.
        rest = ""
        if partial:
            rest += (
                f" {partial} hold configs and no weights — an interrupted "
                f"download rather than a model that is ready."
            )
        if unsized:
            rest += (
                f" {len(unsized)} could not be sized at all, so whether their "
                f"weights arrived is unknown rather than answered."
            )
        # THE SAME CAP `/api/image/available` REPORTS, on the same walk. Both
        # routes read `imaging.scan_cache`, which stops at SCAN_CACHE_LIMIT;
        # `available` says so and this did not — and this is the one the panel
        # renders, because `getImageAvailable` has no consumers. A list that
        # silently stops at 200 reads as "this is everything on the disk",
        # which is the one thing it is not.
        capped = len(rows) >= imaging.SCAN_CACHE_LIMIT
        if capped:
            rest += (
                f" That is as many as one pass of the cache reads "
                f"({imaging.SCAN_CACHE_LIMIT}), so there may be more here that "
                f"are not listed."
            )
        return {
            "models": rows,
            "bytes_on_disk": held,
            "unsized": len(unsized),
            "truncated": capped,
            "scan_limit": imaging.SCAN_CACHE_LIMIT,
            "means": (
                f"{len(rows)} image model(s) on this machine, {len(whole)} of "
                # `fmt.bytes_si`, not `/1e9`. A cache holding only
                # `tiny-stable-diffusion-torch` is about 4 MB, and this
                # sentence called it "0.0 GB in total" in the same breath as
                # saying the weights are present.
                f"them with weights actually present, {fmt.bytes_si(held)} in "
                f"total.{rest}"
            ),
        }

    @app.get("/api/image/size")
    async def image_size(repo: str = "") -> dict:
        """What downloading one model would cost, before any of it moves."""
        from . import image_catalog

        try:
            return await asyncio.to_thread(image_catalog.size_of, repo)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=503)

    @app.post("/api/image/load")
    async def image_load(req: ImageLoadRequest, request: Request):
        """Hold one pipeline, after three refusals that cost nothing.

        Identify from JSON, scan the opcodes, price from real bytes — then
        load. Off the event loop because a pipeline is gigabytes off disk.
        """
        # `repo` is EITHER a Hub id or a directory on this machine, and only
        # the second needs the guard — which is why it is applied on the
        # second and not on both. A Hub id is a public name and refusing it
        # from a remote caller would be refusing the ordinary case.
        #
        # This route had no guard at all, while the four other path-accepting
        # routes in this file have carried one for months. CodeQL found it as
        # an uncontrolled path expression; the real defect is that two routes
        # answered the same question about the same kind of input differently.
        # SHAPE, never the filesystem. The first version of this guard asked
        # `Path(repo).is_dir()` to decide whether the guard applied — and
        # asking that question about caller text ANSWERS it. Measured: a name
        # that exists in the server's working directory took the guard and
        # came back 403, one that does not fell through and came back 500. One
        # bit per request, unauthenticated, to anyone who can reach a server
        # started with `--host 0.0.0.0`.
        #
        # `behavdiff.is_hub_id` already states this rule for
        # `/api/quantdiff/behaviour` — "that question cannot be asked about
        # caller-supplied text without answering it" — so this uses it rather
        # than keeping a second, weaker opinion about what a path looks like.
        # Two code paths answering one question differently is the defect.
        from . import behavdiff
        from . import capacity as _capacity

        here = _from_this_machine(request)
        if not here and not behavdiff.is_hub_id(req.repo):
            refusal = _not_from_this_machine(
                request,
                "loading a model from a path on this machine",
                because=(
                    "a path names a directory on the disk this server is "
                    "running on, not on yours"
                ),
            )
            if refusal is not None:
                return refusal

        already = 0
        if runtime.loaded and runtime.model is not None:
            # A text model resident in THIS process. Both are wanted resident
            # at once, and unlike one oversized model neither can be offloaded
            # to rescue the other.
            #
            # Measured off the module rather than from the checkpoint on disk:
            # what matters is what is in memory NOW, which differs from the
            # file whenever a dtype was cast at load.
            from . import weights_table

            already = sum(
                row.bytes or 0
                for row, _ in weights_table.rows_from_module(runtime.model)
            )

        try:
            status = await asyncio.to_thread(
                app.state.image.load,
                req.repo,
                device=req.device,
                dtype=req.dtype,
                confirm=req.confirm,
                already_held_bytes=already,
                # The shape gate above is necessary and NOT sufficient.
                # `is_hub_id("models")` is True — a bare name is a valid repo
                # id — and `_snapshot` uses an existing directory as-is, so a
                # remote caller naming `models` would have had the server's
                # own `./models` loaded rather than a Hub repo. Worse than the
                # oracle it replaced, because it does not merely reveal that
                # the directory exists, it opens it.
                #
                # So the person at the keyboard keeps "point me at a directory
                # on my disk", and a request from anywhere else gets the Hub
                # branch and nothing else.
                local_ok=here,
            )
            return status.to_dict()
        except ImageLoadCancelled as err:
            # NOT a failure: the reader asked. 200 with a plain sentence, so
            # the panel does not paint a red error over something they did on
            # purpose — the same shape `/api/model/load` uses for its own
            # `LoadCancelled`, because they are the same event.
            return JSONResponse({"cancelled": True, "message": err.sentence})
        # No separate `except Unsafe`. It is a `Refusal`, so the clause below
        # already answers 409 with its sentence intact — the extra arm added
        # nothing and named a type the leak check does not have on its
        # allow-list, which is the check doing its job: every exception a
        # handler publishes has to be provably authored, and proving it by
        # naming subclasses one at a time is how that list stops being true.
        except _capacity.TooBig as err:
            # BEFORE the Refusal arm, and it has to be its own arm at all
            # because `TooBig` subclasses plain `ValueError` rather than
            # `BadRequest` — so `(Refusal, BadRequest)` below does not catch
            # it and it fell through to `except Exception` and a 500.
            #
            # That turned the one refusal this route exists to deliver into
            # "Something inside ModelMRI failed rather than refusing": the
            # capacity guard's whole job is to say "this will not fit, here is
            # what to do" BEFORE a twenty-minute download, and a reader was
            # shown a crash instead. Four other capacity-gated routes have
            # carried this arm since they were written; this one did not.
            return JSONResponse({"error": err.sentence}, status_code=422)
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": err.sentence}, status_code=code)
        except Exception as err:
            return _internal(err, "/api/image/load")

    @app.get("/api/image/progress")
    def image_progress() -> dict:
        """Polled while a pipeline loads — a minutes-long wait needs a pulse.

        A separate tracker from `/api/model/progress`, for the reason
        `progress.py` already documents about pulls: an image pipeline and a
        language model are different jobs held by different handles, either
        can be resident while the other loads, and one shared slot reports one
        job's name against another job's byte counts.
        """
        from .progress import IMAGE_LOADS

        return IMAGE_LOADS.snapshot().to_dict()

    @app.post("/api/image/cancel")
    def image_cancel() -> dict:
        """Stop an in-flight pipeline load.

        `stopping` is False when there was nothing running. The stop lands
        BETWEEN stages: `from_pretrained` is one opaque call that cannot be
        interrupted, so a stop asked for while the pipeline is opening takes
        effect when that returns. Said here rather than implied, because a
        Stop that appears to do nothing for ten minutes is worse than one
        whose limit is stated.
        """
        from .progress import IMAGE_LOADS

        return {
            "stopping": IMAGE_LOADS.request_cancel(),
            "means": (
                "A stop is honoured between stages. If the pipeline is already "
                "opening, that call cannot be interrupted and the stop lands "
                "when it returns."
            ),
        }

    @app.post("/api/image/unload")
    async def image_unload():
        """Drop it and hand the memory back, not merely forget it."""
        try:
            status = await asyncio.to_thread(app.state.image.unload)
        except Refusal as err:
            # A load is in flight and holding the handle. This used to block
            # here with no ceiling, and because the body runs through
            # `asyncio.to_thread` each blocked caller sat on a thread from the
            # default executor — 28 of them starved `/api/model/unload`, whose
            # own refusal was working the whole time and never got a thread.
            return JSONResponse({"error": err.sentence}, status_code=409)
        return status.to_dict()

    def _image_can(what: str):
        """The handle, or a refusal naming what this family cannot do.

        Two different refusals, deliberately: "nothing is loaded" and "this
        architecture has no such thing" are different problems with different
        fixes, and collapsing them would send somebody to load a second model
        that also cannot answer.
        """
        handle = app.state.image
        handle.require()
        status = handle.status()
        if what not in status.capabilities:
            raise Refusal(
                f"{status.repo or 'this model'} is "
                f"{status.family or 'an architecture'}, which has no "
                f"{what.replace('_', ' ')} to measure. Drawing one anyway "
                f"would be a picture of something that does not exist."
            )
        return handle

    @app.get("/api/image/attention/cost")
    def image_attention_cost(steps: int = 20, words: int = 0):
        """Renders and passes, before any are spent.

        Bounded, because this route's entire job is to be trustworthy about
        cost and it was answering "-40 denoising passes" with the same
        confidence as a real plan. A preflight that prices an impossible run
        is worse than no preflight: the number is what the reader decides on.
        """
        from . import image_attention

        try:
            steps = _bounded(steps, "steps", low=1, high=MAX_DENOISE_STEPS)
            words = _bounded(words, "words", low=0, high=MAX_KNOCKOUT_WORDS)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        return image_attention.plan(int(steps), int(words))

    @app.post("/api/image/attention")
    async def image_attention_capture(req: ImageRunRequest):
        """Which words the image attends to, per denoising step.

        Early steps decide layout and late steps decide texture, so a single
        averaged map hides the thing worth seeing.
        """
        from . import image_attention

        try:
            handle = _image_can("cross_attention")
        except (Refusal, BadRequest) as err:
            return JSONResponse({"error": err.sentence}, status_code=409)

        try:
            run = await asyncio.to_thread(
                image_attention.capture,
                handle.require(),
                req.prompt,
                model_name=handle.status().repo,
                steps=req.steps,
                seed=req.seed,
            )
            return run.to_dict()
        except (image_attention.NotSupported, Refusal) as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/image/attention")

    @app.post("/api/image/knockout")
    async def image_knockout(req: ImageKnockoutRequest):
        """Remove one prompt word at a time and measure what actually moved.

        The interventional counterpart to the attention map, and the reason
        this does not stop at a heatmap: a word can be attended to and change
        nothing.
        """
        from . import image_attention

        try:
            handle = _image_can("token_knockout")
        except (Refusal, BadRequest) as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        if not req.words:
            return JSONResponse(
                {
                    "error": (
                        "no words were named, so there is nothing to knock "
                        "out. Pick them from the attention map rather than "
                        "having this choose — which words matter is the "
                        "question, not an implementation detail."
                    )
                },
                status_code=422,
            )

        try:
            return await asyncio.to_thread(
                image_attention.knockout,
                handle.require(),
                req.prompt,
                tokens=[str(w) for w in req.words],
                seed=req.seed,
                steps=req.steps,
            )
        except (image_attention.NotSupported, Refusal) as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/image/knockout")

    @app.get("/api/image/attribution/cost")
    def image_attribution_cost(
        height: int = 224,
        width: int = 224,
        patch: int = 16,
        stride: int = 0,
        batch: int = 32,
    ):
        """How many forward passes covering the image up would take.

        Asked FIRST and answers without a model, because the number it
        produces is the one that decides whether to run at all: the same image
        at stride 1 rather than stride 16 is not a slower run, it is a
        different afternoon. `estimate` never refuses on the ceiling — a
        caller about to be refused needs the number that got them refused.
        """
        from . import vision_attr

        try:
            return vision_attr.estimate(
                height,
                width,
                patch=patch,
                # 0 is the query-string way of saying "not stated"; the module
                # then uses the patch size, which is non-overlapping windows.
                stride=stride or None,
                batch=batch,
            )
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": err.sentence}, status_code=code)

    @app.get("/api/image/cv/cost")
    def image_cv_cost(
        height: int,
        width: int,
        patch: int = 16,
        stride: int | None = None,
        batch: int = 32,
    ):
        """What the three CV measurements cost, before any is spent."""
        from . import image_cv

        try:
            handle = app.state.image
            handle.require()
            shape = image_cv.readout_shape_of(handle.require())
            return image_cv.plan(
                int(height),
                int(width),
                layers=shape.get("layers"),
                heads=shape.get("heads"),
                tokens=shape.get("tokens"),
                patch=int(patch),
                stride=None if stride is None else int(stride),
                batch=int(batch),
            )
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": err.sentence}, status_code=code)

    @app.post("/api/image/cv/predict")
    async def image_cv_predict(req: CVPredictRequest):
        """What a classifier, detector or segmenter says about one image.

        The label names come off the checkpoint's own `id2label`. A checkpoint
        that publishes none gets indices AND a note saying so, rather than
        borrowing ImageNet's names -- a wrong class name is read as the
        model's answer, which is worse than a number nobody can interpret.
        """
        from . import image_cv, image_input

        try:
            handle = app.state.image
            handle.require()
            processor = handle.require_processor()
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": err.sentence}, status_code=code)

        try:
            picture = image_input.decode(req.image)
            # A TENSOR, prepared by the checkpoint's own processor. `image_cv`
            # deliberately does no image loading, so that what the model is
            # shown is exactly what was built for it rather than whatever a
            # convenience path guessed.
            tensor = image_input.to_tensor(
                picture, processor, device=handle.status().device
            )
            found = await asyncio.to_thread(
                image_cv.predict,
                handle.require(),
                tensor,
                top_k=req.top_k,
                processor=processor,
                mask_threshold=req.mask_threshold,
                model_name=handle.status().repo,
            )
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": err.sentence}, status_code=code)
        except Exception as err:
            return _internal(err, "/api/image/cv/predict")

        return found.to_dict()

    @app.post("/api/image/cv/readout")
    async def image_cv_readout(req: CVReadoutRequest):
        """What each layer looked at -- where the architecture has such a thing.

        A ViT has attention; a convolutional backbone does not. The refusal
        says which, rather than producing something shaped like an answer.
        """
        from . import image_cv, image_input

        try:
            handle = _image_can("layer_readout")
            processor = handle.require_processor()
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": err.sentence}, status_code=code)

        try:
            picture = image_input.decode(req.image)
            tensor = image_input.to_tensor(
                picture, processor, device=handle.status().device
            )
            found = await asyncio.to_thread(
                image_cv.layer_readout,
                handle.require(),
                tensor,
                model_name=handle.status().repo,
            )
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": err.sentence}, status_code=code)
        except Exception as err:
            return _internal(err, "/api/image/cv/readout")

        return found.to_dict()

    @app.post("/api/image/cv/attribute")
    async def image_cv_attribute(req: CVAttributeRequest):
        """Cover each window and measure what the PREDICTION did.

        Not always the argmax. "Why did it pick that" and "what supports this
        other class" are different questions, and a detector's third box is a
        different question again from its first.
        """
        from . import image_cv, image_input

        try:
            handle = _image_can("attribution")
            processor = handle.require_processor()
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": err.sentence}, status_code=code)

        try:
            picture = image_input.decode(req.image)
            tensor = image_input.to_tensor(
                picture, processor, device=handle.status().device
            )
            found = await asyncio.to_thread(
                image_cv.attribute,
                handle.require(),
                tensor,
                target=req.target,
                query=req.query,
                region=req.region,
                processor=processor,
                patch=req.patch,
                stride=req.stride,
                fill=req.fill,
                # THE SAME range `/api/image/attribution` reads, from the same
                # processor this route already has in hand two lines above.
                # Without it `image_cv.attribute` infers one from the
                # picture's own extremes, so the two routes occluded with
                # different greys and returned different maps for the same
                # model, the same picture and the same target.
                #
                # MEASURED on google/vit-base-patch16-224, one 224x224 image,
                # patch and stride 112, target 902 "whistle", base logit 4.625
                # on both:
                #   cv/attribute  fill 0.019608, range [-0.686, 0.725] inferred
                #   attribution   fill 0.0,      range [-1.0, 1.0]  from the processor
                # and cell [0][1] came back -0.21875 against -0.25.
                #
                # The processor's range wins because it is a fact about the
                # MODEL. A range inferred from one photograph is a fact about
                # that photograph — a picture of a bright sky never reaches
                # the bottom of the model's input range, so its "grey" is not
                # the model's neutral and every score under it measures the
                # patch as well as the occlusion.
                value_range=image_input.value_range_of(processor),
                batch=req.batch,
                model_name=handle.status().repo,
            )
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": err.sentence}, status_code=code)
        except Exception as err:
            return _internal(err, "/api/image/cv/attribute")

        return found.to_dict()

    @app.get("/api/image/filmstrip/cost")
    def image_filmstrip_cost(
        steps: int = 20,
        every: int | None = None,
        # `at` was named in this route's own refusal — "Pass `every=N` … or
        # `at=[...]` for exactly the steps you want" — and was not in the
        # signature, so FastAPI dropped it and a caller who did exactly what
        # they were told received the identical refusal again.
        # `image_steps.filmstrip_plan` has always taken it; only the wiring
        # was missing.
        at: Annotated[list[int] | None, Query()] = None,
        include_final: bool = True,
        frame_pixels: int = _steps_defaults.DEFAULT_FRAME_PIXELS,
    ):
        """How many decodes a strip costs, and which steps it would hold.

        Answered before the run because a VAE decode per step is a full pass
        through the decoder on top of the denoising being measured -- the
        reason the sibling latent measurement decodes nothing at all.
        """
        from . import image_steps

        try:
            return image_steps.filmstrip_plan(
                int(steps),
                every=None if every is None else int(every),
                at=at,
                include_final=bool(include_final),
                frame_pixels=int(frame_pixels),
            )
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": err.sentence}, status_code=code)

    @app.post("/api/image/filmstrip")
    async def image_filmstrip(req: FilmstripRequest):
        """Run the pipeline once and decode the steps you named.

        The response says which steps were decoded and which were skipped, so
        an 8-frame strip cannot be mistaken for a 50-step run -- and it
        carries the caveat that a decoded frame is not evidence of when the
        LATENT stopped moving, which is what `/api/image/steps` measures.
        """
        from . import image_steps

        try:
            handle = _image_can("latent_trace")
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": err.sentence}, status_code=code)

        try:
            found = await asyncio.to_thread(
                image_steps.filmstrip,
                handle.require(),
                req.prompt,
                model_name=handle.status().repo,
                seed=req.seed,
                steps=req.steps,
                every=req.every,
                at=req.at,
                include_final=req.include_final,
                frame_pixels=req.frame_pixels,
            )
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": err.sentence}, status_code=code)
        except Exception as err:
            return _internal(err, "/api/image/filmstrip")

        return found.to_dict()

    @app.post("/api/image/attribution")
    async def image_attribution(req: ImageAttributionRequest):
        """Cover each window of one image, re-run, and report what moved.

        Every saliency map this project could have drawn for a classifier is a
        gradient or an attention weight, and both are correlational. This is
        the interventional one: it removes evidence and measures the answer,
        which is the same argument `patch.py` makes on the text side and
        `vla_occlude.py` makes for a robot's camera.
        """
        from . import image_input, vision_attr

        try:
            handle = _image_can("attribution")
            processor = handle.require_processor()
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": err.sentence}, status_code=code)

        try:
            picture = image_input.decode(req.image)
            # The checkpoint's own preprocessor does the resize and the
            # normalisation, because those two are what it was trained with
            # and doing either here would be doing them differently.
            tensor = image_input.to_tensor(
                picture, processor, device=handle.status().device
            )
            # READ from the processor rather than inferred from this one
            # picture. `vision_attr` will infer a range from the image's own
            # extremes and say that it did, which is honest but weak: a
            # photograph of a bright sky never reaches the bottom of the
            # model's input range, so "grey" would land somewhere that is not
            # the midpoint. `None` here means the processor published too
            # little, and the inference-with-a-caveat is then the right answer.
            value_range = image_input.value_range_of(processor)

            found = await asyncio.to_thread(
                vision_attr.sweep,
                handle.require(),
                tensor,
                target=req.target,
                patch=req.patch,
                stride=req.stride,
                fill=req.fill,
                value_range=value_range,
                batch=req.batch,
                class_names=_label_names(handle.require()),
                model_name=handle.status().repo,
            )
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": err.sentence}, status_code=code)
        except Exception as err:
            return _internal(err, "/api/image/attribution")

        return found.to_dict()

    @app.post("/api/image/steps")
    async def image_steps_run(req: StepTraceRequest):
        """When the denoiser stopped moving, and what the steps after bought.

        Nothing is decoded: the response carries `vae_decodes: 0` as a
        checkable claim, because a decode would make the answer a property of
        the VAE as much as of the denoiser. `/api/image/filmstrip` is the one
        that draws pictures, and it says what a decoded frame is and is not
        evidence of.
        """
        from . import image_steps

        try:
            handle = _image_can("step_commit")
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": err.sentence}, status_code=code)

        try:
            kwargs = {}
            if req.threshold > 0:
                kwargs["threshold"] = float(req.threshold)
            found = await asyncio.to_thread(
                image_steps.trace,
                handle.require(),
                req.prompt,
                seed=req.seed,
                steps=req.steps,
                model_name=handle.status().repo,
                **kwargs,
            )
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": err.sentence}, status_code=code)
        except Exception as err:
            return _internal(err, "/api/image/steps")

        return found.to_dict()

    @app.get("/api/image/steps/cost")
    def image_steps_cost(steps: int = 20, threshold: float = 0.0):
        """What a latent trace will hold, before it holds it.

        The latent shape is read off the loaded pipeline when there is one, so
        the memory figure is this pipeline's rather than a guess — and `None`
        rather than 0 when there is not, because a run whose memory could not
        be priced is not a run that costs nothing.
        """
        from . import image_steps

        try:
            steps = _bounded(steps, "steps", low=1, high=MAX_DENOISE_STEPS)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        # 0.0 is the query-string way of saying "not stated" and takes the
        # module's own default. Anything else is a value the caller CHOSE, and
        # a negative one used to be replaced by 0.95 and then reported as
        # `"threshold": 0.95` — the payload stating a setting nobody asked
        # for. The refusal written for `threshold > 1` names the same rule.
        if threshold and not 0 < threshold <= 1:
            return JSONResponse(
                {
                    "error": (
                        f"a commit threshold of {threshold:g} is not a share "
                        f"of the movement. It must be greater than 0 and at "
                        f"most 1: at 0 every run 'commits' at its first "
                        f"measured step, which is a fact about the arithmetic "
                        f"rather than about the image."
                    )
                },
                status_code=422,
            )

        shape = None
        if app.state.image.pipe is not None:
            try:
                shape = image_steps.latent_shape_of(app.state.image.pipe)
            except Exception:
                # Best-effort: an unpriceable trace still gets its pass count,
                # and `latent_bytes: null` says the memory half is unknown.
                shape = None
        try:
            kwargs = {"latent_shape": shape}
            if threshold > 0:
                kwargs["threshold"] = float(threshold)
            return image_steps.plan(int(steps), **kwargs)
        except (Refusal, BadRequest) as err:
            return JSONResponse({"error": err.sentence}, status_code=422)

    @app.get("/api/weights/cost")
    def weights_cost(exhaustive: bool = False):
        """What a health scan would read, before it reads a single weight.

        The table half is free and the health half is memory bandwidth, so the
        price of the expensive half is knowable from the cheap half — element
        counts come out of the module's own shapes.
        """
        from . import weights_table

        if not runtime.loaded or runtime.model is None:
            return JSONResponse(
                {
                    "error": (
                        "No model is loaded in this process, so there are no "
                        "weights to price. Load one and ask again."
                    )
                },
                status_code=409,
            )
        counts = [
            row.elements for row, _ in weights_table.rows_from_module(runtime.model)
        ]
        return {
            "tensors": len(counts),
            "elements_total": sum(counts),
            **weights_table.scan_cost(counts, exhaustive=bool(exhaustive)),
        }

    @app.get("/api/weights")
    async def weights_view(
        health: bool = False,
        exhaustive: bool = False,
        limit: int = 0,
    ):
        """The per-tensor table for the loaded model.

        `health=false` by default, and that default is the honest one: the
        table is free, the health scan reads every element it is allowed to,
        and `/api/weights/cost` prices it first.

        Off the event loop when health is on — it is pure memory bandwidth and
        would block every other panel for as long as it ran.
        """
        from . import weights_table

        if not runtime.loaded or runtime.model is None:
            return JSONResponse(
                {
                    "error": (
                        "No model is loaded in this process. This reads the "
                        "module in memory rather than a file on disk, which is "
                        "the whole difference from a checkpoint viewer."
                    )
                },
                status_code=409,
            )
        try:
            kwargs = {
                "health": bool(health),
                "exhaustive": bool(exhaustive),
                "source": runtime.hf_id or "the loaded model",
            }
            if limit > 0:
                kwargs["limit"] = int(limit)
            table = await asyncio.to_thread(
                weights_table.table, runtime.model, **kwargs
            )
            return table.to_dict()
        except (Refusal, BadRequest) as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except Exception as err:
            return _internal(err, "/api/weights")

    @app.post("/api/weights/scan")
    async def weights_scan_path(req: ScanRequest, request: Request):
        """Look inside a checkpoint for anything that executes on load.

        A path, so it carries the same not-from-this-machine guard every other
        file-path route does: a request from elsewhere on the network naming a
        path is a request to read somebody else's disk.
        """
        from . import weights_scan

        refusal = _not_from_this_machine(
            request,
            "reading a path on this machine",
            because="a scan names a file on the disk this server is running on",
        )
        if refusal is not None:
            return refusal

        # `.resolve()`, and the reason is the finding CodeQL raised: without it
        # a path is used as written, so `../../..` walks wherever it likes and
        # nothing downstream can tell that happened. Resolving normalises the
        # traversal FIRST, so the path that gets scanned is the path that gets
        # reported — which matters here more than usual, because every report
        # in the response names the file it describes.
        #
        # It does not become a sandbox and is not meant to be one. Reading a
        # local path IS the feature: this scans the weights on the disk the
        # server is running on, and `_not_from_this_machine` above is what
        # makes that safe by restricting WHO can ask. The same pairing runs on
        # the four other file-path routes in this file.
        try:
            target = Path(req.path).expanduser().resolve()
        except (OSError, ValueError, RuntimeError):
            # A path the OS will not even normalise — a bad drive letter, a
            # symlink loop, a name too long. A refusal rather than a 500,
            # because this is a fact about the input, not a fault in here.
            return JSONResponse(
                {
                    "error": (
                        f"{req.path!r} is not a path this machine can resolve. "
                        f"Check the drive and that no link in it points at "
                        f"itself."
                    )
                },
                status_code=422,
            )

        try:
            reports = await asyncio.to_thread(
                (lambda: weights_scan.scan_dir(target, limit=req.limit))
                if target.is_dir()
                else (lambda: [weights_scan.scan(target)])
            )
        except Exception as err:
            return _internal(err, "/api/weights/scan")

        dangerous = [r for r in reports if r.dangerous]
        unscanned = [r for r in reports if r.verdict == weights_scan.UNSCANNED]
        # What the walk itself could not tell you, carried beside the counts.
        # A single file scan has no walk, so it is neither capped nor unread.
        n_total = getattr(reports, "n_total", len(reports))
        readable = getattr(reports, "readable", True)
        return {
            "reports": [r.to_dict() for r in reports],
            "dangerous": len(dangerous),
            "unscanned": len(unscanned),
            "safe": len(reports) - len(dangerous) - len(unscanned),
            # The cap, REPORTED rather than left to be inferred from a list
            # length. `scan_dir`'s docstring has always said "what it drops is
            # reported by the caller"; this is the caller finally doing it.
            "n_found": n_total,
            "truncated": n_total > len(reports),
            # And whether the folder could be opened at all. `false` here is
            # NOT "no weight files": it is "the contents are unknown", which
            # in a scanner is the difference between a clean bill of health
            # and never having looked.
            "readable": readable,
            "means": _scan_summary(
                reports, dangerous, unscanned, n_total=n_total, readable=readable
            ),
        }

    @app.get("/api/vla")
    def vla_status() -> dict:
        # Names the configured dataset without opening it, so the resting panel
        # can say what a click will read instead of guessing the default.
        return {
            **app.state.vla.status().to_dict(),
            "dataset_repo": getattr(app.state, "vla_dataset", dataset_repo),
            "policy_repo": VLA_DEFAULT_REPO,
        }

    @app.get("/api/policy")
    async def policy_status() -> dict:
        """Whether anything on this machine can say what the robot would DO.

        A GET that reaches out to another process, so it runs off the event
        loop: the sidecar answers in milliseconds when it is up, but a machine
        that has just been suspended can leave the connection hanging for the
        full timeout, and blocking the loop on that would freeze every panel.

        Deliberately never raises. "no action expert" is the resting state of
        most machines, not an error, and a 500 here would paint the panel red
        for a configuration that is completely normal.
        """
        from . import policy as _policy

        state = await asyncio.to_thread(_policy.status)
        return {
            **state.to_dict(),
            "installed": _policy.installed(),
            "venv": str(_policy.venv_dir()),
            "contract_here": _policy.CONTRACT,
            "install_hint": _policy.INSTALL_HINT,
            # What the second process would cost, from the same constants the
            # capacity refusal uses. A panel offering a 6 GB install should
            # say 6 GB before the click, not after.
            "venv_disk_bytes": _policy.VENV_DISK_BYTES,
            "assumed_policy_bytes": _policy.ASSUMED_POLICY_BYTES,
        }

    @app.post("/api/vla/load")
    async def vla_load(req: VLALoadRequest):
        try:
            status = await asyncio.to_thread(app.state.vla.load, req.repo)
            return status.to_dict()
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/vla/load")

    @app.get("/api/vla/datasets")
    async def vla_datasets() -> dict:
        """Every cached LeRobot dataset, not just the configured one."""
        from .vla_data import cached_datasets

        return {
            "datasets": await asyncio.to_thread(cached_datasets),
            "current": getattr(app.state, "vla_dataset", dataset_repo),
        }

    # ---------------- what the policy would DO (ROADMAP #50) ----------------
    #
    # Every route here needs the ACTION sidecar, not just the vision tower, so
    # every one of them can refuse with the command that fixes it. They are
    # POST rather than GET because each spends real forward passes -- a GET
    # that costs three minutes is a GET somebody's browser will retry.

    def _policy_ready():
        """The sidecar, or the refusal that names what to run.

        Returns a `PolicyStatus`. Centralised so the routes below cannot
        disagree with each other about what "ready" means.
        """
        from . import policy as _policy

        state = _policy.status()
        if not state.running:
            raise Refusal(state.means())
        return state

    @app.get("/api/vla/actions/cost")
    def vla_actions_cost(episode: int = 0, stride: int = 0):
        """Forward passes before any are spent.

        Frames and passes, never seconds. A duration guessed from somebody
        else's hardware is the kind of number people plan around; the one
        timing quoted below was measured on this project's own machine and
        says so.
        """
        from . import vla_actions

        # `episodes()` INSIDE the try, not just `_reader()`. `discover()` reads
        # `meta/info.json` and nothing else, so a LeRobot v2.x cache or an
        # interrupted download passes it and refuses here instead — by name,
        # with the directory it looked in. That Refusal escaped as a 500
        # saying "Something inside ModelMRI failed rather than refusing",
        # while the 409 carrying the real sentence sat one frame down. This is
        # a preflight the panel calls on mount, so the panel went red pointing
        # at the terminal.
        try:
            reader = _reader()
            episodes = reader.episodes()
        except ImportError as err:
            return _missing_reader_dep(err)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)

        match = next((e for e in episodes if e.index == episode), None)
        if match is None:
            return JSONResponse(
                {"error": f"episode {episode} is not in this dataset"},
                status_code=422,
            )
        try:
            frames, chosen = vla_actions.plan_frames(match.length, stride=stride)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)

        return {
            "episode": episode,
            "frames_in_episode": match.length,
            "frames_measured": len(frames),
            "frames_skipped": match.length - len(frames),
            "stride": chosen,
            "passes": len(frames),
            "means": (
                f"{len(frames)} forward passes, one per sampled frame, "
                f"covering {len(frames)} of {match.length} frames at a stride "
                f"of {chosen}. No seconds are quoted because this machine has "
                f"not been timed on this policy. For scale only: one SmolVLA "
                f"pass took 49 s on a CPU build of torch during development, "
                f"and far less on a GPU one."
            ),
        }

    @app.post("/api/vla/actions/compare")
    async def vla_actions_compare(req: VLACompareRequest):
        """Predicted against recorded, per dimension, across an episode."""
        from . import vla_actions

        try:
            state = _policy_ready()
            reader = _reader()
        except ImportError as err:
            return _missing_reader_dep(err)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)

        # Units BEFORE any forward pass. Spending three minutes and then
        # refusing to draw the result is a refusal that wasted three minutes,
        # and the answer does not depend on the passes.
        agree, why = vla_actions.units_agree(state.normalisation, reader.action_stats())
        if not agree:
            return JSONResponse({"error": why}, status_code=409)

        try:
            return await asyncio.to_thread(_run_compare, reader, state, req)
        except ImportError as err:
            # The DECODER, not the metadata reader. `_reader()` above only
            # opens the parquet; `av` is imported the first time a frame is
            # actually decoded, which happens in here — so without this arm a
            # machine with pyarrow and no av gets a 500 carrying none of the
            # sentence that says which package to install.
            return _missing_reader_dep(err)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/vla/actions/compare")

    def _run_compare(reader, state, req):
        from . import policy as _policy
        from . import vla_actions

        match = next((e for e in reader.episodes() if e.index == req.episode), None)
        if match is None:
            raise BadRequest(f"episode {req.episode} is not in this dataset")
        wanted, stride = vla_actions.plan_frames(match.length, stride=req.stride)

        rows = []
        for t in wanted:
            sample = reader.frame(req.episode, t)
            answer = _policy.act(
                frames={cam: sample.image for cam in state.cameras},
                state=sample.state,
                instruction=sample.task,
                seed=req.seed,
            )
            chunk = answer.get("action_chunk") or []
            if not chunk:
                raise Refusal(
                    f"the sidecar returned an empty action chunk at frame "
                    f"{t}, so there is nothing to compare there."
                )
            # The FIRST step of the chunk, and only that one. Later steps are
            # predictions about frames the demonstrator had not reached yet,
            # so pairing step 5 with frame t would compare a claim about the
            # future against the present and call the difference an error.
            rows.append((t, chunk[0], sample.action))

        return vla_actions.compare(
            frames=rows,
            joint_names=reader.action_names(),
            stride=stride,
            total_frames=match.length,
            policy_repo=state.policy_repo,
            revision=state.revision,
            seed=req.seed,
        )

    @app.post("/api/vla/actions/swap")
    async def vla_actions_swap(req: VLAFrameRequest):
        """Does the instruction move the action more than the sampler does?"""
        try:
            state = _policy_ready()
            reader = _reader()
        except ImportError as err:
            return _missing_reader_dep(err)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)

        if not state.samples:
            # Refused here rather than after the passes, for the same reason
            # the units check runs first: the answer does not depend on them.
            return JSONResponse(
                {
                    "error": (
                        f"the {state.family or 'loaded'} action head is "
                        f"deterministic, so its own sampling spread is exactly "
                        f"zero. This test measures the instruction effect "
                        f"AGAINST that spread, and a ratio against zero is not "
                        f"a number."
                    )
                },
                status_code=409,
            )

        try:
            return await asyncio.to_thread(_run_swap, reader, state, req)
        except ImportError as err:
            # The DECODER, not the metadata reader. `_reader()` above only
            # opens the parquet; `av` is imported the first time a frame is
            # actually decoded, which happens in here — so without this arm a
            # machine with pyarrow and no av gets a 500 carrying none of the
            # sentence that says which package to install.
            return _missing_reader_dep(err)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/vla/actions/swap")

    def _run_swap(reader, state, req):
        from . import policy as _policy
        from . import vla_actions

        sample = reader.frame(req.episode, req.t)
        frames = {cam: sample.image for cam in state.cameras}

        def first(instruction, seed):
            chunk = _policy.act(
                frames=frames,
                state=sample.state,
                instruction=instruction,
                seed=seed,
            ).get("action_chunk") or [[]]
            return chunk[0]

        # Every DISTINCT task string this dataset contains, read off the
        # episodes. Never invented: a distractor instruction written here
        # would measure a sentence somebody chose, not this policy.
        tasks: list[str] = []
        for ep in reader.episodes():
            task = (ep.task or "").strip()
            if task and task not in tasks:
                tasks.append(task)
        if sample.task and sample.task not in tasks:
            tasks.insert(0, sample.task)
        dropped = max(0, len(tasks) - vla_actions.MAX_INSTRUCTIONS)
        tasks = tasks[: vla_actions.MAX_INSTRUCTIONS]

        base = req.seed if req.seed is not None else 0
        swapped = [(task, first(task, base)) for task in tasks]
        # The reference: the SAME frame and the SAME instruction, re-rolled.
        seeds = [
            first(sample.task, base + i) for i in range(vla_actions.REFERENCE_SEEDS)
        ]
        return vla_actions.instruction_swap(
            own_instruction=sample.task,
            swapped=swapped,
            seed_samples=seeds,
            policy_repo=state.policy_repo,
            dropped_instructions=dropped,
        )

    @app.post("/api/vla/actions/knockout")
    async def vla_actions_knockout(req: VLAFrameRequest):
        """One bar per input, each replaced alone by its episode mean."""
        try:
            state = _policy_ready()
            reader = _reader()
        except ImportError as err:
            return _missing_reader_dep(err)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)

        try:
            return await asyncio.to_thread(_run_knockout, reader, state, req)
        except ImportError as err:
            # The DECODER, not the metadata reader. `_reader()` above only
            # opens the parquet; `av` is imported the first time a frame is
            # actually decoded, which happens in here — so without this arm a
            # machine with pyarrow and no av gets a 500 carrying none of the
            # sentence that says which package to install.
            return _missing_reader_dep(err)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/vla/actions/knockout")

    def _run_knockout(reader, state, req):
        import numpy as np

        from . import policy as _policy
        from . import vla_actions
        from .vla_data import encode_png

        sample = reader.frame(req.episode, req.t)
        seed = req.seed if req.seed is not None else 0
        frames = {cam: sample.image for cam in state.cameras}

        def act(these_frames, this_state, this_instruction):
            chunk = _policy.act(
                frames=these_frames,
                state=this_state,
                instruction=this_instruction,
                seed=seed,
            ).get("action_chunk") or [[]]
            return chunk[0]

        baseline = act(frames, sample.state, sample.task)

        # THIS episode's mean, computed from it. A grey rectangle would be a
        # different baseline than the label claims, and the label is the part
        # a reader trusts.
        episode = next((e for e in reader.episodes() if e.index == req.episode), None)
        length = episode.length if episode else 1
        step = max(1, length // 8)
        sampled = list(range(0, length, step))
        mean_rgb = encode_png(
            np.mean(
                np.stack(
                    [
                        reader.raw_frame(req.episode, t).astype("float64")
                        for t in sampled
                    ]
                ),
                axis=0,
            )
            .round()
            .astype("uint8")
        )
        mean_state = None
        if sample.state:
            rows = [reader.frame(req.episode, t).state for t in sampled]
            mean_state = [sum(col) / len(col) for col in zip(*rows, strict=True)]

        arms = []
        for cam in state.cameras:
            arms.append(
                (
                    cam,
                    f"{cam.split('.')[-1]} \u2192 episode mean",
                    act({**frames, cam: mean_rgb}, sample.state, sample.task),
                )
            )
        # A CONDITION, labelled as one. Never "the instruction did not
        # matter", which is a conclusion about a result nobody has read yet.
        arms.append(("instruction", "no instruction", act(frames, sample.state, "")))
        if mean_state is not None:
            arms.append(
                (
                    "observation.state",
                    "proprioceptive state \u2192 episode mean",
                    act(frames, mean_state, sample.task),
                )
            )

        # The policy's own sampling spread on THIS frame, so a bar can be told
        # from noise. Measured, and when it cannot be measured every bar
        # reports `above_noise: null` rather than a guess.
        spread = None
        if state.samples:
            spread = (
                vla_actions._spread(
                    [act(frames, sample.state, sample.task) for _ in range(3)]
                )
                or None
            )

        return vla_actions.knockout(
            baseline=baseline,
            arms=arms,
            policy_repo=state.policy_repo,
            sampling_spread=spread,
        )

    @app.get("/api/vla/sweep/cost")
    def vla_sweep_cost(
        metric: str = "attention_entropy",
        episode_stride: int = 1,
        frame_stride: int = 25,
        # POST /api/vla/sweep accepts this and prices the run WITH it — an
        # occlusion sweep at stride 1 is sixteen times the passes of one at
        # stride 4. The cost route did not, so it quoted the default's figure
        # for a run the reader had configured differently. `vla_sweep.estimate`
        # has always taken the parameter; only the signature was missing it,
        # which is the same gap `/api/image/filmstrip/cost` had with `at`.
        occlusion_stride: int = 0,
    ):
        """Frames and passes before the sweep starts.

        No seconds unless this machine has been timed: a duration guessed from
        somebody else's hardware is the kind of number people plan around.
        """
        from . import vla_sweep

        try:
            reader = _reader()
        except ImportError as err:
            return _missing_reader_dep(err)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)

        status = app.state.vla.status()
        # `estimate` reaches `reader.episodes()`, which refuses by name on a
        # cache `discover()` accepted — see /api/vla/actions/cost. Without a
        # Refusal arm here that became a 500 on a route the panel calls on
        # mount.
        try:
            return vla_sweep.estimate(
                reader,
                metric,
                episode_stride=episode_stride,
                frame_stride=frame_stride,
                grid=status.grid or None,
                occlusion_stride=occlusion_stride,
            )
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)

    @app.post("/api/vla/sweep")
    async def vla_sweep_run(request: Request):
        """One measurement over a strided sample of every episode.

        Ranked by ONE stated measured quantity, with the stride in every
        summary — a strided ranking can miss the worst frame entirely, and
        that is the first thing a reader needs rather than a footnote.

        No failure-mode names: a cluster labelled "dropped the object" that
        ModelMRI never verified is exactly the fabrication this refuses.
        """
        from . import vla_sweep

        try:
            body = await request.json()
        except Exception:
            # A 422 NAMING THE BODY, the way eight other routes in this file
            # already answer the same bytes. `body = {}` swallowed it, so the
            # request continued on defaults and the reader got whatever
            # sentence the NEXT check produced: `POST /api/custom/ablate` with
            # non-JSON answered "no custom model is loaded" — true, and about
            # something else entirely — and `POST /api/vla/sweep` answered 200
            # after running a full default sweep off a body nobody had read.
            #
            # The worse-formed body already got the better answer: `[1,2,3]`
            # parses, so it reached `_body_object` below and was refused
            # properly. Only bytes that are not JSON at all were let through.
            return JSONResponse({"error": "this request body is not JSON"}, 422)
        if (not_an_object := _body_object(body)) is not None:
            return not_an_object

        try:
            reader = _reader()
        except ImportError as err:
            return _missing_reader_dep(err)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except Exception as err:
            return _internal(err, "/api/vla/sweep")

        def run():
            out = vla_sweep.run(
                app.state.vla,
                reader,
                str(body.get("metric") or "attention_entropy"),
                episode_stride=_whole(body, "episode_stride", 1),
                frame_stride=_whole(body, "frame_stride", 25),
                occlusion_stride=_whole(body, "occlusion_stride", 0),
            )
            # Persisted so a sweep survives the process — the table is the
            # point of running one at all.
            vla_sweep.save(out)
            payload = out.to_dict()
            payload["strip"] = vla_sweep.heat_strip(out)
            return payload

        try:
            return await asyncio.to_thread(run)
        except ImportError as err:
            # The DECODER, not the metadata reader. The `_reader()` guard
            # above opens only the parquet; `av` is imported the first time a
            # frame is actually DECODED, which happens inside this worker. So
            # without this arm a machine with pyarrow and no av gets an opaque
            # 500 while the sentence naming the missing package sits
            # unreachable — the same shape as the bug fixed on
            # /api/vla/actions/* one commit ago.
            return _missing_reader_dep(err)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/vla/sweep")

    @app.post("/api/vla/share")
    async def vla_share(request: Request):
        """Write a robot finding to a `.mri` somebody else can open.

        There is no portable, no-account artifact for robot-policy internals
        anywhere: Foxglove archived its open-source Studio, Rerun's `.rrd`
        carries what the robot recorded rather than what the network computed,
        and HF Spaces need an upload and an account.

        The frame travels at the resolution the POLICY saw, and the section
        says so when it had to shrink it — a causal map is drawn over the
        frame, and one silently shrunk puts every block in the wrong place.
        """
        try:
            body = await request.json()
        except Exception:
            # A 422 NAMING THE BODY, the way eight other routes in this file
            # already answer the same bytes. `body = {}` swallowed it, so the
            # request continued on defaults and the reader got whatever
            # sentence the NEXT check produced: `POST /api/custom/ablate` with
            # non-JSON answered "no custom model is loaded" — true, and about
            # something else entirely — and `POST /api/vla/sweep` answered 200
            # after running a full default sweep off a body nobody had read.
            #
            # The worse-formed body already got the better answer: `[1,2,3]`
            # parses, so it reached `_body_object` below and was refused
            # properly. Only bytes that are not JSON at all were let through.
            return JSONResponse({"error": "this request body is not JSON"}, 422)
        if (not_an_object := _body_object(body)) is not None:
            return not_an_object

        try:
            reader = _reader()
        except ImportError as err:
            return _missing_reader_dep(err)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except Exception as err:
            return _internal(err, "/api/vla/share")

        def run() -> bytes:
            from . import session as session_mod
            from . import vla as vla_mod

            payload = vla_mod.share_payload(
                app.state.vla,
                reader,
                episode=_whole(body, "episode", 0),
                timestep=_whole(body, "t", 0),
                layer=_whole(body, "layer", -1),
                head=_whole(body, "head", -1),
                occlusion=body.get("occlusion") or None,
            )
            status = app.state.vla.status()
            return session_mod.build(
                model_id=status.repo,
                device=status.device,
                dtype=None,
                n_params=None,
                # A robot finding has no token strip and no generation. The
                # section carries the picture; the rest of the format's
                # language-model shape stays empty rather than being filled
                # with placeholders that would render as a run nobody made.
                tokens=[],
                prompt="",
                generation="",
                attention={},
                n_layers=status.n_layers,
                n_heads=status.n_heads,
                note=str(body.get("note") or ""),
                scope="one camera frame of one episode, with its causal map",
                vla=payload,
            )

        try:
            blob = await asyncio.to_thread(run)
        except ImportError as err:
            # The DECODER, not the metadata reader. The `_reader()` guard
            # above opens only the parquet; `av` is imported the first time a
            # frame is actually DECODED, which happens inside this worker. So
            # without this arm a machine with pyarrow and no av gets an opaque
            # 500 while the sentence naming the missing package sits
            # unreachable — the same shape as the bug fixed on
            # /api/vla/actions/* one commit ago.
            return _missing_reader_dep(err)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/vla/share")
        return Response(
            content=blob,
            media_type="application/gzip",
            headers={"Content-Disposition": 'attachment; filename="robot-finding.mri"'},
        )

    @app.post("/api/vla/occlude")
    async def vla_occlude_frame(request: Request):
        """What the policy's vision DEPENDED on, beside what it looked at.

        Occludes each block of the camera frame, re-runs the vision tower and
        reports how far the representation moved — every strong block against
        a null of same-area occlusions at random locations, and the rank
        correlation with the attention map for this same frame.

        PERCEPTION ONLY: this is a shift in an embedding, not an effect on the
        action, and the response says so in those words.
        """
        try:
            body = await request.json()
        except Exception:
            # A 422 NAMING THE BODY, the way eight other routes in this file
            # already answer the same bytes. `body = {}` swallowed it, so the
            # request continued on defaults and the reader got whatever
            # sentence the NEXT check produced: `POST /api/custom/ablate` with
            # non-JSON answered "no custom model is loaded" — true, and about
            # something else entirely — and `POST /api/vla/sweep` answered 200
            # after running a full default sweep off a body nobody had read.
            #
            # The worse-formed body already got the better answer: `[1,2,3]`
            # parses, so it reached `_body_object` below and was refused
            # properly. Only bytes that are not JSON at all were let through.
            return JSONResponse({"error": "this request body is not JSON"}, 422)
        if (not_an_object := _body_object(body)) is not None:
            return not_an_object

        try:
            reader = _reader()
        except ImportError as err:
            return _missing_reader_dep(err)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except Exception as err:
            return _internal(err, "/api/vla/occlude")

        # Read INSIDE a try that converts a refusal. The first pass put
        # these one line above the handler's own try, so the authored
        # sentence was raised and then swallowed by the generic 500 —
        # a guard that is unreachable is not a guard.
        try:
            episode = _whole(body, "episode", 0)
            timestep = _whole(body, "t", 0)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)

        def run():
            from . import vla_occlude

            frame = reader.raw_frame(episode, timestep)
            # Other frames of the SAME episode set the scale. Frames from
            # elsewhere would measure a spread this episode never shows.
            info = next((e for e in reader.episodes() if e.index == episode), None)
            span = info.length if info else 1
            step = max(1, span // vla_occlude.SCALE_FRAMES)
            scale = [reader.raw_frame(episode, t) for t in range(0, span, step)][
                : vla_occlude.SCALE_FRAMES
            ]
            out = app.state.vla.occlude(
                frame,
                scale,
                baseline=str(body.get("baseline") or "episode_mean"),
                stride=_whole(body, "stride", 0),
                layer=_whole(body, "layer", -1),
                head=_whole(body, "head", -1),
                # So a cached attention map from a DIFFERENT frame is not
                # silently ranked against this frame's causal map.
                key=(episode, timestep),
                camera=reader.camera,
            )
            # No post-hoc patching of the three identity fields here any more:
            # this route was the only caller that filled them in, so every
            # other one reported episode 0 timestep 0. They are set inside the
            # sweep now, from the same `key` the staleness check uses.
            return out

        try:
            return await asyncio.to_thread(run)
        except ImportError as err:
            # The DECODER, not the metadata reader. The `_reader()` guard
            # above opens only the parquet; `av` is imported the first time a
            # frame is actually DECODED, which happens inside this worker. So
            # without this arm a machine with pyarrow and no av gets an opaque
            # 500 while the sentence naming the missing package sits
            # unreachable — the same shape as the bug fixed on
            # /api/vla/actions/* one commit ago.
            return _missing_reader_dep(err)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/vla/occlude")

    @app.get("/api/vla/occlude/cost")
    def vla_occlude_cost(stride: int = 0):
        """What the sweep will cost, before it starts.

        A 32x32 grid at stride 1 is over a thousand tower passes. Nobody
        should discover that by waiting.
        """
        try:
            return app.state.vla.occlusion_cost(stride)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)

    @app.get("/api/vla/audit")
    async def vla_audit_dataset():
        """Prove the loaded robot dataset is intact — or say where it is not.

        NOTHING IS DOWNLOADED, no policy is loaded, no GPU is touched and
        lerobot is not imported: this reads the files already on disk.

        Every check reports what it measured and what it compared against.
        There is deliberately no grade — a letter would be a summary of
        somebody else's opinion about what matters in the reader's data.
        """
        from . import vla_audit as audit_mod

        try:
            reader = _reader()
        except ImportError as err:
            return _missing_reader_dep(err)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except Exception as err:
            return _internal(err, "/api/vla/audit")

        try:
            report = await asyncio.to_thread(audit_mod.audit, reader)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/vla/audit")
        return report.to_dict()

    @app.post("/api/vla/dataset")
    async def vla_set_dataset(req: VLADatasetRequest):
        """Switch datasets. Opens the new one before discarding the old."""
        from .vla_data import LeRobotV3Reader

        try:
            reader = await asyncio.to_thread(
                LeRobotV3Reader.discover, None, req.repo_id
            )
        except ImportError as err:
            # ImportError ALONE, and the pairing with FileNotFoundError that
            # used to be here is the point. vla.py and vla_data.py raise
            # `Refusal` now for every "not cached" / "no videos under ..."
            # sentence they wrote, so those are answered by the arm below in
            # their own words. What `except FileNotFoundError` caught in
            # addition was every OSError raised underneath — pyarrow opening a
            # parquet file, av opening a container — and it published those at
            # 409 with their absolute path in the body and no log line.
            # Measured: a reader raising `FileNotFoundError(2, "No such file
            # or directory", <abs path>)` leaked that path on all four of
            # these routes. Those reach `_internal` now.
            #
            # ImportError stays because it means one specific thing — pyarrow
            # or av is not installed — and the fix is one pip line, which is
            # the reader's to run and not something in a traceback.
            return _missing_reader_dep(err)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/vla/dataset")

        # INSIDE the try, and the state assigned only after it succeeds.
        #
        # `discover()` reads `meta/info.json` and nothing else, so a LeRobot
        # v2.x cache or an interrupted download passes it and fails one frame
        # later in `summary()` -> `episodes()`, which refuses by name. That
        # Refusal was raised outside every arm above: the route answered 500
        # "Something inside ModelMRI failed rather than refusing", while the
        # 409 with the real sentence sat unreachable. Reproduced.
        #
        # Worse, the two assignments happened BEFORE the failing call, so a
        # reader that cannot answer was installed process-wide and `_reader()`
        # would not re-discover it.
        try:
            summary = await asyncio.to_thread(reader.summary)
        except ImportError as err:
            return _missing_reader_dep(err)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/vla/dataset")
        app.state.vla_reader = reader
        app.state.vla_dataset = req.repo_id
        return summary

    @app.get("/api/vla/episodes")
    async def vla_episodes(camera: str | None = None):
        # The camera is part of the question, not a property of the dataset:
        # episode routing (which mp4, which span inside it) is stored per
        # camera, so switching views has to re-read the episode table.
        def read() -> dict:
            reader = _reader()
            reader.use_camera(camera)
            return reader.summary()

        try:
            return await asyncio.to_thread(read)
        except ImportError as err:
            # See /api/vla/dataset: ImportError alone, because vla_data.py's
            # own sentences are Refusals now and a library's OSError is not
            # one of them.
            return _missing_reader_dep(err)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/vla/episodes")

    @app.get("/api/vla/frame")
    async def vla_frame(episode: int = 0, t: int = 0, camera: str | None = None):
        def read():
            reader = _reader()
            reader.use_camera(camera)
            return reader.frame(episode, t)

        try:
            sample = await asyncio.to_thread(read)
            return asdict(sample)
        except ImportError as err:
            # See /api/vla/dataset: ImportError alone, because vla_data.py's
            # own sentences are Refusals now and a library's OSError is not
            # one of them.
            return _missing_reader_dep(err)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/vla/frame")

    @app.post("/api/vla/analyse")
    async def vla_analyse(req: VLAAnalyseRequest):
        def run() -> dict:
            rgb = _reader().raw_frame(req.episode, req.t)
            return app.state.vla.analyse(rgb, key=(req.episode, req.t))

        try:
            return await asyncio.to_thread(run)
        except ImportError as err:
            # See /api/vla/dataset: ImportError alone, because vla_data.py's
            # own sentences are Refusals now and a library's OSError is not
            # one of them.
            return _missing_reader_dep(err)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/vla/analyse")

    @app.get("/api/vla/attention/meta")
    def vla_attention_meta() -> dict:
        return app.state.vla.attention_meta()

    @app.get("/api/vla/attention")
    async def vla_attention(layer: int = 0, head: int = -1):
        try:
            return await asyncio.to_thread(app.state.vla.attention, layer, head)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/vla/attention")

    @app.post("/api/otel/v1/traces")
    async def otel_ingest(request: Request):
        """Accept an OTLP/HTTP JSON export, so an already-instrumented team
        does not need this project to write a provider integration.

        Point any OpenTelemetry exporter at it:

            OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:5900/api/otel
            OTEL_EXPORTER_OTLP_PROTOCOL=http/json

        JSON only. OTLP's common wire format is protobuf, and reading it would
        cost either a generated stub set or the OpenTelemetry SDK as a
        dependency -- which `modelmri-record` exists without on purpose. A
        protobuf body is refused with a sentence naming the limit rather than
        mis-parsed into a trace that looks real.
        """
        content_type = (request.headers.get("content-type") or "").split(";")[0].strip()
        if content_type == "application/x-protobuf":
            return JSONResponse(
                {
                    "error": (
                        "this endpoint reads OTLP/HTTP with a JSON body and "
                        "does not speak protobuf. Set "
                        "OTEL_EXPORTER_OTLP_PROTOCOL=http/json on the "
                        "exporter, or put a collector in front that converts. "
                        "Protobuf would mean a generated stub set or the "
                        "OpenTelemetry SDK as a dependency, and this package "
                        "is stdlib-only where it counts."
                    )
                },
                status_code=415,
            )
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse(
                {"error": "this request body is not JSON"}, status_code=422
            )
        try:
            doc = await asyncio.to_thread(otel.ingest, payload)
            trace_id = await asyncio.to_thread(traces.import_trace, doc)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/otel/v1/traces")
        # partialSuccess with rejectedSpans 0 is what an OTLP client expects
        # from a healthy collector, so an exporter pointed here sees a normal
        # answer rather than a shape it has to special-case.
        return {
            "partialSuccess": {},
            "id": trace_id,
            "spans": len(doc.get("steps") or []),
            "semconv": (doc.get("meta") or {}).get("semconv"),
        }

    @app.post("/api/traces/import")
    async def traces_import(doc: dict):
        try:
            trace_id = await asyncio.to_thread(traces.import_trace, doc)
            return {"id": trace_id}
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/traces/import")

    @app.get("/api/traces")
    def traces_list() -> list[dict]:
        return traces.list_traces()

    @app.delete("/api/traces/{trace_id}")
    def trace_delete(trace_id: str):
        """Remove one recording. A flight recorder you cannot clear fills up
        with other people's test runs, which is what happened here."""
        if not traces.delete(trace_id):
            return JSONResponse({"error": "no such trace"}, status_code=404)
        return {"deleted": trace_id}

    @app.delete("/api/traces")
    def traces_clear(keep_demo: bool = True):
        """Clear recordings. Keeps the shipped sample by default, since the
        docs point at it."""
        return {"deleted": traces.clear(keep_demo=keep_demo)}

    # BEFORE /api/traces/{trace_id}. FastAPI matches in definition order,
    # so with this below it the literal path `search` was captured as a
    # trace id and every query answered "trace not found".
    @app.get("/api/telemetry")
    async def read_telemetry():
        """What the last generation cost, with the introspection cost split out.

        Every local runner shows tokens/sec. None of them shows what being
        looked at costs — and ModelMRI is slower than Ollama for a specific
        reason: it forces eager attention and `output_attentions=True`, which
        materialises `n_layers x n_heads x S x S`. At 4,096 tokens on a
        12-layer, 12-head model that is 4.8 GB, larger than the weights.

        204 before anything has been generated, because a bar full of zeros is
        a claim about a run that never happened.
        """
        t = getattr(runtime, "last_telemetry", None)
        if t is None:
            return JSONResponse(
                {"available": False, "reason": "nothing has been generated yet"},
                status_code=200,
            )
        return {"available": True, **t.to_dict()}

    @app.get("/api/gguf")
    async def read_gguf(path: str):
        """What is inside a GGUF, without loading it or touching the GPU.

        The scanner has always found these and then refused them. It still
        cannot RUN one — a quantised GGUF is not something transformers loads —
        but that is a different claim from having nothing to say about it.
        Reads the header only: a few hundred milliseconds and well under a
        megabyte for a multi-gigabyte model.

        Bits-per-weight is computed per tensor from the file's own table, so
        the answer is arithmetic rather than the preset name every other runner
        shows. Measured on Ollama's own blobs: a qwen3-0.6B labelled Q4_K
        reads 5.499 bpw effective, because a third of its bytes are in Q6_K
        and F16.
        """
        try:
            # The same boundary the adapter loader uses, not a second one: a
            # path arriving from a browser is only read if it sits under a
            # root this server was already told about. `resolve_under_roots`
            # raises AdapterError with the sentence to show.
            target = await asyncio.to_thread(custom.resolve_under_roots, path)
            return await asyncio.to_thread(lambda: gguf_read.read(target).to_dict())
        except AdapterError as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/gguf")

    @app.get("/api/gguf/plan")
    async def plan_gguf(path: str, dtype: str | None = None):
        """What loading this GGUF would cost, before loading it.

        The header only — a few hundred kilobytes of a multi-gigabyte file.
        Two figures, because they fail at different moments: the resident size
        (parameters x dtype bytes, which has to sit on the device) and the
        float32 transit (parameters x 4, which has to fit in host RAM while
        the dequantiser runs). Neither is the file size, and the file size is
        what people budget against.
        """
        try:
            target = await asyncio.to_thread(custom.resolve_under_roots, path)
            return await asyncio.to_thread(
                lambda: runtime.plan_gguf(str(target), dtype=dtype)
            )
        except AdapterError as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/gguf/plan")

    @app.post("/api/gguf/load")
    async def load_gguf(body: GgufLoad):
        """Load a GGUF as a full torch module, so every panel works on it.

        Expensive and refusable. The Ollama backend runs the same file faster
        and at its real bit width but can show you nothing; this dequantises
        it into an ordinary module the lens and the ablation sweep can reach.
        `confirm` overrides a tight fit and only a tight fit — "will not fit"
        is that the RAM needed exceeds the RAM that exists.
        """
        try:
            target = await asyncio.to_thread(custom.resolve_under_roots, body.path)
            st = await asyncio.to_thread(
                lambda: runtime.load_gguf(
                    str(target), dtype=body.dtype, confirm=body.confirm
                )
            )
            return st.to_dict()
        except AdapterError as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/gguf/load")

    @app.post("/api/quantdiff/behaviour")
    async def quantdiff_behaviour(body: QuantCompare):
        """What quantisation cost the model's behaviour, on one prompt.

        Expensive: it loads two models, one after the other, and UNLOADS
        whatever is currently held to make room. Per-position KL between the
        two next-token distributions, every position where the argmax flipped
        with both candidates, and per-layer attention divergence.

        One prompt is one sample. The prompt is in the response for that
        reason, and the per-position series is returned whole rather than
        averaged — an average hides the one position where the answer changed,
        which is the position the feature exists to find.
        """
        try:
            q = await asyncio.to_thread(custom.resolve_under_roots, body.quantised)
            # The original may be a local checkpoint directory or a hub id, and
            # which one is decided by SHAPE -- never by asking the filesystem.
            #
            # This used to be `if Path(original).exists()`, which cannot be
            # asked about caller-supplied text without answering it: an
            # existing path took the roots gate and got "outside the
            # directories", a missing one fell through to the hub and got
            # something else, so anyone able to call this route could test for
            # the existence of any file on the machine. CodeQL called it
            # uncontrolled data in a path expression and was right.
            #
            # Everything that is not hub-id-shaped now goes through the gate
            # unconditionally, so both answers are the same sentence.
            original = body.original
            if not behavdiff.is_hub_id(original):
                original = str(
                    await asyncio.to_thread(custom.resolve_dir_under_roots, original)
                )
            return await asyncio.to_thread(
                lambda: runtime.compare_quantisation(
                    str(q), original, body.prompt, want_attention=body.attention
                )
            )
        except AdapterError as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/quantdiff/behaviour")

    @app.get("/api/traces/search")
    async def search_traces(q: str = "", limit: int = 100):
        """Full-text search over every recorded step on this machine.

        Free text plus allow-listed filters — `kind:tool_call`, `error:true`,
        `duration>2000`, `name:pytest`. Results are steps rather than runs,
        because what somebody is looking for is the tool call that failed, not
        the hour it happened in.

        The response names the engine that answered. FTS5 is compiled into
        essentially every CPython SQLite, which is what makes this a `pip
        install` rather than a ClickHouse container — but "essentially every"
        is not "every", and a build without it degrades to a substring scan
        and says so instead of quietly becoming a different feature.
        """
        try:
            return await asyncio.to_thread(traces.search, q, min(int(limit), 500))
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/traces/search")

    @app.get("/api/traces/{trace_id}")
    def trace_get(trace_id: str):
        from . import ledger as ledger_mod

        doc = traces.get_trace(trace_id)
        if doc is None:
            return JSONResponse({"error": "trace not found"}, status_code=404)
        steps = doc.get("steps") or []
        doc["tokens"] = ledger_mod.roll_up(steps).to_dict()
        doc["tokens_by_step"] = {
            sid: roll.to_dict()
            for sid, roll in ledger_mod.subtree_rollups(steps).items()
        }
        # A price file that cannot be read is reported as a field, not raised:
        # the token counts above are complete and useful without it, and
        # failing the whole trace view over a typo in MODELMRI_PRICES would
        # take away the half that works.
        try:
            doc["cost"] = ledger_mod.bill(steps, ledger_mod.load_prices()).to_dict()
        except BadRequest as err:
            doc["cost"] = {"error": err.sentence, "means": err.sentence}
        return doc

    # An Inspect `.eval` log is a zip, so this takes bytes rather than a path:
    # the panel drops a file onto it and the server never learns where on the
    # reader's disk that file lives.
    _EVAL_LIMIT = 200_000_000

    @app.post("/api/traces/import/inspect")
    async def import_inspect(request: Request, sample_id: str = ""):
        from . import inspect_io

        data = await request.body()
        if len(data) > _EVAL_LIMIT:
            return JSONResponse(
                {
                    "error": f"that log is larger than "
                    f"{_EVAL_LIMIT // 1_000_000} MB. Read it with Inspect's "
                    f"own viewer, or split the eval."
                },
                status_code=413,
            )

        def run() -> dict:
            import tempfile

            # Written to a temp file because `zipfile` wants a seekable
            # source and holding a 200 MB archive in memory to read one
            # sample out of it is the opposite of what the lazy reader is
            # for. Deleted whatever happens.
            #
            # `ignore_cleanup_errors` because this runs on Windows, where a
            # still-open handle makes the cleanup itself raise — which would
            # fail a request whose work had already succeeded, throwing away
            # a correct answer over a housekeeping error. The directory is
            # temp; the OS reclaims it either way.
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                path = Path(tmp) / "upload.eval"
                path.write_bytes(data)
                head = inspect_io.header(path)
                out = inspect_io.read_sample(path, sample_id=sample_id)
                stored = traces.import_trace(out.trace)
                refs = inspect_io.samples(path)
                return out.to_dict() | {
                    "trace_id": stored,
                    "header": head.to_dict(),
                    "samples": [s.to_dict() for s in refs],
                    # BOTH numbers. The list is capped at
                    # `MAX_SAMPLES_LISTED`, and the panel was printing its
                    # length as the log's sample count — so a 6,000-sample
                    # file read "5000 samples" as a fact about the reader's
                    # own archive, with the dropdown holding an arbitrary
                    # subset in zip order and a later sample simply
                    # unselectable.
                    "samples_total": refs.n_total,
                    "samples_truncated": refs.truncated,
                }

        try:
            return await asyncio.to_thread(run)
        except BadRequest as err:
            # `InspectError` is one of these: an unrecognised schema version,
            # a file that is not a zip, a sample the log does not carry.
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/traces/import/inspect")

    @app.get("/api/traces/{trace_id}/bundle/preview")
    def bundle_preview(trace_id: str):
        """What a shared bundle would contain — BEFORE it is written.

        A share button that ships a file without showing what is in it asks
        the user to trust a process they cannot see, and this is the one path
        in the project where data leaves the machine.
        """
        from . import bundle as bundle_mod

        doc = traces.get_trace(trace_id)
        if doc is None:
            return JSONResponse({"error": "trace not found"}, status_code=404)
        try:
            # The prompt as the runtime holds it. Absent when nothing has been
            # generated yet, which is fine — the preview is about the trace
            # half either way, and `prepare` scans "" harmlessly.
            return bundle_mod.preview(
                doc, prompt=getattr(runtime, "last_prompt", "") or ""
            ).to_dict()
        except BadRequest as err:
            # `BundleError` is one of these, so a run too long to ship answers
            # 422 through the same path as every other authored refusal.
            return JSONResponse({"error": err.sentence}, status_code=422)

    @app.get("/api/traces/{trace_id}/patterns")
    def trace_patterns(trace_id: str, window_ms: int = 0):
        """Structural findings for one run. No model, no network, no key."""
        from . import patterns as patterns_mod

        doc = traces.get_trace(trace_id)
        if doc is None:
            return JSONResponse({"error": "trace not found"}, status_code=404)
        return patterns_mod.analyse(
            doc.get("steps") or [],
            window_ms=window_ms or patterns_mod.DEFAULT_RETRY_WINDOW_MS,
        ).to_dict()

    # ---------------------------------------------------------- OpenAI /v1
    #
    # So ModelMRI can be dropped into any client that speaks `/v1` — and so
    # that client can ask for the internals of the completion it just got,
    # which no other `/v1` can return.

    @app.get("/v1/models")
    def v1_models():
        from . import openai_api

        return openai_api.models_payload(runtime)

    async def _v1_complete(body: dict, *, chat: bool):
        from . import openai_api

        openai_api.check_parameters(body)
        # BEFORE the stream branch, and that placement is the fix.
        #
        # `runtime.generate_stream` raises this same Refusal, but on the
        # streaming path it does so inside the generator — which FastAPI only
        # starts consuming after `StreamingResponse` has already sent 200 and
        # `text/event-stream`. MEASURED with nothing loaded: `{"stream": true}`
        # returned 200 with a body of ZERO BYTES — no `data:` frame, no
        # `[DONE]` — which an OpenAI client reads as a successful empty
        # completion. Without the flag the same request answered 409 with a
        # sentence naming the fix. One question, two answers, decided by a
        # field that has nothing to do with whether a model is resident.
        if not runtime.loaded:
            raise Refusal("No model loaded. POST /api/model/load first.")
        prompt = openai_api.build_prompt(runtime, body)
        model_name = str(body.get("model") or getattr(runtime, "hf_id", "") or "local")
        max_tokens = int(
            body.get("max_completion_tokens")
            or body.get("max_tokens")
            or openai_api.DEFAULT_MAX_TOKENS
        )
        # Somebody else's client sends this body, so a wrong type here is
        # ordinary rather than exotic — and an OpenAI-compatible surface that
        # answers 500 to a bad `temperature` is one nobody can debug against.
        try:
            temperature = _real(body, "temperature", 0.7)
            top_k = _whole(body, "top_logprobs", 0)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        want_logprobs = bool(body.get("logprobs"))
        ask = body.get("modelmri") if isinstance(body.get("modelmri"), dict) else None

        def generate() -> str:
            return "".join(
                runtime.generate_stream(prompt, max_tokens, temperature, commit=True)
            )

        if not body.get("stream"):
            text = await asyncio.to_thread(generate)
            logprobs = (
                await asyncio.to_thread(openai_api.token_logprobs, runtime, top_k)
                if want_logprobs
                else None
            )
            extension = (
                await asyncio.to_thread(
                    openai_api.internals, runtime, ask, app.state.mri_store
                )
                if ask
                else None
            )
            telemetry = getattr(runtime, "last_telemetry", None)
            return JSONResponse(
                openai_api.completion_payload(
                    text,
                    model_name,
                    chat=chat,
                    prompt_tokens=int(getattr(telemetry, "prompt_tokens", 0) or 0),
                    completion_tokens=int(
                        getattr(telemetry, "generated_tokens", 0) or 0
                    ),
                    logprobs=logprobs,
                    extension=extension,
                )
            )

        # Streaming. The generator is a blocking iterator, so it is consumed
        # off the event loop the same way `/api/generate` does it.
        async def frames():
            first = True
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()
            done = object()

            def pump():
                # `generate_stream` is a BLOCKING iterator; consuming it on
                # the event loop would stall every other request for the
                # length of the generation. The sentinel goes through
                # `finally` so a failure inside the generator still ends the
                # stream instead of hanging the client forever.
                try:
                    for piece in runtime.generate_stream(
                        prompt, max_tokens, temperature, commit=True
                    ):
                        loop.call_soon_threadsafe(queue.put_nowait, piece)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, done)

            task = asyncio.create_task(asyncio.to_thread(pump))
            while True:
                piece = await queue.get()
                if piece is done:
                    break
                yield openai_api.chunk_payload(
                    piece, model_name, chat=chat, first=first
                )
                first = False
            # Re-raises whatever the generator raised, so a mid-stream failure
            # surfaces rather than ending as a clean [DONE].
            await task
            extension = (
                await asyncio.to_thread(
                    openai_api.internals, runtime, ask, app.state.mri_store
                )
                if ask
                else None
            )
            yield openai_api.final_chunk(model_name, chat=chat, extension=extension)

        return StreamingResponse(frames(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def v1_chat(body: dict):
        try:
            return await _v1_complete(body, chat=True)
        except Refusal as err:
            return JSONResponse({"error": {"message": str(err)}}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": {"message": str(err)}}, status_code=400)
        except Exception as err:
            return _internal(err, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def v1_completions(body: dict):
        try:
            return await _v1_complete(body, chat=False)
        except Refusal as err:
            return JSONResponse({"error": {"message": str(err)}}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": {"message": str(err)}}, status_code=400)
        except Exception as err:
            return _internal(err, "/v1/completions")

    @app.get("/v1/mri/{mri_id}")
    async def v1_mri(mri_id: str):
        """A `.mri` this server minted for a `/v1` completion.

        410 for one that WAS held and has been evicted, 404 for one that never
        existed. Collapsing those into a single 404 sends a client to debug the
        wrong thing: "ask again, sooner" and "you have the wrong id" have
        different fixes.
        """
        store = app.state.mri_store
        blob = store.get(mri_id)
        if blob is None:
            if store.was_evicted(mri_id):
                return JSONResponse(
                    {
                        "error": {
                            "message": (
                                f"{mri_id} was held and has been evicted — this "
                                f"server keeps only the last {store.limit}, in "
                                f"memory. Ask for another with "
                                f'\'{{"modelmri": {{"mri": true}}}}\' and fetch '
                                f"it before the run moves on."
                            )
                        }
                    },
                    status_code=410,
                )
            return JSONResponse(
                {
                    "error": {
                        "message": (
                            f"no `.mri` with id {mri_id!r}. This server has "
                            f"never issued that id — ids come back in the "
                            f"`modelmri.mri.id` field of a completion asked "
                            f'for with \'{{"modelmri": {{"mri": true}}}}\'.'
                        )
                    }
                },
                status_code=404,
            )
        return Response(
            content=blob,
            media_type="application/gzip",
            headers={
                "Content-Disposition": f'attachment; filename="{mri_id}.mri"',
            },
        )

    @app.post("/api/judge")
    async def judge_score(body: JudgeRequest):
        """Score a rubric by reading the loaded model's probability mass.

        One forward pass per paraphrase, no generation. Refuses when the model
        did not put enough mass on the verdict tokens to have answered at all,
        which is a 409 like every other deliberate no.
        """
        from . import judge as judge_mod

        def run() -> dict:
            model = getattr(runtime, "model", None)
            tokenizer = getattr(runtime, "tokenizer", None)
            if model is None or tokenizer is None:
                raise Refusal(
                    "No model is loaded to judge with. This reads the "
                    "probability mass of the model on THIS machine — there is "
                    "no hosted judge behind it."
                )
            out = judge_mod.score(
                model,
                tokenizer,
                body.text,
                body.rubric,
                n_paraphrases=body.n_paraphrases,
                device=str(getattr(runtime, "device", "cpu")),
            )
            return out.to_dict()

        try:
            return await asyncio.to_thread(run)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/judge")

    @app.post("/api/judge/plan")
    def judge_plan(body: JudgeRequest):
        """The prompts that would be run, before any of them is."""
        from . import judge as judge_mod

        try:
            prompts = judge_mod.plan(body.text, body.rubric, body.n_paraphrases)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        return {"prompts": prompts, "n_passes": len(prompts)}

    @app.post("/api/rubric/score")
    async def rubric_score(body: RubricScoreRequest, limit: int = 500):
        """Score every recorded run against exact predicates. No model."""
        from . import rubric as rubric_mod

        def run() -> dict:
            # `body.rules`, not `body.get("rules", body)`. That fallback handed
            # the WHOLE request to the parser when the key was missing, and
            # `rubric.parse` reads `.get("rules", [])` off a dict — so any body
            # without the key became an empty rubric and scored nothing, at 200.
            rules = rubric_mod.parse(body.rules)
            available = len(traces.list_traces())
            report = rubric_mod.score(traces.all_traces_with_steps(limit), rules)
            out = report.to_dict()
            # The cap, stated. A rubric answered over 500 of 4,000 runs is a
            # different claim from one answered over all of them, and
            # `slowest_percent` is a claim ABOUT the set it was measured on.
            out["n_traces_available"] = available
            out["truncated"] = max(0, available - report.n_traces)
            return out

        try:
            return await asyncio.to_thread(run)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/rubric/score")

    # ------------------------------------------------------------- scorers
    #
    # 82 KB of tested module with no way to reach it. Every metric here needs
    # NO model — which is the whole argument for them: a scorer that asks a
    # language model whether an answer is good needs a calibration gate
    # nobody ships, costs a key and a round trip per row, and gives a
    # different answer next Tuesday. These are arithmetic and they are
    # reproducible with the network off.

    # -------------------------------------------------------------- sweeps
    #
    # `sweep.save` has existed since the sweep did and nothing ever read it
    # back — a saved sweep was write-only, sitting in the database and
    # unreachable from the tool that wrote it.

    @app.get("/api/sweeps")
    async def sweeps_list(limit: int = 50) -> dict:
        """Every sweep saved on this machine, and how far each one got."""
        from . import sweep as sweep_mod

        rows = await asyncio.to_thread(sweep_mod.saved_sweeps, limit)
        unfinished = [r for r in rows if not r["complete"]]
        return {
            "sweeps": rows,
            "means": (
                f"{len(rows)} saved sweep(s), {len(unfinished)} of them "
                f"unfinished. A sweep that stopped keeps the prompts it "
                f"already measured, so finishing one costs only what is left."
            ),
        }

    @app.get("/api/sweeps/{sweep_id}/resume")
    async def sweeps_resume_plan(sweep_id: str) -> dict:
        """What finishing a stopped sweep would cost, and whether it may run.

        Three things make a resume WRONG rather than merely expensive, and all
        three are checked here before anything is spent — see
        `sweep._resumable`. `blocked` is a sentence, never a warning to
        override.
        """
        from . import sweep as sweep_mod

        try:
            return await asyncio.to_thread(
                sweep_mod.resume_plan, sweep_id, app.state.runtime
            )
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": err.sentence}, status_code=code)

    @app.post("/api/sweeps/{sweep_id}/resume")
    async def sweeps_resume(sweep_id: str):
        """Finish a stopped sweep, keeping every prompt already measured.

        `sweep.resume` was written, tested and PRICED — `GET` on this same path
        answers what finishing would cost — and then had no way to run: no
        route, no CLI flag, no button. The panel rendered "Nothing blocks this
        resume… Finishing it costs only the N prompt(s) below" beside no
        control at all.

        The reachable case is not a crash, which leaves nothing saved because
        `save()` runs after `run()` returns. It is REFUSALS: `remaining()`
        counts every unmeasured row as still-to-run, so any sweep containing a
        prompt the model refused is listed as unfinished forever, prices
        cleanly, and could never be finished by any surface.

        `resume` re-checks `_resumable` itself, so the price and the run cannot
        disagree about whether it may proceed — this route does not re-derive
        that judgement.
        """
        from . import sweep as sweep_mod

        def finish():
            job, rows = sweep_mod.resume(sweep_id, app.state.runtime)
            stats = sweep_mod.aggregate(rows, metric=job.metric)
            measured = sum(1 for r in rows if r.measured)
            return {
                "sweep_id": sweep_id,
                "model": job.model,
                "metric": job.metric,
                "rows": [r.to_dict() for r in rows],
                "stats": [st.to_dict() for st in stats],
                "n_prompts": len(rows),
                "n_measured": measured,
                # REPORTED rather than implied by a shorter list. A prompt the
                # model refused stays unmeasured after a resume, which is the
                # whole reason a sweep can be listed as unfinished forever.
                "n_unmeasured": len(rows) - measured,
                "means": (
                    f"{measured} of {len(rows)} prompt(s) measured for "
                    f"{job.model}."
                    + (
                        ""
                        if measured == len(rows)
                        else (
                            f" {len(rows) - measured} could not be measured and "
                            f"are absent from the ranking rather than scored "
                            f"zero — resuming again will retry them."
                        )
                    )
                ),
            }

        try:
            return await asyncio.to_thread(finish)
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": err.sentence}, status_code=code)
        except Exception as err:
            return _internal(err, "/api/sweeps/{sweep_id}/resume")

    @app.get("/api/scorers")
    def scorers_catalogue() -> dict:
        """Every scorer, WITH the way each one lies.

        `failure_mode` is not documentation garnish, it is the field that
        makes the catalogue honest: `contains_all` finds `kill` inside
        `skill`, `edit_similarity` counts characters rather than meaning, and
        `json_valid` is happy with `{}`. A catalogue whose entries state how
        they fail is a different product from one whose entries state what
        they measure.
        """
        from . import scorers

        rows = scorers.catalogue()
        return {
            "scorers": rows,
            "means": (
                f"{len(rows)} scorers, none of which asks a model anything. "
                f"Each carries the way it FAILS as well as what it measures, "
                f"because a number from a metric whose blind spot you do not "
                f"know is a number you cannot act on."
            ),
        }

    @app.post("/api/scorers/run")
    async def scorers_run(req: ScorerRunRequest):
        """One scorer over one output. Deterministic, and offline."""
        from . import scorers

        try:
            result = await asyncio.to_thread(
                scorers.run,
                req.name,
                output=req.output,
                reference=req.reference,
                **(req.options or {}),
            )
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": err.sentence}, status_code=code)
        except Exception as err:
            return _internal(err, "/api/scorers/run")
        return result.to_dict()

    @app.post("/api/traces/dataset")
    async def traces_to_dataset(req: TraceToDatasetRequest):
        """Turn recorded runs into a dataset, closing the loop they leave.

        You watch an agent go wrong and there is currently no way to make that
        a case you re-run after changing something. Curation needs no model at
        all, which is why this happens offline and in milliseconds.

        Nothing is dropped silently: a run with no prompt-bearing step, a run
        whose prompt repeats an earlier one, and every non-error run when only
        failures were asked for each come back in `skipped` with the reason.
        """
        from . import datasets

        store = app.state.traces

        def _build():
            docs = []
            missing = []
            for trace_id in req.trace_ids:
                doc = store.get_trace(str(trace_id))
                if doc is None:
                    missing.append(str(trace_id))
                else:
                    docs.append(doc)
            data, report = datasets.from_traces(
                docs,
                name=req.name,
                only_errors=req.only_errors,
                description=req.description,
            )
            # A id that is not in the store is its own kind of absence, and it
            # is the caller's mistake rather than the recording's — so it is
            # reported separately from the runs that were read and left out.
            report["not_found"] = missing
            return data, report

        try:
            data, report = await asyncio.to_thread(_build)
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": err.sentence}, status_code=code)
        except Exception as err:
            return _internal(err, "/api/traces/dataset")

        return {"dataset": data.to_dict(), **report}

    # --------------------------------------------------------- experiments
    #
    # The abstraction every observability platform in the competitor analysis
    # shares, and the one this had built and could not reach. `sweep` runs one
    # metric over many prompts; `diff` compares two `.mri` of the SAME prompt.
    # Neither answers "did my edit help on the 40 cases I care about".

    @app.post("/api/experiments/compare")
    async def experiments_compare(req: ExperimentCompareRequest, request: Request):
        """Two runs of one dataset, case by case. Counts and deltas, no verdict.

        Every competitor's experiment row holds an output and a score. This one
        holds the output, the score, AND the internals that produced it — so a
        regression row can say the patching site flipped sign, rather than only
        that a number moved.
        """
        from . import datasets

        # Three paths from a request body, so the same guard the other
        # file-reading routes carry. A path names a file on the disk THIS
        # server runs on.
        refusal = _not_from_this_machine(
            request,
            "reading experiment files",
            because=(
                "an experiment is a file on the disk this server is running "
                "on, not on yours — run ModelMRI where the runs live"
            ),
        )
        if refusal is not None:
            return refusal

        def _run():
            before = datasets.read_experiment(req.before)
            after = datasets.read_experiment(req.after)
            # `None` is not "no dataset": it means nothing looked, and the
            # comparison reports `references: null` rather than 0 so the two
            # stay distinguishable downstream.
            data = datasets.read_dataset(req.dataset) if req.dataset else None
            return datasets.compare_experiments(
                before,
                after,
                metric=req.metric,
                higher_is_better=req.higher_is_better,
                dataset=data,
                floor=req.floor,
                top_k=req.top_k,
            )

        try:
            return (await asyncio.to_thread(_run)).to_dict()
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": err.sentence}, status_code=code)
        except Exception as err:
            return _internal(err, "/api/experiments/compare")

    # ---------------------------------------------------------- trajectory

    @app.get("/api/trajectory/cost")
    def trajectory_cost(reference: int = 0, candidate: int = 0):
        """What aligning two trajectories of these lengths would build.

        Priced first, like every other table in this tool: the alignment is a
        `reference x candidate` grid, and a caller can shorten the span
        themselves rather than discovering the cap by hitting it.
        """
        from . import trajectory

        try:
            return trajectory.plan_comparison(reference, candidate)
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": err.sentence}, status_code=code)

    @app.post("/api/trajectory/compare")
    async def trajectory_compare(req: TrajectoryCompareRequest):
        """What a run did against what it was supposed to do.

        Everybody else scores this with a language-model judge. It is a
        sequence alignment — exact, offline, milliseconds — and it reports
        counts rather than a verdict: two steps missing, one extra, three with
        changed arguments. Never "Plan Adherence 0.71", because a shorter path
        is not a worse path and a number would say it was.
        """
        from . import trajectory

        try:
            found = await asyncio.to_thread(
                trajectory.align, reference=req.reference, candidate=req.candidate
            )
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": err.sentence}, status_code=code)
        except Exception as err:
            return _internal(err, "/api/trajectory/compare")
        return found.to_dict()

    @app.get("/api/rubric")
    def rubric_list():
        return traces.rubrics()

    @app.post("/api/rubric")
    def rubric_save(body: RubricSaveRequest):
        from . import rubric as rubric_mod

        name = body.name.strip()
        if not name:
            return JSONResponse(
                {"error": "a saved rubric needs a name."}, status_code=422
            )
        try:
            rules = rubric_mod.parse(body.rules)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        traces.save_rubric(name, rules)
        return {"name": name, "n_rules": len(rules)}

    @app.delete("/api/rubric/{name}")
    def rubric_delete(name: str):
        return {"deleted": traces.delete_rubric(name)}

    @app.get("/api/patterns/across")
    def patterns_across(name: str = "", limit: int = 50):
        """The same structural finding, counted over many recorded runs.

        `name` narrows to one agent by trace name — a pattern that appears in
        12 of 19 runs of the SAME agent is a different claim from one seen
        across 19 unrelated runs.
        """
        from . import patterns as patterns_mod

        summaries = traces.list_traces()
        if name:
            summaries = [s for s in summaries if str(s.get("name") or "") == name]
        # Capped, and the cap is reported — a truncated sweep whose "12 of 19"
        # silently means "12 of the 19 newest" is a different number.
        chosen = summaries[: max(1, min(int(limit or 50), 500))]
        docs = [d for d in (traces.get_trace(str(s.get("id"))) for s in chosen) if d]
        found = [r.to_dict() for r in patterns_mod.across_runs(docs)]
        return {
            "findings": found,
            "n_runs": len(docs),
            "n_runs_available": len(summaries),
            "truncated": max(0, len(summaries) - len(docs)),
            "name": name,
        }

    @app.get("/api/attention")
    async def attention(layer: int = 0, head: int = 0, variant: str = "live"):
        try:
            return await asyncio.to_thread(runtime.attention, layer, head, variant)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/attention")

    @app.get("/api/attention/diff")
    async def attention_diff(
        layer: int = 0, head: int = 0, a: str = "live", b: str = "steered"
    ):
        """`a` minus `b` for one head, over one token sequence.

        Both sides are forward passes over the same generation, so index i is
        the same token in both by construction — the only arrangement in
        which subtracting two attention matrices means anything.
        """
        try:
            return await asyncio.to_thread(runtime.attention_diff, layer, head, a, b)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/attention/diff")

    @app.get("/api/attention/ablate")
    async def ablate_heads(
        layer: int | None = None, baseline: str = "zero", scope: str = "layer"
    ):
        """Rank heads by how far removing one moves the next-token answer.

        `scope=layer` (default) does n_heads + 2 passes; `scope=all` does
        n_layers x n_heads + 2 — 450 for Qwen3-0.6B. The
        default is the cheap one on purpose.

        The response carries `elapsed_s` and `passes` so a caller can derive
        the rate on ITS machine. Seconds measured on mine do not transfer: the
        same model ranged 12-71 ms/pass across sessions on one RTX 4060, and
        the first ranking after a load runs several times slower than the rest
        while CUDA warms up.
        """
        target = None if scope == "all" else (layer if layer is not None else 0)
        try:
            return await asyncio.to_thread(runtime.ablate_heads, target, baseline)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/attention/ablate")

    @app.get("/api/attention/ablate/estimate")
    async def estimate_ablation(
        layer: int | None = None, baseline: str = "zero", scope: str = "layer"
    ):
        """What this sweep would cost on THIS machine, before it is started.

        The pass count is exact and portable; the seconds and the peak memory
        are projected from one probe pass here and are labelled as one sample.
        Matters most for `baseline=resample`, which is `RESAMPLE_DRAWS` times
        the work — one layer goes from n_heads + 2 passes to
        n_heads * RESAMPLE_DRAWS + 2.
        """
        target = None if scope == "all" else (layer if layer is not None else 0)
        try:
            return await asyncio.to_thread(runtime.estimate_ablation, target, baseline)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/attention/ablate/estimate")

    @app.post("/api/traces/{trace_id}/steps/{step_id}/adopt")
    async def adopt_step(trace_id: str, step_id: str):
        """Open a recorded agent step in the mechanistic panels.

        The join no hosted tracing platform can build: they stop at the API
        boundary and never hold the weights. A step recorded from a local model
        carries its token ids, so it can be re-established as the current
        generation and every panel — attention, lens, ablation, patching, SAE —
        reads it with no changes.

        409 when the step came from a hosted API (no weights here), when the
        wrong model is loaded, or when re-tokenising the prompt disagrees with
        what the recorder captured. That last one is the important refusal:
        near-identical ids would point every panel at a sequence the model
        never saw, and nothing downstream would notice.
        """
        try:
            doc = await asyncio.to_thread(traces.get_trace, trace_id)
            if doc is None:
                return JSONResponse({"error": "no such trace"}, status_code=404)
            step = next((s for s in doc["steps"] if s["id"] == step_id), None)
            if step is None:
                return JSONResponse({"error": "no such step"}, status_code=404)
            return await asyncio.to_thread(runtime.adopt_step, step)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/traces/adopt")

    @app.get("/api/attention/control")
    async def control_ranking(
        layer: int | None = None, baseline: str = "zero", seed: int = 0
    ):
        """The same ranking on an untrained twin of this architecture.

        Answers the question underneath every ranking in this tool: would this
        measurement have produced a confident, ordered list anyway? Builds the
        model from `config.json` alone — no weights fetched, works offline —
        seeds it, runs the identical `rank_heads` over the same tokens, and
        reports the rank correlation with the real one.

        Costs a second model in memory for the duration, and two sweeps.
        """
        try:
            return await asyncio.to_thread(
                runtime.control_ranking,
                layer if layer is not None else 0,
                baseline,
                seed,
            )
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/attention/control")

    @app.get("/api/attention/baselines")
    async def compare_baselines(layer: int | None = None):
        """Every baseline on one layer, and how much they disagree.

        Expensive on purpose — it runs all three, so it is a deliberate action
        rather than something the panel does on load. Ask
        `/api/attention/ablate/estimate?baseline=resample` first; resample
        dominates the total.
        """
        try:
            return await asyncio.to_thread(
                runtime.compare_baselines, layer if layer is not None else 0
            )
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/attention/baselines")

    @app.post("/api/patch")
    async def patch_trace(req: PatchRequest):
        """Where in the model does the answer get decided? Two prompts, one grid.

        Every other ranking here removes something from one prompt. This moves
        an activation from a run that knows the answer into a run that does
        not, at every (layer, position), and reports the share of the gap
        between the two answers that comes back. Ablation says "this
        mattered"; this says "the fact is here".

        The score is SIGNED and so it is not KL, which every other panel uses.
        Patching has a direction and a patch can push the answer further away:
        measured in float32 with "The Eiffel Tower is located in the city
        of" against "The Colosseum is located in the city of", some sites
        moved it away, and KL cannot tell those from a site that
        recovered nothing. The two rankings also disagree — the best sites by
        recovery are not the best by KL-to-clean.

        Cost is `n_layers * n_positions` passes for the grid plus
        `draws + 1` for each of the top 24 sites. Controls are eight draws,
        not one, because one is a
        coin flip — at a single site the draws can straddle zero and span more
        than the real recovery, and the gate passes several times as many sites
        on one draw as on all eight.

        422 when the pair cannot be compared, which is most casually-written
        pairs and is never visible without being told: the two prompts must
        tokenize to the SAME length (2 of 8 natural minimal pairs did not) and
        must actually predict different tokens (2 of 3 did not, making the
        denominator exactly 0). Both refusals name what to change. 409 when
        there is no live HuggingFace model to re-run.
        """
        try:
            return await asyncio.to_thread(runtime.patch_trace, req.clean, req.corrupt)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/patch")

    @app.post("/api/patch/graph")
    async def patch_graph(req: PatchGraphRequest):
        """A PATCHING graph: what wrote the thing that wrote the answer.

        `/api/patch` says WHERE the answer is carried and `path_trace` says
        what wrote into one receiver. This asks that second question again of
        the senders that survived their controls -- which is the question a
        circuit view is actually opened for, and the one neither of the others
        answers.

        NOT AN ATTRIBUTION GRAPH, and the payload says so in `means`.
        circuit-tracer's are built from transcoders, which exist for a handful
        of models and whose gemma-2-2b set does not fit 8 GB. This is a
        different object from a different measurement, built out of nothing but
        the model already loaded. `/api/graph` READS one of theirs; this
        computes one of ours.

        Every edge carries the same eight same-norm draws the node grid uses,
        and an edge that does not beat them is returned marked rather than
        dropped -- "we tested this and it did not survive" and "we never saw
        this" are different findings. The seeding rule and the prune threshold
        travel in the payload for the same reason: edge count is quadratic in
        sites, so anything drawable is a subset, and a graph whose edges were
        chosen by an undisclosed rule is a picture rather than a measurement.

        Expensive. One `path_trace` per receiver per level, each of which is
        one pass per earlier component plus its controls. 409 when there is no
        live model; 422 when the pair cannot be compared or the walk has
        nothing to start from.
        """
        try:
            return await asyncio.to_thread(
                runtime.patch_graph,
                req.clean,
                req.corrupt,
                depth=req.depth,
                max_receivers=req.max_receivers,
            )
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/patch/graph")

    @app.get("/api/attention/attribute")
    async def attribute_tokens(position: int | None = None):
        """Rank the prompt's own tokens by how far masking one moves the answer.

        The input-side companion to `/api/attention/ablate`. `position`
        defaults to the last prompt token, where the next-token distribution is
        the model's answer to the question before any of its own output feeds
        back in.

        Cost is `tested tokens + 8` forward passes: seven inside the ranking
        (a base, a repeat of it for the noise floor, a plain `model(ids)`
        agreement check, one with reversed position_ids that gates on the
        answer MOVING, index 0, one joint mask, one check that masking empties
        the column) plus one to read the model's answer at `position`.
        Measured through this endpoint: 21 on gemma-3-270m-it (13 tested), 24
        on Qwen3-0.6B at
        token 17 (16 tested). Tested tokens are capped at 64 and `truncated`
        says whether it bit. `passes_note` carries the same breakdown to the
        caller.

        `passes` and `elapsed_s` are both returned so a caller can derive a
        rate on ITS machine. Seconds measured on mine do not transfer, and on
        one RTX 4060 they do not transfer between models either: warm and
        back to back, three runs each, those three took 0.14-0.15 s,
        0.81-0.84 s and 1.00-1.03 s, or roughly 13, 39 and 42 ms a pass.

        409 when there is nothing to measure — a recording is open, the model
        is served by Ollama, nothing has been generated, the generation belongs
        to a previous model, or the model's next token at `position` is a
        control token rather than an answer. 422 when `position` is outside the
        sequence.
        """
        try:
            return await asyncio.to_thread(runtime.attribute_tokens, position)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/attention/attribute")

    # ---------------- sessions (.mri) ----------------
    #
    # A `.mri` is one analysis without the model: tokens, attention, the
    # generation and its settings. It is how you show someone the head you
    # found without asking them to download 8 GB and reproduce your prompt.

    @app.get("/api/session/state")
    def session_state() -> dict:
        return runtime.session_info()

    @app.get("/api/session/trace")
    def session_trace() -> dict:
        """The agent run carried by the open `.mri`, if it has one.

        The store at `/api/traces` holds runs somebody IMPORTED. This is the
        other source: a bundle built around a failing step carries the run
        inside it, and until this existed that run was parsed, validated, and
        then shown to nobody — the panel listed the store, found it empty, and
        said "0 recordings" over a file containing the very thing it lists.

        Nothing is written to the store. A recording is read, not adopted:
        importing it would put somebody else's run into this machine's history
        as though it had been captured here.

        `available: False` is a state, not an error. Most sessions carry no
        agent run, and the rolled-up tokens are computed by the same
        `ledger.roll_up` the store's traces go through so the two read
        identically rather than nearly.
        """
        from . import ledger as ledger_mod

        replay = runtime.replay
        if replay is None or not replay.has_trace():
            return {"available": False}
        doc = dict(replay.trace)
        # ONE STEP SHAPE, whichever source it came from. The store fills these
        # from its own columns; a `.mri` carries a deliberately smaller step,
        # so the fields it omits have to arrive as the value that MEANS
        # omitted. `null` is that value and the panel already reads it as "the
        # recorder said nothing" — while a missing key reaches the same test as
        # `undefined`, which is not null, and printed "undefined cache read"
        # next to the real counts. `seq` is positional, so it is derived here
        # rather than left blank: the inspector titles every step "step N".
        steps = [
            {
                "truncated_in": 0,
                "truncated_out": 0,
                "tokens_cache_read": None,
                "tokens_cache_write": None,
                "tokens_reasoning": None,
                # Never adoptable, and false rather than absent: the file
                # carries a run's shape and not the token ids underneath it,
                # so there is nothing for the panels to reopen.
                "adoptable": False,
                **step,
                "seq": i,
            }
            for i, step in enumerate(doc.get("steps") or [])
        ]
        doc["steps"] = steps
        doc["tokens"] = ledger_mod.roll_up(steps).to_dict()
        doc["tokens_by_step"] = {
            sid: roll.to_dict()
            for sid, roll in ledger_mod.subtree_rollups(steps).items()
        }
        # Same treatment as `/api/traces/{id}`: an unreadable price file is a
        # field on the answer, not an exception that takes the run down with
        # it. The token counts above stand without prices.
        try:
            doc["cost"] = ledger_mod.bill(steps, ledger_mod.load_prices()).to_dict()
        except BadRequest as err:
            doc["cost"] = {"error": err.sentence, "means": err.sentence}
        doc["available"] = True
        return doc

    @app.get("/api/graph")
    def graph() -> dict:
        """An attribution graph carried by the open `.mri`, if it has one.

        The same shape the viewer's own shim answers, so a graph looks
        identical whether it is opened in the app or in the zero-install
        viewer. `available: False` is a state, not an error: most sessions
        have no graph.

        Provenance is checked here too, not only at parse. A graph rendered
        under ModelMRI's chrome without saying who computed it is the
        confusion this whole feature exists to prevent, and the guard belongs
        on every path that can reach a screen.
        """
        replay = runtime.replay
        g = getattr(replay, "graph", None) if replay is not None else None
        if not g or not g.get("n_nodes"):
            return {"available": False}
        claim = (g.get("provenance") or {}).get("measured_by")
        # A non-empty string, not merely truthy: `true` passes a truthiness
        # test and renders as nothing.
        if not (isinstance(claim, str) and claim.strip()):
            return {
                "available": False,
                "error": (
                    "this session carries an attribution graph with no "
                    "provenance, so it is not rendered."
                ),
            }
        return {
            "available": True,
            "n_nodes": g["n_nodes"],
            "edges": g.get("edges") or [],
            "provenance": g["provenance"],
            "prompt": g.get("prompt") or "",
            "summary": g.get("summary") or {},
            "notes": g.get("notes") or [],
        }

    @app.get("/api/session/export")
    async def session_export(
        layer: int = 0,
        head: int = 0,
        note: str = "",
        trace_id: str = "",
        step_ref: str = "",
    ):
        # The agent run, when the caller asked for one. Fetched here rather
        # than inside the runtime so the export path keeps knowing nothing
        # about the trace store.
        run = None
        if trace_id:
            run = traces.get_trace(trace_id)
            if run is None:
                return JSONResponse(
                    {"error": f"no trace {trace_id!r} to bundle"}, status_code=404
                )
        try:
            blob = await asyncio.to_thread(
                runtime.export_session, layer, head, note, run, step_ref
            )
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/session/export")
        # The basename, sanitised -- never the id itself. `hf_id` is not always
        # a Hub id: load a model from a local folder and it is an absolute path,
        # so the header carried the whole path, where backslashes are
        # quoted-string escapes. Worse, Starlette encodes header values as
        # latin-1, so a Cyrillic or CJK username raised UnicodeEncodeError and
        # the reader got a generic 500 with nothing naming the cause. Export was
        # simply dead for those users.
        name = re.sub(r"[^A-Za-z0-9._-]", "-", Path(runtime.hf_id or "session").name)
        name = name.strip("-") or "session"
        return Response(
            content=blob,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{name}.mri"',
                # The browser must not treat this as a gzip transfer encoding
                # and silently inflate it -- the bytes are the file.
                "Content-Length": str(len(blob)),
            },
        )

    # A session is JSON and attention matrices; 64 MB is far past any real one
    # and stops an accidental upload of something else from being buffered.
    _SESSION_LIMIT = 64 * 1024 * 1024

    # Raw body rather than multipart: the client already has the bytes, and
    # multipart would pull in python-multipart for nothing.
    @app.post("/api/session/open")
    async def session_open(request: Request):
        data = await request.body()
        if len(data) > _SESSION_LIMIT:
            return JSONResponse(
                {
                    "error": f"that file is larger than {_SESSION_LIMIT // 1_000_000} MB "
                    "— sessions are not that big, so this is probably not one"
                },
                status_code=413,
            )
        try:
            return await asyncio.to_thread(runtime.open_session, data)
        except Refusal as err:
            return JSONResponse({"error": err.sentence}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": err.sentence}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/session/open")

    @app.post("/api/session/close")
    def session_close() -> dict:
        return runtime.close_session()

    @app.websocket("/ws/generate")
    async def ws_generate(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                # PER MESSAGE, because a malformed frame is one client's
                # mistake and not a reason to drop the socket. Measured
                # against this file: `hello` raised JSONDecodeError, `[1,2]`
                # raised AttributeError on `.get`, and a binary frame raised
                # KeyError('text') — each of them escaping to uvicorn, which
                # closes 1011 with no `error` and no `done` frame.
                #
                # Starlette routes the app-level `Exception` handler
                # exclusively through `ServerErrorMiddleware`, which returns
                # early for non-http scopes, so a websocket gets no backstop
                # from it. And `docs/reference/api.md` documents this endpoint
                # as public API that answers with an error frame; the shipped
                # playground registers no `onclose`, so its Generate button
                # would stay disabled forever.
                try:
                    raw = await ws.receive_text()
                    msg = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError, KeyError):
                    await ws.send_json(
                        {
                            "type": "error",
                            "message": (
                                "that frame is not JSON. This socket takes one "
                                "object a frame: "
                                '{"prompt": "...", "max_new_tokens": 256}.'
                            ),
                        }
                    )
                    continue
                if not isinstance(msg, dict):
                    await ws.send_json(
                        {
                            "type": "error",
                            "message": (
                                f"that frame is a {type(msg).__name__}, and this "
                                f"socket takes a JSON object with a `prompt` key."
                            ),
                        }
                    )
                    continue
                if not runtime.loaded:
                    # Same sentence as the POST route above and the ten
                    # `Refusal` sites, for the same reason: this socket is how
                    # the Playground generates, so it is the one an ordinary
                    # click reaches.
                    await ws.send_json(
                        {
                            "type": "error",
                            "message": "No model loaded — pick one first.",
                        }
                    )
                    continue

                queue: asyncio.Queue[str | None] = asyncio.Queue()
                loop = asyncio.get_running_loop()
                failure: list[str] = []
                # This socket is how the playground generates, so this is the
                # path that fills the agents panel for a model loaded here.
                # It always commits — there is no `commit` in the message —
                # so every run over it is recorded.
                rec = _Recording(runtime, str(msg.get("prompt", "")))

                def produce(request: dict = msg) -> None:
                    try:
                        pieces = runtime.generate_stream(
                            str(request.get("prompt", "")),
                            int(request.get("max_new_tokens", 256)),
                            float(request.get("temperature", 0.7)),
                        )
                        for piece in pieces:
                            rec.piece(piece)
                            loop.call_soon_threadsafe(queue.put_nowait, piece)
                    except (Refusal, BadRequest) as err:
                        # A stream that stops mid-sentence has to say why, and
                        # for these two the why is a sentence somebody wrote:
                        # Ollama went away, the model is a recording, the
                        # prompt was rejected. Same words the REST handlers
                        # publish at 409 and 422.
                        failure.append(str(err))
                    except Exception as err:
                        # Everything else. This used to append
                        # f"{type(err).__name__}: {err}", which on the busiest
                        # error path in the app published exactly what the
                        # module header forbids -- measured, a
                        # RuntimeError("CUDA out of memory ... <absolute
                        # path>") reached the browser verbatim.
                        #
                        # test_smoke.py::test_ws_reports_a_mid_stream_crash_as_an_error
                        # asserted that literal text, on the argument that the
                        # stream must say why it stopped. The argument is
                        # right and the assertion was the wrong way to hold
                        # it: "the model failed mid-generation" says why, and
                        # the terminal has the rest. That test now asserts the
                        # reason arrives and torch's text does not.
                        #
                        # `Exception`, not `BaseException`: catching
                        # KeyboardInterrupt and SystemExit here rendered a
                        # shutdown as a chat error message. They now end the
                        # worker thread, and the `finally` below still posts
                        # the sentinel so the socket does not hang.
                        log.exception("generation failed mid-stream", exc_info=err)
                        failure.append(
                            "The model failed mid-generation rather than "
                            "refusing. The full error is in the terminal "
                            "running `modelmri serve`."
                        )
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, None)

                threading.Thread(target=produce, daemon=True).start()
                while (piece := await queue.get()) is not None:
                    await ws.send_json({"type": "token", "text": piece})
                # Filed BEFORE the terminal frame, so a panel refreshed the
                # instant the playground says "done" already has the run in
                # it. A trace that lands a beat after the event that would
                # make you go and look is a trace you have to reload to see.
                await _file(rec, failure[0] if failure else None)
                if failure:
                    await ws.send_json({"type": "error", "message": failure[0]})
                else:
                    await ws.send_json({"type": "done"})
        except WebSocketDisconnect:
            # The reader closed the tab or hit Stop. There is no one left to
            # tell and nothing to clean up — the generation thread is a daemon
            # and the queue goes with the frame. This is the one place in the
            # package where catching and doing nothing is the whole correct
            # behaviour, which is why it says so rather than looking like the
            # seventeen swallowed exceptions that came out of this file's
            # neighbours.
            pass
        except Exception as err:
            # The backstop the app-level handler cannot provide for a
            # websocket. One error frame, then close — rather than uvicorn
            # closing 1011 on a client that is waiting for `done`.
            log.exception("/ws/generate failed", exc_info=err)
            try:
                await ws.send_json(
                    {
                        "type": "error",
                        "message": (
                            "Something inside ModelMRI failed rather than "
                            "refusing. The full error is in the terminal "
                            "running `modelmri serve`."
                        ),
                    }
                )
            except Exception as gone:
                # The socket is already closed — the reader left while the
                # error frame was being written. The failure that mattered is
                # logged above; this one is logged at debug so the swallow is
                # visible without adding noise to an ordinary disconnect.
                log.debug("could not deliver the error frame: %s", type(gone).__name__)

    return app
