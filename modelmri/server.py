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

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, behavdiff, custom, gguf_read, otel, paths
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
from .runtime import DEFAULT_MODEL, ModelRuntime, _load_failed
from .saes import DEFAULT_SAE_HOOK, DEFAULT_SAE_REPO
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


class LoadRequest(BaseModel):
    hf_id: str = DEFAULT_MODEL
    source: str = "hf"  # "hf" | "ollama"
    # The user saw the size warning and chose to proceed anyway.
    confirm: bool = False


class GgufLoad(BaseModel):
    path: str
    # None means "whatever this accelerator prefers". Named explicitly rather
    # than defaulted to float32 here, because the dtype is half of the memory
    # figure and a silent default would make the preflight describe a load
    # nobody asked for.
    dtype: str | None = None
    # Overrides a tight fit, and nothing else.
    confirm: bool = False


class QuantCompare(BaseModel):
    quantised: str
    original: str
    prompt: str = "The capital of France is"
    # Off makes the run cheaper when only the token-level answer is wanted.
    attention: bool = True


class SAELoadRequest(BaseModel):
    repo: str = DEFAULT_SAE_REPO
    hook: str = DEFAULT_SAE_HOOK


class SteerRequest(BaseModel):
    feature_id: int | None = None  # None clears steering
    scale: float = Field(default=0.0, ge=-100.0, le=100.0)


class HubSignInRequest(BaseModel):
    token: str = Field(min_length=1, max_length=400)


class OllamaPullRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    # The user saw the size warning and chose to proceed. Never a default,
    # and never enough to override a disk that has no room.
    confirm: bool = False


class VLALoadRequest(BaseModel):
    # One constant, in vla.py. Two copies of a default is two things to
    # forget to change.
    repo: str = VLA_DEFAULT_REPO


class VLAAnalyseRequest(BaseModel):
    episode: int = Field(default=0, ge=0)
    t: int = Field(default=0, ge=0)


class VLADatasetRequest(BaseModel):
    repo_id: str = Field(min_length=1, max_length=200)


class ScanRequest(BaseModel):
    """A checkpoint or a directory of them.

    `limit` bounds a directory walk, and what it drops is reported rather than
    silently omitted — a scan that stopped at 200 files reads as "200 files,
    all fine".
    """

    path: str = Field(min_length=1, max_length=4096)
    limit: int = Field(default=200, ge=1, le=5000)


class ImageLoadRequest(BaseModel):
    """A cached pipeline directory, or a Hub id.

    No default. The checkpoint decides which panels apply, so guessing one
    would silently decide what the user is looking at.
    """

    repo: str = Field(min_length=1, max_length=400)
    device: str = Field(default="", max_length=32)
    dtype: str = Field(default="", max_length=32)
    confirm: bool = False


class ImageRunRequest(BaseModel):
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

    words: list[str] = Field(default_factory=list, max_length=24)
    seed: int = Field(default=0, ge=0, lt=2**31)


class VLAFrameRequest(BaseModel):
    """One frame, plus the seed that makes the answer reproducible.

    `seed` is optional and `None` is NOT 0. None means "do not fix the
    sampler", which is a different request and a different claim about the
    result — most of these policies sample, so an unseeded run is one draw
    from a distribution rather than the policy's answer.
    """

    episode: int = Field(default=0, ge=0)
    t: int = Field(default=0, ge=0)
    seed: int | None = Field(default=None, ge=0, lt=2**31)


class VLACompareRequest(BaseModel):
    """A whole episode, strided.

    `stride=0` means "choose one that fits the work budget" rather than
    "measure every frame" — an unstrided 200-frame episode is 200 forward
    passes, and `vla_actions.plan_frames` reports whatever it picks.
    """

    episode: int = Field(default=0, ge=0)
    stride: int = Field(default=0, ge=0, le=1000)
    seed: int | None = Field(default=None, ge=0, lt=2**31)


class CustomLoadRequest(BaseModel):
    path: str = Field(min_length=1, max_length=4096)


class CustomRootRequest(BaseModel):
    # A folder to also look in, for a model that does not live where the
    # server was started. Added to the allowed roots rather than bypassing
    # them — see custom.add_root.
    path: str = Field(min_length=1, max_length=4096)


class CustomRunRequest(BaseModel):
    # None means "use the adapter's example_input()"; the panel sends the
    # shape it showed you, so nothing ever runs on a shape you didn't see.
    shape: list[int] | None = Field(default=None, max_length=8)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class PatchGraphRequest(BaseModel):
    """A pair, and how far back to walk from what the grid flagged."""

    clean: str
    corrupt: str
    # 0 means "the module's default". Named rather than defaulted here so the
    # two do not drift apart the way a duplicated constant always does.
    depth: int = 0
    max_receivers: int = 0


class PatchRequest(BaseModel):
    # Two prompts, not one, and the pair is the unit of meaning: neither is
    # usable without the other, so they arrive together rather than as a
    # prompt plus a query parameter.
    clean: str
    corrupt: str


class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    commit: bool = True


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


def _scan_summary(reports, dangerous, unscanned) -> str:
    """One sentence that does not contradict itself.

    The first version said "N file(s) read and nothing executable found" and
    then "N could not be read", which are opposite claims about the same N —
    and on a directory of Python source, where every file is unscanned by
    design, it printed both about all of them. A summary whose two halves
    disagree is worse than either half alone.
    """
    if dangerous:
        return dangerous[0].means()
    if not reports:
        return "Nothing weight-shaped was found at that path."

    read = len(reports) - len(unscanned)
    if read == 0:
        return (
            f"NONE of the {len(reports)} file(s) here could be looked inside — "
            f"they are formats this cannot read, or Python source, which runs "
            f"in full when imported and cannot be made safe by a scan. This is "
            f"not a clean bill of health."
        )
    tail = (
        f" {len(unscanned)} could not be read and are reported as unscanned "
        f"rather than clean — a scanner that answers 'safe' for a file it "
        f"could not open is worse than no scanner."
        if unscanned
        else ""
    )
    return f"{read} file(s) read and nothing executable found.{tail}"


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
        return {
            "app": "modelmri",
            "version": __version__,
            "model": runtime.status().to_dict(),
        }

    @app.post("/api/model/load")
    async def load_model(req: LoadRequest):
        from .capacity import TooBig
        from .runtime import LoadCancelled

        try:
            status = await asyncio.to_thread(
                runtime.load, req.hf_id, req.source, req.confirm
            )
            return status.to_dict()
        except LoadCancelled as err:
            # Not a failure: the user asked. 200 with a plain answer, so the
            # UI does not paint a red error over something it did on purpose.
            # Stays first: LoadCancelled is a RuntimeError, so it survives
            # today only because nothing broader is written above it, and
            # anyone who ever widens an arm here to RuntimeError turns Stop
            # into a red 409.
            return JSONResponse({"cancelled": True, "message": str(err)})
        except TooBig as err:
            # capacity.py's own refusal, raised by _preflight before a byte
            # moves. Still a plain ValueError there, and this arm answers the
            # same 422 the pull path does — the two must not drift, which is
            # why capacity.guard is shared between them in the first place.
            return JSONResponse({"error": str(err)}, status_code=422)
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/model/load")

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
            return JSONResponse({"error": str(err)}, status_code=409)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
        _, free = _capacity.free_space(target)
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
                "warning": str(err),
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
        _, free = _capacity.free_space(target)
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
                "warning": str(err),
            }
        return {
            "name": name,
            "bytes": need,
            "free_bytes": free,
            "ok": True,
            "overridable": False,
            "warning": "",
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": "no model loaded"}, status_code=409)

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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            await _file(rec, str(err))
            return JSONResponse({"error": str(err)}, status_code=422)
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

    @app.post("/api/sae/load")
    async def sae_load(req: SAELoadRequest):
        try:
            status = await asyncio.to_thread(runtime.load_sae, req.repo, req.hook)
            return asdict(status)
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/sae/load")

    @app.get("/api/features/summary")
    async def features_summary(top_k: int = 8):
        try:
            return await asyncio.to_thread(runtime.features_summary, top_k)
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
        the answers differ: measured on gpt2 at blocks.8.hook_resid_pre with
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

        Measured on gpt2 float32 on this CPU: 10.09 s at position scope,
        49.44 s at prompt scope.

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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/features/{feature_id}")

    @app.post("/api/steer")
    def steer(req: SteerRequest):
        try:
            return runtime.set_steering(req.feature_id, req.scale)
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            feature = body.get("feature")
            return await asyncio.to_thread(
                runtime.feature_evidence,
                [str(t) for t in texts],
                feature_id=int(feature) if feature is not None else None,
                corpus_label=label,
                top_k=int(body.get("top_k") or 10),
            )
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
                return JSONResponse({"error": str(err)}, status_code=422)
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

        try:
            # Through the runtime rather than calling `model_diff.compare`
            # directly: that is what keeps the result available to
            # `export_session`, and it is where the accelerator settings and
            # the receipt already live.
            return await asyncio.to_thread(
                runtime.diff_models,
                str(body.get("a") or ""),
                str(body.get("b") or ""),
                [str(p) for p in prompts],
                # OFF by default. The head half costs n_layers x n_heads
                # forward passes per prompt PER SIDE -- 1,176 on gpt2 with
                # four prompts, 5,412 on a 1.7B with six -- which is two
                # orders of magnitude more than everything else in this
                # comparison, so it is opted into rather than out of.
                include_heads=bool(body.get("include_heads")),
                # Far cheaper than the head half — 248 passes for a 24-token
                # prompt over four — but still opted into, because a
                # 500-token prompt is not.
                include_tokens=bool(body.get("include_tokens")),
            )
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            body = {}
        try:
            return await asyncio.to_thread(
                app.state.custom.ablate,
                str(body.get("kind") or "layers"),
                grid=int(body.get("grid") or 0),
            )
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
                return JSONResponse({"error": str(err)}, status_code=422)
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
                max_chunks=int(body.get("max_chunks") or 0),
            )
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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

        try:
            return await asyncio.to_thread(
                runtime.patchscope,
                str(body.get("prompt") or ""),
                source_layer=int(body.get("layer", 0)),
                source_position=int(body.get("position", -1)),
                target_prompt=str(body.get("target") or ""),
                target_layer=(
                    int(body["target_layer"])
                    if body.get("target_layer") is not None
                    else None
                ),
                max_new_tokens=int(body.get("max_new_tokens") or 12),
            )
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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

        try:
            return await asyncio.to_thread(
                runtime.path_trace,
                str(body.get("clean") or ""),
                str(body.get("corrupt") or ""),
                layer=int(body.get("layer", -1)),
                position=int(body.get("position", -1)),
            )
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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

        try:
            return await asyncio.to_thread(
                runtime.probe_layers,
                body.get("examples") or [],
                n_permutations=int(body.get("n_permutations") or 0),
                save_as=str(body.get("save_as") or ""),
            )
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/attention/types")

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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
        steps = int(body.get("steps") or 250)

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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
        return {
            "adapters": adapters,
            "torchscript": scripts,
            "roots": [str(r) for r in custom_mod.allowed_roots()],
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
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return {
                "added": str(root),
                "adapters": adapters,
                "torchscript": scripts,
                "roots": [str(r) for r in custom.allowed_roots()],
            }
        except AdapterError as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
        return {
            "models": [m.to_dict() for m in found],
            "known": sum(1 for m in found if m.known),
            "means": (
                f"{len(found)} image model(s) cached on this machine, "
                f"{sum(1 for m in found if m.known)} of which this can open. "
                f"Nothing was downloaded to answer this."
            ),
        }

    @app.post("/api/image/load")
    async def image_load(req: ImageLoadRequest):
        """Hold one pipeline, after three refusals that cost nothing.

        Identify from JSON, scan the opcodes, price from real bytes — then
        load. Off the event loop because a pipeline is gigabytes off disk.
        """

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
            )
            return status.to_dict()
        # No separate `except Unsafe`. It is a `Refusal`, so the clause below
        # already answers 409 with its sentence intact — the extra arm added
        # nothing and named a type the leak check does not have on its
        # allow-list, which is the check doing its job: every exception a
        # handler publishes has to be provably authored, and proving it by
        # naming subclasses one at a time is how that list stops being true.
        except (Refusal, BadRequest) as err:
            code = 422 if isinstance(err, BadRequest) else 409
            return JSONResponse({"error": str(err)}, status_code=code)
        except Exception as err:
            return _internal(err, "/api/image/load")

    @app.post("/api/image/unload")
    async def image_unload() -> dict:
        """Drop it and hand the memory back, not merely forget it."""
        status = await asyncio.to_thread(app.state.image.unload)
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
        """Renders and passes, before any are spent."""
        from . import image_attention

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
            return JSONResponse({"error": str(err)}, status_code=409)

        try:
            run = await asyncio.to_thread(
                image_attention.capture,
                handle.require(),
                req.prompt,
                steps=req.steps,
                seed=req.seed,
            )
            return run.to_dict()
        except (image_attention.NotSupported, Refusal) as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/image/knockout")

    @app.get("/api/image/steps/cost")
    def image_steps_cost(steps: int = 20, threshold: float = 0.0):
        """What a latent trace will hold, before it holds it.

        The latent shape is read off the loaded pipeline when there is one, so
        the memory figure is this pipeline's rather than a guess — and `None`
        rather than 0 when there is not, because a run whose memory could not
        be priced is not a run that costs nothing.
        """
        from . import image_steps

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
            return JSONResponse({"error": str(err)}, status_code=422)

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
            return JSONResponse({"error": str(err)}, status_code=409)
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
        return {
            "reports": [r.to_dict() for r in reports],
            "dangerous": len(dangerous),
            "unscanned": len(unscanned),
            "safe": len(reports) - len(dangerous) - len(unscanned),
            "means": _scan_summary(reports, dangerous, unscanned),
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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

        try:
            reader = _reader()
        except ImportError as err:
            return _missing_reader_dep(err)
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)

        match = next((e for e in reader.episodes() if e.index == episode), None)
        if match is None:
            return JSONResponse(
                {"error": f"episode {episode} is not in this dataset"},
                status_code=422,
            )
        try:
            frames, chosen = vla_actions.plan_frames(match.length, stride=stride)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)

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
            return JSONResponse({"error": str(err)}, status_code=409)

        # Units BEFORE any forward pass. Spending three minutes and then
        # refusing to draw the result is a refusal that wasted three minutes,
        # and the answer does not depend on the passes.
        agree, why = vla_actions.units_agree(state.normalisation, reader.action_stats())
        if not agree:
            return JSONResponse({"error": why}, status_code=409)

        try:
            return await asyncio.to_thread(_run_compare, reader, state, req)
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)

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
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)

        try:
            return await asyncio.to_thread(_run_knockout, reader, state, req)
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)

        status = app.state.vla.status()
        try:
            return vla_sweep.estimate(
                reader,
                metric,
                episode_stride=episode_stride,
                frame_stride=frame_stride,
                grid=status.grid or None,
            )
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)

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
            body = {}

        try:
            reader = _reader()
        except ImportError as err:
            return _missing_reader_dep(err)
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except Exception as err:
            return _internal(err, "/api/vla/sweep")

        def run():
            out = vla_sweep.run(
                app.state.vla,
                reader,
                str(body.get("metric") or "attention_entropy"),
                episode_stride=int(body.get("episode_stride") or 1),
                frame_stride=int(body.get("frame_stride") or 25),
                occlusion_stride=int(body.get("occlusion_stride") or 0),
            )
            # Persisted so a sweep survives the process — the table is the
            # point of running one at all.
            vla_sweep.save(out)
            payload = out.to_dict()
            payload["strip"] = vla_sweep.heat_strip(out)
            return payload

        try:
            return await asyncio.to_thread(run)
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            body = {}

        try:
            reader = _reader()
        except ImportError as err:
            return _missing_reader_dep(err)
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except Exception as err:
            return _internal(err, "/api/vla/share")

        def run() -> bytes:
            from . import session as session_mod
            from . import vla as vla_mod

            payload = vla_mod.share_payload(
                app.state.vla,
                reader,
                episode=int(body.get("episode", 0)),
                timestep=int(body.get("t", 0)),
                layer=int(body.get("layer", -1)),
                head=int(body.get("head", -1)),
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
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            body = {}

        try:
            reader = _reader()
        except ImportError as err:
            return _missing_reader_dep(err)
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except Exception as err:
            return _internal(err, "/api/vla/occlude")

        episode = int(body.get("episode", 0))
        timestep = int(body.get("t", 0))

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
                stride=int(body.get("stride") or 0),
                layer=int(body.get("layer", -1)),
                head=int(body.get("head", -1)),
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
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)

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
            return JSONResponse({"error": str(err)}, status_code=409)
        except Exception as err:
            return _internal(err, "/api/vla/audit")

        try:
            report = await asyncio.to_thread(audit_mod.audit, reader)
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/vla/dataset")
        app.state.vla_reader = reader
        app.state.vla_dataset = req.repo_id
        return await asyncio.to_thread(reader.summary)

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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=422)
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
            doc["cost"] = {"error": str(err), "means": str(err)}
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
                return out.to_dict() | {
                    "trace_id": stored,
                    "header": head.to_dict(),
                    "samples": [s.to_dict() for s in inspect_io.samples(path)],
                }

        try:
            return await asyncio.to_thread(run)
        except BadRequest as err:
            # `InspectError` is one of these: an unrecognised schema version,
            # a file that is not a zip, a sample the log does not carry.
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=422)

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
        prompt = openai_api.build_prompt(runtime, body)
        model_name = str(body.get("model") or getattr(runtime, "hf_id", "") or "local")
        max_tokens = int(
            body.get("max_completion_tokens")
            or body.get("max_tokens")
            or openai_api.DEFAULT_MAX_TOKENS
        )
        temperature = float(body.get("temperature", 0.7))
        want_logprobs = bool(body.get("logprobs"))
        top_k = int(body.get("top_logprobs") or 0)
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
    async def judge_score(body: dict):
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
                str(body.get("text") or ""),
                str(body.get("rubric") or ""),
                n_paraphrases=int(body.get("n_paraphrases") or 0),
                device=str(getattr(runtime, "device", "cpu")),
            )
            return out.to_dict()

        try:
            return await asyncio.to_thread(run)
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/judge")

    @app.post("/api/judge/plan")
    def judge_plan(body: dict):
        """The prompts that would be run, before any of them is."""
        from . import judge as judge_mod

        try:
            prompts = judge_mod.plan(
                str(body.get("text") or ""),
                str(body.get("rubric") or ""),
                int(body.get("n_paraphrases") or 0),
            )
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
        return {"prompts": prompts, "n_passes": len(prompts)}

    @app.post("/api/rubric/score")
    async def rubric_score(body: dict, limit: int = 500):
        """Score every recorded run against exact predicates. No model."""
        from . import rubric as rubric_mod

        def run() -> dict:
            rules = rubric_mod.parse(body.get("rules", body))
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
            return JSONResponse({"error": str(err)}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/rubric/score")

    @app.get("/api/rubric")
    def rubric_list():
        return traces.rubrics()

    @app.post("/api/rubric")
    def rubric_save(body: dict):
        from . import rubric as rubric_mod

        name = str(body.get("name") or "").strip()
        if not name:
            return JSONResponse(
                {"error": "a saved rubric needs a name."}, status_code=422
            )
        try:
            rules = rubric_mod.parse(body.get("rules", []))
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/attention/diff")

    @app.get("/api/attention/ablate")
    async def ablate_heads(
        layer: int | None = None, baseline: str = "zero", scope: str = "layer"
    ):
        """Rank heads by how far removing one moves the next-token answer.

        `scope=layer` (default) does n_heads + 2 passes; `scope=all` does
        n_layers x n_heads + 2 — 146 for gpt2, 450 for Qwen3-0.6B. The
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
        the work — 98 passes against 14 for one gpt2 layer.
        """
        target = None if scope == "all" else (layer if layer is not None else 0)
        try:
            return await asyncio.to_thread(runtime.estimate_ablation, target, baseline)
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
        measured on gpt2 float32 with "The Eiffel Tower is located in the city
        of" against "The Colosseum is located in the city of", 5 of 132 sites
        moved it away, worst -0.157, and KL cannot tell those from a site that
        recovered nothing. The two rankings also disagree — top-8 by recovery
        against bottom-8 by KL-to-clean overlap on 5 of 8.

        Cost is `n_layers * n_positions` passes for the grid plus
        `draws + 1` for each of the top 24 sites: 350 passes in 9.66 s on gpt2
        for a 12x11 grid. Controls are eight draws, not one, because one is a
        coin flip — at a single site the draws ran -2.038 to +0.616 against a
        real recovery of +0.427, and the gate moves from 76 of 132 sites on
        one draw to 20 on all eight.

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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
        Measured through this endpoint: 11 on gpt2 with "The capital of France
        is" (3 tested), 21 on gemma-3-270m-it (13 tested), 24 on Qwen3-0.6B at
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
                msg = json.loads(await ws.receive_text())
                if not runtime.loaded:
                    await ws.send_json({"type": "error", "message": "no model loaded"})
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

    return app
