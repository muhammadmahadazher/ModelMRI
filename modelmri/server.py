"""FastAPI application: REST for control, WebSocket for token streams."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from dataclasses import asdict
from importlib.resources import files

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pathlib import Path

from . import __version__
from .errors import BadRequest, Refusal
from .runtime import DEFAULT_MODEL, ModelRuntime
from .saes import DEFAULT_SAE_HOOK, DEFAULT_SAE_REPO
from .traces import TraceStore
from .custom import AdapterError, CustomHandle
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


class CustomLoadRequest(BaseModel):
    path: str = Field(min_length=1, max_length=4096)


class CustomRunRequest(BaseModel):
    # None means "use the adapter's example_input()"; the panel sends the
    # shape it showed you, so nothing ever runs on a shape you didn't see.
    shape: list[int] | None = Field(default=None, max_length=8)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    commit: bool = True


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
    app.state.vla = VLAHandle()
    app.state.vla_reader = None
    app.state.custom = CustomHandle()
    if trace_db:
        db_path = str(trace_db)
    else:
        # Platform data dir, but keep using an existing ~/.modelmri database
        # rather than starting an empty one beside it and losing the history.
        from . import paths

        db_path = str(paths.trace_db_path())
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    traces = TraceStore(db_path)
    app.state.traces = traces

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

    @app.get("/api/paths")
    def where() -> dict:
        """Every directory this program reads or writes.

        A tool that puts gigabytes on your disk should be able to say where,
        without you reading its source. Nothing here creates a directory —
        asking is not writing.
        """
        from . import paths

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
            last = {}
            for update in _ollama.pull(req.name):
                last = update
            return {"pulled": req.name, "last": last}

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

        def run() -> str:
            return "".join(
                runtime.generate_stream(
                    req.prompt, req.max_new_tokens, req.temperature, req.commit
                )
            )

        try:
            return {"generation": await asyncio.to_thread(run)}
        except Refusal as err:
            # Ollama quitting mid-session, Ollama refusing the prompt, no
            # model loaded. `runtime.generate_stream` translates ollama.py's
            # plain RuntimeErrors into Refusals at the call, which is what
            # lets this handler tell them apart from the next arm.
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
        except Exception as err:
            # CUDA out of memory, a streamer timeout, an architecture
            # transformers cannot run eagerly. These were 409s carrying
            # "{type}: {err}" — a full GPU reported as a conflict, in torch's
            # words, which can name paths on this machine.
            return _internal(err, "/api/model/prompt")

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
    async def lens(top_k: int = 5):
        """Logit lens — what the model would have said at each layer.

        The fallback for every model with no SAE, which is most of them.
        """
        # Replay first, and this is the only replay-sensitive route in the tree
        # whose guard does not live in runtime.py. Every other one —
        # attention_meta, attention_slice, compare, rank_heads,
        # attribute_tokens, export_session — opens with `if self.replay is not
        # None` because it is a ModelRuntime method. The lens is computed here
        # instead, from `modelmri.lens`, so it never passed a runtime guard and
        # nobody noticed: `runtime.model is None` catches the common case of
        # opening a `.mri` with nothing loaded, which looks like it is working.
        #
        # It stops looking like it is working the moment someone opens a
        # recording while their own model is still loaded. Then `model` is not
        # None, `last_ids` is not None, and the lens happily reports the LIVE
        # model's layers inside a session every other panel is drawing from the
        # recording — with the replay pill on screen saying "recorded, not
        # live".
        if runtime.replay is not None:
            return JSONResponse(
                {
                    "error": "This is a recording. The logit lens means running "
                    "the model, and a `.mri` does not carry one."
                },
                status_code=409,
            )
        if runtime.backend == "ollama":
            return JSONResponse(
                {
                    "error": "Ollama serves text only — the layers never leave its process"
                },
                status_code=409,
            )
        if runtime.model is None:
            return JSONResponse({"error": "no model loaded"}, status_code=409)
        if runtime.last_ids is None:
            return JSONResponse(
                {"error": "generate something first — the lens reads that run"},
                status_code=409,
            )

        from .lens import logit_lens

        def run() -> dict:
            return logit_lens(runtime.model, runtime.tokenizer, runtime.last_ids, top_k)

        try:
            return await asyncio.to_thread(run)
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/lens")

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

    @app.post("/api/custom/unload")
    def custom_unload() -> dict:
        return app.state.custom.unload().to_dict()

    @app.get("/api/vla")
    def vla_status() -> dict:
        # Names the configured dataset without opening it, so the resting panel
        # can say what a click will read instead of guessing the default.
        return {
            **app.state.vla.status().to_dict(),
            "dataset_repo": getattr(app.state, "vla_dataset", dataset_repo),
            "policy_repo": VLA_DEFAULT_REPO,
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
    async def vla_episodes():
        try:
            return await asyncio.to_thread(lambda: _reader().summary())
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
    async def vla_frame(episode: int = 0, t: int = 0):
        try:
            sample = await asyncio.to_thread(lambda: _reader().frame(episode, t))
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

    @app.get("/api/traces/{trace_id}")
    def trace_get(trace_id: str):
        doc = traces.get_trace(trace_id)
        if doc is None:
            return JSONResponse({"error": "trace not found"}, status_code=404)
        return doc

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

    @app.get("/api/session/export")
    async def session_export(layer: int = 0, head: int = 0, note: str = ""):
        try:
            blob = await asyncio.to_thread(runtime.export_session, layer, head, note)
        except Refusal as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=422)
        except Exception as err:
            return _internal(err, "/api/session/export")
        name = (runtime.hf_id or "session").replace("/", "-")
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

                def produce(request: dict = msg) -> None:
                    try:
                        pieces = runtime.generate_stream(
                            str(request.get("prompt", "")),
                            int(request.get("max_new_tokens", 256)),
                            float(request.get("temperature", 0.7)),
                        )
                        for piece in pieces:
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
