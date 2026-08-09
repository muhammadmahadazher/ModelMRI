"""FastAPI application: REST for control, WebSocket for token streams."""

from __future__ import annotations

import asyncio
import json
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
from .runtime import DEFAULT_MODEL, ModelRuntime
from .saes import DEFAULT_SAE_HOOK, DEFAULT_SAE_REPO
from .traces import TraceStore
from .custom import AdapterError, CustomHandle
from .vla import DEFAULT_VLA_REPO as VLA_DEFAULT_REPO
from .vla import VLAHandle


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
    repo: str = "lerobot/smolvla_base"


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
    # read it is not fatal: the CLI already parsed and reported on it, so the
    # server starting with nothing open beats the server not starting.
    if pending := os.environ.get("MODELMRI_OPEN"):
        try:
            runtime.open_session(Path(pending).read_bytes())
        except Exception:
            pass

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
        from .runtime import LoadCancelled

        try:
            status = await asyncio.to_thread(
                runtime.load, req.hf_id, req.source, req.confirm
            )
            return status.to_dict()
        except LoadCancelled as err:
            # Not a failure: the user asked. 200 with a plain answer, so the
            # UI does not paint a red error over something it did on purpose.
            return JSONResponse({"cancelled": True, "message": str(err)})
        except RuntimeError as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except ValueError as err:
            return JSONResponse({"error": str(err)}, status_code=422)

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
        except ValueError as err:
            return JSONResponse({"error": str(err)}, status_code=422)

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
        except RuntimeError as err:
            return JSONResponse({"error": str(err)}, status_code=409)

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
        except RuntimeError as err:
            return JSONResponse({"error": str(err)}, status_code=409)

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
        except RuntimeError as err:
            # Ollama quitting mid-session, a streamer timeout, CUDA OOM: all
            # arrived here as a bare 500 with a traceback. Say what happened.
            return JSONResponse({"error": str(err)}, status_code=409)
        except Exception as err:  # noqa: BLE001 - last line before a 500
            return JSONResponse(
                {"error": f"{type(err).__name__}: {err}"}, status_code=409
            )

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
        except RuntimeError as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except ValueError as err:
            return JSONResponse({"error": str(err)}, status_code=422)

    @app.get("/api/features/summary")
    async def features_summary(top_k: int = 8):
        try:
            return await asyncio.to_thread(runtime.features_summary, top_k)
        except RuntimeError as err:
            return JSONResponse({"error": str(err)}, status_code=409)

    @app.get("/api/features/{feature_id}")
    async def feature_detail(feature_id: int):
        try:
            return await asyncio.to_thread(runtime.feature_detail, feature_id)
        except RuntimeError as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except ValueError as err:
            return JSONResponse({"error": str(err)}, status_code=422)

    @app.post("/api/steer")
    def steer(req: SteerRequest):
        try:
            return runtime.set_steering(req.feature_id, req.scale)
        except RuntimeError as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except ValueError as err:
            return JSONResponse({"error": str(err)}, status_code=422)

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
        are trained per model, and public ones exist for about a dozen models
        in total. The panel says so rather than looking broken.
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
        except RuntimeError as err:
            return JSONResponse({"error": str(err)}, status_code=409)

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
            return JSONResponse({"error": str(err)}, status_code=422)
        except Exception as err:  # a user's own code ran; never 500 on it
            return JSONResponse(
                {"error": f"{type(err).__name__}: {err}"}, status_code=422
            )
        return status.to_dict()

    @app.post("/api/custom/run")
    async def custom_run(req: CustomRunRequest):
        try:
            return await asyncio.to_thread(app.state.custom.run, req.shape, req.seed)
        except AdapterError as err:
            return JSONResponse({"error": str(err)}, status_code=422)
        except Exception as err:
            return JSONResponse(
                {"error": f"{type(err).__name__}: {err}"}, status_code=422
            )

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
        except FileNotFoundError as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except RuntimeError as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except ValueError as err:
            return JSONResponse({"error": str(err)}, status_code=422)

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
        except FileNotFoundError as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except Exception as err:
            return JSONResponse(
                {"error": f"{type(err).__name__}: {err}"}, status_code=409
            )
        app.state.vla_reader = reader
        app.state.vla_dataset = req.repo_id
        return await asyncio.to_thread(reader.summary)

    @app.get("/api/vla/episodes")
    async def vla_episodes():
        try:
            return await asyncio.to_thread(lambda: _reader().summary())
        except (FileNotFoundError, ImportError) as err:
            return JSONResponse({"error": str(err)}, status_code=409)

    @app.get("/api/vla/frame")
    async def vla_frame(episode: int = 0, t: int = 0):
        try:
            sample = await asyncio.to_thread(lambda: _reader().frame(episode, t))
            return asdict(sample)
        except (FileNotFoundError, ImportError) as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except ValueError as err:
            return JSONResponse({"error": str(err)}, status_code=422)

    @app.post("/api/vla/analyse")
    async def vla_analyse(req: VLAAnalyseRequest):
        def run() -> dict:
            rgb = _reader().raw_frame(req.episode, req.t)
            return app.state.vla.analyse(rgb, key=(req.episode, req.t))

        try:
            return await asyncio.to_thread(run)
        except (FileNotFoundError, ImportError, RuntimeError) as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except ValueError as err:
            return JSONResponse({"error": str(err)}, status_code=422)

    @app.get("/api/vla/attention/meta")
    def vla_attention_meta() -> dict:
        return app.state.vla.attention_meta()

    @app.get("/api/vla/attention")
    async def vla_attention(layer: int = 0, head: int = -1):
        try:
            return await asyncio.to_thread(app.state.vla.attention, layer, head)
        except RuntimeError as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except ValueError as err:
            return JSONResponse({"error": str(err)}, status_code=422)

    @app.post("/api/traces/import")
    async def traces_import(doc: dict):
        try:
            trace_id = await asyncio.to_thread(traces.import_trace, doc)
            return {"id": trace_id}
        except ValueError as err:
            return JSONResponse({"error": str(err)}, status_code=422)

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
    async def attention(layer: int = 0, head: int = 0):
        try:
            return await asyncio.to_thread(runtime.attention, layer, head)
        except RuntimeError as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except ValueError as err:
            return JSONResponse({"error": str(err)}, status_code=422)

    @app.get("/api/attention/ablate")
    async def ablate_heads(
        layer: int | None = None, baseline: str = "zero", scope: str = "layer"
    ):
        """Rank heads by how far removing one moves the next-token answer.

        `scope=layer` (default) does n_heads passes; `scope=all` does
        n_layers x n_heads. The default is the cheap one on purpose —
        measured at 0.12-0.68 s per layer against 1.4-19.6 s for a whole
        model — so the button is a click rather than a job.
        """
        target = None if scope == "all" else (layer if layer is not None else 0)
        try:
            return await asyncio.to_thread(runtime.ablate_heads, target, baseline)
        except RuntimeError as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except ValueError as err:
            return JSONResponse({"error": str(err)}, status_code=422)

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
        except RuntimeError as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except ValueError as err:
            return JSONResponse({"error": str(err)}, status_code=422)
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
        except ValueError as err:  # SessionError is one
            return JSONResponse({"error": str(err)}, status_code=422)

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
                    except BaseException as err:
                        # Without this the generation dies in the worker thread,
                        # the finally posts the sentinel, and the browser is told
                        # "done" -- a CUDA OOM or an unsupported architecture
                        # arrives as a successful empty answer.
                        failure.append(f"{type(err).__name__}: {err}")
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
            pass

    return app
