"""FastAPI application: REST for control, WebSocket for token streams."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import asdict
from importlib.resources import files

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
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
    db_path = trace_db or str(Path.home() / ".modelmri" / "traces.sqlite")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    traces = TraceStore(db_path)
    app.state.traces = traces

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
        try:
            status = await asyncio.to_thread(runtime.load, req.hf_id, req.source)
            return status.to_dict()
        except RuntimeError as err:
            return JSONResponse({"error": str(err)}, status_code=409)
        except ValueError as err:
            return JSONResponse({"error": str(err)}, status_code=422)

    @app.get("/api/model/progress")
    def load_progress() -> dict:
        """Polled while a load runs — a minutes-long wait needs a heartbeat."""
        from .progress import TRACKER

        return TRACKER.snapshot().to_dict()

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

    @app.post("/api/ollama/pull")
    async def ollama_pull(req: OllamaPullRequest):
        from . import ollama as _ollama

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
