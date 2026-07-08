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

from . import __version__
from .runtime import DEFAULT_MODEL, ModelRuntime
from .saes import DEFAULT_SAE_HOOK, DEFAULT_SAE_REPO


class LoadRequest(BaseModel):
    hf_id: str = DEFAULT_MODEL


class SAELoadRequest(BaseModel):
    repo: str = DEFAULT_SAE_REPO
    hook: str = DEFAULT_SAE_HOOK


class SteerRequest(BaseModel):
    feature_id: int | None = None  # None clears steering
    scale: float = Field(default=0.0, ge=-100.0, le=100.0)


class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


def create_app() -> FastAPI:
    app = FastAPI(title="ModelMRI", version=__version__)
    runtime = ModelRuntime()
    app.state.runtime = runtime

    # Serve the built React app when present (frontend/ builds into static/app);
    # fall back to the legacy single-file playground otherwise.
    static = files("modelmri") / "static"
    app_index = static / "app" / "index.html"
    if app_index.is_file():
        app.mount("/app", StaticFiles(directory=str(static / "app")), name="app")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        page = app_index if app_index.is_file() else static / "index.html"
        return page.read_text("utf-8")

    @app.get("/api/session")
    def session() -> dict:
        return {
            "app": "modelmri",
            "version": __version__,
            "model": runtime.status().to_dict(),
        }

    @app.post("/api/model/load")
    async def load_model(req: LoadRequest) -> dict:
        status = await asyncio.to_thread(runtime.load, req.hf_id)
        return status.to_dict()

    @app.post("/api/model/prompt")
    async def prompt(req: PromptRequest):
        if not runtime.loaded:
            return JSONResponse({"error": "no model loaded"}, status_code=409)

        def run() -> str:
            return "".join(
                runtime.generate_stream(req.prompt, req.max_new_tokens, req.temperature)
            )

        return {"generation": await asyncio.to_thread(run)}

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

                def produce(request: dict = msg) -> None:
                    try:
                        pieces = runtime.generate_stream(
                            str(request.get("prompt", "")),
                            int(request.get("max_new_tokens", 256)),
                            float(request.get("temperature", 0.7)),
                        )
                        for piece in pieces:
                            loop.call_soon_threadsafe(queue.put_nowait, piece)
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, None)

                threading.Thread(target=produce, daemon=True).start()
                while (piece := await queue.get()) is not None:
                    await ws.send_json({"type": "token", "text": piece})
                await ws.send_json({"type": "done"})
        except WebSocketDisconnect:
            pass

    return app
