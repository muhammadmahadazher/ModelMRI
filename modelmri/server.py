"""FastAPI application: REST for control, WebSocket for token streams."""

from __future__ import annotations

import asyncio
import json
import threading
from importlib.resources import files

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .runtime import DEFAULT_MODEL, ModelRuntime


class LoadRequest(BaseModel):
    hf_id: str = DEFAULT_MODEL


class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


def create_app() -> FastAPI:
    app = FastAPI(title="ModelMRI", version=__version__)
    runtime = ModelRuntime()
    app.state.runtime = runtime

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return files("modelmri").joinpath("static/index.html").read_text("utf-8")

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
