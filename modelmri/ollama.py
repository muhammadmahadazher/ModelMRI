"""Minimal Ollama client (stdlib only): list installed models, stream text.

Ollama serves GGUF models over HTTP — great for *running* any open model
with zero setup, but its API exposes no internals, so attention / SAE
introspection is unavailable in Ollama mode (ModelMRI says so in the UI).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Iterator

DEFAULT_HOST = "http://127.0.0.1:11434"


def status(host: str = DEFAULT_HOST, timeout: float = 1.5) -> dict:
    """{up: bool, models: [name, ...]} — fast, never raises."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as resp:
            data = json.load(resp)
        models = [m.get("name", "") for m in data.get("models", [])]
        return {"up": True, "models": [m for m in models if m]}
    except Exception:
        return {"up": False, "models": []}


def stream_generate(
    model: str,
    prompt: str,
    host: str = DEFAULT_HOST,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> Iterator[str]:
    """Yield response text chunks from Ollama's NDJSON stream."""
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": max_new_tokens, "temperature": temperature},
        }
    ).encode()
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw in resp:
                if not raw.strip():
                    continue
                msg = json.loads(raw)
                if msg.get("error"):
                    raise RuntimeError(f"ollama: {msg['error']}")
                piece = msg.get("response", "")
                if piece:
                    yield piece
                if msg.get("done"):
                    return
    except urllib.error.URLError as err:
        raise RuntimeError(f"ollama unreachable at {host}: {err}") from err
