"""Minimal Ollama client (stdlib only): list installed models, stream text.

Ollama serves GGUF models over HTTP — great for *running* any open model
with zero setup, but its API exposes no internals, so attention / SAE
introspection is unavailable in Ollama mode (ModelMRI says so in the UI).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Iterator

DEFAULT_HOST = "http://127.0.0.1:11434"


def default_host() -> str:
    """Where Ollama is, honouring OLLAMA_HOST like every other Ollama client.

    Read at call time. As an import-time constant this ignored the variable
    entirely, so anyone running Ollama on another port — or on the GPU box
    across the room, which is a normal way to use it — got "Ollama isn't
    running" while it was running fine.

    Ollama's own convention allows a bare `host:port`, so accept that too
    rather than silently building an unusable URL out of it.
    """
    raw = (os.environ.get("OLLAMA_HOST") or "").strip()
    if not raw:
        return DEFAULT_HOST
    if "://" not in raw:
        raw = f"http://{raw}"
    return raw.rstrip("/")


# Popular open models worth suggesting when Ollama is running but empty.
# `ollama pull <name>` fetches these; sizes are the default quantisations.
SUGGESTED = [
    {"name": "qwen3:0.6b", "size": "0.5 GB", "note": "tiny, current"},
    {"name": "qwen3:4b", "size": "2.6 GB", "note": "strong for its size"},
    {"name": "llama3.2:3b", "size": "2.0 GB", "note": "Meta, general"},
    {"name": "gemma3:4b", "size": "3.3 GB", "note": "Google, multimodal"},
    {"name": "phi4-mini:3.8b", "size": "2.5 GB", "note": "Microsoft, reasoning"},
    {"name": "deepseek-r1:1.5b", "size": "1.1 GB", "note": "reasoning traces"},
]


def status(host: str | None = None, timeout: float = 1.5) -> dict:
    """{up, models:[{name,size_gb,family}], suggested} — fast, never raises."""
    host = host or default_host()
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as resp:
            data = json.load(resp)
        models = []
        for m in data.get("models", []):
            name = m.get("name", "")
            if not name:
                continue
            details = m.get("details") or {}
            models.append(
                {
                    "name": name,
                    "size_gb": round((m.get("size") or 0) / 1e9, 2),
                    "family": details.get("family", ""),
                    "params": details.get("parameter_size", ""),
                    "quant": details.get("quantization_level", ""),
                }
            )
        models.sort(key=lambda m: m["name"])
        return {
            "up": True,
            "models": [m["name"] for m in models],  # back-compat
            "installed": models,
            "suggested": SUGGESTED,
            "host": host,
        }
    except Exception:
        return {
            "up": False,
            "models": [],
            "installed": [],
            "suggested": SUGGESTED,
            "host": host,
        }


def pull(name: str, host: str | None = None):
    """Stream `ollama pull` progress as dicts. Blocking generator."""
    host = host or default_host()
    body = json.dumps({"model": name, "stream": True}).encode()
    req = urllib.request.Request(
        f"{host}/api/pull", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw in resp:
                if not raw.strip():
                    continue
                msg = json.loads(raw)
                if msg.get("error"):
                    raise RuntimeError(f"ollama: {msg['error']}")
                total = msg.get("total") or 0
                done = msg.get("completed") or 0
                yield {
                    "status": msg.get("status", ""),
                    "percent": round(100 * done / total, 1) if total else None,
                    "total_gb": round(total / 1e9, 2) if total else None,
                }
    except urllib.error.URLError as err:
        raise RuntimeError(f"ollama unreachable at {host}: {err}") from err


def stream_generate(
    model: str,
    prompt: str,
    host: str | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> Iterator[str]:
    """Yield response text chunks from Ollama's NDJSON stream."""
    host = host or default_host()
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
