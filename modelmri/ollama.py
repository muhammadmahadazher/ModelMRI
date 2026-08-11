"""Minimal Ollama client (stdlib only): list installed models, stream text.

Ollama serves GGUF models over HTTP — great for *running* any open model
with zero setup, but its API exposes no internals, so attention / SAE
introspection is unavailable in Ollama mode (ModelMRI says so in the UI).
"""

from __future__ import annotations

import errno as errno_mod
import http.client
import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterator

from .errors import Refusal

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


REGISTRY = "https://registry.ollama.ai/v2"


# WHY THE TWO FAILURES THIS MODULE RAISES ARE REFUSALS AND NOT 500s.
#
# Both mean "Ollama did not do it", and neither means ModelMRI broke. The
# daemon is a separate process the user starts and stops; when it is down, or
# when it answers with an error of its own, the honest answer is a 409 saying
# so — a 500 reading "something inside ModelMRI failed" would blame this tool
# for another program's state.
#
# `ollama: ` is the marker on relayed text: everything after that prefix is
# Ollama's wording, not ours. It is the one place this project publishes a
# sentence it did not write, and the prefix is what keeps that honest.


def _relayed(message: str) -> Refusal:
    """Ollama's own error, marked as Ollama's."""
    return Refusal(f"ollama: {message}")


def _cause(err: BaseException) -> str:
    """Why a connection failed, with nothing from this machine in it.

    This used to be `getattr(err, "reason", None) or err`, interpolated whole,
    and the docstring argued it was safe because a reason "is an errno
    sentence, never a path from this machine". That is true of the errno case
    and false of every other one. Measured against an https OLLAMA_HOST:

        ollama unreachable at https://ollama.internal: ('CA bundle
        C:/Users/<name>/.../site-packages/certifi/cacert.pem',) ...

    which is a Refusal, so it is relayed to the browser at 409 by design --
    the argument was sound for the case it considered and there was a second
    case.

    The errno is kept, because it is the part a reader can act on
    (ECONNREFUSED means Ollama is not running; a resolution failure means
    OLLAMA_HOST points somewhere wrong) and a number with a fixed name cannot
    carry a path. `os.strerror` is the operating system's own fixed sentence
    for that number. Anything without an errno -- an SSL failure, a bad status
    line -- gives its class, on the same rule the rest of this project
    follows: the class, never the text.
    """
    inner = getattr(err, "reason", None) or err
    number = getattr(inner, "errno", None)
    if isinstance(number, int):
        name = errno_mod.errorcode.get(number, str(number))
        try:
            return f"{name} ({os.strerror(number)})"
        except (ValueError, OverflowError):
            return name
    return type(inner).__name__


def _unreachable(host: str, err: BaseException) -> Refusal:
    """The daemon is not answering. Written once; raised from two streams."""
    return Refusal(
        f"ollama unreachable at {host}: {_cause(err)}. Start Ollama, or set "
        f"OLLAMA_HOST if it listens somewhere else."
    )


def resolve(name: str, timeout: float = 10.0) -> dict:
    """Does this Ollama model exist, and how big is it?

    Ollama publishes no search API — ollama.com/search is a web page, not an
    endpoint — so the picker cannot offer a result list the way the
    HuggingFace tab does. What it can do is let you name any model and answer
    honestly about it, which covers strictly more than a search box would:
    namespaced models (`user/model`), any tag, and anything published after
    whatever list we might have hardcoded.

    Returns {found, bytes, name, error}. `found` false with an empty error
    means the registry answered and does not have it; a non-empty error means
    we could not ask.
    """
    tag = (name or "").strip()
    if not tag:
        return {"found": False, "bytes": 0, "name": "", "error": "type a model name"}
    if any(c.isspace() for c in tag):
        return {
            "found": False,
            "bytes": 0,
            "name": tag,
            "error": "an Ollama model name has no spaces — try `qwen3:8b`",
        }

    repo, _, version = tag.partition(":")
    namespaced = f"{repo}" if "/" in repo else f"library/{repo}"
    url = f"{REGISTRY}/{namespaced}/manifests/{version or 'latest'}"
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            doc = json.load(resp)
    except urllib.error.HTTPError as err:
        if err.code in (401, 404):
            return {
                "found": False,
                "bytes": 0,
                "name": tag,
                "error": "",  # a real answer: the registry does not have it
            }
        return {
            "found": False,
            "bytes": 0,
            "name": tag,
            "error": f"the Ollama registry answered {err.code}",
        }
    except Exception as err:
        return {
            "found": False,
            "bytes": 0,
            "name": tag,
            # The class, never the text. This is returned as data and
            # server.py spreads it into a 200 body, so there is no except arm
            # anywhere to sanitise it: measured, an SSL failure put the CA
            # bundle's absolute path in that body.
            "error": (
                f"could not reach the Ollama registry ({_cause(err)}). "
                "The full error is in the terminal running `modelmri serve`."
            ),
        }

    layers = doc.get("layers") or []
    config = doc.get("config") or {}
    total = int(
        sum(int(layer.get("size") or 0) for layer in layers)
        + int(config.get("size") or 0)
    )
    return {"found": True, "bytes": total, "name": tag, "error": ""}


def manifest_size(name: str, timeout: float = 10.0) -> int:
    """Bytes `ollama pull <name>` will fetch, from the registry's manifest.

    Asked before pulling, not discovered halfway through. `deepseek-r1:671b`
    is 404 GB and nothing in the UI said so; the HuggingFace side had the
    same hole and it cost someone a 1.5 TB download.

    Returns 0 when the registry cannot answer — treated as unknown by the
    guard, never as small.
    """
    repo, _, tag = name.partition(":")
    if "/" not in repo:
        repo = f"library/{repo}"  # ollama's default namespace
    url = f"{REGISTRY}/{repo}/manifests/{tag or 'latest'}"
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            doc = json.load(resp)
    except Exception:
        return 0
    layers = doc.get("layers") or []
    config = doc.get("config") or {}
    return int(
        sum(int(layer.get("size") or 0) for layer in layers)
        + int(config.get("size") or 0)
    )


# A starting point for the Ollama tab, mirroring `hub.SUGGESTED` for the
# HuggingFace one. Names only — every size below is resolved live against the
# registry, because a size written here would be a number nobody rechecks and
# tags are republished.
#
# Ollama publishes no search API, so this is not a substitute for one: the
# name box beside it still reaches strictly more models than any list can.
# This exists because an empty panel is a worse first impression than eight
# names, which is the same reason the HuggingFace tab has curated picks.
SUGGESTED = [
    "qwen3:0.6b",
    "qwen3:1.7b",
    "qwen3:8b",
    "llama3.2:1b",
    "llama3.2:3b",
    "gemma3:1b",
    "gemma3:4b",
    "phi4-mini",
]


def suggested(vram_gb: float | None = None, timeout: float = 6.0) -> list[dict]:
    """The curated Ollama list, sized live and marked against this GPU.

    Fetched concurrently for the same reason the HuggingFace side is: this is
    the view the tab opens on, and eight sequential registry lookups is a
    visible wait on a panel that should feel instant.

    Never raises. Offline, the names still appear with no size — a picker with
    nothing in it is worse than one with names and no metadata.
    """
    from concurrent.futures import ThreadPoolExecutor

    def one(name: str) -> dict:
        try:
            info = resolve(name, timeout=timeout)
            size = int(info.get("bytes") or 0)
        except Exception:
            size = 0
        gb = round(size / 1e9, 2) if size else 0.0
        return {
            "name": name,
            "size_gb": gb,
            # Will this RUN on the GPU, which is the question the chip answers.
            #
            # This used `max(4 * vram_gb, 20.0)` — `capacity.guard`'s ceiling
            # for refusing a DOWNLOAD, which is deliberately generous because
            # a model too big for VRAM still runs on the CPU. Against a 20 GB
            # floor every curated entry here is under 6 GB, so the verdict was
            # `True` for every model at every GPU size, including a 1 GB card:
            # a constant wearing the costume of a measurement.
            #
            # Ollama loads GGUF weights into VRAM and needs room on top for
            # the KV cache and context, so the honest test is the weights
            # against the card with headroom left. It is approximate, and the
            # chip only ever says "bigger than this GPU", never "will fit".
            "fits": None if (not gb or not vram_gb) else gb <= vram_gb * 0.8,
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(one, SUGGESTED))


def is_instruct(
    name: str, host: str | None = None, timeout: float = 3.0
) -> bool | None:
    """Is this Ollama model instruction-tuned, or a base text continuer?

    Ollama publishes both — `llama3.2:1b-text-fp16`, `qwen2.5:0.5b-base` and
    `gemma2:2b-text-q4_0` are all base tags — so this cannot be assumed. It
    was, and the assumption silenced the caveat that explains why a base model
    answers strangely, for exactly the models that need it.

    The signal is Ollama's own: `/api/show` returns the chat template a model
    was published with, and a base model has none. Same distinction the
    HuggingFace path draws from `tokenizer.chat_template`.

    Returns None when the daemon cannot be asked. Unknown is not False —
    False is the positive claim "this is a base model", which the UI states
    in those words.
    """
    host = host or default_host()
    try:
        req = urllib.request.Request(
            f"{host}/api/show",
            data=json.dumps({"model": name}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            doc = json.load(resp)
    except Exception:
        return None
    return bool((doc.get("template") or "").strip())


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
            # Names only. Sizes come from `/api/ollama/suggested`, which
            # resolves each against the registry — this used to carry strings
            # like "2.6 GB" written by hand, which is a number nobody
            # rechecks against tags that get republished.
            "suggested": SUGGESTED,
            "host": host,
        }
    except Exception:
        return {
            "up": False,
            "models": [],
            "installed": [],
            # Names only. Sizes come from `/api/ollama/suggested`, which
            # resolves each against the registry — this used to carry strings
            # like "2.6 GB" written by hand, which is a number nobody
            # rechecks against tags that get republished.
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
                    raise _relayed(msg["error"])
                total = msg.get("total") or 0
                done = msg.get("completed") or 0
                yield {
                    "status": msg.get("status", ""),
                    "percent": round(100 * done / total, 1) if total else None,
                    "total_gb": round(total / 1e9, 2) if total else None,
                }
    except (urllib.error.URLError, OSError, http.client.HTTPException) as err:
        # WHAT A SEPARATE PROCESS DYING ACTUALLY RAISES, MEASURED.
        #
        # This caught URLError alone, which only covers the failure to
        # *connect*. Once the connection is up, urllib wraps nothing: the
        # daemon quitting mid-NDJSON, a proxy resetting, a body that stops
        # short all come out raw. Driven against a local socket that behaves
        # each way:
        #
        #   dies mid-stream    ConnectionAbortedError / ConnectionResetError
        #   never replies      RemoteDisconnected
        #   truncated body     IncompleteRead   (an HTTPException, not OSError)
        #
        # None of those were URLError, so none of them were a Refusal, so the
        # reader was told "something inside ModelMRI failed" about another
        # program's death — the exact mis-blame the split exists to prevent.
        # OSError and HTTPException between them cover all of it.
        raise _unreachable(host, err) from err
    except json.JSONDecodeError as err:
        # A 200 whose body is not NDJSON: a captive portal or a corporate
        # proxy answering with HTML, or the daemon truncating a line as it
        # goes down. Measured — a proxy injecting `<html>...` raised this from
        # `json.loads(raw)`. It is a ValueError, so it was not caught above
        # either, and it says nothing a reader can act on, which is why the
        # sentence below is ours and `err` is not in it.
        raise Refusal(
            f"ollama answered at {host}, but not with the streaming JSON its "
            f"API documents — something between here and the daemon is "
            f"rewriting the response. Check OLLAMA_HOST and any proxy."
        ) from err


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
                    raise _relayed(msg["error"])
                piece = msg.get("response", "")
                if piece:
                    yield piece
                if msg.get("done"):
                    return
    except (urllib.error.URLError, OSError, http.client.HTTPException) as err:
        # WHAT A SEPARATE PROCESS DYING ACTUALLY RAISES, MEASURED.
        #
        # This caught URLError alone, which only covers the failure to
        # *connect*. Once the connection is up, urllib wraps nothing: the
        # daemon quitting mid-NDJSON, a proxy resetting, a body that stops
        # short all come out raw. Driven against a local socket that behaves
        # each way:
        #
        #   dies mid-stream    ConnectionAbortedError / ConnectionResetError
        #   never replies      RemoteDisconnected
        #   truncated body     IncompleteRead   (an HTTPException, not OSError)
        #
        # None of those were URLError, so none of them were a Refusal, so the
        # reader was told "something inside ModelMRI failed" about another
        # program's death — the exact mis-blame the split exists to prevent.
        # OSError and HTTPException between them cover all of it.
        raise _unreachable(host, err) from err
    except json.JSONDecodeError as err:
        # A 200 whose body is not NDJSON: a captive portal or a corporate
        # proxy answering with HTML, or the daemon truncating a line as it
        # goes down. Measured — a proxy injecting `<html>...` raised this from
        # `json.loads(raw)`. It is a ValueError, so it was not caught above
        # either, and it says nothing a reader can act on, which is why the
        # sentence below is ours and `err` is not in it.
        raise Refusal(
            f"ollama answered at {host}, but not with the streaming JSON its "
            f"API documents — something between here and the daemon is "
            f"rewriting the response. Check OLLAMA_HOST and any proxy."
        ) from err
