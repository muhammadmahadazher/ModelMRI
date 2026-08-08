"""modelmri-record — capture agent runs for the ModelMRI timeline.

Zero-config, never crashes the host app. Two ways in:

    from modelmri.record import trace, step

    with trace("fix-tests-run"):
        step("llm_call", name="plan", input=prompt, output=answer,
             duration_ms=1200, tokens_in=900, tokens_out=200)
        with step("subagent", name="test-runner"):
            step("tool_call", name="pytest", input="-q", output="3 failed")

    from modelmri.record import instrument_anthropic
    instrument_anthropic()   # auto-records every Anthropic SDK message call

Delivery: POSTs the finished trace to a running ModelMRI server
(http://127.0.0.1:5900 by default); if unreachable, writes
./modelmri-traces/<name>-<stamp>.json for later import.

Standalone and dependency-free on purpose: instrumenting an agent should not
cost you a 2.5 GB torch install. `pip install modelmri-record` is stdlib only.
The viewer (`pip install modelmri`) is a separate, optional thing.

Credentials are redacted before anything leaves the process. See redact.py --
this is on by default and has to be switched off deliberately.
"""

from __future__ import annotations

import atexit
import contextvars
import json
import time
import urllib.request
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .redact import Redactor, default_redactor, redact_document

__version__ = "0.1.0"

DEFAULT_ENDPOINT = "http://127.0.0.1:5900/api/traces/import"

_current: contextvars.ContextVar["_Trace | None"] = contextvars.ContextVar(
    "modelmri_trace", default=None
)


class _Trace:
    def __init__(self, name: str, endpoint: str, redactor: Redactor | None) -> None:
        self.name = name
        self.endpoint = endpoint
        self.redactor = redactor
        self.delivered = False
        self.id = uuid.uuid4().hex[:12]
        self.t0 = time.monotonic()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.steps: list[dict] = []
        self.parent_stack: list[str] = []

    def now_ms(self) -> int:
        return int((time.monotonic() - self.t0) * 1000)

    def document(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "started_at": self.started_at,
            "meta": {"recorder": f"modelmri-record/{__version__}"},
            "steps": self.steps,
        }


@contextmanager
def trace(
    name: str,
    endpoint: str = DEFAULT_ENDPOINT,
    *,
    redact: Redactor | None | bool = True,
) -> Iterator[None]:
    """Record everything inside this block as one trace, then deliver it.

    redact=True   the default credential scrubber (see redact.py)
    redact=fn     your own str -> str
    redact=False  send payloads verbatim. Only for traces that never leave
                  your machine, and only deliberately.
    """
    redactor: Redactor | None
    if redact is True:
        redactor = default_redactor
    elif redact is False or redact is None:
        redactor = None
    else:
        redactor = redact

    t = _Trace(name, endpoint, redactor)
    # If the interpreter dies mid-run -- a crash, a SIGTERM, a notebook kernel
    # restart -- the finally below never runs and the whole trace is lost,
    # which is exactly the run you most wanted to look at.
    _live.append(t)
    token = _current.set(t)
    try:
        yield
    except Exception as err:
        t.steps.append(
            {
                "kind": "error",
                "name": type(err).__name__,
                "started_ms": t.now_ms(),
                "duration_ms": 0,
                "output": str(err)[:2000],
                "error": True,
            }
        )
        raise
    finally:
        _current.reset(token)
        try:
            _live.remove(t)
        except ValueError:
            pass
        _deliver(t)


class _StepCtx:
    """Returned by step(); usable bare or as a context manager for nesting."""

    def __init__(self, record: dict, tr: _Trace) -> None:
        self._record = record
        self._trace = tr
        self._entered_ms = 0

    def __enter__(self) -> "_StepCtx":
        self._trace.parent_stack.append(self._record["id"])
        self._entered_ms = self._trace.now_ms()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._trace.parent_stack.pop()
        if not self._record["duration_ms"]:
            self._record["duration_ms"] = self._trace.now_ms() - self._entered_ms
        if exc is not None:
            self._record["error"] = True
            self._record["output"] = f"{type(exc).__name__}: {exc}"[:2000]


def step(
    kind: str,
    name: str = "",
    input: object = "",  # noqa: A002 - mirrors the wire field name
    output: object = "",
    duration_ms: int = 0,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    error: bool = False,
    started_ms: int | None = None,  # override for backfilled/synthetic traces
) -> _StepCtx | None:
    """Record one step in the active trace (no-op outside a trace block)."""
    t = _current.get()
    if t is None:
        return None
    record = {
        "id": uuid.uuid4().hex[:10],
        "parent_id": t.parent_stack[-1] if t.parent_stack else None,
        "kind": kind,
        "name": name,
        "started_ms": t.now_ms() if started_ms is None else started_ms,
        "duration_ms": duration_ms,
        "input": input if isinstance(input, str) else json.dumps(input, default=str),
        "output": output
        if isinstance(output, str)
        else json.dumps(output, default=str),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "error": error,
    }
    t.steps.append(record)
    return _StepCtx(record, t)


def instrument_anthropic() -> bool:
    """Monkey-patch anthropic.Messages.create to auto-record llm_call steps.

    Returns False (no crash) when the anthropic package is not installed.
    """
    try:
        from anthropic.resources.messages import Messages
    except Exception:
        return False
    if getattr(Messages.create, "_modelmri_wrapped", False):
        return True
    original = Messages.create

    def wrapped(self, *args, **kwargs):  # noqa: ANN001, ANN202
        t = _current.get()
        started = t.now_ms() if t else 0
        try:
            result = original(self, *args, **kwargs)
        except Exception as err:
            if t is not None:
                step(
                    "llm_call",
                    name=str(kwargs.get("model", "anthropic")),
                    input=_msgs_preview(kwargs),
                    output=f"{type(err).__name__}: {err}",
                    duration_ms=(t.now_ms() - started),
                    error=True,
                )
            raise
        if t is not None:
            usage = getattr(result, "usage", None)
            step(
                "llm_call",
                name=str(kwargs.get("model", "anthropic")),
                input=_msgs_preview(kwargs),
                output=_content_preview(result),
                duration_ms=(t.now_ms() - started),
                tokens_in=getattr(usage, "input_tokens", None),
                tokens_out=getattr(usage, "output_tokens", None),
            )
        return result

    wrapped._modelmri_wrapped = True  # type: ignore[attr-defined]
    Messages.create = wrapped  # type: ignore[assignment]
    return True


def _msgs_preview(kwargs: dict) -> str:
    try:
        return json.dumps(kwargs.get("messages", []), default=str)[:4000]
    except Exception:
        return ""


def _content_preview(result: object) -> str:
    try:
        blocks = getattr(result, "content", [])
        return " ".join(getattr(b, "text", "") for b in blocks)[:4000]
    except Exception:
        return ""


def _deliver(t: _Trace) -> None:
    if t.delivered:
        return
    t.delivered = True
    doc = t.document()
    if t.redactor is not None:
        doc = redact_document(doc, t.redactor)
    body = json.dumps(doc).encode()
    try:
        req = urllib.request.Request(
            t.endpoint, data=body, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=3)
        return
    except Exception:
        pass
    try:
        out = Path("modelmri-traces")
        out.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        (out / f"{t.name}-{stamp}.json").write_text(json.dumps(doc, indent=1))
    except Exception:
        pass  # recording must never crash the host app


# Traces still open when the process ends. Flushed by the atexit hook below.
_live: list[_Trace] = []


def _flush_live() -> None:
    """Deliver anything still in flight at interpreter shutdown."""
    for t in list(_live):
        try:
            _deliver(t)
        except Exception:
            pass


atexit.register(_flush_live)
