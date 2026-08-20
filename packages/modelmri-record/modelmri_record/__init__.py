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
import os
import sys
import time
import urllib.request
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .redact import Redactor, default_redactor, redact_document

__version__ = "0.1.4"

DEFAULT_ENDPOINT = "http://127.0.0.1:5900/api/traces/import"

# What the recorder keeps of a captured payload. NAMED rather than written
# inline five times, because these are now REPORTED and a cap nobody can see is
# a cap nobody can argue with.
#
# `_msgs_preview` and `_content_preview` cut the input and output of every
# `instrument_anthropic()` call — the zero-config headline feature — and 4,000
# is well under the store's own 20,000. So `traces._clip` never fired, no
# marker was written, `truncated_in` read 0, and a 50,000-character prompt
# reached the timeline looking whole. The only sentence a reader ever saw about
# truncation named 20,000, a cap that had not applied to it.
MAX_PREVIEW_CHARS = 4_000
MAX_ERROR_CHARS = 2_000


def _cut(text: str, limit: int) -> str:
    """`text` at `limit`, MARKED.

    The marker is byte-for-byte the one `traces._CLIPPED` parses, so the store
    lifts the count back out into `truncated_in`/`truncated_out` and the panel
    draws it — one shape for a cut payload wherever the cut happened, rather
    than a second mechanism that drifts away from the first.

    Still truncation BEFORE redaction: this runs at `step()` time and
    `redact.py` runs at delivery, so its headless-PEM rule still covers a key
    whose tail this removed.
    """
    if len(text) <= limit:
        return text
    return text[:limit] + f"… [+{len(text) - limit}]"


_current: contextvars.ContextVar[_Trace | None] = contextvars.ContextVar(
    "modelmri_trace", default=None
)


class _Trace:
    def __init__(
        self,
        name: str,
        endpoint: str,
        redactor: Redactor | None,
        meta: dict | None = None,
        deliver_otlp: str | None = None,
    ) -> None:
        self.name = name
        self.endpoint = endpoint
        self.redactor = redactor
        self.deliver_otlp = deliver_otlp
        self.meta = dict(meta or {})
        self.delivered = False
        self.id = uuid.uuid4().hex[:12]
        self.t0 = time.monotonic()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.steps: list[dict] = []
        # NOT a plain list. Concurrent asyncio tasks share the trace, and a
        # shared stack means task B's step becomes a child of task A's open
        # step purely because A happened to be inside a `with` at that moment.
        # A contextvar gives each task its own view of the ancestry while the
        # trace itself stays shared. Verified with asyncio.gather over two
        # agents: before, the second agent's tool call hung off the first.
        self.parents: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
            f"mri_parents_{self.id}", default=()
        )

    def now_ms(self) -> int:
        return int((time.monotonic() - self.t0) * 1000)

    def document(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "started_at": self.started_at,
            # Caller keys first-class, recorder identity last so it cannot be
            # shadowed by a caller who happens to use the same key.
            "meta": {**self.meta, "recorder": f"modelmri-record/{__version__}"},
            "steps": self.steps,
        }


@contextmanager
def trace(
    name: str,
    endpoint: str = DEFAULT_ENDPOINT,
    *,
    redact: Redactor | None | bool = True,
    meta: dict | None = None,
    deliver_otlp: str | None = None,
) -> Iterator[None]:
    """Record everything inside this block as one trace, then deliver it.

    redact=True   the default credential scrubber (see redact.py)
    redact=fn     your own str -> str
    redact=False  send payloads verbatim. Only for traces that never leave
                  your machine, and only deliberately.

    meta          anything you want stored alongside the run — a git sha, the
                  environment, a ticket id. `{"demo": True}` marks a scripted
                  sample so the viewer can label it instead of letting it pass
                  for something you actually recorded.

    deliver_otlp  an OTLP/HTTP endpoint to ALSO send this run to, e.g.
                  "http://localhost:4318". Off by default and deliberately so:
                  this package gets imported into other people's agents, and a
                  recorder that starts talking to the network because it can
                  is a recorder nobody should install.

                  It needs `modelmri` importable — the OTLP mapping lives
                  there in one table so emit and ingest cannot drift, and
                  copying it here would be the second copy that drifts. This
                  package keeps `dependencies = []`; if modelmri is absent the
                  export is skipped and says so, and the normal delivery is
                  untouched either way.
    """
    redactor: Redactor | None
    if redact is True:
        redactor = default_redactor
    elif redact is False or redact is None:
        redactor = None
    else:
        redactor = redact

    t = _Trace(name, endpoint, redactor, meta, deliver_otlp)
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
                # None, not 0. This step is synthesised at the moment an
                # exception escapes the block -- nothing timed it, so there is
                # no duration to report. A 0 here is a positive claim that the
                # failure took no measurable time, and it travelled: the OTLP
                # exporter reads `duration_ms is None` to decide whether to
                # stamp `modelmri.duration.recorded`, so every exported error
                # span asserted a duration nobody measured.
                "duration_ms": None,
                "output": _cut(str(err), MAX_ERROR_CHARS),
                "error": True,
            }
        )
        raise
    finally:
        _current.reset(token)
        try:
            _live.remove(t)
        except ValueError:
            # `list.remove` raises this and nothing else, and only when the
            # trace is already gone. Unreachable by construction today: this
            # is the only line in the package that removes from `_live`, and
            # `_flush_live` iterates a copy. It stays as defence for a caller
            # that reaches into `_live` itself, which the test suite does.
            #
            # Note `remove` compares by identity here — `_Trace` defines no
            # `__eq__` — so it can never take out a different-but-equal trace.
            # Continuing is right either way: `_deliver` runs next and is
            # idempotent through `t.delivered`, so the trace is still sent.
            pass
        _deliver(t)


class _NoStep:
    """What step() returns when there is no trace to record into.

    It has to support `with`, because that is the documented form and a
    library must not explode for callers who never opted into tracing. It
    also has to be falsy, so `if step(...)` keeps working. Returning a bare
    None -- which 0.1.0 did -- made `with step(...)` a TypeError outside a
    trace, and, worse, inside any worker thread: contextvars do not cross
    thread boundaries, so a fan-out agent hit this on every tool call.
    """

    __slots__ = ()

    def __bool__(self) -> bool:
        return False

    def __enter__(self) -> _NoStep:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


_NO_STEP = _NoStep()


class _StepCtx:
    """Returned by step(); usable bare or as a context manager for nesting."""

    def __init__(self, record: dict, tr: _Trace) -> None:
        self._record = record
        self._trace = tr
        self._entered_ms = 0

    def __enter__(self) -> _StepCtx:
        self._token = self._trace.parents.set(
            self._trace.parents.get() + (self._record["id"],)
        )
        self._entered_ms = self._trace.now_ms()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._trace.parents.reset(self._token)
        if not self._record["duration_ms"]:
            self._record["duration_ms"] = self._trace.now_ms() - self._entered_ms
        if exc is not None:
            self._record["error"] = True
            self._record["output"] = _cut(f"{type(exc).__name__}: {exc}", MAX_ERROR_CHARS)


def step(
    kind: str,
    name: str = "",
    input: object = "",  # noqa: A002 - mirrors the wire field name
    output: object = "",
    # None means NOT MEASURED, and it is the default because most callers use
    # this as a context manager and let `_StepCtx.__exit__` time it. 0 was the
    # old default and it is a different claim -- "this took no measurable
    # time" -- which `traces.py` already made the column nullable to express
    # and which the OTLP exporter reads to decide what it may assert.
    duration_ms: int | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    # Same rule as `duration_ms`, one field over. Providers report these only
    # sometimes -- Anthropic returns cache counts only when a cache was in
    # play, reasoning tokens only from models that reason -- so None means the
    # provider said nothing and 0 would mean it said zero. The trace store's
    # columns are nullable to carry that distinction all the way to the panel.
    tokens_cache_read: int | None = None,
    tokens_cache_write: int | None = None,
    tokens_reasoning: int | None = None,
    error: bool = False,
    started_ms: int | None = None,  # override for backfilled/synthetic traces
    meta: dict | None = None,
) -> _StepCtx | _NoStep:
    """Record one step in the active trace (no-op outside a trace block).

    `meta` carries machine facts about a step produced by a LOCAL model —
    model id, the token ids, dtype, device — so ModelMRI can reopen that exact
    generation in its attention, lens, ablation and patching panels.

    **Never put prompt or completion text in `meta`.** `redact.py` runs over
    `input` and `output` at delivery and nothing else, so text hidden in `meta`
    would leave the machine unredacted. Ids and numbers only.
    """
    t = _current.get()
    if t is None:
        return _NO_STEP
    record = {
        "id": uuid.uuid4().hex[:10],
        "parent_id": (t.parents.get() or (None,))[-1],
        "kind": kind,
        "name": name,
        "started_ms": t.now_ms() if started_ms is None else started_ms,
        "duration_ms": duration_ms,
        "input": _encode(input),
        "output": _encode(output),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_cache_read": tokens_cache_read,
        "tokens_cache_write": tokens_cache_write,
        "tokens_reasoning": tokens_reasoning,
        "error": error,
        "meta": meta or {},
    }
    t.steps.append(record)
    return _StepCtx(record, t)


def _encode(value: object) -> str:
    """Payload -> string, without ever raising.

    json.dumps(default=str) still dies on reference cycles and on non-primitive
    dict keys, and agent state graphs are cyclic by construction (a child
    holding a reference to its parent). 0.1.0 let that escape into the caller,
    which breaks the one promise this library makes.
    """
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except Exception:
        try:
            return _cut(repr(value), MAX_PREVIEW_CHARS)
        except Exception:
            return "<unserialisable>"


def instrument_anthropic(force: bool = False) -> bool:
    """Monkey-patch anthropic.Messages.create to auto-record llm_call steps.

    CHECKS THE SDK FIRST. When an attribute the wrapper reads has moved, this
    does not patch: it prints one line naming the package, the version and the
    attribute, and returns False.

    That refusal is the point. The wrapper reads token counts with
    `getattr(usage, "input_tokens", None)`, and the trace store's columns are
    nullable so `None` can mean "the provider reported nothing". If the
    attribute moved, every span would carry that same None — and the ledger
    would report "not reported by provider" about a provider that reported
    perfectly well. The absence would be real and the reason would be wrong.

    Returns False (no crash) when anthropic is not installed, and False when
    the fingerprint does not match. `verify.check()` carries the detail; set
    MODELMRI_FORCE_INSTRUMENT=1 to patch regardless.
    """
    from . import verify

    report = verify.check(force=force)
    if not report.installed:
        return False
    if not report.ok:
        # One line, on stderr, naming what moved. Not an exception: failing to
        # instrument must not take down the program being traced, and
        # `_complain` is already the package's answer to "best-effort is not
        # the same as silent".
        _complain(f"modelmri-record: {report.reason()}")
        return False
    if report.capture != "full":
        _complain(f"modelmri-record: {report.reason()}")

    try:
        from anthropic.resources.messages import Messages
    except Exception:
        return False
    if getattr(Messages.create, "_modelmri_wrapped", False):
        return True
    original = Messages.create
    captured = report.capture

    def wrapped(self, *args, **kwargs):
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
                # Anthropic returns these only when a cache was in play, so a
                # missing attribute is the ordinary case and `None` is the
                # honest reading of it. `getattr(..., None)` rather than
                # `getattr(..., 0)` is the whole distinction: 0 would claim
                # the API reported no cache use on a call that never had a
                # cache field at all.
                tokens_cache_read=getattr(usage, "cache_read_input_tokens", None),
                tokens_cache_write=getattr(usage, "cache_creation_input_tokens", None),
                # `captured` says whether the recorder could read everything
                # it reads. A "partial" step's empty token field is the SDK's
                # shape, not the provider's silence, and the panel needs to be
                # able to tell those apart.
                meta={
                    k: v
                    for k, v in (
                        ("model", str(kwargs.get("model", "")) or None),
                        ("captured", captured),
                    )
                    if v
                }
                or None,
            )
        return result

    wrapped._modelmri_wrapped = True  # type: ignore[attr-defined]
    Messages.create = wrapped  # type: ignore[assignment]
    return True


def instrument_transformers() -> bool:
    """Auto-record `generate()` calls, with the ids needed to reopen them.

    This is what makes an agent step openable in the mechanistic panels. Every
    hosted tracing platform stops at the API boundary; a local model's
    generation can carry its actual token ids, so ModelMRI can re-establish it
    as the current generation and read attention, the logit lens, head ablation
    and patching off the exact sequence the agent produced.

    Records ids, model id, dtype and device — and no text. Prompt and
    completion go through `input`/`output`, which is what `redact.py` covers;
    text smuggled through `meta` would leave the machine unredacted.

    A consequence worth knowing: if redaction rewrites the prompt, ModelMRI
    will refuse to adopt the step, because re-tokenising the redacted text no
    longer reproduces the recorded ids. That is the correct outcome — the model
    saw the unredacted text, and this tool should not reconstruct it.

    Returns False (no crash) when transformers is not installed.
    """
    try:
        from transformers.generation.utils import GenerationMixin
    except Exception:
        return False
    if getattr(GenerationMixin.generate, "_modelmri_wrapped", False):
        return True
    original = GenerationMixin.generate

    def wrapped(self, *args, **kwargs):
        t = _current.get()
        started = t.now_ms() if t else 0
        result = original(self, *args, **kwargs)
        if t is None:
            return result
        try:
            inputs = kwargs.get("input_ids")
            if inputs is None and args:
                inputs = args[0]
            n_prompt = int(inputs.shape[-1]) if inputs is not None else 0
            # `result[0]` assumed a bare [B, S] tensor. With
            # `return_dict_in_generate=True` the return is a
            # GenerateDecoderOnlyOutput whose [0] is the whole `sequences`
            # tensor, so `.tolist()` gave a nested list and int() raised —
            # swallowed by the except below, leaving a step with no ids, no
            # input and no output. Adopt then refused it saying the model was
            # "not on this machine" about a local gpt2 in the same process.
            # The old `hasattr(result, "__getitem__")` guard was inert: both
            # types satisfy it.
            sequences = getattr(result, "sequences", result)
            row = sequences[0]
            ids = [int(i) for i in row.tolist()]
            config = getattr(self, "config", None)
            model_id = str(
                getattr(config, "_name_or_path", "")
                or getattr(config, "name_or_path", "")
            )
            meta = {
                "model": model_id,
                "input_ids": ids,
                "n_prompt_tokens": n_prompt,
                "dtype": str(getattr(self, "dtype", "")),
                "device": str(getattr(self, "device", "")),
            }
        except Exception:
            # Instrumentation must never break the thing it instruments. A
            # step with no meta is simply one that cannot be adopted, which is
            # the same state a hosted call is in — but it is NOT the same as a
            # call that produced nothing, so the payloads below say which.
            meta = {}

        tok = getattr(self, "_modelmri_tokenizer", None)
        step(
            "llm_call",
            name=meta.get("model", "") or "transformers",
            input=_decode_span(tok, meta, 0, meta.get("n_prompt_tokens", 0))
            or "<generation recorded, but its token ids could not be read>",
            output=_decode_span(tok, meta, meta.get("n_prompt_tokens", 0), None)
            or "<generation recorded, but its token ids could not be read>",
            duration_ms=(t.now_ms() - started),
            tokens_in=meta.get("n_prompt_tokens") or None,
            tokens_out=(
                len(meta["input_ids"]) - meta["n_prompt_tokens"]
                if meta.get("input_ids")
                else None
            ),
            meta=meta,
        )
        return result

    wrapped._modelmri_wrapped = True  # type: ignore[attr-defined]
    GenerationMixin.generate = wrapped  # type: ignore[assignment]
    return True


def _decode_span(tokenizer, meta: dict, start: int, end: int | None) -> str:
    """Readable text for a slice of the recorded ids, or a stated placeholder.

    Without a tokenizer there is no honest text to show, so it says that rather
    than printing raw ids as though they were the prompt.
    """
    ids = meta.get("input_ids") or []
    if not ids:
        return ""
    span = ids[start:] if end is None else ids[start:end]
    if tokenizer is None:
        return f"<{len(span)} tokens; attach a tokenizer to see the text>"
    try:
        return tokenizer.decode(span)
    except Exception:
        return f"<{len(span)} tokens; this tokenizer could not decode them>"


def _msgs_preview(kwargs: dict) -> str:
    try:
        return _cut(json.dumps(kwargs.get("messages", []), default=str), MAX_PREVIEW_CHARS)
    except Exception:
        return ""


def _content_preview(result: object) -> str:
    try:
        blocks = getattr(result, "content", [])
        return _cut(" ".join(getattr(b, "text", "") for b in blocks), MAX_PREVIEW_CHARS)
    except Exception:
        return ""


def _undelivered_dir() -> Path:
    """Where a trace goes when the server is not listening.

    Not the working directory. This package is imported *by the user's agent*,
    so the CWD is normally their git repo, and a trace holds full prompts and
    tool output — untracked JSON of their conversations, one `git add -A` from
    being pushed to a public remote.

    MODELMRI_TRACE_DIR wins. Otherwise ask modelmri for the platform data
    directory if it happens to be installed, so both halves agree on one
    location and `modelmri where` can print it. This package stays usable
    without modelmri, hence the fallback and the local import.
    """
    if override := os.environ.get("MODELMRI_TRACE_DIR", "").strip():
        return Path(override).expanduser()
    try:
        from modelmri import paths  # type: ignore[import-not-found]

        return paths.data_dir() / "undelivered"
    except Exception:  # noqa: S110 - the next fallback IS the handling
        pass
    try:
        return Path.home() / ".modelmri" / "undelivered"
    except (RuntimeError, OSError):
        # No home either (container with no passwd entry). A temp directory
        # is a worse place to keep data than the data directory, but it is a
        # far better one than somebody's source tree.
        import tempfile

        return Path(tempfile.gettempdir()) / "modelmri-traces"


def _complain(message: str) -> None:
    """One line to stderr, or nothing at all. Never raises.

    The whole package is built so that recording cannot take down the app it
    is observing, which is why so much here is best-effort. Best-effort is not
    the same as silent, though: a trace that vanished with no explanation is
    indistinguishable from one that was never recorded. This is the one thing
    a library in someone else's process can honestly do about that.

    Guarded because the callers are shutdown paths: `sys.stderr` can be None
    (pythonw.exe, a frozen GUI build) or already closed by the time an atexit
    hook runs, and `print` raises in both cases.
    """
    try:
        stream = sys.stderr
        if stream is not None:
            print(message, file=stream)
    except Exception:  # noqa: S110 - this IS the reporter; see the docstring
        pass  # there is no third place to report to, and raising is worse


def _deliver(t: _Trace) -> None:
    if t.delivered:
        return
    t.delivered = True
    doc = t.document()
    if t.redactor is not None:
        doc = redact_document(doc, t.redactor)
    try:
        body = json.dumps(doc, default=str).encode()
    except Exception:
        return  # nothing deliverable; still must not raise
    try:
        req = urllib.request.Request(
            t.endpoint, data=body, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=3)
        _deliver_otlp(t, doc)
        return
    except Exception:  # noqa: S110 - the disk fallback below is the handling
        # Not narrowed to OSError: `endpoint` is caller-supplied, so a typo in
        # the scheme is a ValueError from urllib rather than a network error.
        # Nothing is swallowed here — the disk fallback below is the handling,
        # and the file it writes is how the trace gets imported later.
        pass
    try:
        # Where an undeliverable trace lands. It used to be a bare
        # Path("modelmri-traces"), which resolves against the CWD of whatever
        # app imported the recorder — usually somebody's git repo. Traces carry
        # full prompts and tool output, so that is untracked JSON of your
        # conversations sitting one `git add -A` from being pushed.
        #
        # MODELMRI_TRACE_DIR overrides it.

        out = _undelivered_dir()
        out.mkdir(parents=True, exist_ok=True)
        # Second-resolution stamps collide: three quick runs of the same
        # agent overwrote each other and only the last survived. The trace id
        # is already unique, so use it.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        safe = "".join(c if c.isalnum() or c in "-_." else "-" for c in t.name)[:60]
        (out / f"{safe}-{stamp}-{t.id}.json").write_text(json.dumps(doc, indent=1))
    except Exception as err:
        # Catches everything rather than the OSError family a write can be
        # expected to raise, because recording must never crash the host app —
        # that is the contract, and it outranks a tidy exception type here.
        # But the server was unreachable and now the disk has refused too, so
        # this trace is gone; saying so costs one line and is the difference
        # between a lost run and a mystery.
        _complain(f"modelmri-record: trace {t.name!r} could not be saved: {err}")
    # OUTSIDE the try, deliberately. It used to be the last statement inside
    # it, so a disk write that raised -- read-only mount, permission denied,
    # no space, a path too long on Windows -- jumped straight to the handler
    # and the collector was never tried. That is the one state where the trace
    # reaches neither destination, and it was the state that skipped the
    # second one. The comment claimed the opposite.
    _deliver_otlp(t, doc)


def _deliver_otlp(t: _Trace, doc: dict) -> None:
    """Also hand this run to an OTLP collector, if the caller asked.

    Runs AFTER the normal delivery and cannot affect it: an export is a
    convenience, and a collector being down must not cost somebody the trace
    itself. Everything here is caught for the same reason the rest of this
    module catches everything — recording must never crash the host app.

    The document handed over is the REDACTED one. Exporting the raw payloads
    to a third-party collector while the local copy is scrubbed would be the
    redactor working exactly backwards.
    """
    if not t.deliver_otlp:
        return
    try:
        from modelmri import otel  # type: ignore[import-not-found]
    except Exception:
        # Named rather than silent: the caller asked for an export and did not
        # get one, and "nothing happened" is the worst possible answer to that.
        _complain(
            f"modelmri-record: deliver_otlp was set but `modelmri` is not "
            f"importable, so trace {t.name!r} was not exported. The OTLP "
            f"mapping lives there in one table; this package stays "
            f"dependency-free. `pip install modelmri` to enable it."
        )
        return
    try:
        # The same 3 s the local delivery uses, not otel.send's 30 s default.
        # This runs in the `finally` of the caller's `with` block and again in
        # the atexit flush, so a stale address behind a firewall that DROPs
        # rather than REJECTs would hold the host app for 30 s per live trace
        # on the way out. An export is a convenience; it does not get to be
        # the slowest thing in somebody's shutdown.
        otel.send(doc, t.deliver_otlp, service_name=t.name or "modelmri", timeout=3.0)
    except (KeyboardInterrupt, SystemExit) as err:
        # Named explicitly rather than caught as BaseException. These two must
        # NOT be swallowed -- a ctrl-c during the export is the user asking to
        # stop, not an export failure -- but they are still worth a line,
        # because otherwise the traceback the user sees comes from a delivery
        # that exists to be optional and reads as the recorder crashing.
        _complain(
            f"modelmri-record: OTLP export of {t.name!r} interrupted "
            f"({type(err).__name__}); the trace itself was already recorded."
        )
        raise
    except Exception as err:
        _complain(
            f"modelmri-record: trace {t.name!r} was recorded but the OTLP "
            f"export to {t.deliver_otlp} failed: {type(err).__name__}: {err}"
        )


# Traces still open when the process ends. Flushed by the atexit hook below.
_live: list[_Trace] = []


def _flush_live() -> None:
    """Deliver anything still in flight at interpreter shutdown."""
    for t in list(_live):
        try:
            _deliver(t)
        except Exception as err:
            # NOT narrowable, and that is the finding rather than a shrug.
            # `_deliver` guards its own json.dumps, its POST and its disk
            # write, so what can still escape it is the code before those
            # guards: `redact_document` running a redactor the CALLER
            # supplied (`trace(..., redact=my_func)`) — arbitrary code with
            # arbitrary exceptions — plus interpreter-shutdown hazards, where
            # module globals may already be torn down and an attribute lookup
            # on `json` or `urllib` raises. KeyboardInterrupt is a
            # BaseException and correctly still passes straight through.
            #
            # It must not raise: this runs from atexit, so raising would mean
            # a recorder crashing the host application on its way out. It
            # must not be silent either — this is the hook that exists for
            # "the interpreter died mid-run", i.e. exactly the trace its owner
            # most wanted, and it was dropping it without a word.
            _complain(
                f"modelmri-record: trace {t.name!r} was lost at shutdown: "
                f"{type(err).__name__}: {err}"
            )


atexit.register(_flush_live)
