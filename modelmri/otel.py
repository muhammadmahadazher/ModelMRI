"""Hand a recorded run to the collector the team already runs.

This is not an "ahead" feature and should not be sold as one. It is the price
of being adoptable: every competitor in the tracing category ingests, and a
local tool that cannot hand its traces onward is a dead end for anybody with
an existing stack. Langfuse, Phoenix, Grafana and Honeycomb all speak OTLP.

Three things make this honest rather than merely working.

**One table, both directions.** `FIELDS` is the single mapping between a
recorded step and its OTLP attributes, and `to_otlp` and `from_otlp` both read
it. Two hand-written mappings drift — one gains a key, the other does not, and
a round-trip silently loses a column. `test_otel.py` round-trips a document
through both and compares every field, so the table is the thing under test
rather than either function.

**The vocabulary is stamped on every span.** `gen_ai.*` was deprecated out of
the main semantic-conventions repo on 2026-06-12 into a `-genai` repo that has
no releases, no tags and nothing marked stable. Emitting against a moving
target is a maintenance obligation that does not go away, and the only thing
that makes it survivable is that a consumer can always tell which generation a
span speaks: every span carries `modelmri.semconv.generation`, and the CLI
prints it. When the vocabulary moves, old exports stay readable because they
say what they were written against.

**JSON only.** OTLP/HTTP with a JSON body, over stdlib `urllib`. No new
dependency, which is what keeps `modelmri-record` importable into somebody
else's agent without dragging the OpenTelemetry SDK along. A collector that
accepts only protobuf is refused with a sentence naming the limit rather than
approximated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .errors import BadRequest, Refusal

# The generation of the GenAI semantic conventions these attribute names were
# written against, stamped on every span and printed by the CLI.
#
# Not a version number, because there is no released version to cite: the
# `gen_ai.*` conventions were moved out of open-telemetry/semantic-conventions
# on 2026-06-12 and the repo they moved to has published no releases and no
# tags. A date is the most precise honest thing available. If this reads as
# unsatisfying, that is the actual state of the specification and pretending
# otherwise with a "1.0" would be worse.
SEMCONV_GENERATION = "gen_ai@2026-06-12"

# OTLP span kinds. Only two are used and both are named rather than left as
# integers at the call site.
SPAN_KIND_INTERNAL = 1
SPAN_KIND_CLIENT = 3

# OTLP status codes.
STATUS_UNSET = 0
STATUS_ERROR = 2

# What a recorded `kind` becomes. `gen_ai.operation.name` is the attribute a
# consumer groups by, so a step kind this table does not know must NOT be
# silently mapped to something plausible — it keeps its own name and is
# reported, because a `subagent` filed under `chat` is a wrong answer that
# looks like a right one.
OPERATION = {
    "llm_call": ("chat", SPAN_KIND_CLIENT),
    "tool_call": ("execute_tool", SPAN_KIND_CLIENT),
    "subagent": ("invoke_agent", SPAN_KIND_INTERNAL),
    "mcp_call": ("execute_tool", SPAN_KIND_CLIENT),
    "user_turn": ("chat", SPAN_KIND_INTERNAL),
    "error": ("chat", SPAN_KIND_INTERNAL),
}


@dataclass(frozen=True)
class Field:
    """One column of a recorded step, and the span attribute it becomes.

    `key` is the OTLP attribute name. `step` is the key in the trace document.
    `kind` decides the OTLP value wrapper AND how `from_otlp` reads it back,
    so the two directions cannot disagree about the type.
    """

    step: str
    key: str
    kind: str  # "str" | "int" | "bool"


# The whole mapping. Adding a row extends BOTH directions, which is the point.
FIELDS: tuple[Field, ...] = (
    Field("name", "gen_ai.tool.name", "str"),
    Field("input", "gen_ai.input.messages", "str"),
    Field("output", "gen_ai.output.messages", "str"),
    Field("tokens_in", "gen_ai.usage.input_tokens", "int"),
    Field("tokens_out", "gen_ai.usage.output_tokens", "int"),
    # The other three counts the store holds. They arrived with `ledger.py`
    # and never reached this table, so an export dropped them -- and it did so
    # invisibly, for exactly the reason the comment below records about
    # parent_id and started_ms: the round-trip test walks FIELDS, so a column
    # missing from FIELDS is both unexported and untested by construction.
    #
    # Under `modelmri.*` rather than `gen_ai.*`: cache and reasoning counts
    # are not in a stable semconv, and this file's own rule is not to squat on
    # a vocabulary nobody agreed to. A collector that does not know them
    # carries them as ordinary attributes; ModelMRI reads them back exactly.
    Field("tokens_cache_read", "modelmri.usage.cache_read_tokens", "int"),
    Field("tokens_cache_write", "modelmri.usage.cache_write_tokens", "int"),
    Field("tokens_reasoning", "modelmri.usage.reasoning_tokens", "int"),
    # ModelMRI's own, namespaced because they are not in anybody's semconv and
    # squatting on `gen_ai.*` for them would make this file the source of a
    # vocabulary nobody agreed to.
    Field("kind", "modelmri.step.kind", "str"),
    Field("seq", "modelmri.step.seq", "int"),
    Field("truncated_in", "modelmri.truncated.input", "bool"),
    Field("truncated_out", "modelmri.truncated.output", "bool"),
    # Whether a duration was recorded at all. See `_span_times`.
    Field("_duration_recorded", "modelmri.duration.recorded", "bool"),
    # The step's real id. The OTLP span id is DERIVED (see `_span_id`), so
    # without this the identity a `.mri` file uses would not survive export.
    Field("id", "modelmri.step.id", "str"),
    # These three were emitted by `to_otlp` and never read back, so a round
    # trip silently destroyed the call tree, stacked every step at t=0, and
    # lost the model. They were invisible to the round-trip test for the
    # reason they were broken: the test walks FIELDS, and they were not in it.
    #
    # `parentSpanId` is a one-way sha256 and cannot be inverted, so the real
    # parent id has to ride as an attribute of its own -- the span link is for
    # the collector, this is for us.
    Field("parent_id", "modelmri.step.parent_id", "str"),
    # Emitted into startTimeUnixNano as an offset from the trace start, which
    # `from_otlp` has no way to subtract back out: it sees one span, not the
    # trace header.
    Field("started_ms", "modelmri.step.started_ms", "int"),
)

# Only the key->field direction is needed: `to_otlp` walks FIELDS in order
# (so the attribute order is stable and diffable) while `from_otlp` has to
# look each attribute up. A step->field index was written here for symmetry
# and never read, which CodeQL correctly called dead.
_BY_KEY = {f.key: f for f in FIELDS}


def _status_phrase(code: object) -> str:
    """ " Not Found" for 404, or "" for a code the stdlib does not know.

    Looked up from the integer rather than read off the response, so the text
    in a refusal is always this machine's own. Empty for an unknown code —
    a made-up phrase would be worse than none.
    """
    import http

    try:
        return f" {http.HTTPStatus(int(code)).phrase}"
    except (ValueError, TypeError):
        return ""


# ------------------------------------------------------------------- ids


def _hex_id(value: str, *, nbytes: int) -> str:
    """A stable OTLP id derived from an arbitrary string.

    OTLP wants 8-byte span ids and 16-byte trace ids; a recorded id is a
    string of whatever shape the recorder chose. Hashing is deterministic, so
    re-exporting the same run twice produces the same ids and a collector
    treats the second as the same trace rather than a duplicate.

    The original is not thrown away — it rides as `modelmri.step.id`.
    """
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return digest[:nbytes].hex()


def span_id(step_id: str) -> str:
    return _hex_id(step_id, nbytes=8)


def trace_id(tid: str) -> str:
    return _hex_id(tid, nbytes=16)


# ------------------------------------------------------------- attributes


def _value(kind: str, raw: Any) -> dict:
    if kind == "int":
        return {"intValue": str(int(raw))}
    if kind == "bool":
        return {"boolValue": bool(raw)}
    return {"stringValue": str(raw)}


def _read(kind: str, wrapper: dict) -> Any:
    if kind == "int":
        return int(wrapper.get("intValue", 0))
    if kind == "bool":
        return bool(wrapper.get("boolValue", False))
    return str(wrapper.get("stringValue", ""))


def _attributes(step: dict) -> list[dict]:
    """Every mapped field that has a value. A None is OMITTED, not zeroed.

    `tokens_in` is nullable and so is `duration_ms`. Emitting a missing token
    count as 0 would put "this call used no input tokens" into somebody's
    dashboard, which is a claim rather than an absence — the same shape as the
    `.get(name, 0.0)` that made 206 robot episodes show one video.
    """
    out = []
    for f in FIELDS:
        raw = step.get(f.step)
        if raw is None:
            continue
        out.append({"key": f.key, "value": _value(f.kind, raw)})
    return out


# ----------------------------------------------------------------- times


def _span_times(step: dict, trace_start_ns: int) -> tuple[int, int, bool]:
    """(start, end, duration_was_recorded), all nanoseconds.

    OTLP has no way to say "this span's end time is unknown": `endTimeUnixNano`
    is required and a collector will render whatever is there. A step with no
    recorded duration therefore has to be emitted as zero-length, and a
    zero-length span reads as "this operation was instantaneous" — a claim
    about something nobody measured.

    So it is emitted, and marked: `modelmri.duration.recorded=false` rides on
    the span and the CLI prints how many spans carry it. That is the most the
    wire format allows. Dropping the step instead would lose it entirely,
    which is worse than an honest zero with a flag on it.
    """
    start = trace_start_ns + int(step.get("started_ms") or 0) * 1_000_000
    ms = step.get("duration_ms")
    if ms is None:
        return start, start, False
    return start, start + int(ms) * 1_000_000, True


# ------------------------------------------------------------------ emit


def to_otlp(doc: dict, *, service_name: str = "modelmri") -> dict:
    """A recorded trace document as an OTLP/HTTP JSON request body.

    The inverse of `from_otlp` over `FIELDS`, which is what the round-trip
    test pins. Times are strings because proto3's JSON mapping encodes uint64
    that way, and a collector handed a JSON number for a nanosecond timestamp
    will either lose precision or reject it.
    """
    from . import __version__

    tid = str(doc.get("id") or "")
    if not tid:
        raise BadRequest("this trace has no id, so it cannot be addressed")
    steps = doc.get("steps") or []
    trace_start_ns = _epoch_ns(doc.get("started_at"))

    spans = []
    for step in steps:
        start, end, recorded = _span_times(step, trace_start_ns)
        kind = str(step.get("kind") or "")
        operation, span_kind = OPERATION.get(kind, (kind, SPAN_KIND_INTERNAL))
        attrs = _attributes({**step, "_duration_recorded": recorded})
        attrs.append(
            {"key": "gen_ai.operation.name", "value": {"stringValue": operation}}
        )
        attrs.append(
            {
                "key": "modelmri.semconv.generation",
                "value": {"stringValue": SEMCONV_GENERATION},
            }
        )
        model = (step.get("meta") or {}).get("model")
        if model:
            attrs.append(
                {"key": "gen_ai.request.model", "value": {"stringValue": str(model)}}
            )

        step_id = str(step.get("id") or "")
        if not step_id:
            # `_hex_id("")` is a constant, so every id-less step would get the
            # SAME span id -- invalid OTLP (span ids must be unique within a
            # trace) and a collector collapses or orphans them. `to_otlp`
            # already refuses a trace with no id; a step with none is the same
            # problem one level down, and deriving a valid-looking id from
            # nothing is exactly what this package refuses to do elsewhere.
            raise BadRequest(
                f"step {step.get('seq', '?')} of trace {tid!r} has no id, so it "
                "cannot be given a span id. Every step in a recorded trace has "
                "one; a document without them was not written by this tool."
            )
        span: dict = {
            "traceId": trace_id(tid),
            "spanId": span_id(step_id),
            "name": str(step.get("name") or kind or "step"),
            "kind": span_kind,
            "startTimeUnixNano": str(start),
            "endTimeUnixNano": str(end),
            "attributes": attrs,
        }
        parent = step.get("parent_id")
        if parent:
            span["parentSpanId"] = span_id(str(parent))
        if step.get("error"):
            # No message: the recorder stores a flag, not text, and inventing
            # one here would put a sentence nobody wrote into a dashboard.
            span["status"] = {"code": STATUS_ERROR}
        else:
            span["status"] = {"code": STATUS_UNSET}
        spans.append(span)

    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {
                            "key": "service.name",
                            "value": {"stringValue": service_name},
                        },
                        {
                            "key": "modelmri.trace.id",
                            "value": {"stringValue": tid},
                        },
                        {
                            "key": "modelmri.trace.name",
                            "value": {"stringValue": str(doc.get("name") or "")},
                        },
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "modelmri", "version": __version__},
                        "spans": spans,
                    }
                ],
            }
        ]
    }


def _epoch_ns(started_at: Any) -> int:
    """The trace's wall-clock start, in nanoseconds since the epoch.

    `started_ms` on a step is an offset from this. 0 when the header cannot be
    parsed, and that is said rather than hidden: every span then sits in 1970,
    which is obviously wrong on a timeline instead of subtly wrong.
    """
    from datetime import datetime

    if not started_at:
        return 0
    from datetime import timezone

    try:
        dt = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return 0
    if dt.tzinfo is None:
        # `dt.timestamp()` on a naive datetime interprets it as LOCAL time, so
        # every span shifts by the machine's UTC offset with nothing saying
        # so. This tool writes offset-aware stamps; an imported trace may not,
        # and UTC is the only defensible reading of a bare ISO timestamp here.
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        return int(dt.timestamp() * 1_000_000_000)
    except (OverflowError, OSError, ValueError):
        # A pre-1970 or absurd date. 0 rather than a crash, and `send` reports
        # `epoch_fallback` so it is not discovered days later in a timeline.
        return 0


# ---------------------------------------------------------------- ingest


def _as_list(value) -> list:
    """A list, or empty for anything that is not one.

    OTLP bodies are written by other people's exporters, so a field holding
    the wrong type is ordinary rather than hostile. `value or []` only covers
    a missing key.
    """
    return value if isinstance(value, list) else []


def from_otlp(body: dict) -> list[dict]:
    """OTLP spans back into recorded steps, over the same `FIELDS`.

    Scoped to spans THIS tool wrote: it reads the attributes in the table and
    nothing else. Ingesting arbitrary third-party OTLP — guessing which of a
    dozen vendor conventions a span speaks — is a separate feature and is not
    this. What this exists for is the round-trip test, which is what stops the
    two directions drifting apart as the table changes.
    """
    steps: list[dict] = []
    # Same guard as the ingest path, and it belongs here too: this is a
    # public function, and `or []` covers a MISSING key rather than a present
    # one holding the wrong type. Fixing the route and leaving this is how the
    # class survives — a caller reaching `from_otlp` directly got the same
    # `TypeError: 'int' object is not iterable`.
    for resource in _as_list(body.get("resourceSpans")):
        if not isinstance(resource, dict):
            continue
        for scope in _as_list(resource.get("scopeSpans")):
            if not isinstance(scope, dict):
                continue
            for span in _as_list(scope.get("spans")):
                if not isinstance(span, dict):
                    continue
                step: dict = {}
                for attr in _as_list(span.get("attributes")):
                    field = _BY_KEY.get(attr.get("key", ""))
                    if field is None:
                        continue
                    step[field.step] = _read(field.kind, attr.get("value") or {})
                # Reconstructed from the times rather than from an attribute:
                # duration is the one field that lives in the span envelope.
                if step.pop("_duration_recorded", False):
                    start = int(span.get("startTimeUnixNano") or 0)
                    end = int(span.get("endTimeUnixNano") or 0)
                    step["duration_ms"] = (end - start) // 1_000_000
                else:
                    step["duration_ms"] = None
                step["error"] = (span.get("status") or {}).get("code") == STATUS_ERROR
                # Read back into `meta`, where `to_otlp` took it from, so the
                # round trip is symmetric rather than one-way.
                for attr in span.get("attributes") or []:
                    if attr.get("key") == "gen_ai.request.model":
                        model = (attr.get("value") or {}).get("stringValue")
                        if model:
                            step.setdefault("meta", {})["model"] = model
                steps.append(step)
    return steps


# ------------------------------------------------------------------ send


@dataclass
class Delivery:
    endpoint: str
    spans: int
    undated_spans: int
    status: int
    semconv: str
    # What the collector said it REJECTED. OTLP returns
    # ExportTraceServiceResponse.partialSuccess with HTTP 200, and a collector
    # over quota or running a filter uses it to say it dropped spans while
    # still answering 200. Reporting `spans` as delivered without reading this
    # is claiming something never measured -- "17 spans -> endpoint (HTTP 200)"
    # when the collector kept none of them.
    #
    # None means the response carried no partialSuccess, which is the normal
    # full-success case. 0 means it said so explicitly. They are different.
    rejected_spans: int | None = None
    reject_message: str = ""
    # Timestamps land at the epoch when the trace header could not be parsed.
    # Counted so the CLI can say it, because 1970 in a collector is discovered
    # days later by someone squinting at a timeline.
    epoch_fallback: bool = False

    @property
    def accepted(self) -> int:
        """Spans the collector did not say it rejected."""
        return self.spans - (self.rejected_spans or 0)

    def to_dict(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "spans": self.spans,
            "accepted": self.accepted,
            "rejected_spans": self.rejected_spans,
            "reject_message": self.reject_message,
            "undated_spans": self.undated_spans,
            "epoch_fallback": self.epoch_fallback,
            "status": self.status,
            "semconv": self.semconv,
        }


def normalise_endpoint(endpoint: str) -> str:
    """Accept a base URL or a full traces path; return the full path.

    Collectors are configured both ways in the wild and the difference is a
    404 twenty seconds later, so it is resolved here rather than left as a
    documentation problem.
    """
    e = (endpoint or "").strip().rstrip("/")
    if not e:
        raise BadRequest("no endpoint given")
    if not e.startswith(("http://", "https://")):
        raise BadRequest(
            f"{endpoint!r} is not an http(s) URL. OTLP/HTTP wants something "
            "like http://localhost:4318 or http://localhost:4318/v1/traces."
        )
    return e if e.endswith("/v1/traces") else e + "/v1/traces"


def send(
    doc: dict,
    endpoint: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    service_name: str = "modelmri",
) -> Delivery:
    """POST one trace as OTLP/HTTP JSON. Stdlib only, by design.

    No OpenTelemetry SDK: `modelmri-record` is imported into other people's
    agents and a dependency there is a liability, so the wire format is
    written by hand and this stays a `urllib` call.
    """
    import urllib.error
    import urllib.request

    url = normalise_endpoint(endpoint)
    body = to_otlp(doc, service_name=service_name)
    spans = body["resourceSpans"][0]["scopeSpans"][0]["spans"]
    undated = sum(
        1
        for s in spans
        for a in s["attributes"]
        if a["key"] == "modelmri.duration.recorded"
        and a["value"].get("boolValue") is False
    )

    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            **(headers or {}),
        },
    )
    rejected: int | None = None
    reject_message = ""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            # The body is not noise. A 200 can still carry partialSuccess.
            try:
                answer = json.loads(response.read() or b"{}")
                partial = (answer or {}).get("partialSuccess") or {}
                if partial:
                    rejected = int(partial.get("rejectedSpans", 0))
                    reject_message = str(partial.get("errorMessage", ""))
            except Exception:  # noqa: S110
                # A collector that answers 200 with something unparseable has
                # still accepted the spans as far as HTTP is concerned. The
                # count stays None -- "it said nothing" -- rather than 0,
                # which would be "it said it rejected none".
                pass
    except urllib.error.HTTPError as err:
        if err.code == 415:
            raise Refusal(
                f"{url} refused a JSON body (415). This sends OTLP/HTTP with "
                "JSON and does not speak protobuf — that would mean either a "
                "generated stub set or the OpenTelemetry SDK as a dependency, "
                "and `modelmri-record` is stdlib-only on purpose because it "
                "gets imported into other people's agents.\n"
                "Most collectors accept JSON on the same port once "
                "`otlp/http` is enabled; the OpenTelemetry Collector does by "
                "default."
            ) from err
        # The status phrase derived LOCALLY from the code, not echoed from the
        # response. `HTTPError.reason` is whatever the remote server chose to
        # put there — an endpoint the user named, but still a third party
        # writing text into a sentence this project publishes. The URLError
        # branch immediately below already made this call with
        # `type(err).__name__`; this one had been left behind, and a widened
        # leak check found it after CodeQL found its sibling in `policy.py`.
        #
        # `err.code` is an int and keeps everything diagnostic about the
        # message: 404 and 401 send you to different places, and the standard
        # phrase for each is a lookup rather than a quotation.
        phrase = _status_phrase(err.code)
        raise Refusal(
            f"{url} answered {err.code}{phrase}. Nothing was recorded as delivered."
        ) from err
    except urllib.error.URLError as err:
        raise Refusal(
            f"could not reach {url}: {type(err).__name__}. OTLP/HTTP is "
            "usually port 4318 — 4317 is gRPC, which this does not speak."
        ) from err
    except (TimeoutError, OSError, ValueError) as err:
        # NOT covered by the arms above, and each is reachable:
        #
        # urllib wraps only `h.request(...)` in `except OSError: raise
        # URLError`. `h.getresponse()` sits OUTSIDE that wrapper, so a
        # collector that accepts the body and never answers -- the standard
        # overloaded-collector failure -- raises a bare TimeoutError, which is
        # a sibling of URLError and not a subclass. Verified against the
        # installed CPython. `http.client.RemoteDisconnected` takes the same
        # route, and `putheader` raises ValueError on a header value this
        # module should have rejected earlier.
        raise Refusal(
            f"{url} did not complete the exchange: {type(err).__name__}. "
            "Nothing was recorded as delivered."
        ) from err

    return Delivery(
        endpoint=url,
        spans=len(spans),
        undated_spans=undated,
        status=status,
        semconv=SEMCONV_GENERATION,
        rejected_spans=rejected,
        reject_message=reject_message,
        epoch_fallback=_epoch_ns(doc.get("started_at")) == 0,
    )


# ---------------------------------------------------------------- ingest


# What a foreign `gen_ai.operation.name` becomes. The inverse of OPERATION,
# plus the spellings other producers actually emit. Anything not here keeps
# its own name in `meta.otel_operation` and is filed as `tool_call` -- "an
# operation ran" -- because VALID_KINDS is closed and inventing a closer fit
# would be filing somebody's span under a claim they never made.
OPERATION_TO_KIND = {
    "chat": "llm_call",
    "text_completion": "llm_call",
    "generate_content": "llm_call",
    "embeddings": "tool_call",
    "execute_tool": "tool_call",
    "invoke_agent": "subagent",
    "create_agent": "subagent",
}

# Where other producers put the prompt, the completion and the token counts.
# One tuple per field, tried in order, because there is no single spelling:
# OpenLLMetry writes `gen_ai.prompt`, OpenInference writes `input.value`, the
# semconv writes `gen_ai.input.messages`, the Vercel AI SDK writes `ai.prompt`.
# First hit wins AND the key that matched is recorded, so a reader can tell
# which vocabulary a step was read through rather than trusting that it was
# read at all.
INPUT_KEYS = (
    "gen_ai.input.messages",
    "gen_ai.prompt",
    "input.value",
    "llm.input_messages",
    "ai.prompt",
)
OUTPUT_KEYS = (
    "gen_ai.output.messages",
    "gen_ai.completion",
    "output.value",
    "llm.output_messages",
    "ai.response.text",
)
TOKENS_IN_KEYS = (
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.prompt_tokens",
    "llm.token_count.prompt",
    "ai.usage.promptTokens",
)
TOKENS_OUT_KEYS = (
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.completion_tokens",
    "llm.token_count.completion",
    "ai.usage.completionTokens",
)
MODEL_KEYS = ("gen_ai.request.model", "gen_ai.response.model", "llm.model_name")


def _flat(attributes: list | None) -> dict:
    """OTLP's `[{key, value: {...Value}}]` as a plain dict.

    Every wrapper shape is unwrapped, because a foreign producer picks them
    and an attribute this does not understand is an attribute that silently
    is not there.
    """
    out: dict = {}
    for attr in attributes or []:
        if not isinstance(attr, dict):
            continue
        key, value = attr.get("key"), attr.get("value")
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        for wrapper, cast in (
            ("stringValue", str),
            ("intValue", int),
            ("doubleValue", float),
            ("boolValue", bool),
        ):
            if wrapper in value:
                try:
                    out[key] = cast(value[wrapper])
                except (TypeError, ValueError):
                    # A producer that wrote a non-numeric intValue. Kept as
                    # its text rather than dropped or zeroed.
                    out[key] = str(value[wrapper])
                break
        else:
            if "arrayValue" in value:
                out[key] = json.dumps(value["arrayValue"])
    return out


def _first(attrs: dict, keys: tuple, cast=None):
    """The first key present, and which one it was. `(None, None)` if absent."""
    for key in keys:
        if key in attrs:
            value = attrs[key]
            if cast is not None:
                try:
                    value = cast(value)
                except (TypeError, ValueError):
                    continue
            return value, key
    return None, None


def ingest(payload: dict) -> dict:
    """A foreign OTLP/HTTP JSON body as a trace document.

    Takes what OpenLLMetry, OpenInference, the Vercel AI SDK or a plain
    OpenTelemetry SDK actually send, so a team that is already instrumented
    does not need this project to write a provider integration for them.

    **Which vocabulary a span spoke is recorded, never assumed.** A span this
    tool wrote carries `modelmri.semconv.generation`; a foreign one usually
    carries nothing, and that is stored as "unstated" rather than backfilled
    with our own pin. `gen_ai.*` left the main semantic-conventions repo on
    2026-06-12 and the names have churned since, so "I do not know which
    generation this is" is a real and common answer, and the honest one.
    """
    if not isinstance(payload, dict):
        raise BadRequest("an OTLP body is a JSON object carrying `resourceSpans`")
    resource_spans = payload.get("resourceSpans")
    if not isinstance(resource_spans, list) or not resource_spans:
        raise BadRequest(
            "this body has no `resourceSpans`, so it is not OTLP/HTTP JSON."
        )

    spans: list[dict] = []
    service = ""
    for resource in resource_spans:
        if not isinstance(resource, dict):
            continue
        res_attrs = _flat((resource.get("resource") or {}).get("attributes"))
        service = service or str(res_attrs.get("service.name") or "")
        # `or []` catches a MISSING key, not a present one holding the wrong
        # type: `5 or []` is 5, and iterating an int is a TypeError. Each
        # ELEMENT was already guarded with `isinstance(...): continue`; the
        # containers around them were not, so an exporter that sent
        # `"scopeSpans": 5` got a 500 from a route whose whole job is to
        # accept a body written by somebody else's software.
        #
        # Skipped rather than refused on the spot, to match how a non-dict
        # element is already treated — and a body with nothing usable left in
        # it then lands on the authored "carries no spans" refusal below,
        # which is the sentence this route's docstring promises.
        for scope in _as_list(resource.get("scopeSpans")):
            if not isinstance(scope, dict):
                continue
            for span in _as_list(scope.get("spans")):
                if isinstance(span, dict):
                    spans.append(span)

    if not spans:
        raise BadRequest("this OTLP body carries no spans")

    # The trace starts when its earliest span does; every step's `started_ms`
    # is an offset from that, which is what the timeline lays out.
    starts = []
    for span in spans:
        try:
            starts.append(int(span.get("startTimeUnixNano") or 0))
        except (TypeError, ValueError):
            continue
    base_ns = min([s for s in starts if s > 0], default=0)

    notes: list[str] = []
    generations: set[str] = set()
    unmapped: set[str] = set()
    steps: list[dict] = []

    ordered = sorted(spans, key=lambda s: str(s.get("startTimeUnixNano") or "0"))
    for seq, span in enumerate(ordered):
        attrs = _flat(span.get("attributes"))
        generations.add(str(attrs.get("modelmri.semconv.generation") or "unstated"))

        operation = str(attrs.get("gen_ai.operation.name") or "")
        kind = OPERATION_TO_KIND.get(operation)
        if kind is None:
            kind = "tool_call"
            if operation:
                unmapped.add(operation)

        try:
            start_ns = int(span.get("startTimeUnixNano") or 0)
            end_ns = int(span.get("endTimeUnixNano") or 0)
        except (TypeError, ValueError):
            start_ns = end_ns = 0

        # An end equal to the start is how OTLP is forced to express "unknown",
        # so it reads back as unknown rather than as a measured zero -- unless
        # the span explicitly says a duration WAS recorded, which is the one
        # case where zero is a measurement.
        duration = None
        if end_ns > start_ns:
            duration = (end_ns - start_ns) // 1_000_000
        elif attrs.get("modelmri.duration.recorded") is True:
            duration = 0

        text_in, in_key = _first(attrs, INPUT_KEYS)
        text_out, out_key = _first(attrs, OUTPUT_KEYS)
        tokens_in, _ = _first(attrs, TOKENS_IN_KEYS, cast=int)
        tokens_out, _ = _first(attrs, TOKENS_OUT_KEYS, cast=int)
        model, _ = _first(attrs, MODEL_KEYS)

        # `step.id` is the PRIMARY KEY of the whole table, and an OTLP span id
        # is only unique WITHIN its trace -- so importing two traces that share
        # one is a UNIQUE constraint failure and a 500, which is exactly what
        # happened the first time this ran against a second body. Namespaced by
        # the OTLP trace id, which also keeps it deterministic: re-importing
        # the same export produces the same ids rather than duplicating it.
        #
        # One body can carry spans from several traces, so the pair is taken
        # per span rather than once for the request.
        otel_trace = str(span.get("traceId") or "")
        raw_span = str(span.get("spanId") or "")

        def _key(sid: str) -> str:
            return f"{otel_trace}:{sid}" if sid else ""

        meta: dict = {"otel_span_id": raw_span, "otel_trace_id": otel_trace}
        if model:
            meta["model"] = str(model)
        if operation:
            meta["otel_operation"] = operation
        if in_key or out_key:
            # Which spelling matched. Two producers write the prompt four
            # different ways, and a reader deserves to know which one this
            # step was read through.
            meta["otel_keys"] = {"input": in_key, "output": out_key}

        steps.append(
            {
                "id": _key(raw_span) or f"{otel_trace or 'otlp'}:span-{seq}",
                # The same scheme, or the tree breaks: a parent named by its
                # bare span id would point at nothing.
                "parent_id": _key(str(span.get("parentSpanId") or "")) or None,
                "kind": kind,
                "name": str(span.get("name") or operation or "span"),
                "started_ms": max(0, (start_ns - base_ns) // 1_000_000),
                "duration_ms": duration,
                "input": "" if text_in is None else str(text_in),
                "output": "" if text_out is None else str(text_out),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "error": (span.get("status") or {}).get("code") == STATUS_ERROR,
                "seq": seq,
                "meta": meta,
            }
        )

    if unmapped:
        notes.append(
            "these operations have no ModelMRI step kind and were filed as "
            "tool_call, with the original kept in meta.otel_operation: "
            + ", ".join(sorted(unmapped))
        )
    if generations == {"unstated"}:
        notes.append(
            "no span said which semantic-convention generation it was written "
            "against, so the attribute names were matched by trying several "
            "spellings in order. gen_ai.* left the main semconv repo on "
            "2026-06-12 and has churned since, so a span that does not say "
            "which vocabulary it speaks cannot be read as though it did."
        )

    started_at = (
        datetime.fromtimestamp(base_ns / 1e9, tz=timezone.utc).isoformat()
        if base_ns
        else datetime.now(timezone.utc).isoformat()
    )
    # A DETERMINISTIC trace id, derived from the OTLP trace ids in the body.
    #
    # Without one, `import_trace` mints a fresh uuid per call while the step
    # ids stay deterministic -- so exporting the same run twice collided on
    # `step.id`, which is the table's primary key, and answered 500. With one,
    # `import_trace`'s existing INSERT OR REPLACE does the right thing: the
    # same export twice is the SAME trace, replaced, which is also what a
    # collector receiving a retry should do.
    #
    # Truncated to the width the store's own ids use. One body can carry spans
    # from several OTLP traces, so the id covers all of them rather than
    # whichever happened to be first.
    otel_trace_ids = sorted(
        {str(s.get("traceId") or "") for s in spans if s.get("traceId")}
    )
    trace_key = hashlib.sha256("|".join(otel_trace_ids).encode()).hexdigest()[:12]
    if len(otel_trace_ids) > 1:
        notes.append(
            f"this body carried {len(otel_trace_ids)} OTLP traces and they are "
            "imported as one ModelMRI trace, because that is what arrived in "
            "one export."
        )

    return {
        "id": trace_key,
        "name": service or "imported OTLP trace",
        "started_at": started_at,
        "meta": {
            "source": "otlp",
            "service": service,
            # Stored and shown in the agents panel. Plural because one body can
            # carry spans from several producers.
            "semconv": ", ".join(sorted(generations)),
            "spans": len(steps),
            "otel_trace_ids": otel_trace_ids,
            "notes": notes,
        },
        "steps": steps,
    }
