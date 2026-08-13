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
)

# Only the key->field direction is needed: `to_otlp` walks FIELDS in order
# (so the attribute order is stable and diffable) while `from_otlp` has to
# look each attribute up. A step->field index was written here for symmetry
# and never read, which CodeQL correctly called dead.
_BY_KEY = {f.key: f for f in FIELDS}


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

        span: dict = {
            "traceId": trace_id(tid),
            "spanId": span_id(str(step.get("id") or "")),
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
    try:
        dt = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
    except ValueError:
        return 0
    return int(dt.timestamp() * 1_000_000_000)


# ---------------------------------------------------------------- ingest


def from_otlp(body: dict) -> list[dict]:
    """OTLP spans back into recorded steps, over the same `FIELDS`.

    Scoped to spans THIS tool wrote: it reads the attributes in the table and
    nothing else. Ingesting arbitrary third-party OTLP — guessing which of a
    dozen vendor conventions a span speaks — is a separate feature and is not
    this. What this exists for is the round-trip test, which is what stops the
    two directions drifting apart as the table changes.
    """
    steps: list[dict] = []
    for resource in body.get("resourceSpans") or []:
        for scope in resource.get("scopeSpans") or []:
            for span in scope.get("spans") or []:
                step: dict = {}
                for attr in span.get("attributes") or []:
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

    def to_dict(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "spans": self.spans,
            "undated_spans": self.undated_spans,
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
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
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
        raise Refusal(
            f"{url} answered {err.code} {err.reason}. Nothing was recorded as "
            "delivered."
        ) from err
    except urllib.error.URLError as err:
        raise Refusal(
            f"could not reach {url}: {type(err).__name__}. OTLP/HTTP is "
            "usually port 4318 — 4317 is gRPC, which this does not speak."
        ) from err

    return Delivery(
        endpoint=url,
        spans=len(spans),
        undated_spans=undated,
        status=status,
        semconv=SEMCONV_GENERATION,
    )
