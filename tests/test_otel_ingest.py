"""Reading OTLP that somebody else's instrumentation produced.

The point of ingest is that a team already running OpenLLMetry, OpenInference
or the Vercel AI SDK does not need this project to write a provider
integration for them. So the fixtures below are the shapes those producers
actually emit, not the shape `to_otlp` writes — a reader tested only against
its own writer is a round-trip test wearing a different hat.

The honesty rule specific to this file: `gen_ai.*` left the main
semantic-conventions repo on 2026-06-12 and the names have churned since. A
span that does not say which generation it speaks must be recorded as not
having said, never backfilled with our own pin.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from modelmri import otel
from modelmri.errors import BadRequest
from modelmri.server import create_app


def _span(
    name, attrs, *, start=1_000_000_000_000_000_000, end=None, span_id="aa", parent=None
):
    span = {
        "traceId": "0" * 32,
        "spanId": span_id,
        "name": name,
        "kind": 3,
        "startTimeUnixNano": str(start),
        "endTimeUnixNano": str(end if end is not None else start),
        "attributes": [{"key": k, "value": v} for k, v in attrs.items()],
        "status": {"code": 0},
    }
    if parent:
        span["parentSpanId"] = parent
    return span


def _body(spans, service="checkout-agent"):
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": service}}
                    ]
                },
                "scopeSpans": [{"scope": {"name": "x"}, "spans": spans}],
            }
        ]
    }


S = lambda v: {"stringValue": v}  # noqa: E731
I = lambda v: {"intValue": str(v)}  # noqa: E731, E741


# ------------------------------------------------- the producers in the wild


def test_the_semconv_spelling_reads():
    doc = otel.ingest(
        _body(
            [
                _span(
                    "chat gpt-4",
                    {
                        "gen_ai.operation.name": S("chat"),
                        "gen_ai.input.messages": S("hello"),
                        "gen_ai.output.messages": S("hi"),
                        "gen_ai.usage.input_tokens": I(11),
                        "gen_ai.usage.output_tokens": I(3),
                        "gen_ai.request.model": S("gpt-4"),
                    },
                    end=1_000_000_001_500_000_000,
                )
            ]
        )
    )
    step = doc["steps"][0]
    assert step["kind"] == "llm_call"
    assert step["input"] == "hello" and step["output"] == "hi"
    assert step["tokens_in"] == 11 and step["tokens_out"] == 3
    assert step["duration_ms"] == 1500
    assert step["meta"]["model"] == "gpt-4"


def test_the_openllmetry_spelling_reads():
    """`gen_ai.prompt` / `gen_ai.completion` / `...prompt_tokens`."""
    doc = otel.ingest(
        _body(
            [
                _span(
                    "openai.chat",
                    {
                        "gen_ai.operation.name": S("chat"),
                        "gen_ai.prompt": S("p"),
                        "gen_ai.completion": S("c"),
                        "gen_ai.usage.prompt_tokens": I(7),
                        "gen_ai.usage.completion_tokens": I(2),
                    },
                )
            ]
        )
    )
    step = doc["steps"][0]
    assert (step["input"], step["output"]) == ("p", "c")
    assert (step["tokens_in"], step["tokens_out"]) == (7, 2)


def test_the_openinference_spelling_reads():
    """`input.value` / `output.value` / `llm.token_count.*`."""
    doc = otel.ingest(
        _body(
            [
                _span(
                    "LLM",
                    {
                        "gen_ai.operation.name": S("chat"),
                        "input.value": S("iv"),
                        "output.value": S("ov"),
                        "llm.token_count.prompt": I(4),
                        "llm.model_name": S("claude"),
                    },
                )
            ]
        )
    )
    step = doc["steps"][0]
    assert (step["input"], step["output"]) == ("iv", "ov")
    assert step["tokens_in"] == 4
    assert step["meta"]["model"] == "claude"


def test_the_vercel_ai_sdk_spelling_reads():
    doc = otel.ingest(
        _body(
            [
                _span(
                    "ai.generateText",
                    {
                        "gen_ai.operation.name": S("chat"),
                        "ai.prompt": S("vp"),
                        "ai.response.text": S("vr"),
                        "ai.usage.promptTokens": I(9),
                    },
                )
            ]
        )
    )
    step = doc["steps"][0]
    assert (step["input"], step["output"], step["tokens_in"]) == ("vp", "vr", 9)


def test_which_spelling_matched_is_recorded():
    """Four producers write the prompt four ways. A reader deserves to know
    which one this step was read through rather than trusting it was read."""
    doc = otel.ingest(
        _body([_span("x", {"gen_ai.operation.name": S("chat"), "input.value": S("i")})])
    )
    assert doc["steps"][0]["meta"]["otel_keys"]["input"] == "input.value"


# ------------------------------------------------------------ the honesty


def test_a_span_that_does_not_state_its_generation_is_recorded_as_unstated():
    """Never backfilled with our own pin. `gen_ai.*` has churned, so "I do not
    know which vocabulary this is" is a real and common answer."""
    doc = otel.ingest(_body([_span("x", {"gen_ai.operation.name": S("chat")})]))
    assert doc["meta"]["semconv"] == "unstated"
    assert any("does not say which vocabulary" in n for n in doc["meta"]["notes"])


def test_a_span_that_states_its_generation_keeps_it():
    doc = otel.ingest(
        _body(
            [
                _span(
                    "x",
                    {
                        "gen_ai.operation.name": S("chat"),
                        "modelmri.semconv.generation": S(otel.SEMCONV_GENERATION),
                    },
                )
            ]
        )
    )
    assert doc["meta"]["semconv"] == otel.SEMCONV_GENERATION
    assert not any("does not say" in n for n in doc["meta"]["notes"])


def test_an_unmapped_operation_keeps_its_own_name_and_is_reported():
    """VALID_KINDS is closed, so an unknown operation has to be filed
    somewhere — but filing it silently would be recording a claim the producer
    never made."""
    doc = otel.ingest(
        _body([_span("x", {"gen_ai.operation.name": S("rerank_documents")})])
    )
    step = doc["steps"][0]
    assert step["kind"] == "tool_call"
    assert step["meta"]["otel_operation"] == "rerank_documents"
    assert any("rerank_documents" in n for n in doc["meta"]["notes"])


def test_a_zero_length_span_reads_back_as_unknown_not_zero():
    """OTLP has no way to say "the end time is unknown", so a zero-length span
    is how it is forced to express it. Reading that back as a measured 0 would
    put "this took no time" into the timeline."""
    doc = otel.ingest(_body([_span("x", {"gen_ai.operation.name": S("chat")})]))
    assert doc["steps"][0]["duration_ms"] is None


def test_a_zero_length_span_that_says_it_was_measured_is_believed():
    """The one case where 0 is a measurement rather than an absence: our own
    exporter stamps `modelmri.duration.recorded`."""
    doc = otel.ingest(
        _body(
            [
                _span(
                    "x",
                    {
                        "gen_ai.operation.name": S("chat"),
                        "modelmri.duration.recorded": {"boolValue": True},
                    },
                )
            ]
        )
    )
    assert doc["steps"][0]["duration_ms"] == 0


def test_a_missing_token_count_stays_none():
    doc = otel.ingest(_body([_span("x", {"gen_ai.operation.name": S("chat")})]))
    assert doc["steps"][0]["tokens_in"] is None
    assert doc["steps"][0]["tokens_out"] is None


# ------------------------------------------------------------- the shape


def test_the_parent_relationship_survives():
    doc = otel.ingest(
        _body(
            [
                _span(
                    "root", {"gen_ai.operation.name": S("invoke_agent")}, span_id="p1"
                ),
                _span(
                    "child",
                    {"gen_ai.operation.name": S("chat")},
                    span_id="c1",
                    parent="p1",
                    start=1_000_000_000_100_000_000,
                ),
            ]
        )
    )
    by_span = {s["meta"]["otel_span_id"]: s for s in doc["steps"]}
    # Namespaced by the OTLP trace id, because `step.id` is a global primary
    # key and a span id is only unique within its trace.
    assert by_span["c1"]["parent_id"] == by_span["p1"]["id"]
    assert by_span["p1"]["parent_id"] is None
    assert by_span["p1"]["kind"] == "subagent"


def test_started_ms_is_an_offset_from_the_earliest_span():
    doc = otel.ingest(
        _body(
            [
                _span("a", {}, span_id="a", start=1_000_000_000_000_000_000),
                _span("b", {}, span_id="b", start=1_000_000_002_000_000_000),
            ]
        )
    )
    offsets = sorted(s["started_ms"] for s in doc["steps"])
    assert offsets == [0, 2000]


def test_the_service_name_becomes_the_trace_name():
    assert otel.ingest(_body([_span("x", {})], service="billing"))["name"] == "billing"


def test_an_error_status_survives():
    span = _span("x", {"gen_ai.operation.name": S("chat")})
    span["status"] = {"code": otel.STATUS_ERROR}
    assert otel.ingest(_body([span]))["steps"][0]["error"] is True


@pytest.mark.parametrize(
    "payload,match",
    [
        ({}, "no `resourceSpans`"),
        ({"resourceSpans": []}, "no `resourceSpans`"),
        ({"resourceSpans": [{"scopeSpans": []}]}, "carries no spans"),
        ("not a dict", "JSON object"),
    ],
)
def test_a_body_that_is_not_otlp_is_refused(payload, match):
    with pytest.raises(BadRequest, match=match):
        otel.ingest(payload)


# -------------------------------------------------------------- the route


def _client():
    return TestClient(create_app())


def test_the_route_imports_a_trace(tmp_path, monkeypatch):
    # MODELMRI_HOME, not MODELMRI_TRACE_DIR: `trace_db_path()` resolves
    # through `data_dir()`, and TRACE_DIR only moves the recorder's
    # undelivered folder. The first version set the wrong one and these
    # tests wrote into the real trace database on this machine.
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    body = _body(
        [
            _span(
                "chat",
                {
                    "gen_ai.operation.name": S("chat"),
                    "gen_ai.input.messages": S("hi"),
                },
                end=1_000_000_000_500_000_000,
            )
        ]
    )
    r = _client().post("/api/otel/v1/traces", json=body)
    assert r.status_code == 200, r.text
    payload = r.json()
    # An OTLP client expects partialSuccess from a healthy collector.
    assert payload["partialSuccess"] == {}
    assert payload["spans"] == 1
    assert payload["semconv"] == "unstated"
    assert payload["id"]


def test_a_protobuf_body_is_refused_by_name():
    """OTLP's common wire format is protobuf. Reading it would cost a
    generated stub set or the OpenTelemetry SDK, so it is refused with the
    limit named rather than mis-parsed into a trace that looks real."""
    r = _client().post(
        "/api/otel/v1/traces",
        content=b"\x0a\x00",
        headers={"Content-Type": "application/x-protobuf"},
    )
    assert r.status_code == 415
    assert "http/json" in r.json()["error"]


def test_a_non_json_body_is_a_422_not_a_500():
    r = _client().post(
        "/api/otel/v1/traces",
        content=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422


def test_a_body_with_no_spans_is_a_422():
    r = _client().post("/api/otel/v1/traces", json={"resourceSpans": []})
    assert r.status_code == 422


def test_two_traces_sharing_a_span_id_can_both_be_imported(tmp_path, monkeypatch):
    """`step.id` is the PRIMARY KEY of the whole table and an OTLP span id is
    only unique WITHIN its trace. Using it raw meant the second import of any
    body reusing a span id was a UNIQUE constraint failure and a 500 — which
    is what happened the first time this ran against a second body.
    """
    # MODELMRI_HOME, not MODELMRI_TRACE_DIR: `trace_db_path()` resolves
    # through `data_dir()`, and TRACE_DIR only moves the recorder's
    # undelivered folder. The first version set the wrong one and these
    # tests wrote into the real trace database on this machine.
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    from modelmri import paths
    from modelmri.traces import TraceStore

    store = TraceStore(paths.trace_db_path())

    def body_for(trace_hex):
        span = _span("chat", {"gen_ai.operation.name": S("chat")}, span_id="aa")
        span["traceId"] = trace_hex
        return _body([span])

    first = store.import_trace(otel.ingest(body_for("1" * 32)))
    second = store.import_trace(otel.ingest(body_for("2" * 32)))
    assert first != second, "two different OTLP traces must not merge"

    # And re-importing the SAME export replaces rather than colliding: the
    # trace id is derived from the OTLP trace id, so a collector retry is the
    # same trace arriving twice, not two traces.
    again = store.import_trace(otel.ingest(body_for("2" * 32)))
    assert again == second

    # And the raw span id survives, so nothing is lost by namespacing it.
    doc = store.get_trace(second)
    assert doc["steps"][0]["meta"]["otel_span_id"] == "aa"


def test_reimporting_the_same_export_produces_the_same_step_ids():
    """Deterministic, so a re-export is recognisable as the same run rather
    than silently duplicated under new ids."""
    span = _span("chat", {"gen_ai.operation.name": S("chat")}, span_id="aa")
    span["traceId"] = "3" * 32
    a = otel.ingest(_body([span]))["steps"][0]["id"]
    b = otel.ingest(_body([span]))["steps"][0]["id"]
    assert a == b
