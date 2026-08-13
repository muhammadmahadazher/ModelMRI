"""Handing a recorded run to somebody else's collector.

The load-bearing test here is the round trip. `to_otlp` and `from_otlp` read
one table, `otel.FIELDS`, and the point of that table is that two hand-written
mappings drift: one gains a key, the other does not, and a column disappears
on the way through while both functions still look correct in isolation. So
the round trip is asserted over `FIELDS` itself rather than over a list of
field names copied into this file, which would be a third thing to drift.

The other half is the refusals and the absences. A missing token count must
not arrive as 0 — that puts "this call used no input tokens" onto somebody's
dashboard — and a step with no recorded duration must not arrive as a plain
zero-length span, because on a waterfall that reads as an instantaneous
operation. OTLP has no way to say "unknown", so the flag is the most the wire
format allows and the tests pin that it is set.

Nothing here opens a socket except the two that deliberately do, against a
`http.server` started in-process.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from modelmri import otel
from modelmri.errors import BadRequest, Refusal


def _doc(**over):
    doc = {
        "id": "trace-1",
        "name": "deploy",
        "started_at": "2026-08-13T10:00:00+00:00",
        "steps": [
            {
                "id": "s1",
                "parent_id": None,
                "kind": "llm_call",
                "name": "chat",
                "started_ms": 0,
                "duration_ms": 1200,
                "input": "hi",
                "output": "hello",
                "tokens_in": 5,
                "tokens_out": 7,
                "error": False,
                "seq": 0,
                "meta": {"model": "qwen3"},
                "truncated_in": False,
                "truncated_out": False,
            }
        ],
    }
    doc.update(over)
    return doc


def _spans(body):
    return body["resourceSpans"][0]["scopeSpans"][0]["spans"]


def _attrs(span):
    return {a["key"]: next(iter(a["value"].values())) for a in span["attributes"]}


# ----------------------------------------------------- the round trip


def test_every_mapped_field_survives_a_round_trip():
    """The table is what is under test, not either function. Adding a row to
    `FIELDS` extends this automatically; adding a key to only one direction
    fails here."""
    doc = _doc()
    back = otel.from_otlp(otel.to_otlp(doc))
    assert len(back) == 1
    original, got = doc["steps"][0], back[0]
    for field in otel.FIELDS:
        if field.step.startswith("_"):
            continue  # derived, not a column
        if original.get(field.step) is None:
            # Correctly OMITTED rather than zeroed — see
            # test_a_missing_token_count_is_omitted_not_zeroed. An absent
            # attribute is the intended representation of an absent value, so
            # demanding it round-trip to a key would be demanding the bug.
            assert field.key not in _attrs(_spans(otel.to_otlp(doc))[0])
            continue
        assert got[field.step] == original[field.step], field.key


def test_the_round_trip_preserves_a_recorded_duration():
    """Duration lives in the span envelope rather than an attribute, so it is
    the one field the table cannot carry and the one most likely to be lost."""
    back = otel.from_otlp(otel.to_otlp(_doc()))
    assert back[0]["duration_ms"] == 1200


def test_the_round_trip_preserves_an_unrecorded_duration_as_none():
    """None, not 0. The whole reason `duration_ms` was made nullable."""
    doc = _doc()
    doc["steps"][0]["duration_ms"] = None
    back = otel.from_otlp(otel.to_otlp(doc))
    assert back[0]["duration_ms"] is None


def test_the_round_trip_preserves_the_error_flag():
    doc = _doc()
    doc["steps"][0]["error"] = True
    assert otel.from_otlp(otel.to_otlp(doc))[0]["error"] is True


# --------------------------------------------------------- absences


def test_a_missing_token_count_is_omitted_not_zeroed():
    """`gen_ai.usage.input_tokens: 0` is a claim that the call used none. The
    same shape as the `.get(name, 0.0)` that made 206 episodes show one video."""
    doc = _doc()
    doc["steps"][0]["tokens_in"] = None
    attrs = _attrs(_spans(otel.to_otlp(doc))[0])
    assert "gen_ai.usage.input_tokens" not in attrs
    assert "gen_ai.usage.output_tokens" in attrs  # the one that IS recorded


def test_a_step_without_a_duration_is_flagged_on_the_span():
    """OTLP requires an end time and has no way to say it is unknown, so the
    span goes as zero-length — and a zero-length span reads as instantaneous.
    The flag is the most the wire format allows."""
    doc = _doc()
    doc["steps"][0]["duration_ms"] = None
    span = _spans(otel.to_otlp(doc))[0]
    assert span["startTimeUnixNano"] == span["endTimeUnixNano"]
    assert _attrs(span)["modelmri.duration.recorded"] is False


def test_a_step_with_a_duration_is_flagged_too():
    """Both states set the attribute. Present-or-absent would make "no
    duration" indistinguishable from "written by an older version"."""
    assert _attrs(_spans(otel.to_otlp(_doc()))[0])["modelmri.duration.recorded"] is True


# ------------------------------------------------------------- shape


def test_ids_are_the_widths_otlp_requires():
    span = _spans(otel.to_otlp(_doc()))[0]
    assert len(span["traceId"]) == 32
    assert len(span["spanId"]) == 16
    int(span["traceId"], 16)  # hex, or this raises
    int(span["spanId"], 16)


def test_ids_are_derived_deterministically():
    """Re-exporting the same run must produce the same ids, or a collector
    files the second send as a second trace."""
    assert otel.to_otlp(_doc()) == otel.to_otlp(_doc())


def test_the_original_step_id_survives_the_derivation():
    """The span id is a hash, so without this the identity a `.mri` uses would
    not survive the export."""
    assert _attrs(_spans(otel.to_otlp(_doc()))[0])["modelmri.step.id"] == "s1"


def test_times_are_strings_not_json_numbers():
    """proto3's JSON mapping encodes uint64 as a string. A collector handed a
    JSON number for a nanosecond timestamp loses precision or rejects it."""
    span = _spans(otel.to_otlp(_doc()))[0]
    assert isinstance(span["startTimeUnixNano"], str)
    assert isinstance(span["endTimeUnixNano"], str)


def test_the_parent_link_is_emitted_only_when_there_is_one():
    doc = _doc()
    doc["steps"].append({**doc["steps"][0], "id": "s2", "parent_id": "s1", "seq": 1})
    a, b = _spans(otel.to_otlp(doc))
    assert "parentSpanId" not in a
    assert b["parentSpanId"] == otel.span_id("s1")


def test_every_span_carries_the_semconv_generation():
    """`gen_ai.*` left the main semconv repo on 2026-06-12 for one with no
    releases and no tags. Which vocabulary a span speaks is a real question,
    and the answer has to travel with the span."""
    for span in _spans(otel.to_otlp(_doc())):
        assert _attrs(span)["modelmri.semconv.generation"] == otel.SEMCONV_GENERATION


def test_an_unknown_step_kind_keeps_its_own_name():
    """A `subagent` filed under `chat` is a wrong answer that looks like a
    right one, so an unmapped kind is passed through rather than guessed."""
    doc = _doc()
    doc["steps"][0]["kind"] = "something_new"
    assert _attrs(_spans(otel.to_otlp(doc))[0])["gen_ai.operation.name"] == (
        "something_new"
    )


@pytest.mark.parametrize(
    "kind,operation",
    [("llm_call", "chat"), ("tool_call", "execute_tool"), ("subagent", "invoke_agent")],
)
def test_known_kinds_map_to_the_genai_operation(kind, operation):
    doc = _doc()
    doc["steps"][0]["kind"] = kind
    assert _attrs(_spans(otel.to_otlp(doc))[0])["gen_ai.operation.name"] == operation


def test_the_error_flag_sets_a_status_without_inventing_a_message():
    """The recorder stores a flag, not text. A sentence here would be one
    nobody wrote appearing in a dashboard."""
    doc = _doc()
    doc["steps"][0]["error"] = True
    span = _spans(otel.to_otlp(doc))[0]
    assert span["status"]["code"] == otel.STATUS_ERROR
    assert "message" not in span["status"]


def test_an_unparseable_start_puts_spans_at_the_epoch_visibly():
    """0 rather than "now": every span in 1970 is obviously wrong on a
    timeline, where a silently substituted current time is subtly wrong."""
    doc = _doc(started_at="not a date")
    assert _spans(otel.to_otlp(doc))[0]["startTimeUnixNano"] == "0"


def test_a_trace_with_no_id_is_refused():
    with pytest.raises(BadRequest, match="no id"):
        otel.to_otlp(_doc(id=""))


def test_the_body_is_json_serialisable():
    json.dumps(otel.to_otlp(_doc()))


# ------------------------------------------------------- the endpoint


@pytest.mark.parametrize(
    "given,expect",
    [
        ("http://localhost:4318", "http://localhost:4318/v1/traces"),
        ("http://localhost:4318/", "http://localhost:4318/v1/traces"),
        ("http://localhost:4318/v1/traces", "http://localhost:4318/v1/traces"),
    ],
)
def test_a_base_url_and_a_full_path_both_work(given, expect):
    """Collectors are configured both ways in the wild, and the difference is
    a 404 twenty seconds later."""
    assert otel.normalise_endpoint(given) == expect


def test_a_bare_host_port_is_refused_naming_the_grpc_trap():
    """4317 is gRPC. Someone reaching for it should be told, not timed out."""
    with pytest.raises(BadRequest) as err:
        otel.normalise_endpoint("localhost:4317")
    assert "http" in str(err.value)


def test_an_empty_endpoint_is_refused():
    with pytest.raises(BadRequest, match="no endpoint"):
        otel.normalise_endpoint("   ")


# ---------------------------------------------------------- delivery


class _Handler(BaseHTTPRequestHandler):
    status = 200
    received: dict = {}

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).received = {
            "path": self.path,
            "content_type": self.headers.get("Content-Type"),
            "auth": self.headers.get("Authorization"),
            "body": json.loads(raw) if raw else None,
        }
        self.send_response(type(self).status)
        self.end_headers()

    def log_message(self, *a):
        pass


@pytest.fixture
def collector():
    """A real HTTP server. The wire format is the feature, so it is tested
    over a socket rather than by asserting on a dict."""

    class H(_Handler):
        status = 200
        received: dict = {}

    server = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", H
    server.shutdown()


def test_a_real_collector_receives_valid_otlp(collector):
    url, handler = collector
    result = otel.send(_doc(), url)
    assert result.status == 200
    assert result.spans == 1
    assert result.semconv == otel.SEMCONV_GENERATION
    assert handler.received["path"] == "/v1/traces"
    assert handler.received["content_type"] == "application/json"
    assert _spans(handler.received["body"])[0]["name"] == "chat"


def test_extra_headers_reach_the_collector(collector):
    """A hosted collector needs an auth token, and a silently dropped one is a
    401 several minutes later."""
    url, handler = collector
    otel.send(_doc(), url, headers={"Authorization": "Bearer t"})
    assert handler.received["auth"] == "Bearer t"


def test_the_delivery_counts_undated_spans(collector):
    """So the CLI can say it out loud rather than leaving the flag in an
    attribute nobody reads."""
    url, _ = collector
    doc = _doc()
    doc["steps"][0]["duration_ms"] = None
    assert otel.send(doc, url).undated_spans == 1


def test_a_protobuf_only_collector_is_refused_with_the_reason(collector):
    """415 means the collector wants protobuf. That would cost either a
    generated stub set or the OpenTelemetry SDK, and `modelmri-record` is
    stdlib-only because it is imported into other people's agents."""
    url, handler = collector
    handler.status = 415
    with pytest.raises(Refusal) as err:
        otel.send(_doc(), url)
    assert "protobuf" in str(err.value)
    assert "stdlib-only" in str(err.value)


def test_another_http_error_says_nothing_was_delivered(collector):
    url, handler = collector
    handler.status = 500
    with pytest.raises(Refusal, match="Nothing was recorded as delivered"):
        otel.send(_doc(), url)


def test_an_unreachable_collector_names_the_port_confusion():
    with pytest.raises(Refusal) as err:
        otel.send(_doc(), "http://127.0.0.1:1", timeout=2)
    assert "4318" in str(err.value) and "4317" in str(err.value)


# ---------------------------------------------------------------------------
# What the first version got wrong
# ---------------------------------------------------------------------------


def test_parent_and_start_survive_the_round_trip():
    """Both were emitted and never read back, so a round trip destroyed the
    call tree and stacked every step at t=0. They were invisible to the
    round-trip test for the same reason they were broken: the test walks
    FIELDS, and they were not in it."""
    doc = _doc()
    doc["steps"].append({**doc["steps"][0], "id": "s2", "parent_id": "s1", "seq": 1})
    back = otel.from_otlp(otel.to_otlp(doc))
    assert back[1]["parent_id"] == "s1"
    assert back[1]["started_ms"] == doc["steps"][1]["started_ms"]


def test_the_model_survives_the_round_trip():
    back = otel.from_otlp(otel.to_otlp(_doc()))
    assert back[0]["meta"]["model"] == "qwen3"


def test_a_step_without_an_id_is_refused_not_given_a_shared_one():
    """`_hex_id("")` is a constant, so every id-less step got the SAME span id
    — invalid OTLP, and collectors collapse or orphan them."""
    doc = _doc()
    doc["steps"][0]["id"] = ""
    with pytest.raises(BadRequest, match="no id"):
        otel.to_otlp(doc)


def test_a_naive_timestamp_is_read_as_utc_not_local():
    """`dt.timestamp()` on a naive datetime uses LOCAL time, so every span
    shifted by the machine's UTC offset with nothing saying so."""
    aware = otel._epoch_ns("2026-08-13T10:00:00+00:00")
    naive = otel._epoch_ns("2026-08-13T10:00:00")
    assert naive == aware


def test_an_unparseable_start_is_reported_not_just_silently_zero(collector):
    """The docstring claimed it was "said rather than hidden" and nothing said
    it. 1970 in a collector is discovered days later by someone squinting."""
    url, _ = collector
    result = otel.send(_doc(started_at="not a date"), url)
    assert result.epoch_fallback is True
    assert otel.send(_doc(), url).epoch_fallback is False


def test_partial_success_is_read_from_the_body(collector):
    """OTLP returns partialSuccess WITH HTTP 200. Reporting `spans` as
    delivered without reading it claims something never measured."""
    url, handler = collector

    class Partial(handler):
        def do_POST(self):  # noqa: N802 - the stdlib's spelling
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.dumps(
                {"partialSuccess": {"rejectedSpans": 1, "errorMessage": "over quota"}}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Partial)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        result = otel.send(_doc(), f"http://127.0.0.1:{server.server_port}")
    finally:
        server.shutdown()
    assert result.status == 200
    assert result.rejected_spans == 1
    assert result.accepted == 0
    assert "over quota" in result.reject_message


def test_a_full_success_reports_nothing_rejected_rather_than_zero(collector):
    """None means the collector said nothing; 0 means it said it rejected
    none. Different answers."""
    url, _ = collector
    assert otel.send(_doc(), url).rejected_spans is None
