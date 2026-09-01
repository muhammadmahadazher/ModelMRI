"""The kinds a recorded step may be, and everything that has to agree on them.

`VALID_KINDS` is a set in one leaf module, but it is not a fact held in one
place: the recorder writes a kind, the store refuses one it does not know, the
OTLP exporter decides a span kind from it, the OTLP reader maps a foreign
operation back onto it, the search grammar parses it, the rubric counts by it,
the wire type in `api.ts` names it, two panels colour it, and three documents
enumerate it. That is nine readers of one set, and a kind added to the set
alone would be recordable, unrenderable and undocumented at the same time.

So the tests here are deliberately written against `VALID_KINDS` itself rather
than against a list of kind names copied into this file — a copy would be the
ninth thing to drift, and it would drift silently, which is the whole failure
mode this file exists to catch.

The other half is the contract for a kind nobody knows. It is REFUSAL, on the
way in, for the whole document: `import_trace` raises `BadRequest` before the
lock and before any INSERT. That is a deliberate design and adding four kinds
must not soften it, so it is pinned here as behaviour rather than left implied
by a test in another file that happens to use a nonsense string.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from modelmri import otel, trace_query
from modelmri.errors import BadRequest
from modelmri.step_kinds import VALID_KINDS
from modelmri.traces import TraceStore

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "frontend" / "src"
RECORD_PKG = ROOT / "packages" / "modelmri-record"

# The four this workstream adds. Named here because the point of the change is
# these four specifically — every OTHER assertion in this file is derived from
# VALID_KINDS so it keeps working for the fifth.
RAG_KINDS = ("retrieval", "embedding", "rerank", "guardrail")


def _client(tmp_path):
    """The route-level app, on a real sqlite file, with no model loaded."""
    from modelmri.server import create_app

    return TestClient(create_app(trace_db=str(tmp_path / "t.sqlite")))


def _doc_of_every_kind():
    """One step per kind, in sorted order so the readback is checkable."""
    kinds = sorted(VALID_KINDS)
    return {
        "name": "rag-run",
        "started_at": "2026-08-31T00:00:00Z",
        "steps": [
            {
                "id": f"s{i}",
                "kind": kind,
                "name": kind,
                "started_ms": i * 10,
                "duration_ms": 5,
            }
            for i, kind in enumerate(kinds)
        ],
    }


# ------------------------------------------------------------------ the set


def test_the_four_rag_shaped_kinds_are_recordable():
    """Retrieval, embedding, rerank and guardrail are what a RAG pipeline is
    made of, and a step kind is the only thing that lets a metric ask "how
    long did retrieval take" without pattern-matching on somebody's step
    names."""
    assert set(RAG_KINDS) <= VALID_KINDS


# ------------------------------------------------------- the round trip


def test_a_run_of_every_kind_imports_and_reads_back(tmp_path):
    c = _client(tmp_path)
    doc = _doc_of_every_kind()
    r = c.post("/api/traces/import", json=doc)
    assert r.status_code == 200, r.text
    tid = r.json()["id"]

    listing = c.get("/api/traces").json()
    assert listing[0]["n_steps"] == len(VALID_KINDS)

    back = c.get(f"/api/traces/{tid}").json()
    assert [s["kind"] for s in back["steps"]] == sorted(VALID_KINDS)


def test_a_trace_recorded_before_these_kinds_existed_is_unaffected(tmp_path):
    """The old two-kind document is the one every existing user has on disk."""
    c = _client(tmp_path)
    old = {
        "name": "t1",
        "started_at": "2026-08-07T00:00:00Z",
        "steps": [
            {"kind": "llm_call", "name": "plan", "started_ms": 0, "duration_ms": 100},
            {
                "kind": "tool_call",
                "name": "pytest",
                "started_ms": 120,
                "duration_ms": 400,
                "error": True,
            },
        ],
    }
    tid = c.post("/api/traces/import", json=old).json()["id"]
    back = c.get(f"/api/traces/{tid}").json()
    assert [s["kind"] for s in back["steps"]] == ["llm_call", "tool_call"]
    assert back["steps"][1]["error"] is True


# ------------------------------------------------- the unknown-kind contract


def test_an_unknown_kind_is_refused_and_takes_the_whole_document_with_it(tmp_path):
    """THE CONTRACT, pinned. Not "rendered as unknown" — refused, and refused
    for the document rather than the step, because a run missing the step that
    matters is a worse answer than a run that did not import.

    THE BAD STEP IS LAST, and that is the whole assertion. Put it first and the
    "nothing was written" check below cannot fail no matter where the kind
    check lives, because no INSERT could have run before the raise either way.
    The hazard `traces.py` documents in its own words is the check drifting
    INSIDE the insert loop, and only a bad step with good steps ahead of it can
    see that happen.
    """
    c = _client(tmp_path)
    bad = _doc_of_every_kind()
    bad["steps"][-1]["kind"] = "telemetry"
    r = c.post("/api/traces/import", json=bad)
    assert r.status_code == 422
    assert c.get("/api/traces").json() == [], (
        "the refusal ran after some steps were written, so a partial run is "
        "in the store"
    )


def test_a_bundle_carries_a_kind_this_build_does_not_know(tmp_path):
    """THE ONE EXCEPTION to the refusal above, and it is deliberate.

    `/api/traces/import` refuses an unknown kind for the whole document. A
    `.mri` bundle does not: `session._trace` refuses an EMPTY kind and passes
    every other one through, because refusing to open a file somebody sent you
    is worse than saying "recorded, and this build does not know what it is".
    That asymmetry is the entire premise of the panel's colour and glyph
    fallbacks and of the hatched bar beside them — and it was stated in the
    guide with nothing holding it, so tightening `session.py` some later
    afternoon would have made three carefully-written fallbacks unreachable
    and this file would not have noticed.
    """
    from modelmri import session

    carried = session._trace(
        {
            "trace": {
                "steps": [
                    {"id": "s1", "kind": "retrieval", "name": "vector-store"},
                    {"id": "s2", "kind": "telemetry", "name": "from a newer build"},
                ]
            }
        }
    )
    assert [s["kind"] for s in carried["steps"]] == ["retrieval", "telemetry"]


def test_the_refusal_names_the_kinds_including_the_new_ones(tmp_path):
    """A refusal that does not say what IS allowed is a dead end."""
    c = _client(tmp_path)
    bad = _doc_of_every_kind()
    bad["steps"][0]["kind"] = "telemetry"
    said = c.post("/api/traces/import", json=bad).json()["error"]
    for kind in sorted(VALID_KINDS):
        assert kind in said, f"the refusal does not name {kind}"


def test_the_refusal_lists_the_kinds_in_a_stable_order(tmp_path):
    """`VALID_KINDS` is a SET, so the `sorted()` in the refusal is what stops
    the same error being a different sentence every run — which is the
    difference between an error somebody can search for and one they cannot.
    The test above passes without it; this one does not."""
    c = _client(tmp_path)
    bad = _doc_of_every_kind()
    bad["steps"][-1]["kind"] = "telemetry"
    said = c.post("/api/traces/import", json=bad).json()["error"]
    assert ", ".join(sorted(VALID_KINDS)) in said


def test_a_reimport_with_a_bad_kind_does_not_empty_the_trace_already_stored(tmp_path):
    """The shape `traces.py` records as having been MEASURED as data loss: a
    good trace stored, then the same id posted again with one bad field, and
    the stored trace left with zero steps and its search entry gone. The kind
    check is the one validator that fires on a document a recorder wrote
    rather than on a hand-written one, so it is the one most likely to meet a
    re-import."""
    c = _client(tmp_path)
    good = _doc_of_every_kind()
    good["id"] = "nightly"
    assert c.post("/api/traces/import", json=good).status_code == 200

    bad = _doc_of_every_kind()
    bad["id"] = "nightly"
    bad["steps"][-1]["kind"] = "telemetry"
    assert c.post("/api/traces/import", json=bad).status_code == 422

    back = c.get("/api/traces/nightly").json()
    assert [s["kind"] for s in back["steps"]] == sorted(VALID_KINDS), (
        "the refused re-import took the stored run's steps with it"
    )
    assert c.get("/api/traces/search", params={"q": "kind:retrieval"}).json()[
        "results"
    ], "the refused re-import took the stored run out of the search index"


def test_a_refusal_names_the_recorder_that_wrote_the_run(tmp_path):
    """The whole of the version-skew policy, and it is one sentence.

    `meta.recorder` has been stamped on every delivered document since the
    recorder shipped and read by nothing. A recorder newer than the viewer
    writes kinds the viewer has never heard of, and without this the refusal
    sends that person hunting for a typo they did not make.
    """
    c = _client(tmp_path)
    bad = _doc_of_every_kind()
    bad["meta"] = {"recorder": "modelmri-record/99.0.0"}
    bad["steps"][0]["kind"] = "telemetry"
    said = c.post("/api/traces/import", json=bad).json()["error"]
    assert "modelmri-record/99.0.0" in said
    assert "upgrade modelmri" in said


def test_a_hand_written_document_gets_no_invented_version_story(tmp_path):
    """Most documents through this route were written by hand or by another
    tool. There is no recorder to name, so nothing is said about one."""
    c = _client(tmp_path)
    bad = _doc_of_every_kind()
    bad["steps"][0]["kind"] = "telemetry"
    said = c.post("/api/traces/import", json=bad).json()["error"]
    assert "recorded by" not in said


def test_a_recorder_stamp_that_is_not_a_string_still_gets_a_422(tmp_path):
    """`meta` is a free-form object of the sender's own keys, so `recorder`
    can be anything — and it reaches `_wrote_this` from a document nobody
    validated. Without the isinstance guard `stamp.strip()` is an
    AttributeError inside a validator, which the route can only answer as a
    500: "something inside ModelMRI failed" about a document that is simply
    the wrong shape. Same class the validators around it exist to prevent."""
    c = _client(tmp_path)
    bad = _doc_of_every_kind()
    bad["meta"] = {"recorder": 123}
    bad["steps"][-1]["kind"] = "telemetry"
    r = c.post("/api/traces/import", json=bad)
    assert r.status_code == 422, r.text
    said = r.json()["error"]
    assert "invalid step kind" in said
    assert "recorded by" not in said, "a number was told as a version story"


def test_a_giant_recorder_stamp_does_not_become_the_whole_refusal(tmp_path):
    """The stamp is the one part of that sentence that came from the document.
    `meta` is stored verbatim and never length-checked, so an unclipped echo
    turns a 200-character refusal into however many megabytes the sender
    chose — returned to them, logged, and rendered in the panel."""
    c = _client(tmp_path)
    bad = _doc_of_every_kind()
    bad["meta"] = {"recorder": "modelmri-record/" + "9" * 500_000}
    bad["steps"][-1]["kind"] = "telemetry"
    said = c.post("/api/traces/import", json=bad).json()["error"]
    assert "modelmri-record/" in said, "the stamp was dropped rather than cut"
    assert len(said) < 1_000, f"the refusal is {len(said)} characters long"


def test_an_unknown_kind_is_still_refused_by_the_search_grammar():
    with pytest.raises(BadRequest, match="unknown step kind"):
        trace_query.parse("kind:retreival")


# ----------------------------------------------------------------- searching


def test_the_new_kinds_are_searchable(tmp_path):
    store = TraceStore(tmp_path / "t.sqlite")
    store.import_trace(
        {
            "id": "t1",
            "name": "nightly",
            "started_at": "2026-01-01T00:00:00Z",
            "steps": [
                {
                    "id": "a",
                    "kind": "retrieval",
                    "name": "vector-store",
                    "input": "how do I fix the flaky test",
                    "output": "3 documents",
                    "started_ms": 0,
                    "duration_ms": 40,
                },
                {
                    "id": "b",
                    "kind": "rerank",
                    "name": "cross-encoder",
                    "started_ms": 40,
                    "duration_ms": 12,
                },
            ],
        }
    )
    assert [r["step_id"] for r in store.search("kind:retrieval")["results"]] == ["a"]
    assert [r["step_id"] for r in store.search("kind:rerank")["results"]] == ["b"]


# --------------------------------------------------------------------- OTLP


def test_every_kind_has_a_deliberate_otlp_operation():
    """`OPERATION.get(kind, (kind, INTERNAL))` means a kind missing from the
    table still exports — as an INTERNAL span, which is a claim about where
    the work happened. A retrieval call goes out over a socket, so the default
    would be quietly wrong for it."""
    assert set(otel.OPERATION) == VALID_KINDS


def test_an_embedding_step_closes_the_otlp_round_trip():
    """`embeddings` was already the operation name this file recognised on the
    way IN, and it landed as `tool_call` because there was no better kind. Now
    there is, and the two directions have to agree."""
    assert otel.OPERATION["embedding"][0] == "embeddings"
    assert otel.OPERATION_TO_KIND["embeddings"] == "embedding"


@pytest.mark.parametrize("kind", RAG_KINDS)
def test_our_own_operation_names_come_back_as_the_kind_they_left_as(kind):
    operation = otel.OPERATION[kind][0]
    assert otel.OPERATION_TO_KIND[operation] == kind


def _otlp_span(kind: str) -> dict:
    """One step of `kind` through the real exporter, as the span it becomes."""
    body = otel.to_otlp(
        {
            "id": "t1",
            "name": "rag-run",
            "started_at": "2026-08-31T10:00:00+00:00",
            "steps": [
                {
                    "id": "s1",
                    "kind": kind,
                    "name": kind,
                    "started_ms": 0,
                    "duration_ms": 40,
                    "seq": 0,
                }
            ],
        }
    )
    return body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]


@pytest.mark.parametrize("kind", ["retrieval", "embedding", "rerank"])
def test_a_retrieval_shaped_span_exports_as_a_client_span(kind):
    """The span kind is the ENTIRE reason these three rows exist. `OPERATION`
    is read through a `.get` whose default is `SPAN_KIND_INTERNAL`, and
    INTERNAL is a claim: it says the work happened in this process. A vector
    store query and a hosted reranker are both a call out over a socket, and a
    waterfall that draws them as in-process time is wrong about where the
    latency lives. The table asserts the operation name in three places and
    the span kind in none, so it is the half that could be quietly reverted."""
    span = _otlp_span(kind)
    attrs = {a["key"]: a["value"] for a in span["attributes"]}
    assert attrs["gen_ai.operation.name"]["stringValue"] == otel.OPERATION[kind][0]
    assert span["kind"] == otel.SPAN_KIND_CLIENT


def test_a_guardrail_span_exports_as_internal_work():
    """The one of the four that genuinely is in-process: a policy check runs
    here. Pinned beside the three above so "they are all CLIENT" cannot become
    the rule by accident."""
    assert _otlp_span("guardrail")["kind"] == otel.SPAN_KIND_INTERNAL


def test_an_embeddings_span_ingests_as_an_embedding_step():
    """`embeddings` is a settled `gen_ai.operation.name`, so a foreign
    producer emitting it was already saying exactly what the span was — and
    this file used to file it as `tool_call`, throwing that away at the door.
    Through the real `ingest`, not through the table: the table is the thing
    that might be right while the reader is not."""
    doc = otel.ingest(
        {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {
                                "key": "service.name",
                                "value": {"stringValue": "rag-agent"},
                            }
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "x"},
                            "spans": [
                                {
                                    "traceId": "0" * 32,
                                    "spanId": "aa",
                                    "name": "embeddings text-embedding-3-small",
                                    "kind": 3,
                                    "startTimeUnixNano": "1000000000000000000",
                                    "endTimeUnixNano": "1000000000180000000",
                                    "attributes": [
                                        {
                                            "key": "gen_ai.operation.name",
                                            "value": {"stringValue": "embeddings"},
                                        },
                                        {
                                            "key": "gen_ai.usage.input_tokens",
                                            "value": {"intValue": "14"},
                                        },
                                    ],
                                    "status": {"code": 0},
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    )
    step = doc["steps"][0]
    assert step["kind"] == "embedding"
    assert step["tokens_in"] == 14


# ---------------------------------------------------------------- the recorder


def test_the_recorder_knows_exactly_the_kinds_the_store_accepts():
    """The recorder cannot import this set — it is stdlib-only by contract and
    a test spawns a fresh interpreter to prove it — so the list is a second
    literal. A second literal gets a test, and the test compares the values
    rather than the source text, because the drift that matters is a kind
    present on one side and absent on the other."""
    import modelmri_record

    assert modelmri_record.KINDS == VALID_KINDS


def test_a_run_of_every_kind_records_offline_and_then_imports(tmp_path, monkeypatch):
    """The whole path, with no viewer running for the first half: the recorder
    writes the parked JSON, and the store takes that exact document.

    EVERY kind, from `VALID_KINDS`, not the four this workstream added — this
    file's own rule, and it applies here more than anywhere: this is the only
    test that carries a kind through the recorder rather than through a
    hand-written dict, so a list copied in here is the one that would leave
    the eleventh kind unexercised on the one path a user actually walks.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODELMRI_TRACE_DIR", str(tmp_path / "parked"))
    from modelmri.record import step, trace

    kinds = sorted(VALID_KINDS)
    with trace("rag-run", endpoint="http://127.0.0.1:1/nope"):
        for kind in kinds:
            step(kind, name=kind, duration_ms=4)

    parked = list((tmp_path / "parked").glob("*.json"))
    assert len(parked) == 1
    doc = json.loads(parked[0].read_text())
    assert [s["kind"] for s in doc["steps"]] == kinds

    c = _client(tmp_path)
    assert c.post("/api/traces/import", json=doc).status_code == 200


def test_the_recorder_stamp_survives_the_import(tmp_path, monkeypatch):
    """`meta.recorder` is the only thing that says which recorder wrote a run,
    and it is the only signal a skew message could ever be built on. It is
    stored verbatim, so it has to still be there on the way out."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODELMRI_TRACE_DIR", str(tmp_path / "parked"))
    import modelmri_record

    from modelmri.record import step, trace

    with trace("stamped", endpoint="http://127.0.0.1:1/nope"):
        step("retrieval", name="vector-store", duration_ms=3)

    doc = json.loads(next((tmp_path / "parked").glob("*.json")).read_text())
    stamp = f"modelmri-record/{modelmri_record.__version__}"
    assert doc["meta"]["recorder"] == stamp

    c = _client(tmp_path)
    tid = c.post("/api/traces/import", json=doc).json()["id"]
    assert c.get(f"/api/traces/{tid}").json()["meta"]["recorder"] == stamp


# ------------------------------------------------------------ the shipped demo


def test_the_shipped_demo_records_a_trace_the_store_accepts(tmp_path, monkeypatch):
    """`examples/record_demo.py` is what the guide tells a new reader to run,
    and it had no test of any kind — so an invalid kind, a broken timing or a
    field the store refuses would have shipped as the first thing anybody saw.

    Run for real, with delivery failing the way it does on a machine with no
    viewer up, and the parked document posted to a real store.
    """
    import runpy
    import urllib.request

    from modelmri.ledger import FIELDS, TOKEN_KINDS

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODELMRI_TRACE_DIR", str(tmp_path / "parked"))

    def nobody_listening(*args, **kwargs):
        raise OSError("no viewer running")

    # Patched rather than pointed at a dead port: the demo uses the default
    # endpoint on purpose, and a developer running `modelmri serve` on this
    # machine would otherwise have the trace delivered and nothing parked.
    monkeypatch.setattr(urllib.request, "urlopen", nobody_listening)
    runpy.run_path(str(ROOT / "examples" / "record_demo.py"), run_name="__main__")

    parked = list((tmp_path / "parked").glob("*.json"))
    assert len(parked) == 1, "the demo did not park a trace"
    doc = json.loads(parked[0].read_text(encoding="utf-8"))

    c = _client(tmp_path)
    r = c.post("/api/traces/import", json=doc)
    assert r.status_code == 200, r.text

    # AND IT MUST NOT DEMONSTRATE A CONTRADICTION. `ledger.TOKEN_KINDS` is
    # `("llm_call",)` deliberately — the rollup is named `n_llm_steps`, so
    # folding embeddings in would restate every stored run — and a step
    # outside that tuple has its counts neither summed nor counted among the
    # ones that reported nothing. The demo shipped an `embedding` step with
    # `tokens_in=14`, so selecting it printed "14 tok in" in the header and,
    # one line below, "no LLM calls here, so there are no tokens to count".
    # Both about the same step. Until the roll-up can say "carries tokens
    # that are not counted here", the shipped sample says nothing at all.
    contradicting = [
        s["name"]
        for s in doc["steps"]
        if s["kind"] not in TOKEN_KINDS and any(s.get(f) is not None for f in FIELDS)
    ]
    assert contradicting == [], (
        f"{contradicting} carry token counts on a kind the roll-up drops in "
        f"silence, so the panel prints the count and 'no tokens to count' "
        f"about the same step"
    )


# ----------------------------------------------------------------- the docs


@pytest.mark.parametrize(
    "doc",
    [
        "docs/guides/agents.md",
        "docs/reference/record.md",
        "packages/modelmri-record/README.md",
    ],
)
def test_every_document_that_enumerates_the_kinds_names_all_of_them(doc):
    """Three hand-written enumerations. A reader who cannot find `retrieval`
    in the reference concludes it is not a thing they may write."""
    text = (ROOT / doc).read_text(encoding="utf-8")
    missing = sorted(k for k in VALID_KINDS if f"`{k}`" not in text)
    assert missing == [], f"{doc} does not name {missing}"


# -------------------------------------------------------------- the panels


def _src(name: str) -> str:
    return (SRC / name).read_text("utf-8", errors="replace")


def _map_body(src: str, name: str) -> str:
    """The body of one `const <name>: Record<StepKind, …> = { … };`.

    With `//` lines removed, which matters: a comment sits above most entries
    in the glyph map, and a naive slice would attach it to the entry before it
    — so two byte-identical drawings with different comments above them would
    read as different bodies, which is the one thing the caller is checking.
    """
    start = src.index(f"const {name}: Record<StepKind,")
    end = src.index("\n};", start)
    lines = src[start:end].splitlines()
    return "\n".join(ln for ln in lines if not ln.strip().startswith("//"))


def _entries(body: str) -> dict[str, str]:
    """`kind -> everything it maps to`, for a map whose entries are one
    two-space-indented `key:` each and whose values may run over many lines."""
    found = re.findall(r"^  (\w+):\s*(.*?)(?=^  \w+:|\Z)", body, re.S | re.M)
    return {k: " ".join(v.split()).rstrip(",") for k, v in found}


def test_the_panel_gives_every_kind_a_colour_and_a_glyph():
    """Colour alone ran out. Eight hues exist in this palette and six were
    already spoken for, so two of the four new kinds share one — which is only
    honest if something else carries the difference, and that something is the
    glyph.

    Both maps are `Record<StepKind, …>`, so TypeScript catches an omission on
    its own. This catches the other direction — a kind added on the Python
    side that never reached the panel at all — and it reads the two maps
    SEPARATELY, because counting entries across both cannot tell "a colour and
    a glyph" from "two colours and no glyph".
    """
    src = _src("stepKinds.tsx")
    colours = _entries(_map_body(src, "KIND_COLOR"))
    glyphs = _entries(_map_body(src, "KIND_GLYPH"))
    assert sorted(colours) == sorted(VALID_KINDS), "the colour map is not the kind list"
    assert sorted(glyphs) == sorted(VALID_KINDS), "the glyph map is not the kind list"


def test_no_two_kinds_are_told_apart_by_colour_alone():
    """The justification for the whole glyph system, asserted.

    `retrieval` and `rerank` share moss deliberately — there is no ninth hue
    to spend and they are the same family — so on the timeline the SHAPE is
    the only thing between them. Nothing held that: making one glyph a copy of
    the other left two kinds genuinely indistinguishable and every test green.
    Stated as the general rule rather than about that pair, so it keeps
    holding when the next pair has to share.
    """
    src = _src("stepKinds.tsx")
    colours = _entries(_map_body(src, "KIND_COLOR"))
    glyphs = _entries(_map_body(src, "KIND_GLYPH"))

    shared: dict[str, list[str]] = {}
    for kind, token in colours.items():
        shared.setdefault(token, []).append(kind)

    by_glyph: dict[str, list[str]] = {}
    for kind, drawing in glyphs.items():
        by_glyph.setdefault(drawing, []).append(kind)
    twins = [ks for ks in by_glyph.values() if len(ks) > 1]
    assert twins == [], f"{twins} are drawn as the same shape"

    # And the reason that matters, said out loud: at least one hue IS shared,
    # so a reader who cannot see the shape cannot see the difference.
    assert any(len(ks) > 1 for ks in shared.values()), (
        "no colour is shared any more — if a hue was freed, say so here rather "
        "than leaving this test claiming a constraint that no longer exists"
    )


def test_every_kind_colour_token_is_defined_in_the_stylesheet():
    """`var(--color-grond)` renders as no background at all, and this project
    has shipped that bug three times — which is what `test_css_vars.py` exists
    for. That file scans `styles.css` only, so these ten references, which
    live in a `.tsx`, are outside everything it can see."""
    css = _src("styles.css")
    colours = _entries(_map_body(_src("stepKinds.tsx"), "KIND_COLOR"))
    tokens = {re.search(r"--[\w-]+", v).group(0) for v in colours.values()}
    tokens.add("--color-mute")  # the fallback in `kindColor`
    missing = sorted(t for t in tokens if f"{t}:" not in css)
    assert missing == [], f"the panel paints with tokens nothing defines: {missing}"


def test_the_rubric_picker_cannot_drift_from_the_kind_list():
    """It was a hand-written array of the six kinds — a plain literal, so
    TypeScript could not see it go stale, and it is exactly the copy that goes
    stale."""
    panel = _src("RubricPanel.tsx")
    assert "STEP_KINDS.map" in panel, "the step-kind picker is not derived any more"
    assert '"mcp_call"' not in panel, "a second hardcoded kind list is back"


def test_a_kind_the_viewer_does_not_know_still_draws():
    """A `.mri` bundle does NOT validate kinds — `session.py` only refuses an
    empty one — so an old viewer opening a new bundle meets a kind it has no
    colour for. `.tl-block` is `all: unset`, so an undefined background is an
    invisible bar: the step is on the timeline and cannot be seen or clicked."""
    kinds = _src("stepKinds.tsx")
    assert '?? "var(--color-mute)"' in kinds, (
        "no colour fallback for a kind this build does not know"
    )
    assert "?? UNKNOWN_GLYPH" in kinds, "no glyph fallback for the same"


def test_an_unknown_kind_is_not_drawn_as_a_user_turn():
    """The fallback above fixed an invisible bar and stopped one step short.

    `--color-mute` is not a neutral: it is the token `user_turn` owns, with
    its own row in the legend. The glyph that would carry the difference is
    hidden on any bar under 22px, so a narrow bar of an unrecognised kind was
    pixel-identical to a narrow user turn — and the legend beside it confirmed
    the misreading. Only the tooltip disagreed. So the bar is marked, and
    hatched by the same idiom `.no-dur` uses for "this is not what it looks
    like", which survives at any width and costs no hue.
    """
    assert "export function isKnownKind" in _src("stepKinds.tsx")
    panel = _src("AgentsPanel.tsx")
    assert "isKnownKind(step.kind)" in panel, "the timeline does not ask"
    assert '"unknown-kind"' in panel, "the bar is not marked"
    css = _src("styles.css")
    assert ".tl-block.unknown-kind" in css, "the mark paints nothing"


def test_the_timeline_bar_asks_its_own_width():
    """The glyph on a bar is gated on the BAR's width, not the panel's, and
    that only works because `.tl-block` is a size container. Drop the
    `container-type` and the `@container` query silently resolves against some
    ancestor or nothing at all — every glyph shown or every glyph hidden, with
    no error either way."""
    css = _src("styles.css")
    assert re.search(r"\.tl-block\s*\{[^}]*container-type:\s*inline-size", css, re.S), (
        "the timeline bar is not a size container"
    )
    assert "@container (min-width: 22px)" in css


def test_the_panel_reads_every_kind_through_the_fallback():
    """The fallbacks live in `kindColor`/`KindGlyph`, so they only protect the
    call sites that go through them. `AgentsPanel` used to index its own local
    map directly, which is exactly the shape that produced the invisible bar,
    and nothing stopped that from coming back."""
    panel = _src("AgentsPanel.tsx")
    assert "KIND_COLOR[" not in panel, "a kind is indexed without the fallback"
    assert panel.count("kindColor(") >= 3, (
        "the bar, the legend and the inspector chip do not all go through it"
    )


def test_the_wire_type_admits_every_kind_the_store_accepts():
    """`TraceStep["kind"]` is the type every trace payload in the frontend is
    read through, and NOTHING fails to compile while it is short — no
    exhaustive switch reads it, every consumer renders the value as a string
    or does one `=== "llm_call"`. So a stale union keeps working and quietly
    makes four real kinds unrepresentable, and there is no other pressure on
    it than this."""
    src = _src("api.ts")
    start = src.index("export interface TraceStep {")
    body = src[start : src.index("\n}", start)]
    union = re.search(r"\n  kind:(.*?);", body, re.S)
    assert union, "TraceStep has no `kind` field any more"
    assert set(re.findall(r'"(\w+)"', union.group(1))) == VALID_KINDS


# ------------------------------------------------------- the stdlib-only rule


def test_the_recorder_still_does_not_import_the_viewer_for_its_kinds():
    """The obvious way to keep the two lists in agreement is an import, and it
    is the one thing this package may not do.

    Read as a SYNTAX TREE rather than as two substrings. `from modelmri.step_kinds`
    and `import modelmri.step_kinds` are two of the spellings; `from modelmri
    import step_kinds` is a third and was matched by neither, and a
    function-level import — the form somebody reaches for precisely because it
    looks cheap — is invisible to a text search that anchors on the margin.
    The tree sees every one of them, and it does not see the `from
    modelmri.record import trace, step` in this module's own docstring, which
    a text search would have to be careful about.

    Two imports of the viewer are legitimate and both are named here rather
    than allowed by shape: `modelmri.otel` in `_deliver_otlp`, reached only
    when the caller asked for an OTLP export, and `modelmri.paths` in
    `_trace_dir`, so both halves agree on where an undelivered trace lands.
    Each is inside a function and inside a `try`, so neither runs at import
    and neither is load-bearing — which is what `test_import_costs_nothing_heavy`
    in the package's own suite measures, in a fresh interpreter.
    """
    import ast

    src = (RECORD_PKG / "modelmri_record" / "__init__.py").read_text("utf-8")
    reached = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            reached.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            # `from modelmri import otel` and `from modelmri.step_kinds import
            # VALID_KINDS` name different modules and have to be told apart,
            # so the submodule is folded back into the name.
            reached.update(f"{node.module}.{a.name}" for a in node.names)
    viewer = sorted(m for m in reached if m == "modelmri" or m.startswith("modelmri."))
    assert viewer == ["modelmri.otel", "modelmri.paths"], (
        f"this package reaches into the viewer as {viewer}. The kind list is a "
        f"second literal on purpose; importing `modelmri.step_kinds` for it "
        f"would make `pip install modelmri-record` cost a 2.5 GB install."
    )
