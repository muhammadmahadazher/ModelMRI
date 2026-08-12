"""Search, truncation markers, and a duration that can be absent.

Three defects of the same family. A filter that quietly matches nothing looks
identical to "there are no tool calls"; a clipped payload that renders as
complete looks identical to a complete one; and `duration_ms NOT NULL DEFAULT
0` made "not recorded" indistinguishable from "took no measurable time". Each
is a confident wrong reading with nothing on screen to contradict it, which is
the same shape as the `.get(name, 0.0)` that made 206 robot episodes show one
video.

The search half also has a security edge the others do not: the box accepts
field names, and field names are the one thing that cannot be a bound
parameter. So it is an allow-list rather than escaping.
"""

from __future__ import annotations

import sqlite3

import pytest

from modelmri import trace_query
from modelmri.errors import BadRequest
from modelmri.traces import TraceStore, _clip, _unclip


# ------------------------------------------------------------- query parsing


def test_free_text_survives_alongside_filters():
    q = trace_query.parse("pytest failed kind:tool_call")
    assert q.text == "pytest failed"
    assert q.kind == "tool_call"


def test_a_filter_anywhere_in_the_string_is_found():
    q = trace_query.parse("error:true timeout")
    assert q.error is True and q.text == "timeout"


def test_quoted_values_carry_spaces():
    q = trace_query.parse('name:"run tests"')
    assert q.name == "run tests"


def test_comparisons_parse_both_ways():
    q = trace_query.parse("duration>2000 duration<9000")
    assert q.duration_gt == 2000 and q.duration_lt == 9000


def test_an_unknown_field_is_treated_as_prose_not_refused():
    q = trace_query.parse("traceback:something broke")
    assert "traceback" in q.text and q.filters_used == []


def test_a_pasted_log_line_is_prose_even_when_it_names_a_real_field():
    """"error: connection refused" is the single most likely thing anybody
    pastes into a search box. With loose binding it parsed as the filter
    `error:connection` and was refused."""
    q = trace_query.parse("error: connection refused")
    assert q.error is None
    assert q.filters_used == []
    assert q.text == "error: connection refused"


def test_a_filter_still_binds_without_the_space():
    q = trace_query.parse("error:true")
    assert q.error is True and q.text == ""


def test_a_bad_kind_is_named_with_the_valid_ones():
    with pytest.raises(BadRequest, match="unknown step kind"):
        trace_query.parse("kind:tolcall")


def test_a_bad_error_value_refuses_rather_than_matching_nothing():
    """Silently matching nothing looks exactly like a trace with no failures."""
    with pytest.raises(BadRequest, match="takes true or false"):
        trace_query.parse("error:maybe")


def test_duration_with_a_colon_explains_what_to_write_instead():
    with pytest.raises(BadRequest, match="duration>2000"):
        trace_query.parse("duration:2000")


def test_duration_needs_a_number():
    with pytest.raises(BadRequest, match="whole number of milliseconds"):
        trace_query.parse("duration>soon")


def test_an_empty_query_is_empty():
    assert trace_query.parse("").is_empty
    assert not trace_query.parse("kind:error").is_empty


# ------------------------------------------------- no SQL is built from input


def test_the_where_clause_binds_every_value():
    q = trace_query.parse("kind:tool_call name:pytest error:true duration>5")
    clause, params = trace_query.where(q)
    assert clause.count("?") == len(params) == 4
    # Not one user-supplied byte in the SQL text.
    for value in ("tool_call", "pytest", "true", "5"):
        assert value not in clause


def test_only_allow_listed_columns_can_appear():
    q = trace_query.parse("kind:error")
    clause, _ = trace_query.where(q)
    for token in clause.replace("(", " ").replace(")", " ").split():
        if token.startswith("s."):
            assert token.split(".")[1] in (
                "kind", "name", "error", "duration_ms",
            )


def test_an_injection_attempt_lands_in_a_bound_parameter():
    q = trace_query.parse("name:x';DROP TABLE step;--")
    clause, params = trace_query.where(q)
    assert "DROP" not in clause
    assert any("DROP" in str(p) for p in params)


def test_a_null_duration_is_excluded_explicitly_not_by_accident():
    """In SQL a comparison against NULL is NULL, so without the guard a step
    with no recorded duration would fail BOTH duration>50 and duration<50 —
    not a filter, a disappearance."""
    clause, _ = trace_query.where(trace_query.parse("duration>5"))
    assert "IS NOT NULL" in clause


# ------------------------------------------------------- truncation markers


def test_clip_and_unclip_are_inverses_for_the_marker():
    text = "x" * 20_050
    clipped = _clip(text)
    body, missing = _unclip(clipped)
    assert missing == 50
    assert len(body) == 20_000
    assert "…" not in body


def test_untruncated_text_reports_nothing_missing():
    body, missing = _unclip("a short output")
    assert body == "a short output" and missing == 0


def test_a_payload_that_merely_mentions_the_marker_is_not_misread():
    """The marker is only the marker at the very end of the string."""
    body, missing = _unclip("the log said … [+5] and then continued")
    assert missing == 0
    assert body.endswith("continued")


def test_the_step_reports_how_much_was_not_stored(tmp_path):
    store = TraceStore(tmp_path / "t.sqlite")
    tid = store.import_trace(
        {
            "name": "run",
            "started_at": "2026-01-01T00:00:00Z",
            "steps": [
                {
                    "id": "s1",
                    "kind": "tool_call",
                    "name": "pytest",
                    "output": "y" * 38_412,
                }
            ],
        }
    )
    step = store.get_trace(tid)["steps"][0]
    assert step["truncated_out"] == 18_412
    assert not step["output"].endswith("]")


# --------------------------------------------------- duration can be absent


def test_a_step_recorded_bare_has_no_duration(tmp_path):
    store = TraceStore(tmp_path / "t.sqlite")
    tid = store.import_trace(
        {
            "name": "run",
            "started_at": "2026-01-01T00:00:00Z",
            "steps": [{"id": "s1", "kind": "tool_call"}],
        }
    )
    assert store.get_trace(tid)["steps"][0]["duration_ms"] is None


def test_an_explicit_zero_is_still_zero(tmp_path):
    """The caller said so. Only ABSENCE becomes None."""
    store = TraceStore(tmp_path / "t.sqlite")
    tid = store.import_trace(
        {
            "name": "run",
            "started_at": "2026-01-01T00:00:00Z",
            "steps": [{"id": "s1", "kind": "tool_call", "duration_ms": 0}],
        }
    )
    assert store.get_trace(tid)["steps"][0]["duration_ms"] == 0


def test_total_ms_ignores_steps_with_no_duration(tmp_path):
    store = TraceStore(tmp_path / "t.sqlite")
    store.import_trace(
        {
            "name": "run",
            "started_at": "2026-01-01T00:00:00Z",
            "steps": [
                {"id": "s1", "kind": "tool_call", "started_ms": 0, "duration_ms": 40},
                {"id": "s2", "kind": "tool_call", "started_ms": 10},
            ],
        }
    )
    assert store.list_traces()[0]["total_ms"] == 40


def test_an_old_store_with_not_null_duration_is_rebuilt(tmp_path):
    """SQLite cannot relax NOT NULL with ALTER TABLE, so this is a real table
    rebuild — and it must not lose the rows."""
    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(str(path))
    old.executescript(
        """
        CREATE TABLE trace (id TEXT PRIMARY KEY, name TEXT NOT NULL,
          started_at TEXT NOT NULL, meta TEXT NOT NULL DEFAULT '{}');
        CREATE TABLE step (id TEXT PRIMARY KEY, trace_id TEXT NOT NULL,
          parent_id TEXT, kind TEXT NOT NULL, name TEXT NOT NULL DEFAULT '',
          started_ms INTEGER NOT NULL, duration_ms INTEGER NOT NULL DEFAULT 0,
          input TEXT NOT NULL DEFAULT '', output TEXT NOT NULL DEFAULT '',
          tokens_in INTEGER, tokens_out INTEGER,
          error INTEGER NOT NULL DEFAULT 0, seq INTEGER NOT NULL);
        INSERT INTO trace VALUES('t1','old','2026-01-01T00:00:00Z','{}');
        INSERT INTO step VALUES('s1','t1',NULL,'tool_call','pytest',0,7,
          'in','out',NULL,NULL,0,0);
        """
    )
    old.commit()
    old.close()

    store = TraceStore(path)
    step = store.get_trace("t1")["steps"][0]
    assert step["name"] == "pytest" and step["duration_ms"] == 7

    notnull = {
        r[1]: r[3] for r in sqlite3.connect(str(path)).execute("PRAGMA table_info(step)")
    }
    assert notnull["duration_ms"] == 0, "duration_ms is still NOT NULL"


# ----------------------------------------------------------------- searching


@pytest.fixture
def store(tmp_path):
    s = TraceStore(tmp_path / "t.sqlite")
    s.import_trace(
        {
            "id": "t1",
            "name": "nightly",
            "started_at": "2026-01-01T00:00:00Z",
            "steps": [
                {"id": "a", "kind": "tool_call", "name": "pytest",
                 "input": "run the suite", "output": "17 passed",
                 "started_ms": 0, "duration_ms": 5000},
                {"id": "b", "kind": "llm_call", "name": "plan",
                 "input": "how do I fix the flaky test",
                 "output": "retry it", "started_ms": 10, "duration_ms": 20},
                {"id": "c", "kind": "tool_call", "name": "git",
                 "input": "git push", "output": "permission denied",
                 "started_ms": 20, "duration_ms": 30, "error": True},
            ],
        }
    )
    return s


def test_search_finds_a_step_by_its_output(store):
    out = store.search("denied")
    assert [r["step_id"] for r in out["results"]] == ["c"]
    assert out["engine"] in ("fts5", "substring-scan")


def test_results_are_steps_not_runs(store):
    out = store.search("the")
    assert all("step_id" in r and "trace_id" in r for r in out["results"])
    assert all(r["trace_name"] == "nightly" for r in out["results"])


def test_the_engine_that_answered_is_named(store):
    out = store.search("pytest")
    assert out["engine"] in ("fts5", "substring-scan")
    assert out["note"]


def test_filters_narrow_the_results(store):
    assert len(store.search("kind:tool_call")["results"]) == 2
    assert len(store.search("error:true")["results"]) == 1
    assert len(store.search("duration>1000")["results"]) == 1


def test_a_filter_and_free_text_combine(store):
    out = store.search("push kind:tool_call")
    assert [r["step_id"] for r in out["results"]] == ["c"]


def test_an_empty_query_returns_nothing_and_says_so(store):
    out = store.search("")
    assert out["results"] == []
    assert "type something" in out["note"]


def test_fts_operators_in_the_query_are_treated_as_text(store):
    """A person pasting an error message is not writing an FTS5 expression."""
    for probe in ('pytest OR "', "NEAR/", "foo(", '"unbalanced'):
        out = store.search(probe)  # must not raise
        assert isinstance(out["results"], list)


def test_deleting_a_trace_removes_it_from_the_index(store):
    assert store.search("denied")["results"]
    store.delete("t1")
    assert store.search("denied")["results"] == []


def test_clearing_removes_everything_from_the_index(store):
    store.clear()
    assert store.search("pytest")["results"] == []


def test_reimporting_does_not_duplicate_index_entries(store):
    store.import_trace(
        {
            "id": "t1",
            "name": "nightly",
            "started_at": "2026-01-01T00:00:00Z",
            "steps": [{"id": "a", "kind": "tool_call", "name": "pytest",
                       "input": "run the suite", "output": "17 passed"}],
        }
    )
    assert len(store.search("pytest")["results"]) == 1


# ------------------------------------------------------------------ routing


def test_the_search_route_is_not_shadowed_by_the_trace_id_route(tmp_path):
    """FastAPI matches in definition order. With `/api/traces/search` declared
    after `/api/traces/{trace_id}`, the literal path `search` was captured as
    a trace id and every query answered "trace not found" — a 404 that looks
    exactly like an empty result set from the browser's side."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from modelmri.server import create_app

    client = TestClient(create_app(trace_db=str(tmp_path / "t.sqlite")))
    r = client.get("/api/traces/search", params={"q": "anything"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "engine" in body and "results" in body
    assert "trace not found" not in r.text


def test_a_bad_filter_reaches_the_browser_as_422(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from modelmri.server import create_app

    client = TestClient(create_app(trace_db=str(tmp_path / "t.sqlite")))
    r = client.get("/api/traces/search", params={"q": "error:maybe"})
    assert r.status_code == 422, r.text
    assert "takes true or false" in r.json()["error"]


def test_search_results_carry_the_truncation_marker(tmp_path):
    s = TraceStore(tmp_path / "t.sqlite")
    s.import_trace(
        {
            "name": "run",
            "started_at": "2026-01-01T00:00:00Z",
            # A space after the word, deliberately: FTS5 matches whole tokens,
            # so "findme" followed immediately by 30,000 z's is a single
            # 30,006-character word that the query "findme" does not match.
            # That is the documented behaviour the response note states, not a
            # defect — but it makes an unrealistic fixture fail confusingly.
            "steps": [{"id": "s1", "kind": "tool_call", "name": "cat",
                       "output": "findme " + "z " * 15_000}],
        }
    )
    hit = s.search("findme")["results"]
    assert hit and hit[0]["truncated_by"] > 0


# ------------------------------------- regressions from the pre-push audit


def test_an_upgraded_store_gets_its_existing_traces_indexed(tmp_path):
    """THE blocking bug. `count(*) FROM step_fts` reads THROUGH an
    external-content table to `step`, so `indexed` equalled `stored` on every
    store an earlier version wrote and the backfill never ran — once, ever.
    Search then answered engine "fts5" with an empty list for every trace the
    user already had, and nothing in the payload dissented."""
    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(str(path))
    old.executescript(
        """
        CREATE TABLE trace (id TEXT PRIMARY KEY, name TEXT NOT NULL,
          started_at TEXT NOT NULL, meta TEXT NOT NULL DEFAULT '{}');
        CREATE TABLE step (id TEXT PRIMARY KEY, trace_id TEXT NOT NULL,
          parent_id TEXT, kind TEXT NOT NULL, name TEXT NOT NULL DEFAULT '',
          started_ms INTEGER NOT NULL, duration_ms INTEGER NOT NULL DEFAULT 0,
          input TEXT NOT NULL DEFAULT '', output TEXT NOT NULL DEFAULT '',
          tokens_in INTEGER, tokens_out INTEGER,
          error INTEGER NOT NULL DEFAULT 0, seq INTEGER NOT NULL);
        INSERT INTO trace VALUES('t1','yesterday','2026-08-01T00:00:00Z','{}');
        INSERT INTO step VALUES('s1','t1',NULL,'tool_call','pytest',0,7,
          'the flaky migration test','17 passed',NULL,NULL,0,0);
        """
    )
    old.commit()
    old.close()

    store = TraceStore(path)
    hits = store.search("flaky")
    assert [h["step_id"] for h in hits["results"]] == ["s1"], hits


def test_reimporting_changed_text_does_not_leave_the_old_words_findable(tmp_path):
    """`INSERT OR REPLACE INTO trace` cascade-deletes the steps, so retracting
    after it read an empty table and every old term survived — bound to rowids
    SQLite then reused."""
    store = TraceStore(tmp_path / "t.sqlite")
    doc = {
        "id": "t1", "name": "run", "started_at": "2026-01-01T00:00:00Z",
        "steps": [{"id": "s1", "kind": "tool_call", "name": "x",
                   "input": "zebra", "output": ""}],
    }
    store.import_trace(doc)
    assert store.search("zebra")["results"]

    doc["steps"][0]["input"] = "giraffe omega"
    store.import_trace(doc)
    assert store.search("zebra")["results"] == []
    assert len(store.search("giraffe")["results"]) == 1


def test_results_are_ordered_by_the_real_clock_not_offset_within_a_run(tmp_path):
    """`step.started_ms` is milliseconds since that trace's own start, so
    ordering by it ranked hits by how deep into their run they happened. A step
    nine minutes into last month's run outranked one a second into today's, and
    the LIMIT then dropped today entirely — a full page of stale hits that
    looks complete."""
    store = TraceStore(tmp_path / "t.sqlite")
    store.import_trace({
        "id": "old", "name": "yesterday", "started_at": "2026-08-01T00:00:00Z",
        "steps": [{"id": "o1", "kind": "tool_call", "name": "pytest",
                   "input": "pytest", "started_ms": 540000}],
    })
    store.import_trace({
        "id": "new", "name": "today", "started_at": "2026-08-13T00:00:00Z",
        "steps": [{"id": "n1", "kind": "tool_call", "name": "pytest",
                   "input": "pytest", "started_ms": 1000}],
    })
    got = [h["trace_name"] for h in store.search("pytest")["results"]]
    assert got == ["today", "yesterday"], got


def test_every_hit_carries_the_time_it_happened(tmp_path):
    store = TraceStore(tmp_path / "t.sqlite")
    store.import_trace({
        "id": "t1", "name": "run", "started_at": "2026-08-13T09:00:00Z",
        "steps": [{"id": "s1", "kind": "tool_call", "name": "x",
                   "input": "needle"}],
    })
    assert store.search("needle")["results"][0]["trace_started_at"] == (
        "2026-08-13T09:00:00Z"
    )
