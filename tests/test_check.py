"""Structural CI assertions, and the exit codes a build script reads.

The sharpest rule in this file is that **nothing may exit 0 unless something
was actually verified**. A green tick from a check that asserted nothing is
how a typo'd flag turns into a build that always passes, and it is the same
defect as `--no-loops` passing on a trace too long to scan.
"""

from __future__ import annotations

import json

import pytest

from modelmri import check
from modelmri.errors import BadRequest


def _thrash(n=9):
    steps = []
    for i in range(n):
        steps.append(
            {"id": f"t{i}", "kind": "llm_call", "name": "think", "input": "next?",
             "started_ms": i * 900, "duration_ms": 300, "seq": len(steps)}
        )
        steps.append(
            {"id": f"a{i}", "kind": "tool_call", "name": "search", "input": "q",
             "started_ms": i * 900 + 400, "duration_ms": 120, "error": True,
             "seq": len(steps)}
        )
    return {"id": "thrash", "name": "react agent", "steps": steps}


def _clean():
    return {
        "id": "ok",
        "name": "clean",
        "steps": [
            {"id": "a", "kind": "llm_call", "name": "plan", "input": "x",
             "started_ms": 0, "duration_ms": 10, "seq": 0}
        ],
    }


# ------------------------------------------------------------ exit codes


def test_a_clean_run_passes_every_gate(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps(_clean()), encoding="utf-8")
    result, code, err = check.check(
        str(path), no_errors=True, max_steps=10, no_loops=True, max_repeat=3
    )
    assert code == check.PASS and result.ok and not err


def test_a_thrashing_run_fails(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps(_thrash()), encoding="utf-8")
    result, code, _ = check.check(
        str(path), no_errors=True, max_steps=12, no_loops=True, max_repeat=3
    )
    assert code == check.FAILED
    assert {a.name for a in result.assertions if not a.ok} == {
        "no-errors", "max-steps", "no-loops", "max-repeat"
    }


def test_no_assertions_chosen_never_exits_zero(tmp_path):
    """A green tick from a run that verified nothing is worse than a red one:
    a typo'd flag would otherwise make the build always pass."""
    path = tmp_path / "t.json"
    path.write_text(json.dumps(_clean()), encoding="utf-8")
    result, code, _ = check.check(str(path))
    assert code == check.NOTHING_CHECKED
    assert code != check.PASS
    assert "No assertions were chosen" in result.report()


def test_an_unreadable_trace_is_not_the_same_code_as_a_failure(tmp_path):
    """A missing file is a broken pipeline, not a broken agent."""
    result, code, err = check.check(str(tmp_path / "nope.json"), no_errors=True)
    assert result is None
    assert code == check.NOTHING_CHECKED != check.FAILED
    assert "no trace" in err


def test_a_file_that_is_not_a_trace_document_says_so(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    with pytest.raises(BadRequest, match="'steps' list"):
        check.load(str(path))


def test_unparseable_json_names_the_problem(tmp_path):
    path = tmp_path / "t.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(BadRequest, match="could not be read as a trace"):
        check.load(str(path))


def test_an_empty_target_is_refused():
    with pytest.raises(BadRequest, match="nothing to check"):
        check.load("")


# ------------------------------------------------------- the assertions


def test_max_steps_counts_every_step_not_only_llm_calls():
    result = check.run(_thrash(2), max_steps=3)
    assert not result.ok
    assert "4 steps, limit 3" in result.assertions[0].detail


def test_no_errors_names_the_failing_steps():
    result = check.run(_thrash(2), no_errors=True)
    detail = result.assertions[0].detail
    assert "2 step(s) recorded an error" in detail and "search" in detail
    assert result.assertions[0].step_ids == ["a0", "a1"]


def test_no_retry_storms_holds_when_failures_are_not_consecutive():
    doc = {
        "id": "x", "name": "x",
        "steps": [
            {"id": "a", "kind": "tool_call", "name": "s", "input": "q", "error": True,
             "started_ms": 0, "seq": 0},
            {"id": "b", "kind": "tool_call", "name": "s", "input": "q",
             "started_ms": 10, "seq": 1},
            {"id": "c", "kind": "tool_call", "name": "s", "input": "q", "error": True,
             "started_ms": 20, "seq": 2},
        ],
    }
    result = check.run(doc, no_retry_storms=True)
    assert result.ok


def test_no_loops_fails_rather_than_passes_when_the_scan_did_not_run():
    """A green check from a scan that did not run is worse than a red one."""
    from modelmri import patterns

    doc = {
        "id": "big", "name": "huge",
        "steps": [
            {"id": f"s{i}", "kind": "tool_call", "name": "a" if i % 2 else "b",
             "input": "x", "started_ms": i, "seq": i}
            for i in range(patterns.MAX_STEPS_FOR_CYCLES + 1)
        ],
    }
    result = check.run(doc, no_loops=True)
    assert not result.ok
    assert "NOT CHECKED" in result.assertions[0].detail


def test_max_repeat_reports_the_worst_offender():
    result = check.run(_thrash(9), max_repeat=3)
    assert not result.ok
    assert "ran 9 times with the same input, limit 3" in result.assertions[0].detail


# ------------------------------------------- timing is opt-in and says so


def test_the_timing_gate_warns_about_wall_clock_on_every_run():
    """The person reading a red CI log is not the person who added the flag."""
    result = check.run(_thrash(2), max_ms=1)
    detail = result.assertions[0].detail
    assert "WALL CLOCK" in detail
    assert "nothing to do with your diff" in detail


def test_untimed_steps_are_excluded_and_declared():
    doc = {
        "id": "x", "name": "x",
        "steps": [
            {"id": "a", "kind": "llm_call", "name": "p", "input": "x",
             "started_ms": 0, "duration_ms": 5, "seq": 0},
            {"id": "b", "kind": "llm_call", "name": "p", "input": "y",
             "started_ms": 5, "seq": 1},
        ],
    }
    result = check.run(doc, max_ms=100)
    detail = result.assertions[0].detail
    assert "5 ms across 1 timed step(s)" in detail
    assert "1 step(s) recorded no duration and are not in this total" in detail


def test_timing_is_never_asserted_unless_asked_for():
    result = check.run(_thrash(9), no_errors=True)
    assert [a.name for a in result.assertions] == ["no-errors"]


# ---------------------------------------------------------------- report


def test_the_report_names_the_trace_and_counts_the_failures():
    text = check.run(_thrash(9), no_errors=True, max_steps=2).report()
    assert "react agent (18 steps)" in text
    assert "2 of 2 assertions failed." in text


def test_the_report_serialises_for_a_script():
    doc = check.run(_thrash(2), no_errors=True).to_dict()
    assert doc["ok"] is False
    assert doc["assertions"][0]["name"] == "no-errors"
    assert doc["n_steps"] == 4


def test_it_imports_nothing_heavy():
    """This has to run in a build container: no torch, no transformers, no
    network. A heavy import here would make the CI step slow and fragile for
    no benefit."""
    import subprocess
    import sys

    code = (
        "import sys; import modelmri.check; "
        "bad=[m for m in ('torch','transformers','numpy') if m in sys.modules]; "
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert out.stdout.strip() == "", f"modelmri.check pulled in {out.stdout.strip()}"
