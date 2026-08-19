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
            {
                "id": f"t{i}",
                "kind": "llm_call",
                "name": "think",
                "input": "next?",
                "started_ms": i * 900,
                "duration_ms": 300,
                "seq": len(steps),
            }
        )
        steps.append(
            {
                "id": f"a{i}",
                "kind": "tool_call",
                "name": "search",
                "input": "q",
                "started_ms": i * 900 + 400,
                "duration_ms": 120,
                "error": True,
                "seq": len(steps),
            }
        )
    return {"id": "thrash", "name": "react agent", "steps": steps}


def _clean():
    return {
        "id": "ok",
        "name": "clean",
        "steps": [
            {
                "id": "a",
                "kind": "llm_call",
                "name": "plan",
                "input": "x",
                "started_ms": 0,
                "duration_ms": 10,
                "seq": 0,
            }
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
        "no-errors",
        "max-steps",
        "no-loops",
        "max-repeat",
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
        "id": "x",
        "name": "x",
        "steps": [
            {
                "id": "a",
                "kind": "tool_call",
                "name": "s",
                "input": "q",
                "error": True,
                "started_ms": 0,
                "seq": 0,
            },
            {
                "id": "b",
                "kind": "tool_call",
                "name": "s",
                "input": "q",
                "started_ms": 10,
                "seq": 1,
            },
            {
                "id": "c",
                "kind": "tool_call",
                "name": "s",
                "input": "q",
                "error": True,
                "started_ms": 20,
                "seq": 2,
            },
        ],
    }
    result = check.run(doc, no_retry_storms=True)
    assert result.ok


def test_no_loops_fails_rather_than_passes_when_the_scan_did_not_run():
    """A green check from a scan that did not run is worse than a red one."""
    from modelmri import patterns

    doc = {
        "id": "big",
        "name": "huge",
        "steps": [
            {
                "id": f"s{i}",
                "kind": "tool_call",
                "name": "a" if i % 2 else "b",
                "input": "x",
                "started_ms": i,
                "seq": i,
            }
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
        "id": "x",
        "name": "x",
        "steps": [
            {
                "id": "a",
                "kind": "llm_call",
                "name": "p",
                "input": "x",
                "started_ms": 0,
                "duration_ms": 5,
                "seq": 0,
            },
            {
                "id": "b",
                "kind": "llm_call",
                "name": "p",
                "input": "y",
                "started_ms": 5,
                "seq": 1,
            },
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


def test_the_json_flag_actually_runs(tmp_path, capsys):
    """`modelmri check --json` called `json.dumps` in a module that never
    imported json — a NameError on every invocation. Every test here drove
    `check.py` directly, so nothing exercised the CLI path and ruff's F821 was
    the only thing that saw it."""
    import argparse

    from modelmri.cli import check_trace

    path = tmp_path / "t.json"
    path.write_text(json.dumps(_clean()), encoding="utf-8")
    args = argparse.Namespace(
        target=str(path),
        no_errors=True,
        max_steps=None,
        no_retry_storms=False,
        no_loops=False,
        max_repeat=None,
        max_ms=None,
        json=True,
    )
    assert check_trace(args) == check.PASS
    doc = json.loads(capsys.readouterr().out)
    assert doc["ok"] is True


def test_the_json_flag_reports_an_unreadable_trace_as_json(tmp_path, capsys):
    import argparse

    from modelmri.cli import check_trace

    args = argparse.Namespace(
        target=str(tmp_path / "nope.json"),
        no_errors=True,
        max_steps=None,
        no_retry_storms=False,
        no_loops=False,
        max_repeat=None,
        max_ms=None,
        json=True,
    )
    assert check_trace(args) == check.NOTHING_CHECKED
    doc = json.loads(capsys.readouterr().out)
    assert doc["ok"] is False and doc["error"]


# ------------------------------------------- the gate that could not go red


def _run(steps, **kw):
    from modelmri import check as check_mod

    result = check_mod.run({"name": "n", "steps": steps}, **kw)
    return next(a for a in result.assertions if a.name == "max-ms")


def test_max_ms_reads_a_duration_recorded_as_a_float():
    """A MERGE GATE THAT COULD NOT GO RED.

    This kept only `isinstance(_, int)`, so a step recorded as `1500.0` — what
    `(t1 - t0) * 1000` produces in any recorder that does not round, and what
    `modelmri-record` stores when the caller passes `duration_ms` directly —
    was silently discarded. Seven of them, 10,500 ms of real work, summed to 0,
    `0 <= 5000` passed, and the log read "0 ms across 0 timed step(s)".

    The same document imported into the store is coerced by `traces._ms` and
    exported by `otel._span_times` as 1500 ms, so three readers agreed about
    this file and the gate disagreed with all of them.
    """
    steps = [
        {"id": f"s{i}", "kind": "llm_call", "name": "call", "duration_ms": 1500.0}
        for i in range(7)
    ]
    a = _run(steps, max_ms=5000)
    assert a.ok is False, a.detail
    assert "10500 ms across 7 timed step(s)" in a.detail

    # And ints are untouched, so the fix is a widening rather than a swap.
    ints = [dict(s, duration_ms=1500) for s in steps]
    assert _run(ints, max_ms=5000).detail == a.detail


def test_max_ms_passes_a_run_that_is_actually_under_the_limit():
    """The guard must not fire on the ordinary case."""
    a = _run(
        [{"id": "s0", "kind": "llm_call", "name": "c", "duration_ms": 1200.0}],
        max_ms=5000,
    )
    assert a.ok is True
    assert "1200 ms across 1 timed step(s)" in a.detail


def test_a_duration_that_is_present_and_unreadable_fails_the_gate():
    """Different from a step that recorded none, and it must not pass.

    Something wrote a number-shaped field this cannot read. Folding that into
    "recorded no duration" and going green is the same failure the float case
    was: a build passing on the strength of measurements that were discarded.
    `no-loops` already states the policy — "a green check from a scan that did
    not run is worse than a red one".
    """
    for bad in ("1500", float("nan"), float("inf"), [1500], {"ms": 1500}):
        a = _run(
            [{"id": "s0", "kind": "llm_call", "name": "c", "duration_ms": bad}],
            max_ms=5000,
        )
        assert a.ok is False, f"{bad!r} was admitted: {a.detail}"
        assert "cannot read" in a.detail


def test_true_is_not_one_millisecond():
    """`isinstance(True, int)` is True, so the bool guard has to come first.

    `session.py` already writes `isinstance(value, int) and not
    isinstance(value, bool)` for this same field; `check.py` was the one place
    that skipped it, and admitted `"duration_ms": true` as a 1 ms step.
    """
    a = _run(
        [{"id": "s0", "kind": "llm_call", "name": "c", "duration_ms": True}],
        max_ms=5000,
    )
    assert a.ok is False
    assert "1 ms across" not in a.detail


def test_a_step_that_recorded_no_duration_is_still_just_absent():
    """Absent is not unreadable, and absent alone does not fail the build."""
    a = _run([{"id": "s0", "kind": "llm_call", "name": "c"}], max_ms=5000)
    assert a.ok is True
    assert "recorded no duration" in a.detail
    assert "cannot read" not in a.detail


def test_a_list_of_three_says_how_many_there_were():
    """`no-errors` lists five and marks its cut with " …"; these two listed
    three and said nothing, so a build with twenty retry storms reported three
    and read as though that was all of them.

    The count is what a reader acts on. Three storms is a flaky dependency;
    twenty is a broken loop, and the detail line said the same thing for both.
    """
    steps = [
        {
            "id": f"s{i}",
            "kind": "tool_call",
            "name": f"fetch{i // 2}",
            "error": True,
            "started_ms": i * 10,
            "duration_ms": 5,
        }
        for i in range(40)
    ]
    from modelmri import check as check_mod

    result = check_mod.run({"name": "n", "steps": steps}, no_retry_storms=True)
    a = next(x for x in result.assertions if x.name == "no-retry-storms")
    assert a.ok is False
    assert "20 in total" in a.detail, a.detail
    assert "and 17 more" in a.detail

    # Three or fewer says nothing extra: "3 in total" under three named ones
    # is noise, and the sibling above behaves the same way.
    few = check_mod.run({"name": "n", "steps": steps[:4]}, no_retry_storms=True)
    b = next(x for x in few.assertions if x.name == "no-retry-storms")
    assert "in total" not in b.detail, b.detail
