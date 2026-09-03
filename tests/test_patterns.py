# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Structural findings, against planted structures.

The hardest assertion in this file is not that the counts are right — it is
that no finding is ever WORDED as a verdict. Paginating an API 14 times and
thrashing against a failing tool 14 times are structurally identical, and the
moment this module says "excessive" it becomes the model judgement it exists
to replace, except without the model.
"""

from __future__ import annotations

import re

from modelmri import patterns


def _step(
    sid, kind="tool_call", name="search", payload="q", *, seq=0, ms=0, error=False
):
    return {
        "id": sid,
        "kind": kind,
        "name": name,
        "input": payload,
        "seq": seq,
        "started_ms": ms,
        "error": error,
    }


# ----------------------------------------------------------------- repeats


def test_the_same_call_fourteen_times_is_counted_exactly():
    steps = [_step(f"s{i}", seq=i, ms=i * 100) for i in range(14)]
    found = patterns.analyse(steps)
    assert len(found.repeats) == 1
    assert found.repeats[0].count == 14
    assert len(found.repeats[0].step_ids) == 14


def test_a_different_input_is_a_different_step():
    steps = [
        _step("a", payload="page=1", seq=0),
        _step("b", payload="page=2", seq=1),
        _step("c", payload="page=1", seq=2),
    ]
    found = patterns.analyse(steps)
    assert len(found.repeats) == 1
    assert found.repeats[0].count == 2
    assert set(found.repeats[0].step_ids) == {"a", "c"}


def test_the_same_input_under_a_different_name_does_not_group():
    steps = [_step("a", name="search", seq=0), _step("b", name="fetch", seq=1)]
    assert patterns.analyse(steps).repeats == []


def test_two_inputs_that_differ_late_are_not_the_same_input():
    """This asserted the opposite, as a documented consequence of hashing only
    the first 4,000 characters.

    It is the case that matters. Agent steps share a system prompt — `otel.py`
    maps `gen_ai.input.messages` onto `input` — so every LLM call in a run
    begins with the same multi-kilobyte preamble and differs only after it.
    Under a prefix hash, twenty-seven distinct calls became "ran 27 times with
    the same input", against sentences promising exact matching, and the count
    feeds `cli.py --max-repeat` into `check.py` — so it failed builds over
    calls that were never repeats.
    """
    shared = "x" * 20_000  # what a real system prompt looks like here
    steps = [
        _step("a", payload=shared + "asked about A", seq=0),
        _step("b", payload=shared + "asked about B", seq=1),
    ]

    assert patterns.analyse(steps).repeats == [], (
        "two different questions behind one preamble are not a repeat"
    )


def test_genuinely_identical_inputs_still_group():
    """The other half, so the fix cannot become "nothing ever repeats"."""
    same = "y" * 20_000
    steps = [_step("a", payload=same, seq=0), _step("b", payload=same, seq=1)]

    found = patterns.analyse(steps)

    assert found.repeats[0].count == 2
    assert set(found.repeats[0].step_ids) == {"a", "b"}


# ----------------------------------------------------------- retry storms


def test_consecutive_failures_of_one_name_are_one_finding():
    steps = [_step(f"e{i}", seq=i, ms=i * 500, error=True) for i in range(6)]
    found = patterns.analyse(steps)
    assert len(found.retry_storms) == 1
    storm = found.retry_storms[0]
    assert storm.count == 6
    assert storm.span_ms == 2500


def test_a_success_between_two_failures_breaks_the_run():
    """Calling that a storm would be the module inventing a narrative."""
    steps = [
        _step("e1", seq=0, ms=0, error=True),
        _step("ok", seq=1, ms=100),
        _step("e2", seq=2, ms=200, error=True),
    ]
    assert patterns.analyse(steps).retry_storms == []


def test_failures_far_apart_are_not_one_storm():
    steps = [
        _step("e1", seq=0, ms=0, error=True),
        _step("e2", seq=1, ms=90_000, error=True),
    ]
    assert patterns.analyse(steps).retry_storms == []


def test_two_different_names_failing_are_two_runs_not_one():
    steps = [
        _step("a", name="search", seq=0, ms=0, error=True),
        _step("b", name="search", seq=1, ms=100, error=True),
        _step("c", name="fetch", seq=2, ms=200, error=True),
        _step("d", name="fetch", seq=3, ms=300, error=True),
    ]
    storms = patterns.analyse(steps).retry_storms
    assert len(storms) == 2
    assert {s.label for s in storms} == {"search", "fetch"}


def test_a_step_with_no_timestamp_ends_the_run_rather_than_being_assumed_near():
    steps = [
        _step("a", seq=0, ms=0, error=True),
        {
            "id": "b",
            "kind": "tool_call",
            "name": "search",
            "input": "q",
            "seq": 1,
            "error": True,
        },
        _step("c", seq=2, ms=200, error=True),
    ]
    # The middle step cannot be placed in the window, so nothing claims a
    # three-failure storm that was never established.
    for storm in patterns.analyse(steps).retry_storms:
        assert storm.count <= 2


# ------------------------------------------------------------------ cycles


def test_a_repeating_sequence_is_found_with_its_length_and_count():
    seq = []
    for i in range(4):
        seq.append(_step(f"a{i}", name="think", seq=len(seq)))
        seq.append(_step(f"b{i}", name="act", seq=len(seq)))
    found = patterns.analyse(seq)
    assert found.cycles, "a 2-step block repeated 4 times was not found"
    cycle = found.cycles[0]
    assert cycle.cycle_length == 2
    assert cycle.count == 4
    assert "think" in cycle.label and "act" in cycle.label


def test_only_the_maximal_repeat_is_reported():
    """A cycle of 2 repeated 6 times also contains one of 4 repeated 3 times.
    Listing both reports one fact as three."""
    seq = []
    for i in range(6):
        seq.append(_step(f"a{i}", name="think", seq=len(seq)))
        seq.append(_step(f"b{i}", name="act", seq=len(seq)))
    cycles = patterns.analyse(seq).cycles
    covered = [sid for c in cycles for sid in c.step_ids]
    assert len(covered) == len(set(covered)), "a step was claimed by two cycles"


def test_no_cycle_where_there_is_none():
    steps = [_step(f"s{i}", name=f"n{i}", seq=i) for i in range(10)]
    assert patterns.analyse(steps).cycles == []


def test_a_trace_too_long_to_scan_says_so_rather_than_reporting_none():
    """'No cycles found' and 'not looked for' are different answers."""
    steps = [
        _step(f"s{i}", name="a" if i % 2 else "b", seq=i)
        for i in range(patterns.MAX_STEPS_FOR_CYCLES + 1)
    ]
    found = patterns.analyse(steps)
    assert found.cycles_scanned is False
    assert found.cycles == []
    assert "CYCLES WERE NOT SCANNED" in found.means()


def test_a_scanned_trace_says_it_was_scanned():
    found = patterns.analyse([_step("a", seq=0)])
    assert found.cycles_scanned is True
    assert "CYCLES WERE NOT SCANNED" not in found.means()


# ------------------------------------------------- counts, never verdicts


VERDICT_WORDS = re.compile(
    r"\b(excessive|redundant|wasteful|inefficient|should|ought|problem|bad|"
    r"poor|wrong|suspicious|anomal\w*|severity|critical|warning)\b",
    re.I,
)


def test_no_finding_is_worded_as_a_verdict():
    """The whole discipline of the module. Paginating an API 14 times and
    thrashing against a failing tool 14 times are structurally identical."""
    seq = []
    for i in range(4):
        seq.append(_step(f"a{i}", name="think", seq=len(seq)))
        seq.append(
            _step(f"b{i}", name="act", seq=len(seq), error=True, ms=len(seq) * 10)
        )
    found = patterns.analyse(seq)
    texts = [f.means() for f in found.all] + [found.means()]
    for text in texts:
        hit = VERDICT_WORDS.search(text)
        assert not hit, f"a finding passes judgement ({hit.group(0)!r}): {text}"


def test_a_finding_carries_no_severity_or_score():
    steps = [_step(f"s{i}", seq=i) for i in range(3)]
    finding = patterns.analyse(steps).repeats[0]
    doc = finding.to_dict()
    for banned in ("severity", "score", "level", "priority", "verdict"):
        assert banned not in doc


def test_the_repeat_wording_names_the_benign_reading_too():
    steps = [_step(f"s{i}", seq=i) for i in range(14)]
    text = patterns.analyse(steps).repeats[0].means()
    assert "page-by-page walk" in text


def test_the_limits_are_stated_rather_than_implied_exhaustive():
    """Hashing the input misses a repeat whose prompt carries a timestamp."""
    found = patterns.analyse([_step("a", seq=0)])
    assert found.near_repeats_not_detected is True
    assert "timestamp" in found.means() and "not counted" in found.means()


# -------------------------------------------------------------- ordering


def test_steps_out_of_order_are_sorted_before_adjacency_is_judged():
    """Comparing consecutive steps in the wrong order invents adjacency."""
    steps = [
        _step("c", seq=2, ms=200, error=True),
        _step("a", seq=0, ms=0, error=True),
        _step("b", seq=1, ms=100),
    ]
    # In recorded order this is fail, ok, fail — not a storm.
    assert patterns.analyse(steps).retry_storms == []


# ------------------------------------------------------------ across runs


def _run(trace_id, n_repeat):
    return {
        "id": trace_id,
        "steps": [_step(f"{trace_id}-{i}", seq=i, ms=i * 10) for i in range(n_repeat)],
    }


def test_a_pattern_is_counted_over_runs_not_occurrences():
    """A pattern in one run is an anecdote."""
    runs = [_run("t1", 3), _run("t2", 5), {"id": "t3", "steps": [_step("x", seq=0)]}]
    out = patterns.across_runs(runs)
    top = out[0]
    assert top.n_runs == 2 and top.of_runs == 3
    assert top.total_count == 8
    assert "in 2 of 3 recorded run(s)" in top.means()


def test_a_pattern_twice_in_one_run_is_still_one_run():
    doc = {
        "id": "t1",
        "steps": [_step(f"s{i}", seq=i) for i in range(4)],
    }
    out = patterns.across_runs([doc])
    assert out[0].n_runs == 1
    assert out[0].total_count == 4


def test_no_runs_is_not_a_crash():
    assert patterns.across_runs([]) == []


def test_a_non_dict_in_the_list_is_skipped_not_fatal():
    out = patterns.across_runs([_run("t1", 3), None, "nonsense"])
    assert out and out[0].of_runs == 1


def test_the_cross_run_wording_is_also_not_a_verdict():
    out = patterns.across_runs([_run("t1", 3), _run("t2", 3)])
    for entry in out:
        hit = VERDICT_WORDS.search(entry.means())
        assert not hit, f"{hit.group(0)!r} in {entry.means()}"


# ----------------------------------------------------------------- shape


def test_an_empty_trace_is_answered_not_refused():
    found = patterns.analyse([])
    assert found.n_steps == 0
    assert found.all == []
    assert "No step ran twice" in found.means()


def test_findings_serialise_for_the_wire():
    seq = [_step(f"s{i}", seq=i, ms=i * 10) for i in range(3)]
    doc = patterns.analyse(seq).to_dict()
    assert doc["n_steps"] == 3
    assert doc["repeats"][0]["count"] == 3
    assert isinstance(doc["means"], str)


def test_the_cycle_length_window_is_reported_not_implied():
    """The constants call both bounds "reported rather than silent" and only
    the step cap was. MIN/MAX_CYCLE_LEN appeared nowhere but the loop range.

    An agent that repeats a 20-step sequence three times is a real loop and
    is outside the window — a legitimate limit. What was not legitimate is
    reporting it as "no sequence repeated back to back", which reads as
    "we looked and your agent does not loop".
    """
    steps = [
        {
            "id": str(i),
            "kind": "tool_call",
            "name": f"s{i % 20}",
            "input": f"x{i % 20}",
        }
        for i in range(60)
    ]
    found = patterns.analyse(steps)

    assert found.cycles == [], "20 is outside the searched window, as designed"
    said = found.means()
    assert f"lengths {patterns.MIN_CYCLE_LEN} to {patterns.MAX_CYCLE_LEN} steps" in said
    assert "not reported as absent" in said

    body = found.to_dict()
    assert body["cycle_len_min"] == patterns.MIN_CYCLE_LEN
    assert body["cycle_len_max"] == patterns.MAX_CYCLE_LEN


def test_the_window_is_stated_even_when_cycles_were_found():
    """ "3 repeating sequences" and "3 repeating sequences between 2 and 12
    steps" are different claims, and the reader acts on the first."""
    steps = [
        {
            "id": str(i),
            "kind": "tool_call",
            "name": f"s{i % 3}",
            "input": f"x{i % 3}",
        }
        for i in range(12)
    ]
    found = patterns.analyse(steps)
    assert found.cycles, "a 3-step loop is inside the window"
    assert "were searched for at lengths" in found.means()


def test_a_trace_too_long_to_scan_says_so_and_omits_the_window():
    """The window describes a scan that ran. When none did, the louder
    sentence is the one that belongs."""
    steps = [
        {"id": str(i), "kind": "tool_call", "name": "s", "input": "x"}
        for i in range(patterns.MAX_STEPS_FOR_CYCLES + 1)
    ]
    found = patterns.analyse(steps)
    assert found.cycles_scanned is False
    said = found.means()
    assert "CYCLES WERE NOT SCANNED" in said
    assert "were searched for at lengths" not in said
