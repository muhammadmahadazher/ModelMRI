# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Exact predicates over recorded runs, and the refusals that keep them exact.

The sharpest rule under test is the roadmap's own caveat: duration outliers
over three traces are not statistics. A "slowest 10%" of four runs is one run,
and calling it an outlier is arithmetic dressed as evidence — so the rule is
SKIPPED with the n printed, never quietly answered "no matches", which reads
identically to having looked.
"""

from __future__ import annotations

import re

import pytest

from modelmri import rubric, step_kinds
from modelmri.errors import BadRequest


def _run(tid, *, steps=None, name="agent"):
    return ({"id": tid, "name": name}, steps or [])


def _step(
    sid, kind="tool_call", *, payload="q", output="ok", ms=0, dur=10, error=False
):
    return {
        "id": sid,
        "kind": kind,
        "name": kind,
        "input": payload,
        "output": output,
        "started_ms": ms,
        "duration_ms": dur,
        "error": error,
    }


# ------------------------------------------------------------ the parsing


def test_a_rule_must_be_named_by_the_person_who_wrote_it():
    """The name appears beside a matching run, and nothing here invents one."""
    with pytest.raises(BadRequest, match="every rule needs a name"):
        rubric.parse_rule({"kind": "has_error"})


def test_an_unknown_kind_is_refused_with_what_is_available():
    with pytest.raises(BadRequest, match="has_error"):
        rubric.parse_rule({"name": "x", "kind": "vibes"})


def test_a_bad_regex_is_refused_and_points_at_the_position():
    """The POSITION, an int — not `str(err)`. Interpolating a caught
    exception's own text is what `test_no_exception_leaks` forbids, and this
    message can author itself."""
    with pytest.raises(BadRequest) as caught:
        rubric.parse_rule(
            {"name": "x", "kind": "any_input_matches", "pattern": "([unclosed"}
        )
    message = str(caught.value)
    assert "not a valid regular expression" in message
    assert "at character" in message
    # The engine's own wording stays on the traceback, not in the message.
    assert "unterminated" not in message


def test_a_matching_rule_with_no_pattern_is_refused():
    with pytest.raises(BadRequest, match="none was given"):
        rubric.parse_rule({"name": "x", "kind": "tool_input_matches"})


def test_an_enormous_pattern_is_refused():
    big = "a" * (rubric.MAX_PATTERN_CHARS + 1)
    with pytest.raises(BadRequest, match="cap is"):
        rubric.parse_rule({"name": "x", "kind": "output_matches", "pattern": big})


def test_a_step_kind_no_step_can_have_is_refused():
    with pytest.raises(BadRequest, match="a step may be one of"):
        rubric.parse_rule({"name": "x", "kind": "kind_count", "step_kind": "telepathy"})


def test_a_nonsense_percentile_is_refused():
    with pytest.raises(BadRequest, match="between 0 and 100"):
        rubric.parse_rule({"name": "x", "kind": "slowest_percent", "value": 140})


def test_a_whole_rubric_parses_from_json():
    rules = rubric.parse('[{"name": "failed", "kind": "has_error"}]')
    assert len(rules) == 1 and rules[0].kind == "has_error"


def test_too_many_rules_are_refused_rather_than_truncated():
    many = [{"name": f"r{i}", "kind": "has_error"} for i in range(rubric.MAX_RULES + 1)]
    with pytest.raises(BadRequest, match="Cut it rather than"):
        rubric.parse(many)


# ----------------------------------------------------------- the matching


def test_has_error_matches_only_runs_with_an_error():
    rules = rubric.parse([{"name": "failed", "kind": "has_error"}])
    out = rubric.score(
        [
            _run("a", steps=[_step("s1")]),
            _run("b", steps=[_step("s1"), _step("s2", error=True)]),
        ],
        rules,
    )
    assert out.counts() == {"failed": 1}
    assert out.rows[1].matched == ["failed"]
    assert out.rows[1].hits[0].step_ids == ["s2"]


def test_kind_count_counts_only_that_kind():
    rules = rubric.parse(
        [
            {
                "name": "chatty",
                "kind": "kind_count",
                "step_kind": "llm_call",
                "op": "gt",
                "value": 2,
            }
        ]
    )
    steps = [_step(f"s{i}", kind="llm_call") for i in range(3)] + [_step("t")]
    out = rubric.score([_run("a", steps=steps)], rules)
    assert out.rows[0].matched == ["chatty"]
    assert "3 llm_call step(s)" in out.rows[0].hits[0].detail


def test_tool_input_matches_ignores_llm_steps():
    rules = rubric.parse(
        [{"name": "rm", "kind": "tool_input_matches", "pattern": r"rm\s+-rf"}]
    )
    out = rubric.score(
        [
            _run("a", steps=[_step("s1", kind="llm_call", payload="rm -rf /")]),
            _run("b", steps=[_step("s1", kind="tool_call", payload="rm -rf /")]),
        ],
        rules,
    )
    assert out.rows[0].matched == []
    assert out.rows[1].matched == ["rm"]


def test_any_input_matches_looks_at_every_kind():
    rules = rubric.parse(
        [{"name": "rm", "kind": "any_input_matches", "pattern": r"rm\s+-rf"}]
    )
    out = rubric.score(
        [_run("a", steps=[_step("s1", kind="llm_call", payload="rm -rf /")])], rules
    )
    assert out.rows[0].matched == ["rm"]


def test_output_matches_reads_the_output_not_the_input():
    rules = rubric.parse(
        [{"name": "refusal", "kind": "output_matches", "pattern": "I cannot"}]
    )
    out = rubric.score(
        [
            _run("a", steps=[_step("s1", payload="I cannot", output="fine")]),
            _run("b", steps=[_step("s1", payload="fine", output="I cannot help")]),
        ],
        rules,
    )
    assert out.rows[0].matched == []
    assert out.rows[1].matched == ["refusal"]


def test_every_operator_works():
    for op, value, expect in (
        ("gt", 2, True),
        ("gte", 3, True),
        ("lt", 4, True),
        ("lte", 3, True),
        ("eq", 3, True),
        ("gt", 3, False),
    ):
        rules = rubric.parse(
            [{"name": "n", "kind": "step_count", "op": op, "value": value}]
        )
        out = rubric.score([_run("a", steps=[_step(f"s{i}") for i in range(3)])], rules)
        assert bool(out.rows[0].matched) is expect, f"{op} {value}"


# ------------------------------------------------------------ the duration


def test_the_span_is_last_end_minus_first_start():
    """An imported trace whose offsets begin at a wall-clock epoch would
    otherwise report ~1.7 trillion ms and win every duration rule."""
    steps = [
        _step("s1", ms=1_700_000_000_000, dur=100),
        _step("s2", ms=1_700_000_000_500, dur=100),
    ]
    assert rubric._total_ms(steps) == 600


def test_a_step_with_no_duration_does_not_shorten_the_run():
    """None means nobody wrote one down; treating it as 0 would make a run
    look faster than it was measured to be."""
    steps = [_step("s1", ms=0, dur=None), _step("s2", ms=500, dur=100)]
    assert rubric._total_ms(steps) == 600


def test_duration_over_uses_the_span():
    rules = rubric.parse([{"name": "slow", "kind": "duration_over", "value": 500}])
    out = rubric.score(
        [
            _run("a", steps=[_step("s1", ms=0, dur=100)]),
            _run("b", steps=[_step("s1", ms=0, dur=100), _step("s2", ms=900, dur=50)]),
        ],
        rules,
    )
    assert out.rows[0].matched == []
    assert out.rows[1].matched == ["slow"]


# ------------------------------- outliers over three traces are not statistics


def test_a_distribution_rule_refuses_below_the_minimum_and_prints_n():
    """THE rule of this module. Quietly answering 'no matches' reads
    identically to having looked."""
    rules = rubric.parse([{"name": "slowest", "kind": "slowest_percent", "value": 10}])
    out = rubric.score([_run(f"t{i}", steps=[_step("s")]) for i in range(4)], rules)
    assert "slowest" in out.skipped
    assert "4 run(s)" in out.skipped["slowest"]
    assert f"at least {rubric.MIN_TRACES_FOR_OUTLIERS}" in out.skipped["slowest"]
    assert "arithmetic dressed as evidence" in out.skipped["slowest"]


def test_a_skipped_rule_produces_no_hits_at_all():
    """Not "matched: false" — that is an answer, and there is not one."""
    rules = rubric.parse([{"name": "slowest", "kind": "slowest_percent", "value": 10}])
    out = rubric.score([_run(f"t{i}", steps=[_step("s")]) for i in range(3)], rules)
    assert all(row.hits == [] for row in out.rows)
    assert out.counts() == {}


def test_the_report_says_a_rule_was_not_evaluated():
    rules = rubric.parse([{"name": "slowest", "kind": "slowest_percent", "value": 10}])
    out = rubric.score([_run("t1", steps=[_step("s")])], rules)
    assert "NOT EVALUATED" in out.means()


def test_with_enough_runs_the_distribution_rule_answers():
    rules = rubric.parse([{"name": "slowest", "kind": "slowest_percent", "value": 20}])
    runs = [
        _run(f"t{i}", steps=[_step("s", ms=0, dur=i * 100)])
        for i in range(rubric.MIN_TRACES_FOR_OUTLIERS + 2)
    ]
    out = rubric.score(runs, rules)
    assert out.skipped == {}
    matched = [r.trace_id for r in out.rows if r.matched]
    assert matched, "nothing was flagged with a full sample"
    # The slowest runs, not the fastest.
    assert "t9" in matched and "t0" not in matched


# ------------------------------------------------ nothing here is a verdict


VERDICT_WORDS = re.compile(
    r"\b(bad|poor|wrong|failed|failure|excessive|wasteful|should|ought|"
    r"severity|critical|suspicious)\b",
    re.I,
)


def test_no_authored_sentence_passes_judgement():
    """A rubric that prints "failed" for something it merely counted is a
    judgement with no judge behind it. The user's own rule NAMES are theirs
    and are excluded — those are their words, not this module's."""
    rules = rubric.parse(
        [
            {"name": "r1", "kind": "has_error"},
            {"name": "r2", "kind": "step_count", "op": "gt", "value": 1},
            {"name": "r3", "kind": "slowest_percent", "value": 10},
        ]
    )
    out = rubric.score([_run("a", steps=[_step("s", error=True)])], rules)
    texts = [out.means()] + [r.means() for r in out.rules]
    texts += [h.detail for row in out.rows for h in row.hits]
    texts += list(out.skipped.values())
    for text in texts:
        hit = VERDICT_WORDS.search(text)
        assert not hit, f"{hit.group(0)!r} in {text}"


def test_the_report_says_no_model_was_asked():
    rules = rubric.parse([{"name": "r", "kind": "has_error"}])
    assert "no model was asked" in rubric.score([_run("a")], rules).means()


# ----------------------------------------------------------------- shapes


def test_no_runs_is_answered_not_refused():
    out = rubric.score([], rubric.parse([{"name": "r", "kind": "has_error"}]))
    assert out.n_traces == 0 and out.rows == []
    assert "No run matched any rule" in out.means()


def test_no_rules_is_answered_not_refused():
    out = rubric.score([_run("a")], [])
    assert out.rows[0].hits == []


def test_the_report_serialises_for_the_wire():
    rules = rubric.parse([{"name": "failed", "kind": "has_error"}])
    doc = rubric.score([_run("a", steps=[_step("s", error=True)])], rules).to_dict()
    assert doc["counts"] == {"failed": 1}
    assert doc["rules"][0]["means"]
    assert isinstance(doc["means"], str)


# ------------------------- found by an adversarial review of this module


def test_duration_over_honours_the_operator_it_validated():
    """`parse_rule` validates `op` against OPERATORS, so a rubric written with
    `op: "lt"` parses without complaint. `_apply` then hardcoded `>` and ran a
    different rule than the one that was accepted — measured, a 100 ms run
    against `lt 500` did not match. Validating a field and then ignoring it is
    worse than not offering it."""
    steps = [_step("s", ms=0, dur=100)]
    quick = rubric.parse(
        [{"name": "quick", "kind": "duration_over", "op": "lt", "value": 500}]
    )
    assert rubric.score([_run("a", steps=steps)], quick).rows[0].matched == ["quick"]

    slow = rubric.parse(
        [{"name": "slow", "kind": "duration_over", "op": "gt", "value": 500}]
    )
    assert rubric.score([_run("a", steps=steps)], slow).rows[0].matched == []


def test_two_rules_cannot_share_a_name():
    """The name is the key: `slow_cut`, `skipped` and `counts()` are all dicts
    keyed by it. Measured, two rules called "same" reported
    `counts() == {"same": 1}` for one that matched and one that did not."""
    with pytest.raises(BadRequest, match="both called"):
        rubric.parse(
            [
                {"name": "same", "kind": "has_error"},
                {"name": "same", "kind": "step_count", "op": "gt", "value": 0},
            ]
        )


def test_a_skipped_rule_cannot_suppress_an_unrelated_one():
    """`score` skips by name, so a duplicate name would have silenced a rule
    that was perfectly answerable."""
    rules = rubric.parse(
        [
            {"name": "slowest", "kind": "slowest_percent", "value": 10},
            {"name": "failed", "kind": "has_error"},
        ]
    )
    out = rubric.score([_run("a", steps=[_step("s", error=True)])], rules)
    assert "slowest" in out.skipped
    assert out.counts() == {"failed": 1}, "the answerable rule still answered"


# ------------------------------------------- a threshold nobody wrote


def test_a_threshold_that_is_not_a_number_is_refused_not_zeroed():
    """`"500"` -- a string, the ordinary JSON slip -- used to become 0.0.

    `duration_over gt 0` is satisfied by every run that has ever been
    recorded, so the rubric reported a full-marks match against a threshold
    nobody wrote, and the document it came from still read `"500"`.
    """
    for bad in ("500", None, True, [500], {"ms": 500}):
        with pytest.raises(rubric.RubricError) as caught:
            rubric.parse_rule(
                {"name": "slow", "kind": "duration_over", "op": "gt", "value": bad}
            )
        assert "has to be a number" in str(caught.value)


def test_a_nonfinite_threshold_is_refused():
    """NaN loses every comparison and inf wins every one, both in silence."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(rubric.RubricError):
            rubric.parse_rule(
                {"name": "odd", "kind": "step_count", "op": "gt", "value": bad}
            )


def test_every_counting_kind_is_covered_by_the_number_check():
    """Derived from KINDS, so a new counting kind cannot quietly opt out."""
    for kind in rubric.KINDS:
        rule = {"name": "r", "kind": kind, "value": "nonsense"}
        if kind.endswith("_matches"):
            rule["pattern"] = "x"
        if kind == "kind_count":
            rule["step_kind"] = sorted(step_kinds.VALID_KINDS)[0]

        if kind in rubric.NUMERIC_KINDS:
            with pytest.raises(rubric.RubricError):
                rubric.parse_rule(rule)
        else:
            # No threshold to misread: these read `pattern` or nothing, and
            # must stay writable without a `value` at all.
            assert rubric.parse_rule(rule).value == 0.0


def test_a_rule_with_no_threshold_to_compare_still_parses():
    assert rubric.parse_rule({"name": "any", "kind": "has_error"}).value == 0.0
    assert (
        rubric.parse_rule(
            {"name": "grep", "kind": "output_matches", "pattern": "boom"}
        ).value
        == 0.0
    )


# ------------------------------------------------- a row has to BE one run


def test_a_row_carries_what_tells_one_run_from_another():
    """The defect a reader reported as "random data".

    Runs share names — every playground generation is called after the model,
    and one attempted with nothing loaded is called "generation" — so a
    hundred rows carrying only a name and a matched rule are a hundred rows
    nobody can order, date or choose between. The store already answers all
    of this in `list_traces`; the row was throwing it away.
    """
    summary = {
        "id": "abc123",
        "name": "generation",
        "started_at": "2026-08-17T17:31:18.088339+00:00",
        "total_ms": 2,
        "n_steps": 1,
        "n_errors": 1,
        "source": "app",
        "demo": False,
    }
    rules = [rubric.parse_rule({"name": "bad", "kind": "has_error"})]
    row = rubric.score([(summary, [_step("s", error=True)])], rules).rows[0].to_dict()

    assert row["started_at"] == summary["started_at"]
    assert row["total_ms"] == 2
    assert row["n_steps"] == 1
    assert row["n_errors"] == 1
    assert row["source"] == "app"
    assert row["demo"] is False
    assert row["matched"] == ["bad"]


def test_a_run_with_no_recorded_duration_reports_none_rather_than_zero():
    """`None` is "nobody recorded how long this took". `0` is "it finished
    inside a millisecond". Collapsing them invents the fastest run in the
    list, and the panel sorts and reads against exactly that number."""
    rules = [rubric.parse_rule({"name": "bad", "kind": "has_error"})]
    unknown = {"id": "a", "name": "x"}
    instant = {"id": "b", "name": "x", "total_ms": 0}

    rows = rubric.score([(unknown, [_step("s")]), (instant, [_step("s")])], rules).rows
    assert rows[0].total_ms is None
    assert rows[1].total_ms == 0


def test_the_kind_of_run_survives_to_the_row():
    """`demo` and `source` are distinctions the trace store keeps on purpose:
    scripted sample data, a playground generation and a run of the reader's
    own agent code are three different things. A row that dropped them is a
    row where somebody debugs a demo."""
    rules = [rubric.parse_rule({"name": "bad", "kind": "has_error"})]
    scripted = {"id": "a", "name": "demo-run", "demo": True, "source": ""}
    mine = {"id": "b", "name": "my-agent", "demo": False, "source": ""}

    rows = rubric.score([(scripted, [_step("s")]), (mine, [_step("s")])], rules).rows
    assert rows[0].demo is True
    assert rows[1].demo is False


# ------------------------------ an unrecorded duration is not a fast one


def _steps(*durations, start=0):
    """Steps at successive starts; `None` means nobody recorded a length."""
    out, at = [], start
    for i, d in enumerate(durations):
        step = {"id": f"s{i}", "kind": "llm_call", "name": "call", "started_ms": at}
        if d is not None:
            step["duration_ms"] = d
        out.append(step)
        at += int(d or 100)
    return out


def test_a_run_nobody_timed_has_no_wall_clock():
    """`_total_ms`'s own docstring says treating an absent length as 0 "would
    shorten a run to make it look faster than it was measured to be" — and
    that is exactly what it did when NO step carried one.

    The span from first start to last end collapses to 0, and 0 went on to
    `_compare` as a measurement: a run nobody timed matched "under 500 ms"
    with the detail "0 ms of recorded wall clock".
    """
    from modelmri import rubric as r

    assert r._total_ms(_steps(None, None, None)) is None
    assert r._total_ms([]) is None
    # Some durations is a floor, which is the honest best available.
    assert r._total_ms(_steps(400, None)) == 400
    assert r._total_ms(_steps(1200)) == 1200


def test_a_float_duration_counts():
    """`isinstance(length, int)` dropped a step recorded as `1500.0`, which is
    what `(t1 - t0) * 1000` produces in any recorder that does not round."""
    from modelmri import rubric as r

    assert r._total_ms(_steps(300.0)) == 300
    # And a bool is not a duration, for the reason `isinstance(True, int)` is
    # True everywhere else in this codebase.
    assert r._total_ms(_steps(True)) is None


def test_a_duration_rule_does_not_match_a_run_it_could_not_time():
    from modelmri import rubric as r

    runs = [
        ({"id": "A", "name": "untimed"}, _steps(None, None)),
        ({"id": "B", "name": "slow"}, _steps(1200)),
    ]
    rule = r.Rule(
        name="under 500ms",
        kind="duration_over",
        pattern="",
        step_kind="",
        op="lt",
        value=500,
    )
    by_name = {row.name: row for row in r.score(runs, [rule]).rows}
    untimed = by_name["untimed"].hits[0]
    assert untimed.matched is False
    assert "unknown rather than zero" in untimed.detail
    # The timed run is judged on its real number, unchanged.
    assert by_name["slow"].hits[0].matched is False
    assert "1200 ms" in by_name["slow"].hits[0].detail
