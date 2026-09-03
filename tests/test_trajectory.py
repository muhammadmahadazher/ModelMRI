# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Trajectory alignment, and the conclusions each refusal exists to prevent.

The hardest assertion in this file is the one that asserts an ABSENCE: that
nowhere in the output is there a ratio. "2 steps missing, 1 extra, 3 with
changed arguments" is a description of two sequences; "Plan Adherence 0.71" is
a claim that the run was 29% wrong, and a shorter path is not a worse path.
One division is all it would take, so a structural test watches for it.

The rest are about the ways a sequence comparison lies quietly: a moved step
counted twice, a repeated call matched by membership so the third one goes
unnoticed, a prefix aligned when the run was too long so the tail of the plan
reads as skipped, and two different API keys redacted before they are compared
and therefore reported as unchanged.

Nothing here needs a model, a key or a network — `trajectory` does no I/O and
holds nothing, which is exactly what makes the arithmetic checkable.
"""

from __future__ import annotations

import json

import pytest

from modelmri import trajectory as tj


def _step(name, kind="tool_call", *, sid="", seq=None, args=None, payload=None):
    """One recorded step, in the shape `modelmri-record` writes."""
    out = {"kind": kind, "name": name}
    if sid:
        out["id"] = sid
    if seq is not None:
        out["seq"] = seq
    if args is not None:
        out["args"] = args
    if payload is not None:
        out["input"] = payload
    return out


def _plan(*names):
    return [_step(n) for n in names]


def _run(*names):
    return [_step(n, sid=f"s{i}", seq=i) for i, n in enumerate(names)]


# ------------------------------------------------------- the rule: no score


def test_nothing_in_the_output_is_a_ratio():
    """The single most important constraint, asserted structurally rather than
    trusted. A count is an integer; a score is a float, and one division is
    all it would take for `n_matched / n_reference` to appear beside them."""
    out = tj.align(
        reference=_plan("search", "read", "write"),
        candidate=_run("search", "write", "lint"),
    ).to_dict()

    floats = []

    def walk(value, where):
        if isinstance(value, bool):
            return
        if isinstance(value, float):
            floats.append(f"{where} = {value}")
        elif isinstance(value, dict):
            for k, v in value.items():
                walk(v, f"{where}.{k}")
        elif isinstance(value, list):
            for n, v in enumerate(value):
                walk(v, f"{where}[{n}]")

    walk(out, "alignment")
    assert not floats, (
        "a ratio appeared in a report that must only count: " + ", ".join(floats)
    )

    # And no FIELD may be named like one either. Checked on the key names
    # rather than on the whole document, because the sentence forbidding a
    # score has to contain the word "score" to forbid it.
    def keys_of(value, found):
        if isinstance(value, dict):
            for k, v in value.items():
                found.add(k)
                keys_of(v, found)
        elif isinstance(value, list):
            for v in value:
                keys_of(v, found)
        return found

    named = keys_of(out, set())
    assert not [
        k
        for k in named
        if any(
            word in k.lower()
            for word in ("adherence", "score", "similarity", "accuracy", "ratio")
        )
    ], sorted(named)


def test_the_no_score_sentence_travels_in_the_body_not_the_docs():
    """The reader who needs "a shorter path is not a worse path" is looking at
    the panel, not at a module docstring."""
    said = tj.align(
        reference=_plan("search", "read", "write"), candidate=_run("search", "write")
    ).means()
    assert "NOT A SCORE" in said
    assert "SHORTER PATH IS NOT A WORSE PATH" in said
    assert "does not know what any step was FOR" in said


def test_a_missing_step_is_never_worded_as_a_fault():
    """`patterns.py` holds this line for structural findings and it holds here:
    the moment a count is worded as a verdict it is a model judgement with no
    model behind it."""
    out = tj.align(
        reference=_plan("search", "read", "write"), candidate=_run("search", "write")
    )
    row = next(r for r in out.rows if r.status == tj.MISSING)
    said = row.means().lower()
    assert "skipped step or a shorter path" in said
    for verdict in ("should", "failed", "wrong", "incorrect", "violation"):
        assert verdict not in said


# --------------------------------------------------------------- alignment


def test_a_skipped_middle_step_is_one_missing_and_not_a_whole_mismatch():
    """The regression this module exists for. Index-by-index comparison would
    call `write` a substitution AND report the tail as changed; the alignment
    says one step is missing and the rest is intact."""
    out = tj.align(
        reference=_plan("search", "read", "write"), candidate=_run("search", "write")
    )
    assert (out.n_matched, out.n_missing, out.n_extra, out.n_reordered) == (2, 1, 0, 0)
    assert [r.name for r in out.rows if r.status == tj.MISSING] == ["read"]


def test_an_unplanned_step_is_extra_and_nothing_is_reported_missing():
    out = tj.align(
        reference=_plan("search", "write"),
        candidate=_run("search", "lint", "write"),
    )
    assert (out.n_matched, out.n_missing, out.n_extra) == (2, 0, 1)
    assert [r.name for r in out.rows if r.status == tj.EXTRA] == ["lint"]


def test_identical_trajectories_report_nothing_missing_and_nothing_extra():
    out = tj.align(
        reference=_plan("search", "read", "write"),
        candidate=_run("search", "read", "write"),
    )
    assert out.n_matched == 3
    assert out.n_missing == out.n_extra == out.n_reordered == 0
    assert all(r.status == tj.MATCHED for r in out.rows)


def test_a_moved_step_is_one_reordered_not_a_missing_plus_an_extra():
    """An edit distance alone reports a moved step twice — "read is missing,
    and there is an unexpected read" — which is two findings about one event
    and reads as worse than what happened."""
    out = tj.align(
        reference=_plan("search", "read", "write"),
        candidate=_run("search", "write", "read"),
    )
    assert out.n_reordered == 1
    assert out.n_missing == 0 and out.n_extra == 0
    moved = next(r for r in out.rows if r.status == tj.REORDERED)
    assert moved.reference_index is not None
    assert moved.candidate_index is not None
    assert moved.reference_index != moved.candidate_index


def test_reordered_never_claims_the_order_was_wrong():
    """A plan is frequently a set of things to do rather than a sequence, and
    nothing here knows which kind it was handed."""
    out = tj.align(
        reference=_plan("a", "b"),
        candidate=_run("b", "a"),
    )
    assert out.n_reordered == 1
    assert "SOMEWHERE ELSE, NOT OUT OF ORDER" in out.means()
    moved = next(r for r in out.rows if r.status == tj.REORDERED)
    assert "not the same claim as the order being wrong" in moved.means()


def test_two_disjoint_trajectories_say_nothing_aligned():
    """Nine missing and nine extra is arithmetically true and reads as a run
    that did everything wrong. The likelier cause is that the plan names the
    tools differently from the recorder, and the sentence has to say so before
    anybody acts on the counts."""
    out = tj.align(reference=_plan("a", "b", "c"), candidate=_run("x", "y", "z"))
    assert out.n_matched == 0
    assert out.n_missing == 3 and out.n_extra == 3
    assert "NOTHING ALIGNED" in out.means()
    assert "names the tools the way the recorder does" in out.means()


def test_rows_read_in_the_order_the_run_happened():
    """A listing sorted by status would put every missing step first and lose
    where in the run the gap actually is."""
    out = tj.align(
        reference=_plan("search", "read", "write"),
        candidate=_run("search", "lint", "write"),
    )
    assert [(r.status, r.name) for r in out.rows] == [
        (tj.MATCHED, "search"),
        (tj.MISSING, "read"),
        (tj.EXTRA, "lint"),
        (tj.MATCHED, "write"),
    ]


def test_a_row_points_at_the_recorded_step_it_is_about():
    out = tj.align(reference=_plan("search"), candidate=_run("search"))
    assert out.rows[0].step_id == "s0"


# ------------------------------------------------------- repeated identities


def test_three_identical_calls_against_two_is_one_missing_not_zero():
    """The bug in every naive matcher. Membership testing says "search is in
    both, everything matched"; a longest common subsequence says two of the
    three were made."""
    out = tj.align(
        reference=_plan("search", "search", "search"),
        candidate=_run("search", "search"),
    )
    assert out.n_matched == 2
    assert out.n_missing == 1
    assert out.n_extra == 0


def test_two_identical_calls_against_three_is_one_extra_not_zero():
    out = tj.align(
        reference=_plan("search", "search"),
        candidate=_run("search", "search", "search"),
    )
    assert out.n_matched == 2
    assert out.n_extra == 1
    assert out.n_missing == 0


def test_a_repeated_identity_says_which_occurrence_is_named_is_arbitrary():
    """Three alignments of exactly this length exist. `matched` is the same
    number in all of them; the row it is attributed to is not, and a reader who
    assumes the named occurrence is the one that went wrong is reading
    something this cannot tell them."""
    out = tj.align(
        reference=_plan("search", "search", "search"),
        candidate=_run("search", "search"),
    )
    assert out.repeated_identities == [
        {"identity": "tool_call search", "in_reference": 3, "in_candidate": 2}
    ]
    said = out.means()
    assert "Several alignments of exactly this length exist" in said
    assert "WHICH occurrence is called missing" in said


def test_an_identity_that_repeats_only_on_one_side_is_not_ambiguous():
    """A tool the plan never names produces `extra` rows however often it
    repeats, and there is no question about which planned step went
    unmatched — reporting it would be noise dressed as a caveat."""
    out = tj.align(
        reference=_plan("search"),
        candidate=_run("search", "lint", "lint", "lint"),
    )
    assert out.n_extra == 3
    assert out.repeated_identities == []


# ------------------------------------------------------------- empty inputs


def test_an_empty_plan_is_refused_because_it_makes_every_step_extra():
    """ "12 unplanned steps" reads as a finding about the run and is a fact
    about the plan."""
    with pytest.raises(tj.TrajectoryError) as caught:
        tj.align(reference=[], candidate=_run("search", "read"))
    said = str(caught.value)
    assert "no reference trajectory" in said
    assert "fact about the plan" in said


def test_an_empty_run_is_reported_rather_than_refused_and_says_what_it_cannot_tell():
    """Every planned step missing is a real finding. That a run which did
    nothing and a run that was never recorded are the same document here is a
    real limit, and it ships beside the finding."""
    out = tj.align(reference=_plan("search", "read"), candidate=[])
    assert out.n_missing == 2
    assert out.n_matched == out.n_extra == 0
    assert "never recorded are the same document" in out.means()


def test_both_empty_refuses_on_the_plan_rather_than_reporting_a_clean_run():
    """Zero missing and zero extra is what a perfect run looks like."""
    with pytest.raises(tj.TrajectoryError, match="no reference trajectory"):
        tj.align(reference=[], candidate=[])


# ------------------------------------------------------------------ bounds


def test_a_ten_thousand_step_pair_is_refused_rather_than_aligned_on_a_prefix():
    """A prefix comparison reports every step past the cut as missing, which
    is the exact wrong conclusion this measurement exists to get right."""
    with pytest.raises(tj.TrajectoryError) as caught:
        tj.align(
            reference=[_step(f"p{i}") for i in range(10_000)],
            candidate=_run(*[f"p{i}" for i in range(10_000)]),
        )
    said = str(caught.value)
    assert "past the cut as missing" in said
    assert "subtree" in said


def test_a_one_step_plan_against_a_huge_run_is_refused_too():
    """The cell budget alone does not catch this: 1 x 250,000 fits the table
    and does not fit anybody's screen or this process's memory."""
    with pytest.raises(tj.TrajectoryError) as caught:
        tj.align(
            reference=_plan("search"),
            candidate=[_step("x", seq=i) for i in range(tj.MAX_STEPS_PER_SIDE + 1)],
        )
    assert "a side" in str(caught.value)


def test_the_cost_can_be_priced_before_it_is_paid():
    """A caller should be able to shorten the span itself rather than
    discovering the cap by hitting it."""
    small = tj.plan_comparison(20, 40)
    assert small["fits"] is True
    assert small["cells"] == 800
    big = tj.plan_comparison(10_000, 10_000)
    assert big["fits"] is False
    assert big["cells"] == 100_000_000
    assert "past the" in big["means"]


def test_plan_comparison_guards_bool_because_isinstance_true_int_is_true():
    """`plan_comparison(True, True)` would otherwise report a tidy one-cell
    comparison for a call that passed no lengths at all."""
    with pytest.raises(tj.TrajectoryError, match="whole number of steps"):
        tj.plan_comparison(True, 4)
    with pytest.raises(tj.TrajectoryError, match="whole number of steps"):
        tj.plan_comparison(4, -1)


def test_a_capped_listing_still_counts_what_it_is_not_showing():
    """A truncated listing that reads as the whole alignment is how you debug
    the wrong step for an hour. The counts are computed before the cut, and
    both causes of a dropped row are counted apart."""
    reference = [_step(f"p{i}") for i in range(2_400)]
    candidate = [_step(f"p{i}", seq=i) for i in range(100)]
    out = tj.align(reference=reference, candidate=candidate)

    assert (out.n_matched, out.n_missing) == (100, 2_300), (
        "the counts must be over the whole alignment, not over the listing"
    )
    assert len(out.rows) == tj.MAX_ROWS_LISTED
    assert out.matched_rows_not_listed == 100
    assert out.rows_not_listed == 2_400 - tj.MAX_ROWS_LISTED
    assert "not listed" in out.means()
    assert "over the whole alignment rather than over what is listed" in out.means()


def test_the_capped_listing_keeps_the_differences_and_drops_the_matches():
    """A reader opening this is looking for what differs, and a matched row is
    the one whose absence from the listing costs least."""
    reference = [_step(f"p{i}") for i in range(2_050)]
    candidate = [_step(f"p{i}", seq=i) for i in range(100)]
    out = tj.align(reference=reference, candidate=candidate)

    assert out.n_matched == 100 and out.n_missing == 1_950
    assert len(out.rows) == 1_950, "every difference survives the cap"
    assert all(r.status == tj.MISSING for r in out.rows)
    assert out.matched_rows_not_listed == 100
    assert out.rows_not_listed == 100


def test_the_largest_alignment_this_allows_actually_runs():
    """A cap nobody ever exercised is a cap nobody knows the cost of. 500 x 500
    is `MAX_ALIGN_CELLS`, and this is the measurement behind the number in the
    module docstring."""
    names = [f"step{i % 97}" for i in range(500)]
    out = tj.align(
        reference=[_step(n) for n in names],
        candidate=[_step(n, seq=i) for i, n in enumerate(names)],
    )
    assert out.cells == tj.MAX_ALIGN_CELLS
    assert out.n_matched == 500


# --------------------------------------------------------------- arguments


def test_a_changed_argument_at_a_matched_position_is_reported():
    out = tj.align(
        reference=[_step("search", args={"query": "cats", "limit": 10})],
        candidate=[_step("search", args={"query": "dogs", "limit": 10})],
    )
    row = out.rows[0]
    assert row.status == tj.MATCHED
    assert out.n_changed_arguments == 1
    assert [(d.key, d.status) for d in row.argument_diffs] == [("query", tj.CHANGED)]
    assert (row.argument_diffs[0].reference, row.argument_diffs[0].candidate) == (
        "cats",
        "dogs",
    )


def test_two_different_secrets_are_not_reported_as_one_unchanged_argument():
    """The defect that decides the order of operations. Redact first and both
    keys become `[redacted:api-key]`, compare equal, and the step reports its
    arguments unchanged — a false all-clear on the field most likely to be the
    thing that broke."""
    old = "sk-" + "a" * 24
    new = "sk-" + "b" * 24
    out = tj.align(
        reference=[_step("call", args={"token": old})],
        candidate=[_step("call", args={"token": new})],
    )
    diff = out.rows[0].argument_diffs[0]
    assert diff.status == tj.CHANGED, "equality must be decided on the raw value"
    assert diff.redacted is True
    assert diff.reference == diff.candidate == "[redacted:api-key]"
    assert "still compared as different" in out.means()


def test_no_credential_leaves_this_module_in_the_clear():
    """Argument values travel into a panel, an `.mri` and eventually a GitHub
    issue."""
    secret = "sk-" + "z" * 30
    out = tj.align(
        reference=[_step("call", args={"token": "placeholder"})],
        candidate=[_step("call", args={"token": secret}, payload=secret)],
    )
    assert secret not in json.dumps(out.to_dict())
    assert out.redactions == [{"label": "api-key", "count": 1}]


def test_a_credential_in_a_step_name_is_scrubbed_but_still_aligns_on_the_raw_one():
    """Two steps whose names differ only by a secret must not be merged into
    one by the scrubbing that makes them printable."""
    a = "fetch-sk-" + "a" * 20
    b = "fetch-sk-" + "b" * 20
    out = tj.align(reference=[_step(a), _step(b)], candidate=[_step(a, seq=0)])
    assert out.n_matched == 1 and out.n_missing == 1
    text = json.dumps(out.to_dict())
    assert a not in text and b not in text
    assert "[redacted:api-key]" in text


def test_a_plan_that_names_no_arguments_does_not_report_them_all_as_added():
    """A plan that names a step and not its arguments is the ordinary case.
    Reporting five `only_in_candidate` diffs for it would bury the one real
    difference in a run under noise nobody asked for."""
    out = tj.align(
        reference=_plan("search"),
        candidate=[_step("search", args={"query": "cats", "limit": 10})],
    )
    row = out.rows[0]
    assert row.argument_diffs == []
    assert row.arguments_compared is False
    assert "not reported as additions" in row.arguments_note
    assert out.n_changed_arguments == 0


def test_arguments_that_could_not_be_compared_are_not_reported_as_agreement():
    """ "0 changed arguments" over five pairs that were never compared is not
    the same finding as over five that were."""
    out = tj.align(
        reference=_plan("search", "read"),
        candidate=_run("search", "read"),
    )
    assert out.n_changed_arguments == 0
    assert out.n_arguments_not_compared == 2
    assert all(r.arguments_compared is False for r in out.rows)
    assert "not evidence that their arguments agreed" in out.means()


def test_a_key_order_change_in_a_nested_object_is_not_a_difference():
    """JSON objects have no order, and a diff that thinks they do reports a
    difference nobody made."""
    out = tj.align(
        reference=[_step("post", args={"body": {"a": 1, "b": 2}})],
        candidate=[_step("post", args={"body": {"b": 2, "a": 1}})],
    )
    assert out.rows[0].argument_diffs == []
    assert out.rows[0].arguments_compared is True


def test_a_long_value_is_compared_whole_and_only_shown_cut():
    """Comparing the truncated forms would call two payloads that diverge at
    character 1,500 identical."""
    shared = "x" * 1_500
    out = tj.align(
        reference=[_step("post", args={"body": shared + "ONE"})],
        candidate=[_step("post", args={"body": shared + "TWO"})],
    )
    diff = out.rows[0].argument_diffs[0]
    assert diff.status == tj.CHANGED, "the cut must not decide equality"
    assert diff.truncated is True
    assert "characters not shown" in diff.reference
    assert out.arg_values_truncated == 2
    assert "compared whole" in out.means()


def test_argument_keys_past_the_cap_are_counted_rather_than_dropped_silently():
    """Neither changed nor unchanged — a third answer, and it has to be
    visible."""
    many = {f"k{i}": i for i in range(tj.MAX_ARG_KEYS + 5)}
    out = tj.align(
        reference=[_step("call", args=dict(many, k0="one"))],
        candidate=[_step("call", args=dict(many, k0="two"))],
    )
    assert out.arg_keys_dropped == 5
    assert "neither reported as changed nor as unchanged" in out.means()


def test_a_json_input_string_is_read_as_arguments_rather_than_as_one_blob():
    """A recorder writes tool arguments into `input` as JSON, and comparing
    that as free text reports one changed payload where two of six arguments
    moved."""
    out = tj.align(
        reference=[_step("search", payload='{"query": "cats", "limit": 10}')],
        candidate=[_step("search", payload='{"query": "dogs", "limit": 10}')],
    )
    assert [d.key for d in out.rows[0].argument_diffs] == ["query"]


def test_free_text_input_is_compared_as_one_argument_rather_than_dropped():
    """A changed prompt is a real difference. Dropping it because it is not
    keyed would report the step as identical."""
    out = tj.align(
        reference=[_step("ask", kind="llm_call", payload="who is the king")],
        candidate=[_step("ask", kind="llm_call", payload="who is the queen")],
    )
    assert [d.key for d in out.rows[0].argument_diffs] == ["input"]


def test_an_argument_present_on_one_side_only_is_neither_added_nor_removed():
    """`only_in_candidate` rather than "added": calling it an addition is a
    judgement about whether the plan was complete."""
    out = tj.align(
        reference=[_step("search", args={"query": "cats"})],
        candidate=[_step("search", args={"query": "cats", "cursor": "p2"})],
    )
    diff = out.rows[0].argument_diffs[0]
    assert (diff.key, diff.status) == ("cursor", tj.ONLY_IN_CANDIDATE)
    assert diff.reference is None, "absent is not the empty string"
    assert diff.candidate == "p2"


# -------------------------------------------------------------------- kinds


def test_a_plan_of_bare_names_aligns_on_name_and_says_it_dropped_kind():
    """A plan written as tool names against a run recording `tool_call` would
    otherwise match nothing, and every step would report as both missing and
    extra."""
    out = tj.align(reference=["search", "write"], candidate=_run("search", "write"))
    assert out.n_matched == 2
    assert out.matched_on == "name alone"
    assert "Kinds were dropped" in out.means()
    assert "one step here" in out.means()


def test_a_fully_kinded_plan_keeps_kind_in_the_key():
    """An MCP call and a direct call of the same name are different steps when
    both sides say so."""
    out = tj.align(
        reference=[_step("search", kind="mcp_call")],
        candidate=[_step("search", kind="tool_call", seq=0)],
    )
    assert out.matched_on == "kind and name"
    assert out.n_matched == 0
    assert out.n_missing == 1 and out.n_extra == 1


def test_an_unknown_kind_is_refused_by_name_rather_than_matching_nothing():
    """A kind no recorder writes matches nothing, so every step naming it
    reports as missing — which looks like a finding about the run."""
    with pytest.raises(tj.TrajectoryError) as caught:
        tj.align(
            reference=[{"kind": "tool", "name": "search"}], candidate=_run("search")
        )
    assert "is one of" in str(caught.value)
    assert "looks like a finding about the run" in str(caught.value)


def test_a_kind_prefix_in_a_bare_name_is_refused_rather_than_split_by_guess():
    """`slack:post` is either a kind and a name or a tool whose name has a
    colon in it, and the two readings align differently."""
    with pytest.raises(tj.TrajectoryError) as caught:
        tj.align(reference=["slack:post"], candidate=_run("slack:post"))
    assert "not a step kind" in str(caught.value)
    assert '{"name":' in str(caught.value)


def test_an_explicit_kind_prefix_is_honoured():
    out = tj.align(reference=["tool_call:search"], candidate=_run("search"))
    assert out.matched_on == "kind and name"
    assert out.n_matched == 1


# ----------------------------------------------------------------- ordering


def test_steps_are_ordered_by_seq_so_a_shuffled_document_is_not_a_reordering():
    """An ingested trace need not arrive in order, and comparing it as it
    arrived invents a reordering that never happened."""
    run = _run("search", "read", "write")
    out = tj.align(
        reference=_plan("search", "read", "write"),
        candidate=[run[2], run[0], run[1]],
    )
    assert out.n_matched == 3
    assert out.n_reordered == 0


def test_a_boolean_seq_does_not_move_a_step():
    """`isinstance(True, int)` is True, so `"seq": true` would sort as
    position 1 and silently move the step ahead of position 2."""
    candidate = [
        {"kind": "tool_call", "name": "search", "seq": 0},
        {"kind": "tool_call", "name": "read", "seq": True},
        {"kind": "tool_call", "name": "write", "seq": 2},
    ]
    out = tj.align(reference=_plan("search", "read", "write"), candidate=candidate)
    assert out.n_matched == 3, "an unreadable seq must not reorder the document"
    assert out.n_reordered == 0


def test_a_plan_keeps_the_order_it_was_written_in():
    """A written plan has no `seq`, and sorting it by a missing field would
    make its order arbitrary."""
    out = tj.align(
        reference=_plan("c", "a", "b"),
        candidate=_run("c", "a", "b"),
    )
    assert [r.name for r in out.rows] == ["c", "a", "b"]


# --------------------------------------------------------------- read_plan


def test_a_plan_document_names_the_field_it_could_not_find():
    with pytest.raises(tj.TrajectoryError, match="no 'steps', 'plan' or"):
        tj.read_plan({"expected": ["search"]})


def test_a_plan_that_is_not_json_is_refused_as_such():
    with pytest.raises(tj.TrajectoryError, match="not readable JSON"):
        tj.read_plan("search, read, write")


def test_a_nameless_plan_step_is_refused_because_it_would_match_every_other():
    with pytest.raises(tj.TrajectoryError, match="match every other nameless"):
        tj.read_plan([{"kind": "tool_call"}])


def test_a_malformed_step_is_caught_when_the_plan_is_read_not_mid_comparison():
    """Half an alignment already on screen is a worse place to discover a
    broken plan than the moment it was loaded."""
    with pytest.raises(tj.TrajectoryError, match="neither an object nor a name"):
        tj.read_plan([{"name": "search"}, 7])


def test_a_plan_document_round_trips_into_an_alignment():
    """`read_plan` returns the raw entries and `align` normalises both sides
    through one function — a second normaliser here is the drift this project
    has already had to rescue a module from."""
    plan = tj.read_plan('{"steps": ["search", {"name": "write"}]}')
    out = tj.align(reference=plan, candidate=_run("search", "write"))
    assert out.n_matched == 2


# ------------------------------------------------------------------ shapes


def test_every_row_carries_its_own_sentence():
    out = tj.align(
        reference=_plan("search", "read"), candidate=_run("search", "lint")
    ).to_dict()
    assert all(row["means"] for row in out["rows"])


def test_an_index_of_zero_is_a_position_and_not_an_absence():
    """`None` and 0 are different answers, and position 0 is a real
    position."""
    out = tj.align(reference=_plan("search"), candidate=_run("lint"))
    missing = next(r for r in out.rows if r.status == tj.MISSING)
    extra = next(r for r in out.rows if r.status == tj.EXTRA)
    assert missing.reference_index == 0 and missing.candidate_index is None
    assert extra.candidate_index == 0 and extra.reference_index is None


def test_an_entry_that_is_not_a_step_is_counted_rather_than_dropped():
    """MEASURED: `{"reference":[{"name":"x"}],"candidate":[null]}` answered
    "the plan names x at position 0 and no step in the run aligned with it" —
    a finding about the run, produced by discarding the run.

    `_ordered` dropped nulls with a bare comprehension and told nobody, in a
    module that counts every other thing it leaves out.
    """
    out = tj.align(reference=[{"name": "x"}], candidate=[None]).to_dict()
    assert out["n_candidate"] == 0
    assert out["n_candidate_unusable"] == 1
    assert "could not be read as a step" in out["means"]

    mixed = tj.align(
        reference=[{"name": "x"}],
        candidate=[None, 42, ["a"], {"name": "x"}],
    ).to_dict()
    assert mixed["n_candidate"] == 1
    assert mixed["n_candidate_unusable"] == 3


def test_a_plan_of_bare_names_is_still_a_plan():
    """The regression the first version of the fix introduced, and the reason
    it is pinned here.

    `_normalise` takes a mapping OR a bare string — `["search", "write"]` is a
    valid plan and the module is explicitly written for it. Filtering the
    "unusable" entries to dicts alone discarded every bare name and turned a
    written plan into an empty one, which `align` then refused outright.
    "Unusable" has to mean "the next function cannot read this", not "it is
    not the shape I had in mind".
    """
    run = [{"kind": "tool_call", "name": "search"}]
    out = tj.align(reference=["search"], candidate=run).to_dict()

    assert out["n_reference"] == 1
    assert out["n_reference_unusable"] == 0
    assert [r["status"] for r in out["rows"]] == ["matched"]
