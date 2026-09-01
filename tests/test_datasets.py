"""Datasets, experiments, and the arithmetic of comparing two runs.

The failure this file is mostly about is a comparison that LOOKS complete and
is not: a case dropped from the denominator, a missing side read as no change,
a metric whose direction was guessed, or a difference in the last binary digit
reported as an improvement. Each of those produces a plausible table that says
something other than what the reader thinks.

Every test is named for the wrong conclusion it prevents.

Nothing here needs torch, a model or a GPU — `datasets` holds no model and its
only I/O is four readers and writers over plain JSONL.
"""

from __future__ import annotations

import json

import pytest

from modelmri import datasets as ds
from modelmri.errors import BadRequest, Refusal


def _dataset(n=3, *, name="cases", references=None):
    return ds.Dataset(
        name=name,
        cases=[
            ds.Case(
                case_id=f"c{i}",
                input_text=f"prompt {i}",
                reference=(references[i] if references is not None else None),
            )
            for i in range(n)
        ],
    ).validated()


def _experiment(name, dataset, scores, *, refused=None, internals=None, floors=None):
    """One row per case, with `scores[case_id]` or a refusal sentence."""
    refused = refused or {}
    internals = internals or {}
    rows = []
    for case in dataset.cases:
        if case.case_id in refused:
            rows.append(
                ds.Result(case_id=case.case_id, could_not_measure=refused[case.case_id])
            )
            continue
        rows.append(
            ds.Result(
                case_id=case.case_id,
                output=f"answer for {case.case_id}",
                scores=dict(scores.get(case.case_id, {})),
                internals=dict(internals.get(case.case_id, {})),
            )
        )
    return ds.Experiment(
        name=name,
        dataset_name=dataset.name,
        dataset_fingerprint=dataset.fingerprint(),
        results=rows,
        metric_floors=dict(floors or {}),
    ).validated()


# ------------------------------------------------------------------ the files


def test_a_file_with_no_schema_version_is_refused_rather_than_assumed_current(tmp_path):
    """Guessing the version reads unknown fields as known ones, which is how a
    row silently loses the field that made it interesting."""
    path = tmp_path / "d.jsonl"
    path.write_text('{"kind": "dataset", "name": "x"}\n', encoding="utf-8")
    with pytest.raises(BadRequest, match="no schema version"):
        ds.read_dataset(path)


def test_a_schema_version_of_true_is_not_read_as_version_one(tmp_path):
    """isinstance(True, int) is True. A header saying `true` states nothing."""
    path = tmp_path / "d.jsonl"
    path.write_text(
        '{"modelmri_schema": true, "kind": "dataset", "name": "x"}\n', encoding="utf-8"
    )
    with pytest.raises(BadRequest, match="no schema version"):
        ds.read_dataset(path)


def test_a_file_from_a_newer_modelmri_is_refused_by_version(tmp_path):
    """Reading it under the old rules would ignore whatever the new version
    added, and say nothing about having done so."""
    path = tmp_path / "d.jsonl"
    path.write_text(
        json.dumps({"modelmri_schema": ds.SCHEMA_VERSION + 1, "kind": "dataset"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(BadRequest, match="newer ModelMRI"):
        ds.read_dataset(path)


def test_reading_an_experiment_as_a_dataset_names_what_it_actually_is(tmp_path):
    """A run of results read as a set of cases produces a dataset with no
    inputs, which then compares against nothing."""
    data = _dataset()
    exp = _experiment("run", data, {"c0": {"f": 1.0}})
    path = tmp_path / "e.jsonl"
    ds.write_experiment(exp, path)
    with pytest.raises(BadRequest, match="experiment file"):
        ds.read_dataset(path)


def test_an_empty_file_is_refused_rather_than_read_as_a_set_that_all_passed(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(BadRequest, match="is empty"):
        ds.read_dataset(path)


def test_an_unreadable_row_says_which_line_rather_than_being_skipped(tmp_path):
    """A file with an unreadable row is not a file with one fewer row — that
    row was a case somebody meant to measure."""
    data = _dataset()
    path = tmp_path / "d.jsonl"
    ds.write_dataset(data, path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    with pytest.raises(BadRequest, match="line 5"):
        ds.read_dataset(path)


def test_a_truncated_file_says_how_many_rows_are_missing(tmp_path):
    """A run killed at case 2 of 3 leaves a file that reads perfectly as a
    complete 2-case run, and every percentage from it is right about a
    denominator nobody chose."""
    data = _dataset(3)
    path = tmp_path / "d.jsonl"
    ds.write_dataset(data, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    back = ds.read_dataset(path)
    assert len(back.cases) == 2
    assert "1 are missing" in back.truncated
    assert "not zero" in back.truncated


def test_a_dataset_edited_after_it_was_written_says_so(tmp_path):
    """Its header still claims the fingerprint every earlier experiment ran
    against, and that claim is now false."""
    data = _dataset(2)
    path = tmp_path / "d.jsonl"
    ds.write_dataset(data, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    row["input_text"] = "somebody changed this"
    lines[1] = json.dumps(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    back = ds.read_dataset(path)
    assert "edited after it was written" in back.edited


def test_a_dataset_round_trips_including_the_absence_of_a_reference(tmp_path):
    """`None` must survive as `None`. Coercing it to "" claims the expected
    output IS the empty string, which is a different and stranger dataset."""
    data = _dataset(2, references=["gold", None])
    path = tmp_path / "d.jsonl"
    ds.write_dataset(data, path)
    back = ds.read_dataset(path)
    assert back.cases[0].reference == "gold"
    assert back.cases[1].reference is None
    assert back.n_references == 1
    assert back.fingerprint() == data.fingerprint()
    assert back.truncated == "" and back.edited == ""


def test_an_experiment_round_trips_its_refusal_sentences(tmp_path):
    """A refused row that comes back measured is a run that reports a success
    it never had."""
    data = _dataset(2)
    exp = _experiment(
        "run", data, {"c0": {"f": 0.5}}, refused={"c1": "the SAE was not loaded"}
    )
    path = tmp_path / "e.jsonl"
    ds.write_experiment(exp, path)
    back = ds.read_experiment(path)
    assert len(back.results) == 2
    assert back.results[1].measured is False
    assert back.results[1].could_not_measure == "the SAE was not loaded"
    assert back.n_measured == 1


def test_an_empty_output_and_no_output_survive_as_different_things(tmp_path):
    """A run that emitted nothing and a run that emitted the empty string are
    different runs."""
    data = _dataset(2)
    exp = ds.Experiment(
        name="run",
        dataset_name=data.name,
        dataset_fingerprint=data.fingerprint(),
        results=[
            ds.Result(case_id="c0", output="", scores={"f": 1.0}),
            ds.Result(case_id="c1", output=None, scores={"f": 1.0}),
        ],
    )
    path = tmp_path / "e.jsonl"
    ds.write_experiment(exp, path)
    back = ds.read_experiment(path)
    assert back.results[0].output == ""
    assert back.results[1].output is None


def test_duplicate_case_ids_are_refused_because_the_join_cannot_tell_them_apart():
    """The denominator would count both and the comparison one."""
    data = ds.Dataset(
        name="x",
        cases=[ds.Case("same", "a"), ds.Case("same", "b")],
    )
    with pytest.raises(BadRequest, match="positions 0 and 1"):
        data.validated()


def test_two_results_for_one_case_are_refused_rather_than_last_wins():
    with pytest.raises(BadRequest, match="whichever it read last"):
        ds.Experiment(
            name="r",
            dataset_name="x",
            results=[ds.Result(case_id="c0"), ds.Result(case_id="c0")],
        ).validated()


def test_from_inputs_refuses_a_repeated_input_naming_both_positions():
    """Two identical inputs hash to one id: keeping both puts two cases in the
    denominator and one in the comparison."""
    with pytest.raises(BadRequest, match="inputs 0 and 2 are identical"):
        ds.from_inputs("x", ["a", "b", "a"])


def test_from_inputs_refuses_a_reference_list_of_the_wrong_length():
    """Paired by position, so a mismatch scores at least one case against
    another case's answer."""
    with pytest.raises(BadRequest, match="scored against another case"):
        ds.from_inputs("x", ["a", "b"], references=["gold"])


def test_a_non_finite_score_is_refused_at_write_time_naming_the_case(tmp_path):
    """`NaN` is not JSON, and a NaN in a scores dict sorts as a number."""
    exp = ds.Experiment(
        name="r",
        dataset_name="x",
        dataset_fingerprint="f",
        results=[ds.Result(case_id="c0", scores={"f": float("nan")})],
    )
    with pytest.raises(BadRequest, match="case c0"):
        ds.write_experiment(exp, tmp_path / "e.jsonl")


def test_results_stream_one_at_a_time_without_reading_the_whole_file(tmp_path):
    data = _dataset(4)
    exp = _experiment("run", data, {c.case_id: {"f": 1.0} for c in data.cases})
    path = tmp_path / "e.jsonl"
    ds.write_experiment(exp, path)
    streamed = list(ds.stream_results(path))
    assert [r.case_id for r in streamed] == ["c0", "c1", "c2", "c3"]


# ------------------------------------------------------------- the identity


def test_reordering_a_dataset_does_not_change_which_dataset_it_is():
    """Otherwise every experiment becomes incomparable the moment somebody
    sorts the file."""
    a = _dataset(3)
    b = ds.Dataset(name="cases", cases=list(reversed(a.cases))).validated()
    assert a.fingerprint() == b.fingerprint()


def test_editing_a_reference_output_makes_it_a_different_dataset():
    """A score computed against one answer key is not comparable to a score
    computed against another, however identical the ids look."""
    a = _dataset(2, references=["gold", "gold"])
    b = _dataset(2, references=["gold", "silver"])
    assert a.fingerprint() != b.fingerprint()


def test_a_missing_reference_and_an_empty_one_are_different_datasets():
    """`None` means nobody wrote an expected output; `""` means the expected
    output is the empty string."""
    a = _dataset(1, references=[None])
    b = _dataset(1, references=[""])
    assert a.fingerprint() != b.fingerprint()


def test_a_dataset_with_no_cases_is_refused_not_treated_as_all_passing():
    with pytest.raises(BadRequest, match="nothing here to run"):
        ds.Dataset(name="x", cases=[]).validated()


def test_a_dataset_with_no_references_says_so_rather_than_reporting_zero_of_zero():
    said = _dataset(3).means()
    assert "No case here carries an expected output" in said
    assert "compares two runs against each other" in said


# --------------------------------------------------------- refusing to compare


def test_two_experiments_over_different_datasets_are_refused_naming_the_cases():
    """Comparing row 12 of one set against row 12 of another is a table of
    real numbers about nothing."""
    a = _dataset(2, name="a")
    b = ds.Dataset(
        name="b", cases=[ds.Case("c0", "prompt 0"), ds.Case("z9", "different")]
    ).validated()
    before = _experiment("before", a, {"c0": {"f": 1.0}, "c1": {"f": 1.0}})
    after = _experiment("after", b, {"c0": {"f": 1.0}, "z9": {"f": 1.0}})
    with pytest.raises(ds.DifferentDatasets) as caught:
        ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    said = str(caught.value)
    assert "c1" in said and "z9" in said
    assert "two different sets" in said


def test_the_same_ids_over_an_edited_input_is_still_refused_and_says_the_ids_match():
    """The most dangerous case: everything joins perfectly and the numbers
    describe two different questions."""
    a = _dataset(2)
    b = ds.Dataset(
        name="cases",
        cases=[ds.Case("c0", "prompt 0"), ds.Case("c1", "SOMEBODY EDITED THIS")],
    ).validated()
    before = _experiment("before", a, {"c0": {"f": 1.0}, "c1": {"f": 1.0}})
    after = _experiment("after", b, {"c0": {"f": 1.0}, "c1": {"f": 1.0}})
    with pytest.raises(ds.DifferentDatasets) as caught:
        ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    said = str(caught.value)
    assert "same 2 case ids" in said
    assert "edited between the two runs" in said


def test_an_experiment_that_recorded_no_dataset_cannot_be_compared():
    """Nothing can tell whether the two runs covered the same cases, and a
    comparison that cannot tell must not pretend."""
    data = _dataset(2)
    before = _experiment("before", data, {"c0": {"f": 1.0}, "c1": {"f": 1.0}})
    after = _experiment("after", data, {"c0": {"f": 1.0}, "c1": {"f": 1.0}})
    after.dataset_fingerprint = ""
    with pytest.raises(ds.DifferentDatasets, match="did not record which dataset"):
        ds.compare_experiments(before, after, metric="f", higher_is_better=True)


def test_a_supplied_dataset_that_is_not_the_one_that_was_run_is_refused():
    """Resolving references from it would attach the wrong expected output to
    every case."""
    data = _dataset(2)
    other = _dataset(2, references=["gold", "gold"])
    before = _experiment("before", data, {"c0": {"f": 1.0}, "c1": {"f": 1.0}})
    after = _experiment("after", data, {"c0": {"f": 1.0}, "c1": {"f": 1.0}})
    with pytest.raises(ds.DifferentDatasets, match="not the one these runs"):
        ds.compare_experiments(
            before, after, metric="f", higher_is_better=True, dataset=other
        )


def test_a_metric_nobody_recorded_is_refused_naming_the_metrics_present():
    """Reporting forty `unmeasurable` rows for a typo buries the typo."""
    data = _dataset(2)
    before = _experiment("before", data, {"c0": {"faith": 1.0}, "c1": {"faith": 1.0}})
    after = _experiment("after", data, {"c0": {"faith": 1.0}, "c1": {"faith": 1.0}})
    with pytest.raises(Refusal) as caught:
        ds.compare_experiments(before, after, metric="fatih", higher_is_better=True)
    assert "faith" in str(caught.value)


def test_a_negative_floor_is_refused_because_it_would_make_everything_changed():
    data = _dataset(1)
    before = _experiment("before", data, {"c0": {"f": 1.0}})
    after = _experiment("after", data, {"c0": {"f": 1.0}})
    with pytest.raises(BadRequest, match="including the identical ones"):
        ds.compare_experiments(
            before, after, metric="f", higher_is_better=True, floor=-0.1
        )


def test_a_direction_that_is_not_a_boolean_is_refused_rather_than_coerced():
    """It decides the sign of every conclusion in the comparison."""
    data = _dataset(1)
    before = _experiment("before", data, {"c0": {"f": 1.0}})
    after = _experiment("after", data, {"c0": {"f": 2.0}})
    with pytest.raises(BadRequest, match="has to be True or False"):
        ds.compare_experiments(before, after, metric="f", higher_is_better=1)


# ------------------------------------------------------------- the denominator


def test_a_case_missing_from_one_run_is_a_row_not_a_shrunken_denominator():
    """An intersection would report "2 unchanged, 0 worse" about a run that
    died after case 2 of 3."""
    data = _dataset(3)
    before = _experiment(
        "before", data, {"c0": {"f": 1.0}, "c1": {"f": 1.0}, "c2": {"f": 1.0}}
    )
    after = _experiment("after", data, {"c0": {"f": 1.0}, "c1": {"f": 1.0}})
    after.results = after.results[:2]
    out = ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    assert out.n_cases == 3
    assert sum(out.counts.values()) == 3
    missing = next(r for r in out.rows if r.case_id == "c2")
    assert missing.status == ds.UNMEASURABLE
    assert "wrote no row for this case" in missing.detail


def test_the_four_counts_always_sum_to_the_number_of_cases():
    data = _dataset(4)
    before = _experiment(
        "before",
        data,
        {"c0": {"f": 1.0}, "c1": {"f": 1.0}, "c2": {"f": 1.0}},
        refused={"c3": "the model refused"},
    )
    after = _experiment(
        "after",
        data,
        {"c0": {"f": 2.0}, "c1": {"f": 0.5}, "c2": {"f": 1.0}},
        refused={"c3": "the model refused"},
    )
    out = ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    assert out.counts == {
        ds.BETTER: 1,
        ds.WORSE: 1,
        ds.UNCHANGED: 1,
        ds.UNMEASURABLE: 1,
    }
    assert sum(out.counts.values()) == out.n_cases == 4


def test_all_four_count_keys_are_present_even_at_zero():
    """`counts.get("worse", 0)` cannot tell an absent key from a real zero."""
    data = _dataset(1)
    before = _experiment("before", data, {"c0": {"f": 1.0}})
    after = _experiment("after", data, {"c0": {"f": 1.0}})
    out = ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    assert set(out.counts) == {ds.BETTER, ds.WORSE, ds.UNCHANGED, ds.UNMEASURABLE}


def test_a_refused_row_carries_its_sentence_into_the_comparison():
    """ "It failed" is not actionable; "the SAE was not loaded" is."""
    data = _dataset(1)
    before = _experiment("before", data, {"c0": {"f": 1.0}})
    after = _experiment(
        "after", data, {}, refused={"c0": "no SAE is loaded for this layer"}
    )
    out = ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    assert out.rows[0].status == ds.UNMEASURABLE
    assert "no SAE is loaded for this layer" in out.rows[0].detail


def test_a_measured_row_with_no_score_for_this_metric_lists_what_it_does_have():
    """Different from a refusal: the case ran, and this particular number was
    never recorded."""
    data = _dataset(1)
    before = _experiment("before", data, {"c0": {"faith": 1.0, "kl": 0.2}})
    after = _experiment("after", data, {"c0": {"kl": 0.3}})
    out = ds.compare_experiments(before, after, metric="faith", higher_is_better=True)
    assert out.rows[0].status == ds.UNMEASURABLE
    assert "recorded no `faith` score" in out.rows[0].detail
    assert "it carries kl" in out.rows[0].detail


def test_a_nan_score_is_unmeasurable_rather_than_ordered():
    data = _dataset(1)
    before = _experiment("before", data, {"c0": {"f": 1.0}})
    after = _experiment("after", data, {"c0": {"f": float("inf")}})
    out = ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    assert out.rows[0].status == ds.UNMEASURABLE
    assert "not a finite number" in out.rows[0].detail


def test_a_boolean_score_is_not_read_as_one_point_zero():
    """isinstance(True, int) is True, so `{"passed": True}` would rank."""
    data = _dataset(1)
    before = _experiment("before", data, {"c0": {"f": 0.0}})
    after = _experiment("after", data, {"c0": {"f": True}})
    out = ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    assert out.rows[0].status == ds.UNMEASURABLE
    assert "is a bool" in out.rows[0].detail


def test_delta_is_none_when_a_side_is_unmeasurable_never_zero():
    """A zero would read as "no change" and would be averaged in as one."""
    data = _dataset(1)
    before = _experiment("before", data, {"c0": {"f": 1.0}})
    after = _experiment("after", data, {}, refused={"c0": "out of memory"})
    out = ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    assert out.rows[0].delta is None
    assert out.rows[0].after is None


# ------------------------------------------------------------------ direction


def test_the_same_numbers_read_opposite_ways_under_opposite_directions():
    """KL is better lower and faithfulness better higher. A module that guessed
    would invert every conclusion in half of all comparisons and look fine."""
    data = _dataset(1)
    before = _experiment("before", data, {"c0": {"m": 1.0}})
    after = _experiment("after", data, {"c0": {"m": 2.0}})
    up = ds.compare_experiments(before, after, metric="m", higher_is_better=True)
    down = ds.compare_experiments(before, after, metric="m", higher_is_better=False)
    assert up.rows[0].status == ds.BETTER
    assert down.rows[0].status == ds.WORSE
    assert up.rows[0].delta == down.rows[0].delta == 1.0


# ---------------------------------------------------------------------- floor


def test_a_difference_under_the_stated_floor_is_unchanged_not_better():
    data = _dataset(1)
    before = _experiment("before", data, {"c0": {"f": 1.0}}, floors={"f": 0.01})
    after = _experiment("after", data, {"c0": {"f": 1.005}}, floors={"f": 0.01})
    out = ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    assert out.rows[0].status == ds.UNCHANGED
    assert out.floor == 0.01


def test_the_coarser_of_the_two_files_floors_is_the_one_used():
    """Two runs cannot be compared more finely than the coarser of them can
    represent."""
    data = _dataset(1)
    before = _experiment("before", data, {"c0": {"f": 1.0}}, floors={"f": 0.001})
    after = _experiment("after", data, {"c0": {"f": 1.02}}, floors={"f": 0.05})
    out = ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    assert out.floor == 0.05
    assert out.rows[0].status == ds.UNCHANGED
    assert "coarser" in out.floor_note


def test_one_file_stating_a_floor_says_the_other_did_not():
    data = _dataset(1)
    before = _experiment("before", data, {"c0": {"f": 1.0}}, floors={"f": 0.1})
    after = _experiment("after", data, {"c0": {"f": 1.05}})
    out = ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    assert out.floor == 0.1
    assert "states none" in out.floor_note


def test_with_no_floor_anywhere_the_comparison_says_it_is_exact():
    """A difference in the last binary digit counting as "better" is a fact
    about float arithmetic, not about the model."""
    data = _dataset(1)
    before = _experiment("before", data, {"c0": {"f": 1.0}})
    after = _experiment("after", data, {"c0": {"f": 1.0 + 1e-15}})
    out = ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    assert out.rows[0].status == ds.BETTER
    assert "EXACT" in out.floor_note
    assert "pass `floor=`" in out.floor_note


def test_a_caller_floor_overrides_what_the_files_state():
    data = _dataset(1)
    before = _experiment("before", data, {"c0": {"f": 1.0}}, floors={"f": 10.0})
    after = _experiment("after", data, {"c0": {"f": 1.5}}, floors={"f": 10.0})
    out = ds.compare_experiments(
        before, after, metric="f", higher_is_better=True, floor=0.1
    )
    assert out.floor == 0.1
    assert out.rows[0].status == ds.BETTER


# --------------------------------------------------------------- distribution


def test_the_distribution_n_is_not_the_denominator_and_says_so():
    """Two different numbers that both look like "how many cases"."""
    data = _dataset(3)
    before = _experiment(
        "before", data, {"c0": {"f": 1.0}, "c1": {"f": 1.0}}, refused={"c2": "no"}
    )
    after = _experiment(
        "after", data, {"c0": {"f": 2.0}, "c1": {"f": 3.0}}, refused={"c2": "no"}
    )
    out = ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    assert out.n_cases == 3
    assert out.delta_distribution["n"] == 2
    assert "NOT the 3 cases" in out.means()


def test_the_distribution_is_none_when_nothing_was_measured_on_both_sides():
    """Not a distribution centred on zero, which is what a dict of zeros would
    look like."""
    data = _dataset(2)
    before = _experiment("before", data, {"c0": {"f": 1.0}}, refused={"c1": "no"})
    after = _experiment("after", data, {}, refused={"c0": "no", "c1": "no"})
    out = ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    assert out.delta_distribution is None
    assert "no delta distribution at all" in out.means()


def test_the_summary_is_a_median_and_a_spread_never_a_mean():
    """A mean over a set hides the one case that carried it."""
    data = _dataset(4)
    before = _experiment("before", data, {c.case_id: {"f": 0.0} for c in data.cases})
    after = _experiment(
        "after",
        data,
        {"c0": {"f": 0.1}, "c1": {"f": 0.1}, "c2": {"f": 0.1}, "c3": {"f": 100.0}},
    )
    out = ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    dist = out.delta_distribution
    assert dist["median"] == pytest.approx(0.1)
    assert dist["hi"] == pytest.approx(100.0)
    assert "mean" not in dist


# ------------------------------------------------------------------ references


def test_without_a_dataset_the_reference_count_is_none_not_zero():
    """ "This comparison did not look" and "the dataset has none" are different
    answers, and only one says something about the data."""
    data = _dataset(2, references=["gold", "gold"])
    before = _experiment("before", data, {"c0": {"f": 1.0}, "c1": {"f": 1.0}})
    after = _experiment("after", data, {"c0": {"f": 1.0}, "c1": {"f": 1.0}})
    out = ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    assert out.references is None
    assert all(r.has_reference is None for r in out.rows)
    assert "unknown rather than zero" in out.means()


def test_with_a_dataset_the_rows_say_which_had_a_reference():
    data = _dataset(2, references=["gold", None])
    before = _experiment("before", data, {"c0": {"f": 1.0}, "c1": {"f": 1.0}})
    after = _experiment("after", data, {"c0": {"f": 1.0}, "c1": {"f": 1.0}})
    out = ds.compare_experiments(
        before, after, metric="f", higher_is_better=True, dataset=data
    )
    assert out.references == 1
    assert [r.has_reference for r in out.rows] == [True, False]


# ------------------------------------------------------------------ internals


def test_a_reordered_ranking_is_changed_rather_than_the_same_set():
    """Same five heads in a different order is a different answer about which
    head carries the case."""
    data = _dataset(1)
    ranking = ["L6H9", "L2H1", "L4H7"]
    before = _experiment(
        "before", data, {"c0": {"f": 1.0}}, internals={"c0": {"top_heads": ranking}}
    )
    after = _experiment(
        "after",
        data,
        {"c0": {"f": 1.0}},
        internals={"c0": {"top_heads": ["L2H1", "L6H9", "L4H7"]}},
    )
    out = ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    finding = out.rows[0].internals[0]
    assert finding.status == ds.CHANGED
    assert "moved rank" in finding.detail


def test_a_head_that_entered_the_top_five_is_named_rather_than_counted():
    data = _dataset(1)
    before = _experiment(
        "before",
        data,
        {"c0": {"f": 1.0}},
        internals={"c0": {"heads": ["a", "b", "c", "d", "e", "z"]}},
    )
    after = _experiment(
        "after",
        data,
        {"c0": {"f": 0.5}},
        internals={"c0": {"heads": ["a", "b", "c", "d", "z", "e"]}},
    )
    out = ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    finding = out.rows[0].internals[0]
    assert finding.measured["entered"] == ["z"]
    assert finding.measured["left"] == ["e"]
    assert "z entered the top 5" in finding.detail
    # And the row's own sentence points at it, because the row is what a
    # reader scanning a regression sees first.
    assert "What the measurement saw also moved" in out.rows[0].detail


def test_a_patching_site_that_flipped_sign_is_named():
    """The claim this module exists to make: not "0.71 -> 0.63" but "L6.resid
    is now pushing the answer the other way"."""
    data = _dataset(1)
    before = _experiment(
        "before",
        data,
        {"c0": {"f": 1.0}},
        internals={"c0": {"sites": {"L6.resid": 0.42, "L2.attn": -0.11}}},
    )
    after = _experiment(
        "after",
        data,
        {"c0": {"f": 0.6}},
        internals={"c0": {"sites": {"L6.resid": -0.31, "L2.attn": -0.10}}},
    )
    out = ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    finding = out.rows[0].internals[0]
    assert finding.status == ds.CHANGED
    assert "L6.resid flipped sign" in finding.detail
    assert list(finding.measured["flipped"]) == ["L6.resid"]
    assert out.internals_changed == [out.rows[0]]


def test_a_site_that_vanished_is_reported_as_gone_not_as_a_zero():
    """A missing block is not a zero — the same rule `mri_diff` keeps."""
    data = _dataset(1)
    before = _experiment(
        "before",
        data,
        {"c0": {"f": 1.0}},
        internals={"c0": {"sites": {"L6.resid": 0.42, "L2.attn": 0.11}}},
    )
    after = _experiment(
        "after",
        data,
        {"c0": {"f": 1.0}},
        internals={"c0": {"sites": {"L6.resid": 0.42}}},
    )
    out = ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    finding = out.rows[0].internals[0]
    assert finding.status == ds.CHANGED
    assert "no longer recorded" in finding.detail
    assert finding.measured["only_in_before"] == ["L2.attn"]


def test_a_magnitude_move_with_no_sign_flip_reports_the_values_and_judges_nothing():
    """There is no floor for an internal, and calling a move significant
    without one would be inventing the threshold."""
    data = _dataset(1)
    before = _experiment(
        "before", data, {"c0": {"f": 1.0}}, internals={"c0": {"s": {"a": 0.1}}}
    )
    after = _experiment(
        "after", data, {"c0": {"f": 1.0}}, internals={"c0": {"s": {"a": 0.9}}}
    )
    out = ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    finding = out.rows[0].internals[0]
    assert finding.status == ds.SAME
    assert finding.measured["before"] == {"a": 0.1}
    assert finding.measured["after"] == {"a": 0.9}
    assert "inventing the threshold" in finding.detail


def test_internals_present_on_one_side_only_are_not_comparable_not_unchanged():
    data = _dataset(1)
    before = _experiment(
        "before", data, {"c0": {"f": 1.0}}, internals={"c0": {"heads": ["a"]}}
    )
    after = _experiment("after", data, {"c0": {"f": 1.0}}, internals={"c0": {}})
    out = ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    finding = out.rows[0].internals[0]
    assert finding.status == ds.NOT_COMPARABLE
    assert "certainly not a zero" in finding.detail


def test_a_mapping_of_booleans_is_not_read_as_signed_sites():
    """`True` would read as +1 and flip against `False`."""
    data = _dataset(1)
    before = _experiment(
        "before", data, {"c0": {"f": 1.0}}, internals={"c0": {"flags": {"a": True}}}
    )
    after = _experiment(
        "after", data, {"c0": {"f": 1.0}}, internals={"c0": {"flags": {"a": False}}}
    )
    out = ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    assert out.rows[0].internals[0].status == ds.NOT_COMPARABLE


def test_internals_are_recognised_by_shape_not_by_key_name():
    """A name map identifies exactly what was written into it and silently
    declines to compare everything added since — which looks like "nothing
    changed"."""
    findings = ds.compare_internals(
        {"a_panel_nobody_has_heard_of": {"site": 1.0}},
        {"a_panel_nobody_has_heard_of": {"site": -1.0}},
    )
    assert findings[0].kind == "sites"
    assert findings[0].status == ds.CHANGED


def test_a_bare_number_internal_is_pointed_at_scores_rather_than_compared():
    """It has no floor here, and `scores` is where a floor can be stated."""
    findings = ds.compare_internals({"depth": 3.0}, {"depth": 4.0})
    assert findings[0].status == ds.NOT_COMPARABLE
    assert "belongs in `scores`" in findings[0].detail


def test_an_unchanged_ranking_says_what_it_did_not_look_at():
    findings = ds.compare_internals(
        {"heads": ["a", "b", "c", "d", "e", "f"]},
        {"heads": ["a", "b", "c", "d", "e", "zzz"]},
    )
    assert findings[0].status == ds.SAME
    assert "Positions past 5 are not judged" in findings[0].detail


# ------------------------------------------------------------------- the prose


def test_the_comparison_reports_no_single_score_and_says_why():
    """Two cases collapsing and three improving slightly average out to fine,
    which is exactly the regression an aggregate hides."""
    data = _dataset(2)
    before = _experiment("before", data, {"c0": {"f": 1.0}, "c1": {"f": 1.0}})
    after = _experiment("after", data, {"c0": {"f": 2.0}, "c1": {"f": 0.1}})
    out = ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    said = out.means()
    assert "NO SINGLE NUMBER HERE ON PURPOSE" in said
    keys = set(out.to_dict())
    assert not keys & {"score", "verdict", "passed", "overall", "grade"}


def test_the_means_sentence_carries_the_denominator_and_the_floor():
    data = _dataset(2)
    before = _experiment("before", data, {"c0": {"f": 1.0}, "c1": {"f": 1.0}})
    after = _experiment("after", data, {"c0": {"f": 2.0}, "c1": {"f": 1.0}})
    said = ds.compare_experiments(
        before, after, metric="f", higher_is_better=True
    ).means()
    assert "2 cases" in said
    assert "always sum to it" in said
    assert "EXACT" in said


def test_a_truncated_run_says_so_in_the_comparison_it_produced(tmp_path):
    """The whole point of noticing truncation is that it reaches the reader of
    the comparison, not just the reader of the file."""
    data = _dataset(3)
    before = _experiment("before", data, {c.case_id: {"f": 1.0} for c in data.cases})
    after = _experiment("after", data, {c.case_id: {"f": 1.0} for c in data.cases})
    path = tmp_path / "after.jsonl"
    ds.write_experiment(after, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    partial = ds.read_experiment(path)
    out = ds.compare_experiments(before, partial, metric="f", higher_is_better=True)
    assert out.n_cases == 3
    assert out.counts[ds.UNMEASURABLE] == 1
    assert "not finished being written" in out.means()


def test_a_row_in_a_run_but_not_in_the_dataset_is_counted_and_flagged():
    """Dropping it would shrink the denominator to make the file tidy."""
    data = _dataset(2)
    before = _experiment("before", data, {"c0": {"f": 1.0}, "c1": {"f": 1.0}})
    after = _experiment("after", data, {"c0": {"f": 1.0}, "c1": {"f": 1.0}})
    stray = ds.Result(case_id="ghost", scores={"f": 1.0})
    before.results.append(stray)
    after.results.append(ds.Result(case_id="ghost", scores={"f": 1.0}))
    out = ds.compare_experiments(
        before, after, metric="f", higher_is_better=True, dataset=data
    )
    assert out.n_cases == 3
    assert "ghost" in " ".join(out.notes)


def test_the_terminal_table_puts_the_worse_rows_first():
    """A comparison is read to find what broke."""
    data = _dataset(3)
    before = _experiment(
        "before", data, {"c0": {"f": 1.0}, "c1": {"f": 1.0}, "c2": {"f": 1.0}}
    )
    after = _experiment(
        "after", data, {"c0": {"f": 2.0}, "c1": {"f": 0.1}, "c2": {"f": 1.0}}
    )
    out = ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    text = ds.render(out)
    assert text.index("c1") < text.index("c0")
    assert "1 better · 1 worse · 1 unchanged" in text


def test_the_terminal_table_prints_why_a_row_could_not_be_measured():
    """ "unmeasurable" with no reason is the same non-answer as a gap, and the
    reason is already written."""
    data = _dataset(1)
    before = _experiment("before", data, {"c0": {"f": 1.0}})
    after = _experiment("after", data, {}, refused={"c0": "the GPU ran out of memory"})
    text = ds.render(
        ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    )
    assert "the GPU ran out of memory" in text


def test_a_case_id_too_long_for_the_column_is_cut_visibly_and_counted():
    """A silently shortened id is a row a reader cannot look up."""
    long_id = "an-authored-case-id-that-is-far-too-long-to-fit-in-a-column"
    data = ds.Dataset(name="x", cases=[ds.Case(long_id, "prompt")]).validated()
    before = _experiment("before", data, {long_id: {"f": 1.0}})
    after = _experiment("after", data, {long_id: {"f": 2.0}})
    text = ds.render(
        ds.compare_experiments(before, after, metric="f", higher_is_better=True)
    )
    assert "…" in text
    assert "1 case id(s) were shortened" in text
    assert "full ids are in the comparison" in text


def test_rows_past_the_terminal_limit_are_counted_rather_than_dropped_quietly():
    """A table that stops at twelve rows and says nothing reads as a
    twelve-case comparison."""
    data = _dataset(20)
    scores = {c.case_id: {"f": 1.0} for c in data.cases}
    before = _experiment("before", data, scores)
    after = _experiment("after", data, scores)
    text = ds.render(
        ds.compare_experiments(before, after, metric="f", higher_is_better=True),
        limit=5,
    )
    assert "15 more rows" in text
    assert "carries all 20" in text


# ------------------------------------------ what ONE run recorded, counted


def test_a_boolean_score_is_summarised_by_its_values_and_never_by_a_median():
    """`bool` is an `int`, so a metric of True/False would take a median and
    print "0.5" — a number for a thing this project refuses to rank. Every
    other reader here already refuses a boolean; this one has to as well."""
    data = _dataset(2)
    exp = _experiment("run", data, {"c0": {"passed": True}, "c1": {"passed": False}})
    summary = ds.score_summary(exp)
    metric = summary["metrics"][0]
    assert metric["numbers"] is False
    assert metric["median"] is None, "no median, not a median of zero"
    assert metric["values"] == {"false": 1, "true": 1}
    assert "none can be ordered against another run" in summary["means"]


def test_a_metric_only_some_rows_carry_says_how_many_do_not():
    """A metric on 1 of 3 rows read as a metric on 3 is how a scorer that
    crashed two thirds of the way through looks like a complete one."""
    data = _dataset(3)
    exp = _experiment(
        "run",
        data,
        {"c0": {"f": 1.0, "rubric": 7.0}, "c1": {"f": 2.0}, "c2": {"f": 3.0}},
    )
    summary = ds.score_summary(exp)
    by_name = {m["metric"]: m for m in summary["metrics"]}
    assert by_name["rubric"]["n"] == 1
    assert by_name["rubric"]["n_missing"] == 2
    assert by_name["f"]["n_missing"] == 0
    assert "2 row(s) do not carry it" in ds.render_scores(exp)


def test_a_run_with_more_metrics_than_the_table_shows_says_how_many_are_hidden():
    """The same rule the comparison table keeps: a list that stops and says
    nothing reads as the whole list."""
    data = _dataset(1)
    exp = _experiment("run", data, {"c0": {f"m{i}": float(i) for i in range(9)}})
    text = ds.render_scores(exp, limit=4)
    assert "5 more metric(s), not shown" in text
    assert "summary itself carries all of them" in text


def test_a_run_that_scored_nothing_is_not_rendered_as_a_run_that_scored_zero():
    data = _dataset(2)
    exp = _experiment("run", data, {}, refused={"c0": "the SAE was not loaded"})
    text = ds.render_scores(exp)
    assert "no metric was recorded" in text
    assert "the SAE was not loaded" in text


# ------------------------------------- times the file does not get to invent


def test_an_experiment_with_no_start_time_is_not_dated_to_the_write(tmp_path):
    """`started_at or _now()` dated an imported eval to the moment somebody
    read it, which is a value that was not recorded sitting in the field a
    reader trusts for "when did this happen". An unknown stays unknown."""
    data = _dataset(1)
    exp = _experiment("run", data, {"c0": {"f": 1.0}})
    ds.write_experiment(exp, tmp_path / "e.jsonl")
    assert ds.read_experiment(tmp_path / "e.jsonl").started_at == ""


def test_a_dataset_with_no_creation_time_is_not_dated_to_the_write(tmp_path):
    data = _dataset(1)
    assert data.created_at == ""
    ds.write_dataset(data, tmp_path / "d.jsonl")
    assert ds.read_dataset(tmp_path / "d.jsonl").created_at == ""


def test_a_start_time_that_really_was_recorded_survives_the_file(tmp_path):
    """The other half: dropping the invented default must not drop a real
    one."""
    data = _dataset(1)
    exp = _experiment("run", data, {"c0": {"f": 1.0}})
    exp.started_at = "2026-03-04T09:00:00+00:00"
    ds.write_experiment(exp, tmp_path / "e.jsonl")
    assert ds.read_experiment(tmp_path / "e.jsonl").started_at == (
        "2026-03-04T09:00:00+00:00"
    )


def test_a_gap_the_reader_left_behind_survives_the_file_it_is_written_into(tmp_path):
    """`truncated` is set by every reader and was written by no writer, so a
    run somebody had already been told was partial came back looking whole:
    the header's `n_results` and the rows under it agree, and the recomputed
    sentence is empty."""
    data = _dataset(2)
    exp = _experiment("run", data, {"c0": {"f": 1.0}, "c1": {"f": 2.0}})
    exp.truncated = "only the first 2 of 40 cases were read."
    ds.write_experiment(exp, tmp_path / "e.jsonl")
    assert "first 2 of 40" in ds.read_experiment(tmp_path / "e.jsonl").truncated
