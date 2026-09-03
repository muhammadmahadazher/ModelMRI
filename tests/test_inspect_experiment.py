# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Inspect's per-sample scores, carried the last step into an `Experiment`.

`inspect_io._scores` has always parsed the scores out of an `.eval` log, and
the import path has always dropped them one call before `datasets.Experiment`
— so importing an eval produced a timeline without the one thing an eval
produces. These tests pin the whole carry: log bytes in, an experiment file
out, with the scores in it, the provenance on every row, and the states this
project refuses to render as a zero.

Every fixture is a real zip, for the reason `test_inspect_io.py` gives: the
feature's claim is "stdlib only, no new dependency, works air-gapped" and a
mocked archive would not check it.
"""

from __future__ import annotations

import json
import zipfile

import pytest

from modelmri import datasets, inspect_io, receipts
from modelmri.errors import BadRequest


def _log(tmp_path, *, version=2, samples=None, name="run.eval", header=None):
    """The builder from `test_inspect_io.py`, unchanged, on purpose."""
    path = tmp_path / name
    head = {
        "version": version,
        "status": "success",
        "eval": {"task": "arc_easy", "model": "openai/gpt-4o", "created": "2026-08-15"},
        "results": {"total_samples": len(samples or [])},
    }
    if header:
        head.update(header)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("header.json", json.dumps(head))
        for sample in samples or []:
            z.writestr(
                f"samples/{sample['id']}_epoch_{sample.get('epoch', 1)}.json",
                json.dumps(sample),
            )
    return path


def _sample(sid="s1", **over):
    doc = {
        "id": sid,
        "epoch": 1,
        "input": "what is 2+2?",
        "target": "4",
        "events": [
            {
                "event": "model",
                "timestamp": "2026-08-15T00:00:00+00:00",
                "model": "openai/gpt-4o",
                "input": [{"role": "user", "content": "what is 2+2?"}],
                "output": {
                    "choices": [{"message": {"role": "assistant", "content": "4"}}],
                    "usage": {"input_tokens": 12, "output_tokens": 1},
                },
            }
        ],
        "scores": {"match": {"value": "C"}},
    }
    doc.update(over)
    return doc


def _imported(tmp_path, samples, **kw):
    run = inspect_io.read_scores(_log(tmp_path, samples=samples))
    return datasets.from_inspect(run, **kw)


# ------------------------------------------------------- the scores arrive


def test_the_experiment_carries_exactly_the_scores_the_log_states(tmp_path):
    """The gap this whole workstream exists to close."""
    experiment, _ = _imported(
        tmp_path,
        [_sample("a"), _sample("b", scores={"match": {"value": "I"}})],
    )
    # The WHOLE dict per row, hand-written, not `scores["match"]`. "Exactly"
    # is the word in this test's name and an assertion on one key does not
    # check it: a converter that invented a second metric on every row would
    # pass, and an invented metric is the thing `--metric` then ranks runs on.
    got = {r.case_id: r.scores for r in experiment.results}
    assert got == {
        "a:1": {"match": "C", "match_correct": 1.0},
        "b:1": {"match": "I", "match_correct": 0.0},
    }


def test_the_case_id_carries_the_epoch_so_two_epochs_are_two_rows(tmp_path):
    """`Experiment.validated()` refuses a duplicate case id, and a two-epoch
    eval repeats every sample id. Keying on the bare id would make the whole
    import raise on an entirely ordinary log."""
    experiment, _ = _imported(tmp_path, [_sample("a"), dict(_sample("a"), epoch=3)])
    assert [r.case_id for r in experiment.results] == ["a:1", "a:3"]
    experiment.validated()  # would raise on a duplicate


# ------------------------------------------------ an unknown is not a zero


def test_a_sample_the_log_never_scored_says_so_rather_than_scoring_zero(tmp_path):
    """House rule 5. `scores={"match": 0}` here would be this reader inventing
    a mark nobody gave, and it would sort as a real number in every
    comparison."""
    experiment, _ = _imported(tmp_path, [_sample("a", scores={})])
    row = experiment.results[0]
    assert row.scores == {}
    # `Result.could_not_measure` is documented as "the sentence saying why
    # this row has no scores. Empty when it does." This row does not.
    assert row.could_not_measure
    # "not a score of zero" appears in TWO of the three sentences this
    # converter can write, so asserting on it passes against the unreadable-
    # entry sentence — which would be a false claim about the reader's file,
    # saying every entry was unreadable when the log recorded none. The
    # wording below belongs to this branch and to no other.
    assert "nothing measured it" in row.could_not_measure
    assert "recorded no score for this sample at all" in row.could_not_measure


def test_a_sample_with_no_scores_key_at_all_reads_the_same_way(tmp_path):
    doc = _sample("a")
    doc.pop("scores")
    experiment, _ = _imported(tmp_path, [doc])
    assert experiment.results[0].scores == {}
    assert experiment.results[0].could_not_measure


def test_a_sample_the_log_marks_errored_says_so_rather_than_saying_nothing_scored_it(
    tmp_path,
):
    """Three different reasons a row has no score, and the reader has to be
    able to tell them apart: nothing scored it, the scores were unreadable,
    and the SAMPLE ITSELF crashed. Only the last one is a bug in the eval."""
    experiment, _ = _imported(
        tmp_path, [_sample("a", scores={}, error="the tool crashed")]
    )
    row = experiment.results[0]
    assert row.scores == {}
    assert "errored" in row.could_not_measure
    assert "the tool crashed" in row.could_not_measure


def test_an_errored_sample_that_was_still_scored_keeps_its_scores_and_its_error(
    tmp_path,
):
    """Inspect can mark a sample errored AND score it — a scorer that ran on a
    partial transcript. The scores are real and the row stays measured, so
    `could_not_measure` (which flips `measured`) is the wrong place for the
    error. Dropping it entirely was the alternative, and then the one fact the
    log recorded about a row somebody will compare reaches nowhere at all.
    """
    experiment, _ = _imported(tmp_path, [_sample("a", error="the tool crashed")])
    row = experiment.results[0]
    assert row.scores["match"] == "C"
    assert row.measured, "a scored row is a measured row"
    assert experiment.meta["sample_errors"]["a:1"] == "the tool crashed"
    assert "1 scored sample(s)" in experiment.meta["means"]


def test_a_non_finite_score_beside_a_good_one_leaves_the_row_measured(tmp_path):
    """The NaN is dropped with a sentence; the metric next to it is a real
    measurement and must not be thrown out with it."""
    run = inspect_io.read_scores(_log(tmp_path, samples=[_sample("a")]))
    run.rows[0].scores = {"match": "C", "rubric": float("inf")}
    experiment, _ = datasets.from_inspect(run)
    row = experiment.results[0]
    assert row.scores["match"] == "C"
    assert "rubric" not in row.scores
    assert row.measured, "one unusable metric does not unmeasure the row"
    assert "non-finite" in experiment.meta["skipped_scores"]["a:1"]["rubric"]
    datasets.write_experiment(experiment, tmp_path / "out.jsonl")  # must not raise


def test_an_unscored_run_still_reports_how_many_rows_it_has(tmp_path):
    experiment, _ = _imported(tmp_path, [_sample("a", scores={})])
    assert len(experiment.results) == 1
    assert experiment.n_measured == 0


# --------------------------------------------- malformed entries, and the
# --------------------------------------------- tolerance that skips them


def test_a_malformed_score_entry_is_skipped_and_named_rather_than_dropped(tmp_path):
    """`_scores` has always skipped a value that is not a scalar. It did it
    SILENTLY, which the module's own docstring forbids: "what was dropped is
    reported, not implied"."""
    doc = _sample(
        "a",
        scores={
            "good": {"value": 1.0},
            "nested": {"value": {"x": 1}},
            "listy": [1, 2],
            "nothing": {},
        },
    )
    run = inspect_io.read_scores(_log(tmp_path, samples=[doc]))
    row = run.rows[0]
    assert row.scores == {"good": 1.0}
    assert set(row.skipped_scores) == {"nested", "listy", "nothing"}
    assert "dict" in row.skipped_scores["nested"]


def test_the_skipped_entries_survive_into_the_experiment(tmp_path):
    experiment, _ = _imported(
        tmp_path,
        [_sample("a", scores={"good": {"value": 1.0}, "nested": {"value": {}}})],
    )
    # The SENTENCE, not merely a truthy value: the field exists so a reader
    # learns what was there instead, and `True` teaches nobody anything.
    assert "dict" in experiment.meta["skipped_scores"]["a:1"]["nested"]
    assert "good" not in experiment.meta["skipped_scores"]["a:1"]


def test_a_row_whose_only_score_was_malformed_is_not_a_measured_row(tmp_path):
    experiment, _ = _imported(
        tmp_path, [_sample("a", scores={"nested": {"value": {"x": 1}}})]
    )
    row = experiment.results[0]
    assert row.scores == {}
    assert "nested" in row.could_not_measure


def test_a_non_finite_score_never_reaches_the_file(tmp_path):
    """`write_experiment` refuses a NaN anywhere in a row and says the row
    should have carried `could_not_measure` instead. It is this converter's
    job to put it there, not the file writer's job to raise."""
    run = inspect_io.read_scores(_log(tmp_path, samples=[_sample("a")]))
    run.rows[0].scores = {"broken": float("nan")}
    experiment, _ = datasets.from_inspect(run)
    row = experiment.results[0]
    assert "broken" not in row.scores
    assert "broken" in row.could_not_measure
    datasets.write_experiment(experiment, tmp_path / "out.jsonl")  # must not raise


# ------------------------------------------------------------- provenance


def test_every_row_carries_a_receipt_naming_the_log_the_task_and_the_model(tmp_path):
    experiment, _ = _imported(tmp_path, [_sample("a")])
    receipt = experiment.results[0].receipt
    assert receipt["op"] == "inspect_import"
    assert receipt["model"] == "openai/gpt-4o"
    assert receipt["request"]["task"] == "arc_easy"
    assert receipt["request"]["log_name"] == "run.eval"
    assert receipt["request"]["sample_id"] == "a"
    assert receipt["request"]["epoch"] == 1


def test_the_receipt_never_carries_the_path_the_log_was_read_from(tmp_path):
    """A receipt is the part of a finding most likely to be forwarded to a
    stranger, and `tmp_path` is somebody's home directory."""
    experiment, _ = _imported(tmp_path, [_sample("a")])
    blob = json.dumps(experiment.results[0].receipt)
    # An empty receipt contains no path either, so the absence is only worth
    # something once the receipt is known to carry the log's identity.
    assert "run.eval" in blob
    assert str(tmp_path) not in blob
    assert str(tmp_path.parent) not in blob
    assert tmp_path.name not in blob


def test_two_different_logs_do_not_share_one_identity(tmp_path):
    one = inspect_io.read_scores(_log(tmp_path, samples=[_sample("a")], name="a.eval"))
    two = inspect_io.read_scores(_log(tmp_path, samples=[_sample("b")], name="b.eval"))
    assert one.log_sha256 != two.log_sha256
    again = inspect_io.read_scores(tmp_path / "a.eval")
    assert again.log_sha256 == one.log_sha256


def test_the_receipt_identifies_the_log_by_its_bytes_and_not_by_its_name(tmp_path):
    """`len(...) == 16` pins a LENGTH, and a digest of the filename is also
    sixteen characters long. Two runs of the same task written to `run.eval`
    on two days would then carry one identity, which is the exact thing this
    field exists to prevent."""
    first = inspect_io.read_scores(_log(tmp_path, samples=[_sample("a")]))
    exp_first, _ = datasets.from_inspect(first)
    # The same NAME, rewritten with different contents — `run.eval` is what an
    # eval harness calls its output every time it runs.
    second = inspect_io.read_scores(
        _log(tmp_path, samples=[_sample("a", scores={"match": {"value": "I"}})])
    )
    exp_second, _ = datasets.from_inspect(second)

    stamped = exp_first.results[0].receipt["request"]["log_sha256"]
    assert stamped == first.log_sha256[: receipts.DIGEST_CHARS]
    assert stamped != exp_second.results[0].receipt["request"]["log_sha256"]


def test_the_receipt_dates_the_eval_when_it_ran_and_not_when_it_was_read(tmp_path):
    """`measured_at` is the log's own `created`. Stamping the import moment
    here would date somebody else's measurement to whenever this machine
    happened to open the file, and the receipt is the part of a finding most
    likely to be forwarded to a stranger who cannot tell the difference."""
    experiment, _ = _imported(tmp_path, [_sample("a")])
    assert experiment.results[0].receipt["measured_at"] == "2026-08-15"


def test_the_receipt_carries_the_digest_of_the_prompt_the_sample_asked(tmp_path):
    """Not the sample id and not the output: what was ASKED is what makes two
    rows in two files the same question."""
    experiment, _ = _imported(tmp_path, [_sample("a")])
    assert experiment.results[0].receipt["prompt_sha256"] == receipts.digest(
        "what is 2+2?"
    )


# --------------------------------------------- Inspect's C/I, transcribed


def test_the_correct_marker_becomes_a_named_number_and_the_string_survives(tmp_path):
    """Inspect's canonical score value is the string "C" or "I", and
    `compare_experiments` refuses a string. Rewriting `match` to 1.0 would
    make the file claim a number the log never wrote, so the number arrives
    under its OWN name beside the marker it came from."""
    experiment, _ = _imported(
        tmp_path,
        [_sample("a"), _sample("b", scores={"match": {"value": "I"}})],
    )
    rows = {r.case_id: r.scores for r in experiment.results}
    assert rows["a:1"]["match"] == "C"
    assert rows["a:1"]["match_correct"] == 1.0
    assert rows["b:1"]["match"] == "I"
    assert rows["b:1"]["match_correct"] == 0.0


def test_a_marker_this_reader_has_no_number_for_is_left_alone(tmp_path):
    """Inspect also writes P (partial) and N (no answer). A number for those
    would be this reader deciding what a partial credit is worth."""
    experiment, _ = _imported(
        tmp_path, [_sample("a", scores={"match": {"value": "P"}})]
    )
    row = experiment.results[0]
    assert row.scores["match"] == "P"
    assert "match_correct" not in row.scores


def test_the_mapping_from_marker_to_number_is_stated_in_the_file(tmp_path):
    experiment, _ = _imported(tmp_path, [_sample("a")])
    assert experiment.meta["score_markers"]["C"] == 1.0
    assert "match_correct" in experiment.meta["derived_scores"]


def test_a_log_that_already_uses_the_companion_name_is_not_overwritten(tmp_path):
    """A scorer named `match_correct` in the log owns that key. Writing over
    it would replace a number somebody measured with one derived here."""
    doc = _sample(
        "a", scores={"match": {"value": "C"}, "match_correct": {"value": 0.25}}
    )
    experiment, _ = _imported(tmp_path, [doc])
    assert experiment.results[0].scores["match_correct"] == 0.25


def test_a_scorer_that_owns_the_companion_name_owns_it_on_every_row(tmp_path):
    """A scorer that ran on one sample and errored on the next still OWNS its
    column, and the row it missed must not be filled in from the marker.

    Deciding per row is the trap: `match_correct` is absent from row `a`, so a
    per-row check sees a free name and writes a derived 1.0 into a column
    somebody else measured. The file then carries one column holding two
    kinds of number, `derived_scores` calls the whole column derived when half
    of it was measured, and `--metric match_correct` ranks the two against
    each other with nothing on the row saying which is which.
    """
    experiment, _ = _imported(
        tmp_path,
        [
            _sample("a"),
            _sample(
                "b",
                scores={"match": {"value": "C"}, "match_correct": {"value": 0.25}},
            ),
        ],
    )
    rows = {r.case_id: r.scores for r in experiment.results}
    assert rows["a:1"] == {"match": "C"}, "no number nobody measured"
    assert rows["b:1"] == {"match": "C", "match_correct": 0.25}
    assert experiment.meta["derived_scores"] == []
    # Not silently: the marker got no numeric column and the file says which
    # marker it was and why, because "there is no `match_correct` to compare"
    # is otherwise indistinguishable from "this converter forgot".
    assert "match" in experiment.meta["markers_not_derived"]
    assert "match_correct" in experiment.meta["markers_not_derived"]["match"]
    assert "match_correct" in experiment.meta["means"]


# ------------------------------------ the file format needed no redesign


def test_the_experiment_round_trips_through_an_experiment_file(tmp_path):
    """The proof that `Experiment` carries this without a schema change: it
    is written and read back by the shipped reader, unmodified."""
    experiment, dataset = _imported(tmp_path, [_sample("a"), _sample("b")])
    path = datasets.write_experiment(experiment, tmp_path / "run.jsonl")
    back = datasets.read_experiment(path)
    # Hand-written, not `== {r.case_id: r.scores for r in experiment}`. Both
    # sides of that comparison move together — a converter that produced no
    # scores at all would write none, read none back, and pass — so it checks
    # the file format and nothing about the scores it is named for.
    assert {r.case_id: r.scores for r in back.results} == {
        "a:1": {"match": "C", "match_correct": 1.0},
        "b:1": {"match": "C", "match_correct": 1.0},
    }
    assert back.results[0].receipt["request"]["task"] == "arc_easy"
    assert back.meta["source"] == "inspect"
    datasets.write_dataset(dataset, tmp_path / "cases.jsonl")
    assert (
        datasets.read_dataset(tmp_path / "cases.jsonl").fingerprint()
        == dataset.fingerprint()
    )


def test_the_dataset_the_import_builds_joins_to_the_experiment(tmp_path):
    experiment, dataset = _imported(tmp_path, [_sample("a"), _sample("b")])
    assert experiment.dataset_fingerprint == dataset.fingerprint()
    # Written out, not compared to itself: both lists come from the same
    # converter, so an epoch dropped from the case id changes both and the
    # equality holds while the join it stands for has quietly changed.
    assert [c.case_id for c in dataset.cases] == ["a:1", "b:1"]
    assert [r.case_id for r in experiment.results] == ["a:1", "b:1"]
    assert dataset.cases[0].reference == "4", "Inspect's target is a real answer key"


def test_a_log_that_states_no_start_time_is_not_dated_to_the_moment_it_was_read(
    tmp_path,
):
    """The receipt one field over refuses to do exactly this, and the file
    header did it anyway: `write_experiment` filled an empty `started_at` with
    this machine's clock, so an eval that ran in March came back dated today
    and nothing on the row said otherwise. An unknown is not a value that was
    not recorded — least of all in a file whose purpose is to be forwarded."""
    path = _log(
        tmp_path,
        samples=[_sample("a")],
        header={"eval": {"task": "arc_easy", "model": "openai/gpt-4o"}},
    )
    experiment, dataset = datasets.from_inspect(inspect_io.read_scores(path))
    assert experiment.started_at == ""
    assert experiment.results[0].receipt["measured_at"] == ""

    back = datasets.read_experiment(
        datasets.write_experiment(experiment, tmp_path / "run.jsonl")
    )
    assert back.started_at == "", "an unrecorded time is not this machine's clock"
    cases = datasets.read_dataset(
        datasets.write_dataset(dataset, tmp_path / "cases.jsonl")
    )
    assert cases.created_at == ""
    # And said, not merely left blank: a reader looking at a file with no date
    # on it has to learn whether the log had none or the converter lost it.
    assert "states no time at which it ran" in experiment.meta["means"]


def test_a_sample_with_no_target_keeps_an_absent_reference(tmp_path):
    doc = _sample("a")
    doc.pop("target")
    _, dataset = _imported(tmp_path, [doc])
    assert dataset.cases[0].reference is None, "absent is not the empty string"


# ---------------------------------------------- two imports, compared


def test_two_imported_runs_compare_on_the_derived_number(tmp_path):
    """The end the whole carry exists for."""
    before = inspect_io.read_scores(
        _log(
            tmp_path,
            samples=[_sample("a"), _sample("b", scores={"match": {"value": "I"}})],
            name="before.eval",
        )
    )
    after = inspect_io.read_scores(
        _log(
            tmp_path,
            samples=[_sample("a", scores={"match": {"value": "I"}}), _sample("b")],
            name="after.eval",
        )
    )
    exp_before, data = datasets.from_inspect(before, name="before")
    exp_after, _ = datasets.from_inspect(after, name="after")
    comparison = datasets.compare_experiments(
        exp_before,
        exp_after,
        metric="match_correct",
        higher_is_better=True,
        dataset=data,
    )
    statuses = {r.case_id: r.status for r in comparison.rows}
    assert statuses == {"a:1": datasets.WORSE, "b:1": datasets.BETTER}


def test_a_row_scored_on_one_side_only_compares_as_unmeasurable_with_its_reason(
    tmp_path,
):
    """The end house rule 5 exists for. The second run never scored `b`, and
    the honest answer is "nobody measured this", not a zero that reads as a
    total collapse — while `a`, which both runs scored, still compares."""
    before = inspect_io.read_scores(
        _log(tmp_path, samples=[_sample("a"), _sample("b")], name="before.eval")
    )
    after = inspect_io.read_scores(
        _log(
            tmp_path,
            samples=[_sample("a"), _sample("b", scores={})],
            name="after.eval",
        )
    )
    exp_before, data = datasets.from_inspect(before, name="before")
    exp_after, _ = datasets.from_inspect(after, name="after")
    comparison = datasets.compare_experiments(
        exp_before,
        exp_after,
        metric="match_correct",
        higher_is_better=True,
        dataset=data,
    )
    rows = {r.case_id: r for r in comparison.rows}
    assert rows["a:1"].status == datasets.UNCHANGED, "the row both runs scored"
    assert rows["b:1"].status == datasets.UNMEASURABLE
    assert rows["b:1"].delta is None, "never 0.0, which reads as no change"
    assert "nothing measured it" in rows["b:1"].detail


# --------------------------------------------------- the per-score summary


def test_the_summary_counts_every_metric_in_the_run(tmp_path):
    experiment, _ = _imported(
        tmp_path,
        [
            _sample("a"),
            _sample("b", scores={"match": {"value": "I"}}),
            _sample("c", scores={}),
        ],
    )
    summary = datasets.score_summary(experiment)
    by_name = {m["metric"]: m for m in summary["metrics"]}
    assert by_name["match"]["n"] == 2
    assert by_name["match"]["values"] == {"C": 1, "I": 1}
    assert by_name["match_correct"]["n"] == 2
    assert by_name["match_correct"]["median"] == 0.5
    # The row nobody scored is COUNTED, not folded into the metrics above.
    assert summary["n_unmeasured"] == 1
    assert summary["n_results"] == 3


def test_the_summary_of_a_run_with_no_metrics_says_so(tmp_path):
    experiment, _ = _imported(tmp_path, [_sample("a", scores={})])
    summary = datasets.score_summary(experiment)
    assert summary["metrics"] == []
    assert "no metric" in summary["means"]


def test_the_rendered_summary_names_every_metric(tmp_path):
    experiment, _ = _imported(tmp_path, [_sample("a"), _sample("b")])
    text = datasets.render_scores(experiment)
    # Both, and told apart: "match" alone is a substring of "match_correct",
    # so an assertion on it would pass against a render that lost the marker
    # column entirely. The C×2 count is what only the marker column can print.
    assert "Cx2" in text
    assert "match_correct" in text
    assert "median 1" in text
    assert "arc_easy" in text


def test_a_comparison_names_the_other_metrics_the_files_carry(tmp_path):
    """Picking `--metric match` on an Inspect import gets a table of
    unmeasurable rows, because the string is not orderable. The comparison
    already knows every metric present; saying so is what turns that dead end
    into the next command to run."""
    experiment, data = _imported(tmp_path, [_sample("a"), _sample("b")])
    comparison = datasets.compare_experiments(
        experiment,
        experiment,
        metric="match_correct",
        higher_is_better=True,
        dataset=data,
    )
    assert comparison.metrics_present == ["match", "match_correct"]
    # `"match" in render(...)` would pass on "match_correct" alone, which is
    # the metric already in the header line — the whole point is the OTHER one.
    assert "also recorded: match\n" in datasets.render(comparison)


# ------------------------------------------------------ reading the log


def test_read_scores_refuses_a_log_format_it_does_not_speak(tmp_path):
    with pytest.raises(BadRequest, match="version 7"):
        inspect_io.read_scores(_log(tmp_path, version=7, samples=[_sample()]))


def test_read_scores_leaves_no_open_handle(tmp_path):
    """On Windows a still-open handle makes a caller's TemporaryDirectory
    cleanup raise, failing a request whose work had already succeeded."""
    import gc

    path = _log(tmp_path, samples=[_sample("a")], name="held.eval")
    inspect_io.read_scores(path)
    gc.collect()
    path.unlink()
    assert not path.exists()


def test_a_log_with_no_samples_is_refused_rather_than_read_as_an_empty_run(tmp_path):
    # "no samples" is in `read_sample`'s refusal too, and this is the reader
    # that has to be under test — the clause below belongs to `read_scores`.
    with pytest.raises(BadRequest, match="nothing here that was scored"):
        inspect_io.read_scores(_log(tmp_path, samples=[]))


def test_the_listing_cap_is_reported_rather_than_stated_as_the_size(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(inspect_io, "MAX_SAMPLES_LISTED", 3)
    run = inspect_io.read_scores(
        _log(tmp_path, samples=[_sample(f"s{i}") for i in range(6)])
    )
    assert len(run.rows) == 3
    assert run.n_total == 6
    assert run.truncated is True
    experiment, _ = datasets.from_inspect(run)
    assert "6" in experiment.meta["means"]
    assert "3 row(s) short" in experiment.truncated


def test_the_gap_a_capped_read_leaves_survives_into_the_file(tmp_path, monkeypatch):
    """The warning is worth nothing if it dies at the writer.

    `read_experiment` RECOMPUTES `truncated` from the header's `n_results`
    against the rows under it — and those agree, because 3 were declared and 3
    were written. So a sentence set in memory and not carried in the header
    comes back empty, and an experiment 3 rows into a 6-sample eval compares
    as though it were the whole run: `compare_experiments` builds its notes
    from `before.truncated`/`after.truncated` and would have nothing to say.
    """
    monkeypatch.setattr(inspect_io, "MAX_SAMPLES_LISTED", 3)
    run = inspect_io.read_scores(
        _log(tmp_path, samples=[_sample(f"s{i}") for i in range(6)])
    )
    experiment, _ = datasets.from_inspect(run)
    back = datasets.read_experiment(
        datasets.write_experiment(experiment, tmp_path / "run.jsonl")
    )
    assert "3 row(s) short" in back.truncated
    assert "3 row(s) short" in datasets.score_summary(back)["truncated"]
    comparison = datasets.compare_experiments(
        back, back, metric="match_correct", higher_is_better=True
    )
    assert any("row(s) short" in note for note in comparison.notes)


def test_a_file_that_was_capped_and_then_killed_says_both(tmp_path, monkeypatch):
    """Two different gaps, and neither may swallow the other: the cap is what
    the READER left behind, the short file is what the WRITER never finished,
    and a reader holding the file has to be told about both."""
    monkeypatch.setattr(inspect_io, "MAX_SAMPLES_LISTED", 3)
    run = inspect_io.read_scores(
        _log(tmp_path, samples=[_sample(f"s{i}") for i in range(6)])
    )
    experiment, _ = datasets.from_inspect(run)
    path = datasets.write_experiment(experiment, tmp_path / "run.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")  # killed

    back = datasets.read_experiment(path)
    assert "3 row(s) short" in back.truncated, "the cap the reader applied"
    assert "not finished being written" in back.truncated, "the killed write"


# ------------------------------------ the trace the panel imports, too


def test_the_imported_trace_carries_its_scores_into_the_store(tmp_path):
    """The panel's own path. `traces.import_trace` has always persisted a
    `meta` dict and the Inspect trace has never filled one, so an imported
    sample landed in the store with its scores stripped."""
    from modelmri import traces

    store = traces.TraceStore(tmp_path / "t.sqlite")
    try:
        out = inspect_io.read_sample(_log(tmp_path, samples=[_sample("a")]))
        back = store.get_trace(store.import_trace(out.trace))
        assert back["meta"]["scores"] == {"match": "C"}
        assert back["meta"]["source"] == "inspect"
        assert back["meta"]["task"] == "arc_easy"
        assert back["meta"]["model"] == "openai/gpt-4o"
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def test_the_panel_payload_names_the_scores_it_could_not_read(tmp_path):
    doc = _sample("a", scores={"good": {"value": 1}, "nested": {"value": {}}})
    out = inspect_io.read_sample(_log(tmp_path, samples=[doc])).to_dict()
    assert out["scores"] == {"good": 1}
    assert "nested" in out["skipped_scores"]
    assert "nested" in out["means"]
