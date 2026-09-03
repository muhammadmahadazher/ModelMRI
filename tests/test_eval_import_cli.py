# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""`modelmri eval-import`: an Inspect log becomes an experiment file.

The verb that gives `modelmri experiments` something to compare. Its exit
codes follow `compare_experiments` exactly — 2 means IT DID NOT RUN, and a
converter that could not run is not one that produced an empty file.

Driven by calling the function, not a subprocess, the way
`test_experiments_cli.py` does: the exit code and what was printed are the
whole contract.
"""

from __future__ import annotations

import json
import zipfile

import pytest

from modelmri import datasets
from modelmri.cli import import_eval


def _log(tmp_path, *, version=2, samples=None, name="run.eval"):
    path = tmp_path / name
    head = {
        "version": version,
        "status": "success",
        "eval": {"task": "arc_easy", "model": "openai/gpt-4o", "created": "2026-08-15"},
        "results": {"total_samples": len(samples or [])},
    }
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("header.json", json.dumps(head))
        for sample in samples or []:
            z.writestr(
                f"samples/{sample['id']}_epoch_{sample.get('epoch', 1)}.json",
                json.dumps(sample),
            )
    return path


def _sample(sid="s1", value="C"):
    return {
        "id": sid,
        "epoch": 1,
        "input": f"question {sid}",
        "target": "4",
        "events": [
            {
                "event": "model",
                "model": "openai/gpt-4o",
                "input": [{"role": "user", "content": "q"}],
                "output": {"choices": [{"message": {"content": "4"}}]},
            }
        ],
        "scores": {"match": {"value": value}},
    }


@pytest.fixture
def log(tmp_path):
    return str(_log(tmp_path, samples=[_sample("a"), _sample("b", value="I")]))


def test_reading_a_log_without_writing_anything_prints_the_scores(log, capsys):
    """A read with no `--out` is a look, not a conversion. Somebody pointing
    this at a log wants to know what is in it before deciding where it goes."""
    assert import_eval(log) == 0
    said = capsys.readouterr().out
    # Not `"match" in said`: that is a substring of `match_correct`, the
    # DERIVED column, so it passes against a print that lost the log's own
    # marker entirely. The counts are what only the marker column can print.
    assert "Cx1, Ix1" in said
    assert "arc_easy" in said
    assert "openai/gpt-4o" in said


def test_the_written_file_is_one_the_experiments_verb_can_read(log, tmp_path):
    out = tmp_path / "run.jsonl"
    assert import_eval(log, out=str(out)) == 0
    experiment = datasets.read_experiment(out)
    assert {r.case_id: r.scores["match"] for r in experiment.results} == {
        "a:1": "C",
        "b:1": "I",
    }


def test_the_dataset_is_written_beside_it_when_asked(log, tmp_path):
    cases = tmp_path / "cases.jsonl"
    assert (
        import_eval(log, out=str(tmp_path / "run.jsonl"), dataset_out=str(cases)) == 0
    )
    data = datasets.read_dataset(cases)
    experiment = datasets.read_experiment(tmp_path / "run.jsonl")
    assert data.fingerprint() == experiment.dataset_fingerprint


def test_a_log_that_cannot_be_read_exits_two_and_writes_nothing(tmp_path, capsys):
    """Exit 2, not 1: a conversion that did not happen is not a conversion
    that produced an empty file."""
    bad = tmp_path / "not-a-log.eval"
    bad.write_text("this is not a zip", encoding="utf-8")
    out = tmp_path / "run.jsonl"
    assert import_eval(str(bad), out=str(out)) == 2
    assert not out.exists()
    # The authored sentence, not "something was printed": a truthy check on
    # stderr passes against a leaked traceback, which is the one thing a
    # refusal is supposed to replace.
    said = capsys.readouterr().err
    assert "modelmri eval-import:" in said
    assert "not a readable Inspect log" in said
    assert "Traceback" not in said


def test_an_unreadable_version_names_the_version_it_found(tmp_path, capsys):
    path = _log(tmp_path, version=7, samples=[_sample("a")])
    assert import_eval(str(path)) == 2
    assert "version 7" in capsys.readouterr().err


def test_json_output_carries_the_per_metric_summary(log, capsys):
    assert import_eval(log, as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    names = {m["metric"] for m in payload["metrics"]}
    assert {"match", "match_correct"} <= names
    assert payload["n_results"] == 2


def test_the_derived_number_is_named_in_what_it_prints(log, capsys):
    """The one thing a reader must not discover by surprise: a metric in the
    file that the log did not write."""
    import_eval(log)
    said = capsys.readouterr().out
    assert "match_correct" in said
    assert "DERIVED" in said, "the reconciliation sentence, not only the table"
    assert "Cx1, Ix1" in said


def test_a_score_entry_it_could_not_read_is_named_even_on_a_scored_row(
    tmp_path, capsys
):
    """The row that would otherwise be reported nowhere. It kept `match`, so
    it is a MEASURED row and never appears among the unmeasured reasons — and
    its unreadable `depth` entry lives only in the file's own meta."""
    doc = _sample("a")
    doc["scores"]["depth"] = {"value": {"nested": 1}}
    path = _log(tmp_path, samples=[doc])
    assert import_eval(str(path)) == 0
    said = capsys.readouterr().out
    assert "1 sample(s) carried a score entry" in said
    assert "skipped_scores" in said


def test_the_name_and_label_flags_are_wired_through_to_the_file(log, tmp_path):
    """Nothing else in the suite constructs the `eval-import` parser or passes
    either flag, so a dest typo would ship silently — and `--label` is what a
    comparison prints as "what was under test"."""
    out = tmp_path / "run.jsonl"
    assert import_eval(log, out=str(out), name="nightly", label="gpt-4o @ 3f21c9") == 0
    experiment = datasets.read_experiment(out)
    assert experiment.name == "nightly"
    assert experiment.label == "gpt-4o @ 3f21c9"
    # The log's own task and model are still recorded — the flags rename the
    # run, they do not overwrite what the log said about itself.
    assert experiment.meta["task"] == "arc_easy"
    assert experiment.meta["model"] == "openai/gpt-4o"


def test_nothing_is_written_when_no_destination_is_given(log, tmp_path, capsys):
    """A read is a look. It must not leave a file behind in the working
    directory that nobody asked for."""
    before = set(tmp_path.iterdir())
    assert import_eval(log) == 0
    assert set(tmp_path.iterdir()) == before
    assert "nothing was written" in capsys.readouterr().out
