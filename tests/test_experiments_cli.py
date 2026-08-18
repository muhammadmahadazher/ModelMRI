"""`modelmri experiments` as a CI gate, and what its exit codes mean.

The roadmap asks for this to run on every pull request "in milliseconds, on a
machine with no accelerator": both sides are already measured, so the
comparison is arithmetic over JSONL and pays for no torch.

The exit code IS the feature. Three values, and the third is the one that
matters most:

    0  ran, and no more cases got worse than the gate allows
    1  ran, and too many got worse
    2  COULD NOT RUN

A gate that cannot run is not a gate that passed. Collapsing 2 into either of
the others is how a broken comparison gets read as a green build.
"""

from __future__ import annotations

import pytest

from modelmri import datasets
from modelmri.cli import compare_experiments


@pytest.fixture
def runs(tmp_path):
    """One dataset, and two runs where one case improved and one regressed."""
    data = datasets.from_inputs("probe", ["2+2?", "capital?", "who?"])
    datasets.write_dataset(data, tmp_path / "probe.jsonl")

    def run(name, scores):
        rows = [
            datasets.Result(case_id=c.case_id, output=str(s), scores={"faith": s})
            for c, s in zip(data.cases, scores, strict=True)
        ]
        experiment = datasets.Experiment(
            name=name,
            dataset_name=data.name,
            dataset_fingerprint=data.fingerprint(),
            results=rows,
        )
        path = tmp_path / f"{name}.jsonl"
        datasets.write_experiment(experiment, path)
        return str(path)

    return {
        "before": run("before", [0.80, 0.70, 0.60]),
        "after": run("after", [0.90, 0.55, 0.60]),
        "dataset": str(tmp_path / "probe.jsonl"),
    }


def test_a_regression_fails_the_gate_by_default(runs, capsys):
    """Default `--fail-on-worse 0`, because any regression should fail a gate.
    A threshold above zero is somebody deciding in advance how much breakage
    is acceptable, which is a decision to make out loud."""
    code = compare_experiments(
        runs["before"], runs["after"], metric="faith", higher_is_better=True
    )
    assert code == 1
    assert "got worse" in capsys.readouterr().err


def test_the_allowance_is_honoured_when_it_is_stated(runs):
    """One case regressed, and one is allowed."""
    assert (
        compare_experiments(
            runs["before"],
            runs["after"],
            metric="faith",
            higher_is_better=True,
            fail_on_worse=1,
        )
        == 0
    )


def test_a_run_compared_against_itself_passes(runs):
    assert (
        compare_experiments(
            runs["before"], runs["before"], metric="faith", higher_is_better=True
        )
        == 0
    )


def test_the_direction_inverts_the_verdict(runs):
    """The same two files, the same metric, opposite conclusions — which is
    exactly why `higher_is_better` has no default anywhere in this stack.

    Read lower-is-better, the case that went 0.70 -> 0.55 IMPROVED and the one
    that went 0.80 -> 0.90 regressed. Still one worse, so the gate still
    fails; what changes is WHICH case it is unhappy about, and a wrong guess
    would send somebody to investigate the wrong one.
    """
    lower = compare_experiments(
        runs["before"], runs["after"], metric="faith", higher_is_better=False
    )
    assert lower == 1


def test_a_gate_that_cannot_run_is_not_a_gate_that_passed(runs, capsys):
    """Exit 2, and the refusal names the metrics that ARE recorded — a caller
    who typed a metric that does not exist has no other way to learn which
    ones do."""
    code = compare_experiments(
        runs["before"], runs["after"], metric="not-recorded", higher_is_better=True
    )
    assert code == 2, "an unrunnable comparison must not report as pass or fail"
    said = capsys.readouterr().err
    assert "not-recorded" in said
    assert "faith" in said


def test_a_missing_file_is_also_two_not_one(runs, tmp_path, capsys):
    """Same rule: the comparison did not happen, so it did not pass and it did
    not find regressions."""
    code = compare_experiments(
        str(tmp_path / "nope.jsonl"),
        runs["after"],
        metric="faith",
        higher_is_better=True,
    )
    assert code == 2
    assert capsys.readouterr().err.strip()


def test_json_output_carries_the_counts(runs, capsys):
    """For a CI job that wants to post the numbers rather than the prose."""
    import json

    compare_experiments(
        runs["before"],
        runs["after"],
        metric="faith",
        higher_is_better=True,
        fail_on_worse=99,
        as_json=True,
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["counts"]["worse"] == 1
    assert payload["counts"]["better"] == 1
    assert sum(payload["counts"].values()) == payload["n_cases"]
