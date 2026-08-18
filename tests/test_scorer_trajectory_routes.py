"""Two modules that existed, were tested, and could not be reached.

`scorers.py` (82 KB) and `trajectory.py` (44 KB) had no route on any surface —
not HTTP, not the CLI. A feature nobody can open is a feature that does not
ship, and this repo has now found that pattern three times: `vision_attr` was
the last one.

These tests are about the SEAM rather than the modules, whose own suites are
thorough. What they pin is that the route relays the honest half: the failure
mode beside each metric, the counts without a verdict, and the price before
the table is built.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from modelmri.server import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


# ------------------------------------------------------------------ scorers


def test_every_scorer_states_how_it_fails_and_not_only_what_it_measures(client):
    """`failure_mode` is what makes the catalogue honest rather than a menu.

    A number from a metric whose blind spot you do not know is a number you
    cannot act on, and every one of these has a blind spot: `contains_all`
    finds `kill` inside `skill`, `edit_similarity` counts characters rather
    than meaning, `json_valid` is satisfied by `{}`.
    """
    d = client.get("/api/scorers").json()
    assert d["scorers"], "the catalogue came back empty"
    for row in d["scorers"]:
        assert row["name"]
        assert row["summary"]
        assert row["failure_mode"], f"{row['name']} does not say how it fails"


def test_the_catalogue_promises_no_model_is_asked(client):
    """The whole argument for these metrics. A scorer that asks a language
    model needs a calibration gate nobody ships and answers differently next
    Tuesday."""
    assert "asks a model" in client.get("/api/scorers").json()["means"]


def test_a_scorer_that_is_not_in_the_catalogue_is_refused_by_name(client):
    """Naming the alternatives, because a caller who typed a metric that does
    not exist has no other way to learn which ones do."""
    r = client.post("/api/scorers/run", json={"name": "vibes", "output": "x"})
    assert r.status_code == 422
    said = r.json()["error"]
    assert "vibes" in said
    assert "exact_match" in said, "the refusal did not offer the real ones"


def test_the_documented_substring_trap_really_behaves_that_way(client):
    """The catalogue says `kill` is found inside `skill`. If the route did not
    reproduce that, the warning would be describing a different function than
    the one being run."""
    r = client.post(
        "/api/scorers/run",
        json={"name": "contains_all", "output": "she has skill", "reference": ["kill"]},
    )
    assert r.status_code == 200
    assert r.json()["passed"] is True


def test_exact_match_is_whitespace_sensitive_over_http(client):
    """Its own stated failure mode, through the seam."""
    same = client.post(
        "/api/scorers/run",
        json={"name": "exact_match", "output": "42", "reference": "42"},
    ).json()
    spaced = client.post(
        "/api/scorers/run",
        json={"name": "exact_match", "output": "42 ", "reference": "42"},
    ).json()
    assert same["passed"] is True
    assert spaced["passed"] is False


def test_a_scorer_needs_a_name(client):
    assert client.post("/api/scorers/run", json={"output": "x"}).status_code == 422
    assert (
        client.post("/api/scorers/run", json={"name": "", "output": "x"}).status_code
        == 422
    )


# --------------------------------------------------------------- trajectory


def test_the_alignment_is_priced_before_the_table_is_built(client):
    """A `reference x candidate` grid, and the caller can shorten the span
    themselves rather than discovering the cap by hitting it."""
    d = client.get("/api/trajectory/cost?reference=3&candidate=4").json()
    assert d["cells"] == 12
    assert d["fits"] is True


def test_a_span_past_the_cap_is_refused_rather_than_truncated(client):
    """Aligning a prefix would report every step past the cut as MISSING,
    which is the exact wrong conclusion this measurement exists to reach."""
    d = client.get("/api/trajectory/cost?reference=5000&candidate=5000").json()
    assert d["fits"] is False
    assert "refuses rather than aligning a prefix" in d["means"]


def test_a_comparison_reports_counts_and_refuses_to_score(client):
    """The rule this module is built on. Everybody else scores plan adherence
    with a language-model judge; a shorter path is not a worse path, and a
    single figure says that it is."""
    r = client.post(
        "/api/trajectory/compare",
        json={
            "reference": ["search", "read_file", "edit_file", "run_tests", "commit"],
            "candidate": ["search", "read_file", "run_tests", "run_tests", "commit"],
        },
    )
    assert r.status_code == 200
    d = r.json()
    assert d["n_missing"] == 1, "edit_file was planned and not run"
    assert d["n_extra"] == 1, "run_tests ran twice"
    assert "NOT A SCORE" in d["means"]
    # And there is genuinely no score field to read by accident.
    assert not any("adherence" in k.lower() or k == "score" for k in d)


def test_an_empty_plan_is_a_fact_about_the_plan_not_the_run(client):
    """With nothing to compare against, every recorded step is "extra" — which
    reads as a finding about the run and is not one."""
    r = client.post(
        "/api/trajectory/compare", json={"reference": [], "candidate": ["a", "b"]}
    )
    assert r.status_code == 422
    assert "no reference trajectory" in r.json()["error"]


# -------------------------------------------------------------- experiments


@pytest.fixture
def two_runs(tmp_path):
    """A dataset and two runs of it, one better on a case and worse on another."""
    from modelmri import datasets

    data = datasets.from_inputs("probe", ["2+2?", "capital of France?"])
    datasets.write_dataset(data, tmp_path / "probe.jsonl")

    def run(name, scores):
        rows = [
            datasets.Result(
                case_id=case.case_id, output=str(s), scores={"faithfulness": s}
            )
            for case, s in zip(data.cases, scores, strict=True)
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
        "before": run("before", [0.80, 0.70]),
        "after": run("after", [0.90, 0.55]),
        "dataset": str(tmp_path / "probe.jsonl"),
    }


def test_a_comparison_counts_both_directions_rather_than_averaging(
    client, two_runs, monkeypatch
):
    """The whole point of a per-case comparison. A mean would report these two
    runs as roughly unchanged and hide that one case got materially worse —
    which is the regression somebody needed to see."""
    monkeypatch.setattr("modelmri.server._not_from_this_machine", lambda *a, **k: None)
    r = client.post(
        "/api/experiments/compare",
        json={
            "before": two_runs["before"],
            "after": two_runs["after"],
            "metric": "faithfulness",
            "higher_is_better": True,
            "dataset": two_runs["dataset"],
        },
    )
    assert r.status_code == 200
    counts = r.json()["counts"]
    assert counts["better"] == 1
    assert counts["worse"] == 1
    assert sum(counts.values()) == r.json()["n_cases"]


def test_the_direction_of_good_has_to_be_stated(client, two_runs, monkeypatch):
    """`higher_is_better` has NO default anywhere in this stack. There is no
    way to tell from a metric's name which way is good — KL is better lower,
    faithfulness better higher — and a wrong guess inverts every conclusion
    while producing output that looks entirely reasonable."""
    monkeypatch.setattr("modelmri.server._not_from_this_machine", lambda *a, **k: None)
    r = client.post(
        "/api/experiments/compare",
        json={
            "before": two_runs["before"],
            "after": two_runs["after"],
            "metric": "faithfulness",
        },
    )
    assert r.status_code == 422


def test_no_dataset_supplied_is_null_and_not_zero(client, two_runs, monkeypatch):
    """`references: null` means nothing looked. `0` means it looked and there
    were none. Collapsing them would say a set has no expected answers when in
    fact nobody opened it."""
    monkeypatch.setattr("modelmri.server._not_from_this_machine", lambda *a, **k: None)
    body = {
        "before": two_runs["before"],
        "after": two_runs["after"],
        "metric": "faithfulness",
        "higher_is_better": True,
    }
    nothing_looked = client.post("/api/experiments/compare", json=body).json()
    looked = client.post(
        "/api/experiments/compare", json={**body, "dataset": two_runs["dataset"]}
    ).json()
    assert nothing_looked["references"] is None
    assert looked["references"] == 0


def test_reading_experiments_from_elsewhere_on_the_network_is_refused(client, two_runs):
    """Three paths arrive in this body, and a path names a file on the disk
    THIS server runs on — the same guard every other file-reading route
    carries."""
    r = client.post(
        "/api/experiments/compare",
        json={
            "before": two_runs["before"],
            "after": two_runs["after"],
            "metric": "faithfulness",
            "higher_is_better": True,
        },
    )
    assert r.status_code == 403
    assert "only possible from this machine" in r.json()["error"]
