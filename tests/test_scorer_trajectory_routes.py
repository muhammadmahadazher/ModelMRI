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
