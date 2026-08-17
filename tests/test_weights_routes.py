"""The weight table and the scanner, over HTTP.

Netron reads a file and TensorBoard's Debugger V2 needs a run instrumented in
advance. These read the module already sitting in this process's memory, which
is the whole difference — and it is also why every one of them has to refuse
clearly when there is no model loaded.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from modelmri import weights_scan  # noqa: E402
from modelmri.server import create_app  # noqa: E402


class _Tiny(torch.nn.Module):
    """Three parameters and one buffer, so the three-way split is exercised."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(64, 32)
        self.block = torch.nn.Linear(32, 32)
        self.register_buffer("scale", torch.ones(32))


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.fixture
def loaded():
    app = create_app()
    app.state.runtime.model = _Tiny()
    app.state.runtime.hf_id = "tiny/test"
    return TestClient(app)


# ------------------------------------------------- nothing loaded is a refusal


def test_the_table_refuses_when_no_model_is_in_memory(client):
    """ "There is nothing to describe" must not come back as an empty table —
    an empty table reads as a model with no weights."""
    r = client.get("/api/weights")
    assert r.status_code == 409
    assert "No model is loaded" in r.json()["error"]
    assert "rather than a file on disk" in r.json()["error"]


def test_the_cost_refuses_too_rather_than_quoting_zero(client):
    """Zero passes over zero tensors is a plausible-looking number for a
    question that was never asked."""
    r = client.get("/api/weights/cost")
    assert r.status_code == 409
    assert "no weights to price" in r.json()["error"]


# ------------------------------------------------------------ the cheap half


def test_the_cost_is_answerable_without_reading_a_weight(loaded):
    """The table half is free and the health half is memory bandwidth, so the
    price of the expensive half comes out of the cheap half."""
    d = loaded.get("/api/weights/cost").json()
    assert d["tensors"] == 4
    assert d["elements_total"] == 64 * 32 + 32 * 32 + 32 + 32
    assert d["scanned"] > 0


def test_exhaustive_costs_at_least_as_much_as_sampled(loaded):
    sampled = loaded.get("/api/weights/cost").json()
    every = loaded.get("/api/weights/cost?exhaustive=true").json()
    assert every["scanned"] >= sampled["scanned"]


# ----------------------------------------------------------------- the table


def test_the_three_categories_sum_to_the_total_over_http(loaded):
    """The invariant the module holds must survive serialisation — a JSON
    body where two of three numbers are visible and the third is missing is
    the silent gap all over again."""
    d = loaded.get("/api/weights").json()
    assert (
        d["trainable_elements"] + d["frozen_elements"] + d["buffer_elements"]
        == d["elements_total"]
    )
    assert d["buffer_elements"] == 32


def test_health_is_off_by_default_and_says_so(loaded):
    """Off is the honest default: the table is free, the scan reads every
    element it is allowed to, and the cost route prices it first."""
    assert loaded.get("/api/weights").json()["health_checked"] is False
    assert loaded.get("/api/weights?health=true").json()["health_checked"] is True


def test_the_source_names_the_model_that_was_read(loaded):
    assert loaded.get("/api/weights").json()["source"] == "tiny/test"


# ---------------------------------------------------------------- the scanner


def test_a_scan_from_elsewhere_on_the_network_is_refused(client, tmp_path):
    """A path names a file on the disk THIS server runs on, so a request from
    another machine naming one is a request to read somebody else's disk."""
    r = client.post("/api/weights/scan", json={"path": str(tmp_path)})
    assert r.status_code == 403
    assert "only possible from this machine" in r.json()["error"]


def test_the_summary_never_claims_files_were_read_and_unread_at_once():
    """The bug this test exists for: the first version said "N file(s) read and
    nothing executable found" AND "N could not be read" — opposite claims about
    the same N. On a directory of Python source, where every file is unscanned
    by design, it printed both about all of them."""
    from modelmri.server import _scan_summary

    unscanned = [
        weights_scan.Report(path="a.py", verdict=weights_scan.UNSCANNED, reason="x"),
        weights_scan.Report(path="b.py", verdict=weights_scan.UNSCANNED, reason="x"),
    ]
    said = _scan_summary(unscanned, [], unscanned)
    assert "NONE of the 2" in said
    assert "not a clean bill of health" in said
    assert "read and nothing executable found" not in said


def test_a_mixed_directory_counts_only_what_was_actually_read():
    from modelmri.server import _scan_summary

    reports = [
        weights_scan.Report(path="ok.safetensors", verdict=weights_scan.SAFE),
        weights_scan.Report(path="a.py", verdict=weights_scan.UNSCANNED, reason="x"),
    ]
    said = _scan_summary(reports, [], reports[1:])
    assert "1 file(s) read" in said
    assert "1 could not be read" in said


def test_a_dangerous_finding_speaks_for_the_whole_scan():
    from modelmri.server import _scan_summary

    bad = weights_scan.Report(
        path="evil.bin",
        verdict=weights_scan.DANGEROUS,
        findings=[weights_scan.Finding("executes on load", "names `os.system`")],
    )
    said = _scan_summary([bad], [bad], [])
    assert "executes something when it is loaded" in said


def test_an_empty_directory_is_not_reported_as_clean():
    from modelmri.server import _scan_summary

    assert "Nothing weight-shaped" in _scan_summary([], [], [])
