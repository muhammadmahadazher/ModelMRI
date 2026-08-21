"""One import must not be able to kill the agents panel permanently.

REPRODUCED in an isolated store, before any of this existed:

    POST /api/traces/import  {"id":"poison", …, "meta":"notadict", "steps":[…]}
        -> 200 {"id":"poison"}
    GET  /api/traces          -> 500
    GET  /api/patterns/across -> 500
    POST /api/rubric/score    -> 500

The import SUCCEEDED. Every reader of the store died afterwards, across
restarts, until somebody found and deleted the row by hand.

Two halves, and both are needed. Refusing the poison on the way in helps
nobody who already imported one — that row is on their disk — so the readers
have to survive damage as well.
"""

from __future__ import annotations

import tempfile

import pytest
from fastapi.testclient import TestClient

from modelmri import traces
from modelmri.errors import BadRequest


@pytest.fixture()
def app_in_its_own_store(monkeypatch):
    """A store of its own. A poisoned row that survives is the whole point,
    and it must not be written into the developer's real one."""
    monkeypatch.setenv("MODELMRI_HOME", tempfile.mkdtemp(prefix="mri-test-traces-"))
    from modelmri.server import create_app

    return create_app()


GOOD_STEP = {"kind": "llm_call", "started_ms": 0, "ended_ms": 1, "name": "generate"}


def test_a_trace_whose_meta_is_not_an_object_is_refused_on_the_way_in(
    app_in_its_own_store,
):
    client = TestClient(app_in_its_own_store, raise_server_exceptions=False)
    r = client.post(
        "/api/traces/import",
        json={
            "id": "poison",
            "name": "p",
            "started_at": "2026-01-02T00:00:00Z",
            "meta": "notadict",
            "steps": [GOOD_STEP],
        },
    )
    assert r.status_code == 422, r.text
    assert "meta" in r.text

    # And the store is still readable, which is what the 500s destroyed.
    for path in ("/api/traces", "/api/patterns/across"):
        assert client.get(path).status_code == 200


def test_a_parent_id_that_is_not_a_scalar_is_refused_by_name(app_in_its_own_store):
    """It went straight into an sqlite bind parameter, so the crash was an
    InterfaceError from the driver — at somebody who sent a nested object."""
    client = TestClient(app_in_its_own_store, raise_server_exceptions=False)
    r = client.post(
        "/api/traces/import",
        json={
            "id": "pid",
            "name": "x",
            "started_at": "2026-01-05T00:00:00Z",
            "steps": [{**GOOD_STEP, "parent_id": {"a": 1}}],
        },
    )
    assert r.status_code == 422, r.text
    assert "parent_id" in r.text


def test_a_store_that_is_already_poisoned_still_reads(app_in_its_own_store):
    """The other half. A row written by an older build is on disk, and a fix
    that only guards the entrance leaves that reader with a dead panel."""
    client = TestClient(app_in_its_own_store, raise_server_exceptions=False)
    client.post(
        "/api/traces/import",
        json={
            "id": "real",
            "name": "a real run",
            "started_at": "2026-01-01T00:00:00Z",
            "steps": [GOOD_STEP],
        },
    )

    # The app's OWN store, not a second one: the point is a row already
    # sitting in the database this server is reading from.
    store = app_in_its_own_store.state.traces
    store._db.execute(
        "INSERT OR REPLACE INTO trace (id, name, started_at, meta) VALUES (?,?,?,?)",
        ("poisoned", "from an older build", "2026-01-02T00:00:00Z", "notadict"),
    )
    store._db.commit()

    r = client.get("/api/traces")
    assert r.status_code == 200, r.text
    listed = {t["id"] for t in r.json()}
    # BOTH are listed. The damaged row is not hidden either — a trace that
    # vanishes from the panel is its own kind of wrong answer.
    assert listed == {"real", "poisoned"}
    assert client.get("/api/patterns/across").status_code == 200


def test_damage_reads_as_empty_rather_than_raising():
    """`_loads` exists for exactly this and its docstring says so. The two
    places that read TRACE meta called bare `json.loads` instead, which is how
    the hazard it names came true."""
    assert traces._loads(None) == {}
    assert traces._loads("") == {}
    assert traces._loads("notadict") == {}
    assert traces._loads('"a json string"') == {}
    assert traces._loads("[1, 2, 3]") == {}
    assert traces._loads('{"a": 1}') == {"a": 1}


def test_an_otlp_field_of_the_wrong_type_lands_on_the_authored_refusal():
    """`resource.get("scopeSpans") or []` covers a MISSING key, not a present
    one holding 5 — and iterating an int is a TypeError. This route's entire
    job is to accept a body written by somebody else's exporter."""
    from modelmri import otel

    assert otel._as_list([1, 2]) == [1, 2]
    assert otel._as_list(5) == []
    assert otel._as_list(None) == []
    assert otel._as_list({"a": 1}) == []

    # `from_otlp` is the round-trip reader: it returns the steps it could
    # read, so an empty list is its right answer for a body with none. What it
    # must not do is raise TypeError from inside a for-loop.
    assert otel.from_otlp({"resourceSpans": [{"scopeSpans": 5}]}) == []
    assert otel.from_otlp({"resourceSpans": 5}) == []

    # `ingest` is the ROUTE's function, and it owns the authored sentence.
    with pytest.raises(BadRequest) as err:
        otel.ingest({"resourceSpans": [{"scopeSpans": 5}]})
    assert "spans" in str(err.value)
