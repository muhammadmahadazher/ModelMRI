"""The agent-run section of a `.mri`, held to the same standard as `patch`.

A `.mri` is meant to be forwarded, so `_trace` runs on bytes a stranger sent.
The steps reach the viewer's timeline as loop bounds and their milliseconds
reach a pixel offset, so a 400,000-step claim or a string where a number
belongs has to stop at parse rather than in whoever's browser opened the file.
"""

from __future__ import annotations

import pytest

from modelmri import session
from modelmri.session import SessionError

API_KEY = "sk-ant-api03-" + "A" * 88


def _trace(n=2, **over):
    doc = {
        "id": "t1",
        "name": "a run",
        "started_at": "2026-08-14T00:00:00Z",
        "steps": [
            {
                "id": f"s{i}",
                "kind": "llm_call" if i % 2 == 0 else "tool_call",
                "name": "plan" if i % 2 == 0 else "search",
                "input": "go",
                "output": "ok",
                "started_ms": i * 10,
                "duration_ms": 5,
                "error": i == 1,
            }
            for i in range(n)
        ],
    }
    doc.update(over)
    return doc


def _build(**over) -> bytes:
    args = dict(
        model_id="gpt2",
        device="cpu",
        dtype="float32",
        n_params=124_000_000,
        tokens=["a", "b"],
        prompt="hello",
        generation="world",
        attention={(0, 0): [[1.0, 0.0], [0.5, 0.5]]},
        n_layers=1,
        n_heads=1,
        n_prompt=1,
    )
    args.update(over)
    return session.build(**args)


# ------------------------------------------------------------ round trip


def test_a_run_survives_the_round_trip():
    parsed = session.parse(_build(trace=_trace()))
    assert parsed.has_trace()
    assert len(parsed.trace["steps"]) == 2
    assert parsed.trace["steps"][1]["error"] is True


def test_no_trace_is_the_ordinary_case():
    parsed = session.parse(_build())
    assert not parsed.has_trace()
    assert parsed.trace == {}
    assert parsed.failing_step() is None


def test_the_failing_step_resolves():
    parsed = session.parse(_build(trace=_trace(), step_ref="s1"))
    step = parsed.failing_step()
    assert step is not None
    assert step["id"] == "s1" and step["name"] == "search"


def test_a_step_ref_naming_a_step_the_file_lacks_is_refused_at_write():
    """Otherwise the viewer opens a bundle whose highlighted step is a dead
    link."""
    from modelmri.bundle import BundleError

    with pytest.raises(BundleError, match="not among the steps"):
        _build(trace=_trace(), step_ref="nope")


def test_the_same_ref_is_refused_at_read_on_a_hand_edited_file():
    """Write-time validation protects files this tool wrote. A stranger's
    file goes through the reader."""
    bad = _trace()
    bad["step_ref"] = "does-not-exist"
    with pytest.raises(SessionError, match="nothing for the viewer to open"):
        session._trace({"trace": bad})


# --------------------------------------------------- redaction, at export


def test_a_credential_in_a_step_never_reaches_the_file():
    """The recorder redacts at DELIVERY, which is behind us by the time steps
    come out of the store."""
    doc = _trace()
    doc["steps"][0]["input"] = f"call with {API_KEY}"
    blob = _build(trace=doc)
    assert API_KEY.encode() not in blob
    parsed = session.parse(blob)
    assert "[redacted:api-key]" in parsed.trace["steps"][0]["input"]


def test_a_credential_in_the_prompt_never_reaches_the_file():
    """This holds whether or not there is a trace: a key pasted into a prompt
    is a key either way."""
    blob = _build(prompt=f"my key {API_KEY}")
    assert API_KEY.encode() not in blob
    assert "[redacted:api-key]" in session.parse(blob).prompt


# ------------------------------------------------------------- the bounds


def test_a_run_with_too_many_steps_is_refused_at_read():
    bad = _trace(n=1)
    bad["steps"] = bad["steps"] * (session.MAX_TRACE_STEPS + 1)
    with pytest.raises(SessionError, match="format holds"):
        session._trace({"trace": bad})


def test_a_step_with_no_kind_is_refused():
    bad = _trace()
    del bad["steps"][0]["kind"]
    with pytest.raises(SessionError, match="has no kind"):
        session._trace({"trace": bad})


def test_a_step_that_is_not_an_object_is_refused():
    bad = _trace()
    bad["steps"][0] = "nonsense"
    with pytest.raises(SessionError, match="is not an object"):
        session._trace({"trace": bad})


def test_a_section_that_is_not_an_object_is_refused():
    with pytest.raises(SessionError, match="not a set of fields"):
        session._trace({"trace": ["nope"]})


def test_a_section_with_no_steps_list_is_refused():
    with pytest.raises(SessionError, match="no steps list"):
        session._trace({"trace": {"id": "x"}})


def test_absent_is_fine_and_returns_nothing():
    assert session._trace({}) == {}


# ------------------------------------------------- null is not zero, still


def test_a_step_with_no_duration_keeps_null_rather_than_zero():
    """`traces.py` made the column nullable to express exactly this, and the
    export must not undo it."""
    doc = _trace()
    del doc["steps"][0]["duration_ms"]
    parsed = session.parse(_build(trace=doc))
    assert parsed.trace["steps"][0]["duration_ms"] is None


def test_a_string_where_a_number_belongs_becomes_null_not_a_guess():
    bad = _trace()
    bad["steps"][0]["duration_ms"] = "quick"
    out = session._trace({"trace": bad})
    assert out["steps"][0]["duration_ms"] is None


def test_a_boolean_is_not_an_integer_here():
    bad = _trace()
    bad["steps"][0]["tokens_in"] = True
    out = session._trace({"trace": bad})
    assert out["steps"][0]["tokens_in"] is None


def test_truncation_is_carried_so_the_reader_knows_it_is_partial():
    doc = _trace(n=3)
    doc["n_steps_total"] = 900
    doc["truncated"] = 897
    out = session._trace({"trace": doc})
    assert out["n_steps_total"] == 900 and out["truncated"] == 897


# ------------------------------------------------------------ additive


def test_the_format_version_does_not_move_for_this_section():
    """An older reader ignores an unknown key, which is how `patch` already
    works — moving the version would make every existing file look stale."""
    import gzip
    import json

    def version_of(blob):
        return json.loads(gzip.decompress(blob))["format_version"]

    assert version_of(_build()) == version_of(_build(trace=_trace()))
    assert version_of(_build(trace=_trace())) == session.FORMAT_VERSION


def test_an_older_reader_ignores_the_section_rather_than_failing():
    """The additive contract: a reader that predates `trace` sees a key it
    does not know and carries on, which is why the version can stay put."""
    import gzip
    import json

    doc = json.loads(gzip.decompress(_build(trace=_trace())))
    assert "trace" in doc
    del doc["trace"]
    # The same bytes without the section still parse — nothing else in the
    # file depends on it.
    revived = session.parse(gzip.compress(json.dumps(doc).encode()))
    assert not revived.has_trace()
    assert revived.prompt == "hello"


def test_a_bundle_stays_small_enough_to_open():
    blob = _build(trace=_trace(n=200))
    assert len(blob) < 300_000, f"{len(blob):,} bytes"


# ------------------------------------------- the run reaching a screen


def _client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from modelmri.server import create_app

    return TestClient(create_app())


def test_a_carried_run_is_reported_on_the_session_state():
    """The three siblings — patch, ground, patch_graph — were all here and
    this one was not, so a bundle built around a failing step opened to a
    panel reading "0 recordings" with the run inside the file."""
    # `step_ref` is an argument to `build`, not a field the caller writes
    # into the trace: the writer checks it names a step the file carries.
    blob = _build(trace=_trace(3), step_ref="s1")
    with _client() as c:
        assert c.post("/api/session/open", content=blob).status_code == 200
        state = c.get("/api/session/state").json()["trace"]
        assert state["available"] is True
        assert state["n_steps"] == 3
        # `bundle.prepare` is the authority on these two, not the caller, so a
        # complete run reports itself complete rather than taking a claim.
        assert state["n_steps_total"] == 3
        assert state["truncated"] == 0
        # The step the bundle was built AROUND is the reason it was sent.
        assert state["step_ref"] == "s1"
        c.post("/api/session/close")


def _hand_authored(trace: dict) -> bytes:
    """A `.mri` written by something other than this library.

    Which is the case the whole section exists for: a bundle is meant to be
    forwarded, so `_trace` runs on bytes a stranger sent, and a sender whose
    run was longer than the format holds says so in the file rather than
    silently shipping a section as a whole.
    """
    import json

    return json.dumps(
        {
            "format": session.FORMAT,
            "format_version": session.FORMAT_VERSION,
            "meta": {"model": "Qwen/Qwen3-1.7B"},
            "prompt": "hello",
            "generation": "world",
            "tokens": ["a", "b"],
            "attention": {},
            "trace": trace,
        }
    ).encode("utf-8")


def test_a_capped_run_reports_what_the_senders_run_held_not_what_fits():
    """3 steps out of 9 is a section of a run, and reporting 3 for both counts
    would present it as the whole of one. The cut is REPORTED, never silent."""
    blob = _hand_authored(_trace(3, n_steps_total=9, truncated=6))
    with _client() as c:
        assert c.post("/api/session/open", content=blob).status_code == 200
        state = c.get("/api/session/state").json()["trace"]
        assert (state["n_steps"], state["n_steps_total"], state["truncated"]) == (
            3,
            9,
            6,
        )
        doc = c.get("/api/session/trace").json()
        assert len(doc["steps"]) == 3
        assert doc["n_steps_total"] == 9
        assert doc["truncated"] == 6
        c.post("/api/session/close")


def test_the_carried_run_is_served_with_the_same_rollup_the_store_gets():
    """Two sources for one shape. A run read from a file and a run read from
    the store go through the same `ledger.roll_up`, so they read identically
    rather than nearly."""
    blob = _build(trace=_trace(4))
    with _client() as c:
        c.post("/api/session/open", content=blob)
        doc = c.get("/api/session/trace").json()
        assert doc["available"] is True
        assert len(doc["steps"]) == 4
        assert set(doc["tokens_by_step"]) == {"s0", "s1", "s2", "s3"}
        assert "counts" in doc["tokens"]
        # Priced through the same biller, and a price file that cannot be read
        # is a field rather than a 500.
        assert "means" in doc["cost"]
        c.post("/api/session/close")


def test_no_session_and_no_run_are_both_the_same_ordinary_state():
    """`available: False`, not an error. Most sessions carry no agent run, and
    a panel that treats "nothing here" as a failure shows a red box on the
    common case."""
    with _client() as c:
        assert c.get("/api/session/trace").json() == {"available": False}
        c.post("/api/session/open", content=_build())
        assert c.get("/api/session/trace").json() == {"available": False}
        assert c.get("/api/session/state").json()["trace"]["available"] is False
        c.post("/api/session/close")


def test_reading_a_carried_run_does_not_file_it_in_this_machines_history():
    """A recording is read, never adopted. Importing it would put a stranger's
    run into the store as though it had been captured here."""
    with _client() as c:
        before = {t["id"] for t in c.get("/api/traces").json()}
        c.post("/api/session/open", content=_build(trace=_trace(2, id="t-outside")))
        carried = c.get("/api/session/trace").json()
        assert carried["available"] is True
        after = {t["id"] for t in c.get("/api/traces").json()}
        assert after == before
        # And it is not reachable through the store's own route either, which
        # is the check that would catch a future "just import it" shortcut.
        assert c.get(f"/api/traces/{carried['id']}").status_code == 404
        c.post("/api/session/close")


def test_a_recording_without_attention_says_so_rather_than_going_quiet():
    """`available: False` with no sentence is a panel that vanishes.

    Every branch of `runtime.attention_meta` carries a `reason` and the replay
    branch did not, so a bundle exported for the agent run it carries — which
    has no attention slices — reached the panel as a bare unavailable. The
    panel then removed itself from a page that is otherwise entirely about
    that file, with nothing anywhere saying why.
    """
    empty = session.parse(_build(attention={}))
    meta = empty.attention_meta()
    assert meta["available"] is False
    assert "no attention maps" in meta["reason"]

    # And the ordinary case is untouched: a sentence there would be a caption
    # on a panel that is about to draw the thing it describes.
    full = session.parse(_build())
    assert full.attention_meta()["available"] is True
    assert "reason" not in full.attention_meta()
