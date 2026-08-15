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
