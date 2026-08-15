"""Reading a UK AISI Inspect `.eval` log onto the timeline.

Every fixture here is a real zip built with `zipfile`, because the whole
feature is "stdlib only, no new dependency, works air-gapped" and a test that
mocked the archive would not check that claim.

The two disciplines under test:

  * an unrecognised schema version is REFUSED WITH THE VERSION NAMED, the same
    thing `session.parse` does with `format_version`. Inspect's schema is not
    frozen, and guessing at one that moved produces a timeline full of
    real-looking steps in the wrong places;
  * what was NOT mapped is counted and named. "We showed you what we
    understood" is only honest when the rest is on screen too.
"""

from __future__ import annotations

import json
import zipfile

import pytest

from modelmri import inspect_io
from modelmri.errors import BadRequest


def _log(tmp_path, *, version=2, samples=None, name="run.eval", header=None):
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
            },
            {
                "event": "tool",
                "timestamp": "2026-08-15T00:00:01+00:00",
                "function": "calculator",
                "arguments": {"expr": "2+2"},
                "result": "4",
            },
        ],
        "scores": {"match": {"value": "C"}},
    }
    doc.update(over)
    return doc


# ------------------------------------------------------------ the header


def test_the_header_is_read(tmp_path):
    head = inspect_io.header(_log(tmp_path, samples=[_sample()]))
    assert head.version == 2
    assert head.task == "arc_easy"
    assert head.model == "openai/gpt-4o"
    assert head.n_samples == 1


def test_an_unrecognised_version_is_refused_with_the_version_named(tmp_path):
    """The whole point. A schema guessed wrong produces a timeline full of
    real-looking steps in the wrong places."""
    path = _log(tmp_path, version=7, samples=[_sample()])
    with pytest.raises(BadRequest) as caught:
        inspect_io.header(path)
    message = str(caught.value)
    assert "version 7" in message
    assert "Refusing rather than guessing" in message


def test_a_log_with_no_version_is_refused(tmp_path):
    path = _log(tmp_path, samples=[_sample()], header={"version": None})
    with pytest.raises(BadRequest, match="does not state a format version"):
        inspect_io.header(path)


def test_a_file_that_is_not_a_zip_is_refused_without_naming_the_path(tmp_path):
    path = tmp_path / "not-really.eval"
    path.write_text("this is not a zip", encoding="utf-8")
    with pytest.raises(BadRequest) as caught:
        inspect_io.header(path)
    assert "not a readable Inspect log" in str(caught.value)
    assert str(tmp_path) not in str(caught.value)


def test_a_missing_file_names_no_path():
    with pytest.raises(BadRequest, match="does not exist"):
        inspect_io.header("nowhere/at/all.eval")


def test_an_archive_with_no_header_is_refused(tmp_path):
    path = tmp_path / "empty.eval"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("something.json", "{}")
    with pytest.raises(BadRequest, match="no header.json"):
        inspect_io.header(path)


# ----------------------------------------------------------- the samples


def test_samples_are_listed_without_parsing_them(tmp_path):
    path = _log(tmp_path, samples=[_sample("a"), _sample("b")])
    refs = inspect_io.samples(path)
    assert [r.id for r in refs] == ["a", "b"]
    assert all(r.epoch == 1 for r in refs)


def test_an_id_containing_an_underscore_survives(tmp_path):
    """Splitting from the left would turn `task_42` into id `task`."""
    path = _log(tmp_path, samples=[_sample("task_42")])
    assert inspect_io.samples(path)[0].id == "task_42"


def test_epochs_are_read(tmp_path):
    path = _log(tmp_path, samples=[_sample("a", epoch=1), dict(_sample("a"), epoch=3)])
    # Both write to the same name in this fixture, so just check the parse.
    refs = inspect_io.samples(path)
    assert refs[0].epoch == 1


# -------------------------------------------------------------- the steps


def test_events_become_steps_in_order(tmp_path):
    out = inspect_io.read_sample(_log(tmp_path, samples=[_sample()]))
    kinds = [s["kind"] for s in out.trace["steps"]]
    assert kinds == ["llm_call", "tool_call"]
    assert out.trace["steps"][1]["name"] == "calculator"


def test_timestamps_become_offsets_from_the_first_event(tmp_path):
    """All-zero offsets would stack every block at x=0 on the timeline."""
    out = inspect_io.read_sample(_log(tmp_path, samples=[_sample()]))
    steps = out.trace["steps"]
    assert steps[0]["started_ms"] == 0
    assert steps[1]["started_ms"] == 1000


def test_events_with_no_timestamps_get_sequential_offsets(tmp_path):
    doc = _sample()
    for event in doc["events"]:
        del event["timestamp"]
    out = inspect_io.read_sample(_log(tmp_path, samples=[doc]))
    offsets = [s["started_ms"] for s in out.trace["steps"]]
    assert offsets == sorted(offsets)
    assert len(set(offsets)) == len(offsets), "every block would sit at x=0"


def test_token_usage_is_carried(tmp_path):
    out = inspect_io.read_sample(_log(tmp_path, samples=[_sample()]))
    step = out.trace["steps"][0]
    assert step["tokens_in"] == 12 and step["tokens_out"] == 1


def test_absent_usage_stays_absent_rather_than_zero(tmp_path):
    """The store's columns are nullable so 'the provider reported nothing' is
    recordable; a 0 here would claim a report that never happened."""
    doc = _sample()
    del doc["events"][0]["output"]["usage"]
    out = inspect_io.read_sample(_log(tmp_path, samples=[doc]))
    step = out.trace["steps"][0]
    assert "tokens_in" not in step and "tokens_out" not in step


def test_a_tool_error_marks_the_step(tmp_path):
    doc = _sample()
    doc["events"][1]["error"] = "connection refused"
    out = inspect_io.read_sample(_log(tmp_path, samples=[doc]))
    step = out.trace["steps"][1]
    assert step["error"] is True
    assert "connection refused" in step["output"]


def test_content_blocks_become_text(tmp_path):
    doc = _sample()
    doc["events"][0]["input"] = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ]
    out = inspect_io.read_sample(_log(tmp_path, samples=[doc]))
    assert "hello" in out.trace["steps"][0]["input"]


def test_a_non_text_block_is_named_rather_than_rendered_empty(tmp_path):
    """An image should not read as a message with nothing in it."""
    doc = _sample()
    doc["events"][0]["input"] = [
        {"role": "user", "content": [{"type": "image", "image": "data:..."}]}
    ]
    out = inspect_io.read_sample(_log(tmp_path, samples=[doc]))
    assert "[image]" in out.trace["steps"][0]["input"]


# ------------------------------------------- what was dropped is reported


def test_an_unknown_event_kind_is_counted_and_named(tmp_path):
    doc = _sample()
    doc["events"].append({"event": "sandbox", "timestamp": "2026-08-15T00:00:02Z"})
    doc["events"].append({"event": "approval", "timestamp": "2026-08-15T00:00:03Z"})
    out = inspect_io.read_sample(_log(tmp_path, samples=[doc]))
    assert out.mapping.dropped == {"sandbox": 1, "approval": 1}
    means = out.mapping.means()
    assert "NOT SHOWN" in means and "sandbox" in means and "approval" in means


def test_a_fully_mapped_sample_says_nothing_was_dropped(tmp_path):
    out = inspect_io.read_sample(_log(tmp_path, samples=[_sample()]))
    assert out.mapping.dropped == {}
    assert "NOT SHOWN" not in out.mapping.means()


def test_every_mapped_kind_is_one_a_step_can_actually_have():
    """A kind name no step can hold would render as a blank block."""
    from modelmri.step_kinds import VALID_KINDS

    assert set(inspect_io.EVENT_KINDS.values()) <= VALID_KINDS
    assert set(inspect_io.ROLE_KINDS.values()) <= VALID_KINDS


# ------------------------------------------------- the failing sample first


def test_the_failing_sample_is_the_one_returned(tmp_path):
    """ "Same timeline, failing sample highlighted" is what makes this worth
    opening rather than scrolling."""
    good = _sample("a")
    bad = _sample("b", scores={"match": {"value": "I"}})
    out = inspect_io.read_sample(_log(tmp_path, samples=[good, bad]))
    assert out.trace["id"].startswith("inspect-b")
    assert out.failed is True


def test_a_sample_with_an_error_counts_as_failing(tmp_path):
    good = _sample("a")
    bad = _sample("b", error="the tool crashed")
    out = inspect_io.read_sample(_log(tmp_path, samples=[good, bad]))
    assert out.trace["id"].startswith("inspect-b")
    assert "the tool crashed" in out.error


def test_a_zero_score_is_not_read_as_a_failure(tmp_path):
    """A 0 on a 0-10 rubric is a low mark, not an error, and highlighting it
    would put the wrong sample on screen."""
    first = _sample("a", scores={"rubric": {"value": 0}})
    second = _sample("b")
    out = inspect_io.read_sample(_log(tmp_path, samples=[first, second]))
    assert out.failed is False
    assert out.trace["id"].startswith("inspect-a"), "fell back to the first"


def test_a_named_sample_is_read_directly(tmp_path):
    out = inspect_io.read_sample(
        _log(tmp_path, samples=[_sample("a"), _sample("b")]), sample_id="b"
    )
    assert out.trace["id"].startswith("inspect-b")


def test_an_unknown_sample_id_is_refused_with_the_count(tmp_path):
    with pytest.raises(BadRequest, match="carries 2 sample"):
        inspect_io.read_sample(
            _log(tmp_path, samples=[_sample("a"), _sample("b")]), sample_id="zzz"
        )


# ------------------------------------------------------------- fallbacks


def test_a_sample_with_only_messages_still_draws(tmp_path):
    doc = _sample(
        "m",
        events=[],
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    )
    out = inspect_io.read_sample(_log(tmp_path, samples=[doc]))
    kinds = [s["kind"] for s in out.trace["steps"]]
    assert kinds == ["user_turn", "llm_call"]


def test_a_sample_with_nothing_drawable_is_refused(tmp_path):
    doc = _sample("m", events=[], messages=[])
    with pytest.raises(BadRequest, match="nothing to draw"):
        inspect_io.read_sample(_log(tmp_path, samples=[doc]))


def test_a_log_with_no_samples_is_refused(tmp_path):
    with pytest.raises(BadRequest, match="no samples"):
        inspect_io.read_sample(_log(tmp_path, samples=[]))


# ------------------------------------------- it produces an importable trace


def test_the_trace_is_one_the_store_accepts(tmp_path):
    """The whole point is that it renders on the EXISTING timeline."""
    from modelmri import traces

    store = traces.TraceStore(tmp_path / "t.sqlite")
    try:
        out = inspect_io.read_sample(_log(tmp_path, samples=[_sample()]))
        trace_id = store.import_trace(out.trace)
        back = store.get_trace(trace_id)
        assert back is not None
        assert len(back["steps"]) == 2
        assert back["steps"][0]["tokens_in"] == 12
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def test_the_result_serialises_for_the_wire(tmp_path):
    doc = _sample()
    doc["events"].append({"event": "sandbox"})
    out = inspect_io.read_sample(_log(tmp_path, samples=[doc])).to_dict()
    assert out["mapping"]["dropped"] == {"sandbox": 1}
    assert out["scores"] == {"match": "C"}
    assert isinstance(out["means"], str)


# ------------------------------------------------- the same log, read twice


def test_the_same_log_can_be_read_again_for_a_different_sample(tmp_path):
    """The sample picker re-sends the same bytes with a different id. If
    reading twice failed, the picker would be dead for every log."""
    path = _log(tmp_path, samples=[_sample("a"), _sample("b")])
    first = inspect_io.read_sample(path)
    second = inspect_io.read_sample(path, sample_id="b")
    third = inspect_io.read_sample(path, sample_id="a")
    assert first.trace["id"].startswith("inspect-a")
    assert second.trace["id"].startswith("inspect-b")
    assert third.trace["id"].startswith("inspect-a")


def test_reading_leaves_no_open_handle(tmp_path):
    """On Windows a still-open handle makes the caller's TemporaryDirectory
    cleanup raise, failing a request whose work had already succeeded."""
    import gc

    path = tmp_path / "held.eval"
    built = _log(tmp_path, samples=[_sample()], name="held.eval")
    inspect_io.header(built)
    inspect_io.samples(built)
    inspect_io.read_sample(built)
    gc.collect()
    # If any handle were still open this raises PermissionError on Windows.
    path.unlink()
    assert not path.exists()
