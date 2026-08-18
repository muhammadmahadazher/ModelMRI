"""Finishing a sweep that stopped, and the three ways that is WRONG.

`sweep.save` has existed since the sweep did and nothing ever read it back, so
a saved sweep was write-only: in the database, and unreachable from the tool
that wrote it. A sweep that died at prompt 180 of 200 started over, and losing
four hours to a sleeping laptop is a worse failure than any missing feature.

The interesting half is not the resuming. It is the refusing: a resume that
merges rows from two different prompt sets, or two different models, produces
one table of numbers that looks exactly like a table from one run.
"""

from __future__ import annotations

import pytest

from modelmri import receipts, sweep
from modelmri.errors import BadRequest


@pytest.fixture
def stopped(tmp_path, monkeypatch):
    """A four-prompt sweep that measured two and was cancelled on the rest."""
    monkeypatch.setattr(
        "modelmri.paths.trace_db_path", lambda: tmp_path / "traces.sqlite3"
    )
    job = sweep.Job(model="m/x", prompts=["a", "b", "c", "d"], metric="heads")
    rows = [
        sweep.Row(index=0, prompt_sha256=receipts.digest("a"), scores={"0.1": 1.0}),
        sweep.Row(index=1, prompt_sha256=receipts.digest("b"), scores={"0.1": 2.0}),
        sweep.Row(
            index=2,
            prompt_sha256=receipts.digest("c"),
            could_not_measure="the sweep was cancelled before this prompt ran",
        ),
        sweep.Row(
            index=3,
            prompt_sha256=receipts.digest("d"),
            could_not_measure="the sweep was cancelled before this prompt ran",
        ),
    ]
    sweep.save(job, rows, started_at="2026-08-18T00:00:00Z", sweep_id="s1")
    return job, rows


def test_a_cancelled_prompt_is_not_a_result(stopped):
    """ "The sweep was cancelled before this prompt ran" is a reason to try
    again, not an outcome. A resume that treated it as done would report a
    partial sweep as finished."""
    job, rows = sweep.load_sweep("s1")
    assert sweep.remaining(job, rows) == [2, 3]


def test_resuming_keeps_the_measurements_and_reruns_only_the_rest(stopped, monkeypatch):
    """The merge, which is the new code. `run` numbers rows from 0 over the
    job it is given, so without restoring the original indices every resumed
    row would claim to be prompt 0..n of the original set and the join back
    would be silently wrong."""
    given = {}

    def fake_run(job, runtime, **kwargs):
        given["prompts"] = list(job.prompts)
        return [
            sweep.Row(index=i, prompt_sha256=receipts.digest(p), scores={"9.9": 42.0})
            for i, p in enumerate(job.prompts)
        ]

    # `resume` prices the remainder now, and `plan` needs a real runtime; the
    # projection has its own test below.
    monkeypatch.setattr(sweep, "plan", lambda job, runtime: {})
    monkeypatch.setattr(sweep, "run", fake_run)
    job, merged = sweep.resume("s1", object())

    assert given["prompts"] == ["c", "d"], "it re-ran prompts it had already done"
    assert [r.index for r in merged] == [0, 1, 2, 3]
    # The old rows survived untouched...
    assert merged[0].scores == {"0.1": 1.0}
    # ...and the new ones carry the ORIGINAL index, not 0 and 1.
    assert merged[2].index == 2
    assert merged[3].index == 3
    for row in merged:
        assert row.prompt_sha256 == receipts.digest(job.prompts[row.index])


def test_an_edited_prompt_blocks_the_resume_by_index(stopped):
    """Checked by DIGEST, not by count. A set with the same number of prompts
    and one of them edited would otherwise attach the old row to the new
    prompt, and every number in it would be about text that is no longer
    there."""
    edited = sweep.Job(model="m/x", prompts=["a", "EDITED", "c", "d"], metric="heads")
    sweep.save(edited, stopped[1], started_at="2026-08-18T00:01:00Z", sweep_id="s2")

    plan = sweep.resume_plan("s2")
    assert plan["blocked"]
    assert "prompt 1" in plan["blocked"]

    with pytest.raises(BadRequest, match="cannot be resumed"):
        sweep.resume("s2", object())


def test_a_different_model_blocks_the_resume(stopped):
    """Finishing it would put two models' numbers in one table, which looks
    exactly like one model's."""

    class _Other:
        hf_id = "other/model"

    plan = sweep.resume_plan("s1", _Other())
    assert plan["blocked"]
    assert "m/x" in plan["blocked"]
    assert "other/model" in plan["blocked"]


def test_the_same_model_does_not_block_it(stopped):
    """The guard must not fire on the ordinary case."""

    class _Same:
        hf_id = "m/x"

    assert sweep.resume_plan("s1", _Same())["blocked"] is None


def test_a_sweep_that_is_not_here_is_refused_by_id(stopped):
    with pytest.raises(BadRequest, match="no saved sweep"):
        sweep.load_sweep("definitely-not-a-sweep")


def test_a_finished_sweep_reports_nothing_left(stopped, monkeypatch):
    """And resuming it is a no-op that returns what was already there rather
    than re-running anything."""
    job = sweep.Job(model="m/x", prompts=["a", "b"], metric="heads")
    rows = [
        sweep.Row(index=0, prompt_sha256=receipts.digest("a"), scores={"0.1": 1.0}),
        sweep.Row(index=1, prompt_sha256=receipts.digest("b"), scores={"0.1": 2.0}),
    ]
    sweep.save(job, rows, started_at="2026-08-18T00:02:00Z", sweep_id="done")

    def _must_not_run(*a, **k):
        raise AssertionError("a finished sweep re-ran a prompt")

    monkeypatch.setattr(sweep, "plan", lambda job, runtime: {})
    monkeypatch.setattr(sweep, "run", _must_not_run)
    assert sweep.resume_plan("done")["n_remaining"] == 0
    _, merged = sweep.resume("done", object())
    assert len(merged) == 2


def test_the_listing_says_how_far_each_one_got(stopped):
    """`n_remaining` is the number that decides whether resuming is worth
    anything, so the list carries it rather than making a reader subtract."""
    found = {s["sweep_id"]: s for s in sweep.saved_sweeps()}
    assert found["s1"]["n_remaining"] == 2
    assert found["s1"]["complete"] is False


# ------------------------------- what the review found after the first pass


def test_finishing_a_sweep_updates_what_the_database_advertises(stopped, monkeypatch):
    """It persisted NOTHING. `saved_sweeps` kept reporting `n_remaining: 2`
    for a sweep that had just been finished, so the listing invited a second
    resume that would re-run the same prompts."""
    monkeypatch.setattr(sweep, "plan", lambda job, runtime: {})
    monkeypatch.setattr(
        sweep,
        "run",
        lambda job, runtime, **kw: [
            sweep.Row(index=i, prompt_sha256=receipts.digest(p), scores={"9.9": 1.0})
            for i, p in enumerate(job.prompts)
        ],
    )
    before = {s["sweep_id"]: s for s in sweep.saved_sweeps()}["s1"]
    assert before["n_remaining"] == 2

    sweep.resume("s1", object())

    after = {s["sweep_id"]: s for s in sweep.saved_sweeps()}["s1"]
    assert after["n_remaining"] == 0
    assert after["complete"] is True
    # And the sweep still began when it began. Overwriting this would reorder
    # the listing, putting a sweep started Monday and finished Friday above
    # one that ran wholly on Thursday.
    assert after["started_at"] == before["started_at"]


def test_a_resume_is_priced_like_a_fresh_run(stopped, monkeypatch):
    """A resume that skipped the projection let somebody finish a very long
    remainder without ever being shown the number — the one thing `plan`
    exists to prevent, bypassed by the path most likely to be long."""
    priced = {}
    monkeypatch.setattr(
        sweep,
        "plan",
        lambda job, runtime: priced.setdefault("prompts", list(job.prompts)),
    )
    monkeypatch.setattr(
        sweep,
        "run",
        lambda job, runtime, **kw: [
            sweep.Row(index=i, prompt_sha256=receipts.digest(p), scores={"9.9": 1.0})
            for i, p in enumerate(job.prompts)
        ],
    )
    sweep.resume("s1", object())
    assert priced["prompts"] == ["c", "d"], "the remainder was not priced"


def test_a_resume_does_not_overwrite_the_first_prompts_analysis(
    stopped, monkeypatch, tmp_path
):
    """`run` names one `.mri` per prompt by POSITION. A resume handing it the
    remainder would write prompt 2 as position 0, overwriting the first
    prompt's file and leaving two rows pointing at one analysis."""
    handed = {}
    monkeypatch.setattr(sweep, "plan", lambda job, runtime: {})

    def fake_run(job, runtime, **kw):
        handed["out_dir"] = job.out_dir
        return [
            sweep.Row(index=i, prompt_sha256=receipts.digest(p), scores={"9.9": 1.0})
            for i, p in enumerate(job.prompts)
        ]

    monkeypatch.setattr(sweep, "run", fake_run)

    job, rows = sweep.load_sweep("s1")
    sweep.save(
        sweep.Job(**{**__import__("dataclasses").asdict(job), "out_dir": tmp_path}),
        rows,
        started_at="2026-08-18T00:00:00Z",
        sweep_id="s1",
    )
    sweep.resume("s1", object())
    assert handed["out_dir"] is None


def test_a_negative_row_index_is_refused_rather_than_read_backwards(stopped):
    """Python indexes backwards from a negative, so -1 would quietly read the
    LAST prompt and compare its digest against a row about the first."""
    job, rows = sweep.load_sweep("s1")
    rows[2] = sweep.Row(index=-1, prompt_sha256="x", could_not_measure="cancelled")
    sweep.save(job, rows, started_at="2026-08-18T00:03:00Z", sweep_id="neg")

    blocked = sweep.resume_plan("neg")["blocked"]
    assert blocked and "-1" in blocked
    assert "backwards" in blocked


def test_an_unmeasured_row_past_the_end_still_blocks_it(stopped):
    """The guard only looked at MEASURED rows, so a shortened prompt set went
    through: the unmeasured row past the end was dropped by `remaining` and
    the resume ran a set that no longer matched what was saved."""
    job, rows = sweep.load_sweep("s1")
    shorter = sweep.Job(
        **{**__import__("dataclasses").asdict(job), "prompts": ["a", "b"]}
    )
    sweep.save(shorter, rows, started_at="2026-08-18T00:04:00Z", sweep_id="short")

    blocked = sweep.resume_plan("short")["blocked"]
    assert blocked and "only 2 prompt(s)" in blocked


def test_a_sweep_from_a_different_version_is_a_refusal_not_a_500(stopped, tmp_path):
    """`Job(**raw)` raises TypeError on both an unexpected key and a missing
    one, and that reached the route as an unexplained 500 — a fact about the
    FILE reported as a fault in the server."""
    import json
    import sqlite3

    from modelmri import paths

    db = sqlite3.connect(str(paths.trace_db_path()))
    db.execute(
        "UPDATE sweep SET job = ? WHERE id = 's1'",
        (json.dumps({"model": "m/x", "prompts": ["a"], "a_field_from_the_future": 1}),),
    )
    db.commit()
    db.close()

    with pytest.raises(BadRequest, match="different version of ModelMRI"):
        sweep.load_sweep("s1")
