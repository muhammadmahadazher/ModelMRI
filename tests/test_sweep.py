"""The loop that turns a number into a distribution.

The failure this file is mostly about is a sweep that LOOKS complete and is
not: refusals dropped instead of recorded, a mean standing in for a spread, or
an aggregate over rows that do not mean the same thing in every prompt. Each
of those produces a plausible table that describes something other than what
the reader thinks.
"""

from __future__ import annotations

import json
import os

import pytest

from modelmri import sweep
from modelmri.errors import BadRequest, Refusal


def _row(index: int, scores: dict, *, refused: str = "") -> sweep.Row:
    top = [k for k, _ in sorted(scores.items(), key=lambda kv: -kv[1])]
    return sweep.Row(
        index=index,
        prompt_sha256=f"hash{index}",
        scores=scores,
        top=top,
        could_not_measure=refused,
    )


# ------------------------------------------------------------------- the job


def test_a_sweep_with_no_prompts_is_refused():
    with pytest.raises(BadRequest, match="no prompts"):
        sweep.Job(model="Qwen/Qwen3-1.7B", prompts=[]).validated()
    with pytest.raises(BadRequest, match="no prompts"):
        sweep.Job(model="Qwen/Qwen3-1.7B", prompts=["", "   "]).validated()


def test_an_unknown_metric_names_the_ones_that_exist():
    with pytest.raises(BadRequest, match="heads"):
        sweep.Job(model="Qwen/Qwen3-1.7B", prompts=["a"], metric="vibes").validated()


def test_blank_lines_are_dropped_but_real_prompts_are_not():
    job = sweep.Job(model="Qwen/Qwen3-1.7B", prompts=["a", "", "  ", "b"]).validated()
    assert job.prompts == ["a", "b"]


# ---------------------------------------------------------------- prompt file


def test_prompts_load_from_plain_text(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text("first prompt\n\nsecond prompt\n", encoding="utf-8")
    assert sweep.load_prompts(path) == ["first prompt", "second prompt"]


def test_prompts_load_from_jsonl(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text(
        '{"prompt": "first"}\n{"prompt": "second", "extra": 1}\n', encoding="utf-8"
    )
    assert sweep.load_prompts(path) == ["first", "second"]


def test_a_jsonl_line_with_no_prompt_field_is_refused_by_line_number(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text('{"prompt": "ok"}\n{"text": "wrong key"}\n', encoding="utf-8")
    with pytest.raises(BadRequest, match="line 2"):
        sweep.load_prompts(path)


def test_a_broken_jsonl_line_says_which_line(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text('{"prompt": "ok"}\n{not json\n', encoding="utf-8")
    with pytest.raises(BadRequest, match="line 2"):
        sweep.load_prompts(path)


def test_an_empty_file_is_refused_rather_than_swept(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text("\n\n  \n", encoding="utf-8")
    with pytest.raises(BadRequest, match="no prompts"):
        sweep.load_prompts(path)


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(BadRequest, match="could not be read"):
        sweep.load_prompts(tmp_path / "nope.txt")


# --------------------------------------------------------------- aggregating


def test_the_aggregate_is_order_statistics_not_a_mean():
    """A mean over twenty prompts hides the head that carried one of them, and
    that head is usually the interesting one."""
    rows = [
        _row(0, {"L0H1": 2.0, "L0H2": 0.1}),
        _row(1, {"L0H1": 0.1, "L0H2": 0.1}),
        _row(2, {"L0H1": 0.1, "L0H2": 0.1}),
    ]
    stats = {s.key: s for s in sweep.aggregate(rows, metric="heads")}

    spiky = stats["L0H1"]
    assert spiky.median == 0.1, "the median is not dragged by the one big prompt"
    assert spiky.hi == 2.0, "and the outlier is still visible in the range"
    assert spiky.n == 3


def test_n_is_the_prompts_a_head_was_measured_on(tmp_path):
    """Not the prompts in the job. A head that appears in three of twenty
    rankings has n=3, and saying 20 would be a different claim."""
    rows = [
        _row(0, {"L0H1": 1.0, "L0H2": 0.5}),
        _row(1, {"L0H1": 1.0}),  # H2 was not ranked here
        _row(2, {"L0H1": 1.0}),
    ]
    stats = {s.key: s for s in sweep.aggregate(rows, metric="heads")}
    assert stats["L0H1"].n == 3
    assert stats["L0H2"].n == 1


def test_the_top_k_rate_is_over_the_prompts_it_was_measured_on():
    rows = [
        _row(0, {"L0H1": 1.0, "L0H2": 0.5}),
        _row(1, {"L0H1": 1.0, "L0H2": 0.5}),
    ]
    stats = {s.key: s for s in sweep.aggregate(rows, metric="heads")}
    assert stats["L0H1"].top_k_hits == 2
    assert stats["L0H1"].top_k_rate == 1.0


def test_a_single_prompt_reports_no_spread_it_did_not_measure():
    """With one point the median IS the distribution. q1 and q3 are that same
    number, with n=1 beside them, rather than a spread nobody took."""
    stats = sweep.aggregate([_row(0, {"L0H1": 0.5})], metric="heads")
    (only,) = stats
    assert only.n == 1
    assert only.median == only.q1 == only.q3 == 0.5
    assert only.iqr == 0.0


def test_refused_rows_do_not_enter_the_aggregate():
    rows = [
        _row(0, {"L0H1": 1.0}),
        _row(1, {}, refused="the attribution position is a control token"),
    ]
    (stat,) = sweep.aggregate(rows, metric="heads")
    assert stat.n == 1, "a refusal contributes no score"


def test_a_position_metric_is_refused_rather_than_averaged():
    """Rule 3, enforced in code. Position 3 is a different token in every
    prompt, so a median over it is a number about nothing."""
    rows = [_row(0, {"P3' France'": 1.0})]
    with pytest.raises(Refusal, match="different token in every prompt"):
        sweep.aggregate(rows, metric="tokens")


def test_features_do_aggregate_because_a_feature_id_is_stable():
    rows = [_row(0, {"F42": 1.0}), _row(1, {"F42": 0.5})]
    (stat,) = sweep.aggregate(rows, metric="features")
    assert stat.n == 2
    assert stat.median == 0.75


# ---------------------------------------------------------------- the output


def test_a_refusal_is_written_as_a_row_with_its_sentence(tmp_path):
    """If refusals were skipped, the file would quietly describe only the
    prompts that happened to work."""
    rows = [
        _row(0, {"L0H1": 1.0}),
        _row(1, {}, refused="No SAE loaded, so there are no features to remove."),
    ]
    path = sweep.write_jsonl(rows, tmp_path / "out" / "rows.jsonl")
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]

    assert len(lines) == 2, "the refused prompt is a row, not a gap"
    assert lines[1]["could_not_measure"].startswith("No SAE loaded")
    assert lines[1]["scores"] == {}


def test_the_rendered_table_distinguishes_refused_from_unaggregatable(tmp_path):
    """A tokens sweep whose prompts all succeeded said "every prompt was
    refused" — two different facts reported as one."""
    job = sweep.Job(model="Qwen/Qwen3-1.7B", prompts=["a"] * 5, metric="tokens")
    rows = [_row(i, {"P3'x'": 1.0}) for i in range(5)]
    text = sweep.render(job, rows, [])
    assert "5 measured · 0 could not be measured" in text
    assert "every prompt was refused" not in text
    assert "not aggregated across prompts" in text


def test_the_table_says_a_median_is_not_a_mean(tmp_path):
    job = sweep.Job(model="Qwen/Qwen3-1.7B", prompts=["a", "b"])
    rows = [_row(0, {"L0H1": 1.0}), _row(1, {"L0H1": 0.5})]
    text = sweep.render(job, rows, sweep.aggregate(rows, metric="heads"))
    assert "never a mean" in text
    assert "n" in text


# ------------------------------------------------------------- persistence


def test_a_sweep_is_findable_after_the_shell_closes(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELMRI_HOME", str(tmp_path))
    job = sweep.Job(model="Qwen/Qwen3-1.7B", prompts=["a", "b"])
    rows = [_row(0, {"L0H1": 1.0}), _row(1, {}, refused="nope")]
    sweep.save(job, rows, started_at="2026-08-13T00:00:00+00:00", sweep_id="abc123")

    import sqlite3

    from modelmri import paths

    db = sqlite3.connect(str(paths.trace_db_path()))
    try:
        row = db.execute(
            "SELECT model, metric, n_prompts, n_measured, n_refused FROM sweep "
            "WHERE id=?",
            ("abc123",),
        ).fetchone()
    finally:
        db.close()
    assert row == ("Qwen/Qwen3-1.7B", "heads", 2, 1, 1)


# ------------------------------------------------- against a real model


@pytest.fixture(scope="module")
def gpt2():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from modelmri import receipts as _receipts

    if _receipts.revision_of("gpt2")[0] is None and not os.environ.get(
        "MODELMRI_TEST_DOWNLOAD"
    ):
        pytest.skip("gpt2 is not in the local model cache")

    from modelmri.runtime import ModelRuntime

    runtime = ModelRuntime()
    try:
        runtime.load("gpt2")
    except Exception as err:
        pytest.skip(f"gpt2 is not available here: {err}")
    yield runtime
    runtime.unload()


def test_the_projection_is_made_before_anything_runs(gpt2):
    """Cost is N prompts x per-prompt cost, and resample multiplies through
    every row. A sweep about to take an hour should say so beforehand."""
    job = sweep.Job(model="gpt2", prompts=["a"] * 10, layer=0)
    one_layer = sweep.plan(job, gpt2)
    every_layer = sweep.plan(
        sweep.Job(model="gpt2", prompts=["a"] * 10, layer=None), gpt2
    )

    assert one_layer["passes_total"] == one_layer["passes_per_prompt"] * 10
    assert every_layer["passes_per_prompt"] > one_layer["passes_per_prompt"], (
        "sweeping every layer costs more than one, and the projection says so"
    )
    assert every_layer["aggregatable"] is True


def test_the_resample_baseline_shows_its_draws_in_the_projection(gpt2):
    plain = sweep.plan(sweep.Job(model="gpt2", prompts=["a"], layer=0), gpt2)
    drawn = sweep.plan(
        sweep.Job(model="gpt2", prompts=["a"], layer=0, baseline="resample"), gpt2
    )
    assert drawn["passes_per_prompt"] > plain["passes_per_prompt"]


def test_a_real_sweep_reports_a_distribution(gpt2):
    job = sweep.Job(
        model="gpt2",
        prompts=[
            "The capital of France is",
            "The capital of Italy is",
            "The Eiffel Tower is located in the city of",
        ],
        layer=0,
        max_new_tokens=3,
    )
    rows = sweep.run(job, gpt2)
    assert len(rows) == 3
    assert all(r.measured for r in rows), [r.could_not_measure for r in rows]
    # Each row carries the setup that produced it, so a sweep's JSONL is
    # auditable the same way a `.mri` is.
    assert all(r.receipt.get("op") == "ablate_heads" for r in rows)
    assert all(r.prompt_sha256 for r in rows)

    stats = sweep.aggregate(rows, metric="heads")
    assert stats, "gpt2 layer 0 has heads to rank"
    assert all(s.n == 3 for s in stats), "every head was measured on every prompt"
    assert any(s.hi > s.lo for s in stats), (
        "at least one head scores differently across prompts — which is the "
        "entire reason this feature exists"
    )


def test_one_mri_per_prompt_when_asked(gpt2, tmp_path):
    """So a single row of a sweep can be opened, forwarded or verified like
    any other finding."""
    job = sweep.Job(
        model="gpt2",
        prompts=["The capital of France is", "The capital of Italy is"],
        layer=0,
        max_new_tokens=3,
        out_dir=tmp_path / "runs",
    )
    rows = sweep.run(job, gpt2)
    written = sorted((tmp_path / "runs").glob("*.mri"))
    assert len(written) == 2
    assert [r.mri for r in rows] == ["0000.mri", "0001.mri"]

    from modelmri import session

    parsed = session.parse(written[0].read_bytes())
    assert parsed.has_ranking(), "and it carries the ranking that row measured"


# ------------------------- a stopped run is not a short run


class _Counting:
    """A runtime that trips a cancel event partway through."""

    last_n_prompt_tokens = 3

    def __init__(self, cancel, after: int):
        self.cancel, self.after, self.seen = cancel, after, 0

    def generate_stream(self, *a, **k):
        self.seen += 1
        if self.seen >= self.after:
            self.cancel.set()
        return iter(["x"])

    def ablate_heads(self, **k):
        return {"ranked": [{"layer": 0, "head": 0, "kl": 1.0}], "receipt": {}}


def test_a_cancelled_sweep_still_accounts_for_every_prompt():
    """`break` left the unrun prompts with no row and no note, so a sweep
    stopped after 3 of 20 was saved and rendered as a complete 3-prompt sweep.
    The aggregate, the top-k and the "measured on N prompts" line were all
    true of a run nobody meant to take as final.

    The `except` arm a few lines below already states the rule this broke:
    "A ROW, NOT A GAP".
    """
    import threading

    cancel = threading.Event()
    job = sweep.Job(model="m", prompts=[f"p{i}" for i in range(20)], metric="heads")
    rows = sweep.run(job, _Counting(cancel, after=3), cancel=cancel)

    assert len(rows) == 20, "the unrun prompts vanished"
    assert sum(1 for r in rows if r.measured) == 3
    assert [r.index for r in rows] == list(range(20))
    for row in rows[3:]:
        assert "cancelled" in row.could_not_measure


def test_the_cancelled_rows_are_skipped_by_the_aggregate():
    """They must not dilute the statistics they are absent from — `measured`
    is what `aggregate` filters on, and a cancelled row is not measured."""
    import threading

    cancel = threading.Event()
    job = sweep.Job(model="m", prompts=[f"p{i}" for i in range(10)], metric="heads")
    rows = sweep.run(job, _Counting(cancel, after=2), cancel=cancel)

    stats = sweep.aggregate(rows, metric="heads")
    assert stats, "the measured rows still aggregate"
    assert all(s.n <= 2 for s in stats), "a cancelled row was counted as measured"


def test_an_uncancelled_sweep_is_unchanged():
    """The rows must not grow a phantom cancellation when nothing stopped."""
    import threading

    cancel = threading.Event()
    job = sweep.Job(model="m", prompts=["a", "b", "c"], metric="heads")
    rows = sweep.run(job, _Counting(cancel, after=99), cancel=cancel)

    assert len(rows) == 3
    assert all(r.measured for r in rows)


def test_a_head_sweep_on_an_unreadable_config_is_refused_not_under_quoted():
    """`getattr(config, ..., 0) or 0` turns an architecture that names these
    fields differently into a model with no layers and no heads, and the
    projection then quotes a whole-model head sweep at 2 passes per prompt
    instead of n_layers x n_heads + 2 — 2 against 450 on Qwen3-0.6B.

    `model_diff.head_pass_estimate` already refuses exactly this, and says
    why: "A preflight that under-quotes is worse than no preflight, because
    it is the number somebody plans around."
    """

    class Runtime:
        model = type("M", (), {"config": type("C", (), {})()})()

    job = sweep.Job(model="m", prompts=["a", "b"], metric="heads")
    with pytest.raises(Refusal) as caught:
        sweep.plan(job, Runtime())
    said = str(caught.value)
    assert "does not state how many layers" in said
    assert "cannot be projected" in said


def test_a_token_sweep_needs_no_head_counts():
    """The refusal is scoped to the metric that depends on them."""

    class Runtime:
        model = type("M", (), {"config": type("C", (), {})()})()

    job = sweep.Job(model="m", prompts=["a", "b"], metric="tokens")
    out = sweep.plan(job, Runtime())
    assert out["passes_total"] > 0


def test_a_file_that_is_not_utf8_is_refused_rather_than_500ing(tmp_path):
    """`except OSError` does not catch `UnicodeDecodeError` — it is a
    `ValueError`.

    So pointing `file` at a `.safetensors` answered 500 on `/api/diff/models`,
    `/api/ground`, `/api/lens/tune` and `/api/features/evidence`, and printed a
    raw traceback in the terminal, while a malformed `.jsonl` line one branch
    down correctly answered 422. One wrong file type, two entirely different
    answers, depending on which byte offended.

    `datasets.py` already handles this with a test pinning it; `load_prompts`
    was missed by that pass.
    """
    binary = tmp_path / "model.safetensors"
    binary.write_bytes(b"\x00\x01\x02\xff\xfe" * 64)

    with pytest.raises(BadRequest) as caught:
        sweep.load_prompts(binary)

    said = caught.value.sentence
    assert "not UTF-8" in said
    assert ".txt" in said and ".jsonl" in said, "name what it does read"


def test_a_missing_file_still_says_it_is_missing(tmp_path):
    """The arm that already worked, so the new one cannot swallow it."""
    with pytest.raises(BadRequest, match="could not be read"):
        sweep.load_prompts(tmp_path / "nope.txt")
