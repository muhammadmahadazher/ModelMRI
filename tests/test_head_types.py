"""A detector that labels everything is as useless as one with no null.

The value of this module is entirely in what it REFUSES to label, so most of
these tests are about the gates rather than the scores. Each gate exists
because the ones before it were measured on gpt2 and found insufficient.
"""

from __future__ import annotations

import json
import os

import pytest

from modelmri import head_types as ht
from modelmri.errors import Refusal


class _Tokenizer:
    def __init__(self, vocab_size, special=()):
        self.vocab_size = vocab_size
        self.all_special_ids = list(special)


# ------------------------------------------------------------- sampling


def test_a_byte_level_vocabulary_is_refused_not_approximated():
    """A repeat of a random byte is not the event these detectors name."""
    with pytest.raises(Refusal, match="byte-level or character-level"):
        ht._sampleable(_Tokenizer(256))


def test_a_vocabulary_that_is_almost_all_special_tokens_is_refused():
    with pytest.raises(Refusal, match="special token"):
        ht._sampleable(_Tokenizer(1200, special=range(1200)))


def test_special_tokens_are_never_sampled():
    usable = ht._sampleable(_Tokenizer(5000, special=[0, 1, 2]))
    assert 0 not in usable and 1 not in usable and 2 not in usable
    assert len(usable) == 4997


def test_a_repeated_sequence_really_repeats_and_a_null_one_does_not():
    pytest.importorskip("torch")
    usable = list(range(3000))
    repeated = ht._sequences(usable, seq_len=8, count=2, repeat=True, seed=0)
    fresh = ht._sequences(usable, seq_len=8, count=2, repeat=False, seed=0)

    assert repeated.shape == (2, 16)
    for row in repeated:
        assert row[:8].tolist() == row[8:].tolist()
    for row in fresh:
        assert row[:8].tolist() != row[8:].tolist()


def test_a_null_sequence_has_no_repeated_token_inside_it():
    """A draw that happened to repeat would put an induction target into the
    null and quietly raise it -- the one thing the null must not contain."""
    pytest.importorskip("torch")
    fresh = ht._sequences(list(range(3000)), seq_len=12, count=4, repeat=False, seed=7)
    for row in fresh:
        assert len(set(row.tolist())) == len(row), "no token appears twice"


def test_the_same_seed_gives_the_same_sequences():
    pytest.importorskip("torch")
    a = ht._sequences(list(range(3000)), seq_len=8, count=3, repeat=True, seed=5)
    b = ht._sequences(list(range(3000)), seq_len=8, count=3, repeat=True, seed=5)
    assert a.tolist() == b.tolist()


# ------------------------------------------------------------- the offsets


def test_each_pattern_points_at_the_position_it_names():
    offsets = ht._offsets(index=30, seq_len=24)
    assert offsets["duplicate-token"] == 6, "the earlier copy of this token"
    assert offsets["induction"] == 7, "the token AFTER the earlier copy"
    assert offsets["previous-token"] == 29
    assert offsets["sink"] == 0


def test_the_two_nulls_are_assigned_by_whether_repetition_matters():
    """A previous-token head attends to i-1 whether or not anything repeats,
    so a non-repeating null for it is just the same number again."""
    assert ht.NULL_KIND["induction"] == "repeat"
    assert ht.NULL_KIND["duplicate-token"] == "repeat"
    assert ht.NULL_KIND["previous-token"] == "chance"
    assert ht.NULL_KIND["sink"] == "chance"


# ------------------------------------------------------ against a real model


@pytest.fixture(scope="module")
def report():
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
    try:
        yield ht.label_heads(
            runtime.model, runtime.tokenizer, seq_len=24, n_sequences=6
        )
    finally:
        runtime.unload()


def test_it_finds_the_induction_heads_gpt2_is_known_for(report):
    """L5H1 and L5H5 are the canonical gpt2-small induction heads in the
    literature. A detector that missed them would be measuring something
    else."""
    induction = {(r.layer, r.head) for r in report.named if r.label == "induction"}
    assert (5, 1) in induction
    assert (5, 5) in induction
    assert (6, 9) in induction


def test_the_strongest_induction_head_dominates_its_own_attention(report):
    strongest = max(
        (r for r in report.named if r.label == "induction"),
        key=lambda r: r.scores["induction"],
    )
    assert strongest.scores["induction"] > 0.8, "not a marginal effect"
    assert strongest.times_chance > 10


def test_some_heads_get_no_label_at_all(report):
    """No type detected is a result, not a gap. A detector that labels every
    head has stopped being a detector."""
    assert report.counts()["no type detected"] > 0


def test_every_label_is_the_heads_own_peak_target(report):
    """A type label claims this is what the head looks at. Choosing by margin
    instead labelled L5H8 induction at 0.089 while 70% of its attention sat on
    position 0."""
    for row in report.named:
        assert row.scores[row.label] >= row.peak - 1e-9


def test_every_label_beats_chance_under_the_causal_mask(report):
    """Significance without effect size labelled a head that put 0.15x chance
    on the induction offset."""
    for row in report.named:
        assert row.times_chance >= ht.MIN_TIMES_CHANCE


def test_an_unlabelled_head_has_no_margin_rather_than_a_zero_one(report):
    """0.0 would read as exactly at the null, which is a measurement. None is
    the absence of one."""
    for row in report.labels:
        if row.label is None:
            assert row.margin is None
            assert row.times_chance is None


def test_the_label_carries_which_null_it_cleared(report):
    """They are not interchangeable and a reader comparing two labels needs to
    know they were not gated the same way."""
    for row in report.named:
        assert row.null_kind in ("repeat", "chance")
        assert row.null_kind == ht.NULL_KIND[row.label]


def test_the_report_says_these_are_not_claims_about_real_text(report):
    means = report.means()
    assert "REPEATED RANDOM TOKENS" in means
    assert "not be read as explaining the ablation ranking" in means


def test_a_label_held_by_most_heads_is_flagged_as_a_model_fact(report):
    """gpt2 attends to the first token throughout, so most of its heads have
    position 0 as their peak -- true, and useless read as these are special."""
    counts = report.counts()
    dominant = [p for p in ht.PATTERNS if counts[p] > len(report.labels) / 2]
    if dominant:
        assert "fact about the model" in report.means()


def test_the_report_survives_json(report):
    out = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert out["counts"]["no type detected"] >= 0
    assert len(out["labels"]) == out["n_layers"] * out["n_heads"]
