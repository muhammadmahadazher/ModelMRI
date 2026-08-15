"""The third baseline, and the number that says the baseline is deciding.

`ablate.py` has documented since it was written that zero and mean disagree —
on gpt2 layer 0, zero ranks heads 7/10/9 and mean ranks 3/1/10, so head 7 goes
from first to sixth. It documented it and did nothing, which means every
ranking the tool has ever shown was one of several answers with no indication
that the others existed.

Resampling adds the on-distribution baseline: replace a head with what it
really does compute on some other sentence. Its failure modes are different
from the other two and worse, because a donor that does not fit can be made to
fit by padding, truncating or falling back — and all three produce a number
that looks exactly like a resample score and is not one. Every one of those is
a refusal here.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from modelmri import ablate  # noqa: E402
from modelmri.errors import BadRequest  # noqa: E402

WIDTH, HEADS, VOCAB, HEAD_DIM = 4, 2, 6, 2


class ToyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.o_proj = torch.nn.Linear(WIDTH, VOCAB, bias=False)


class ToyModel(torch.nn.Module):
    """Small but real: the pre-hook fires on an actual Linear's input.

    A stub that never calls the projection would let a broken hook pass, which
    is the one thing these tests exist to catch.
    """

    def __init__(self, layers: int = 1) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(VOCAB, WIDTH)
        self.blocks = torch.nn.ModuleList([ToyBlock() for _ in range(layers)])
        # estimate_cost reads vocab_size to price the retained base
        # distribution, so the toy needs the same surface a real config has.
        self.config = type("Cfg", (), {"vocab_size": VOCAB})()

    def forward(self, ids):
        x = self.embed(ids)
        out = None
        for block in self.blocks:
            out = block.self_attn.o_proj(x)
        return type("Out", (), {"logits": out})()

    __call__ = forward


def donors_from(model, seqs, layers):
    return [
        ablate.capture_projection_inputs(model, lambda i: model.blocks[i], s, layers)
        for s in seqs
    ]


# ------------------------------------------------------------- the mechanism


def test_resample_is_position_matched_not_broadcast():
    """Token i is replaced by the donor's token i.

    Broadcasting one position across the sequence replaces a varying signal
    with a constant, which is what `mean` already does — the two baselines
    would quietly measure the same thing.
    """
    x = torch.zeros(1, 3, WIDTH)
    donor = torch.arange(12.0).reshape(3, WIDTH)
    (out,) = ablate._cut(0, HEAD_DIM, "resample", donor)(None, (x,))

    assert out[0, 0, 0] == 0.0
    assert out[0, 1, 0] == 4.0  # donor row 1, not row 0 repeated
    assert out[0, 2, 0] == 8.0
    # Head 1's columns are untouched.
    assert torch.equal(out[0, :, HEAD_DIM:], torch.zeros(3, HEAD_DIM))


def test_a_longer_donor_is_truncated_from_the_front_not_sampled():
    x = torch.zeros(1, 2, WIDTH)
    donor = torch.arange(20.0).reshape(5, WIDTH)
    (out,) = ablate._cut(0, HEAD_DIM, "resample", donor)(None, (x,))
    assert out[0, 0, 0] == 0.0 and out[0, 1, 0] == 4.0


def test_capture_returns_the_tensor_cut_slices():
    model = ToyModel(layers=2)
    ids = torch.tensor([[1, 2, 3]])
    captured = ablate.capture_projection_inputs(
        model, lambda i: model.blocks[i], ids, [0, 1]
    )
    assert set(captured) == {0, 1}
    for layer in (0, 1):
        assert captured[layer].shape == (3, WIDTH)
    assert torch.allclose(captured[0], model.embed(ids)[0])


def test_capture_removes_every_hook_it_installed():
    model = ToyModel(layers=2)
    ids = torch.tensor([[1, 2]])
    ablate.capture_projection_inputs(model, lambda i: model.blocks[i], ids, [0, 1])
    for block in model.blocks:
        assert not block.self_attn.o_proj._forward_pre_hooks


# ---------------------------------------------------------------- refusals


def test_resample_without_a_corpus_is_refused():
    model = ToyModel()
    with pytest.raises(BadRequest, match="needs a corpus"):
        ablate.rank_heads(
            model,
            lambda i: model.blocks[i],
            torch.tensor([[1, 2]]),
            position=0,
            layers=[0],
            n_heads=HEADS,
            baseline="resample",
        )


def test_a_donor_shorter_than_the_prompt_is_refused_with_both_lengths():
    """Padding it out would score the padding. Falling back to mean would
    return a different measurement under the name of this one."""
    model = ToyModel()
    donors = donors_from(model, [torch.tensor([[1, 2]])], [0])
    with pytest.raises(ablate.AblationError, match="2 tokens and this prompt is 4"):
        ablate.rank_heads(
            model,
            lambda i: model.blocks[i],
            torch.tensor([[1, 2, 3, 4]]),
            position=0,
            layers=[0],
            n_heads=HEADS,
            baseline="resample",
            donors=donors,
            corpus="toy",
        )


def test_a_donor_missing_the_layer_is_refused():
    model = ToyModel(layers=2)
    donors = donors_from(model, [torch.tensor([[1, 2, 3]])], [0])  # layer 1 absent
    with pytest.raises(ablate.AblationError, match="no capture for layer 1"):
        ablate.rank_heads(
            model,
            lambda i: model.blocks[i],
            torch.tensor([[1, 2]]),
            position=0,
            layers=[1],
            n_heads=HEADS,
            baseline="resample",
            donors=donors,
            corpus="toy",
        )


def test_the_refusal_names_the_corpus():
    model = ToyModel()
    donors = donors_from(model, [torch.tensor([[1]])], [0])
    with pytest.raises(ablate.AblationError, match="wikitext-sample"):
        ablate.rank_heads(
            model,
            lambda i: model.blocks[i],
            torch.tensor([[1, 2, 3]]),
            position=0,
            layers=[0],
            n_heads=HEADS,
            baseline="resample",
            donors=donors,
            corpus="wikitext-sample",
        )


# ------------------------------------------------------------- what it says


def _run_resample(n_donors=4, layers=1):
    model = ToyModel(layers=layers)
    ids = torch.tensor([[1, 2, 3]])
    seqs = [
        torch.tensor([[(i + 2) % VOCAB, (i + 3) % VOCAB, (i + 4) % VOCAB]])
        for i in range(n_donors)
    ]
    donors = donors_from(model, seqs, list(range(layers)))
    return ablate.rank_heads(
        model,
        lambda i: model.blocks[i],
        ids,
        position=0,
        layers=list(range(layers)),
        n_heads=HEADS,
        baseline="resample",
        donors=donors,
        corpus="toy-corpus",
        decode=lambda t: f"<{t}>",
    )


def test_every_row_reports_the_spread_not_one_draw():
    out = _run_resample(n_donors=4)
    for row in out["ranked"]:
        assert row["draws"] == 4
        assert row["kl_min"] <= row["kl"] <= row["kl_max"]


def test_the_corpus_travels_with_the_numbers():
    out = _run_resample()
    assert out["corpus"] == "toy-corpus"
    assert out["draws"] == 4


def test_passes_counts_every_draw():
    """Cost is draws x the sweep, which is why this is gated behind a preflight."""
    out = _run_resample(n_donors=4, layers=2)
    assert out["passes"] == 2 * HEADS * 4 + 2
    assert out["baseline"] == "resample"


def test_flips_top_requires_every_draw_to_flip():
    """A flip that depends on which donor arrived is not a property of a head."""
    out = _run_resample(n_donors=4)
    for row in out["ranked"]:
        assert isinstance(row["flips_top"], bool)


def test_resample_is_an_allowed_baseline():
    assert "resample" in ablate.BASELINES
    assert ablate.RESAMPLE_DRAWS == 8


def test_estimating_a_resample_sweep_does_not_need_a_corpus():
    """Regression: the probe built `_cut(..., "resample")` with no donor, so
    the hook indexed None inside a forward pass and
    /api/attention/ablate/estimate?baseline=resample answered 500. A browser
    found it; no unit test did, because estimate_cost was only ever called
    with the default baseline."""
    model = ToyModel()
    out = ablate.estimate_cost(
        model,
        lambda i: model.blocks[i],
        torch.tensor([[1, 2, 3]]),
        position=0,
        layers=[0],
        n_heads=HEADS,
        baseline="resample",
    )
    assert out["estimate"]["passes"] == HEADS + 2
    assert out["baseline"] == "resample"


def test_estimating_works_for_every_baseline():
    model = ToyModel()
    for name in ablate.BASELINES:
        out = ablate.estimate_cost(
            model,
            lambda i: model.blocks[i],
            torch.tensor([[1, 2]]),
            position=0,
            layers=[0],
            n_heads=HEADS,
            baseline=name,
        )
        assert out["estimate"]["passes"] > 0, name


# --------------------------------------------------- the disagreement figure


def test_spearman_is_one_for_an_identical_ranking():
    assert ablate.spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == 1.0


def test_spearman_is_minus_one_for_a_reversal():
    assert ablate.spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0


def test_spearman_is_none_when_one_side_is_constant():
    """Not 0.0: 'uncorrelated' and 'that is not a ranking' are different."""
    assert ablate.spearman([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) is None
    assert ablate.spearman([1.0], [2.0]) is None
    assert ablate.spearman([1.0, 2.0], [1.0]) is None


def test_ties_get_average_ranks_so_they_do_not_manufacture_order():
    assert ablate._ranks([5.0, 5.0, 9.0]) == [1.5, 1.5, 3.0]


def test_compare_baselines_counts_the_top_k_disagreement():
    zero = [
        {"layer": 0, "head": h, "kl": k} for h, k in enumerate([9.0, 8.0, 1.0, 0.5])
    ]
    resample = [
        {"layer": 0, "head": h, "kl": k} for h, k in enumerate([0.5, 1.0, 8.0, 9.0])
    ]
    out = ablate.compare_baselines({"zero": zero, "resample": resample}, top=2)

    pair = out["pairs"][0]
    assert pair["baselines"] == ["resample", "zero"]
    assert pair["spearman"] == -1.0
    assert pair["top_k"] == 2
    assert pair["top_k_shared"] == 0
    assert pair["top_k_disagree"] == 2


def test_compare_baselines_agrees_with_itself():
    rows = [{"layer": 0, "head": h, "kl": float(4 - h)} for h in range(4)]
    out = ablate.compare_baselines({"a": rows, "b": list(rows)}, top=3)
    pair = out["pairs"][0]
    assert pair["spearman"] == 1.0
    assert pair["top_k_disagree"] == 0


def test_compare_baselines_only_compares_heads_both_measured():
    a = [{"layer": 0, "head": 0, "kl": 1.0}, {"layer": 0, "head": 1, "kl": 2.0}]
    b = [{"layer": 0, "head": 1, "kl": 2.0}, {"layer": 1, "head": 0, "kl": 3.0}]
    pair = ablate.compare_baselines({"a": a, "b": b})["pairs"][0]
    assert pair["heads_compared"] == 1
