"""Head ranking must be a measurement, not an ordered opinion.

This is the feature most able to produce a confident, ordered, wrong list:
it outputs a leaderboard, and a leaderboard is read as truth. Every test
here guards one of the four ways it could lie.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from modelmri import ablate  # noqa: E402


class FakeProj(torch.nn.Linear):
    """Stands in for c_proj / o_proj."""


def block_with(width: int, head_dim: int | None = None) -> torch.nn.Module:
    block = torch.nn.Module()
    attn = torch.nn.Module()
    attn.o_proj = FakeProj(width, 8, bias=False)
    if head_dim is not None:
        attn.head_dim = head_dim
    block.self_attn = attn
    return block


# ------------------------------------------------- where the cut goes


def test_head_dim_is_read_from_the_projection_not_guessed():
    """`hidden_size // n_heads` is right for gpt2 and wrong by 2x on
    Qwen3-0.6B (128, not 64) and wrong on gemma-3-270m (256, not 160).
    The wrong value ablates half of one head plus half of the next."""
    # 16 heads x 128 = 2048 wide, exactly Qwen3-0.6B's shape.
    assert ablate.head_geometry(block_with(2048), 16) == 128
    # 4 x 256 = 1024, gemma-3-270m-it's shape.
    assert ablate.head_geometry(block_with(1024), 4) == 256


def test_an_explicit_head_dim_on_the_module_wins():
    assert ablate.head_geometry(block_with(2048, head_dim=128), 16) == 128


def test_a_geometry_that_does_not_divide_is_refused_not_guessed():
    """Refusing is the difference between a wrong number and no number."""
    with pytest.raises(ablate.AblationError, match="cannot tell where one head"):
        ablate.head_geometry(block_with(1000), 7)


def test_a_model_with_no_recognisable_projection_says_so():
    empty = torch.nn.Module()
    empty.self_attn = torch.nn.Module()
    with pytest.raises(ablate.AblationError, match="output projection"):
        ablate.out_projection(empty)

    nothing = torch.nn.Module()
    with pytest.raises(ablate.AblationError, match="attn"):
        ablate.out_projection(nothing)


def test_the_head_slices_tile_the_projection_input_exactly():
    """Zeroing every head one slice at a time must equal zeroing the whole
    tensor. If the slices overlapped or left a gap, each 'head' would be
    measuring something that is not a head."""
    width, heads = 24, 6
    head_dim = width // heads
    x = torch.arange(float(width)).reshape(1, 1, width)

    stacked = x.clone()
    for head in range(heads):
        (stacked,) = ablate._cut(head, head_dim, "zero")(None, (stacked,))
    assert torch.equal(stacked, torch.zeros_like(x))


def test_zeroing_one_head_leaves_every_other_column_untouched():
    x = torch.ones(1, 2, 12)
    (out,) = ablate._cut(1, 4, "zero")(None, (x,))
    assert out[..., 4:8].abs().sum() == 0
    assert out[..., :4].sum() == 8 and out[..., 8:].sum() == 8
    # and the caller's tensor is not mutated under it
    assert x.sum() == 24


def test_the_mean_baseline_removes_variation_not_magnitude():
    """Two different questions. Zero asks 'does this head contribute?';
    mean asks 'does its variation across positions matter?'."""
    x = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]])  # 1 head, dim 2, 2 positions
    (out,) = ablate._cut(0, 2, "mean")(None, (x,))
    assert torch.allclose(out, torch.tensor([[[3.0, 5.0], [3.0, 5.0]]]))


# ------------------------------------------------- the metric


class FakeModel:
    """A model whose next-token logits depend on which head is cut.

    `shift` adds a constant to every logit — softmax is invariant to that,
    so a metric that notices it is measuring nothing.
    """

    def __init__(self, block, shift: float = 0.0):
        self._block = block
        self._shift = shift

    def __call__(self, ids):
        width = self._block.self_attn.o_proj.in_features
        x = torch.ones(1, 1, width)
        (cut,) = _CURRENT["hook"](None, (x,)) if _CURRENT["hook"] else (x,)
        # A logit vector that reacts to how much of the input survived.
        alive = float(cut.sum())
        base = torch.tensor([[[3.0, 1.0, 0.5, 0.2]]]) * (alive / width)
        return type("Out", (), {"logits": base + self._shift})()


_CURRENT: dict = {"hook": None}


def test_ranking_uses_kl_so_a_constant_logit_shift_scores_zero():
    """Ablation shifts whole logit vectors. Measured on gpt2 L0H0 with the
    prompt "The capital of France is": the top token's logit moves -0.258
    while the vocabulary mean moves -0.145, so the honest residual is -0.113
    and a raw logit difference calls that head 2.3x more important than it is.
    """
    p = torch.tensor([0.7, 0.2, 0.1])
    shifted = torch.log(p) + 12.345  # same distribution, different logits
    q = torch.softmax(shifted, dim=-1)
    # Not zero — float32 round-tripping through log/softmax costs ~1e-8.
    # The bound is set against the smallest per-head signal worth resolving on
    # gpt2 layer 0 (0.0028, head 6), which this is three orders of magnitude
    # below.
    assert ablate._kl(p, q) < 1e-6


def test_kl_is_finite_even_when_the_ablated_distribution_zeroes_a_token():
    p = torch.tensor([0.5, 0.5])
    q = torch.tensor([1.0, 0.0])
    value = ablate._kl(p, q)
    assert math.isfinite(value) and value > 0


def test_kl_is_zero_for_an_identical_distribution():
    p = torch.tensor([0.25, 0.25, 0.5])
    assert ablate._kl(p, p) == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------- what the answer says


def _tiny_run(baseline: str = "zero") -> dict:
    """A 2-head toy where head 0 matters and head 1 does not."""
    width, heads = 4, 2
    block = block_with(width)

    class Model:
        def __call__(self, ids):
            x = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
            if _CURRENT["hook"]:
                (x,) = _CURRENT["hook"](None, (x,))
            keep = float(x.sum())
            return type(
                "Out", (), {"logits": torch.tensor([[[keep, 0.5, 0.25, 0.1]]])}
            )()

    real_cut = ablate._cut

    def spy(head, head_dim, base):
        hook = real_cut(head, head_dim, base)

        def wrapped(module, args):
            return hook(module, args)

        _CURRENT["hook"] = wrapped
        return wrapped

    ablate._cut = spy
    try:
        proj = ablate.out_projection(block)
        original_register = proj.register_forward_pre_hook

        class Handle:
            def remove(self):
                _CURRENT["hook"] = None

        proj.register_forward_pre_hook = lambda fn: Handle()  # type: ignore[assignment]
        try:
            return ablate.rank_heads(
                Model(),
                lambda _l: block,
                torch.tensor([[1, 2]]),
                position=0,
                layers=[0],
                n_heads=heads,
                baseline=baseline,
                decode=lambda t: f"<{t}>",
            )
        finally:
            proj.register_forward_pre_hook = original_register  # type: ignore[assignment]
            _CURRENT["hook"] = None
    finally:
        ablate._cut = real_cut


def test_the_answer_names_its_baseline():
    """On gpt2 layer 0, zero-ablation ranks heads 7, 10, 9; replacing each
    head with its mean ranks 3, 1, 10. Same model, same prompt, different
    question — so an unlabelled number is the lie."""
    assert _tiny_run("zero")["baseline"] == "zero"
    assert _tiny_run("mean")["baseline"] == "mean"


def test_an_unknown_baseline_is_refused():
    with pytest.raises(ablate.AblationError, match="unknown baseline"):
        _tiny_run("vibes")


def test_the_answer_carries_a_measured_noise_floor():
    """The floor is measured, not assumed. On this code path it comes out at
    exactly 0.0 — checked on CPU and CUDA in fp32, bf16 and fp16, because one
    unbatched sequence replayed through the same kernels is bit-identical.
    That is the argument for spending the pass rather than dropping it: the
    floor is zero *here*, and this is what proves it. Batching, TF32 or a
    different accelerator can lift it above the smallest real signals, and
    nothing else would notice."""
    out = _tiny_run()
    assert "noise_floor_kl" in out
    assert out["noise_floor_kl"] >= 0


def test_the_answer_says_the_scores_are_not_shares():
    """Measured on gpt2 layer 0: the twelve per-head KLs sum to 1.995 while
    ablating the whole layer gives 0.208. Presenting these as portions of a
    prediction would be a fabrication."""
    means = _tiny_run()["means"].lower()
    assert "not" in means and ("add up" in means or "share" in means)


def test_the_ranking_is_sorted_and_complete():
    out = _tiny_run()
    scores = [r["kl"] for r in out["ranked"]]
    assert scores == sorted(scores, reverse=True)
    assert len(out["ranked"]) == 2
    assert {(r["layer"], r["head"]) for r in out["ranked"]} == {(0, 0), (0, 1)}


def test_every_row_reports_what_happened_to_the_top_token():
    for row in _tiny_run()["ranked"]:
        assert {"p_top_before", "p_top_after", "flips_top"} <= set(row)
        assert 0.0 <= row["p_top_before"] <= 1.0
        assert 0.0 <= row["p_top_after"] <= 1.0
