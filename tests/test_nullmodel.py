"""The control has to be able to say the measurement is uninformative.

That is the whole reason it exists, and it is the sentence a tool has every
incentive not to ship. So these tests pin the wording at each end: when the
untrained twin ranks heads the same way, the verdict says the ranking is
mostly reporting the architecture; when it does not, the verdict says so
without overclaiming.

The other half is arithmetic honesty — a correlation that does not exist must
come back as "no ranking to compare", never as 0.0, because 0.0 reads as
"unrelated" which is a positive finding.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from modelmri import nullmodel  # noqa: E402
from modelmri.errors import Refusal  # noqa: E402

# ------------------------------------------------------------ the verdict


def test_a_twin_that_agrees_says_the_ranking_is_the_architecture():
    said = nullmodel.verdict(0.92, top_k_shared=5, top_k=5)
    assert "architecture" in said
    assert "0.92" in said
    assert "not what this model learned" in said


def test_a_twin_that_disagrees_says_the_finding_survives():
    said = nullmodel.verdict(0.05, top_k_shared=0, top_k=5)
    assert "differently" in said
    assert "not just the shape" in said


def test_a_partial_match_is_not_rounded_to_either_end():
    said = nullmodel.verdict(0.45, top_k_shared=2, top_k=5)
    assert "partly" in said
    assert "Some of what you are seeing is the architecture" in said


def test_sharing_almost_every_top_head_counts_as_agreement():
    """Spearman over every head can be middling while the heads a reader
    actually looks at are identical. The top-k share is the one they see."""
    said = nullmodel.verdict(0.4, top_k_shared=5, top_k=5)
    assert "architecture" in said


def test_no_correlation_is_reported_as_absent_not_as_zero():
    """0.0 reads as 'unrelated', which is a positive finding. An untrained
    twin whose scores are all equal supports no finding at all."""
    said = nullmodel.verdict(None, top_k_shared=0, top_k=5)
    assert "no ranking to compare" in said
    assert "0.00" not in said


def test_every_verdict_quotes_the_number_it_came_from():
    for rho in (0.95, 0.5, 0.01):
        assert f"{rho:.2f}" in nullmodel.verdict(rho, top_k_shared=1, top_k=5)


# --------------------------------------------------------------- the twin


def test_the_twin_is_built_from_config_without_fetching_weights():
    """`from_config` initialises; it never downloads. That is what makes the
    control work air-gapped and what keeps it to one model's memory."""
    from transformers import GPT2Config

    cfg = GPT2Config(n_layer=1, n_head=2, n_embd=8, vocab_size=16, n_positions=16)
    twin = nullmodel.build_twin(cfg, seed=0, dtype=torch.float32, device="cpu")
    assert twin.config.num_hidden_layers == 1
    assert not twin.training  # eval(), or dropout would make it non-deterministic


def test_the_seed_makes_the_twin_reproducible():
    """A control whose seed is not stated is a control nobody can re-run."""
    from transformers import GPT2Config

    cfg = GPT2Config(n_layer=1, n_head=2, n_embd=8, vocab_size=16, n_positions=16)
    a = nullmodel.build_twin(cfg, seed=7, dtype=torch.float32, device="cpu")
    b = nullmodel.build_twin(cfg, seed=7, dtype=torch.float32, device="cpu")
    c = nullmodel.build_twin(cfg, seed=8, dtype=torch.float32, device="cpu")

    pa = dict(a.named_parameters())
    pb = dict(b.named_parameters())
    pc = dict(c.named_parameters())
    key = next(k for k in pa if pa[k].numel() > 4)
    assert torch.equal(pa[key], pb[key])
    assert not torch.equal(pa[key], pc[key])


def test_a_config_that_cannot_be_instantiated_refuses_in_words():
    class Unbuildable:
        pass

    with pytest.raises(Refusal, match="cannot be built from its config"):
        nullmodel.build_twin(Unbuildable(), seed=0, dtype=torch.float32, device="cpu")


def test_the_refusal_does_not_republish_the_exception():
    class Unbuildable:
        pass

    try:
        nullmodel.build_twin(Unbuildable(), seed=0, dtype=torch.float32, device="cpu")
    except Refusal as err:
        # errors.py: never interpolate a caught exception's text — it is
        # machinery talking to itself and carries paths from this machine.
        assert "Traceback" not in str(err)
        assert "site-packages" not in str(err)


def test_teardown_does_not_raise_without_cuda():
    from transformers import GPT2Config

    cfg = GPT2Config(n_layer=1, n_head=2, n_embd=8, vocab_size=16, n_positions=16)
    twin = nullmodel.build_twin(cfg, seed=0, dtype=torch.float32, device="cpu")
    nullmodel.teardown(twin)


# ---------------------------- regressions from the round-2 audit


def test_teardown_actually_releases_accelerator_memory():
    """`del twin` inside teardown unbinds only its own parameter — the caller's
    variable is still a live reference, so gc collects nothing. Measured: a
    gpt2 twin allocated 255.3 MB and 255.3 MB was still allocated after
    teardown returned, while its docstring claimed the memory came back."""
    if not torch.cuda.is_available():
        pytest.skip("needs an accelerator to observe the allocation")
    from transformers import GPT2Config

    cfg = GPT2Config(n_layer=4, n_head=4, n_embd=256, vocab_size=512, n_positions=64)
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    twin = nullmodel.build_twin(cfg, seed=0, dtype=torch.float32, device="cuda")
    allocated = torch.cuda.memory_allocated() - before
    assert allocated > 0, "the twin did not allocate anything to free"

    nullmodel.teardown(twin)
    # `twin` is deliberately still bound here — that is the caller shape the
    # old implementation could not handle.
    retained = torch.cuda.memory_allocated() - before
    assert retained < allocated * 0.15, (
        f"{retained / 1e6:.1f} MB of {allocated / 1e6:.1f} MB still held"
    )


def test_a_verdict_with_nothing_compared_does_not_draw_a_conclusion():
    """`top_k_shared >= max(1, top_k - 1)` is false at top_k 0, but a high rho
    still took the "mostly the architecture" branch and printed "sharing 0 of
    the top 0"."""
    said = nullmodel.verdict(0.9, top_k_shared=0, top_k=0)
    assert "top 0" not in said
    assert "nothing to say" in said
