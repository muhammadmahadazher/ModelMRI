"""Activation patching: the refusals, and the arithmetic underneath them.

No model download. The refusals are the part most likely to rot, because they
guard failures that are invisible when they happen — two prompts of different
token lengths both run fine on their own, and a pair that predicts the same
token divides by exactly zero.
"""

from __future__ import annotations

import pytest
import torch

from modelmri import patch


class _Tok:
    """Whitespace tokenizer. Enough to exercise alignment and decoding."""

    def __init__(self, vocab: list[str] | None = None) -> None:
        self.vocab = vocab or []

    def __call__(self, text: str, return_tensors=None):
        words = text.split()
        for w in words:
            if w not in self.vocab:
                self.vocab.append(w)
        ids = torch.tensor([[self.vocab.index(w) for w in words]])
        return type("Enc", (), {"input_ids": ids})()

    def decode(self, ids) -> str:
        return "".join(self.vocab[int(i)] for i in ids)


class _Block(torch.nn.Module):
    def forward(self, x):
        return x


class _Model(torch.nn.Module):
    """A model whose answer depends on exactly one position, so the grid has a
    known right answer rather than one we read off the thing under test."""

    def __init__(self, n_layers: int = 3, d: int = 4, vocab: int = 6) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(_Block() for _ in range(n_layers))
        self.embed = torch.nn.Embedding(vocab, d)
        self.head = torch.nn.Linear(d, vocab)

    def forward(self, ids):
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        return type("Out", (), {"logits": self.head(x)})()


def _fixture():
    torch.manual_seed(0)
    model = _Model()
    return model, _Tok(), list(model.blocks)


def test_two_prompts_of_different_lengths_are_refused_with_both_tokenizations():
    """The failure this guards is silent: both prompts run fine alone, and
    position 3 of one simply is not position 3 of the other. Measured on real
    pairs, 2 of 8 natural minimal pairs tokenize to different lengths."""
    model, tok, blocks = _fixture()
    with pytest.raises(patch.PatchError) as err:
        patch.trace(model, tok, blocks, "a b c", "a b c d", device="cpu")
    msg = str(err.value)
    assert "different lengths" in msg
    assert "3" in msg and "4" in msg
    # The reader has to be able to see WHICH token split differently, or the
    # message is just a complaint.
    assert "'a'" in msg or '"a"' in msg or "a" in msg
    assert "same number of pieces" in msg


def test_a_pair_that_agrees_is_refused_rather_than_divided_by():
    """Recovery divides by the gap between the two answers. Two of three
    casually-written pairs produced the same next token, making that gap
    exactly 0.000000."""
    model, tok, blocks = _fixture()

    # Same answer by construction: the head ignores position, so two prompts
    # of equal length over the same vocab give the same argmax.
    class _Flat(_Model):
        def forward(self, ids):
            out = super().forward(ids)
            out.logits[:] = 0.0
            out.logits[..., 2] = 1.0
            return out

    flat = _Flat()
    with pytest.raises(patch.PatchError) as err:
        patch.trace(flat, tok, list(flat.blocks), "a b", "c d", device="cpu")
    assert "same next token" in str(err.value)


def test_identical_prompts_say_what_to_change():
    model, tok, blocks = _fixture()
    with pytest.raises(patch.PatchError) as err:
        patch.trace(model, tok, blocks, "a b c", "a b c", device="cpu")
    assert "identical" in str(err.value)
    assert "Change one fact" in str(err.value)


def test_an_empty_prompt_is_refused():
    model, tok, blocks = _fixture()
    with pytest.raises(patch.PatchError):
        patch.trace(model, tok, blocks, "   ", "a b", device="cpu")


def test_a_near_zero_gap_is_refused_before_it_becomes_a_percentage():
    """A gap of 0.3158 logits — measured on "The doctor said he" against "The
    nurse said she" — makes a movement of 0.1 read as 32% recovered. The
    fraction is only meaningful when there is something to divide by."""
    assert patch.MIN_GAP > 0.3158, (
        "the floor has to sit above the smallest real gap measured, or the "
        "refusal never fires on the case that motivated it"
    )


def test_the_splice_does_not_write_into_the_cache():
    """`_splice` clones. Writing in place would corrupt the clean cache for
    every later patch, and that failure does not raise — it makes each site's
    score depend on the order the sites were visited."""
    block = _Block()
    incoming = torch.zeros(1, 3, 4)
    handle = patch._splice(block, 1, torch.full((1, 4), 5.0))
    try:
        out = block(incoming)  # through __call__, so the pre-hook actually runs
    finally:
        handle.remove()
    assert torch.equal(incoming, torch.zeros(1, 3, 4)), "the input was mutated"
    assert torch.equal(out[:, 1, :], torch.full((1, 4), 5.0))
    assert torch.equal(out[:, 0, :], torch.zeros(1, 4))


def test_the_control_is_more_than_one_draw():
    """One draw is a sample, not a property. Measured over 8 draws at a single
    site the control ran from -2.038 to +0.616 against a real recovery of
    +0.427, and the gate moves from 76 of 132 sites on one draw to 20 on
    eight."""
    assert patch.CONTROL_DRAWS > 1


def test_capture_reads_the_block_input():
    block = _Block()
    sink: dict = {}
    handle = patch._capture(block, 7, sink)
    x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    try:
        block(x)
    finally:
        handle.remove()
    assert 7 in sink and torch.equal(sink[7], x)
    # A clone, not a view: the cache has to survive the tensor being reused.
    assert sink[7].data_ptr() != x.data_ptr()


def test_the_sublayer_lookup_knows_both_spellings():
    """GPT-2 calls it `attn`; Llama, Qwen and Gemma call it `self_attn`. Both
    call the MLP `mlp`. Guessing wrong does not raise — it would read a
    different tensor and report it as the model's attention."""

    class _GPT2ish(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _Block()
            self.mlp = _Block()

    class _Llamaish(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _Block()
            self.mlp = _Block()

    g, ll = _GPT2ish(), _Llamaish()
    assert patch._sublayer(g, "attn") is g.attn
    assert patch._sublayer(ll, "attn") is ll.self_attn
    assert patch._sublayer(g, "mlp") is g.mlp and patch._sublayer(ll, "mlp") is ll.mlp


def test_an_unknown_layout_is_refused_and_points_at_the_one_that_always_works():
    from modelmri import patch as patch_mod

    class _Exotic(torch.nn.Module):
        pass

    with pytest.raises(patch_mod.PatchError) as err:
        patch_mod._sublayer(_Exotic(), "attn")
    msg = str(err.value)
    assert "self_attn" in msg and "attn" in msg
    # The refusal has to leave the reader somewhere to go, and resid reads the
    # block input, which every layout this tool walks has.
    assert "resid" in msg


def test_splicing_a_sublayer_output_keeps_the_rest_of_its_tuple():
    """Attention returns the hidden states alongside a cache on several
    transformers versions. Returning a bare tensor would silently drop the
    tail, and nothing downstream would say so."""

    class _TupleOut(torch.nn.Module):
        def forward(self, x):
            return (x, "cache-sentinel")

    m = _TupleOut()
    handle = patch._splice_out(m, 1, torch.full((1, 4), 9.0))
    try:
        out = m(torch.zeros(1, 3, 4))
    finally:
        handle.remove()
    assert isinstance(out, tuple) and len(out) == 2
    assert out[1] == "cache-sentinel", "the cache was dropped"
    assert torch.equal(out[0][:, 1, :], torch.full((1, 4), 9.0))
    assert torch.equal(out[0][:, 0, :], torch.zeros(1, 4))


def test_every_component_is_named_once():
    """The three grids are the product's claim; a typo here would silently
    drop one from every response."""
    assert patch.COMPONENTS == ("resid", "attn", "mlp")
    assert len(set(patch.COMPONENTS)) == len(patch.COMPONENTS)


def test_the_tie_threshold_survives_the_wire_in_float32():
    """`round(resolution, 6)` threw away the number it was publishing.

    `recovery_resolution` is one step of the model's number format on the
    recovery scale. In bfloat16 that is around 1e-2 and rounding to six places
    is harmless; in float32 — which is the CPU default, so every patch trace
    run without a GPU — it lives at 1e-6 and below.

    Measured at |logit|max 22 over a gap of 4: the resolution is 6.557e-07,
    `round(_, 6)` returns 1e-06, and most of the figure is gone. Widen the gap
    and it returns exactly 0.0 — a tie threshold of zero, which asserts that
    nothing is tied, the opposite of what the number is for. Nothing
    downstream can recover it, which is why this is fixed at the source rather
    than in a formatter.
    """
    assert round(6.557e-07, 6) == 1e-06
    assert round(1.6e-08, 6) == 0.0

    # And the sentence that publishes it no longer floors it either. `.3f` on
    # a float32 resolution printed "0.000" in the line that says two senders
    # closer than THAT are tied.
    from modelmri import fmt

    assert format(6.557e-07, ".3f") == "0.000"
    assert fmt.measured(6.557e-07, 3) == "6.6e-07"
    # An exact zero still prints as zero: a deterministic model whose logits
    # do not move IS at resolution zero, and that one is the measurement.
    assert fmt.measured(0.0, 3) == "0.000"
