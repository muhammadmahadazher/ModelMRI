# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""A neuron browser is the panel that runs when the SAE panel cannot.

Which means it is the one a reader is most likely to mistake for the SAE
panel — same three readouts, same span cards, same firing-rate table, a
completely different basis underneath. Most of these tests are about the
things that must NOT transfer: the caveat that neurons are polysemantic, the
firing-rate threshold that would flag a whole layer if it were carried over,
the labels nothing here is allowed to attach, and the caps that have to be
reported rather than merely applied.

The fixture is a hand-built `nn.Module` whose MLP hidden activation is a
lookup table indexed by token id. That makes every activation in these tests
an exact known number rather than a plausible one, so the arithmetic can be
checked with `==` instead of a tolerance — which a real checkpoint could never
support.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from modelmri import neurons  # noqa: E402
from modelmri.errors import BadRequest, Refusal  # noqa: E402

# Eight neurons over a five-token vocabulary, chosen by hand so that every
# quantity below can be worked out on paper.
#
#   neuron 0  fires on every token           -- the dense, uninteresting case
#   neuron 1  fires on tokens 0 and 1 only   -- co-fires with 2
#   neuron 2  fires on tokens 0 and 1 only   -- co-fires with 1
#   neuron 3  fires on 3 and 4, NEGATIVE on 0 and 1  -- co-fires with 4, and
#             it is the neuron that gives `clip` something real to discard
#   neuron 4  fires on tokens 3 and 4 only   -- co-fires with 3
#   neuron 5  fires once, on token 2, hard   -- the sharp, rare case
#   neuron 6  never positive, always -0.5    -- "not seen", and negative mass
#   neuron 7  always exactly zero            -- never fired, no negative mass
TABLE = torch.tensor(
    [
        # n0    n1    n2    n3    n4    n5    n6   n7
        [1.0, 2.0, 3.0, -1.0, 0.0, 0.0, -0.5, 0.0],  # token 0
        [1.0, 4.0, 6.0, -2.0, 0.0, 0.0, -0.5, 0.0],  # token 1
        [1.0, 0.0, 0.0, 0.0, 0.0, 9.0, -0.5, 0.0],  # token 2
        [1.0, 0.0, 0.0, 2.0, 1.0, 0.0, -0.5, 0.0],  # token 3
        [1.0, 0.0, 0.0, 4.0, 2.0, 0.0, -0.5, 0.0],  # token 4
    ],
    dtype=torch.float32,
)

VOCAB = TABLE.shape[0]
D_MLP = TABLE.shape[1]
# Equal to VOCAB so that the embedding can be the identity on one-hot vectors
# AND the block can keep its residual connection. Both matter: the identity
# embedding is what makes the projection's input an exact table lookup, and
# the residual connection is what makes this block the shape the capture hook
# is written against rather than a bare MLP.
D_MODEL = VOCAB


class Projection(torch.nn.Linear):
    """Stands in for down_proj / c_proj / dense_4h_to_h / fc2."""


class TinyMLP(torch.nn.Module):
    """`hidden = TABLE[token]`, exactly, so the capture is a known tensor.

    No nonlinearity: the point of the fixture is that the number arriving at
    the projection is one this file wrote down, not one an activation function
    produced to eleven decimal places.
    """

    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(VOCAB, D_MLP, bias=False)
        with torch.no_grad():
            self.up.weight.copy_(TABLE.T)
        self.down_proj = Projection(D_MLP, D_MODEL, bias=False)
        with torch.no_grad():
            self.down_proj.weight.copy_(
                torch.arange(D_MODEL * D_MLP, dtype=torch.float32).reshape(
                    D_MODEL, D_MLP
                )
            )

    def forward(self, x):
        return self.down_proj(self.up(x))


class TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMLP()

    def forward(self, x):
        return x + self.mlp(x)


class Out:
    def __init__(self, logits):
        self.logits = logits


class TinyModel(torch.nn.Module):
    """One block, a one-hot embedding, and an unembedding, on CPU.

    The embedding is the identity on one-hot vectors, so `up @ onehot(t)` is
    exactly `TABLE[t]` and the projection's input is a table lookup.
    """

    def __init__(self):
        super().__init__()
        self.block = TinyBlock()
        self.norm = torch.nn.Identity()
        self.head = torch.nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, ids):
        one_hot = torch.nn.functional.one_hot(ids, num_classes=VOCAB).float()
        return Out(self.head(self.block(one_hot)))

    def get_output_embeddings(self):
        return self.head


class Tokenizer:
    """Whitespace-splitting, digits are token ids. `"0 1 2"` -> `[0, 1, 2]`."""

    def __call__(self, text, return_tensors=None):
        ids = [int(part) for part in text.split()]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, ids):
        return f"<{int(ids[0])}>"


def rig():
    model = TinyModel().eval()
    return model, model.block, Tokenizer()


# Six sequences. Tokens 0 and 1 appear together, tokens 3 and 4 appear
# together, and token 2 appears alone -- so neurons 1/2 co-fire, 3/4 co-fire,
# and 5 fires on its own. That is the structure `decompose` has to find.
CORPUS = [
    "0 1 0 1",
    "1 0 1 0",
    "3 4 3 4",
    "4 3 4 3",
    "0 1 3 4",
    "2 2 0 3",
]


# ------------------------------------------------------- where a neuron is


def test_the_projection_is_found_under_every_family_spelling():
    """Four families, four names, and none of them guessable. Reading the
    wrong module returns a tensor of the wrong width and every neuron index
    below it would be about a different neuron."""
    for name in ("down_proj", "c_proj", "dense_4h_to_h", "fc2", "w2"):
        block = torch.nn.Module()
        holder = torch.nn.Module()
        setattr(holder, name, torch.nn.Linear(11, 4, bias=False))
        block.mlp = holder
        assert neurons.neuron_count(block) == 11


def test_a_projection_hung_straight_off_the_block_is_found():
    """OPT has no `mlp` at all -- `fc1`/`fc2` sit on the block. A lookup that
    only ever reads `block.mlp` refuses a whole architecture family."""
    block = torch.nn.Module()
    block.fc2 = torch.nn.Linear(7, 4, bias=False)
    assert neurons.neuron_count(block) == 7


def test_a_block_with_no_recognisable_projection_refuses_and_names_the_spellings():
    """A refusal that lists what IS supported is a next step. One that says
    'unsupported' is a dead end."""
    block = torch.nn.Module()
    block.mlp = torch.nn.Module()
    with pytest.raises(Refusal, match="down_proj") as err:
        neurons.neuron_count(block)
    assert "fc2" in err.value.sentence


def test_d_mlp_is_read_off_the_projection_not_a_config():
    """The same rule `ablate.head_geometry` holds for attention: a width taken
    from anywhere but the tensor being sliced is free to disagree with it, and
    the disagreement is silent."""
    _, block, _ = rig()
    assert neurons.neuron_count(block) == D_MLP


def test_a_conv1d_style_projection_is_read_from_the_transposed_weight():
    """GPT-2's Conv1D holds `[in, out]` and has no `in_features`. Falling back
    to `weight.shape[0]` on an nn.Linear would give the OUTPUT width, so the
    two paths have to be told apart rather than merged."""

    class Conv1D(torch.nn.Module):
        def __init__(self, n_in, n_out):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(n_in, n_out))

    block = torch.nn.Module()
    holder = torch.nn.Module()
    holder.c_proj = Conv1D(3072, 768)
    block.mlp = holder
    assert neurons.neuron_count(block) == 3072


def test_the_write_direction_is_the_projections_column_exactly():
    """A neuron's contribution to the residual stream IS one column of the
    down-projection. Taking the row instead returns a vector of the right
    length belonging to a different neuron, which does not error and makes
    every logit below it confidently wrong."""
    _, block, _ = rig()
    proj = neurons.mlp_projection(block)
    for neuron in range(D_MLP):
        assert torch.equal(
            neurons.write_direction(proj, neuron), proj.weight[:, neuron]
        )


def test_the_write_direction_of_a_conv1d_is_the_row():
    class Conv1D(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(12.0).reshape(4, 3))

    proj = Conv1D()
    assert neurons.write_direction(proj, 2).tolist() == [6.0, 7.0, 8.0]


def test_a_neuron_index_outside_the_layer_is_refused_with_the_real_width():
    _, block, _ = rig()
    proj = neurons.mlp_projection(block)
    with pytest.raises(BadRequest, match="outside this layer") as err:
        neurons.write_direction(proj, D_MLP)
    assert "0 to 7" in err.value.sentence


def test_a_bool_is_not_a_neuron_index():
    """`isinstance(True, int)` is True, so `write_direction(proj, True)` would
    silently read neuron 1."""
    _, block, _ = rig()
    with pytest.raises(BadRequest, match="whole number"):
        neurons.write_direction(neurons.mlp_projection(block), True)


# ------------------------------------------------------------ the capture


def test_the_capture_returns_the_projections_input_exactly():
    """This is the tensor `write_direction` indexes into. Capturing the MLP's
    OUTPUT and inverting would be a second implementation of this module's
    geometry -- and is not invertible anyway, since down_proj maps d_mlp to
    d_model."""
    model, block, _ = rig()
    ids = torch.tensor([[0, 2, 4]])
    acts = neurons.capture_neuron_activations(model, block, ids)
    assert acts.shape == (3, D_MLP)
    assert torch.equal(acts, TABLE[torch.tensor([0, 2, 4])])


def test_a_block_that_never_runs_refuses_rather_than_returning_nothing():
    """An empty capture used to be an IndexError one frame later. The block
    that was hooked is simply not on the forward path, and saying so is a
    fact the caller can act on."""
    model, _, _ = rig()
    orphan = TinyBlock()
    with pytest.raises(Refusal, match="never fired"):
        neurons.capture_neuron_activations(model, orphan, torch.tensor([[0]]))


# ------------------------------------------------------------- the sweep


def test_every_firing_count_and_peak_is_exact():
    """The whole reason the fixture is a lookup table. Over CORPUS, token 0
    appears 6 times and token 1 6 times, so neurons 1 and 2 fired 12 times."""
    model, block, tok = rig()
    stats, rows, _ = sweep_it(model, block, tok)

    counts = {t: sum(seq.split().count(str(t)) for seq in CORPUS) for t in range(VOCAB)}
    assert stats.n_tokens == sum(counts.values()) == 24

    assert rows[0].n_fired == 24  # fires on every token
    assert rows[1].n_fired == counts[0] + counts[1]
    assert rows[2].n_fired == counts[0] + counts[1]
    assert rows[5].n_fired == counts[2]
    assert rows[1].max_activation == 4.0
    assert rows[2].max_activation == 6.0
    assert rows[5].max_activation == 9.0


def sweep_it(model, block, tok, **over):
    kwargs = dict(device="cpu", layer=0, corpus_label="fixture", sample_rows=24)
    kwargs.update(over)
    return neurons.sweep(model, block, tok, CORPUS, **kwargs)


def test_a_neuron_that_never_went_positive_has_no_mean_and_is_not_given_a_zero():
    """UNKNOWN NEVER COLLAPSES INTO ZERO. A neuron with no positive
    activations has no mean positive activation, and 0.0 would make it
    indistinguishable from one that fired constantly at exactly zero -- which
    neuron 7 in this fixture actually is."""
    model, block, tok = rig()
    _, rows, _ = sweep_it(model, block, tok)
    assert rows[6].mean_positive is None
    assert rows[7].mean_positive is None
    assert rows[6].n_fired == 0 and rows[7].n_fired == 0
    # And the two are still told apart, by the quantity that was measured.
    assert rows[6].min_activation == -0.5
    assert rows[7].min_activation == 0.0
    assert rows[6].n_negative == 24
    assert rows[7].n_negative == 0


def test_every_neuron_is_in_the_table_including_the_silent_ones():
    """Unlike an SAE feature, a neuron produces a value at every token. A
    neuron missing from the table would be a fact about the table."""
    model, block, tok = rig()
    _, rows, _ = sweep_it(model, block, tok)
    assert sorted(rows) == list(range(D_MLP))


def test_a_never_fired_neuron_is_not_seen_here_never_dead():
    """Dead means the neuron does nothing. Not seen means this text never
    showed it anything it responds to, and only one of those is about the
    model. Carried over from feature_corpus verbatim because the distinction
    does not depend on the basis."""
    model, block, tok = rig()
    stats, _, _ = sweep_it(model, block, tok)
    assert stats.n_never_fired == 2
    assert "NOT SEEN IN THIS CORPUS, not dead" in stats.means()


def test_the_layer_median_firing_rate_travels_with_every_rate():
    """An SAE feature on 20% of tokens is not selecting anything;
    `feature_corpus.evidence` says so. A post-GELU neuron near a coin flip is
    the ordinary case. Carrying that threshold over would flag the whole
    layer, so the reference is measured on this layer instead of chosen."""
    model, block, tok = rig()
    stats, _, _ = sweep_it(model, block, tok)
    assert stats.layer_median_firing_rate is not None
    assert "median neuron in this layer" in stats.means()
    # Eight neurons: rates sorted are two zeros, then 4 of 24, then 12, 12,
    # 12, 24 over 24. The median is the mean of the 4th and 5th.
    rates = sorted(row.firing_rate for row in sweep_it(model, block, tok)[1].values())
    assert stats.layer_median_firing_rate == pytest.approx((rates[3] + rates[4]) / 2)


def test_the_negative_mass_share_is_measured_not_assumed_small():
    """A GELU's negative lobe is bounded near -0.17. A gated MLP feeds
    silu(gate) * up into the projection and `up` is an unbounded linear map,
    so there is no architectural bound on this and clamping it is not a
    rounding decision."""
    model, block, tok = rig()
    stats, _, _ = sweep_it(model, block, tok)
    # Every token contributes 0.5 of negative mass from neuron 6.
    total = float(TABLE.abs().sum(dim=1)[torch.tensor([0, 1, 0, 1])].sum())
    assert stats.negative_mass_share is not None
    assert 0 < stats.negative_mass_share < 1
    assert "NEGATIVE" in stats.means()
    assert total > 0  # the fixture would be pointless otherwise


def test_the_sweep_refuses_an_empty_corpus_with_a_next_step():
    model, block, tok = rig()
    with pytest.raises(BadRequest, match="needs text") as err:
        neurons.sweep(model, block, tok, [], device="cpu", layer=0)
    assert "downloaded" in err.value.sentence


def test_a_corpus_that_tokenises_to_nothing_is_a_refusal_not_a_zero_table():
    """A table of eight neurons that all read 'never fired' over zero tokens
    is a page of numbers that describe nothing."""
    model, block, tok = rig()
    with pytest.raises(Refusal, match="tokenised to nothing"):
        neurons.sweep(model, block, tok, ["", "  "], device="cpu", layer=0)


# --------------------------------------------------------------- the caps


def test_a_sequence_longer_than_the_cap_is_cut_and_the_cut_is_counted(monkeypatch):
    """MAX_SEQUENCE_TOKENS, not MAX_TOKENS, is what bounds the peak: the sweep
    streams sequence by sequence, so what lives at once is `[S, d_mlp]`. A
    200k-token corpus as ONE document would be 6.5 GB at d_mlp=8192, and
    nothing in a .txt file stops it being one document."""
    monkeypatch.setattr(neurons, "MAX_SEQUENCE_TOKENS", 3)
    model, block, tok = rig()
    stats, _, _ = neurons.sweep(
        model, block, tok, ["0 1 2 3 4", "0 1"], device="cpu", layer=0, sample_rows=8
    )
    assert stats.n_tokens == 5  # 3 from the first, 2 from the second
    assert stats.n_sequences_cut == 1
    assert stats.n_tokens_dropped == 2
    assert "2 tokens were dropped off the end" in stats.means()


def test_the_corpus_cap_names_how_many_sequences_were_never_read(monkeypatch):
    """EVERY CAP IS REPORTED, never merely applied. A truncated corpus that
    does not say so turns 'this neuron never fired' into a claim about text
    nobody read."""
    monkeypatch.setattr(neurons, "MAX_TOKENS", 9)
    model, block, tok = rig()
    stats, _, _ = sweep_it(model, block, tok)
    assert stats.truncated is True
    assert stats.n_sequences < stats.n_sequences_offered == len(CORPUS)
    assert "were not read at all" in stats.means()


def test_the_first_sequence_is_read_even_when_it_alone_exceeds_the_cap(monkeypatch):
    """Otherwise a corpus that is one long document measures nothing at all
    and reports it as 'this neuron never fired' -- the worst possible way to
    fail, because it looks like an answer."""
    monkeypatch.setattr(neurons, "MAX_TOKENS", 2)
    model, block, tok = rig()
    stats, _, _ = neurons.sweep(
        model, block, tok, ["0 1 2 3 4"], device="cpu", layer=0, sample_rows=4
    )
    assert stats.n_tokens == 5


def test_the_sweep_and_the_evidence_pass_read_the_same_corpus_prefix(monkeypatch):
    """feature_corpus.py:359 records what happens when they do not: one
    response reported two different sizes for 'this corpus', and the firing
    rate shown for a feature was over a different denominator from the rates
    beside it. Deciding the cap inside `_stream` makes them agree by
    construction rather than by two matching constants."""
    monkeypatch.setattr(neurons, "MAX_TOKENS", 9)
    model, block, tok = rig()
    stats, _, _ = sweep_it(model, block, tok)
    shown = neurons.evidence(model, block, tok, CORPUS, 1, device="cpu")
    assert shown["n_tokens"] == stats.n_tokens


# ------------------------------------------------------------ the sample


def test_the_reservoir_takes_every_token_when_the_corpus_fits():
    """Algorithm R fills before it replaces, so a corpus smaller than the
    capacity is sampled whole and `sampled_share` reads 1.0."""
    model, block, tok = rig()
    _, _, sample = sweep_it(model, block, tok, sample_rows=100)
    assert sample.n_rows == sample.n_tokens_seen == 24
    assert sample.to_dict()["sampled_share"] == 1.0


def test_the_reservoir_is_capped_and_says_what_it_sampled_from():
    """A sample of 2,048 rows out of 2,048 tokens is the whole corpus; out of
    200,000 it is one percent of it, and the two are not the same evidence."""
    model, block, tok = rig()
    _, _, sample = sweep_it(model, block, tok, sample_rows=6)
    assert sample.n_rows == 6
    assert sample.n_tokens_seen == 24
    assert sample.to_dict()["sampled_share"] == 0.25


def test_the_reservoir_slot_is_uniform_over_the_stream():
    """Not the first N tokens: those are the first few sequences, and a
    factorisation of the first few sequences is a factorisation of whatever
    they happened to be about. Measured here rather than argued -- 2,000 runs
    of a 10-slot reservoir over a 500-token stream."""
    import random

    chosen = [0] * 500
    trials = 2000
    for trial in range(trials):
        run = random.Random(trial)
        reservoir = [None] * 10
        for index in range(500):
            slot = neurons._reservoir_slot(run, index, 10)
            if slot is not None:
                reservoir[slot] = index
        for value in reservoir:
            chosen[value] += 1

    # THE CLAIM, stated as the thing a prefix sampler would fail: the first
    # ten positions must be picked about as often as the last ten. Taking the
    # first N tokens would give the head 2,000 apiece and the tail zero.
    head, tail = sum(chosen[:10]), sum(chosen[-10:])
    assert head == pytest.approx(tail, rel=0.35), (
        f"the head of the stream was sampled {head} times against the tail's "
        f"{tail}, which is what a prefix sampler looks like"
    )

    # And no position is unreachable or dominant. The band is wide on purpose:
    # each count is Binomial(2000, 0.02), sd 6.3, and the MAXIMUM over 500
    # such bins sits around three standard deviations above the mean by
    # construction. A tighter band here would be a test of the pseudo-random
    # stream rather than of the algorithm, and it would fail on some seeds.
    expected = trials * 10 / 500
    assert min(chosen) > expected * 0.4, "some positions are near-unreachable"
    assert max(chosen) < expected * 2.0, "some positions are over-represented"


def test_the_same_seed_samples_the_same_rows():
    """A reported seed that does not reproduce the run is not a reported
    seed."""
    model, block, tok = rig()
    first = sweep_it(model, block, tok, sample_rows=6, seed=3)[2]
    second = sweep_it(model, block, tok, sample_rows=6, seed=3)[2]
    assert first.positions == second.positions
    assert torch.equal(first.values, second.values)


def test_the_sample_holds_signed_activations_so_both_policies_see_the_same_data():
    """The negative policy is applied in `decompose`, not in the sweep, so the
    two policies can be compared on the SAME activations. Clipping at capture
    time would make that comparison impossible without a second forward pass
    over the corpus."""
    model, block, tok = rig()
    _, _, sample = sweep_it(model, block, tok, sample_rows=100)
    # Neuron 3's floor, which is the deepest negative in the fixture.
    assert float(sample.values.min()) == -2.0


# ------------------------------------------------------- one neuron, close


def test_the_top_spans_are_the_real_peaks_and_carry_their_offset():
    """The offset is not derivable by searching the window for the token: a
    window containing the same token twice would highlight both, telling the
    reader the neuron fired twice there."""
    model, block, tok = rig()
    shown = neurons.evidence(model, block, tok, CORPUS, 2, device="cpu", top_k=3)
    assert shown["max_activation"] == 6.0
    assert [s["activation"] for s in shown["spans"]] == [6.0, 6.0, 6.0]
    for span in shown["spans"]:
        assert span["token"] == "<1>"
        assert (
            span["text"][span["offset"] : span["offset"] + len(span["token"])]
            == (span["token"])
        )


def test_the_span_cap_travels_beside_the_true_count():
    """EVERY CAP IS REPORTED. A list of three spans with no count beside it
    reads as three spans existing."""
    model, block, tok = rig()
    shown = neurons.evidence(model, block, tok, CORPUS, 2, device="cpu", top_k=3)
    assert len(shown["spans"]) == 3
    # Token 1 occurs 5 times across CORPUS and token 0 six times; neuron 2
    # fires on both, so eleven spans exist and three are shown.
    assert shown["n_spans_available"] == 11


def test_the_span_list_is_bounded_no_matter_how_dense_the_neuron_is():
    """Keeping one Span per firing token is fine for a sparse SAE feature and
    wrong here: a neuron at the layer median fires on about half of
    everything, so a full 200,000-token corpus would build 100,000 span
    objects -- each carrying a context string -- in order to show ten. Neuron 0
    fires on every token in this fixture, which is the dense case in
    miniature."""
    model, block, tok = rig()
    shown = neurons.evidence(model, block, tok, CORPUS, 0, device="cpu", top_k=2)
    assert shown["n_spans_shown"] == len(shown["spans"]) == 2
    assert shown["n_spans_available"] == shown["n_tokens"] == 24


def test_bounding_the_span_list_does_not_change_which_spans_come_back():
    """The bound has to be a memory change and nothing else. Neuron 5 peaks at
    9.0 on token 2 and neuron 2 at 6.0 on token 1; asking for one span must
    still return the global maximum rather than whichever one arrived first."""
    model, block, tok = rig()
    for neuron, peak in ((5, 9.0), (2, 6.0), (3, 4.0)):
        one = neurons.evidence(model, block, tok, CORPUS, neuron, device="cpu", top_k=1)
        assert one["spans"][0]["activation"] == peak == one["max_activation"]

    # And a larger request is still in descending order.
    many = neurons.evidence(model, block, tok, CORPUS, 3, device="cpu", top_k=6)
    activations = [span["activation"] for span in many["spans"]]
    assert activations == sorted(activations, reverse=True)


def test_the_histogram_covers_the_negative_lobe():
    """feature_corpus bins over [0, max] because an SAE feature cannot be
    negative. A neuron can, and a histogram that starts at zero silently drops
    the half of the distribution the NMF section then has to decide about."""
    model, block, tok = rig()
    shown = neurons.evidence(model, block, tok, CORPUS, 6, device="cpu")
    assert shown["bin_edges"][0] == -0.5
    assert sum(shown["histogram"]) == shown["n_tokens"]
    assert shown["n_negative"] == shown["n_tokens"]
    assert shown["n_fired"] == 0


def test_a_neuron_with_one_constant_value_gets_one_bin_not_an_empty_chart():
    """`torch.histc` over a zero-width range returns nothing useful. Neuron 7
    in this fixture is exactly zero everywhere."""
    model, block, tok = rig()
    shown = neurons.evidence(model, block, tok, CORPUS, 7, device="cpu")
    assert shown["histogram"] == [shown["n_tokens"]]
    assert shown["bin_edges"] == [0.0, 0.0]


def test_evidence_never_attaches_a_label():
    """NO FABRICATED LABELS. A neuron is polysemantic, so a name would assert
    the one thing this measurement cannot support -- which is a stronger
    version of the rule feature_corpus.py:32-35 states for SAE features."""
    model, block, tok = rig()
    shown = neurons.evidence(model, block, tok, CORPUS, 5, device="cpu")
    assert shown["label"] is None
    assert "label" in shown  # present and null, not quietly absent


def test_the_polysemanticity_caveat_is_on_the_result_not_in_a_footnote():
    """The panel LOOKS like the SAE panel. The one sentence that says it is a
    different instrument has to travel with the numbers, because a caller
    rendering a dict into a browser cannot render a module docstring."""
    model, block, tok = rig()
    stats, _, _ = sweep_it(model, block, tok)
    shown = neurons.evidence(model, block, tok, CORPUS, 1, device="cpu")
    weights = neurons.logit_weights(model, tok, block, 1)
    for payload in (stats.to_dict(), shown, weights):
        assert "POLYSEMANTIC" in payload["polysemantic"]
        assert "sparse autoencoders" in payload["polysemantic"]


def test_a_firing_rate_is_reported_against_the_layer_median_not_a_threshold():
    model, block, tok = rig()
    shown = neurons.evidence(model, block, tok, CORPUS, 0, device="cpu")
    assert shown["firing_rate"] == 1.0
    assert shown["layer_median_firing_rate"] is not None
    assert "median of" in shown["means"]
    assert "not against an SAE feature's" in shown["means"]


def test_a_neuron_index_outside_the_layer_is_refused_before_any_pass():
    model, block, tok = rig()
    with pytest.raises(BadRequest, match="outside this layer"):
        neurons.evidence(model, block, tok, CORPUS, 99, device="cpu")


def test_top_k_zero_is_a_bad_request_rather_than_an_empty_list():
    model, block, tok = rig()
    with pytest.raises(BadRequest, match="at least 1"):
        neurons.evidence(model, block, tok, CORPUS, 1, device="cpu", top_k=0)


# ------------------------------------------------------- the exact readout


def test_the_logit_readout_is_the_write_column_through_the_unembedding():
    """EXACT weight math, no corpus. Checked against the arithmetic done by
    hand rather than against itself."""
    model, block, tok = rig()
    out = neurons.logit_weights(model, tok, block, 3, top_k=2)
    direction = model.block.mlp.down_proj.weight[:, 3].detach()
    expected = model.head(direction).detach()
    expected = expected - expected.mean()
    best = int(torch.argmax(expected))
    assert out["promotes"][0]["token"] == f"<{best}>"
    assert out["promotes"][0]["logit"] == pytest.approx(float(expected[best]), abs=1e-5)
    assert out["exact"] is True


def test_the_logit_readout_states_that_it_ranks_rather_than_predicts():
    """The norm's real scale depends on the stream this direction would be
    added to. Applying it at unit scale is a choice, and a response that did
    not say so would be presenting a rank as an amount."""
    model, block, tok = rig()
    out = neurons.logit_weights(model, tok, block, 0)
    assert "rank tokens rather than predict logit amounts" in out["means"]
    assert "polysemantic neuron can promote several unrelated groups" in out["means"]


def test_the_two_logit_lists_can_never_overlap():
    """Caught by running the fixture rather than by reading the code. The two
    lists are taken from opposite ends of ONE ranking, so a `top_k` above half
    the vocabulary makes them overlap -- and the fixture's five-token
    vocabulary with top_k=3 published `<0>` as both promoted and suppressed,
    at -7.33584, its own logit printed twice with one of them negated. Real
    vocabularies are 32k-150k and never reach it, which is exactly why it
    would have shipped."""
    model, block, tok = rig()
    out = neurons.logit_weights(model, tok, block, 3, top_k=3)
    promoted = {entry["token"] for entry in out["promotes"]}
    suppressed = {entry["token"] for entry in out["suppresses"]}
    assert not (promoted & suppressed)
    # And the cut is REPORTED rather than made quietly.
    assert out["top_k_requested"] == 3
    assert out["top_k_applied"] == 2
    assert "rather than the 3 asked for" in out["means"]


def test_a_normal_vocabulary_is_not_cut_at_all():
    """The guard must not narrow the ordinary case. Half of 5 is 2, so top_k=2
    is the largest this fixture supports and it comes back untouched."""
    model, block, tok = rig()
    out = neurons.logit_weights(model, tok, block, 3, top_k=2)
    assert out["top_k_applied"] == out["top_k_requested"] == 2
    assert "asked for" not in out["means"]


def test_a_model_with_no_unembedding_refuses_rather_than_crashing():
    model, block, tok = rig()
    model.get_output_embeddings = lambda: None
    with pytest.raises(Refusal, match="no output embedding"):
        neurons.logit_weights(model, tok, block, 0)


# ------------------------------------------------- the non-negative decision


def test_clipping_reports_the_mass_it_discarded():
    """Clamping negatives is a decision that changes the factorisation. What
    makes it acceptable is that its cost is measured on THIS corpus rather
    than assumed small."""
    values = torch.tensor([[1.0, -3.0], [2.0, -4.0]])
    matrix, report = neurons.non_negative(values, "clip")
    assert torch.equal(matrix, torch.tensor([[1.0, 0.0], [2.0, 0.0]]))
    # 7 of 10 units of absolute mass were negative.
    assert report["discarded_mass_share"] == pytest.approx(0.7)
    assert report["columns_out"] == 2
    assert "CLIPPED TO ZERO" in report["means"]
    assert "70.00%" in report["means"] or "70.0%" in report["means"]


def test_splitting_discards_nothing_and_doubles_the_columns():
    values = torch.tensor([[1.0, -3.0]])
    matrix, report = neurons.non_negative(values, "split")
    assert matrix.tolist() == [[1.0, 0.0, 0.0, 3.0]]
    assert report["discarded_mass_share"] == 0.0
    assert report["columns_out"] == 4
    assert "SPLIT OFF rather than discarded" in report["means"]


def test_an_unknown_policy_names_the_ones_that_exist():
    with pytest.raises(BadRequest, match="clip, split") as err:
        neurons.non_negative(torch.zeros(2, 2), "abs")
    assert "doubles the columns" in err.value.sentence


def test_the_negative_report_travels_with_the_factorisation():
    """Neuron 3 goes negative on tokens 0 and 1 and positive on 3 and 4, so
    the clip policy really does throw something away here — and the amount is
    on the response rather than in this module's docstring."""
    model, block, tok = rig()
    _, rows, sample = sweep_it(model, block, tok, sample_rows=100)
    out = neurons.decompose(sample, rows, n_components=3).to_dict()
    assert out["negatives"]["policy"] == "clip"
    assert out["negatives"]["discarded_mass_share"] > 0
    assert "CLIPPED TO ZERO" in out["negatives"]["means"]


def test_which_neurons_are_eligible_depends_on_the_negative_policy():
    """The gap this test exists for: selecting on `n_fired` alone drops a
    neuron that only ever goes negative — neuron 6 in this fixture — and
    `split` exists precisely to keep that neuron's lobe. Under `clip` it
    contributes an all-zero column and is rightly dropped; under `split` its
    negative lobe is a real column and dropping it would discard data while
    the response claimed nothing had been discarded."""
    model, block, tok = rig()
    _, rows, sample = sweep_it(model, block, tok, sample_rows=100)

    clipped = neurons.decompose(sample, rows, n_components=2, negatives="clip")
    split = neurons.decompose(sample, rows, n_components=2, negatives="split")

    # Neurons 0-5 fire; 6 is negative-only; 7 is exactly zero and is never
    # eligible under either policy.
    assert clipped.n_neurons_offered == 6
    assert split.n_neurons_offered == 7


# ----------------------------------------------------------------- the NMF


def test_nmf_recovers_a_matrix_that_is_exactly_rank_two():
    """The arithmetic check. `V = W H` built by hand from two non-negative
    factors is recoverable to a residual near zero, so a residual that is not
    near zero on this input means the update rules are wrong rather than the
    data being hard."""
    w = torch.tensor([[2.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 4.0]])
    h = torch.tensor([[1.0, 2.0, 0.0], [0.0, 0.0, 5.0]])
    v = w @ h
    _, _, info = neurons.nmf(v, 2, iterations=500, seed=0)
    assert info["residual"] < 1e-3


def test_the_residual_is_reported_as_the_factorisations_own_resolution():
    """EVERY NUMBER CARRIES ITS OWN RESOLUTION. dla.py reports a
    reconstruction residual as its readability floor; the same number does the
    same job here."""
    v = torch.rand(20, 8)
    _, _, info = neurons.nmf(v, 3, iterations=50, seed=0)
    assert 0.0 <= info["residual"] <= 1.0
    assert info["iterations"] <= 50
    assert isinstance(info["converged"], bool)


def test_a_fit_that_ran_out_says_so_rather_than_reporting_it_as_converged():
    v = torch.rand(30, 10)
    _, _, info = neurons.nmf(v, 4, iterations=2, seed=0)
    assert info["iterations"] == 2
    assert info["iterations_offered"] == 2
    assert info["converged"] is False


def test_more_components_than_rank_is_refused_with_the_rank_named():
    with pytest.raises(BadRequest, match="rank is at most 3") as err:
        neurons.nmf(torch.ones(3, 5), 4)
    assert "ask for fewer components" in err.value.sentence


def test_a_negative_matrix_is_refused_and_points_at_the_policy():
    """Silently clipping inside `nmf` would hide the one decision this module
    insists on making out loud."""
    with pytest.raises(BadRequest, match="non_negative") as err:
        neurons.nmf(torch.tensor([[1.0, -1.0], [1.0, 1.0]]), 1)
    assert "reports what that policy cost" in err.value.sentence


def test_an_all_zero_matrix_is_a_refusal_not_a_meaningless_fit():
    with pytest.raises(Refusal, match="every entry in this matrix is zero"):
        neurons.nmf(torch.zeros(4, 4), 2)


def test_the_same_seed_fits_the_same_factors():
    v = torch.rand(20, 8)
    first = neurons.nmf(v, 3, iterations=40, seed=7)
    second = neurons.nmf(v, 3, iterations=40, seed=7)
    assert torch.allclose(first[1], second[1])


# ----------------------------------------------------------- the components


def test_the_components_find_the_neurons_that_actually_co_fire():
    """The fixture's structure: neurons 1 and 2 only fire on tokens 0 and 1,
    neurons 3 and 4 only on tokens 3 and 4. Whatever else the factorisation
    does, the two pairs must not end up in the same component."""
    model, block, tok = rig()
    _, rows, sample = sweep_it(model, block, tok, sample_rows=100)
    out = neurons.decompose(sample, rows, n_components=3, seed=0)

    homes = {}
    for component in out.components:
        for entry in component.neurons:
            homes.setdefault(entry["neuron"], []).append(
                (component.component, entry["loading"])
            )

    def strongest(neuron):
        return max(homes[neuron], key=lambda pair: pair[1])[0]

    assert strongest(1) == strongest(2)
    assert strongest(3) == strongest(4)
    assert strongest(1) != strongest(3)


def test_a_component_is_never_given_a_label():
    """NMF gives components, not concepts. 'These neurons co-fire on these
    spans' is the whole claim, and naming it is the reader's job."""
    model, block, tok = rig()
    _, rows, sample = sweep_it(model, block, tok, sample_rows=100)
    out = neurons.decompose(sample, rows, n_components=2).to_dict()
    for component in out["components"]:
        assert component["label"] is None
    assert "NOTHING MORE" in out["means"]


def test_each_component_carries_its_true_neuron_count_beside_the_shown_list():
    model, block, tok = rig()
    _, rows, sample = sweep_it(model, block, tok, sample_rows=100)
    out = neurons.decompose(sample, rows, n_components=2, top_neurons=1).to_dict()
    for component in out["components"]:
        assert component["n_neurons_shown"] == 1
        assert component["n_neurons_loaded"] >= component["n_neurons_shown"]


def test_a_component_span_is_a_weight_not_an_activation():
    """A shared key name would invite a reader to compare a component's weight
    on a row against a neuron's activation there. They are different
    quantities."""
    model, block, tok = rig()
    _, rows, sample = sweep_it(model, block, tok, sample_rows=100)
    out = neurons.decompose(sample, rows, n_components=2).to_dict()
    for component in out["components"]:
        for span in component["spans"]:
            assert "weight" in span
            assert "activation" not in span


def test_the_component_spans_point_at_real_sampled_rows():
    model, block, tok = rig()
    _, rows, sample = sweep_it(model, block, tok, sample_rows=100)
    out = neurons.decompose(sample, rows, n_components=2)
    for component in out.components:
        for span in component.spans:
            assert 0 <= span["sequence"] < len(CORPUS)
            assert span["token"] in sample.tokens
            assert span["text"][span["offset"] :].startswith(span["token"])


def test_the_neuron_cap_is_reported_beside_the_number_of_neurons_offered():
    """A neuron that fires rarely and sharply is exactly what this cap drops
    -- neuron 5 in this fixture. The reader has to be able to see that the
    selection happened."""
    model, block, tok = rig()
    _, rows, sample = sweep_it(model, block, tok, sample_rows=100)
    out = neurons.decompose(sample, rows, n_components=2, max_neurons=2).to_dict()
    assert out["n_cols"] == 2
    assert out["max_neurons"] == 2
    assert out["n_neurons_offered"] == 6  # neurons 6 and 7 never fired
    assert out["n_neurons_selected"] == 2
    assert "2 most active of 6 neurons" in out["means"]


def test_the_split_policy_labels_which_lobe_a_component_loaded():
    """A component here can group 'this neuron firing' with 'that neuron going
    negative', which is a different object from a co-firing group."""
    model, block, tok = rig()
    _, rows, sample = sweep_it(model, block, tok, sample_rows=100)
    out = neurons.decompose(sample, rows, n_components=3, negatives="split").to_dict()
    lobes = {
        entry["lobe"]
        for component in out["components"]
        for entry in component["neurons"]
    }
    assert lobes <= {"positive", "negative"}
    assert out["negatives"]["columns_out"] == 2 * out["negatives"]["columns_in"]


def test_the_control_is_run_and_both_residuals_are_published():
    """nullmodel.py's question, asked of a factorisation. A k-component NMF
    reduces the residual of ANY matrix, so a fit that is not compared against
    one on structureless data says nothing."""
    model, block, tok = rig()
    _, rows, sample = sweep_it(model, block, tok, sample_rows=100)
    out = neurons.decompose(sample, rows, n_components=3, seed=0).to_dict()
    assert out["control_residual"] is not None
    assert out["control_seed"] is not None
    assert out["control_margin"] == pytest.approx(
        out["control_residual"] - out["residual"], abs=1e-9
    )
    assert isinstance(out["beats_control"], bool)


def test_a_skipped_control_reads_as_unknown_and_never_as_a_pass():
    """None, not False. 'We did not check' and 'we checked and it failed' are
    different answers and only one of them is about the data."""
    model, block, tok = rig()
    _, rows, sample = sweep_it(model, block, tok, sample_rows=100)
    out = neurons.decompose(sample, rows, n_components=2, control=False)
    assert out.control_residual is None
    assert out.beats_control is None
    assert "NO CONTROL WAS RUN" in out.means()


def test_the_control_shuffle_preserves_every_columns_own_distribution():
    """It has to destroy co-firing and nothing else. A shuffle that also
    changed the values would be testing against a different matrix, and the
    comparison would be meaningless in both directions."""
    v = torch.rand(50, 6).double()
    shuffled = neurons._shuffle_columns(v, 3)
    for column in range(6):
        assert torch.allclose(
            torch.sort(v[:, column]).values, torch.sort(shuffled[:, column]).values
        )
    assert not torch.equal(v, shuffled)


def test_stability_is_measured_against_a_second_seed():
    """NMF has no unique minimiser. A low agreement means the components a
    reader is looking at are an artefact of an initialisation."""
    model, block, tok = rig()
    _, rows, sample = sweep_it(model, block, tok, sample_rows=100)
    out = neurons.decompose(sample, rows, n_components=3, seed=0).to_dict()
    assert out["stability"] is not None
    assert 0.0 <= out["stability"] <= 1.0
    assert out["stability_seed"] == 1


def test_skipped_stability_reads_as_unknown():
    model, block, tok = rig()
    _, rows, sample = sweep_it(model, block, tok, sample_rows=100)
    out = neurons.decompose(sample, rows, n_components=2, stability=False)
    assert out.stability is None
    assert "Stability was not measured" in out.means()


def test_a_layer_where_nothing_fired_refuses_rather_than_factorising_zeros():
    model, block, tok = rig()
    _, rows, sample = sweep_it(model, block, tok, sample_rows=100)
    silent = {i: row for i, row in rows.items() if row.n_fired == 0}
    with pytest.raises(Refusal, match="nothing that could co-fire"):
        neurons.decompose(sample, silent, n_components=2)


def test_decompose_refuses_anything_that_is_not_the_sweeps_sample():
    with pytest.raises(BadRequest, match="TokenSample"):
        neurons.decompose(torch.zeros(4, 4), {}, n_components=2)


def test_the_component_selection_is_reproducible_from_the_reported_seed():
    """A ranking stable only up to dict order is not reproducible, and the
    whole point of reporting a seed is reproducibility."""
    model, block, tok = rig()
    _, rows, sample = sweep_it(model, block, tok, sample_rows=100)
    first = neurons.decompose(sample, rows, n_components=3, seed=5).to_dict()
    second = neurons.decompose(sample, rows, n_components=3, seed=5).to_dict()
    assert first["components"] == second["components"]
    assert first["residual"] == second["residual"]


# --------------------------------------------------- cost before spending it


def test_the_cost_is_priced_in_forward_passes_before_a_pass_is_spent():
    """budget.py's rule: anything expensive gets a companion that prices it
    first."""
    out = neurons.cost(CORPUS, 8192)
    assert out["passes"] == len(CORPUS)
    assert out["sweep_passes"] == len(CORPUS)
    assert out["evidence_passes"] == 0


def test_the_pass_count_says_that_it_is_an_upper_bound():
    """The sweep stops at MAX_TOKENS, and which sequence that lands in depends
    on token counts nobody has without tokenising the corpus -- which is most
    of the work this is supposed to price first."""
    out = neurons.cost(CORPUS, 8192)
    assert out["passes_are_an_upper_bound"] is True
    assert "upper bound rather than the figure" in out["means"]


def test_the_byte_figures_are_labelled_as_arithmetic_rather_than_measured():
    """budget.py insists on the distinction: `max_memory_allocated` is a
    measured peak, this is a count of the arrays this file asks for."""
    out = neurons.cost(CORPUS, 8192)
    assert "not a measured peak" in out["bytes_basis"]
    # 4096 x 8192 x 4 bytes for the live sequence, exactly.
    assert out["live_sequence_bytes"] == 4096 * 8192 * 4
    assert out["reservoir_bytes"] == 2048 * 8192 * 4


def test_the_split_policy_is_priced_at_twice_the_columns():
    clip = neurons.cost(CORPUS, 8192, negatives="clip")
    split = neurons.cost(CORPUS, 8192, negatives="split")
    assert split["matrix_shape"][1] == 2 * clip["matrix_shape"][1]
    assert split["fit_bytes"] > clip["fit_bytes"]


def test_the_cost_of_a_narrow_mlp_is_bounded_by_its_own_width():
    """min(max_neurons, d_mlp): pricing 512 columns for a 64-wide MLP would
    over-refuse on exactly the small models this panel is most useful on."""
    out = neurons.cost(CORPUS, 64)
    assert out["matrix_shape"] == [2048, 64]


def test_cost_refuses_a_width_it_was_not_given():
    with pytest.raises(BadRequest, match="neuron_count"):
        neurons.cost(CORPUS, 0)


# ------------------------------------------------- what is not a number
#
# NaN and inf compare False against every bound, so a guard written as
# `if x < LIMIT` lets them straight through and the module publishes a
# confident sentence about arithmetic that never happened. Each test below is
# pinned to a way that actually happened on this fixture.


def nan_rig(column: int = 1, value: float = float("nan")):
    """The fixture with one neuron's activations replaced by NaN or inf.

    What an fp16 MLP that overflows produces, on a fixture where every other
    number is still exactly known -- so the excluded column can be checked
    against the arithmetic of the seven that remain.
    """
    model, block, tok = rig()
    table = TABLE.clone()
    table[:, column] = value
    with torch.no_grad():
        model.block.mlp.up.weight.copy_(table.T)
    return model, block, tok


def test_a_neuron_with_no_finite_activation_is_unmeasured_not_silent():
    """It used to come back BYTE-IDENTICAL to neuron 7, which is exactly 0.0
    on every token: both read `n_fired 0, mean_positive None, firing_rate
    0.0`. One of those neurons was measured and found silent; the other
    produced nothing that could be compared to zero, and a table printing the
    same row for both has published a measurement nobody took."""
    model, block, tok = nan_rig(1)
    stats, rows, _ = sweep_it(model, block, tok)

    unreadable = rows[1].to_dict()
    truly_silent = rows[7].to_dict()
    assert unreadable != truly_silent
    assert unreadable["firing_rate"] is None
    assert unreadable["max_activation"] is None
    assert unreadable["min_activation"] is None
    assert unreadable["n_finite"] == 0
    assert unreadable["n_nonfinite"] == stats.n_tokens
    # Neuron 7 IS measured, and it is silent. Zero is the right answer there.
    assert truly_silent["firing_rate"] == 0.0
    assert truly_silent["max_activation"] == 0.0
    assert truly_silent["n_finite"] == stats.n_tokens


def test_an_unreadable_neuron_is_not_counted_as_one_that_never_fired():
    """`n_never_fired` feeds the sentence 'never went positive here - that is
    NOT SEEN IN THIS CORPUS, not dead', which is a claim about the TEXT. A
    neuron whose arithmetic overflowed supports no claim about the text at
    all, and counting it there inflates the number a reader draws that
    conclusion from."""
    model, block, tok = rig()
    clean, _, _ = sweep_it(model, block, tok)
    model, block, tok = nan_rig(1)
    stats, _, _ = sweep_it(model, block, tok)

    # Neurons 6 and 7 never go positive in this fixture, with or without the
    # NaN column. The NaN column must not join them.
    assert clean.n_never_fired == 2
    assert stats.n_never_fired == 2
    assert stats.n_neurons_unmeasured == 1
    assert stats.n_nonfinite_entries == stats.n_tokens
    assert "produced no finite activation at all" in stats.means()
    assert "excluded from every count above" in stats.means()


def test_the_layer_median_is_taken_over_neurons_that_produced_a_number():
    """The median is the reference EVERY other rate is reported against, so a
    NaN column counted as a rate-0 neuron moves the number the whole table is
    read against. Measured on this fixture: 0.458333 with the column excluded,
    0.270833 with it folded in as a zero."""
    model, block, tok = rig()
    clean, _, _ = sweep_it(model, block, tok)
    model, block, tok = nan_rig(1)
    stats, _, _ = sweep_it(model, block, tok)

    assert clean.layer_median_firing_rate == pytest.approx(0.458333, abs=1e-6)
    assert stats.layer_median_firing_rate == pytest.approx(0.458333, abs=1e-6)


def test_an_infinite_activation_does_not_read_as_zero_negative_mass():
    """`negative_mass / absolute_mass` with one `inf` in the denominator is
    `finite / inf` = 0.0, and the response then states that 0.0% of the
    layer's mass was negative. The true share on the clean fixture is
    14.1414%, and neuron 6 -- which carries most of it -- is untouched by an
    overflow four columns over."""
    model, block, tok = rig()
    clean, _, _ = sweep_it(model, block, tok)
    assert clean.negative_mass_share == pytest.approx(0.141414, abs=1e-6)

    model, block, tok = nan_rig(2, float("inf"))
    stats, _, _ = sweep_it(model, block, tok)
    # Neuron 2's own mass leaves with it -- it is positive-only, so the
    # negative mass is unchanged and only the denominator shrinks.
    assert stats.negative_mass_share is not None
    assert stats.negative_mass_share > clean.negative_mass_share
    assert "0.0% of the absolute activation" not in stats.means()


def test_a_neuron_with_no_readable_activation_refuses_rather_than_torch_erroring():
    """`torch.histc` raises `RuntimeError: range of [-nan, -nan] is not
    finite`, which is a raw torch error out of a module whose whole error
    contract is Refusal or BadRequest."""
    model, block, tok = nan_rig(1)
    with pytest.raises(Refusal, match="NaN or infinite"):
        neurons.evidence(model, block, tok, CORPUS, 1, device="cpu")


class LookupMLP(torch.nn.Module):
    """`hidden = table[token]` by INDEXING rather than by a one-hot matmul.

    The main fixture reaches the table through `one_hot(ids) @ up.weight.T`,
    and a matmul propagates a NaN through the ZERO entries of the one-hot --
    `0 * nan` is `nan` -- so every token comes back NaN as soon as any token
    does. That fixture cannot express "this neuron overflowed on THIS token
    and not that one", which is the ordinary shape of an fp16 overflow.
    Indexing can.
    """

    def __init__(self, table):
        super().__init__()
        self.register_buffer("table", table)
        self.down_proj = Projection(D_MLP, D_MODEL, bias=False)

    def forward(self, ids):
        return self.down_proj(self.table[ids])


class LookupBlock(torch.nn.Module):
    def __init__(self, table):
        super().__init__()
        self.mlp = LookupMLP(table)

    def forward(self, ids):
        return self.mlp(ids)


class LookupModel(torch.nn.Module):
    def __init__(self, table):
        super().__init__()
        self.block = LookupBlock(table)
        self.norm = torch.nn.Identity()
        self.head = torch.nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, ids):
        return Out(self.head(self.block(ids)))

    def get_output_embeddings(self):
        return self.head


def lookup_rig(table):
    model = LookupModel(table).eval()
    return model, model.block, Tokenizer()


def test_evidence_reports_what_it_excluded_beside_the_rate_it_kept():
    """A cap applied and not reported is the same defect whether the cap is a
    token budget or an unreadable number. Here a quarter of this neuron's
    tokens are unreadable and the denominator has to say so."""
    table = TABLE.clone()
    table[0, :] = float("nan")  # every neuron overflows on token id 0
    model, block, tok = lookup_rig(table)
    out = neurons.evidence(model, block, tok, ["0 1 0 1", "1 1 1 1"], 1, device="cpu")
    assert out["n_tokens"] == 8
    assert out["n_nonfinite"] == 2  # the two token-0 positions
    assert out["n_finite"] == 6
    # Six readable tokens, and EVERY one of them is token 1, where this neuron
    # reads 4.0 — so six firings, not four. The first version of this test said
    # four, which is the arithmetic mistake the counts exist to make visible:
    # the rate's denominator is the readable tokens and its numerator is the
    # firings among them, and getting either wrong produces a plausible number.
    assert out["n_fired"] == 6
    assert out["firing_rate"] == pytest.approx(6 / 6, abs=1e-9)
    assert out["max_activation"] == 4.0
    assert "excluded from every count above" in out["means"]


def test_a_partly_unreadable_neuron_keeps_its_own_denominator():
    """The rate is over the tokens where this neuron produced a NUMBER.
    Dividing by the corpus size instead counts every unreadable token as a
    token it did not fire on, which is a measurement nobody took."""
    table = TABLE.clone()
    table[0, :] = float("nan")
    model, block, tok = lookup_rig(table)
    stats, rows, _ = sweep_it(model, block, tok, sample_rows=100)
    # Token 0 appears SIX times across CORPUS — 2 + 2 + 0 + 0 + 1 + 1. Counted
    # rather than recalled: the first version of this test said five, and a
    # denominator that is off by one is exactly the kind of wrong number that
    # still looks like a rate.
    assert sum(seq.split().count("0") for seq in CORPUS) == 6
    assert stats.n_tokens == 24
    assert rows[1].n_nonfinite == 6
    assert rows[1].n_finite == 18
    assert rows[1].firing_rate == pytest.approx(rows[1].n_fired / 18, abs=1e-9)
    assert stats.n_neurons_unmeasured == 0
    assert stats.n_nonfinite_entries == 6 * D_MLP


def test_a_nan_matrix_is_refused_before_it_becomes_a_nan_residual():
    """`nan < 0` is False, so a NaN matrix walks straight through a guard
    written as `if v.min() < 0` and the fit returns a NaN residual. The ORDER
    of the two checks is what this pins."""
    with pytest.raises(Refusal, match="NaN or infinite"):
        neurons.nmf(torch.full((6, 4), float("nan")).double(), 2)
    v = torch.rand(6, 4).double()
    v[2, 1] = float("inf")
    with pytest.raises(Refusal, match="NaN or infinite"):
        neurons.nmf(v, 2)


def test_a_nan_residual_is_never_published_as_the_control_not_being_beaten():
    """THE WORST ONE. `nan < nan` is False, so a fit whose arithmetic was
    entirely NaN published 'THE CONTROL WAS NOT BEATEN ... these components
    are not evidence of neurons firing together' -- the exact thing
    `beats_control`'s own docstring promises it will never do."""
    broken = neurons.Decomposition(
        components=[],
        negatives={},
        residual=float("nan"),
        control_residual=float("nan"),
        control_residuals=[float("nan")] * 3,
        control_spread=float("nan"),
        control_repeats=3,
        control_seed=977,
        stability=None,
        stability_seed=None,
        iterations=1,
        iterations_offered=1,
        converged=False,
        seed=0,
        n_rows=4,
        n_cols=4,
        n_tokens_seen=4,
        n_neurons_selected=4,
        n_neurons_offered=4,
        max_neurons=4,
        bytes_bound=0,
    )
    assert broken.beats_control is None
    assert broken.control_verdict == "not a number"
    assert "THE CONTROL COMPARISON DID NOT HAPPEN" in broken.means()
    assert "not evidence of neurons firing together" not in broken.means()


def test_an_overflowed_neuron_is_left_out_of_the_factorisation_entirely():
    """End to end, from the model an fp16 overflow produces to the response.
    An unmeasured neuron has no activity to rank on, so it cannot be selected
    -- and the rows it poisoned are still finite in every column that was."""
    model, block, tok = nan_rig(1)
    _, rows, sample = sweep_it(model, block, tok, sample_rows=100)
    out = neurons.decompose(sample, rows, n_components=2)
    loaded = {entry["neuron"] for c in out.components for entry in c.neurons}
    assert 1 not in loaded
    assert math.isfinite(out.residual)
    assert out.beats_control is not None


def test_sampled_rows_that_hold_a_nan_are_excluded_and_counted():
    """Not zeroed. Zeroing puts a token the model produced nothing readable on
    into the fit as a token where every selected neuron happened to be
    silent."""
    values = torch.rand(20, 4)
    values[3, 2] = float("nan")
    values[11, 0] = float("inf")
    sample = neurons.TokenSample(
        values=values,
        tokens=["t"] * 20,
        contexts=["c"] * 20,
        offsets=[0] * 20,
        sequences=[0] * 20,
        positions=[0] * 20,
        n_tokens_seen=20,
        capacity=20,
        seed=0,
    )
    rows = {
        i: neurons.NeuronRow(
            neuron=i,
            n_fired=19,
            n_negative=0,
            max_activation=1.0,
            min_activation=0.0,
            mean_positive=0.5,
            firing_rate=0.95,
            n_finite=20,
        )
        for i in range(4)
    }
    out = neurons.decompose(sample, rows, n_components=2, max_neurons=4)
    assert out.n_rows_nonfinite == 2
    assert out.n_rows == 18
    assert "excluded before the fit" in out.means()


def test_a_write_direction_of_nans_refuses_rather_than_ranking_them():
    """Sorting NaNs produces an arbitrary order, and the response would print
    it as this neuron's top promoted tokens."""
    model, block, tok = rig()
    with torch.no_grad():
        model.block.mlp.down_proj.weight[:, 3] = float("nan")
    with pytest.raises(Refusal, match="NaN or infinite"):
        neurons.logit_weights(model, tok, block, 3)


def test_an_all_nan_matrix_is_refused_by_the_policy_before_it_prices_one():
    """With one `inf` in it, `values.abs().sum()` is `inf` and every share
    taken against it rounds to 0.0 -- so `non_negative` reported that clipping
    discarded nothing, about a matrix nobody could measure."""
    v = torch.rand(5, 3)
    v[1, 1] = float("inf")
    with pytest.raises(Refusal, match="NaN or infinite"):
        neurons.non_negative(v, "clip")
