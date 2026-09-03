# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""An SAE that reconstructs the activations can still ruin the predictions.

`SAECalibration` measures FVU and L0, and both are activation-space numbers:
they ask whether the reconstruction is close to the vector the SAE was handed.
The model does not care about that vector, it cares about the logits, and the
directions carrying the residual stream's variance are not the directions the
next token depends on. So an SAE can post an excellent FVU and still cost the
model most of its predictive loss, and nothing above `ce_recovered` in saes.py
could ever notice.

Every test here guards one of the ways the output-space number could lie:
reporting a percentage without saying which ablation floor produced it,
quoting a loss without saying what text it is a loss on, splicing the
reconstruction back in without the per-token mean that centering removed,
clamping a negative answer up to zero, or printing a ratio whose denominator
was noise.

The synthetic model below is small and deliberately PREDICTIVE. A toy with a
random readout scores about ln(vocab) on every version of the stream, which
makes `CE_ablate - CE_clean` a rounding error and every ratio here meaningless
— the exact degeneracy `ce_recovered` refuses, so the whole file would pass by
refusing rather than by measuring. So its readout is fitted by gradient descent
on its own clean mixed stream: measured on the corpus below, the clean loss is
0.5156 nats against the 2.3979 (ln 11) that zero-ablation returns, and the
denominator is a real quantity to divide by.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from modelmri import saes  # noqa: E402
from modelmri.errors import BadRequest, Refusal  # noqa: E402
from modelmri.saes import SAEHandle  # noqa: E402

D_IN = 8
VOCAB = 11


# ------------------------------------------------------------- the stand-ins


def _centered_basis(d: int, k: int) -> torch.Tensor:
    """[d, k] orthonormal columns inside the mean-zero subspace."""
    g = torch.Generator().manual_seed(0)
    p = torch.eye(d) - torch.ones(d, d) / d
    q, _ = torch.linalg.qr(p @ torch.randn(d, d, generator=g))
    return q[:, :k]


def identity_sae() -> SAEHandle:
    """Reconstructs whatever it is given, EXACTLY, in the raw convention.

    W_enc = [I, -I] and W_dec = [I; -I]: relu(a) - relu(-a) == a, and with
    b_dec zero every product in both matmuls is either the value itself or a
    literal 0, so the round trip is bit-exact rather than merely close. That is
    what makes `ce_recovered == 1.0` an assertion about the splice landing
    rather than about float tolerance.
    """
    return SAEHandle(
        repo="synthetic/identity",
        hook="blocks.0.hook_resid_pre",
        point="resid_pre",
        layer=0,
        W_enc=torch.cat([torch.eye(D_IN), -torch.eye(D_IN)], dim=1),
        b_enc=torch.zeros(2 * D_IN),
        W_dec=torch.cat([torch.eye(D_IN), -torch.eye(D_IN)], dim=0),
        b_dec=torch.zeros(D_IN),
        apply_b_dec_to_input=None,
    )


def centered_sae(b_dec_scale: float = 3.0) -> SAEHandle:
    """Exact, but only in the `centered+b_dec` convention.

    Its decoder can only emit mean-zero vectors, so it reconstructs `x - mu`
    and never `x`; its b_dec is non-zero so skipping the subtraction leaves a
    constant error. Splicing its output back WITHOUT re-adding the per-token
    mean would hand the model a stream shifted by mu at every token — which is
    exactly the drift the shared splice exists to prevent, and the reason this
    fixture is here beside the raw identity one.
    """
    q = _centered_basis(D_IN, D_IN - 1)
    g = torch.Generator().manual_seed(1)
    b_dec = torch.randn(D_IN, generator=g)
    b_dec = (b_dec - b_dec.mean()) * b_dec_scale
    return SAEHandle(
        repo="synthetic/centered",
        hook="blocks.0.hook_resid_pre",
        point="resid_pre",
        layer=0,
        W_enc=torch.cat([q, -q], dim=1),
        b_enc=torch.zeros(2 * (D_IN - 1)),
        W_dec=torch.cat([q.T, -q.T], dim=0),
        b_dec=b_dec,
        apply_b_dec_to_input=None,
    )


def lossy_sae(k: int = 6) -> SAEHandle:
    """Keeps a `k`-dimensional slice of the stream and throws the rest away.

    A real SAE's partial reconstruction, without the randomness: the answer is
    strictly between the identity's 1.0 and the garbage one's negative number,
    which is what makes it the fixture the two floors are compared on.

    `k` is 6 of the 7 dimensions the mean-zero subspace has at D_IN=8, and 7
    would be the wrong fixture rather than a stricter one — a basis spanning
    the whole subspace reconstructs the centered stream exactly and scores
    1.0, which is the identity test again under another name.
    """
    q = _centered_basis(D_IN, k)
    return SAEHandle(
        repo="synthetic/lossy",
        hook="blocks.0.hook_resid_pre",
        point="resid_pre",
        layer=0,
        W_enc=torch.cat([q, -q], dim=1),
        b_enc=torch.zeros(2 * k),
        W_dec=torch.cat([q.T, -q.T], dim=0),
        b_dec=torch.zeros(D_IN),
        apply_b_dec_to_input=None,
    )


def garbage_sae() -> SAEHandle:
    """Emits a large random direction. Worse than deleting the activation.

    Not merely "reconstructs badly": the decoder is scaled up, so the stream it
    hands back is confidently wrong rather than small, and the model built on
    it predicts worse than it does with the stream zeroed. That is the case a
    clamp at zero would hide.
    """
    g = torch.Generator().manual_seed(2)
    return SAEHandle(
        repo="synthetic/garbage",
        hook="blocks.0.hook_resid_pre",
        point="resid_pre",
        layer=0,
        W_enc=torch.randn(D_IN, 6, generator=g),
        b_enc=torch.ones(6),
        W_dec=torch.randn(6, D_IN, generator=g) * 6.0,
        b_dec=torch.zeros(D_IN),
        apply_b_dec_to_input=None,
    )


class ToyBlock(torch.nn.Module):
    """Identity, so the hooks are the only thing that changes the stream."""

    def forward(self, x):
        return x


class _Out:
    def __init__(self, logits):
        self.logits = logits


class ToyModel:
    """Logits from a CAUSAL running mean of the stream at the block.

    The mixing matters: a readout of position t alone would make the loss at t
    independent of every other token, and a corpus would then be N independent
    one-token measurements rather than a language-model loss.

    `double` calls the block twice on different tensors, which stands in for
    the failure the write-back check exists to catch — the capture records the
    first call and the edit replaces both, so the stream the model receives is
    not the stream that was read out. `deaf` ignores the block's output
    entirely while still running it, which is the hook point that does not
    matter for this text.
    """

    def __init__(self, block, embed, readout, *, double=False, deaf=False):
        self.block, self.embed, self.readout = block, embed, readout
        self.double, self.deaf = double, deaf
        self.constant = torch.linspace(-1.0, 1.0, VOCAB)

    def mixed(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.block(self.embed[ids[0]].unsqueeze(0))
        if self.double:
            h = h + self.block(self.embed[ids[0]].unsqueeze(0) * 0.5)
        n = torch.arange(1, h.shape[1] + 1, dtype=h.dtype).view(1, -1, 1)
        return h.cumsum(1) / n

    def __call__(self, ids: torch.Tensor) -> _Out:
        m = self.mixed(ids)
        if self.deaf:
            return _Out(self.constant.view(1, 1, -1).expand(1, m.shape[1], VOCAB))
        return _Out(m @ self.readout)


def toy_corpus(seed: int = 3, lengths=(9, 7, 6)) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randint(0, VOCAB, (1, n), generator=g, dtype=torch.long) for n in lengths
    ]


def _embed(seed: int = 4) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(VOCAB, D_IN, generator=g) * 5.0 + 4.0  # +4 is a real d_in mean


#: Gradient steps fitting the toy readout, and the number is load-bearing in
#: both directions. Too few and the clean pass predicts no better than the
#: uniform distribution zero-ablation returns, so the denominator vanishes and
#: every test here passes by refusing rather than by measuring. Too many and
#: the readout is sharp enough that any imperfect stream is confidently wrong:
#: measured on this corpus at 140 steps the 6-dimensional SAE below scores
#: -0.1243 against the zero floor, which is a real answer about an overfitted
#: toy and a useless fixture for comparing two floors. At 60 steps it scores
#: 0.4777 and 0.6996, and the clean loss is 0.5156 against ln(11)=2.3979.
READOUT_STEPS = 60


def toy_model(sequences, *, double=False, deaf=False, steps: int = READOUT_STEPS):
    """A block, and a model whose clean pass actually predicts this corpus.

    The readout is fitted, by gradient descent, on the model's OWN clean mixed
    stream — so `CE_clean` lands well under ln(vocab) and destroying the stream
    costs the model something real. Without that the denominator is a rounding
    error and every ratio in this file would be noise, which is precisely what
    `ce_recovered` refuses.

    No bias term, deliberately: it makes zero-ablation return exactly the
    uniform distribution, so `CE_ablate` for that floor is ln(vocab) and can be
    asserted against a closed form rather than against a recorded number.
    """
    block = ToyBlock()
    embed = _embed()
    probe = ToyModel(block, embed, torch.zeros(D_IN, VOCAB))
    rows, targets = [], []
    for ids in sequences:
        rows.append(probe.mixed(ids)[0][:-1])
        targets.append(ids[0, 1:])
    rows, targets = torch.cat(rows), torch.cat(targets)
    readout = torch.zeros(D_IN, VOCAB, requires_grad=True)
    opt = torch.optim.Adam([readout], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(rows @ readout, targets).backward()
        opt.step()
    return block, ToyModel(block, embed, readout.detach(), double=double, deaf=deaf)


def run(sequences=None, *, sae=None, floor=saes.FLOOR_ZERO, label="toy corpus", **kw):
    sequences = toy_corpus() if sequences is None else sequences
    block, model = toy_model(
        sequences, double=kw.pop("double", False), deaf=kw.pop("deaf", False)
    )
    return saes.ce_recovered(
        model,
        block,
        sequences,
        sae or identity_sae(),
        floor=floor,
        corpus_label=label,
        **kw,
    )


# --------------------------------------------------- the fixture is honest


def test_the_toy_model_actually_predicts_its_corpus():
    """Guards every other test in this file.

    If the clean pass scored like a uniform distribution, `CE_ablate -
    CE_clean` would be a rounding error, `ce_recovered` would refuse, and the
    assertions below would be passing against a refusal rather than a
    measurement. Measured here: the clean loss sits far under ln(11)=2.3979,
    which zero-ablation returns exactly because a zeroed stream makes every
    logit 0.
    """
    got = run()
    assert got.ce_clean < 1.0, f"the toy predicts nothing: CE {got.ce_clean}"
    assert abs(got.ce_ablate - math.log(VOCAB)) < 1e-4, (
        f"zero-ablating this toy must leave a uniform distribution; got {got.ce_ablate}"
    )
    assert got.denominator > 1.0


# ------------------------------------------------------------- the identity


def test_a_perfect_sae_recovers_exactly_one():
    """CE_recon == CE_clean, so the ratio is 1.0 EXACTLY, not 0.9999.

    The identity SAE's round trip is bit-exact, so any drift from 1.0 here is
    the splice, not the SAE: a reconstruction written to the wrong positions, a
    per-token mean added when the convention did not remove one, a hook landing
    on the other side of the block. An `abs(x - 1) < 1e-3` would pass through
    all three.
    """
    got = run(sae=identity_sae())
    assert got.ce_recovered == 1.0
    assert got.ce_recon == got.ce_clean
    assert got.numerator == got.denominator


def test_the_splice_re_adds_the_mean_the_convention_removed():
    """A centered SAE reconstructs `x - mu`, and `x` is what the model gets back.

    Splicing `recon` alone would shift the stream by the per-token mean at
    every token — for this corpus a mean near 4 against a stream of scale 5,
    which is not a subtle error. It would still look like a reconstruction, and
    the number would still print. This is the same `+ mu[p]` feature_ablate.py
    holds at its residual baseline, which is why the splice is shared rather
    than rewritten.
    """
    sae = centered_sae()
    got = run(sae=sae)
    assert got.calibration.convention == "centered+b_dec"
    assert got.calibration.center is True
    # Not bit-exact like the identity: this round trip is two real matmuls
    # through an orthonormal basis. Measured 1.0 to six places on this corpus.
    assert abs(got.ce_recovered - 1.0) < 1e-4, (
        f"the mean the convention removed was not re-added: {got.ce_recovered}"
    )


def test_the_convention_reported_is_the_one_the_splice_used():
    """Two numbers on one panel under two conventions describe two splices."""
    sae = centered_sae()
    got = run(sae=sae)
    assert sae.calibration is not None
    assert got.calibration.convention == sae.calibration.convention
    assert got.calibrated_here is True, "this run took the calibration"
    assert "centered+b_dec" in got.means()


def test_a_calibration_taken_elsewhere_is_reused_and_said_so():
    """The panel's convention and this number's convention must be the same one.

    Re-calibrating here on a different corpus could pick a different convention
    from the one the features panel is plotting, and both numbers would be
    correct about different splices with nothing saying so.
    """
    sae = centered_sae()
    sae.calibrate(torch.randn(20, D_IN) * 5.0 + 4.0)
    before = sae.calibration
    got = run(sae=sae)
    assert got.calibrated_here is False
    assert got.calibration is before


# ------------------------------------------------------- below zero is real


def test_a_garbage_sae_scores_below_zero_and_is_not_clamped():
    """Worse than deleting the activation is a real, reportable answer.

    A clamp at 0 would file "this reconstruction actively misleads the model"
    under the same label as "this reconstruction contributes nothing", and only
    one of those is a reason to throw the SAE away. Measured on this fixture
    the recovered figure is well below zero because the decoder hands back a
    large wrong direction rather than a small one.
    """
    got = run(sae=garbage_sae())
    assert got.ce_recovered < 0.0, f"garbage scored {got.ce_recovered}"
    assert got.ce_recon > got.ce_ablate, "the premise of the test did not hold"
    assert "below zero" in got.means()


def test_an_unusable_sae_is_measured_rather_than_refused():
    """`feature_ablate` refuses FVU >= 1; this must not.

    There, ranking the causal effects of a non-decomposition ranks arbitrary
    directions and the ranking is meaningless. Here the whole point is to say
    how badly it does, and refusing would suppress the finding.
    """
    got = run(sae=garbage_sae())
    assert got.calibration.usable is False
    assert got.calibration.fvu >= saes.FVU_UNUSABLE
    assert got.ce_recovered < 0.0


# ------------------------------------------------------ the floor is the answer


def test_the_two_floors_give_different_answers_from_the_same_losses():
    """This is the reason the floor is named in the payload.

    The clean loss and the reconstruction's loss do not depend on the floor at
    all; only the denominator does. So the same SAE on the same corpus posts
    two different percentages, both correct, and a bare "recovered 84%" cannot
    be checked against either. The three raw losses are reported so a reader
    holding the other convention can renormalise without re-running anything.
    """
    corpus = toy_corpus()
    sae_zero, sae_mean = lossy_sae(), lossy_sae()
    zero = run(corpus, sae=sae_zero, floor=saes.FLOOR_ZERO)
    mean = run(corpus, sae=sae_mean, floor=saes.FLOOR_MEAN)

    assert zero.ce_clean == mean.ce_clean
    assert zero.ce_recon == mean.ce_recon
    assert zero.ce_ablate != mean.ce_ablate
    assert zero.ce_recovered != mean.ce_recovered
    assert zero.floor == saes.FLOOR_ZERO and mean.floor == saes.FLOOR_MEAN
    # Both are between the identity's 1.0 and the garbage SAE's negative, or
    # the fixture is not a partial reconstruction and the comparison is empty.
    for got in (zero, mean):
        assert 0.0 < got.ce_recovered < 1.0, f"{got.floor}: {got.ce_recovered}"
        assert got.floor in got.means()
    # A reader holding the other normalisation can rebuild it from what is
    # published, which is the whole reason all three losses travel.
    assert (
        abs(
            (mean.ce_ablate - zero.ce_recon) / (mean.ce_ablate - zero.ce_clean)
            - mean.ce_recovered
        )
        < 1e-5
    )


def test_the_mean_floor_names_the_tokens_it_was_averaged_over():
    """And the zero floor reports None there, never 0.

    Zero would read as "the mean was taken over an empty corpus", which is a
    measurement nobody could have made. The two are different facts.
    """
    corpus = toy_corpus()
    mean = run(corpus, sae=lossy_sae(), floor=saes.FLOOR_MEAN)
    zero = run(corpus, sae=lossy_sae(), floor=saes.FLOOR_ZERO)
    assert mean.n_floor_tokens == sum(int(ids.shape[1]) for ids in corpus)
    assert zero.n_floor_tokens is None
    assert "mean-ablation" in mean.floor_means
    assert "zero vector" in zero.floor_means


def test_an_unknown_floor_names_the_ones_that_exist():
    with pytest.raises(BadRequest) as err:
        run(floor="whatever")
    message = err.value.sentence
    assert saes.FLOOR_MEAN in message and saes.FLOOR_ZERO in message
    assert "no default" in message


# --------------------------------------------------- the corpus is the answer


def test_the_corpus_is_named_counted_and_hashed():
    """ "97% recovered" and "97% on the 4,000 tokens you gave it" differ.

    The token count is of PREDICTED tokens: the first token of each sequence is
    fed and never scored, so a three-sequence corpus loses three. Reporting the
    fed count as the measured one would overstate the sample every result is
    read against.
    """
    corpus = toy_corpus()
    got = run(corpus, label="nine sentences of toy")
    assert got.corpus_label == "nine sentences of toy"
    assert got.n_sequences == len(corpus)
    assert got.n_tokens == sum(int(ids.shape[1]) - 1 for ids in corpus)
    assert got.n_tokens_seen == sum(int(ids.shape[1]) for ids in corpus)
    assert len(got.corpus_sha256) == 64
    assert "nine sentences of toy" in got.means()


def test_two_corpora_are_distinguishable_even_under_one_label():
    """The label is what the reader called it; the sha is what actually ran."""
    first = run(toy_corpus(seed=3), label="corpus")
    second = run(toy_corpus(seed=99), label="corpus")
    assert first.corpus_sha256 != second.corpus_sha256
    assert first.ce_clean != second.ce_clean


def test_an_unnamed_corpus_is_refused():
    with pytest.raises(BadRequest) as err:
        run(label="   ")
    assert "corpus_label" in err.value.sentence


def test_the_cap_is_reported_and_not_merely_applied():
    """A sequence the cap dropped was NOT MEASURED, which is not a finding."""
    corpus = toy_corpus()
    got = run(corpus, max_sequences=2)
    assert got.n_sequences == 2
    assert got.n_sequences_given == len(corpus)
    assert got.truncated is True
    assert got.n_tokens == sum(int(ids.shape[1]) - 1 for ids in corpus[:2])
    assert "NOT MEASURED" in got.means()
    assert run(corpus).truncated is False


def test_a_sequence_with_no_predicted_token_is_refused_by_index():
    """One token is fed and never scored, so it contributes nothing at all.

    Counted in the corpus and scoring zero, it would dilute every per-token
    loss reported above by a token that was never predicted.
    """
    corpus = toy_corpus()
    corpus.insert(1, torch.zeros(1, 1, dtype=torch.long))
    with pytest.raises(BadRequest) as err:
        run(corpus)
    assert "sequence 1" in err.value.sentence


# ----------------------------------------------- refusing a meaningless ratio


def test_a_degenerate_denominator_is_refused_with_its_three_losses():
    """A percentage over a denominator of zero is noise amplified to infinity.

    This model runs the block and ignores its output, which is what a hook
    point that does not matter for a given text looks like from the outside:
    all three losses come back identical and the ratio is 0/0. The refusal
    still carries the three losses, because they were really measured and they
    are the answer to "does this hook point matter here" — which is what the
    reader actually needs next.
    """
    with pytest.raises(Refusal) as err:
        run(deaf=True)
    message = err.value.sentence
    assert "no meaning" in message
    assert "blocks.0.hook_resid_pre" in message
    assert "toy corpus" in message
    # The three losses, by name, in the refusal itself.
    for word in ("model's own", "reconstruction spliced in", "floor spliced in"):
        assert word in message
    assert "predicted tokens" in message


def test_a_write_back_that_does_not_land_is_refused_before_any_ratio():
    """If the splice does not land, all three losses measure the splice.

    This model calls the block twice and the capture records the first call, so
    writing the captured stream back unchanged changes the answer. Every
    number downstream would be the difference between one block call and two.
    """
    with pytest.raises(Refusal) as err:
        run(double=True)
    message = err.value.sentence
    assert "unchanged moves its loss" in message
    assert "no hook at all" in message


# ------------------------------------------------------- cost before spending


def test_the_pass_count_is_knowable_before_it_is_spent():
    """Three per sequence plus two, and the run has to actually spend that.

    A cost function that drifts from the loop it prices is worse than none —
    it is a promise. So the projection and the measured `passes` are asserted
    equal rather than merely both existing.
    """
    corpus = toy_corpus()
    assert saes.ce_recovered_passes(1) == 5
    assert saes.ce_recovered_passes(len(corpus)) == 3 * len(corpus) + 2
    got = run(corpus)
    assert got.passes == saes.ce_recovered_passes(len(corpus))
    # The cap changes what is scored, so it changes the cost too.
    assert run(corpus, max_sequences=2).passes == saes.ce_recovered_passes(2)


def test_the_pass_count_refuses_a_flag_and_an_empty_corpus():
    """`isinstance(True, int)` is True, so a stray flag would price one sequence."""
    with pytest.raises(BadRequest):
        saes.ce_recovered_passes(True)
    with pytest.raises(BadRequest):
        saes.ce_recovered_passes(0)


def test_the_estimate_prices_a_real_iteration_on_this_machine():
    """The pass count transfers between machines; the seconds do not.

    Both come back, and the count in the estimate is the same number
    `ce_recovered` will spend — a projection built on a different formula from
    the loop is the failure `budget.probe_pass` documents.
    """
    corpus = toy_corpus()
    block, model = toy_model(corpus)
    out = saes.estimate_ce_recovered_cost(
        model, block, corpus[0], identity_sae(), n_sequences=len(corpus)
    )
    assert out["passes"] == saes.ce_recovered_passes(len(corpus))
    assert out["estimate"]["passes"] == out["passes"]
    assert out["probe"]["seconds"] >= 0.0
    assert out["probed_sequence_length"] == int(corpus[0].shape[1])
    assert str(out["passes"]) in out["means"]


def test_the_price_and_the_gate_come_from_the_same_arithmetic():
    """One threshold, published beside the count it applies to.

    A panel or a CLI that recomputed `needs_confirmation` would be a second
    copy of the threshold, and the copy is what drifts from the thing it is
    supposed to describe.
    """
    priced = saes.ce_recovered_price(10)
    assert priced["passes"] == saes.ce_recovered_passes(10) == 32
    assert priced["n_sequences"] == 10
    assert priced["confirm_above"] == saes.CE_CONFIRM_ABOVE_PASSES
    assert priced["needs_confirmation"] is False
    assert "32" in priced["means"]

    over = saes.ce_recovered_price(saes.CE_CONFIRM_ABOVE_PASSES)
    assert over["passes"] > saes.CE_CONFIRM_ABOVE_PASSES
    assert over["needs_confirmation"] is True
    assert "max_sequences" in over["means"]


def test_a_corpus_over_the_gate_is_refused_and_confirming_returns_the_price():
    """The gate names the count, both flags, and the cheaper alternative.

    A refusal that states the problem and no next step is a wall, and
    `max_sequences` is the next step that reprices exactly — which is why the
    same call with a cap under the gate has to come back rather than refuse.
    """
    n = saes.CE_CONFIRM_ABOVE_PASSES
    with pytest.raises(Refusal) as caught:
        saes.confirm_ce_recovered(n)
    said = caught.value.sentence
    assert "forward passes" in said
    assert "`confirm: true`" in said and "--yes" in said
    assert "max_sequences" in said

    assert saes.confirm_ce_recovered(n, confirm=True)["passes"] == (
        saes.ce_recovered_passes(n)
    )
    # Under the gate it never asks, which is what stops a ten-line paste
    # needing a flag.
    assert saes.confirm_ce_recovered(1)["needs_confirmation"] is False


# ---------------------------------------------------------- the receipt itself


def test_every_published_number_travels_with_what_it_is_a_number_of():
    """The payload has to survive `asdict` and carry its own sentence.

    The server returns these through a JSON encoder, so a field that is only a
    property never reaches the browser and the panel would have to carry its
    own copy of the arithmetic — putting the decision about what a number means
    in the one place that cannot measure it.
    """
    got = run(sae=lossy_sae())
    payload = got.to_dict()
    for key in (
        "ce_clean",
        "ce_recon",
        "ce_ablate",
        "ce_recovered",
        "numerator",
        "denominator",
        "floor",
        "floor_means",
        "corpus_label",
        "corpus_sha256",
        "n_tokens",
        "passes",
        "means",
    ):
        assert key in payload, key
    # The activation-space half travels with the output-space half, nested and
    # already flattened for the encoder.
    assert isinstance(payload["calibration"], dict)
    assert "fvu" in payload["calibration"] and "l0" in payload["calibration"]
    # The resolution the denominator is read against, measured on this run.
    assert payload["splice_deviation_nats"] == 0.0
    assert payload["replay_deviation_nats"] == 0.0
