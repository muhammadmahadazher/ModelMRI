"""A gradient attribution whose bars do not add up is a picture of nothing.

Integrated gradients has exactly one property no intervention in this package
has: an axiom that can be checked against the run that just happened. The
attributions must sum to `f(input) - f(baseline)`, exactly in the limit and
approximately at any finite step count, so the GAP between the sum and the
measured move is the only number on the page that says whether the
approximation converged. Most of these tests are about that gap — that it is
always reported, that it is exact where the arithmetic says it must be, that it
shrinks with the step count, and that a gap too large to be a rounding error is
refused rather than drawn.

The rest are about the two things that would make this module dangerous rather
than merely wrong: `.backward()` leaving a second copy of a model's weights on
the card as `.grad` buffers, and a peak that grows with the step count.

Every fixture here is a hand-built `nn.Module` with EVERY weight laid out by
`torch.linspace`, so the arithmetic is exact and identical on every machine and
every torch version — an RNG-seeded fixture is reproducible only until the RNG
algorithm changes, and a real checkpoint could not check any of these
identities at all. That claim used to be false of `DeepToyLM`, whose layers
were left at `nn.Linear`'s random init: the memory test built on it drew a
different model every run, diverged on 9 of 12 draws, and so was red far more
often than green. A test that fails on a coin flip is not a test, and it was
the only guard on the one claim this whole module is shaped around.
"""

from __future__ import annotations

import json
import math

import pytest

from modelmri.errors import BadRequest, Refusal

torch = pytest.importorskip("torch")


VOCAB, D_MODEL, SEQ = 11, 8, 6


class _Out:
    """What a HuggingFace causal LM returns, reduced to the field read here."""

    def __init__(self, logits):
        self.logits = logits


class ToyLM(torch.nn.Module):
    """A causal LM small enough to check by hand, in three degrees of nonlinearity.

    `cumsum` over the sequence is the causal mixing — every position reads
    every position before it and none after, which is the only property of a
    decoder that matters to a path integral over the inputs. `shape` then
    decides what the function of the embeddings IS:

      "linear"    f is exactly linear, so completeness is exact for any rule
                  at any step count and the only gap is float rounding.
      "quadratic" f is exactly quadratic, so the integrand is linear in alpha
                  and the MIDPOINT rule is exact at one step while a
                  left-endpoint rule is not. This is what pins `RULE`.
      "tanh"      genuinely nonlinear, so the gap is real, large at low step
                  counts, and shrinks as the step count rises.
    """

    def __init__(self, shape: str = "linear", gain: float = 3.0):
        super().__init__()
        self.shape = shape
        self.gain = gain
        self.embed = torch.nn.Embedding(VOCAB, D_MODEL)
        self.unembed = torch.nn.Linear(D_MODEL, VOCAB, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(
                torch.linspace(-1.0, 1.0, VOCAB * D_MODEL).reshape(VOCAB, D_MODEL)
            )
            self.unembed.weight.copy_(
                torch.linspace(0.9, -0.7, VOCAB * D_MODEL).reshape(VOCAB, D_MODEL)
            )
        self.eval()

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds=None, **_ignored):
        h = inputs_embeds.cumsum(dim=1)
        if self.shape == "quadratic":
            h = h * h
        elif self.shape == "tanh":
            h = torch.tanh(self.gain * h)
        return _Out(self.unembed(h))


DEEP_SEQ = 128


class DeepToyLM(torch.nn.Module):
    """Deep enough that a retained backward graph is a measurable allocation.

    `ToyLM` is 2 tensors and 6 tokens: its activations round to nothing and a
    peak taken over it measures the allocator. This one has real layers and a
    real sequence, so `max_memory_allocated` has something to see — which is
    the only way to check the claim the module is built around, that the peak
    does not grow with the step count.

    EVERY weight is laid out by `torch.linspace`, like every other fixture in
    this file. It was not: `nn.Linear` and `nn.Embedding` were left at their
    random init, which made this a different model on every run. Measured over
    12 unseeded draws the gap share at 2 steps ran .12 to .985, and 9 of the 12
    were past the 0.4 refusal line — so the peak test raised `Diverged` before
    it ever read a peak, and the claim it exists to guard went untested for as
    long as the fixture was random.

    The scales are the other half of that: the embedding is divided by the
    sequence length because a `cumsum` over 128 positions of an O(1) ramp
    saturates every `tanh` in the stack, and a saturated fixture has no
    gradient to integrate and no move to attribute — measured, a delta of
    -0.000002 and a gap share that stopped falling with the step count. At
    these scales the same run converges: gap share 0.0073 at 2 steps.
    """

    def __init__(
        self,
        vocab: int = 512,
        d_model: int = 256,
        layers: int = 6,
        seq: int = DEEP_SEQ,
    ):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, d_model)
        self.blocks = torch.nn.ModuleList(
            torch.nn.Linear(d_model, d_model) for _ in range(layers)
        )
        self.unembed = torch.nn.Linear(d_model, vocab, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(
                torch.linspace(-1.0, 1.0, vocab * d_model).reshape(vocab, d_model) / seq
            )
            for block in self.blocks:
                block.weight.copy_(
                    torch.linspace(-1.0, 1.0, d_model * d_model).reshape(
                        d_model, d_model
                    )
                    * (3.0 / d_model)
                )
                block.bias.copy_(torch.linspace(-0.1, 0.1, d_model))
            self.unembed.weight.copy_(
                torch.linspace(0.7, -0.9, vocab * d_model).reshape(vocab, d_model)
            )
        self.eval()

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds=None, **_ignored):
        h = inputs_embeds.cumsum(dim=1)
        for block in self.blocks:
            h = torch.tanh(block(h))
        return _Out(self.unembed(h))


class DriftingLM(ToyLM):
    """Eval mode, and still not bit-reproducible — which is why the floor exists.

    `endpoint_floor` is measured by running both endpoints TWICE and reporting
    how far the repeats moved. On a deterministic CPU fixture that is always
    0.0, so the branch that consults it was never once exercised by a test.
    Here the third and fourth forward passes — the two repeats — disagree with
    the first two by exactly 2.0, giving a floor of 4.0, which is larger than
    the gap and smaller than the move.
    """

    def __init__(self, drift: float = 2.0):
        super().__init__("tanh")
        self.drift = drift
        self.calls = 0

    def forward(self, inputs_embeds=None, **kw):
        self.calls += 1
        out = super().forward(inputs_embeds=inputs_embeds, **kw)
        # A constant added to every logit: it moves the repeats and leaves
        # every gradient, and therefore every attribution, untouched.
        if self.calls > 2:
            out.logits = out.logits + self.drift
        return out


class NaNLogitsLM(ToyLM):
    """A model whose forward pass is not a number. Nothing else about it is odd."""

    def forward(self, inputs_embeds=None, **kw):
        out = super().forward(inputs_embeds=inputs_embeds, **kw)
        out.logits = out.logits * float("nan")
        return out


class _NaNBackward(torch.autograd.Function):
    """Identity forwards, NaN backwards — the half a finite endpoint check misses."""

    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def backward(ctx, grad):
        return grad * float("nan")


class NaNGradLM(ToyLM):
    """Finite at every point on the path, and every gradient along it is NaN.

    The endpoint guard cannot see this one: `f(input)` and `f(baseline)` are
    ordinary numbers and the move between them is real. It is the accumulation
    that comes back non-finite, which is the case where a bar with no number
    reaches the payload.
    """

    def forward(self, inputs_embeds=None, **kw):
        return super().forward(inputs_embeds=_NaNBackward.apply(inputs_embeds), **kw)


class ToyTokenizer:
    """Enough tokenizer for a readout: a decode and a pad id that may be None."""

    def __init__(self, pad_token_id: int | None = 3):
        self.pad_token_id = pad_token_id

    def decode(self, ids):
        return "".join(f"<{int(i)}>" for i in ids)


@pytest.fixture
def ids():
    return torch.arange(SEQ).unsqueeze(0) % VOCAB


@pytest.fixture
def tok():
    return ToyTokenizer()


# ------------------------------------------------------- completeness is the feature


def test_the_three_completeness_numbers_are_always_in_the_payload(ids, tok):
    """Not behind a flag, not only when they are bad. A reader shown a bar
    chart with no gap cannot tell a converged attribution from one that has not
    begun to settle, and both look equally convincing."""
    from modelmri import gradients

    got = gradients.integrated_gradients(
        ToyLM("linear"), tok, ids, baseline="zero", steps=8
    ).to_dict()

    c = got["completeness"]
    assert set(("sum_of_attributions", "measured_delta", "gap")) <= set(c)
    assert c["rule"] == gradients.RULE and c["steps"] == 8
    # 2e-6, and the number comes from the payload rather than from taste. All
    # THREE of these are rounded to 6 places INDEPENDENTLY, so
    # `round(a, 6) - round(b, 6)` can differ from `round(a - b, 6)` by up to
    # 1e-6 from the two operands plus 5e-7 from the result. A tolerance of
    # 1e-6 therefore sits exactly on the boundary and passes or fails on which
    # way the floats land — it failed on macos-latest/py3.12 with
    # `0.073919 == 0.07391799999999993`, having passed on three other
    # platforms in the same run. The identity being checked is that the
    # published gap is the published difference; the rounding is the
    # resolution that check has.
    assert c["gap"] == pytest.approx(
        c["measured_delta"] - c["sum_of_attributions"], abs=2e-6
    )


def test_completeness_is_exact_on_an_exactly_linear_model(ids, tok):
    """The identity this whole module rests on, checked where it must hold
    exactly: for a linear f the gradient is constant along the path, so ANY
    number of steps of ANY Riemann rule integrates it perfectly and the only
    residue is float32 rounding. A gap here would mean the accumulation itself
    is wrong, not that the approximation is coarse."""
    from modelmri import gradients

    got = gradients.integrated_gradients(
        ToyLM("linear"), tok, ids, baseline="zero", target_kind="logit", steps=1
    )
    assert abs(got.completeness.gap) < 1e-5
    assert got.completeness.verdict == "converged"


def test_the_midpoint_rule_is_exact_for_a_quadratic_at_one_step(ids, tok):
    """This is what pins `RULE`, and it is the reason the module does not use
    the left-endpoint rule that most implementations ship. For a quadratic f
    the integrand is linear in alpha, which the midpoint rule integrates
    exactly and the left-endpoint rule misses by the whole first-order term.
    Swap the rule and this test fails while every other completeness test still
    passes at a high step count — which is exactly the silent quality loss it
    exists to catch."""
    from modelmri import gradients

    got = gradients.integrated_gradients(
        ToyLM("quadratic"), tok, ids, baseline="zero", target_kind="logit", steps=1
    )
    assert abs(got.completeness.measured_delta) > 1.0, "a real move to attribute"
    assert abs(got.completeness.gap) < 1e-4


def test_the_gap_shrinks_as_the_step_count_rises(ids, tok):
    """The step count IS the approximation's resolution, which is why it is
    reported beside the gap it produced rather than left in the request. On a
    genuinely nonlinear model the gap must fall monotonically as steps rise; a
    gap that did not would mean the step count buys nothing and the number is
    decorative."""
    from modelmri import gradients

    model = ToyLM("tanh")
    gaps = [
        abs(
            gradients.integrated_gradients(
                model,
                tok,
                ids,
                baseline="zero",
                target_kind="logit",
                steps=n,
                on_gap="report",
            ).completeness.gap
        )
        for n in (2, 8, 32, 128)
    ]
    assert gaps == sorted(gaps, reverse=True), gaps
    assert gaps[0] > 10 * gaps[-1], gaps


def test_a_gap_that_is_a_large_share_of_the_delta_is_refused(ids, tok):
    """A completeness gap of 40% means the attributions do not add up to what
    happened, and drawing them anyway is the failure mode this feature would
    otherwise have. The refusal names all three numbers and a step count to
    retry with, because a refusal with no next step is just a wall."""
    from modelmri import gradients

    with pytest.raises(gradients.Diverged) as caught:
        gradients.integrated_gradients(
            ToyLM("tanh", gain=12.0),
            tok,
            ids,
            baseline="zero",
            target_kind="logit",
            steps=1,
        )
    err = caught.value
    assert isinstance(err, Refusal), "the server must answer this 409, not 500"
    assert err.suggested_steps > err.steps
    assert "steps=" in err.sentence and "completeness" in err.sentence
    assert f"{err.measured_delta:.6f}" in err.sentence


def test_a_partly_converged_run_is_called_approximate_and_not_converged(ids, tok):
    """The band between the two thresholds is not decoration. A run whose gap
    is 16% of the delta is a real attribution whose every bar is short by its
    share of that gap, and calling it "converged" or refusing it outright are
    both wrong answers. Measured: steps=4 on the tanh fixture gives a gap share
    of 0.159753."""
    from modelmri import gradients

    got = gradients.integrated_gradients(
        ToyLM("tanh"), tok, ids, baseline="zero", target_kind="logit", steps=4
    )
    assert got.completeness.verdict == "approximate"
    assert got.completeness.gap_share == pytest.approx(0.159753, abs=1e-5)
    assert "every bar below is short by its share of it" in got.completeness.sentence


def test_the_thresholds_sit_where_the_measured_ladder_says_they_do(ids, tok):
    """The two constants are placed against a convergence ladder measured on
    this fixture, not chosen: zero baseline, logit target, gap share .9879 at 1
    step, .7257 at 2, .3705 at 3, .1598 at 4, .0640 at 5, .0248 at 6, .0035 at
    8. This pins the verdict each of those earns, so moving a threshold has to
    be a decision somebody makes rather than a number that drifts."""
    from modelmri import gradients

    model = ToyLM("tanh")
    verdicts = {}
    for n in (1, 2, 3, 4, 5, 6, 8):
        got = gradients.integrated_gradients(
            model,
            tok,
            ids,
            baseline="zero",
            target_kind="logit",
            steps=n,
            on_gap="report",
        )
        verdicts[n] = (round(got.completeness.gap_share, 4), got.completeness.verdict)

    assert verdicts == {
        1: (0.9879, "diverged"),
        2: (0.7257, "diverged"),
        3: (0.3705, "approximate"),
        4: (0.1598, "approximate"),
        5: (0.0640, "approximate"),
        6: (0.0248, "converged"),
        8: (0.0035, "converged"),
    }, verdicts


def test_report_returns_the_diverged_attribution_with_the_failure_named(ids, tok):
    """Refusing is right for a chart; a caller that wants to SHOW the
    divergence needs the numbers, and it must not get them without the word."""
    from modelmri import gradients

    got = gradients.integrated_gradients(
        ToyLM("tanh", gain=12.0),
        tok,
        ids,
        baseline="zero",
        target_kind="logit",
        steps=1,
        on_gap="report",
    )
    assert got.completeness.verdict == "diverged"
    assert "do not add up" in got.completeness.sentence
    assert "not a decomposition" in got.completeness.sentence


def test_the_gap_share_is_none_rather_than_zero_when_there_is_no_move(ids, tok):
    """UNKNOWN NEVER COLLAPSES INTO ZERO. When the baseline and the input give
    the token the same score, there is no move for the attributions to be a
    share OF — and a 0.0 in `gap_share` would read as "no gap", the opposite of
    what was found."""
    from modelmri import gradients

    # The baseline IS the input: every embedding is already the pad row, so the
    # path has zero length and the delta is exactly zero.
    flat = torch.full((1, SEQ), 3, dtype=torch.long)
    got = gradients.integrated_gradients(
        ToyLM("tanh"), tok, flat, baseline="pad", target_kind="logit", steps=4
    )
    assert got.completeness.measured_delta == 0.0
    assert got.completeness.gap_share is None
    assert got.completeness.verdict == "undefined"
    assert "nothing here to attribute" in got.completeness.sentence


def test_the_delta_carries_the_resolution_of_its_own_measurement(ids, tok):
    """`measured_delta` comes from two forward passes and every other number
    here is compared against it, so it has a resolution: both endpoints are run
    twice and the disagreement is reported. On this deterministic CPU fixture
    the repeats agree exactly and the floor is 0.0 — which is a measurement,
    not an assumption, and the field has to exist for the machine where it is
    not."""
    from modelmri import gradients

    got = gradients.integrated_gradients(
        ToyLM("linear"), tok, ids, baseline="zero", steps=4
    )
    assert got.completeness.endpoint_floor == 0.0
    assert got.forward_passes == 4, "two endpoints, and both again for the floor"


def test_a_nonzero_endpoint_floor_is_measured_and_used(ids, tok):
    """The floor branch, on a fixture where the floor is not zero — which no
    test in this file ever built, so the whole feature ran unexercised. The
    repeats disagree by 2.0 each, so the floor is 4.0, and a gap under it is a
    gap that cannot be told from running the same pass twice."""
    from modelmri import gradients

    got = gradients.integrated_gradients(
        DriftingLM(drift=2.0),
        tok,
        ids,
        baseline="zero",
        target_kind="logit",
        steps=32,
        on_gap="report",
    )
    assert got.completeness.endpoint_floor == pytest.approx(4.0, abs=1e-5)
    assert got.completeness.verdict == "converged"
    assert "moved on its own" in got.completeness.sentence


def test_the_endpoint_floor_does_not_overrule_the_refusal_line(ids, tok):
    """`endpoint_floor` is a resolution, not an amnesty.

    The floor was consulted BEFORE the share thresholds, so a run whose
    attributions accounted for 27% of the measured move came back
    `verdict='converged'` — with `gap_share=0.7257` sitting in the same
    payload, and the default `on_gap='refuse'` returning it rather than
    refusing. That is the one branch the whole feature exists for and it turned
    the 40% refusal off.

    Measured on this fixture: floor 4.0, delta 5.085055, gap 3.689976, share
    0.725651. Both statements are true — the gap is inside the noise of the two
    passes that measured the move, and it is most of the move — and neither
    verdict is available, so it is `undefined` and it is refused."""
    from modelmri import gradients

    run = dict(baseline="zero", target_kind="logit", steps=2)
    reported = gradients.integrated_gradients(
        DriftingLM(drift=2.0), tok, ids, on_gap="report", **run
    )
    c = reported.completeness
    assert c.endpoint_floor == pytest.approx(4.0, abs=1e-5)
    assert c.gap_share == pytest.approx(0.725651, abs=1e-5)
    assert abs(c.gap) <= c.endpoint_floor, "the branch under test"
    assert c.verdict == "undefined", c.verdict
    assert "do not agree" in c.sentence
    assert f"{c.gap_share:.2%}" in c.sentence

    # And the default is still to refuse a chart that accounts for 27% of what
    # happened, whatever word the verdict landed on.
    with pytest.raises(gradients.Diverged) as caught:
        gradients.integrated_gradients(DriftingLM(drift=2.0), tok, ids, **run)
    assert "not scored" in caught.value.sentence

    # The same run with a floor of 0.0 is an ordinary divergence, so the
    # fixture is not smuggling the refusal in by some other route.
    with pytest.raises(gradients.Diverged):
        gradients.integrated_gradients(DriftingLM(drift=0.0), tok, ids, **run)


# ------------------------------------------------------- the baseline is the answer


def test_three_baselines_give_three_different_attributions(ids, tok):
    """The baseline is not a detail of the method — it is the point every
    attribution is relative to. If these agreed, naming it in the payload would
    be decoration."""
    from modelmri import gradients

    model = ToyLM("tanh")
    runs = {
        name: gradients.integrated_gradients(
            model,
            tok,
            ids,
            baseline=name,
            target_kind="logit",
            steps=64,
            on_gap="report",
        )
        for name in gradients.BASELINES
    }
    deltas = {n: r.completeness.measured_delta for n, r in runs.items()}
    assert len(set(round(v, 6) for v in deltas.values())) == 3, deltas
    for name, run in runs.items():
        assert run.baseline == name
        assert run.baseline_note and name.replace("zero", "zero vector") in (
            run.baseline_note + name
        )


def test_the_baseline_is_named_in_words_the_reader_can_check(ids, tok):
    """ "pad" does not name the same vector on two tokenizers, so the id is
    published rather than left for the reader to look up."""
    from modelmri import gradients

    got = gradients.integrated_gradients(
        ToyLM("linear"), tok, ids, baseline="pad", target_kind="logit", steps=4
    )
    assert "id 3" in got.baseline_note
    assert got.to_dict()["baseline_note"] == got.baseline_note


def test_the_pad_baseline_refuses_when_there_is_no_pad_token(ids):
    """attribute.py refuses token substitution outright because which token
    stands in depends on the tokenizer. This method cannot refuse a baseline —
    there must be one — so it refuses the one it cannot build, and names the
    two that need nothing from the tokenizer."""
    from modelmri import gradients

    with pytest.raises(Refusal) as caught:
        gradients.integrated_gradients(
            ToyLM("linear"), ToyTokenizer(pad_token_id=None), ids, baseline="pad"
        )
    assert "'zero'" in caught.value.sentence and "'mean'" in caught.value.sentence


# ------------------------------------------------------- it does not damage the model


def test_no_parameter_is_left_holding_a_gradient_buffer(ids, tok):
    """The single most expensive mistake this module could make. `.backward()`
    populates `.grad` on every parameter — on a 1.7B model that is a second
    copy of the weights, permanently, attached to the caller's object on a card
    chosen to just fit the first copy. `autograd.grad(scalar, inputs=[point])`
    prunes every path that does not reach the input, so nothing accumulates."""
    from modelmri import gradients

    model = ToyLM("tanh")
    gradients.integrated_gradients(
        model, tok, ids, baseline="zero", target_kind="logit", steps=16
    )
    holding = [n for n, p in model.named_parameters() if p.grad is not None]
    assert holding == [], f"left .grad buffers on {holding}"


def test_the_model_is_returned_exactly_as_it_arrived(ids, tok):
    """No eval(), no requires_grad_(False), no zero_grad(). vision_attr.py
    refuses to flip a model to eval as a side effect of drawing a picture — "a
    change to their object that they did not ask for" — and that applies with
    more force to something that builds a graph."""
    from modelmri import gradients

    model = ToyLM("tanh")
    before = {n: p.requires_grad for n, p in model.named_parameters()}
    gradients.integrated_gradients(
        model, tok, ids, baseline="zero", target_kind="logit", steps=4
    )
    assert {n: p.requires_grad for n, p in model.named_parameters()} == before
    assert model.training is False


def test_a_model_in_training_mode_is_refused_and_not_switched(ids, tok):
    """Dropout makes the same input two different answers, so every gradient
    along the path would be that noise and the completeness check would fail
    for a reason that has nothing to do with the step count."""
    from modelmri import gradients

    model = ToyLM("linear")
    model.train()
    with pytest.raises(Refusal) as caught:
        gradients.integrated_gradients(model, tok, ids)
    assert "model.eval()" in caught.value.sentence
    assert model.training is True, "refusing must not be a mutation either"


def test_it_runs_inside_the_no_grad_regime_this_package_serves_under(ids, tok):
    """Every attribution path in this repo is under torch.no_grad. This module
    is the one place a graph is built, so it has to reopen grad itself rather
    than requiring its caller to have left it open."""
    from modelmri import gradients

    with torch.no_grad():
        got = gradients.integrated_gradients(
            ToyLM("linear"), tok, ids, baseline="zero", target_kind="logit", steps=4
        )
    assert abs(got.completeness.gap) < 1e-5


def test_inference_mode_is_refused_in_a_sentence_rather_than_by_autograd(ids, tok):
    """torch.enable_grad() reopens no_grad and does NOT reopen inference_mode,
    where tensors are permanently barred from a graph. Without this check the
    failure is a torch message about inference tensors — machinery talking to
    itself, in a browser."""
    from modelmri import gradients

    with torch.inference_mode():
        with pytest.raises(Refusal) as caught:
            gradients.integrated_gradients(ToyLM("linear"), tok, ids)
    assert "inference_mode" in caught.value.sentence


# ------------------------------------------------------- caps, counts and arguments


def test_top_k_cuts_the_list_and_never_the_sum(ids, tok):
    """EVERY CAP IS REPORTED. The completeness sum is over every token, taken
    before the cut, so it stays comparable to the measured delta whatever the
    caller asked to see — and the true count travels beside the short list."""
    from modelmri import gradients

    model, kw = (
        ToyLM("linear"),
        dict(baseline="zero", target_kind="logit", steps=8, on_gap="report"),
    )
    full = gradients.integrated_gradients(model, tok, ids, **kw)
    cut = gradients.integrated_gradients(model, tok, ids, top_k=2, **kw)

    assert cut.n_listed == 2 and cut.n_tokens == SEQ == full.n_listed
    assert cut.completeness.sum_of_attributions == (
        full.completeness.sum_of_attributions
    )
    assert sum(t.attribution for t in cut.tokens) != pytest.approx(
        cut.completeness.sum_of_attributions
    )
    assert "of 6 tokens are listed" in cut.means()
    assert "still inside sum_of_attributions" in cut.means()


def test_unreadable_means_under_the_gap_and_is_counted_before_the_cut(ids, tok):
    """dla.py's floor, in this module's units: a token attributing less than
    the approximation's own error cannot be told from that error. It is not the
    same statement as "this token did not matter", and the count is taken over
    every token rather than over the strongest few — which are the least likely
    to be unreadable."""
    from modelmri import gradients

    run = dict(baseline="zero", target_kind="logit", steps=2, on_gap="report")
    full = gradients.integrated_gradients(ToyLM("tanh"), tok, ids, **run)
    cut = gradients.integrated_gradients(ToyLM("tanh"), tok, ids, top_k=1, **run)

    floor = abs(full.completeness.gap)
    assert floor > 0
    for row in full.tokens:
        assert row.unreadable == (abs(row.attribution) < floor)

    # The count published beside a one-row table is the count over all six
    # tokens, not over the one row that survived the cut. Counting after the
    # cut publishes 1 here instead of 6 — and the strongest row is the least
    # likely to be unreadable, so counting after the cut always understates.
    assert full.n_unreadable == 6 and full.n_tokens == 6
    assert cut.n_listed == 1 and cut.n_tokens == 6
    assert cut.n_unreadable == full.n_unreadable
    assert cut.n_unreadable > cut.n_listed
    assert "6 of 6 tokens attribute less than that gap" in cut.means()


def test_a_share_is_none_rather_than_zero_when_nothing_moved(tok):
    """Same rule as gap_share, one level down: with no movement at all there is
    no share to take, and 0.0 would be a measurement nobody made."""
    from modelmri import gradients

    flat = torch.full((1, SEQ), 3, dtype=torch.long)
    got = gradients.integrated_gradients(
        ToyLM("linear"), tok, flat, baseline="pad", target_kind="logit", steps=2
    )
    assert all(t.attribution == 0.0 for t in got.tokens)
    assert all(t.share is None for t in got.tokens)


def test_the_denominator_every_share_was_taken_against_is_published(ids, tok):
    """A share whose denominator is nowhere in the payload cannot be checked.
    The only other sum here is the SIGNED one, and the two differ whenever a
    token pushed the target down — which is most prompts."""
    from modelmri import gradients

    got = gradients.integrated_gradients(
        ToyLM("tanh"), tok, ids, baseline="zero", target_kind="logit", steps=32
    )
    total_abs = got.sum_of_absolute_attributions
    assert total_abs is not None and total_abs > 0
    assert total_abs != got.completeness.sum_of_attributions, "signed is not absolute"
    for row in got.tokens:
        assert row.share == pytest.approx(abs(row.attribution) / total_abs, abs=1e-5)
    assert got.to_dict()["sum_of_absolute_attributions"] == total_abs


# ------------------------------------------------------- a NaN is not a number


def test_a_non_finite_endpoint_score_is_refused_rather_than_scored(ids, tok):
    """UNKNOWN NEVER COLLAPSES INTO ZERO, and NaN is the loudest unknown.

    With NaN logits `abs(measured_delta) > endpoint_floor` is False — NaN
    compares False against every bound — so the run came back `undefined` and
    the published sentence said "The baseline and your input give this token
    the same score to within the arithmetic, so there is nothing here to
    attribute". That is a confident statement about a measurement nobody made,
    and it was reached after paying for every backward pass."""
    from modelmri import gradients

    with pytest.raises(Refusal) as caught:
        gradients.integrated_gradients(
            NaNLogitsLM(),
            tok,
            ids,
            baseline="zero",
            target_kind="logit",
            steps=4,
            target=VOCAB - 1,
            on_gap="report",
        )
    said = caught.value.sentence
    assert "not a number" in said
    assert "f(input)" in said and "f(baseline)" in said
    assert "same score" not in said, "the sentence a NaN used to produce"


def test_a_non_finite_attribution_is_never_published_as_a_readable_bar(ids, tok):
    """The half the endpoint guard cannot see: finite at both ends of the path,
    NaN in the backward pass.

    `abs(nan) < floor` is False, so every NaN bar came back `unreadable=False`
    and `n_unreadable=0` — six bars of unknown height, all marked readable, on
    a chart. A bar with no number is the thing that flag exists to mark."""
    from modelmri import gradients

    got = gradients.integrated_gradients(
        NaNGradLM(),
        tok,
        ids,
        baseline="zero",
        target_kind="logit",
        steps=2,
        on_gap="report",
    )
    assert got.n_nonfinite == SEQ, "every attribution came back non-finite"
    assert got.n_unreadable == SEQ, "and none of them can be read"
    assert all(row.unreadable for row in got.tokens)
    assert all(row.share is None for row in got.tokens)
    assert all(math.isnan(row.attribution) for row in got.tokens)

    c = got.completeness
    assert c.verdict == "undefined"
    assert c.gap_share is None
    assert "not a number" in c.sentence
    assert "same score" not in c.sentence
    assert "came back non-finite" in got.means()


def test_the_payload_is_json_even_when_a_number_is_not(ids, tok):
    """`json.dumps` writes bare `NaN` literals by default and no parser on the
    other end of the /api route accepts one — the payload silently stops being
    JSON at the first non-finite number. `None` is the word this module already
    uses for "not measured"."""
    from modelmri import gradients

    got = gradients.integrated_gradients(
        NaNGradLM(),
        tok,
        ids,
        baseline="zero",
        target_kind="logit",
        steps=2,
        on_gap="report",
    ).to_dict()

    # `allow_nan=False` is the assertion: it raises rather than writing a bare
    # `NaN` literal, so this call succeeding is the payload being real JSON.
    # (The word appears in the prose sentence, which is a string and fine.)
    json.dumps(got, allow_nan=False)
    assert got["completeness"]["gap"] is None
    assert got["sum_of_absolute_attributions"] is None
    assert all(row["attribution"] is None for row in got["tokens"])


def test_the_undefined_verdict_names_no_gap_it_did_not_measure(ids, tok):
    """`means()` appended "N of M tokens attribute less than that gap" to every
    verdict, including the one whose preceding sentence named no gap and
    measured none. A count against a quantity nobody has is worse than no
    sentence."""
    from modelmri import gradients

    flat = torch.full((1, SEQ), 3, dtype=torch.long)
    got = gradients.integrated_gradients(
        ToyLM("tanh"), tok, flat, baseline="pad", target_kind="logit", steps=4
    )
    assert got.completeness.verdict == "undefined"
    assert "attribute less than that gap" not in got.means()
    assert "nothing here to attribute" in got.means()


def test_bad_arguments_are_422_and_name_what_is_acceptable(ids, tok):
    """errors.py gives "an unknown baseline name" as its type example of a
    BadRequest: the caller has to change the call they just made, which is a
    different fix from a Refusal's "not here, not like this"."""
    from modelmri import gradients

    model = ToyLM("linear")
    cases = {
        "baseline": dict(baseline="grey"),
        "target": dict(target_kind="probability"),
        "steps": dict(steps=0),
        "max": dict(steps=gradients.MAX_STEPS + 1),
        "on_gap": dict(on_gap="shrug"),
        "top_k": dict(top_k=-1),
        "position": dict(position=SEQ),
    }
    for name, kwargs in cases.items():
        with pytest.raises(BadRequest) as caught:
            gradients.integrated_gradients(model, tok, ids, **kwargs)
        assert caught.value.sentence, name


def test_bool_is_not_an_int_here(ids, tok):
    """isinstance(True, int) is True, so `steps=True` would sail through as one
    step and report itself as "1" — a coarser approximation than anybody asked
    for, wearing the step count they did ask for."""
    from modelmri import gradients

    model = ToyLM("linear")
    for kwargs in (dict(steps=True), dict(top_k=True), dict(target=True)):
        with pytest.raises(BadRequest):
            gradients.integrated_gradients(model, tok, ids, **kwargs)


def test_a_target_outside_the_vocabulary_is_422(ids, tok):
    from modelmri import gradients

    with pytest.raises(BadRequest) as caught:
        gradients.integrated_gradients(ToyLM("linear"), tok, ids, target=VOCAB + 5)
    assert str(VOCAB) in caught.value.sentence


def test_cost_refuses_exactly_the_targets_the_run_refuses(ids, tok):
    """`cost()` is the call every docstring here tells a reader to make FIRST,
    so it must not be the one with no argument checking.

    It was: `cost(target=True)` priced a run `integrated_gradients` answers 422
    for, `cost(target=-3)` did the same, and `cost(target=99999)` raised a bare
    IndexError — a 500 with torch's own words in the body — where the run gives
    a 422 naming the vocabulary size."""
    from modelmri import gradients

    for bad in (True, -3, VOCAB, VOCAB + 5, 1.0, "cat"):
        with pytest.raises(BadRequest) as priced:
            gradients.cost(ToyLM("linear"), tok, ids, target=bad, steps=2)
        with pytest.raises(BadRequest) as ran:
            gradients.integrated_gradients(
                ToyLM("linear"), tok, ids, target=bad, steps=2, on_gap="report"
            )
        assert priced.value.sentence == ran.value.sentence, bad

    # And the id that IS in the vocabulary is priced and run alike.
    assert gradients.cost(ToyLM("linear"), tok, ids, target=VOCAB - 1, steps=2)
    assert (
        gradients.integrated_gradients(
            ToyLM("linear"), tok, ids, target=VOCAB - 1, steps=2, on_gap="report"
        ).target_token_id
        == VOCAB - 1
    )


def test_ids_that_are_not_token_ids_are_422_and_not_a_torch_message(tok):
    """Three ways to hand this the wrong prompt, and all three used to leave
    through torch: "'list' object has no attribute 'dim'", a scalar-type
    message naming argument #1, and "index out of range in self". None of those
    is a sentence anybody wrote for a reader, and all three are the caller's
    own request — which is a 422."""
    from modelmri import gradients

    model = ToyLM("linear")
    cases = {
        "not a tensor": [[0, 1, 2]],
        "float dtype": torch.zeros(1, SEQ),
        "bool dtype": torch.ones(1, SEQ, dtype=torch.bool),
        "outside the vocabulary": torch.full((1, SEQ), VOCAB + 7, dtype=torch.long),
        "negative id": torch.full((1, SEQ), -1, dtype=torch.long),
    }
    for name, bad in cases.items():
        for call in (gradients.integrated_gradients, gradients.cost):
            with pytest.raises(BadRequest) as caught:
                call(model, tok, bad, steps=2)
            assert caught.value.sentence, name
    with pytest.raises(BadRequest) as caught:
        gradients.integrated_gradients(model, tok, cases["outside the vocabulary"])
    assert str(VOCAB) in caught.value.sentence


def test_a_negative_position_is_reported_as_an_absolute_index(ids, tok):
    """The default is -1 and the payload has to say which token that was, or a
    reader cannot line the attribution up against their own prompt."""
    from modelmri import gradients

    got = gradients.integrated_gradients(
        ToyLM("linear"), tok, ids, position=-1, target_kind="logit", steps=2
    )
    assert got.position == SEQ - 1


def test_a_model_without_input_embeddings_is_refused_with_the_reason(ids, tok):
    """Token ids are not differentiable; the path runs between embedding
    vectors, and a model that exposes none has no continuous space to run it
    in. The refusal says so and points at the measurement that still works."""
    from modelmri import gradients

    class Opaque(torch.nn.Module):
        def get_input_embeddings(self):
            return None

    model = Opaque()
    model.eval()
    with pytest.raises(Refusal) as caught:
        gradients.integrated_gradients(model, tok, ids)
    assert "not differentiable" in caught.value.sentence


# ------------------------------------------------------- the price, in the right unit


def test_cost_prices_the_run_in_forward_and_backward_passes(ids, tok):
    """COST BEFORE SPENDING IT — and in the unit actually spent. A backward
    pass retains the forward's activations, so a count of "passes" that mixes
    the two is not a price. `ratio` is measured here rather than asserted to be
    the "about 2x" everyone repeats."""
    from modelmri import gradients

    priced = gradients.cost(
        ToyLM("tanh"), tok, ids, baseline="zero", target_kind="logit", steps=64
    )
    assert priced.backward_passes == 64
    assert priced.forward_passes == 4
    assert priced.step_seconds > priced.forward_seconds > 0
    assert priced.ratio > 1.0
    assert priced.estimate.passes == 64
    assert "forward-and-backward" in priced.estimate.basis
    assert any("forward-AND-backward" in n for n in priced.estimate.notes)
    assert "forward-and-backward" in priced.to_dict()["means"]


def test_cost_counts_what_the_loop_actually_holds_across_steps(ids, tok):
    """budget.project's `retained_bytes` means what a loop keeps BETWEEN
    iterations, and getting it wrong in either direction is the bug that
    module's docstring is about.

    It is FOUR [S, d] tensors and it was counted as three: the float32
    accumulator, the input embeddings, the baseline's, and `delta_emb`, their
    difference, which is a real allocation held for the whole loop and was
    missing. This asserts against the measured sizes of those four tensors —
    not against `per * 4 + 2 * per * 4`, which is the implementation's own
    expression and is why the old version of this test could not see that a
    tensor was absent from it."""
    from modelmri import gradients

    priced = gradients.cost(
        ToyLM("linear"), tok, ids, baseline="zero", target_kind="logit", steps=4
    )

    # The four tensors by name, built here and measured rather than priced.
    model = ToyLM("linear")
    with torch.no_grad():
        x = model.get_input_embeddings()(ids)
    b = torch.zeros_like(x)
    delta_emb = x - b
    accumulator = torch.zeros(SEQ, D_MODEL, dtype=torch.float32)
    live = (accumulator, x, b, delta_emb)

    assert priced.retained_bytes == sum(t.numel() * t.element_size() for t in live)
    assert priced.estimate.retained_bytes == priced.retained_bytes
    # A three-tensor count would land here, and it is what was published.
    assert priced.retained_bytes > SEQ * D_MODEL * 4 * 3


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="no allocator to read a live set from without an accelerator",
)
def test_the_retained_set_is_what_the_allocator_says_it_is(tok):
    """The same number again, measured off the allocator during the real run
    rather than off tensors a test built to look like the loop's.

    A forward pre-hook on the FIRST of the four endpoint passes reads the
    allocator at the one moment when exactly the three model-dtype tensors held
    across the whole run are alive — `x`, `b` and their difference — and
    nothing of the loop's own. The float32 accumulator is the fourth, and its
    size is measured off a real one rather than multiplied out."""
    from modelmri import gradients

    seq, d_model = 64, 256
    model = DeepToyLM(vocab=512, d_model=d_model, layers=2, seq=seq).cuda()
    ids = (torch.arange(seq).unsqueeze(0) % 512).cuda()

    # Warm the allocator and the fixture before anything is read off it.
    gradients.integrated_gradients(
        model, tok, ids, target_kind="logit", steps=2, on_gap="report"
    )

    seen = []

    def watch(_module, _args, _kwargs):
        seen.append(torch.cuda.memory_allocated())

    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    handle = model.register_forward_pre_hook(watch, with_kwargs=True)
    try:
        gradients.integrated_gradients(
            model, tok, ids, target_kind="logit", steps=3, on_gap="report"
        )
    finally:
        handle.remove()

    assert len(seen) == 4 + 3, "four endpoint passes and one per step"
    held_in_model_dtype = seen[0] - resident
    accumulator = torch.zeros(seq, d_model, dtype=torch.float32, device="cuda")
    measured = held_in_model_dtype + accumulator.numel() * accumulator.element_size()

    priced = gradients.cost(model, tok, ids, target_kind="logit", steps=3)
    assert priced.retained_bytes == measured, (priced.retained_bytes, measured)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="only an accelerator's allocator can show a view pinning its storage",
)
def test_an_endpoint_row_does_not_pin_the_whole_logits_tensor(tok):
    """`logits[0, at]` is a VIEW. Holding one row holds the entire `[1, S, V]`
    tensor it was sliced out of, for as long as the row lives — and the row
    lives across the whole accumulation loop, uncounted by `retained_bytes` and
    invisible in the payload.

    Measured on Qwen3-1.7B's vocabulary (V=151,936) at S=400 that is 243 MB
    pinned by a one-row read, on a module written for an 8 GB card. Isolated on
    a small tensor: a row of a [1,64,1024] logits tensor kept 262,144 bytes
    alive after the tensor was deleted, against the 4,096 a copy costs. The fix
    is a `.clone()` and this is what says it is still there."""
    from modelmri import gradients

    vocab, seq = 8192, 64
    model = DeepToyLM(vocab=vocab, d_model=32, layers=2, seq=seq).cuda()
    ids = (torch.arange(seq).unsqueeze(0) % vocab).cuda()

    gradients.integrated_gradients(
        model, tok, ids, target_kind="logit", steps=2, on_gap="report"
    )

    held = []

    def watch(_module, _args, _kwargs):
        held.append(torch.cuda.memory_allocated())

    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    handle = model.register_forward_pre_hook(watch, with_kwargs=True)
    try:
        gradients.integrated_gradients(
            model, tok, ids, target_kind="logit", steps=2, on_gap="report"
        )
    finally:
        handle.remove()

    logits_bytes = seq * vocab * 4
    # At the first step of the loop, nothing from the endpoint passes may still
    # be alive: a pinned logits tensor here is 2 MB against a retained set of
    # 32 KB.
    live_across_steps = held[4] - resident
    assert live_across_steps < logits_bytes / 4, (live_across_steps, logits_bytes)


def test_an_unmeasurable_peak_is_none_and_says_why(ids, tok):
    """On CPU there is no allocator high-water mark to read. `None` means "not
    measured" and a 0 would mean "this run needed no memory" — budget.py calls
    reading one as the other the bug that made 206 robot episodes show one
    video, and this is the same field."""
    from modelmri import gradients

    got = gradients.integrated_gradients(
        ToyLM("linear"), tok, ids, baseline="zero", target_kind="logit", steps=2
    )
    if got.peak_bytes is None:
        assert got.peak_note, "an unmeasured peak has to say why"
    else:
        assert got.peak_bytes >= 0 and got.peak_note


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="no allocator high-water mark to read without an accelerator",
)
def test_the_peak_does_not_grow_with_the_step_count(tok):
    """The claim the whole loop is shaped around, on the only device here that
    can check it. `steps` gradients must never coexist: they are summed into
    one buffer as they arrive and each step's graph is freed before the next
    allocates, so a 256-step run must peak exactly where a 2-step run does.

    Without this, the accumulate-and-free loop could be replaced by a list
    comprehension over steps and every other test in this file would still
    pass, right up until somebody ran it on a real model at 256 steps.
    Measured on that mutant: 2,097,664 bytes at 2 steps, 2,884,096 at 8,
    10,224,128 at 64 and 43,909,632 at 256 — 22x the flat 1,966,592 this
    reads.

    Two things about the shape. One throwaway call first, because the first
    call on a device reads five times the rest — that reading follows the first
    call and not the step count, it is the caching allocator growing its pool,
    and warming it up deliberately is better than dropping whichever reading
    turned out to be the odd one. And `on_gap='report'`, because this measures
    MEMORY: a convergence verdict must never be able to turn a peak
    measurement into a raised refusal, which is exactly how this test spent its
    life failing."""
    from modelmri import gradients

    model = DeepToyLM().cuda()
    ids = (torch.arange(DEEP_SEQ).unsqueeze(0) % 512).cuda()
    run = dict(baseline="zero", target_kind="logit", on_gap="report")

    gradients.integrated_gradients(model, tok, ids, steps=2, **run)

    peaks = {}
    for n in (2, 8, 64, 256):
        got = gradients.integrated_gradients(model, tok, ids, steps=n, **run)
        assert got.peak_bytes is not None, "cuda reports a peak; it must be read"
        peaks[n] = got.peak_bytes

    assert len(set(peaks.values())) == 1, peaks
    assert peaks[2] > 0
    assert peaks[256] == peaks[2], peaks


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="no allocator high-water mark to read without an accelerator",
)
def test_the_reported_peak_is_the_peak_of_the_whole_call(tok):
    """`peak_bytes` has to be a number a reader could reproduce by wrapping the
    call in `max_memory_allocated`, or it is a memory claim about a region
    nobody else can name.

    It was not. The peak region used to open AFTER the four endpoint passes, so
    the `[1, S, V]` logits tensor those passes allocate was in neither snapshot
    — not resident on the way in, not in the high-water mark on the way out.
    Measured on a Qwen3-sized vocabulary: a true call peak of 176,521,728
    against a reported 128,107,008, understating by 48.4 MB on a module aimed
    at an 8 GB card, and understating is the one direction budget.py says an
    estimate must never be wrong in.

    The vocabulary here is 16,384 wide so that the logits tensor is 4 MB — two
    orders of magnitude past the slack this allows."""
    from modelmri import gradients

    model = DeepToyLM(vocab=16384, d_model=64, layers=4, seq=64).cuda()
    ids = (torch.arange(64).unsqueeze(0) % 16384).cuda()

    gradients.integrated_gradients(
        model, tok, ids, target_kind="logit", steps=2, on_gap="report"
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()
    got = gradients.integrated_gradients(
        model, tok, ids, target_kind="logit", steps=4, on_gap="report"
    )
    torch.cuda.synchronize()
    true_peak = torch.cuda.max_memory_allocated() - resident

    logits_bytes = 64 * 16384 * 4
    assert true_peak > logits_bytes, "the fixture has to make the logits visible"
    assert got.peak_bytes is not None
    assert got.peak_bytes <= true_peak, (got.peak_bytes, true_peak)
    # What is excluded is only the three [S, d] tensors `_prepare` builds
    # before the region opens: 64 * 64 * 4 * 3 = 49,152 bytes.
    assert true_peak - got.peak_bytes <= 3 * 64 * 64 * 4, (got.peak_bytes, true_peak)
    assert "logits tensors included" in got.peak_note


def test_the_payload_says_it_is_not_a_causal_measurement(ids, tok):
    """vision_attr.py's argument, carried into this module's own output rather
    than left in a docstring nobody reading the panel will open: a gradient
    says what the answer was sensitive to, not what it needed."""
    from modelmri import gradients

    got = gradients.integrated_gradients(
        ToyLM("linear"), tok, ids, baseline="zero", target_kind="logit", steps=4
    ).to_dict()
    assert "NOT A CAUSAL MEASUREMENT" in got["means"]
    assert "Mask a token out" in got["means"]
    assert got["completeness"]["sentence"] in got["means"]


def test_what_the_payload_publishes_is_what_the_object_measured(ids, tok):
    """The serialised dict is what a ROUTE ships, and nothing pinned it.

    MEASURED: replacing `"retained_bytes": self.retained_bytes` with a literal
    `0` in `Cost.to_dict` left all 43 tests in this file green. The tests
    assert the ATTRIBUTE — `priced.retained_bytes` — and a reader over HTTP
    never sees the attribute. So a budget figure this module argues about at
    length, whose whole purpose is to stop a run the card cannot hold, could
    have been published as zero with the suite green.

    Every numeric field, not just that one: a serialiser is exactly the place
    a field goes missing or gets rounded to nothing, and checking one of them
    checks one of them.
    """
    from modelmri import gradients

    priced = gradients.cost(
        ToyLM("linear"), tok, ids, baseline="zero", target_kind="logit", steps=3
    )
    said = priced.to_dict()

    for field in ("steps", "backward_passes", "forward_passes", "retained_bytes"):
        assert said[field] == getattr(priced, field), (
            f"{field}: payload says {said[field]!r}, object measured "
            f"{getattr(priced, field)!r}"
        )
    # `retained_bytes` is the one with a consequence, so it is also checked for
    # being a real figure rather than a present-but-empty one.
    assert said["retained_bytes"] > 0
    assert said["basis"] == priced.basis

    # The rounded fields are allowed to lose precision and NOT allowed to lose
    # the number: `None` means this machine would not report it, and a zero
    # would say it took no time. Pinned as the EXACT rounding `to_dict`
    # performs rather than with a relative tolerance, because at these
    # magnitudes a tolerance is the wrong instrument — MEASURED on this toy,
    # `forward_seconds` is 0.000308 and `round(_, 4)` publishes 0.0003, which
    # is 2.6% off. That is fine for a displayed timing and would fail any
    # tolerance tight enough to catch a dropped field, which is what this is
    # actually for.
    # The seconds round by SIGNIFICANT FIGURES, not decimal places, and the
    # reason is a CI failure: `round(x, 4)` publishes 0.0 for anything under
    # 50 microseconds, and a toy forward pass on macos-latest/py3.12 is
    # exactly that — it came back 0.0 where this machine measures 0.000308.
    # "0.0 seconds" tells a reader the pass took no time, which is the same
    # defect as a zero standing in for an unknown, and this module publishes
    # a RATIO derived from these two.
    for field in ("forward_seconds", "step_seconds"):
        value = getattr(priced, field)
        if value is None:
            assert said[field] is None, field
        else:
            assert said[field] == gradients._seconds(value), field
            assert said[field] > 0, f"{field} rounded away to nothing"
            # Four significant figures survives a fast machine.
            assert said[field] == pytest.approx(value, rel=1e-3), field
    if priced.ratio is None:
        assert said["ratio"] is None
    else:
        assert said["ratio"] == round(priced.ratio, 2)
        assert said["ratio"] > 0
