"""What the answer was sensitive to — and the arithmetic that says how much of it we captured.

Every other attribution in this package is an intervention. `ablate.py` cuts a
head out and measures what the answer did. `attribute.py` masks a token out of
every later position's attention and measures what the answer did.
`vision_attr.py` slides an occluder over the image and measures what the answer
did. That module's own docstring says why it stays that way:

    a gradient says what the output was *sensitive to* in the limit of an
    infinitesimal nudge, and that is not an answer to "would it still say cat
    if that were not there"

and it keeps a `no_grad` regime deliberately, "which is also why this cannot be
turned into a gradient method by accident". That argument is right. This module
does not overturn it — it is the one place in the package where a graph is
built, it is opt-in, it is never on an intervention path, and it imports
nothing from `runtime.py` so the serving regime stays `no_grad` whatever this
file does.

What the argument constrains is what THIS may claim. So, plainly:

**An integrated gradient is not a causal measurement.** A token with a large
attribution here is a token the target was sensitive to along a straight line
from a baseline to the input. It is not a token the model needed. If you want
"would it still answer that", mask it out — `attribute.rank_tokens` runs that
experiment and returns a KL. The two will disagree, and neither is a correction
for the other; the same relationship `dla.py` has with `ablate.py`.

## Then what is this for

One thing gradients have that no intervention in this package has: an axiom
that can be CHECKED against the run that just happened.

Integrated gradients satisfies completeness — the attributions sum to
`f(input) - f(baseline)` exactly, in the limit of infinitely many steps. Any
real run takes finitely many, so the sum misses. That miss is not a nuisance to
be hidden; it is the only number on the page that says whether the attribution
converged. A reader shown a bar chart with no gap cannot tell a converged
attribution from one where the Riemann sum has not begun to settle, and both
look exactly as convincing.

So `sum_of_attributions`, `measured_delta` and `gap` are in the payload always,
never behind a flag, and a gap that is a large fraction of the delta is refused
rather than drawn. This is the same shape `dla.py` uses for its own
decomposition: it reports a reconstruction residual, it USES that residual as
the floor below which a bar is "unreadable" rather than "small", and it refuses
outright when the approximation it is built on does not hold. Here the gap is
that residual and it does both jobs.

## The delta has a resolution too, and it is measured rather than assumed

`measured_delta` comes from two forward passes, and a forward pass is not
bit-reproducible on every accelerator. So both endpoints are run TWICE and the
disagreement between the repeats is reported as `endpoint_floor` — the same
trick `ablate.rank_heads` uses to find out what zero looks like, applied to the
one quantity everything else here is compared against. A gap under that floor
is a gap that cannot be told from running the same pass twice.

## The baseline is part of the answer, not a detail of the method

There is no neutral input. A zero embedding is a point the model has never
seen; the pad embedding is a real token with real meaning to the model; the
mean embedding is the centroid of a vocabulary the model does not treat as a
word. Each gives different attributions for the same prompt, because each asks
a different question: "compared to nothing", "compared to filler", "compared to
the average word". `vision_attr.py` says this about its occluder fill — grey,
black, white and the image mean are four baselines and "there is no neutral
one" — and it is the same statement here. `attribute.py` went further and
refused token substitution entirely, because which token stands in depends on
the tokenizer. That refusal is not available to this method: integrated
gradients is defined by a path from a baseline, so there must be one. What is
available is naming it, so `baseline` is in the payload and in `means()`, and
the three are offered by name rather than one being chosen silently.

Every position is replaced, including the chat template's own delimiters and
any BOS. That is a real distortion on a templated prompt — the path passes
through inputs that are not sentences — and it is the honest reading of "the
baseline is the point the attribution is relative to".

## It works on the input EMBEDDINGS, because token ids are not differentiable

There is no derivative with respect to "token 4382". The path runs through the
continuous embedding space between the baseline's embeddings and the input's,
and every point on it except the endpoints is a vector that corresponds to no
token at all. A per-token number here is the sum over that token's embedding
dimensions, not a property of the token in the vocabulary.

## Memory: this is the largest thing this package does

A backward pass retains every activation the forward produced, which is exactly
what `no_grad` exists to avoid — so one step of this costs materially more than
one forward pass of the same model, and neither the time nor the peak
transfers. `budget.py`'s whole argument is that a cost figure from somebody
else's card is worthless; a cost figure in the wrong UNIT is worse. So `cost()`
below prices this in FORWARD-AND-BACKWARD passes, times one of those measured
on this machine right now, and it separately times a forward-only pass so the
ratio between them is a measured number rather than the "about 2x" everyone
repeats. Measured on a 6-layer toy on cuda, S=128, five `cost()` calls back to
back: 3.48, 4.46, 2.35, 2.54, 4.07. One probe is one sample, budget.py says so
about its own, and the spread here is the reason the field is a ratio the
caller can see rather than a constant baked into a formula.

**THE PEAK IS FLAT IN THE STEP COUNT, and that is the whole design.** Measured
on that same toy (cuda, S=128, d=256): 1,966,592 bytes allocated by PyTorch for
the accumulation loop at 2 steps, and the identical 1,966,592 at 8, at 64 and
at 256. Three things buy that:

  * The accumulation is step by step. `steps` gradients are summed into one
    `[S, d]` float32 buffer as they arrive; no list of per-step gradients is
    ever held, so the peak does not grow with `steps`.
  * `torch.autograd.grad(scalar, inputs=[point])`, never `scalar.backward()`.
    `.backward()` populates `.grad` on every parameter of the model — for a
    1.7B model that is a second copy of the weights, permanently, on a card
    that was chosen to just fit the first copy — and it would do it to the
    caller's model object rather than to anything this function owns.
  * The graph is freed each step (`retain_graph` left False) and the step's
    tensors are dropped before the next one allocates.

The peak is therefore: one forward's retained activations, plus the `[S, d]`
accumulator, the input embeddings and the baseline embeddings — three tensors
of `S * d * 4` bytes, which `cost()` reports as `retained_bytes` because that
is what `budget.project` means by the word.

One reading in that run was not flat: 10,486,272 bytes, five times the rest.
It follows the FIRST call on the device and not the step count — re-run in the
order 256, 64, 8, 2 and the large reading moves to 256 — so it is the caching
allocator growing its pool, the same warm-up budget.py measured on its own
probe (3.05 s, then 0.80, then 0.78 for identical work). A first reading here
is a reading of the allocator.

## Nothing here mutates the model

Not `eval()`, not `requires_grad_(False)`, not `zero_grad()`. `vision_attr.py`
refuses to flip a model to eval as a side effect of drawing a picture —
"a change to their object that they did not ask for" — and the same rule
applies with more force to something that could leave gradient buffers behind.
A model in training mode is refused here for the same reason it is refused
there: dropout makes the same input two different answers, and every gradient
would be that noise.
"""

from __future__ import annotations

import math
import operator
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .errors import BadRequest, Refusal

# The three points the path may start from. Named, never inferred: see the
# module docstring. "zero" is the literature's default and is the only one that
# needs nothing from the tokenizer; "pad" is a real token the model has seen;
# "mean" is the vocabulary centroid.
BASELINES = ("zero", "pad", "mean")

# What scalar the attributions are OF. Completeness holds for either — it is a
# statement about a path integral of any differentiable scalar — but they are
# different quantities and the payload says which.
#
# "logprob" is the default because a raw logit is not shift-invariant and
# softmax is: dla.py shift-corrects every component against its own vocabulary
# mean for exactly this reason, and an attribution to a logit includes whatever
# moved the whole vocabulary together, which moved nothing. log p is also in
# nats, the unit `ablate.kl_nats` and `attribute.rank_tokens` already report in.
TARGETS = ("logprob", "logit")

# The midpoint rule, stated because the payload reports it and a test pins it.
# Left-endpoint Riemann is the common implementation and it is strictly worse
# at the same cost: the midpoint rule integrates any linear integrand exactly,
# so completeness is exact for a quadratic model at ONE step, while
# left-endpoint is exact only for a model that is already linear. Measured on
# the quadratic fixture in tests/test_gradients.py at steps=1: midpoint gap
# -0.000008, left-endpoint gap 52.128719 against a delta of 52.128719 — the
# whole answer, missed, at the same price.
#
# It also never evaluates the baseline itself, which matters for the 'zero'
# baseline: at a zero embedding a saturating model can have a gradient of
# almost nothing, and a rule that weights that point is a rule that starts from
# no information.
RULE = "midpoint Riemann"

DEFAULT_STEPS = 32

# The two thresholds on the completeness gap, placed against a measured
# convergence ladder rather than chosen. On the tanh fixture (S=6, d=8,
# float32, zero baseline, logit target) the gap share ran:
#
#   steps    1      2      3      4      5      6      8     16     32
#   share  .9879  .7257  .3705  .1598  .0640  .0248  .0035  .0000  .0000
#
# Two things follow from that shape. The method goes from useless to converged
# over a very narrow range of step counts, so a threshold anywhere in the
# middle separates the same runs; and the fall is steep enough that "raise the
# steps" is always the right next sentence, which is why the refusal ends in
# one.
#
# 0.4 is the "this chart is a fabrication" line — it is the figure the roadmap
# item names, a completeness gap of 40% means the attributions do not add up to
# what happened, and on the ladder above it lands between 3 and 4 steps, i.e.
# it refuses only the step counts nobody should have asked for.
REFUSE_ABOVE_GAP_SHARE = 0.4

# 0.05 is the "converged" line, and it is deliberately not tighter. It sits
# between 5 and 6 steps on the ladder, where the gap is already falling by
# roughly 2.5x per step; a threshold below that would spend a large multiple of
# the cost to move a verdict, and the gap has its own floor anyway — see
# `endpoint_floor`, which is what the arithmetic itself can resolve. Between
# the two the run is called `approximate` and says in a sentence that every bar
# is short by its share of the gap.
WARN_ABOVE_GAP_SHARE = 0.05

# A request for more steps than this is a BadRequest naming the range. Not a
# silent cap: `EVERY CAP IS REPORTED`, and a truncated step count would be a
# quietly worse approximation wearing the number that was asked for.
#
# This is a budget line and not a convergence one, and the ladder above is why
# it can be said plainly: that fixture was inside float noise by 32 steps, and
# nothing is known here about where a real model lands. 512 stops a query
# string asking for a million forward-and-backward passes. `cost()` is the
# guard that actually knows what this machine can afford.
MAX_STEPS = 512


class Diverged(Refusal):
    """The completeness check failed: the attributions do not add up.

    A `Refusal` rather than a bug, and 409 rather than 500, because nothing
    broke — the measurement was taken and it says the approximation has not
    converged at this step count. Carries the three numbers so the caller can
    put them on screen instead of the chart, and `suggested_steps` so the
    sentence can end in something the reader can do.
    """

    def __init__(
        self,
        message: str,
        *,
        gap: float,
        measured_delta: float,
        steps: int,
        suggested_steps: int,
    ) -> None:
        super().__init__(message)
        self.gap = gap
        self.measured_delta = measured_delta
        self.steps = steps
        self.suggested_steps = suggested_steps


@dataclass
class TokenAttribution:
    """One input token's share of the move from baseline to input.

    `attribution` is signed and is summed over that token's embedding
    dimensions — a token can and does push the target DOWN. `share` is against
    the total absolute attribution, and that denominator is published as
    `sum_of_absolute_attributions` on the `Attribution` so a reader can check
    the division rather than take it. It is `None` rather than 0.0 when the
    total is itself zero, or not finite, and there is no share to take.
    """

    index: int
    token: str
    token_id: int
    # Non-finite when the backward pass produced one. It is left as the float
    # it was so nothing here rounds a NaN into a number, and `to_dict` turns it
    # into `null` rather than a bare `NaN` literal that no JSON parser accepts.
    attribution: float
    share: float | None
    # True when |attribution| is under the completeness gap, i.e. this token's
    # attribution is smaller than the error the approximation already makes.
    # NOT the same as "this token did not matter" — dla.py's `unreadable`
    # carries the identical meaning against its reconstruction residual.
    #
    # Also True when the attribution is not finite, and when the GAP is not
    # finite: `abs(nan) < floor` is False and so is `x < nan`, so a comparison
    # written the obvious way publishes every NaN bar as readable. A bar that
    # cannot be compared with the error is exactly the thing this flag exists
    # to mark.
    unreadable: bool


@dataclass
class Completeness:
    """Did the attributions add up to what actually happened?

    `sum_of_attributions` is over EVERY token and every embedding dimension,
    taken before any `top_k` cut, so it is comparable to `measured_delta`
    whatever the caller asked to see.
    """

    steps: int
    rule: str
    sum_of_attributions: float
    measured_delta: float
    gap: float
    # None when `measured_delta` is under `endpoint_floor` — there is no share
    # of a quantity we could not resolve, and 0.0 would read as "no gap".
    gap_share: float | None
    # Measured: how far the two endpoint passes moved when repeated. The
    # resolution of `measured_delta`, and the floor under which `gap` cannot be
    # told from running the same forward twice.
    endpoint_floor: float
    verdict: str  # converged | approximate | diverged | undefined
    sentence: str

    def to_dict(self) -> dict:
        return {k: _jsonable(v) for k, v in asdict(self).items()}


@dataclass
class Attribution:
    """One integrated-gradients run, with the check that says how far it got."""

    target_token: str
    target_token_id: int
    position: int
    target_kind: str
    baseline: str
    # What the baseline actually was, in words, because "pad" is not the same
    # sentence on two tokenizers and the reader cannot see the id from here.
    baseline_note: str
    completeness: Completeness
    tokens: list[TokenAttribution] = field(default_factory=list)
    # How many tokens were ATTRIBUTED against how many are carried here. A
    # `top_k` cut drops rows from the list and not from the sum.
    n_tokens: int = 0
    n_listed: int = 0
    n_unreadable: int = 0
    # How many token attributions came back non-finite. EVERY EXCLUSION IS
    # REPORTED: those rows carry no number, they are marked unreadable, and
    # `means()` says how many there were rather than letting them pass as
    # bars of unknown height.
    n_nonfinite: int = 0
    # The denominator every `share` was taken against — sum of |attribution|
    # over every token, before any `top_k` cut. Published because a share
    # whose denominator is nowhere in the payload cannot be checked: the only
    # other sum here is the SIGNED one, and the two differ whenever a token
    # pushed the target down. `None` when it is not finite.
    sum_of_absolute_attributions: float | None = None
    # Split out, because they do not cost the same and the whole point of
    # `cost()` is that they must not be added together.
    backward_passes: int = 0
    forward_passes: int = 0
    elapsed_s: float = 0.0
    peak_bytes: int | None = None
    peak_note: str = ""

    def to_dict(self) -> dict:
        return {
            **{
                k: _jsonable(v)
                for k, v in asdict(self).items()
                if k not in ("tokens", "completeness")
            },
            "completeness": self.completeness.to_dict(),
            "tokens": [
                {k: _jsonable(v) for k, v in asdict(t).items()} for t in self.tokens
            ],
            "means": self.means(),
        }

    def means(self) -> str:
        dropped = max(0, self.n_tokens - self.n_listed)
        cut = (
            ""
            if not dropped
            else (
                f"{self.n_listed} of {self.n_tokens} tokens are listed, the "
                f"strongest by magnitude; the other {dropped} were attributed "
                f"and are not shown, and they are still inside "
                f"sum_of_attributions. "
            )
        )
        # The unreadable count is a count against the GAP, so it is only a
        # sentence when a gap was scored. In the `undefined` verdict no gap was
        # named in the sentence before it and none could be measured, and "0 of
        # 6 tokens attribute less than that gap" there is a number about a
        # quantity nobody has.
        if self.completeness.verdict == "undefined":
            readable = (
                f"No gap was scored, so no bar here can be told from the "
                f"approximation's own error; all {self.n_tokens} are marked "
                f"unreadable for that reason and not because they are small. "
                if self.n_unreadable == self.n_tokens and self.n_tokens
                else ""
            )
        else:
            readable = (
                f"{self.n_unreadable} of {self.n_tokens} tokens attribute less "
                f"than that gap, so their share cannot be told from the "
                f"approximation's own error. "
            )
        nonfinite = (
            ""
            if not self.n_nonfinite
            else (
                f"{self.n_nonfinite} of {self.n_tokens} attributions came back "
                f"non-finite and carry no number at all — they are published as "
                f"null rather than as a bar of unknown height. "
            )
        )
        return (
            cut + f"Integrated gradients of {self.target_kind} for "
            f"{self.target_token!r} at position {self.position}, along a "
            f"straight path from the {self.baseline!r} baseline "
            f"({self.baseline_note}) to your input, summed over each token's "
            f"embedding dimensions. {self.completeness.sentence} "
            + readable
            + nonfinite
            + "THIS IS NOT A CAUSAL MEASUREMENT: it "
            "says what the answer was sensitive to along that path, not what "
            "it needed. Mask a token out and re-run to ask the other "
            "question — the two disagree and neither corrects the other."
        )


@dataclass
class Cost:
    """What this run will cost, priced in the units it actually spends.

    `forward_seconds` and `step_seconds` are both measured here, one probe
    each, so `ratio` is a number from this machine rather than the "about
    twice" that gets repeated about backward passes. Measured on the CPU
    fixture, two runs back to back: 4.39x and 4.09x. It is not a constant, it
    is not 2, and it is not carried between runs.

    `estimate` is `budget.Estimate` unchanged, so the verdict, the fraction of
    free memory and the refusal threshold are the ones the rest of the package
    already uses — see budget.py. What is different is the unit it is counting:
    `passes` on that object is a count of forward-AND-backward steps.
    """

    steps: int
    backward_passes: int
    forward_passes: int
    forward_seconds: float | None
    step_seconds: float | None
    ratio: float | None
    retained_bytes: int
    estimate: Any
    basis: str = "one forward probe and one forward-and-backward probe on this machine"

    def to_dict(self) -> dict:
        d = {
            "steps": self.steps,
            "backward_passes": self.backward_passes,
            "forward_passes": self.forward_passes,
            "forward_seconds": (
                None if self.forward_seconds is None else round(self.forward_seconds, 4)
            ),
            "step_seconds": (
                None if self.step_seconds is None else round(self.step_seconds, 4)
            ),
            "ratio": None if self.ratio is None else round(self.ratio, 2),
            "retained_bytes": self.retained_bytes,
            "basis": self.basis,
            "estimate": self.estimate.to_dict(),
        }
        d["means"] = (
            f"{self.backward_passes} forward-and-backward passes plus "
            f"{self.forward_passes} forward-only ones. A backward pass retains "
            f"the forward's activations, so the two are not interchangeable "
            f"units: measured here, one step cost "
            + (
                f"{self.ratio:.2f}x a forward pass"
                if self.ratio is not None
                else "an amount this machine would not report"
            )
            + ". Seconds are projected from one probe of each and are a "
            "sample, not a property of the model."
        )
        return d


def _torch():
    import torch

    return torch


def _finite(value: Any) -> bool:
    """True only for a real number.

    NON-FINITE INPUT IS NOT A NUMBER. A NaN compares False against every bound
    in both directions, so `if abs(x) < limit` lets it through as "small" and
    `if abs(x) > limit` lets it through as "fine"; an infinity passes exactly
    one of the two. Every threshold in this module goes through here first so
    the answer to "is this under the floor" is never accidentally yes.
    """
    return isinstance(value, (int, float)) and math.isfinite(value)


def _jsonable(value: Any) -> Any:
    """`None` for a float that is not a number, the value otherwise.

    `json.dumps` writes bare `NaN` and `Infinity` literals by default and no
    JSON parser on the other end of the /api route accepts either — the
    payload silently stops being JSON at the first non-finite number. `None`
    is the same word this module already uses for "not measured", which is
    what a NaN attribution is.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _grad_is_possible(torch, weight) -> None:
    """Refuse before spending anything if autograd cannot run here at all.

    This package runs `no_grad` everywhere else, and `torch.enable_grad()`
    reopens that — but it does NOT reopen `torch.inference_mode()`, which marks
    its tensors so they can never take part in a graph. A model loaded or
    called under inference mode fails deep inside autograd with a message about
    inference tensors, which is machinery talking to itself. Both halves are
    checked: the ambient mode, and the embedding weight itself, because a model
    whose weights were CREATED under inference mode carries the mark long after
    the context exited.
    """
    if torch.is_inference_mode_enabled():
        raise Refusal(
            "this call is inside torch.inference_mode(), where tensors cannot "
            "take part in a gradient graph at all. Integrated gradients needs "
            "a backward pass. Call it outside inference mode — no_grad is "
            "fine, this function reopens that itself."
        )
    if torch.is_inference(weight):
        raise Refusal(
            "this model's input embeddings are inference tensors — they were "
            "created inside torch.inference_mode(), which permanently marks "
            "them as unusable in a gradient graph. Load the model outside "
            "inference mode to attribute through it. Ablation and token "
            "masking work on it as it is, because neither needs a backward "
            "pass."
        )


def _embeddings(model):
    """The input embedding module, or a sentence saying why there is none."""
    getter = getattr(model, "get_input_embeddings", None)
    embed = getter() if callable(getter) else None
    if embed is None or getattr(embed, "weight", None) is None:
        raise Refusal(
            "this model does not expose an input embedding table, so there is "
            "no continuous space to integrate a gradient along. Token ids are "
            "not differentiable — the path integral runs between embedding "
            "vectors, and without get_input_embeddings() there are none to "
            "read. Token masking (attribute.rank_tokens) needs no embeddings "
            "and works here."
        )
    return embed


def _baseline_embeddings(name: str, tokenizer, embed, x):
    """`(tensor, sentence)` for the point the path starts from.

    The sentence is published: which baseline was used is not a detail of the
    method, it is half of what the number means, and "pad" does not name the
    same vector on two tokenizers.
    """
    torch = _torch()
    if name == "zero":
        return torch.zeros_like(x), (
            "every position replaced by the zero vector, which is not a token "
            "and is a point the model has never seen"
        )

    weight = embed.weight.detach()
    if name == "mean":
        # `dtype=` on the reduction, not `weight.float().mean(...)`. The second
        # form upcasts the whole table first, which on a 150k x 2048 bf16
        # vocabulary is a 1.2 GB temporary to produce one 2048-wide vector —
        # and this package's rule is that nothing materialises a
        # vocabulary-square or anything near it. `mean(dtype=float32)`
        # accumulates in float32 while reading the rows in place.
        centroid = weight.mean(dim=0, dtype=torch.float32)
        return centroid.to(x.dtype).expand_as(x).contiguous(), (
            f"every position replaced by the mean of all {weight.shape[0]:,} "
            "embedding rows — the vocabulary's centroid, which is not a word"
        )

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        raise Refusal(
            "this tokenizer has no pad token, so the 'pad' baseline has no "
            "vector to start from. attribute.py refuses token substitution "
            "outright for this reason — which token stands in depends on the "
            "tokenizer, and where it resolves to an end-of-document marker the "
            "attribution is about being told the document ended. Use the "
            "'zero' or 'mean' baseline, both of which need nothing from the "
            "tokenizer."
        )
    row = weight[int(pad_id)]
    return row.to(x.dtype).expand_as(x).contiguous(), (
        f"every position replaced by the embedding of the pad token "
        f"(id {int(pad_id)}), a real token this model has seen"
    )


def _scalar(logits, token_id: int, target_kind: str):
    """The differentiable quantity being attributed, upcast before the softmax.

    `.float()` for the same reason `ablate.distribution` does it: a log_softmax
    taken in bf16 over a 150k vocabulary loses more than the quantities being
    compared. The upcast is inside the graph, so the gradient comes back
    through it.
    """
    torch = _torch()
    row = logits.float()
    if target_kind == "logit":
        return row[token_id]
    return torch.log_softmax(row, dim=-1)[token_id]


def _refuse_non_finite_endpoints(f_x, f_b, f_x_again, f_b_again, target_kind: str):
    """Stop before the loop when the model's own score is not a number.

    UNKNOWN NEVER COLLAPSES INTO ZERO, and a NaN is the loudest kind of
    unknown. `measured_delta` is the quantity every attribution in this module
    is checked against; if it is NaN then `abs(measured_delta) > endpoint_floor`
    is False, the run is labelled `undefined`, and the sentence that comes out
    says the baseline and the input give this token the same score — a
    measurement nobody made, published with total confidence.

    So it is refused, by name, before `steps` forward-and-backward passes are
    spent producing bars that could not be checked against anything.
    """
    named = {
        "f(input)": f_x,
        "f(baseline)": f_b,
        "f(input) on the repeat": f_x_again,
        "f(baseline) on the repeat": f_b_again,
    }
    bad = [f"{k} = {v}" for k, v in named.items() if not _finite(v)]
    if not bad:
        return
    raise Refusal(
        f"this model returned a score that is not a number at an endpoint of "
        f"the path: {', '.join(bad)}. Completeness is a comparison against "
        f"that move, and a NaN or an infinity compares False against every "
        f"bound in both directions — so the check would not fail, it would "
        f"come back 'converged' or 'undefined' having measured nothing. "
        f"Nothing was attributed and no backward pass was spent. Run one "
        f"forward pass yourself and look at .logits before attributing "
        f"through this model: a non-finite {target_kind} usually means the "
        f"forward overflowed in a half precision the model needs more than, "
        f"or that the input embeddings already carry one."
    )


def _check_target(target, vocab_size: int | None = None) -> int | None:
    """The token id being attributed, checked identically wherever it arrives.

    `cost()` is the call the docstrings tell a reader to make FIRST, so it must
    not be the one with the weaker guard: an id that `integrated_gradients`
    answers 422 for and `cost()` accepts is a price for a run that cannot
    happen, and an id that reaches `logits[token_id]` unchecked is an
    IndexError, which is a 500 with torch's own words in it.

    Called twice on each path: once before any forward pass, where the type is
    knowable and the vocabulary is not, and once with the vocabulary in hand.
    """
    if target is None:
        return None
    # bool BEFORE int, here as everywhere: isinstance(True, int) is True, and
    # True would be read as token id 1 — a real row of a real vocabulary.
    if isinstance(target, bool):
        raise BadRequest(
            "target must be a token id or None, not a bool — True would be "
            "read as token id 1."
        )
    try:
        token_id = operator.index(target)
    except TypeError:
        raise BadRequest(
            f"target must be a token id or None, got {type(target).__name__}. "
            "A token id is an index into this model's vocabulary; there is no "
            "derivative with respect to a token string."
        ) from None
    if vocab_size is not None and not 0 <= token_id < vocab_size:
        raise BadRequest(
            f"target token id {token_id} is outside this model's vocabulary of "
            f"{vocab_size}."
        )
    return token_id


def _check_ids(torch, ids) -> None:
    """The prompt itself, before anything indexes into an embedding table.

    Without this the three ways to get it wrong all leave through torch:
    `'list' object has no attribute 'dim'`, a scalar-type message naming
    argument #1, and `index out of range in self`. None of those is a sentence
    anybody wrote for a reader, and all three are the caller's own request.
    """
    if not isinstance(ids, torch.Tensor):
        raise BadRequest(
            f"ids must be a torch tensor of token ids shaped [1, S], got "
            f"{type(ids).__name__}. Tokenize the prompt first — "
            f"tokenizer(text, return_tensors='pt').input_ids is the shape this "
            f"takes."
        )
    # torch.bool is deliberately not in this list: True would index row 1.
    if ids.dtype not in (
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
    ):
        raise BadRequest(
            f"ids must be whole token ids, got a tensor of {ids.dtype}. An "
            f"embedding table is indexed by row, not interpolated between "
            f"rows — the path integral here runs between EMBEDDINGS, and the "
            f"ids that name them are still integers."
        )


def _check_ids_in_vocabulary(ids, vocab_size: int) -> None:
    """Every id names a row of this model's table, or a 422 says which does not."""
    if int(ids.numel()) == 0:
        return
    lo, hi = int(ids.min()), int(ids.max())
    if lo < 0 or hi >= vocab_size:
        bad = hi if hi >= vocab_size else lo
        raise BadRequest(
            f"token id {bad} is outside this model's vocabulary of "
            f"{vocab_size}. These ids came from a different tokenizer than "
            f"this model's, or from a tokenizer with added tokens the "
            f"checkpoint never learned."
        )


def _check_arguments(ids, position: int, baseline: str, steps: int, target_kind: str):
    """Everything wrong with the REQUEST, before anything expensive runs.

    All `BadRequest` (422) rather than `Refusal` (409): each names a parameter
    the caller passed, and errors.py gives "an unknown baseline name" as its
    type example of a bad request. The refusals in this module are about the
    model and the measurement, which is a different thing for the reader to
    fix.
    """
    if baseline not in BASELINES:
        raise BadRequest(
            f"unknown baseline {baseline!r} — this measurement offers "
            f"{', '.join(BASELINES)}. The baseline is not a detail of the "
            "method: it is the point every attribution is relative to, and the "
            "three give different answers for the same prompt."
        )
    if target_kind not in TARGETS:
        raise BadRequest(
            f"unknown target {target_kind!r} — this measurement offers "
            f"{', '.join(TARGETS)}. 'logprob' attributes log p(token), which "
            "softmax makes shift-invariant; 'logit' attributes the raw logit, "
            "which includes whatever moved the whole vocabulary together."
        )
    # bool BEFORE int: isinstance(True, int) is True, and steps=True would
    # otherwise sail through as one step and report itself as "1".
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise BadRequest(f"steps must be a whole number of path points, got {steps!r}.")
    if not 1 <= steps <= MAX_STEPS:
        raise BadRequest(
            f"steps must be between 1 and {MAX_STEPS}, got {steps}. Each step "
            "is a forward AND a backward pass over the whole model — price it "
            "with gradients.cost() before spending it."
        )
    if ids.dim() != 2 or int(ids.shape[0]) != 1:
        raise BadRequest(
            f"integrated gradients needs one unbatched sequence shaped [1, S], "
            f"got {tuple(ids.shape)}."
        )
    seq = int(ids.shape[1])
    at = position if position >= 0 else seq + position
    if not 0 <= at < seq:
        raise BadRequest(f"position {position} is outside a sequence of {seq} tokens.")
    return at


def _prepare(model, tokenizer, ids, *, position, baseline, steps, target_kind):
    """Shared setup for `integrated_gradients` and `cost`.

    Both need the same embeddings, the same baseline and the same argument
    checks, and a cost estimate built from a DIFFERENT setup than the run it
    prices is the failure `budget.probe_pass` warns about at length — "a probe
    that does less work than the loop it projects is not a cheap estimate, it
    is a wrong one".
    """
    torch = _torch()
    if getattr(model, "training", False) is True:
        raise Refusal(
            "this model is in training mode, where dropout and batch-norm make "
            "the same input give a different answer each time — so the "
            "gradient at one point on the path would be that noise plus the "
            "model, and the completeness check would fail for a reason that "
            "has nothing to do with the step count. Call model.eval() first; "
            "this function will not do it to your object as a side effect."
        )

    _check_ids(torch, ids)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    at = _check_arguments(ids, position, baseline, steps, target_kind)

    embed = _embeddings(model)
    _grad_is_possible(torch, embed.weight)

    device = embed.weight.device
    ids = ids.to(device)
    _check_ids_in_vocabulary(ids, int(embed.weight.shape[0]))
    with torch.no_grad():
        x = embed(ids).detach()
    b, note = _baseline_embeddings(baseline, tokenizer, embed, x)
    return ids, at, x, b.detach(), note


def integrated_gradients(
    model,
    tokenizer,
    ids,
    *,
    position: int = -1,
    target: int | None = None,
    baseline: str = "zero",
    target_kind: str = "logprob",
    steps: int = DEFAULT_STEPS,
    top_k: int = 0,
    on_gap: str = "refuse",
    forward_kwargs: dict | None = None,
) -> Attribution:
    """Attribute one token's score to the input embeddings, and check it adds up.

    Blocking; call from a worker thread. `model` and `tokenizer` are passed in
    the way `dla.attribute` and `attribute.rank_tokens` take them — nothing
    here reaches for a runtime, and nothing here is on the serving path.

    `target` is the token id being attributed; `None` takes whatever the model
    predicts at `position`, read from the same forward pass that provides
    `f(input)`. `top_k` trims the returned token list to the strongest by
    magnitude — 0 returns all of them, and `n_tokens` carries the true count
    beside whatever is listed.

    `on_gap` is "refuse" or "report". Refusing is the default because a chart
    whose bars do not sum to what happened is the failure mode this whole
    feature would otherwise have; "report" is for a caller that wants to SHOW
    the divergence, and it comes back with `verdict="diverged"` and a sentence
    saying so rather than quietly.

    Costs `steps` forward-and-backward passes plus four forward-only ones: the
    two endpoints, and both of them again to measure how far a repeat moves. Do
    not add those two numbers together — see `cost()`.
    """
    torch = _torch()
    started = time.perf_counter()

    if on_gap not in ("refuse", "report"):
        raise BadRequest(
            f"unknown on_gap {on_gap!r} — 'refuse' declines to return an "
            "attribution whose completeness check failed, 'report' returns it "
            "with the failure named."
        )
    # bool BEFORE int, here as everywhere: isinstance(True, int) is True, and
    # top_k=True would silently become a one-row table.
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise BadRequest(
            f"top_k must be a count of rows to keep, 0 for all of them, got {top_k!r}."
        )
    # The half of the target check that needs no vocabulary, run before
    # anything is spent. The other half is below, once a forward pass has said
    # how wide the vocabulary is.
    _check_target(target)

    ids, at, x, b, baseline_note = _prepare(
        model,
        tokenizer,
        ids,
        position=position,
        baseline=baseline,
        steps=steps,
        target_kind=target_kind,
    )
    extra = dict(forward_kwargs or {})
    delta_emb = x - b
    seq = int(ids.shape[1])

    forward_passes = 0

    def endpoint(point) -> Any:
        """One forward pass at a fixed point on the path, no graph retained.

        Every pass in this function sees the same sequence length, the same
        (absent) attention mask and therefore the same positions. That is why
        this module does not carry attribute.py's `position_ids` machinery: it
        never masks anything, so there is no suffix to re-phase and no way for
        a position shift to be billed to a token.

        `.clone()` on the row, and it is not a nicety. `logits[0, at]` is a
        VIEW: it shares storage with the whole `[1, S, V]` logits tensor, so
        holding the row holds all of it for as long as the row lives. On
        Qwen3-1.7B at S=400 that is 400 * 151,936 * 4 = 243 MB pinned by a
        1-row read, on a module aimed at an 8 GB card — and none of it would
        appear in `cost().retained_bytes`, which is the direction budget.py
        warns about: an estimate that approves a run the card then cannot
        hold. The copy costs V floats.
        """
        nonlocal forward_passes
        forward_passes += 1
        with torch.no_grad():
            return model(inputs_embeds=point, **extra).logits[0, at].clone()

    # The measured region starts HERE, before the endpoint passes, because
    # they are this function's allocations too. Starting it after them meant
    # the `[1, S, V]` logits tensor they allocate was billed to nobody: it was
    # not in `before` on the way in and it was not in the peak on the way out,
    # so `peak_bytes` came back 48 MB under the true peak of the same call on a
    # Qwen3-sized vocabulary. What is excluded is only what was already
    # resident when this was called plus `_prepare`'s embeddings — which is
    # what MARGINAL means here and in budget.probe_pass.
    before, peak_note = _start_peak(torch, x.device)

    # THE ENDPOINTS, AND THEN THE ENDPOINTS AGAIN. The second pair is not
    # waste: `measured_delta` is the number every attribution is checked
    # against, and on an accelerator that is not bit-reproducible it has a
    # resolution. `ablate.rank_heads` runs the same forward twice to find out
    # what zero looks like; this is that, for the one quantity this module
    # cannot do without.
    logits_x = endpoint(x)
    vocab_size = int(logits_x.shape[-1])
    token_id = (
        int(logits_x.argmax()) if target is None else _check_target(target, vocab_size)
    )
    f_x = float(_scalar(logits_x, token_id, target_kind))
    # Dropped before the loop rather than at the end of the function: it is a
    # `[V]` row, it is never read again, and `retained_bytes` says four `[S,d]`
    # tensors are what this holds across steps. That has to stay true.
    del logits_x
    f_b = float(_scalar(endpoint(b), token_id, target_kind))
    f_x_again = float(_scalar(endpoint(x), token_id, target_kind))
    f_b_again = float(_scalar(endpoint(b), token_id, target_kind))
    _refuse_non_finite_endpoints(f_x, f_b, f_x_again, f_b_again, target_kind)
    endpoint_floor = abs(f_x - f_x_again) + abs(f_b - f_b_again)
    measured_delta = f_x - f_b

    # One buffer, float32, allocated once and added into. `steps` gradients
    # never coexist — that is the difference between a peak that is flat in
    # `steps` and one that is linear in it.
    total = torch.zeros(seq, x.shape[-1], dtype=torch.float32, device=x.device)

    with torch.enable_grad():
        for k in range(steps):
            # Midpoint, not left-endpoint. See RULE: at the same cost it
            # integrates a linear integrand exactly, and it never evaluates the
            # baseline itself, where a zero embedding can put a model somewhere
            # its gradient means very little.
            alpha = (k + 0.5) / steps
            point = (b + alpha * delta_emb).detach().requires_grad_(True)
            out = model(inputs_embeds=point, **extra)
            scalar = _scalar(out.logits[0, at], token_id, target_kind)
            # `autograd.grad`, never `.backward()`: with `inputs` named,
            # autograd prunes every path that does not reach `point`, so no
            # parameter of the caller's model gets a `.grad` buffer. On a 1.7B
            # model `.backward()` here would allocate a second copy of the
            # weights and leave it attached to their object.
            (grad,) = torch.autograd.grad(scalar, point)
            total += grad.detach()[0].float()
            # Explicit, before the next iteration allocates. The graph goes
            # with `out`, and it is the largest thing in the loop.
            del out, scalar, grad, point

    peak_bytes, measured_note = _end_peak(torch, x.device, before)
    peak_note = peak_note or measured_note

    # The Riemann average times the straight-line displacement. Summed over the
    # embedding dimensions per token, because a per-dimension number is not
    # about anything a reader can name.
    attributions = (delta_emb[0].float() * (total / steps)).sum(dim=-1)
    sum_of_attributions = float(attributions.sum())
    gap = measured_delta - sum_of_attributions

    completeness = _completeness(
        steps=steps,
        sum_of_attributions=sum_of_attributions,
        measured_delta=measured_delta,
        gap=gap,
        endpoint_floor=endpoint_floor,
    )
    # `verdict == "diverged"` is not the only way to be past the refusal line.
    # A gap that is 73% of the move but is also under `endpoint_floor` scores
    # `undefined` rather than `diverged` — see `_completeness` — and drawing it
    # would be exactly the chart this feature exists to prevent. The line is
    # the share, wherever the verdict landed.
    over_the_line = (
        completeness.gap_share is not None
        and completeness.gap_share > REFUSE_ABOVE_GAP_SHARE
    )
    if over_the_line and on_gap == "refuse":
        unresolved = (
            ""
            if abs(gap) > endpoint_floor
            else (
                f" That gap is also at or under the {endpoint_floor:.6f} two "
                f"repeats of the same forward passes moved on their own, so it "
                f"cannot be told whether the approximation has not converged or "
                f"the move itself is too small for this device to resolve. "
                f"Either way it is not scored."
            )
        )
        raise Diverged(
            f"the completeness check failed: the attributions sum to "
            f"{sum_of_attributions:.6f} against a measured move of "
            f"{measured_delta:.6f} from the {baseline!r} baseline, a gap of "
            f"{gap:.6f} — {abs(gap) / abs(measured_delta):.1%} of what "
            f"happened.{unresolved} Integrated gradients is only exact in the "
            f"limit, and {steps} step{'' if steps == 1 else 's'} of the {RULE} "
            f"rule {'has' if steps == 1 else 'have'} not got there on this "
            f"model, so these bars are not a decomposition of the answer. "
            f"Re-run with steps={min(MAX_STEPS, steps * 4)}, or pass "
            f"on_gap='report' to see the attribution with the gap named "
            f"beside it.",
            gap=gap,
            measured_delta=measured_delta,
            steps=steps,
            suggested_steps=min(MAX_STEPS, steps * 4),
        )

    # `.tolist()` once, rather than `float(attributions[i])` per row: each of
    # those is a separate device-to-host synchronisation on CUDA, and a
    # 400-token prompt would pay 1,200 of them to fill a table.
    values = attributions.tolist()
    n_nonfinite = sum(1 for v in values if not _finite(v))
    # The floor a bar is read against. When the gap itself is not finite there
    # is no floor: `abs(v) < nan` is False for every v, so the obvious
    # comparison would publish every bar as readable against an error nobody
    # could measure. `None` here means "no bar can be read", and every row is
    # marked unreadable for that reason rather than for being small.
    floor = abs(gap) if _finite(gap) else None
    total_abs = float(attributions.abs().sum())
    shareable = _finite(total_abs) and total_abs > 0
    token_ids = ids[0].tolist()
    rows = [
        TokenAttribution(
            index=i,
            token=_decode(tokenizer, int(token_ids[i])),
            token_id=int(token_ids[i]),
            attribution=round(values[i], 6) if _finite(values[i]) else values[i],
            # None, not 0.0, when there is nothing to take a share of — and
            # none of them when the total itself is not a number.
            share=(
                round(abs(values[i]) / total_abs, 6)
                if shareable and _finite(values[i])
                else None
            ),
            unreadable=(
                not _finite(values[i]) or floor is None or abs(values[i]) < floor
            ),
        )
        for i in range(seq)
    ]
    # Counted over every token, before the cut — both of these describe the
    # attribution, not the slice of it that fits in a table. Counting after the
    # cut would publish the unreadable count of the STRONGEST few rows, which
    # are the least likely to be unreadable, under a name that says otherwise.
    n_tokens = len(rows)
    n_unreadable = sum(1 for r in rows if r.unreadable)
    # Non-finite bars sort last rather than wherever the comparison happens to
    # put a NaN — a row with no number is not the strongest row.
    rows.sort(key=lambda r: (not _finite(r.attribution), -abs(r.attribution)))
    if top_k > 0:
        rows = rows[:top_k]

    return Attribution(
        target_token=_decode(tokenizer, token_id),
        target_token_id=token_id,
        position=at,
        target_kind=target_kind,
        baseline=baseline,
        baseline_note=baseline_note,
        completeness=completeness,
        tokens=rows,
        n_tokens=n_tokens,
        n_listed=len(rows),
        n_unreadable=n_unreadable,
        n_nonfinite=n_nonfinite,
        sum_of_absolute_attributions=(
            round(total_abs, 6) if _finite(total_abs) else None
        ),
        backward_passes=steps,
        forward_passes=forward_passes,
        elapsed_s=round(time.perf_counter() - started, 3),
        peak_bytes=peak_bytes,
        peak_note=peak_note,
    )


def _completeness(
    *,
    steps: int,
    sum_of_attributions: float,
    measured_delta: float,
    gap: float,
    endpoint_floor: float,
) -> Completeness:
    """Turn the three numbers into a verdict, and never divide by a zero delta.

    `gap_share` is `None` when the delta is under the floor two repeated
    forward passes measured. That is the case where the baseline and the input
    give the same answer to within the noise, so there is no move for the
    attributions to be a share of — and a 0.0 there would read as "no gap",
    which is the opposite of what was found.

    `endpoint_floor` is a resolution, not an amnesty. A gap under it is a gap
    that cannot be told from running the same forward twice — which is a
    reason not to CALL it, and not a reason to call it converged. When the same
    gap is also most of the move, both statements are true at once and neither
    verdict is available: the run is `undefined` and says both numbers.
    """
    if not _finite(sum_of_attributions) or not _finite(gap):
        return Completeness(
            steps=steps,
            rule=RULE,
            sum_of_attributions=sum_of_attributions,
            measured_delta=measured_delta,
            gap=gap,
            gap_share=None,
            endpoint_floor=round(endpoint_floor, 9),
            verdict="undefined",
            sentence=(
                f"The completeness check could not be scored: the {steps}-step "
                f"sum of attributions came back as {sum_of_attributions}, which "
                f"is not a number, so there is nothing to compare with the "
                f"measured move of {measured_delta}. A NaN or an infinity "
                f"compares False against every bound in both directions, so any "
                f"verdict here would be a word with no measurement under it. "
                f"The bars are marked unreadable for the same reason. The "
                f"forward passes at both endpoints were finite, so this came "
                f"out of the backward pass — a gradient that overflowed, or a "
                f"point on the path where this model's derivative does not "
                f"exist."
            ),
        )

    resolved = _finite(measured_delta) and abs(measured_delta) > endpoint_floor
    share = abs(gap) / abs(measured_delta) if resolved and measured_delta else None

    if share is None:
        verdict = "undefined"
        sentence = (
            f"The completeness check could not be scored: the {steps}-step sum "
            f"is {sum_of_attributions:.6f} against a measured move of "
            f"{measured_delta:.6f}, which is not above the {endpoint_floor:.6f} "
            f"that repeating the same two forward passes moved on its own. "
            f"The baseline and your input give this token the same score to "
            f"within the arithmetic, so there is nothing here to attribute."
        )
    else:
        if abs(gap) <= endpoint_floor and share > REFUSE_ABOVE_GAP_SHARE:
            # BOTH things are true and they point opposite ways. The gap is
            # inside the noise of the two passes that measured the move, so it
            # cannot be called; and it is most of the move, so the attributions
            # are not a decomposition of anything. Calling this `converged` —
            # which is what checking the floor first did — publishes a chart
            # accounting for 27% of what happened under the word that means it
            # accounts for all of it, with the 72.6% share sitting in the same
            # payload. `undefined` is the module's own word for a quantity it
            # could not resolve, and this is one.
            verdict = "undefined"
            tail = (
                f"or {share:.2%} of the move — and at the same time at or under "
                f"the {endpoint_floor:.6f} that repeating the same two forward "
                f"passes moved on their own. Both of those are true, and they "
                f"do not agree: most of the move is unaccounted for, and the "
                f"move is only {abs(measured_delta) / endpoint_floor:.2f}x the "
                f"resolution of the passes that measured it. This is not scored "
                f"either way. Re-run with steps="
                f"{min(MAX_STEPS, steps * 4)}, and on a device whose forward "
                f"pass repeats exactly if you have one — that floor is what a "
                f"repeat moved here."
            )
        elif abs(gap) <= endpoint_floor:
            verdict = "converged"
            tail = (
                f"which is at or under the {endpoint_floor:.6f} that repeating "
                f"the same two forward passes moved on its own, so it cannot "
                f"be told from the arithmetic."
            )
        elif share <= WARN_ABOVE_GAP_SHARE:
            verdict = "converged"
            tail = f"or {share:.2%} of the move — the sum accounts for the rest."
        elif share <= REFUSE_ABOVE_GAP_SHARE:
            verdict = "approximate"
            tail = (
                f"or {share:.2%} of the move, which is real: {steps} "
                f"step{'' if steps == 1 else 's'} of the {RULE} rule "
                f"{'has' if steps == 1 else 'have'} not converged here, and "
                f"every bar below is short by its share of it. More steps will "
                f"shrink it."
            )
        else:
            verdict = "diverged"
            tail = (
                f"or {share:.2%} of the move. The attributions do not add up "
                f"to what happened, so they are not a decomposition of it."
            )
        sentence = (
            f"Completeness: the attributions sum to {sum_of_attributions:.6f} "
            f"against a measured move of {measured_delta:.6f} from baseline to "
            f"input, a gap of {gap:.6f} — {tail}"
        )

    return Completeness(
        steps=steps,
        rule=RULE,
        sum_of_attributions=round(sum_of_attributions, 6),
        measured_delta=round(measured_delta, 6),
        gap=round(gap, 6),
        gap_share=None if share is None else round(share, 6),
        endpoint_floor=round(endpoint_floor, 9),
        verdict=verdict,
        sentence=sentence,
    )


def _decode(tokenizer, token_id: int) -> str:
    """Whatever this tokenizer calls that id, or the id itself.

    A tokenizer that cannot decode is not a reason to lose the numbers, and a
    row labelled with its own id is honest about what is known — an empty
    string there would be a token that looks like it decoded to nothing.
    """
    try:
        return tokenizer.decode([token_id])
    except Exception:
        return f"<id {token_id}>"


def _start_peak(torch, device):
    """Zero the allocator's high-water mark and note what was already resident.

    Returns `(baseline_bytes, reason)`; `reason` is empty exactly when the
    counter took. `budget.reset_peak` does the same thing keyed by a device
    KIND string, and this module has a `torch.device` in hand — the API asked
    is the same one, and the `None`-not-zero rule is the same rule. A peak of
    zero bytes is a claim; an unmeasured peak is not, and the two must never
    arrive in the same field.
    """
    kind = device.type
    mod = getattr(torch, kind, None)
    reset = getattr(mod, "reset_peak_memory_stats", None) if mod else None
    current = getattr(mod, "memory_allocated", None) if mod else None
    if reset is None or current is None:
        return None, (
            f"there is no allocator peak to read on {kind}, so the memory this "
            f"run needed was not measured"
        )
    try:
        before = int(current())
        reset()
    except Exception:
        return None, (
            f"the {kind} allocator would not report a peak for this region, so "
            f"the memory this run needed was not measured"
        )
    return before, ""


def _end_peak(torch, device, before: int | None):
    """`(peak_bytes, note)` for the region since `_start_peak`.

    MARGINAL, like `budget.probe_pass`: the model's weights and the four
    `[S, d]` tensors `_prepare` and its caller built are already allocated when
    the region opens, and billing them to the attribution would make every run
    look like it needed the whole card. `max(..., 0)` for the same reason
    budget.py does it — a region that frees more than it takes has no
    measurable cost, not a negative one.

    The region covers the endpoint passes as well as the loop, so the `[1,S,V]`
    logits tensor a forward allocates is inside the number rather than in the
    gap between two snapshots. It is the largest single allocation on a real
    vocabulary and it used to appear in neither.
    """
    if before is None:
        return None, ""
    mod = getattr(torch, device.type, None)
    fn = getattr(mod, "max_memory_allocated", None) if mod else None
    if fn is None:
        return None, ""
    try:
        peak = int(fn())
    except Exception:
        return None, (
            f"the {device.type} allocator did not report a peak for this region"
        )
    return max(peak - before, 0), (
        "allocated by PyTorch for this attribution — the four endpoint forward "
        "passes and the accumulation loop, logits tensors included — on top of "
        "what was already resident when it was called: the model, the input "
        "embeddings, the baseline's and their difference"
    )


# ------------------------------------------------------------------- the price


def cost(
    model,
    tokenizer,
    ids,
    *,
    position: int = -1,
    target: int | None = None,
    baseline: str = "zero",
    target_kind: str = "logprob",
    steps: int = DEFAULT_STEPS,
    forward_kwargs: dict | None = None,
) -> Cost:
    """Price the run before spending it, in forward-AND-backward passes.

    Two probes, both real: one forward-only pass at the input, and one complete
    step of the loop `integrated_gradients` runs — the interpolated point, the
    forward, the scalar, `autograd.grad`, and the accumulation. `budget.py`
    makes the case for probing the real loop body rather than a bare forward at
    length, and it is sharper here than anywhere else in the package, because
    the two differ by the entire retained graph.

    The ratio between them is returned. It is measured, and it is the reason
    this module does not report a pass count the way `ablate` and `patch` do: a
    number of "passes" that mixes the two units is not a price.

    Cheap in the same sense as the rest of budget.py — this costs two passes,
    one of which is the expensive kind, against the `steps` this is pricing.
    """
    torch = _torch()
    from . import budget

    # Same check, same order, same errors as the run this is pricing. A
    # `cost()` that accepts an id `integrated_gradients` answers 422 for prices
    # a call that cannot be made, and the docstrings send the reader HERE
    # first — so this must not be the lenient one.
    _check_target(target)

    ids, at, x, b, _note = _prepare(
        model,
        tokenizer,
        ids,
        position=position,
        baseline=baseline,
        steps=steps,
        target_kind=target_kind,
    )
    extra = dict(forward_kwargs or {})
    delta_emb = x - b
    device_kind = x.device.type

    with torch.no_grad():
        # `.clone()` for the reason `integrated_gradients.endpoint` clones: the
        # row is a view onto the whole `[1, S, V]` logits tensor and holding it
        # holds all of it, across both probes below, uncounted.
        logits_x = model(inputs_embeds=x, **extra).logits[0, at].clone()
    vocab_size = int(logits_x.shape[-1])
    token_id = (
        int(logits_x.argmax()) if target is None else _check_target(target, vocab_size)
    )
    del logits_x

    def forward_only():
        with torch.no_grad():
            out = model(inputs_embeds=x, **extra)
            _scalar(out.logits[0, at], token_id, target_kind)

    def one_step():
        """Exactly the body of the loop, including the accumulation."""
        buffer = torch.zeros(
            x.shape[1], x.shape[-1], dtype=torch.float32, device=x.device
        )
        with torch.enable_grad():
            point = (b + 0.5 * delta_emb).detach().requires_grad_(True)
            out = model(inputs_embeds=point, **extra)
            scalar = _scalar(out.logits[0, at], token_id, target_kind)
            (grad,) = torch.autograd.grad(scalar, point)
            buffer += grad.detach()[0].float()

    forward_probe = budget.probe_pass(forward_only, device_kind)
    step_probe = budget.probe_pass(one_step, device_kind)

    # What the loop holds ACROSS steps, which is what budget.project means by
    # the word. FOUR `[S, d]` tensors, not three: the float32 accumulator, the
    # input embeddings `x`, the baseline's `b`, AND their difference
    # `delta_emb`, which is a real allocation held for the whole loop and was
    # missing from this count — measured, 524,288 live bytes against 393,216
    # reported at S=128, d=256, fp32. Counted from the real shapes rather than
    # estimated, and the accumulator is float32 whatever the model's dtype is,
    # so its size is not the other three's.
    per = int(x.shape[1]) * int(x.shape[-1])
    retained = per * 4 + 3 * per * x.element_size()

    estimate = budget.project(step_probe, steps, retained_bytes=retained)
    # `budget.project` labels every estimate "one probe pass on this machine",
    # which is true and, here, ambiguous in the one way that matters. The unit
    # is named instead. Setting the field rather than reimplementing project():
    # the verdict, the threshold and the free-memory reading should be the same
    # ones the rest of the package uses, and a second copy of that logic would
    # drift.
    estimate.basis = "one probe forward-and-backward step on this machine"
    estimate.notes = [
        *estimate.notes,
        "passes here are forward-AND-backward steps, not forward passes",
    ]

    fwd = forward_probe.seconds
    step = step_probe.seconds
    ratio = (step / fwd) if fwd and fwd > 0 else None

    return Cost(
        steps=steps,
        backward_passes=steps,
        # The two endpoints, and both again for the repeat floor.
        forward_passes=4,
        forward_seconds=fwd,
        step_seconds=step,
        ratio=ratio,
        retained_bytes=retained,
        estimate=estimate,
    )


def check(priced: Cost, *, confirm: bool = False) -> Cost:
    """Raise `budget.TooCostly` when the projection says this will not fit.

    Delegates to `budget.check` so the threshold, the wording and the
    overridable flag are the package's, not a second set. Returns the `Cost`
    otherwise, so a caller can report the price it just approved.
    """
    from . import budget

    budget.check(
        priced.estimate,
        label=(
            f"integrated gradients at {priced.steps} steps "
            f"({priced.backward_passes} forward-and-backward passes)"
        ),
        confirm=confirm,
    )
    return priced
