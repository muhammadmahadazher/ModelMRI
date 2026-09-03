# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""The smallest set of your words that HOLDS the answer on its own.

`attribute.rank_tokens` masks one token out and measures how far the answer
moved. That is a NECESSITY question — "was this token needed?" — and it is the
only question this package could ask until now. An anchor is the opposite one:
the smallest set of tokens which, ON THEIR OWN, hold the prediction while
everything else in the prompt is replaced. SUFFICIENCY.

**They are different questions and they give different answers, which is the
entire reason this file exists.** Take a model whose next token is decided by
"is 7 present OR is 9 present", with both in the prompt. Masking 7 changes
nothing — 9 still carries it — so a necessity ranking scores it at the floor
and the reader concludes it did not matter. Measured on the OR fixture in
tests/test_anchors.py, fp32 on CPU: `rank_tokens` scores those two tokens at
4.755e-07 and 8.052e-07 nats against a noise floor of 0.0 — both print as
0.00000 in the five decimals that payload carries — while EITHER ONE ALONE
holds the prediction through 24 of 24 perturbations of every other candidate.
Not necessary, and completely sufficient.

The necessity list there is worse than flat, which is the part worth sitting
with. Index 1 holds a token nothing depends on and scores 4.164e-06, 8.8x the
trigger: below the floor the ranking is ordering the position term, so a panel
that shows it puts an irrelevant word above the word that decides the answer.

The mirror case is an AND, where both tokens are necessary and neither is
sufficient. Same prompt, same position: `rank_tokens` scores them 10.00451 and
9.92551 nats — a real, large, correct necessity signal — while each of them
alone anchors at 0 of 24 and the pair anchors at 24 of 24. A panel that shows
one number and calls it "importance" is answering whichever question it
happened to implement.

So the two live side by side, over THE SAME candidate set — `range(1, position)`
minus the control tokens, index 0 excluded as an attention sink and `position`
excluded as its own query, all for the reasons `attribute.rank_tokens` writes
out. A necessity score and a sufficiency score taken over different sets of
tokens could not be read on one screen, and reading them together is the point.

PRECISION IS THE MEASUREMENT, AND IT IS A SAMPLE

An anchor's precision is the fraction of perturbations of the non-anchor tokens
for which the top-1 prediction still holds. There is no closed form for it: the
number of perturbations is a sample, so it comes back as a count and an
interval, never a bare float. 8 of 10 and 800 of 1000 are both "80%" and they
are not the same evidence — Wilson at 95% gives [0.4902, 0.9433] for the
first and [0.7741, 0.8236] for the second, and a payload that printed 0.8 twice
would have hidden a nine-fold difference in width. Wilson rather than the normal
approximation because precision lives at the ends: 64 of 64 is the case this
search is looking for, and the normal interval there has zero width, which is a
claim of certainty from 64 draws.

THE CEILING, AND THE FLOOR

A precision of 55% means nothing on its own. Both ends are therefore measured:

  `base_rate`  the empty anchor — every candidate perturbed, nothing held. This
               is what the prediction does with none of your words, and it is
               the denominator the anchor has to beat. Measured 0 of 24 on the OR
               fixture, and 24 of 24 on the fixture whose answer is carried by a
               token the search cannot touch — where the run returns an anchor of
               size 0 rather than a token picked to have something to show.
  `ceiling`    the full candidate set held — the best precision ANY anchor
               drawn from this candidate set could reach. Below the target, the
               search is arithmetic that cannot succeed and is refused before it
               is paid for: measured 0 of 24 with the trigger outside the search
               window, and the run stops at 51 passes instead of paying 243.

When the candidate set covers every perturbable position the ceiling perturbs
nothing, so there is no sample to take and none is faked: `measured` is False,
`point` is None, and `implied` carries the 1.0 that follows from the base pass
being stable. A number nobody measured does not get printed as if somebody had.

THE PERTURBATION DISTRIBUTION IS PART OF THE CLAIM

"The prediction survives perturbation" is not a statement until you say
perturbed with WHAT. Replacement ids are drawn from corpus.py — this project's
donor corpus, the same sentences ablate.py resamples heads from, bundled rather
than fetched so the number reproduces offline. Its docstring already argues why
the donor distribution is part of the measurement rather than provenance
trivia, and that argument transfers wholesale.

What does NOT transfer is `corpus.donor_ids`. That function enforces a length
rule — a donor must be at least as long as the analysed prompt — because it
hands back position-matched ACTIVATIONS, and a short donor has nothing to put
at the later positions. This file needs a bag of token ids and no alignment at
all, so that rule would refuse for a reason that does not apply here. So
`donor_pool` calls `corpus.load()` for the sentences and the label, and does its
own tokenizing. The pool is frequency-weighted: an id occurring twice in the
corpus is twice as likely to be drawn, which makes the perturbation the corpus's
own unigram distribution rather than uniform over its types. Uniform-over-types
is equally defensible and is a DIFFERENT measurement, so the payload names which
one was used.

`donor_pool` refuses a corpus that tokenizes to fewer than `MIN_DISTINCT_IDS`
distinct ids, and that rule protects only the callers who go through it.
`find_anchor` takes a `pool` argument directly — every hand-built analysis and
every test does — so the pool it is handed can be anything, and a pool of one
id used to publish `held: 24, samples: 24, low: 0.862` for 24 copies of a
single sequence. `pool_size` could not have shown it either: it is the length
of the list, so `[1] * 200` printed 200. Two counts travel with the interval
now — `pool_distinct_ids`, and `distinct_templates` for how many of the
`n_samples` drawn rows are actually different — with a `quality.note` naming
the shortfall when there is one. Refusing here instead would delete the fixture
this module's arithmetic is checked on, where 8 distinct ids and no trigger
among them is exactly what makes every precision 0 of n or n of n.

Control tokens are never drawn and never perturbed. Replacing `<|im_start|>`
with the word "coffee" breaks the template, and the prediction collapses for a
reason that has nothing to do with the reader's words — every anchor would look
weak and the base rate would sit at zero on every templated prompt. They are
held fixed, counted, and listed.

COMMON RANDOM NUMBERS: EVERY ANCHOR IS SCORED ON THE SAME DRAWS

One block of `n_samples` perturbation templates is drawn up front — one donor id
for every perturbable position, per sample — and every anchor is evaluated
against that same block, with the anchor's own positions written back over it.
So "A beat B" is a comparison on identical noise rather than two independent
lotteries, and the screening pass is a nested prefix of the verification pass
(`templates[:n_search]`) rather than a fresh draw. The seed is reported, because
a sample whose seed is not stated cannot be re-run.

Memory is the template block and nothing else: `n_samples x len(perturbable)`
int64, plus one fp32 vector over the vocabulary for the base distribution. That
is 32 KiB for 64 draws over a 64-token prompt. The per-sample pass takes an
argmax and keeps an integer — no distribution is materialised per draw.

That sentence was false as written until somebody measured it, which is the
point of measuring it. `forward_logits` used to append the mask and the
position_ids to two Python lists on EVERY pass, so the peak grew with the PASS
COUNT rather than staying flat in it, and `estimate_cost.retained_bytes` did
not count them at all. tracemalloc, same prompt, same pool, same seed, a model
that allocates nothing per call so the number is about this file:

    n_samples   passes    peak WITH the lists    peak WITHOUT them
           64      691             64,091 B             51,567 B
        1,024    9,331            202,101 B             51,217 B

138,010 B of that 202,101 B was two pointers per forward pass: 8,640 extra
passes at 15.97 B each, against the 16 B two pointers cost. The lists are two
counters now — they were only ever compared against the very objects they
stored, so the check lost nothing it actually had — and 13.5x the passes now
costs 0.99x the peak. `test_the_peak_does_not_grow_with_the_pass_count` runs
that comparison rather than restating it.

`estimate_cost` reported 960 bytes retained on the test fixture, and 960 was
960 only because that fixture has no `config.vocab_size`: the missing width was
read as zero, so the fp32 vocabulary vector was published as free. The width
comes from the warm pass's own logits now, and the same fixture reports 1,024.

MINIMAL MEANS MINIMAL, AND THIS SEARCH IS GREEDY

Forward selection: grow the anchor one token at a time, always taking the token
that screens best. That is a heuristic and it is named as one everywhere it
appears — `smaller_may_exist: true` is in every payload this module produces. It
is NOT "the minimal set". Two things it can miss: a smaller set it never
assembled, and an equally small DIFFERENT set. The OR fixture is the second case
in its purest form — both triggers screened at 6 of 6, both are minimal
sufficient sets, and the run returned the one at index 4 because the tie-break
is recency (the same argument the candidate cap uses). Nothing about that
payload is wrong and nothing in it says "and the other one would have done".
The screened list is returned for exactly that reason.

Forward selection also OVERSHOOTS, which is not a footnote: on the AND fixture
in tests/test_anchors.py both triggers screen at exactly 0.0 on the first step,
the tie-break takes a token carrying nothing, and greedy then needs three tokens
to clear a bar two of them clear alone. So `prune=True` runs a backward
elimination sweep — offer every element for removal against the set as it
stands, repeat until a sweep removes nothing — and the anchor comes back at two.
The sweep that ended it tested every element against the FINAL anchor, so what
`irreducible_under_single_removal` claims is exactly this: no SUBSET of this
anchor is sufficient. It does not claim no other set of that size is.

**position_ids.** attribute.py hands one `arange(S)` to every pass because
masking an interior entry shifts `attention_mask.cumsum(-1) - 1` for the whole
suffix, and HF decoders derive positions from exactly that. Substitution cannot
cause it: the mask stays all-ones and the length never changes, so cumsum and
arange agree by construction on every pass here. The discipline is kept anyway
and asserted at the end — one mask object, one position_ids object, every pass —
because "cannot arise" is a claim about today's code, and the failure it would
cause is silent.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Sequence
from statistics import NormalDist
from typing import Any

import torch

from . import corpus
from .ablate import distribution, kl_nats
from .errors import BadRequest, Refusal

# Draws behind a reported precision. THE SAMPLE SIZE AND THE TARGET ARE ONE
# DECISION, and getting it wrong makes the search impossible rather than merely
# noisy. The search gates on the interval's LOWER bound, and Wilson's lower
# bound for a perfect result collapses to exactly `n / (n + z**2)` — every other
# term cancels when k == n. So 64 perfect draws publish [0.943, 1.000] and a
# target of 0.95 could never be met by any anchor, however sufficient: the
# search would silently run to `max_size` every time and hand back an
# over-large anchor with `found: false`. `_impossible_target` refuses that
# combination up front rather than letting it look like a finding.
#
# 64 and 0.90 leave room: a perfect run clears at 0.943, one failure in 64 still
# clears at 0.917, two do not (0.893). Reaching 0.95 would need 73 draws and
# tolerate no failures at all, which is a slower search buying a bar nobody
# asked for.
N_SAMPLES = 64

# Draws behind a SCREENING estimate, which only has to order candidates rather
# than support a published number. A prefix of the same block, so screening and
# verification never disagree about which draws they saw.
N_SEARCH = 16

# Nearest `position`, the same rule and the same argument as
# `attribute.MAX_CANDIDATES`: recency is the only ordering available before the
# measurement is taken. Smaller than attribute's 64 because cost here is
# quadratic in this number rather than linear — see `cost_passes`.
MAX_CANDIDATES = 12

# A four-token anchor over a twelve-token window is already a weak claim about
# sufficiency; past that the honest report is "no small set holds this".
MAX_ANCHOR_SIZE = 4

# The lower bound of the interval has to clear this, not the point estimate.
# 0.90 rather than the 0.95 the anchors literature usually quotes, and the
# reason is the arithmetic at N_SAMPLES above rather than a softer standard:
# 0.95 is unreachable at 64 draws, so quoting it would be a bar that reads
# stricter and measures nothing. It travels in the payload either way, because
# an anchor is only "sufficient" relative to a bar somebody set.
PRECISION_TARGET = 0.90

CONFIDENCE = 0.95

INTERVAL_METHOD = "Wilson score interval"

# base, a repeat of it for the noise floor, and one plain model(ids).
FIXED_PASSES = 3

# Held-fixed positions can be most of a long prompt. The list is cut and the
# true count travels beside it.
HELD_FIXED_LISTED = 32

# Below this many DISTINCT ids the pool stops being "ordinary English" and
# becomes a statement about a handful of words: with 8 distinct ids every draw
# has a 1-in-8 chance of reinstating the very token it was meant to remove, and
# precision rises for that reason rather than because the anchor holds anything.
MIN_DISTINCT_IDS = 16


class AnchorError(Refusal):
    """We cannot take this measurement, and we say why rather than guess.

    A `Refusal` rather than its own root, following `budget.TooCostly`: every
    raise below is ModelMRI understanding the request perfectly and declining
    it in a sentence written for the reader, which is what errors.py reserves
    409 for. It also means a route needs no translation arm to avoid answering
    500 for a decision this file made on purpose.
    """


def wilson_interval(
    successes: int, samples: int, *, confidence: float = CONFIDENCE
) -> tuple[float, float]:
    """A binomial confidence interval for `successes / samples`, Wilson's.

    Wilson rather than the normal approximation, and the reason is the end of
    the range this file lives at. A perfect anchor is `successes == samples`,
    where the normal interval has width exactly zero — 64 of 64 would publish
    [1.0, 1.0], a claim of certainty out of 64 draws. Wilson gives
    [0.943, 1.000] there, and its lower bound is what the search actually gates
    on. At the other end 0 of 64 is [0.000, 0.057] rather than a point at zero.

    Clamped to [0, 1] because the interval is over a proportion and Wilson's
    bounds can round a hair outside it in float arithmetic.
    """
    if isinstance(successes, bool) or isinstance(samples, bool):
        # `isinstance(True, int)` is True, so a bool would sail through the
        # range checks below and silently become 1 or 0.
        raise BadRequest("successes and samples are counts, not booleans.")
    if samples < 1:
        raise BadRequest(
            f"a Wilson interval needs at least one sample, got {samples}. A "
            "proportion over no draws is not a wide interval, it is no "
            "measurement."
        )
    if not 0 <= successes <= samples:
        raise BadRequest(
            f"{successes} successes out of {samples} samples is not a "
            "proportion. Successes must be in [0, samples]."
        )
    if not 0.0 < confidence < 1.0:
        raise BadRequest(
            f"confidence must be strictly between 0 and 1, got {confidence}."
        )

    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    n, k = float(samples), float(successes)
    denominator = n + z * z
    centre = (k + z * z / 2.0) / denominator
    spread = z / denominator * ((k * (n - k) / n) + z * z / 4.0) ** 0.5
    return max(0.0, centre - spread), min(1.0, centre + spread)


def _interval(successes: int, samples: int, *, confidence: float) -> dict:
    """One measured proportion, with everything needed to read it."""
    low, high = wilson_interval(successes, samples, confidence=confidence)
    return {
        "measured": True,
        "held": int(successes),
        "samples": int(samples),
        "point": round(successes / samples, 4),
        "low": round(low, 4),
        "high": round(high, 4),
        "confidence": confidence,
        "method": INTERVAL_METHOD,
        "reason": "",
    }


def reachable_target(samples: int, *, confidence: float = CONFIDENCE) -> float:
    """The highest target `samples` draws could ever clear, and it is exact.

    Wilson's lower bound for a PERFECT result cancels down to `n / (n + z**2)`:
    at `k == n` the spread term is `z**2 / (2 * (n + z**2))` and it subtracts
    exactly the `z**2 / 2` the centre added. So this is not a heuristic — it is
    the best lower bound arithmetic allows from that many draws.

    Measured against the shipped defaults: 64 draws reach 0.9434, 73 reach
    0.9500, 128 reach 0.9709. A target above the returned value cannot be met
    by any anchor at any level of sufficiency.
    """
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    return samples / (samples + z * z)


def publishable_ceiling(samples: int, *, confidence: float = CONFIDENCE) -> float:
    """`reachable_target` cut to 4 places DOWNWARDS, so it can be passed back.

    `round()` is the wrong direction for a boundary and it shipped that way.
    At the default 64 draws the exact ceiling is 0.9433759402 and `round(_, 4)`
    is 0.9434 — ABOVE it. The payload advertised 0.9434 as the best any anchor
    could reach, and handing that same number back as `target` was refused by a
    sentence that printed 0.9434 twice and said one could not reach the other.
    Truncation toward zero cannot do that: the published value is always <= the
    exact one, so `_impossible_target` accepts it by construction, which is
    what `test_the_published_ceiling_is_a_target_that_is_actually_accepted`
    checks at every sample size it can reach.

    Measured, exact vs published: 16 -> 0.8063923194 / 0.8063; 24 ->
    0.8620237953 / 0.862; 32 -> 0.8928208017 / 0.8928; 64 -> 0.9433759402 /
    0.9433 (round would have said 0.9434); 73 -> 0.9500079920 / 0.95; 128 ->
    0.9708630437 / 0.9708 (round would have said 0.9709). The exact float
    travels beside it as `target_ceiling_exact`, so the 4-place cut costs a
    reader nothing.
    """
    return (
        math.floor(reachable_target(samples, confidence=confidence) * 10_000) / 10_000
    )


def _impossible_target(samples: int, target: float, confidence: float) -> None:
    """Refuse a (samples, target) pair that no result could satisfy.

    Without this the run is not wrong, it is worse than wrong: every search
    walks to `max_size`, every anchor comes back `found: false` with a
    precision of 1.000, and the payload looks like a model that has no anchors
    rather than a caller who asked for a bound their sample size cannot carry.
    """
    reachable = reachable_target(samples, confidence=confidence)
    if target <= reachable:
        return
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    needed = -(-(target * z * z / (1.0 - target)) // 1) if target < 1.0 else None
    # `publishable_ceiling` rather than a format spec, for the reason written
    # out there: any rounding that can go UP names a target this same function
    # would refuse a second later. `{reachable:.2f}` did exactly that at
    # samples=1000, where the exact bound is 0.9961732415 and `.2f` prints
    # "1.00". The number offered here is always one that is accepted.
    offer = publishable_ceiling(samples, confidence=confidence)
    raise BadRequest(
        f"a target of {target} cannot be met with {samples} draws: even a "
        f"perfect {samples} of {samples} has a {confidence:.0%} Wilson lower "
        f"bound of {reachable:.10f}, and the search gates on the lower bound. "
        + (
            f"Raise n_samples to at least {int(needed)}, or lower the target to "
            f"{offer} or below."
            if needed is not None
            else "A target of exactly 1.0 is unreachable from any finite "
            "sample; lower it."
        )
    )


def _proportion(successes: int, samples: int, *, confidence: float) -> dict:
    """`_interval` when there were draws, `_unmeasured` when there were none.

    Zero draws happens for one reason and it is not an error: the anchor being
    scored already holds every perturbable position, so no draw perturbs
    anything and the prediction survives by construction. That is `implied=1.0`
    and `point=None` — a proportion over no samples is not 1.0 measured, and the
    difference is the whole reason both keys exist.
    """
    if samples < 1:
        return _unmeasured(
            "nothing outside this anchor was perturbable, so no draw changed "
            "the sequence and no sufficiency was demonstrated.",
            implied=1.0,
        )
    return _interval(successes, samples, confidence=confidence)


def _unmeasured(reason: str, *, implied: float | None = None) -> dict:
    """The same shape, for a proportion nobody sampled.

    `point` is None and stays None. The one thing this must never do is put a
    0 or a 1 in the field a measured proportion uses — `implied` is a separate
    key precisely so that a reader (and a chart) can tell arithmetic from
    evidence.
    """
    return {
        "measured": False,
        "held": None,
        "samples": 0,
        "point": None,
        "low": None,
        "high": None,
        "confidence": None,
        "method": None,
        "implied": implied,
        "reason": reason,
    }


def cost_passes(
    *,
    n_candidates: int,
    steps: int,
    n_samples: int = N_SAMPLES,
    n_search: int = N_SEARCH,
    ceiling_measured: bool,
    prune_size: int = 0,
    covering_final_step: bool = False,
) -> int:
    """Exactly how many forward passes a run of this shape takes.

    Exact, not an estimate: `find_anchor` counts its own passes and the tests
    assert the two agree to the pass. The parts, in the order they are spent:

        3                            base, its repeat for the floor, one plain
                                     model(ids) for the agreement check
        n_samples                    the base rate — the empty anchor
        n_samples                    the ceiling, when there is anything
                                     outside the candidate set left to perturb
        n_search * sum(C - i)        screening, step i having C - i left
        n_samples * steps            verifying the anchor after each step
        n_samples * prune_size       backward elimination

    `steps` is not known before the run — greedy stops when the target is met —
    so `estimate_cost` prices the shortest and the longest run and says which
    is which rather than averaging them into a number that describes neither.

    `prune_size` is the number of removal evaluations that actually spent
    passes, which is NOT the anchor's size: elimination sweeps until one removes
    nothing, so an anchor of k costs between k (one clean sweep) and
    k(k+1)/2 (every sweep removing one). A sweep over a one-token anchor spends
    nothing at all — dropping its only element leaves the empty anchor, which
    was already measured as the base rate.

    `covering_final_step` is the one path on which "n_search per trial and
    n_samples per step" is NOT what the loop spends, and leaving it out is what
    made the word "exact" above false. When the anchor plus the last remaining
    candidate covers every perturbable position, `evaluate` has nothing to
    perturb: it returns (0, 0) and spends ZERO passes, so that step's single
    screening trial and its verification are both free. Measured on a
    three-perturbable prompt where the rule needs all three (steps=3,
    n_candidates=3, n_samples=24, n_search=6): the loop spends 105 passes and
    this function said 135 until the flag existed — 30 out, which is exactly
    the missing `n_search + n_samples`.

    That path is only reachable when the candidate window is the whole
    perturbable set and the search consumed all of it, so `steps` must equal
    `n_candidates`; anything else is a caller describing a run that cannot
    happen, and is refused rather than priced.
    """
    for name, value in (
        ("n_candidates", n_candidates),
        ("steps", steps),
        ("n_samples", n_samples),
        ("n_search", n_search),
        ("prune_size", prune_size),
    ):
        if isinstance(value, bool):
            raise BadRequest(f"{name} is a count, not a boolean.")
    if steps > n_candidates:
        raise BadRequest(
            f"a greedy search over {n_candidates} candidates cannot take "
            f"{steps} steps — each step consumes one candidate."
        )
    if covering_final_step and steps != n_candidates:
        raise BadRequest(
            "a covering final step means the anchor plus the last remaining "
            "candidate held every perturbable position, which can only happen "
            f"once the search has consumed every candidate. Got {steps} steps "
            f"over {n_candidates} candidates."
        )
    trials = sum(n_candidates - i for i in range(steps))
    free_trials = 1 if covering_final_step else 0
    free_steps = 1 if covering_final_step else 0
    return (
        FIXED_PASSES
        + n_samples
        + (n_samples if ceiling_measured else 0)
        + (trials - free_trials) * n_search
        + (steps - free_steps) * n_samples
        + prune_size * n_samples
    )


def donor_pool(
    tokenizer: Any,
    *,
    control_ids: Iterable[int] = (),
    sentences: Sequence[str] | None = None,
    label: str | None = None,
) -> tuple[list[int], dict]:
    """Replacement token ids, and a description of where they came from.

    Returns `(ids, description)`. `ids` is frequency-weighted and NOT deduped —
    see the module docstring: the pool is the corpus's own unigram distribution,
    which is a different measurement from uniform-over-types and is named as
    such in `description`.

    `sentences`/`label` override the bundled corpus together; passing one
    without the other is refused, because a pool labelled "built-in" that is not
    the built-in corpus is the exact failure corpus.py exists to prevent.
    """
    if (sentences is None) != (label is None):
        raise BadRequest(
            "pass both `sentences` and `label` or neither. A pool of one "
            "corpus wearing another's label cannot be reproduced by anyone."
        )
    if sentences is None:
        sentences, label = corpus.load()

    control = {int(c) for c in control_ids}
    ids: list[int] = []
    dropped = 0
    for sentence in sentences:
        encoded = tokenizer(sentence, return_tensors="pt").input_ids
        for token_id in encoded[0].tolist():
            if int(token_id) in control:
                dropped += 1
            else:
                ids.append(int(token_id))

    distinct = len(set(ids))
    if distinct < MIN_DISTINCT_IDS:
        raise AnchorError(
            f"the donor corpus ({label}) tokenizes to {distinct} distinct ids "
            f"once its {dropped} control tokens are removed, and a perturbation "
            f"drawn from fewer than {MIN_DISTINCT_IDS} of them is a statement "
            "about those few words: each draw has better than a one-in-"
            f"{max(distinct, 1)} chance of putting back the very token it was "
            "meant to remove, which raises precision without any anchor "
            "holding anything. Point MODELMRI_CORPUS at a larger text file, "
            "one sentence per line."
        )

    return ids, {
        "corpus": label,
        "sentences": len(sentences),
        "draws_in_pool": len(ids),
        "distinct_ids": distinct,
        "control_ids_dropped": dropped,
        "weighting": (
            "frequency-weighted — an id occurring twice in the corpus is twice "
            "as likely to be drawn, so the pool is that corpus's own unigram "
            "distribution rather than uniform over its types. The two are "
            "different measurements."
        ),
    }


def _screen_key(row: dict) -> tuple[float, int]:
    """Order screened candidates: precision first, then nearest `position`.

    A row with no draws behind it (`measured` False) can only mean the anchor
    plus this candidate covers everything perturbable, where the prediction
    holds by construction — so it sorts by its `implied` value. Reading `point`
    there would be reading a None, and defaulting it to 0.0 would rank a
    candidate that trivially holds the answer BELOW one that does not.

    HOW FAR THAT CLAIM GOES, because a mutation test asked. Replacing the
    `implied` branch with a bare 0.0 leaves `find_anchor`'s answers unchanged,
    and the reason is arithmetic rather than luck: an unmeasured screening row
    needs `perturbable` to be a subset of `anchor | {index}`, and `anchor` is
    always a subset of `candidates = perturbable[-max_candidates:]`, so it
    needs `candidates == perturbable` AND `len(perturbable) - len(anchor) == 1`
    — at which point `remaining` holds exactly that one index and `max` is
    picking from a list of one. So the ordering below cannot change an answer
    through today's `find_anchor`, and
    `test_an_unmeasured_screen_row_outranks_a_measured_one` tests this function
    directly instead of pretending otherwise. It is a guarantee about the
    function, kept so that a future caller with a wider candidate window
    inherits the right ranking rather than a silent 0.0.
    """
    value = row["point"] if row["measured"] else (row.get("implied") or 0.0)
    return float(value), int(row["index"])


def _pool_quality(distinct_ids: int, distinct_templates: int, n_samples: int) -> dict:
    """Whether the perturbation was a distribution or a handful of words.

    `MIN_DISTINCT_IDS` is argued at length where it is defined and was enforced
    in exactly one place: `donor_pool`. `find_anchor` takes `pool` directly —
    which is how every hand-built analysis and every test reaches it — and
    accepted a pool of one id, then published `held: 24, samples: 24` with a
    Wilson interval of [0.862, 1.000] as though 24 independent perturbations
    had been tried. They were 24 copies of one sequence, and nothing in the
    payload let a reader notice: `pool_size` reported the LENGTH OF THE LIST,
    so `[1] * 200` printed 200.

    Not a refusal, deliberately, and the line is worth stating. `donor_pool`
    refuses at the place where a corpus is CHOSEN, which is where the reader
    can act. `find_anchor` is also the entry point for a deliberately tiny
    fixture pool — the one in tests/test_anchors.py has 8 distinct ids on
    purpose, so that no draw can reinstate a trigger and every precision is
    exactly 0 of n or exactly n of n. Refusing there would delete the only
    fixture on which this module's arithmetic is checkable. What it must never
    do is publish the interval without the counts that say how much evidence is
    behind it.
    """
    weak_pool = distinct_ids < MIN_DISTINCT_IDS
    repeated = distinct_templates < n_samples
    reasons = []
    if weak_pool:
        reasons.append(
            f"the pool holds {distinct_ids} distinct "
            f"{'id' if distinct_ids == 1 else 'ids'}, below the "
            f"{MIN_DISTINCT_IDS} `donor_pool` requires: each draw has about a 1 "
            f"in {max(distinct_ids, 1)} chance of reinstating the very token it "
            "was meant to remove, which raises precision without any anchor "
            "holding anything"
        )
    if repeated:
        reasons.append(
            f"the {n_samples} draws produced only {distinct_templates} distinct "
            f"perturbation {'template' if distinct_templates == 1 else 'templates'}"
            ", so the sample size behind every interval here overstates how "
            "many different perturbations were tried"
        )
    return {
        "distinct_ids": distinct_ids,
        "distinct_templates": distinct_templates,
        "below_min_distinct_ids": weak_pool,
        "templates_repeat": repeated,
        "note": (
            "; ".join(reasons).capitalize() + "."
            if reasons
            else (
                f"{distinct_ids} distinct ids and {distinct_templates} distinct "
                f"templates behind {n_samples} draws."
            )
        ),
    }


def _group_of(
    index: int, typed_span: tuple[int, int] | None, n_prompt: int | None
) -> str:
    """typed / template / generated / unknown, on `attribute.rank_tokens`'s rules.

    Same four values and the same order of tests, deliberately: an anchor row
    and an attribution row describing the same index must not disagree about
    what that index IS. The order is load-bearing and the reasoning is written
    out at `attribute.rank_tokens` — past the prompt nothing is template and
    nothing is typed, so `generated` is checked first; and an absent span is
    `unknown` rather than a permissive "all of it is yours", because "we could
    not find your words" and "all of them are yours" are opposite claims.

    Duplicated rather than imported because it lives inside `rank_tokens` and
    has no name of its own. If it ever grows one, this should call it.
    """
    if n_prompt is not None and index >= n_prompt:
        return "generated"
    if typed_span is None:
        return "unknown"
    if typed_span[0] <= index < typed_span[1]:
        return "typed"
    return "template"


def _reject_bools(pairs: Iterable[tuple[str, Any]]) -> None:
    """Bools first, everywhere, because `isinstance(True, int)` is True.

    Kept as a function rather than a loop in each entry point because the two
    entry points below drifted apart once already: `find_anchor` guarded six
    names and `n_prompt` was not one of them, so `n_prompt=True` became
    `n_prompt == 1` and `_group_of` labelled EVERY anchor token "generated" —
    a provenance claim about the reader's own text, fabricated, in the
    published payload. One list, both callers.
    """
    for name, value in pairs:
        if isinstance(value, bool):
            raise BadRequest(f"{name} is a number, not a boolean.")


def _finite(name: str, value: Any) -> float:
    """A float that is really a float, and really finite.

    NaN and inf compare False against every bound, so `if not 0 < x < 1` does
    catch them — by accident, and with a sentence about a range rather than
    about what was actually wrong. This says which it was.
    """
    if isinstance(value, bool):
        raise BadRequest(f"{name} is a number, not a boolean.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise BadRequest(
            f"{name} must be a number, got {type(value).__name__}."
        ) from None
    if not math.isfinite(number):
        raise BadRequest(
            f"{name} must be a finite number, got {number}. NaN and inf compare "
            "False against every bound, so a value like this does not fail a "
            "range check — it passes one, and every number downstream of it is "
            "arithmetic on something nobody measured."
        )
    return number


def _check_confidence(confidence: Any) -> float:
    """`confidence` is a public parameter and used to reach `inv_cdf` unguarded.

    `NormalDist().inv_cdf` raises `statistics.StatisticsError` for 1.0 and
    `ValueError` for NaN — neither is a `Refusal`, so a route answered 500 for
    a request this file understood perfectly. And 0.0 or -0.2 DID refuse, but
    only from inside `wilson_interval`, 27 forward passes later. Checked once,
    here, before anything is spent.
    """
    number = _finite("confidence", confidence)
    if not 0.0 < number < 1.0:
        raise BadRequest(
            f"confidence must be strictly between 0 and 1, got {number}. It is "
            "the coverage of the Wilson interval every precision below is "
            "reported with; 1.0 would be an interval of infinite width and 0.0 "
            "one of no width."
        )
    return number


def _check_span(typed_span: Any, n_prompt: Any, seq: int) -> None:
    """`typed_span` and `n_prompt` decide what each anchor row CLAIMS to be.

    `_group_of` reads both, so a malformed one is not a crash in the arithmetic
    — it is a wrong word next to the reader's own token. `typed_span=True` used
    to reach `typed_span[0]` and raise an untyped TypeError (a 500); a span
    that runs backwards would silently make `typed` unreachable and label
    everything `template`.
    """
    if n_prompt is not None:
        if isinstance(n_prompt, bool):
            raise BadRequest(
                "n_prompt is a number, not a boolean. `n_prompt=True` is 1 to "
                "`isinstance`, which would label every token past index 0 "
                "'generated' — a claim about where the reader's words came "
                "from, made up."
            )
        if not isinstance(n_prompt, int) or n_prompt < 0:
            raise BadRequest(
                f"n_prompt is the number of prompt tokens, got {n_prompt!r}. It "
                "must be a non-negative integer, or None for 'the caller could "
                'not tell\', which is what `group: "unknown"` reports.'
            )
    if typed_span is None:
        return
    if isinstance(typed_span, bool) or not isinstance(typed_span, (tuple, list)):
        raise BadRequest(
            f"typed_span must be a (start, end) pair or None, got "
            f"{type(typed_span).__name__}. None means 'the caller could not "
            'locate your words\' and those rows come back `group: "unknown"` — '
            "it does not mean 'all of it is yours'."
        )
    if len(typed_span) != 2:
        raise BadRequest(
            f"typed_span must be a (start, end) pair, got {len(typed_span)} values."
        )
    start, end = typed_span
    if isinstance(start, bool) or isinstance(end, bool):
        raise BadRequest("typed_span bounds are numbers, not booleans.")
    if not isinstance(start, int) or not isinstance(end, int):
        raise BadRequest("typed_span bounds must be integer token indices.")
    if not 0 <= start <= end <= seq:
        raise BadRequest(
            f"typed_span {(start, end)} is not a half-open range inside a "
            f"sequence of {seq} tokens. A backwards span would make `typed` "
            "unreachable and quietly label every token 'template'."
        )


def _check_pool(pool_tensor: torch.Tensor, vocab: int | None, source: str) -> None:
    """Every donor id must be one the model can actually embed.

    The guard shipped checking only the TOP of the range, so a negative id
    sailed past it and produced exactly the failure the message describes — an
    IndexError partway through the sweep, untyped, a 500 with no sentence.
    `nn.Embedding` treats a negative index as out of range rather than wrapping,
    which is measurable: id -1 against a real `nn.Embedding(16, 8)` raises
    "index out of range in self".

    `vocab` may be None, and None is not 0. A model that does not report a
    vocabulary size means the TOP of the range cannot be checked here; the
    caller says so in the payload rather than the guard silently passing
    everything, which is what `int(getattr(..., 0) or 0)` did.
    """
    if not int(pool_tensor.numel()):
        return
    lowest = int(pool_tensor.min())
    if lowest < 0:
        raise BadRequest(
            f"the donor pool contains id {lowest}, and a token id is an index "
            "into an embedding table. Negative indices are out of range there "
            "rather than counted from the end, so this is an index error "
            "partway through the sweep, not a perturbation."
        )
    if vocab is None:
        return
    highest = int(pool_tensor.max())
    if highest >= vocab:
        raise BadRequest(
            f"the donor pool contains id {highest} and this model's vocabulary "
            f"is {vocab} wide ({source}). A replacement the model cannot embed "
            "is an index error partway through the sweep, not a perturbation."
        )


def _declared_vocab(model: Any) -> int | None:
    """`config.vocab_size` if the model states one, and None if it does not.

    None rather than 0. `int(getattr(...) or 0)` turned "this model does not
    say" into "the vocabulary is zero tokens wide", which disabled the pool
    guard entirely and dropped the fp32 vocabulary vector out of
    `retained_bytes` as if it were free. Both callers below fall back to
    MEASURING the width from a real pass's logits, which is a number rather
    than a default.
    """
    declared = getattr(getattr(model, "config", None), "vocab_size", None)
    if declared is None or isinstance(declared, bool):
        return None
    try:
        width = int(declared)
    except (TypeError, ValueError):
        return None
    return width if width > 0 else None


def _check_shape_and_counts(
    ids: torch.Tensor,
    *,
    position: int,
    max_candidates: int,
    max_size: int,
    n_samples: int,
    n_search: int,
    pool: Sequence[int],
) -> int:
    """The request surface `find_anchor` and `estimate_cost` must agree about.

    They did not. `estimate_cost` priced runs `find_anchor` refuses — an empty
    pool, a batched `ids`, `max_candidates=0` (where `perturbable[-0:]` is
    Python for THE WHOLE LIST, so it silently priced a 5-candidate search when
    asked about a 0-candidate one) — and crashed untyped on a position outside
    the sequence. A cost that describes a run nobody can make is worse than no
    cost. One function, both entry points, so they cannot drift again.

    Returns the sequence length, which both callers want next.
    """
    _reject_bools(
        (
            ("position", position),
            ("max_candidates", max_candidates),
            ("max_size", max_size),
            ("n_samples", n_samples),
            ("n_search", n_search),
        )
    )
    if ids.dim() != 2 or int(ids.shape[0]) != 1:
        # A plain RuntimeError and not a Refusal, on attribute.py's rule: the
        # caller of this is runtime.py building `ids` from `last_ids`, so a
        # violation is this package contradicting itself and belongs on the 500
        # path with a traceback rather than in a sentence for a reader who
        # cannot act on it.
        raise RuntimeError(
            f"anchors need one unbatched sequence shaped [1, S], got "
            f"{tuple(ids.shape)}."
        )
    seq = int(ids.shape[1])
    if not 0 <= position < seq:
        raise BadRequest(f"position {position} is outside a sequence of {seq} tokens.")
    if n_samples < 1 or n_search < 1:
        raise BadRequest("n_samples and n_search must each be at least 1.")
    if n_search > n_samples:
        raise BadRequest(
            f"n_search ({n_search}) cannot exceed n_samples ({n_samples}) — "
            "screening reads a prefix of the verification draws so that the "
            "two never disagree about which perturbations they saw."
        )
    if max_candidates < 1 or max_size < 1:
        raise BadRequest("max_candidates and max_size must each be at least 1.")
    if len(pool) < 1:
        raise BadRequest(
            "the perturbation needs replacement ids to draw from and the pool "
            "is empty. `donor_pool(tokenizer)` builds one from the bundled "
            "corpus."
        )
    return seq


def find_anchor(
    model: Any,
    ids: torch.Tensor,
    *,
    position: int,
    pool: Sequence[int],
    perturbation: dict | None = None,
    control_ids: Iterable[int] = (),
    typed_span: tuple[int, int] | None = None,
    n_prompt: int | None = None,
    max_candidates: int = MAX_CANDIDATES,
    max_size: int = MAX_ANCHOR_SIZE,
    n_samples: int = N_SAMPLES,
    n_search: int = N_SEARCH,
    target: float = PRECISION_TARGET,
    confidence: float = CONFIDENCE,
    seed: int = 0,
    prune: bool = True,
    decode=None,
) -> dict:
    """The smallest set of tokens found that holds the prediction on its own.

    `ids` is one unbatched sequence `[1, S]`; `position` is the index whose
    next-token argmax is the claim. `pool` is the replacement ids, and
    `perturbation` is the description `donor_pool` returned beside them — pass
    it, or the payload cannot say what the tokens were replaced WITH, and a
    precision whose distribution is unnamed cannot be interpreted or
    reproduced.

    `typed_span` and `n_prompt` do the same job as in `attribute.rank_tokens`
    and carry the same warning: `None` for the span is "the caller could not
    locate your words", not "all of it is yours", and those rows come back
    `group: "unknown"`.

    Returns a dict; `anchor_indices` is the answer and everything else is the
    evidence for it. The search is greedy — `minimality.smaller_may_exist` is
    True in every payload this function produces, and it means it.
    """
    # Bools first, everywhere, because `isinstance(True, int)` is True and
    # `position=True` would otherwise quietly attribute at index 1. The list
    # lives in `_reject_bools` and `_check_span` now, because keeping it inline
    # is how `n_prompt` and `confidence` came to be missing from it.
    _reject_bools((("seed", seed),))
    seq = _check_shape_and_counts(
        ids,
        position=position,
        max_candidates=max_candidates,
        max_size=max_size,
        n_samples=n_samples,
        n_search=n_search,
        pool=pool,
    )
    confidence = _check_confidence(confidence)
    target = _finite("target", target)
    if not 0.0 < target <= 1.0:
        raise BadRequest(
            f"target must be a precision in (0, 1], got {target}. It is the bar "
            "the interval's LOWER bound has to clear."
        )
    _check_span(typed_span, n_prompt, seq)
    _impossible_target(n_samples, target, confidence)

    started = time.perf_counter()
    control = {int(c) for c in control_ids}
    device = ids.device

    # THE CANDIDATE SET. Identical to `attribute.rank_tokens`'s, minus control
    # tokens, and the sameness is the point — see the module docstring. Index 0
    # is an attention sink whose score follows the position rather than the
    # token; `position` is its own query; anything after `position` cannot reach
    # it through a causal mask. Control tokens are held because perturbing the
    # chat template measures the template.
    perturbable = [i for i in range(1, position) if int(ids[0, i]) not in control]
    movable = set(perturbable)
    held_fixed = [i for i in range(seq) if i not in movable]
    if not perturbable:
        nothing = (
            "is the first token and has read nothing at all"
            if position == 0
            else "has read nothing but index 0 (an attention sink rather than "
            "content), itself, and control tokens"
        )
        raise AnchorError(
            f"position {position} {nothing}, so there is no set of your words "
            "to hold the answer with. Attribute at a position that has read at "
            "least one ordinary token."
        )

    candidates = perturbable[-max_candidates:]
    anchorable = set(candidates)
    n_candidates, n_tested = len(perturbable), len(candidates)
    outside = [i for i in perturbable if i not in anchorable]

    pool_tensor = torch.tensor([int(p) for p in pool], dtype=ids.dtype, device=device)
    declared_vocab = _declared_vocab(model)
    # A model that states its width is checked before a single pass is spent,
    # which is what `test_a_pool_the_model_cannot_embed_is_refused_before_the_sweep`
    # measures. A model that does not is checked below against the width its
    # own logits turn out to have — one pass later, and still before anything
    # perturbed reaches it. The negative arm needs neither and runs here.
    _check_pool(pool_tensor, declared_vocab, "model.config.vocab_size")
    # The pool is a distribution, and `len(pool)` is not its size. A pool of
    # [1] * 200 has pool_size 200 and exactly one draw in it; `pool_size` alone
    # let that publish 24 of 24 with a Wilson interval as though 24 independent
    # perturbations had been tried. The true count travels beside the length.
    pool_distinct = int(torch.unique(pool_tensor).numel())

    # ONE block of templates, drawn once, reused by every anchor — common random
    # numbers, so a comparison between two anchors is not also a comparison
    # between two lotteries. `[n_samples, len(perturbable)]` int64: 32 KiB at
    # the defaults, and nothing else here scales with the sample count.
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    draws = torch.randint(
        len(pool_tensor), (n_samples, len(perturbable)), generator=generator
    )
    templates = pool_tensor[draws.to(device)]
    # How many DISTINCT perturbations that block actually contains. `n_samples`
    # is how many draws were TAKEN, and the two are not the same number: with a
    # one-id pool all 24 rows are the same sequence, and publishing 24 of 24
    # with a Wilson interval over that is one trial restated 24 times wearing
    # the clothes of 24. Counting them sorts a copy of the block, so the cost is
    # a small multiple of the block itself and does not scale with the passes.
    distinct_templates = int(torch.unique(templates, dim=0).shape[0])
    perturbable_at = torch.tensor(perturbable, dtype=torch.long, device=device)

    ones = torch.ones((1, seq), dtype=torch.long, device=ids.device)
    position_ids = torch.arange(seq, device=ids.device).unsqueeze(0)
    # Two counters rather than two lists. The version that shipped appended
    # `ones` and `position_ids` on EVERY pass and compared the lists at the
    # end, which is 2 Python pointers per forward pass — measured at 322,187 B
    # of a 758,240 B peak at 20,595 passes, against 20595*8*2 = 329,520 B of
    # accounting — and it made this module's "a thousand samples cost one
    # pass's peak and a thousand bytes" false by the pass count. A counter
    # carries the same guarantee about every call site in this function at a
    # fixed size, and the counts are PUBLISHED rather than only raised on.
    other_mask_passes = 0
    other_position_passes = 0
    passes = 0

    def forward_logits(
        row: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        nonlocal passes, other_mask_passes, other_position_passes
        passes += 1
        mask = ones if mask is None else mask
        positions = position_ids if positions is None else positions
        other_mask_passes += int(mask is not ones)
        other_position_passes += int(positions is not position_ids)
        out = model(input_ids=row, attention_mask=mask, position_ids=positions)
        if tuple(row.shape) != (1, seq):
            raise RuntimeError(
                f"a perturbed sequence came out shaped {tuple(row.shape)} "
                f"rather than {(1, seq)}; substitution must preserve every "
                "index, or the ranking is about the renumbering."
            )
        return out.logits[0, position]

    with torch.no_grad():
        base_logits = forward_logits(ids)
        base = distribution(base_logits)
        base_top = int(base.argmax())

        # The floor, on ablate.py's pattern: the same pass twice. It is not
        # decoration here — every precision below is an argmax comparison
        # against `base_top`, so if the argmax is not even stable against a
        # repeat of its own pass there is nothing to hold.
        repeat = distribution(forward_logits(ids))
        floor = kl_nats(base, repeat)
        if int(repeat.argmax()) != base_top:
            raise AnchorError(
                "this model's own top token at that position changed between "
                "two identical forward passes, so 'the prediction still holds' "
                "has nothing stable to hold against. Every precision below "
                "would be measuring non-determinism. Try a shorter sequence or "
                "a deterministic kernel."
            )

        # The plain call every other reader of this model makes. Substitution
        # cannot re-phase RoPE the way masking can (the mask stays all-ones and
        # the length never changes), but "our explicit arguments select the same
        # computation" is still a claim, and it is one pass to check.
        passes += 1
        plain = distribution(model(ids).logits[0, position])
        agreement = kl_nats(plain, base)
        if int(plain.argmax()) != base_top:
            raise AnchorError(
                f"supplying an all-ones attention mask and explicit "
                f"position_ids changes this model's top token at position "
                f"{position} (it moved the distribution by {agreement:.3e} "
                "nats). Its position semantics are not the ones this "
                "measurement assumes, so every precision below would include "
                "that change. It would work on a model that takes position_ids "
                "at face value."
            )

        if base_top in control:
            raise AnchorError(
                f"the token being held is a control token "
                f"({decode(base_top) if decode else base_top!r}), which a chat "
                "template all but guarantees at the end of a prompt. An anchor "
                "for it would be a set of your words that keeps the template "
                "working. Generate at least one token and anchor at a position "
                "where the model is answering."
            )

        # THE WIDTH THIS MODEL ACTUALLY HAS, which is a measurement rather than
        # a field a checkpoint may not carry. `config.vocab_size` used to
        # collapse to 0 when absent and `if vocab and ...` then turned the whole
        # out-of-range guard off — silently, on exactly the models that could
        # not tell us. The base pass ran on the ORIGINAL ids, so nothing
        # unembeddable has reached the model yet and this is still "before the
        # sweep".
        measured_vocab = int(base.numel())
        _check_pool(pool_tensor, measured_vocab, "measured from the base pass's logits")

        # topk rather than a full sort: the second-largest probability is all
        # that is wanted and sorting the vocabulary to get it is V log V for two
        # numbers.
        top2 = base.topk(2).values if measured_vocab > 1 else None
        margin = None if top2 is None else float(top2[0] - top2[1])

        # Evaluations that perturbed nothing and therefore spent no passes. Not
        # a curiosity: it is the difference between `cost_passes` and the loop
        # on the covering path, and it travels in the payload beside `passes`
        # so a reader can reconcile the two without re-deriving the search.
        free_evaluations = 0

        def evaluate(anchor: set[int], n: int) -> tuple[int, int]:
            """How many of the first `n` templates hold `base_top`."""
            nonlocal free_evaluations
            free = [k for k, index in enumerate(perturbable) if index not in anchor]
            if not free:
                # Nothing to perturb: every draw is the original sequence. The
                # caller decides what to do with that; spending n passes to
                # re-learn `base_top` is not it.
                free_evaluations += 1
                return 0, 0
            columns = torch.tensor(free, dtype=torch.long, device=device)
            where = perturbable_at[columns]
            held = 0
            for sample in range(n):
                row = ids.clone()
                row[0, where] = templates[sample, columns]
                held += int(forward_logits(row).argmax()) == base_top
            return held, n

        # --------------------------------------------------- the two ends
        floor_held, floor_n = evaluate(set(), n_samples)
        base_rate = _proportion(floor_held, floor_n, confidence=confidence)

        steps: list[dict] = []
        anchor: list[int] = []
        verified = base_rate
        stopped = ""

        # An empty anchor that already clears the bar is a real finding and not
        # a failure: nothing in the candidate window is holding the prediction,
        # so it is being held by something this search cannot touch — the
        # template, the sink, the position, or the model's prior.
        if base_rate["low"] >= target:
            stopped = "empty-anchor-sufficient"
            ceiling = _unmeasured(
                "not measured: the empty anchor already clears the target, so "
                "the best any anchor could do was never the question."
            )
        else:
            if outside:
                ceiling_held, ceiling_n = evaluate(anchorable, n_samples)
                ceiling = _proportion(ceiling_held, ceiling_n, confidence=confidence)
                ceiling["n_perturbed"] = len(outside)
            else:
                ceiling = _unmeasured(
                    "the candidate set covers every perturbable position, so "
                    "holding all of it perturbs nothing and there is no sample "
                    "to take. The implied value follows from the base pass "
                    "being stable across a repeat, not from a measurement.",
                    implied=1.0,
                )
                ceiling["n_perturbed"] = 0

            if ceiling["measured"] and ceiling["high"] < target:
                # Arithmetic, not pessimism: no anchor drawn from this window
                # can beat the window itself, so the whole greedy budget would
                # buy a guaranteed failure. Refusing to spend it is the point of
                # measuring the ceiling before the search rather than after.
                stopped = "ceiling-below-target"
            else:
                stopped = "max-size"
                remaining = list(candidates)
                for step in range(min(max_size, len(candidates))):
                    screened = []
                    for index in remaining:
                        held, n = evaluate(set(anchor) | {index}, n_search)
                        screened.append(
                            {
                                "index": index,
                                "token": _token(ids, index, decode),
                                **_proportion(held, n, confidence=confidence),
                            }
                        )
                    # Best screened precision; ties to the token nearest
                    # `position`. On an AND every candidate screens at exactly
                    # zero on the first step, so the tie-break is what decides
                    # which half of the pair the reader sees first, and recency
                    # is the same ordering the candidate cap already uses.
                    best = max(screened, key=_screen_key)
                    anchor.append(best["index"])
                    remaining = [i for i in remaining if i != best["index"]]

                    held, n = evaluate(set(anchor), n_samples)
                    verified = _proportion(held, n, confidence=confidence)
                    steps.append(
                        {
                            "step": step + 1,
                            "added_index": best["index"],
                            "added_token": best["token"],
                            "screened": screened[:5],
                            "n_screened": len(screened),
                            "screened_truncated": len(screened) > 5,
                            "screen_samples": n_search,
                            "verified": verified,
                        }
                    )
                    if not verified["measured"]:
                        # The anchor now holds every perturbable token, so no
                        # draw changes anything and its precision is 1.0 by
                        # construction. `found` stays False: nothing was
                        # demonstrated, and calling that sufficiency would be
                        # the one fabricated verdict this module exists to
                        # avoid.
                        stopped = "anchor-covers-every-perturbable-token"
                        break
                    if verified["low"] >= target:
                        stopped = "target-reached"
                        break

        found = stopped in ("target-reached", "empty-anchor-sufficient")

        # ------------------------------------------- backward elimination
        #
        # Greedy forward selection OVERSHOOTS, and not rarely. Measured on the
        # AND fixture in tests/test_anchors.py: with both triggers screening at
        # exactly 0.0 on the first step, the tie-break takes a token that turns
        # out to carry nothing, and the search then needs three tokens to reach
        # a target two of them hold on their own. Reporting that three-token set
        # as the anchor would be reporting the tie-break.
        #
        # So every element is offered for removal against the set as it
        # currently stands, and the sweep repeats until one completes with
        # nothing removed. That terminating sweep is the evidence: every element
        # was tested against the FINAL anchor, so no single removal survives.
        # It is bounded — each removal shortens the anchor — and every
        # evaluation it spends is recorded in `drops` rather than only its
        # conclusion.
        drops: list[dict] = []
        irreducible: bool | None = None
        if prune and found and anchor:
            while True:
                removed = None
                for index in list(anchor):
                    rest = set(anchor) - {index}
                    if rest:
                        held, n = evaluate(rest, n_samples)
                        without = _proportion(held, n, confidence=confidence)
                        reused = ""
                    else:
                        # Dropping the only element leaves the empty anchor,
                        # which IS the base rate and was measured before the
                        # search started. Re-running it would spend n_samples
                        # passes to reproduce a number already in this payload.
                        without = base_rate
                        reused = (
                            "base_rate — dropping the only element leaves the "
                            "empty anchor, which was already measured"
                        )
                    still = bool(without["measured"] and without["low"] >= target)
                    drops.append(
                        {
                            "dropped_index": index,
                            "dropped_token": _token(ids, index, decode),
                            "anchor_at_the_time": sorted(anchor),
                            "still_sufficient": still,
                            "removed": still,
                            "reused": reused,
                            **without,
                        }
                    )
                    if still:
                        anchor = [i for i in anchor if i != index]
                        # The set that survived the removal is the new anchor,
                        # and this measurement of it is its precision. Taking it
                        # again would be paying twice for one number.
                        verified = without
                        removed = index
                        break
                if removed is None:
                    break
            # The sweep that broke the loop removed nothing, and it tested every
            # element against the anchor as it now stands. That is exactly the
            # claim, and it is no larger than the claim.
            irreducible = True

    # The instrumentation check, on attribute.py's pattern and for its reason:
    # one mask object and one position_ids object across every scoring pass,
    # tested against the objects rather than their values so that a
    # rebuilt-but-equal tensor still fails it.
    #
    # What this DOES and DOES NOT prove, because the list version overstated
    # it. It is a guard on the call sites in this function — every one of them
    # goes through `forward_logits`, and anything handed there that is not the
    # single pair is counted. It is not evidence about what the model received,
    # because this function cannot see inside the model; that half is measured
    # from the model's own recorded arguments in
    # `test_every_pass_is_handed_the_same_mask_and_the_same_position_ids`.
    if other_mask_passes or other_position_passes:
        raise RuntimeError(
            f"{other_mask_passes} passes were handed an attention_mask and "
            f"{other_position_passes} a position_ids other than the single "
            "all-ones/arange pair. Substitution is only safe from the "
            "re-phasing attribute.py documents while those two never change."
        )

    return {
        "position": position,
        "target_token": _token_id(base_top, decode),
        "target_token_id": base_top,
        "base_p_top": round(float(base[base_top]), 5),
        # The resolution of the claim being anchored. A top-1 that leads by
        # 0.002 flips on arithmetic, and every precision below it is a count of
        # coin flips rather than of a prediction holding. None only when the
        # vocabulary has one entry, which is not a model anybody runs.
        "base_margin": None if margin is None else round(margin, 5),
        "noise_floor_kl": round(floor, 6),
        "agreement_kl": round(agreement, 6),
        "anchor_indices": sorted(anchor),
        "anchor": [
            {
                "index": index,
                "token": _token(ids, index, decode),
                "group": _group_of(index, typed_span, n_prompt),
                "added_at_step": order + 1,
            }
            for order, index in enumerate(anchor)
        ],
        "size": len(anchor),
        "found": found,
        "stopped_because": stopped,
        "precision": verified,
        "base_rate": base_rate,
        "ceiling": ceiling,
        "target": target,
        # The bar and the best any sample of this size could clear, side by
        # side. A reader who sees `found: false` at a precision of 1.000 should
        # be able to tell "no anchor holds this" from "these draws cannot carry
        # that bound" without leaving the payload.
        # Cut DOWNWARDS to 4 places, not rounded — see `publishable_ceiling`.
        # `round` published 0.9434 at the shipped default of 64 draws and the
        # exact bound is 0.9433759402, so a reader who acted on the number
        # beside `target` was refused by a sentence that printed their own
        # value back at them.
        "target_ceiling": publishable_ceiling(n_samples, confidence=confidence),
        "target_ceiling_exact": reachable_target(n_samples, confidence=confidence),
        "steps": steps,
        "minimality": {
            "search": "greedy forward selection, then backward elimination",
            "smaller_may_exist": True,
            "drop_one_checked": bool(drops),
            "irreducible_under_single_removal": irreducible,
            "removed_by_elimination": [
                d["dropped_index"] for d in drops if d["removed"]
            ]
            if drops
            else [],
            "drops": drops,
            "note": (
                "Greedy. This is the smallest set THIS SEARCH assembled, not "
                "the minimal set: a smaller sufficient set it never tried may "
                "exist, and so may a different set of the same size — on an OR "
                "of two tokens both singletons are sufficient and the tie-break "
                "decides which one you see. The elimination sweep, when it ran, "
                "says only that no SUBSET of this anchor is sufficient."
            ),
        },
        "perturbation": {
            **(perturbation or {}),
            "pool_size": len(pool_tensor),
            # The length of the list and the size of the distribution are two
            # numbers, and only the first one used to be published. A pool of
            # [1] * 200 reported pool_size 200 for one distinct draw.
            "pool_distinct_ids": pool_distinct,
            "min_distinct_ids": MIN_DISTINCT_IDS,
            "distinct_templates": distinct_templates,
            "vocabulary": {
                "size": declared_vocab
                if declared_vocab is not None
                else measured_vocab,
                "source": (
                    "model.config.vocab_size"
                    if declared_vocab is not None
                    else "measured from the base pass's logits"
                ),
                "declared": declared_vocab,
                "measured": measured_vocab,
            },
            "quality": _pool_quality(pool_distinct, distinct_templates, n_samples),
            "replaces": (
                "every perturbable position outside the anchor, independently, "
                "one draw per position per sample"
            ),
            "samples": n_samples,
            "screen_samples": n_search,
            "paired": (
                "one block of perturbation templates is drawn once and every "
                "anchor is scored against the same block, so two precisions "
                "here differ by their anchors and not by their draws"
            ),
            "seed": seed,
            # "candidate" was the wrong word and the same payload contradicted
            # it two keys away: `evaluate` perturbs every PERTURBABLE position
            # outside the anchor, including the ones the `max_candidates` cap
            # kept out of the search. On a 13-token prompt capped at 3 that is
            # 11 positions perturbed while this sentence said 3.
            "sentence": (
                "Precision is the fraction of "
                f"{n_samples} perturbations in which this model's top-1 token "
                f"at position {position} was still "
                f"{_token_id(base_top, decode)!r}, where a perturbation "
                f"replaces all {n_candidates} perturbable tokens outside the "
                f"anchor — not just the {n_tested} the search could choose "
                "from — with ids drawn from "
                f"{(perturbation or {}).get('corpus', 'the supplied pool')}."
            ),
        },
        "candidates": {
            "n_candidates": n_candidates,
            "n_tested": n_tested,
            "truncated": n_tested < n_candidates,
            "tested_span": [candidates[0], candidates[-1] + 1],
            "max_candidates": max_candidates,
            "max_size": max_size,
            "coverage": (
                f"{n_tested} of {n_candidates} perturbable tokens could be "
                "chosen for the anchor; the other "
                f"{n_candidates - n_tested} were perturbed on every sample and "
                "were never offered to the search, which is not the same as "
                "having been tried and rejected."
            ),
        },
        "held_fixed": {
            "count": len(held_fixed),
            "indices": held_fixed[:HELD_FIXED_LISTED],
            "listed": min(len(held_fixed), HELD_FIXED_LISTED),
            "why": (
                "Index 0 (an attention sink, whose score follows the position "
                f"rather than the token), position {position} itself, "
                "everything after it (unreachable under a causal mask), and "
                "every control token — perturbing a chat template measures the "
                "template. These are never perturbed and never anchored, so "
                "whatever they contribute is inside every number here."
            ),
        },
        "seed": seed,
        "passes": passes,
        "accounting": {
            # Evaluations that perturbed nothing, spent no passes, and are the
            # entire difference between `cost_passes` and the loop on the
            # covering path. `cost_passes` said 135 for a run that spent 105
            # until this was counted rather than assumed away.
            "free_evaluations": free_evaluations,
            "covering_final_step": stopped == "anchor-covers-every-perturbable-token",
            "passes_with_another_mask": other_mask_passes,
            "passes_with_another_position_ids": other_position_passes,
            "why": (
                "an evaluation whose anchor already covers every perturbable "
                "position has nothing to perturb, so it spends zero forward "
                "passes rather than n_samples. Every pass this run spent is in "
                "`passes`; `cost_passes(..., covering_final_step=True)` "
                "reproduces it exactly on that path."
            ),
        },
        "elapsed_s": round(time.perf_counter() - started, 2),
        "means": (
            "An anchor is a SUFFICIENCY claim: these tokens, on their own, held "
            "the model's top-1 next token at this position while every other "
            "PERTURBABLE token was replaced — all "
            f"{n_candidates} of them, including the "
            f"{n_candidates - n_tested} the candidate cap kept out of the "
            "search. It is not a necessity claim and the "
            "two disagree — `attribute.rank_tokens` masks one token out and "
            "asks whether the answer needed it, and a token can be sufficient "
            "without being necessary (an OR) or necessary without being "
            "sufficient (an AND). Precision is a sample: read the interval and "
            "the sample count, not the point. Read it against `base_rate` (what "
            "the prediction does with none of these tokens held) and `ceiling` "
            "(the best any anchor from this candidate set could reach)."
        ),
    }


def _token(ids: torch.Tensor, index: int, decode) -> str:
    return _token_id(int(ids[0, index]), decode)


def _token_id(token_id: int, decode) -> str:
    return decode(token_id) if decode else str(token_id)


def estimate_cost(
    model: Any,
    ids: torch.Tensor,
    *,
    position: int,
    pool: Sequence[int],
    control_ids: Iterable[int] = (),
    max_candidates: int = MAX_CANDIDATES,
    max_size: int = MAX_ANCHOR_SIZE,
    n_samples: int = N_SAMPLES,
    n_search: int = N_SEARCH,
    device_kind: str = "cpu",
) -> dict:
    """What would `find_anchor` cost here? Measured, then multiplied.

    The pass count is exact and portable, and it is a RANGE rather than a
    number: greedy stops when the target is met, so a run can be as short as
    one step or as long as `max_size`. What a pass costs is not portable at
    all, so this runs ONE real iteration on this machine — a perturbed sequence
    through the same forward call, with the same argmax — and projects from it.
    `budget.probe_pass` records what happens to a probe that does less work
    than the loop it prices.

    WHAT `minimum` MEANS, because it used to mean something narrower than the
    word does. It was "one step, target reached", and three cheaper endings
    exist: an empty anchor that already clears the bar spends the fixed passes
    and the base rate and stops before the ceiling is even measured; a ceiling
    below the target stops before any screening; and a one-candidate window
    that covers everything perturbable screens and verifies for free. Measured:
    a 3-token prompt with exactly one perturbable index costs 27 passes, and
    `minimum` said 57 — the bracket this function advertises was violated
    outright at the bottom. So `minimum` is now the floor across every stopping
    path, `one_step` carries the old figure, and both are named.

    This function REFUSES what `find_anchor` would refuse, and for the same
    reasons in the same words — an empty pool, a batched `ids`, a position
    outside the sequence, `max_candidates=0` (where `perturbable[-0:]` is
    Python for the whole list, so it used to price a 5-candidate search when
    asked about a 0-candidate one). A price for a run that cannot be made is
    worse than no price.

    Retained bytes are the template block plus the base distribution over the
    vocabulary. The width comes from the warm pass's own logits rather than
    from `config.vocab_size`, which not every checkpoint carries and which used
    to collapse to 0 — publishing the fp32 vocabulary vector as free.
    """
    from . import budget

    _check_shape_and_counts(
        ids,
        position=position,
        max_candidates=max_candidates,
        max_size=max_size,
        n_samples=n_samples,
        n_search=n_search,
        pool=pool,
    )
    control = {int(c) for c in control_ids}
    seq = int(ids.shape[1])
    perturbable = [i for i in range(1, position) if int(ids[0, i]) not in control]
    if not perturbable:
        raise AnchorError(
            f"position {position} has no ordinary token before it to anchor "
            "with, so there is no search to price."
        )
    candidates = perturbable[-max_candidates:]
    steps = min(max_size, len(candidates))
    ceiling_measured = len(candidates) < len(perturbable)

    ones = torch.ones((1, seq), dtype=torch.long, device=ids.device)
    position_ids = torch.arange(seq, device=ids.device).unsqueeze(0)
    pool_tensor = torch.tensor([int(p) for p in pool], dtype=ids.dtype)
    _check_pool(pool_tensor, _declared_vocab(model), "model.config.vocab_size")
    probe_row = ids.clone()
    probe_row[0, perturbable[-1]] = pool_tensor[0].to(ids.device)

    def one_iteration() -> None:
        with torch.no_grad():
            out = model(
                input_ids=probe_row, attention_mask=ones, position_ids=position_ids
            )
            int(out.logits[0, position].argmax())

    # Warm first, for ablate.estimate_cost's reason: the first pass after a load
    # pays initialisation and probing it prices a sweep several times slower
    # than the one that runs. It is also where the vocabulary width is measured,
    # so `retained_bytes` never has to guess it.
    with torch.no_grad():
        warm = model(ids)
        vocab = int(warm.logits.shape[-1])
    _check_pool(pool_tensor, vocab, "measured from the warm pass's logits")

    probe = budget.probe_pass(one_iteration, device_kind)
    # The floor across every stopping path: the fixed three and the base rate.
    # `empty-anchor-sufficient` stops exactly there, and so does a one-candidate
    # window whose single step covers everything perturbable.
    floor = FIXED_PASSES + n_samples
    one_step = cost_passes(
        n_candidates=len(candidates),
        steps=1,
        n_samples=n_samples,
        n_search=n_search,
        ceiling_measured=ceiling_measured,
        prune_size=0,
        covering_final_step=not ceiling_measured and len(candidates) == 1,
    )
    minimum = min(floor, one_step)
    worst = cost_passes(
        n_candidates=len(candidates),
        steps=steps,
        n_samples=n_samples,
        n_search=n_search,
        ceiling_measured=ceiling_measured,
        # Every elimination sweep removing exactly one element: k + (k-1) + ...
        # The cheap end of elimination is one clean sweep, and it is the far
        # more common one — this is the bound, not the expectation.
        prune_size=steps * (steps + 1) // 2,
    )
    retained = n_samples * len(perturbable) * 8 + vocab * 4

    return {
        "passes": {
            "minimum": minimum,
            "one_step": one_step,
            "worst_case": worst,
            "why_a_range": (
                "greedy stops as soon as the interval's lower bound clears the "
                f"target, so the run is between 0 and {steps} steps. `minimum` "
                f"({minimum}) is the floor across every stopping path: an empty "
                "anchor that already clears the target, or a ceiling below it, "
                "stops before the search starts. `one_step` "
                f"({one_step}) is one full step with the target met, whose "
                "elimination sweep is free because it reuses the base rate. "
                f"`worst_case` ({worst}) is a bound rather than an expectation: "
                f"{steps} steps and an elimination that removes one element per "
                "sweep."
            ),
            "formula": (
                "3 + n_samples (base rate) + n_samples (ceiling, when anything "
                "outside the candidate set is perturbable) + n_search * "
                "sum(C - i) over steps + n_samples per step + n_samples per "
                "removal evaluation, minus n_search + n_samples when the last "
                "step's anchor covers every perturbable position and therefore "
                "perturbs nothing"
            ),
        },
        "estimate_worst_case": budget.project(
            probe, worst, retained_bytes=retained
        ).to_dict(),
        "estimate_minimum": budget.project(
            probe, minimum, retained_bytes=retained
        ).to_dict(),
        "probe": probe.to_dict(),
        "candidates": len(candidates),
        "perturbable": len(perturbable),
        "ceiling_measured": ceiling_measured,
        "retained_bytes": retained,
        "vocab_size": vocab,
        "vocab_source": "measured from the warm pass's logits",
        "retained_note": (
            f"the perturbation templates ({n_samples} x {len(perturbable)} "
            f"int64 = {n_samples * len(perturbable) * 8} B) and one fp32 vector "
            f"over the {vocab}-wide vocabulary ({vocab * 4} B). What the search "
            "holds across passes, and it is the only thing that scales with "
            "n_samples: the per-sample pass takes an argmax and keeps an "
            "integer. Measured with tracemalloc on a model that retains "
            "nothing, a 7-token prompt and 5 perturbable positions: 128 draws / "
            "1,267 passes peaks at 417,433 B and 2,048 draws / 20,595 passes at "
            "436,373 B — 16.25x the passes for 1.045x the peak. Before the two "
            "bookkeeping lists this function used to append to on every pass "
            "were removed, the same two runs peaked at 431,067 B and 758,240 B. "
            "Counting the distinct templates sorts a copy of the block once, "
            "which is a multiple of the block rather than of the pass count."
        ),
    }
