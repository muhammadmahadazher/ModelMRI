"""What a policy would DO, next to what a human actually did.

Three measurements — plus a fourth that rides inside the first and never
touches a recorded action — and all of them are shaped by the same worry: an
action curve is the most persuasive-looking thing this whole project draws.
Two lines on one axis, one labelled "policy" and one "recorded", read as a
score, and a reader will conclude something about a robot from the gap between
them.

So each of these refuses more than it reports.

## Predicted versus recorded

Per-dimension, across an episode. Click a spike, jump the scrubber, hand the
frame to the occlusion map.

Ahead of NVIDIA GR00T, whose open-loop predicted-vs-ground-truth curves are
terminal output: they say WHERE the policy differs and never why. Here the
divergence is a coordinate you can click into the internals with.

**A recorded action is one human demonstration, not ground truth.** A policy
can be right and differ — a different grasp that also works is a large
divergence and a fine policy. That sentence travels in the response body, not
in the docs, because the person who needs it is looking at the chart.

**Open-loop teacher forcing is not closed-loop behaviour.** Every prediction
here is conditioned on the human's observations, so error never compounds the
way it would on a real robot. A policy that looks excellent here can still
drift immediately in the world.

**Different units are a refusal, not a rescale.** A policy normalises actions
against ITS training statistics; a dataset records its own. Overlaying two
curves in different units is the plausible-wrong output ROADMAP #50 names
explicitly, and refusing needs both sides to have stated their units — which
is why `PolicyStatus.normalisation` travels and why empty means "do not
overlay" rather than "identity".

## Chunk consistency, which rides along inside that comparison

A VLA asked about frame `t` answers with a CHUNK: its claim about frames `t`
through `t + H − 1`. The comparison above uses step 0 of each chunk and only
step 0, for the reason the server's own comment gives — pairing step 5 with
frame `t` would call a claim about the future an error about the present.

The rest of the chunk is not waste. The chunk from frame `t` and the chunk
from frame `t + s` both describe the absolute timesteps they overlap on, and
the gap between those two claims is action-chunk consistency: the policy
saying, one observation later, that it now wants something else at a frame it
had already committed to. It is a published failure-prediction signal (Agia
et al.'s STAC, ActProbe's TCE, VLA-FAIL's ACC) and it costs ZERO extra
forward passes, because the chunks were already computed and thrown away.

**Nothing recorded is on either side of that subtraction.** Both operands are
predictions; the demonstrator's action never enters. That is why it is a
separate block with its own `means` rather than a column in the error table —
a policy that predicts the same wrong action every time is perfectly
consistent here, and a reader who has just read "A RECORDED ACTION IS NOT
GROUND TRUTH" will fold the number into that frame unless it says otherwise.

**Not measurable is not zero.** When the stride exceeds the horizon the two
chunks never describe the same timestep and there is nothing to compare;
`measurable: False` and a sentence naming the stride that would work, never an
empty list and never a 0.0 — 0.0 is what a policy that agreed with itself
perfectly would score.

**The index algebra is established; the rest of it is ours.** `chunk_p[dt+k]`
against `chunk_q[k]` over `min(H_q, H_p − dt)` steps is what all three papers
do, unanimously. The distance, the aggregation and the per-steps-ahead
breakdown are not: those three use squared L2, per-dimension L1 over a
velocity-normalised denominator, and MMD between resampled distributions, and
they mean, EMA and cumulatively sum a single adjacent-step number. So this
takes the module's own L2, reports a median with `n` beside it, and says in
the body that the curve over steps ahead is this project's extension rather
than a convention borrowed from somewhere.

## Instruction swap

Run the policy on ONE frame with its own task string and with every other
distinct task string the dataset contains. Compare that spread against the
spread across noise seeds on the identical frame.

The reference is the policy's OWN sampling variance, so no threshold is
invented and no calibration is borrowed from another paper. If swapping "pick
up the red block" for "close the drawer" moves the action less than re-rolling
the sampler does, this says so in those words.

Refuses a single-task dataset by name — fabricating a distractor instruction
would be inventing the experiment — and refuses a deterministic policy, where
the reference collapses to zero and a ratio against it is not a number.

## Input-stream knockout

One bar per input the policy consumes: each camera, the instruction, the
proprioceptive state. Each replaced, alone, by its episode mean.

VLA-Trace did modality knockout and never released code.

**Mean substitution is a specific baseline, not "removal".** The mean frame of
an episode is a real image the policy has opinions about; it is not absence.
And single-stream knockouts DO NOT ADD UP — the sum of individual effects is
not the effect of removing several, because the policy's inputs interact. Both
sentences ship in the response, following the `means` convention `ablate.py`
and `attribute.py` already set.

The empty-instruction arm is labelled "no instruction", never "the instruction
did not matter".
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from . import fmt
from .errors import BadRequest, Refusal

# How many frames a predicted-vs-recorded run may cover before it must be
# strided. Each one is a forward pass through a VLA -- measured at 49 seconds
# on CPU for SmolVLA -- so an unstrided 200-frame episode is nearly three
# hours. The cap is on WORK, and what was dropped is always reported.
MAX_FRAMES_PER_RUN = 64

# Seeds drawn for the sampling-variance reference. Small because each is a
# forward pass, and enough that the spread is a spread rather than a pair.
# The number travels in the response: a variance over 5 samples is a different
# claim from one over 500, and the reader has to be able to tell.
REFERENCE_SEEDS = 5

# The most distinct instructions an instruction-swap will try. A dataset with
# forty tasks would otherwise be forty forward passes per frame; the ones
# dropped are named.
MAX_INSTRUCTIONS = 8


class NotComparable(Refusal):
    """The two sides cannot be put on one axis, and the message says why.

    Its own class because the caller's correct response differs from every
    other refusal here: not "install something" or "load something", but
    "this comparison should not be drawn at all".
    """


def _finite(values) -> list[float]:
    """Floats, with NaN and infinity refused rather than plotted.

    A NaN in an action chunk draws as a gap in a line, which reads as "the
    policy did nothing here" -- indistinguishable from a real hold.
    """
    out = []
    for v in values:
        f = float(v)
        if not math.isfinite(f):
            raise NotComparable(
                "the policy returned a non-finite action value, so there is "
                "nothing here that can be plotted against a recorded one. A "
                "NaN drawn on this chart is a gap in a line, which reads as "
                "the policy deciding to hold still."
            )
        out.append(f)
    return out


def units_agree(policy_norm: dict, dataset_stats: dict) -> tuple[bool, str]:
    """May a policy's actions be drawn on the same axis as a dataset's?

    Returns `(agree, why)`. `why` is always a sentence, including when they do
    agree, because a chart that CAN be drawn still needs to say on what basis.

    Empty on either side is a refusal, and that is the substantive rule. A
    policy that never published its action statistics and a dataset that never
    published its own are not "probably the same" -- they are two unlabelled
    axes, and the fact that both are lists of numbers of the same length is
    exactly what makes the mistake easy.
    """
    if not policy_norm:
        return False, (
            "This policy does not publish the statistics its actions are "
            "normalised against, so nothing here knows what units they are "
            "in. Drawing them over the dataset's recorded actions would be "
            "putting two different scales on one axis and letting the shape "
            "of the chart imply they match."
        )
    if not dataset_stats:
        return False, (
            "This dataset does not publish action statistics, so its recorded "
            "actions have no stated units either. Two unlabelled axes that "
            "happen to be the same length is precisely the case that looks "
            "comparable and is not."
        )

    p_dim = _width_of(policy_norm)
    d_dim = _width_of(dataset_stats)
    if p_dim and d_dim and p_dim != d_dim:
        return False, (
            f"The policy emits {p_dim} action dimensions and this dataset "
            f"recorded {d_dim}. These are different action spaces — different "
            f"robots, or the same robot described differently — and pairing "
            f"dimension 3 with dimension 3 across them would be comparing two "
            f"unrelated joints."
        )
    return True, (
        f"Both sides publish action statistics over {p_dim or d_dim} "
        f"dimensions, so the curves are on one scale. They are still a "
        f"policy's output against ONE human demonstration, not against ground "
        f"truth."
    )


def _width_of(stats: dict) -> int:
    """How many action dimensions a statistics blob describes, or 0."""
    for value in stats.values():
        if isinstance(value, dict):
            for inner in value.values():
                if isinstance(inner, (list, tuple)) and inner:
                    return len(inner)
        elif isinstance(value, (list, tuple)) and value:
            return len(value)
    return 0


@dataclass
class Divergence:
    """One frame's policy action beside the human's, and the gap between."""

    t: int
    predicted: list[float] = field(default_factory=list)
    recorded: list[float] = field(default_factory=list)
    # Per-dimension signed difference. Signed rather than absolute because
    # "the policy consistently reaches further" and "the policy is noisy" are
    # different findings and an absolute value erases the first.
    delta: list[float] = field(default_factory=list)
    # L2 over the dimensions, for ranking frames. A summary, and labelled as
    # one: it cannot say WHICH joint moved and the per-dimension list can.
    distance: float = 0.0

    def to_dict(self) -> dict:
        return {
            "t": self.t,
            "predicted": self.predicted,
            "recorded": self.recorded,
            "delta": self.delta,
            "distance": self.distance,
        }


def compare(
    *,
    frames: list[tuple[int, list[float], list[float]]],
    joint_names: list[str] | None = None,
    stride: int = 1,
    total_frames: int = 0,
    policy_repo: str = "",
    revision: str = "",
    seed: int | None = None,
    chunks: list[tuple[int, list[list[float]]]] | None = None,
) -> dict:
    """Predicted against recorded, per dimension, across an episode.

    `frames` is `(t, predicted, recorded)` already measured by the caller —
    this module does no I/O and holds no model, which is what lets it be
    tested without a policy, a GPU or a dataset.

    `chunks` is `(t, chunk)` for the SAME frames, carrying the whole action
    chunk each of those `predicted` vectors is step 0 of. It buys the
    `chunk_consistency` block and nothing else: `rows`, `bias`, `worst_frame`,
    `worst_distance` and `means` are computed from `frames` alone and are
    byte-identical with it and without it.

    Two arguments rather than one on purpose. Widening `frames` to carry the
    chunk would change the shape every existing caller and every existing test
    builds, to add a measurement that never touches the recorded action those
    tuples exist to pair against. `None` is the honest default — "nobody
    plumbed the chunks through", which the block reports as unmeasured rather
    than as a consistency of zero.
    """
    if not frames:
        raise BadRequest(
            "no frames were measured, so there is nothing to compare. This is "
            "not an empty result — it means the run did not happen."
        )

    rows: list[Divergence] = []
    for t, predicted, recorded in frames:
        pred = _finite(predicted)
        rec = _finite(recorded)
        if len(pred) != len(rec):
            raise NotComparable(
                f"at frame {t} the policy emitted {len(pred)} action "
                f"dimensions and the dataset recorded {len(rec)}. Pairing "
                f"them by position would compare unrelated joints."
            )
        delta = [p - r for p, r in zip(pred, rec, strict=True)]
        rows.append(
            Divergence(
                t=t,
                predicted=pred,
                recorded=rec,
                delta=delta,
                distance=round(math.sqrt(sum(d * d for d in delta)), 6),
            )
        )

    width = len(rows[0].predicted)
    for row in rows:
        if len(row.predicted) != width:
            # The width check inside the loop above is PER FRAME -- predicted
            # against recorded at that ONE frame -- so an episode whose frames
            # each agree internally and disagree with EACH OTHER walks past it
            # and reaches `bias` below, where `r.delta[d]` indexes off the end
            # of the narrow row and raises a bare IndexError. That is a 500
            # carrying none of the sentence, out of the module whose whole
            # argument is that a refusal is an authored sentence naming what
            # is missing. `units_agree`'s reasoning, one loop over: two widths
            # are two action spaces, and averaging a bias over a column only
            # some frames have would divide a sum of a few numbers by the
            # count of all of them.
            raise NotComparable(
                f"frame {rows[0].t} has {width} action dimensions and frame "
                f"{row.t} has {len(row.predicted)}. An episode cannot change "
                f"action space partway through, and pairing dimension 3 with "
                f"dimension 3 across the two halves would compare unrelated "
                f"joints."
            )

    names = list(joint_names or [])
    if names and len(names) != width:
        # The dataset named its dimensions and the count disagrees. Dropped
        # rather than truncated: a chart with six curves and five labels
        # mislabels at least one, and a mislabelled joint is worse than an
        # unlabelled one.
        names = []

    worst = max(rows, key=lambda r: r.distance)
    # Per-dimension mean of the SIGNED delta. Bias, not error: a policy that
    # reaches 2 cm further every single frame has a mean of 2 cm here and a
    # mean absolute error that says the same thing about a policy that is
    # randomly wrong by 2 cm in both directions.
    bias = [round(sum(r.delta[d] for r in rows) / len(rows), 6) for d in range(width)]

    if chunks is not None and [t for t, _ in chunks] != [r.t for r in rows]:
        # The server builds both lists in one loop, so this cannot happen from
        # the route. It can happen from anywhere else, and the failure is
        # invisible: the block would aggregate a different set of frames from
        # the section above it and both would render as one measurement.
        raise BadRequest(
            "the action chunks and the first-step rows describe different "
            "frames, so they did not come out of one run. This is not a "
            "partial result — the two halves of the response would be talking "
            "about different parts of the episode."
        )

    dropped = max(0, total_frames - len(rows)) if total_frames else 0
    return {
        "rows": [r.to_dict() for r in rows],
        "joint_names": names,
        "dimensions": width,
        "frames_measured": len(rows),
        "frames_in_episode": total_frames or len(rows),
        "stride": stride,
        "frames_skipped": dropped,
        "worst_frame": worst.t,
        "worst_distance": worst.distance,
        "bias": bias,
        "policy_repo": policy_repo,
        "revision": revision,
        "seed": seed,
        # A SEPARATE block, below the error section and never mixed into it.
        # Both of its operands are predictions; the recorded action is on
        # neither side, so a number from here does not belong in the same
        # column as one from above and does not belong under the same `means`.
        "chunk_consistency": _chunk_consistency(
            chunks=chunks,
            names=names,
            stride=stride,
            policy_repo=policy_repo,
            revision=revision,
            seed=seed,
        ),
        "means": _compare_means(
            n=len(rows),
            total=total_frames or len(rows),
            stride=stride,
            worst=worst,
            names=names,
            policy_repo=policy_repo,
            revision=revision,
            seed=seed,
        ),
    }


def _compare_means(
    *, n, total, stride, worst, names, policy_repo, revision, seed
) -> str:
    where = (
        f"dimension {worst.delta.index(max(worst.delta, key=abs))}"
        if not names
        else names[worst.delta.index(max(worst.delta, key=abs))]
    )
    skipped = (
        f" {total - n} frames were skipped by a stride of {stride}, so a "
        f"divergence between sampled frames is not in this chart."
        if total > n
        else ""
    )
    seeded = (
        f" Sampled at seed {seed}; another seed gives another curve."
        if seed is not None
        else " No seed was fixed, so re-running gives a different curve."
    )
    return (
        f"{n} of {total} frames, comparing {policy_repo or 'the policy'} at "
        f"revision {revision or 'unknown'} against what the demonstrator did. "
        f"The largest gap is at frame {worst.t}, mostly in {where}."
        f"{skipped}{seeded} "
        f"A RECORDED ACTION IS ONE HUMAN DEMONSTRATION, NOT GROUND TRUTH: a "
        f"policy can differ and be right, because a different grasp that also "
        f"works looks exactly like an error here. And every prediction was "
        f"made on the human's own observations, so this is open-loop teacher "
        f"forcing — error never compounds the way it would on a robot."
    )


# ----------------------------------------------------------- chunk consistency
#
# `chunk_p[dt + k]` against `chunk_q[k]`, for `k` in `range(min(H_q, H_p - dt))`
# with `dt = t_q - t_p`. Both of those elements describe the ONE absolute
# timestep `t_p + dt + k == t_q + k`, and the derivation is three lines: the
# chunk emitted at frame `t` has step `m` predicting frame `t + m` -- lerobot's
# `action_delta_indices` is `range(chunk_size)` for ACT, SmolVLA and pi0 alike,
# and the diffusion families slice their chunk "from the current observation"
# -- so `t_p + m == t_q + k` gives `m = dt + k`. All three published overlap
# detectors align their chunks exactly this way; it is the one part of this
# measurement there is a convention to follow.
#
# The tempting `chunk_p[k]` against `chunk_q[k]` compares two frames that are
# `dt` apart and reports a divergence for a policy that revised nothing.
# `chunk_p[dt + k]` against `chunk_q[k + 1]` is off by one the other way. The
# pair COUNT separates all three -- H=2 one frame apart shares exactly one
# timestep, where the first pairing finds two and the second finds none -- so
# the tests assert on the count as well as on the value.


def _consistency_fields(stride: int, means: str) -> dict:
    """The block when it could not be measured: a sentence, and no numbers.

    Every quantity is `None` or `[]` rather than 0, because 0.0 is a real and
    very different reading here -- it is what a policy that agreed with itself
    perfectly scores -- and this is the shape `scorers.Result` already uses for
    exactly that distinction, down to the sentence saying so in words.
    """
    return {
        "measurable": False,
        "means": means,
        "horizon": None,
        "horizon_min": None,
        "horizon_max": None,
        # Duplicated from the response around it, deliberately. The strip that
        # draws this has to read standalone, and a refusal naming a stride the
        # reader has to scroll for is a refusal they will not check.
        "stride": stride,
        "pairs": None,
        "overlapping_steps": None,
        "median": None,
        "p25": None,
        "p75": None,
        "worst_pair": None,
        "by_steps_ahead": [],
        "by_dimension": [],
        "pairs_skipped_same_frame": None,
    }


def _middle_half(values: list[float]) -> tuple[float, float, float]:
    """Median and the middle half, exactly, for any n.

    `statistics.quantiles` needs two points and raises below that. ONE shared
    timestep is an ordinary answer here -- a horizon of 2 at a stride of 1
    gives exactly that -- and the honest report of it is the median being the
    whole distribution with the quartiles equal to it, `n` beside them saying
    how few there were, rather than a spread nothing measured. The same shape
    and the same reason as `model_diff._quartiles`.

    Exact rather than interpolated off a histogram: `n` here is at most a few
    thousand and every value is in hand, unlike `vla_data`'s percentiles.
    """
    if not values:
        return 0.0, 0.0, 0.0
    if len(values) < 2:
        return values[0], values[0], values[0]
    cuts = statistics.quantiles(values, n=4, method="inclusive")
    return statistics.median(values), cuts[0], cuts[2]


def _chunk_consistency(*, chunks, names, stride, policy_repo, revision, seed) -> dict:
    """Does the policy still want, one observation later, what it committed to?

    `chunks` is `(t, chunk)` with `chunk` a `(H, D)` list of lists. Every pair
    of chunks that share an absolute timestep contributes one L2 distance per
    shared timestep, and nothing here reads a recorded action.

    L2 over the dimensions, unsquared, because that is this module's metric in
    four other places and the number ends up in the same response as
    `Divergence.distance`. ActProbe squares its L2 and VLA-FAIL takes a
    per-dimension L1 divided by a clamped velocity; the second needs a `v_min`
    the paper never fixes, which would be a constant invented here, and this
    module already refuses to substitute a threshold from somewhere else.

    ALL pairs, not only consecutive ones. The three papers compare adjacent
    inference steps and report one number; a curve over steps ahead is worth
    more to somebody looking at a panel, and the `means` says which part of it
    is the extension.

    Refuses by returning, never by raising. A raise here would take the
    first-step comparison down with it, and that comparison is correct on
    every input that reaches this -- including the ones this block cannot use.
    """
    if chunks is None:
        return _consistency_fields(
            stride,
            "CHUNK CONSISTENCY WAS NOT MEASURED. This comparison was given "
            "only the first step of each prediction rather than the whole "
            "action chunk, so no two predictions here describe the same "
            "absolute timestep twice and there is nothing to difference. THIS "
            "IS NOT A CONSISTENCY OF ZERO — nothing was compared at all.",
        )

    # Sorted by frame. `dt` has to be positive for the algebra to mean what it
    # says, and ascending order is not a promise the caller made: only
    # `plan_frames` guarantees it, and `compare(frames=...)` is open to anyone.
    ordered = sorted(chunks, key=lambda row: row[0])

    clean: list[tuple[int, list[list[float]]]] = []
    width = 0
    # Which chunk fixed `width`, so a later chunk that disagrees can be named
    # BESIDE it rather than against a number the sentence implies came from
    # itself. Without this the between-chunks case prints "wide at step 0 and
    # 1 at step 0" -- one frame, two widths, step 0 twice.
    width_frame = 0
    for row_index, (t, chunk) in enumerate(ordered):
        if not chunk:
            return _consistency_fields(
                stride,
                f"CHUNK CONSISTENCY WAS NOT MEASURED. The chunk at frame {t} "
                f"has no steps in it, so there is nothing there that "
                f"describes a timestep at all.",
            )
        if not width:
            width = len(chunk[0])
            width_frame = t
        if not width:
            return _consistency_fields(
                stride,
                f"CHUNK CONSISTENCY WAS NOT MEASURED. The chunk at frame {t} "
                f"is zero action dimensions wide, so the distance between two "
                f"of its steps is a sum over nothing rather than a small "
                f"number.",
            )
        steps: list[list[float]] = []
        for m, step in enumerate(chunk):
            try:
                values = [float(v) for v in step]
            except (TypeError, ValueError):
                return _consistency_fields(
                    stride,
                    f"CHUNK CONSISTENCY WAS NOT MEASURED. Step {m} of the "
                    f"chunk at frame {t} holds a value that is not a number, "
                    f"so nothing there can be subtracted from a later "
                    f"prediction of the same timestep.",
                )
            if len(values) != width:
                if m == 0 and row_index:
                    # BETWEEN two chunks, not inside one. `width` came from
                    # the FIRST chunk and the sentence below reads as though
                    # it came from this one, which on this input prints two
                    # different widths at the same step of the same frame --
                    # a refusal that contradicts itself in the clause whose
                    # job is naming what went wrong.
                    return _consistency_fields(
                        stride,
                        f"CHUNK CONSISTENCY WAS NOT MEASURED. The chunk at "
                        f"frame {width_frame} is {width} dimensions wide and "
                        f"the chunk at frame {t} is {len(values)}, so these "
                        f"are not the same action space and pairing dimension "
                        f"3 with dimension 3 across them would compare "
                        f"unrelated joints.",
                    )
                return _consistency_fields(
                    stride,
                    f"CHUNK CONSISTENCY WAS NOT MEASURED. The chunk at frame "
                    f"{t} is {width} dimensions wide at step 0 and "
                    f"{len(values)} at step {m}, so these are not the same "
                    f"action space and pairing dimension 3 with dimension 3 "
                    f"across them would compare unrelated joints.",
                )
            if any(not math.isfinite(v) for v in values):
                # NOT `_finite`, which raises. A NaN at step 7 must not take
                # down a `compare()` that succeeds today on step 0 alone. And
                # not a dropped frame either: a frame missing from an
                # aggregate is invisible, and its absence reads as agreement.
                return _consistency_fields(
                    stride,
                    f"CHUNK CONSISTENCY WAS NOT MEASURED. The policy returned "
                    f"a non-finite value at step {m} of the chunk at frame "
                    f"{t}, so the disagreement between chunks is not a number "
                    f"there. The first-step comparison above is unaffected — "
                    f"it reads only step 0 of each chunk. This block refuses "
                    f"rather than dropping that frame, because a dropped "
                    f"frame is invisible inside an aggregate and would read "
                    f"as agreement.",
                )
            steps.append(values)
        clean.append((t, steps))

    distinct = sorted({t for t, _ in clean})
    if len(distinct) < 2:
        # Three leads, because "one frame" and "one prediction" are not the
        # same claim and the middle case is the one where they come apart:
        # several chunks CAN sit on a single frame, and saying the policy made
        # one prediction there would be false in a sentence whose whole job is
        # to say what was and was not measured.
        if len(clean) > len(distinct):
            alone = (
                f"One frame was sampled and {len(clean)} chunks were "
                f"predicted on it, so they all begin at the same timestep and "
                f"none of them is a revision of another. They are not folded "
                f"in: two chunks at one frame differ by sampling noise, which "
                f"is what the instruction swap's seed reference measures and "
                f"is a different question from this one."
            )
        elif len(distinct) == 1:
            alone = (
                "One frame was sampled, so this policy made one prediction "
                "and there is no second one to disagree with it."
            )
        else:
            alone = (
                "No chunks reached this block, so there is no prediction here "
                "for another to disagree with."
            )
        return _consistency_fields(
            stride,
            f"CHUNK CONSISTENCY WAS NOT MEASURED. {alone} THIS IS NOT A "
            f"CONSISTENCY OF ZERO. Two or more sampled frames, less than a "
            f"horizon apart, are what this measurement needs.",
        )

    horizons = [len(chunk) for _, chunk in clean]
    lo, hi = min(horizons), max(horizons)
    gap = min(b - a for a, b in zip(distinct[:-1], distinct[1:], strict=True))

    distances: list[float] = []
    per_dt: dict[int, list[float]] = {}
    per_dt_pairs: dict[int, int] = {}
    per_dim: list[list[float]] = [[] for _ in range(width)]
    pairs = 0
    same_frame = 0
    worst: tuple[int, int, int, int, float] | None = None
    worst_delta: list[float] = []

    for i, (t_p, chunk_p) in enumerate(clean):
        for t_q, chunk_q in clean[i + 1 :]:
            dt = t_q - t_p
            if dt == 0:
                # Two chunks at ONE frame differ by sampling noise, and that
                # is the instruction swap's reference measurement rather than
                # this one. Counted rather than silently dropped.
                same_frame += 1
                continue
            overlap = min(len(chunk_q), len(chunk_p) - dt)
            if overlap <= 0:
                continue
            pairs += 1
            per_dt_pairs[dt] = per_dt_pairs.get(dt, 0) + 1
            for k in range(overlap):
                earlier = chunk_p[dt + k]
                later = chunk_q[k]
                # LATER minus EARLIER, and signed, so "this policy revises
                # this joint upward as it gets closer" survives the
                # aggregation. The same argument `Divergence.delta` makes.
                delta = [b - a for a, b in zip(earlier, later, strict=True)]
                distance = math.sqrt(sum(d * d for d in delta))
                distances.append(distance)
                per_dt.setdefault(dt, []).append(distance)
                for c in range(width):
                    per_dim[c].append(delta[c])
                if worst is None or distance > worst[4]:
                    worst = (t_p, t_q, dt, k, distance)
                    worst_delta = delta

    if not distances:
        if hi <= 1:
            return _consistency_fields(
                stride,
                "CHUNK CONSISTENCY WAS NOT MEASURED. Every chunk this policy "
                "returned is ONE step long, so it predicted nothing beyond "
                "the frame it was asked about and there is no future for a "
                "later chunk to disagree about. THIS IS NOT A CONSISTENCY OF "
                "ZERO. This measurement needs a horizon of at least 2 and a "
                "stride below it; a one-step chunk comes from a policy queried "
                "a single step at a time, and no stride makes it overlap.",
            )
        # `gap` and `hi` are two numbers the reader will do arithmetic on, and
        # that arithmetic only closes when the horizons agree. A pair overlaps
        # when `H_p > dt`: `min(H_q, H_p - dt)` is zero the moment the EARLIER
        # chunk is no longer than the gap, however long the later one is. So
        # the stride that overlaps every consecutive pair is set by the
        # SHORTEST chunk, `lo - 1`, and `hi - 1` is only the ceiling above
        # which nothing could overlap at all. They are the same number under
        # one horizon, which is exactly what made reading both off `hi` look
        # right -- and under ragged horizons it printed a cause its own two
        # numbers refute ("1 frame apart, 3-step chunks, so nothing overlaps")
        # and then recommended a stride the run had just used.
        apart = f"{gap} frame{'' if gap == 1 else 's'} apart"
        if lo == hi:
            return _consistency_fields(
                stride,
                f"CHUNK CONSISTENCY WAS NOT MEASURED. The closest two sampled "
                f"frames here are {apart} and every chunk this policy returned "
                f"is {hi} steps, so no two chunks ever describe the same "
                # "1 frames later" is the "every 2th eligible row" typo
                # `fmt.ordinal` exists to stop, one sentence over: a stride of
                # H-1 is the case this refusal is most often read in, and
                # hi-1 is 1 exactly there.
                f"absolute timestep: a chunk predicted at one frame ends "
                f"{hi - 1} frame{'' if hi == 2 else 's'} later, before the "
                f"next sampled frame begins. "
                f"THIS IS NOT A CONSISTENCY OF ZERO, which is what a policy "
                f"that agreed with itself perfectly would score. A stride of "
                f"{hi - 1} or less overlaps — at {hi - 1} each consecutive "
                f"pair shares one timestep, and the overlap grows by one for "
                f"every frame the stride comes down. This run was planned at "
                f"a stride of {stride}.",
            )
        # Ragged horizons. The remedy is `lo - 1`, and when `lo` is 1 there is
        # no stride at all that brings THAT chunk in -- while the long ones
        # can still be reached, so "no stride works" would be false as well.
        remedy = (
            f"A stride of {lo - 1} or less overlaps every consecutive pair: "
            f"the SHORTEST chunk sets that number, because the earlier chunk "
            f"of a pair is the one that has to reach the later frame, and a "
            f"stride of {hi - 1} is only the ceiling above which nothing here "
            f"could overlap at all."
            if lo > 1
            else (
                f"The shortest chunk here is ONE step, and a one-step chunk "
                f"predicts nothing past its own frame, so no stride makes "
                f"THAT one overlap. The earlier chunk of a pair is the one "
                f"that has to reach the later frame, so a stride of {hi - 1} "
                f"or less is what the {hi}-step chunks would need, and the "
                f"one-step ones stay out of this measurement at any stride."
            )
        )
        return _consistency_fields(
            stride,
            f"CHUNK CONSISTENCY WAS NOT MEASURED. The chunks were not all the "
            f"same length — they ran from {lo} to {hi} steps — and the "
            f"closest two sampled frames here are {apart}. Every one of them "
            f"ended before the next frame it was compared against: a chunk of "
            f"n steps predicted at one frame describes frames up to n-1 "
            f"later, and in each pair it is the EARLIER chunk that has to "
            f"reach the later frame, whatever horizon the later one has. THIS "
            f"IS NOT A CONSISTENCY OF ZERO, which is what a policy that "
            f"agreed with itself perfectly would score. {remedy} This run was "
            f"planned at a stride of {stride}.",
        )

    median, p25, p75 = _middle_half(distances)
    horizon = lo if lo == hi else None
    labelled = names if len(names) == width else []

    by_steps_ahead = []
    for dt in sorted(per_dt):
        step_median, step_p25, step_p75 = _middle_half(per_dt[dt])
        by_steps_ahead.append(
            {
                "steps_ahead": dt,
                "pairs": per_dt_pairs[dt],
                # `n` travels with every row. A median over one shared
                # timestep and a median over eighty are different claims.
                "overlapping_steps": len(per_dt[dt]),
                "median": fmt.measured_value(step_median, 6),
                "p25": fmt.measured_value(step_p25, 6),
                "p75": fmt.measured_value(step_p75, 6),
            }
        )

    by_dimension = []
    for c in range(width):
        signed = per_dim[c]
        by_dimension.append(
            {
                "dimension": c,
                "name": labelled[c] if labelled else None,
                # The pair `compare` already draws: a signed mean, which says
                # WHICH WAY the policy keeps revising, beside a median
                # absolute, which says how far it moves either way.
                "revision_bias": fmt.measured_value(sum(signed) / len(signed), 6),
                "disagreement": fmt.measured_value(
                    statistics.median([abs(v) for v in signed]), 6
                ),
            }
        )

    return {
        "measurable": True,
        "means": _consistency_means(
            n=len(distances),
            pairs=pairs,
            stride=stride,
            gap=gap,
            horizon=horizon,
            lo=lo,
            hi=hi,
            median=median,
            p25=p25,
            p75=p75,
            worst=worst,
            worst_delta=worst_delta,
            names=labelled,
            policy_repo=policy_repo,
            revision=revision,
            seed=seed,
            same_frame=same_frame,
            widest_dt=max(per_dt),
        ),
        "horizon": horizon,
        "horizon_min": lo,
        "horizon_max": hi,
        "stride": stride,
        "pairs": pairs,
        "overlapping_steps": len(distances),
        # `measured_value`, not `round`. A perfectly self-consistent policy
        # scores exactly 0.0 and a near-consistent one scores 3e-07, and
        # `round(x, 6)` prints the second as `0.0` -- the first policy's
        # reading, on the second policy.
        "median": fmt.measured_value(median, 6),
        "p25": fmt.measured_value(p25, 6),
        "p75": fmt.measured_value(p75, 6),
        "worst_pair": {
            "t_earlier": worst[0],
            "t_later": worst[1],
            "steps_ahead": worst[2],
            "step": worst[3],
            "distance": fmt.measured_value(worst[4], 6),
        },
        "by_steps_ahead": by_steps_ahead,
        "by_dimension": by_dimension,
        "pairs_skipped_same_frame": same_frame,
    }


def _consistency_means(
    *,
    n,
    pairs,
    stride,
    gap,
    horizon,
    lo,
    hi,
    median,
    p25,
    p75,
    worst,
    worst_delta,
    names,
    policy_repo,
    revision,
    seed,
    same_frame,
    widest_dt,
) -> str:
    where = (
        f"dimension {worst_delta.index(max(worst_delta, key=abs))}"
        if not names
        else names[worst_delta.index(max(worst_delta, key=abs))]
    )
    horizon_phrase = (
        f"a {horizon}-step horizon"
        if horizon is not None
        else f"horizons that VARIED between {lo} and {hi} steps"
    )
    seeded = (
        f"Sampled at seed {seed}; another seed gives another set of chunks."
        if seed is not None
        else "No seed was fixed, so re-running gives different chunks."
    )
    skipped = (
        f" {same_frame} pair(s) of chunks predicted at the identical frame "
        f"were skipped: two chunks at one frame differ by sampling noise, "
        f"which is what the instruction swap's seed reference measures, and "
        f"folding it in would quietly turn this into a different metric."
        if same_frame
        else ""
    )
    beyond = (
        # `{gap} steps` reads "1 steps" on the ordinary run, where the stride
        # is 1 and every row past the first is the extension being named.
        f" Rows past {gap} step{'' if gap == 1 else 's'} ahead compare "
        f"NON-ADJACENT chunks: the "
        f"published detectors compare consecutive inference steps only, so "
        f"that part of the curve is this project's extension of them."
        if widest_dt > gap
        else ""
    )
    return (
        # `fmt.measured`, not a format string. This quantity is exactly 0.0
        # for a policy that never revises anything and 3e-07 for one that
        # barely does, and `{:,.4f}` prints the second as the first -- the
        # failure `fmt.py` exists to record, on the one number in this block
        # a reader is most likely to act on.
        f"{policy_repo or 'This policy'}'s chunks disagree with each other by "
        f"a median of {fmt.measured(median, 4)} over {n} shared timestep(s), "
        f"from {pairs} chunk pair(s) at a stride of {stride} and "
        f"{horizon_phrase}. The middle half runs {fmt.measured(p25, 4)} to "
        f"{fmt.measured(p75, 4)}, and the widest single disagreement is "
        f"{fmt.measured(worst[4], 4)} — between the chunk from frame "
        f"{worst[0]} and the chunk from frame {worst[1]}, about frame "
        f"{worst[1] + worst[3]}, mostly in {where}. {seeded}{skipped}{beyond} "
        f"THIS COMPARES {policy_repo or 'the policy'} AT REVISION "
        f"{revision or 'unknown'} AGAINST ITSELF, never against the "
        f"demonstrator: no recorded action is on either side of any "
        f"subtraction here, and a policy that predicts the same wrong action "
        f"every single time is perfectly consistent by this measure. The "
        f"numbers are in the policy's own action units, the ones its chunks "
        f"come out in, and every chunk was produced open-loop on the human's "
        f"observations — a robot running this policy would visit different "
        f"states and produce a different sequence of chunks. A chunk step is "
        f"assumed to be one dataset frame, which holds only if this policy "
        f"was trained at this dataset's frame rate, and nothing here can "
        f"check that. The median over the overlap and the breakdown by steps "
        f"ahead are this project's own choices rather than a published "
        f"convention: the detectors this follows mean, sum or exponentially "
        f"average a single adjacent-step number and none of them publishes a "
        f"curve. NO THRESHOLD IS APPLIED — this is a measured divergence, not "
        f"a failure prediction, and turning it into one needs a calibration "
        f"nothing here has made. It needs no sampling reference, so a "
        f"deterministic policy is measured by it rather than refused."
    )


def instruction_swap(
    *,
    own_instruction: str,
    swapped: list[tuple[str, list[float]]],
    seed_samples: list[list[float]],
    policy_repo: str = "",
    dropped_instructions: int = 0,
) -> dict:
    """Does the instruction move the action more than the sampler does?

    `swapped` is `(instruction, action)` for the policy's own task string and
    every distinct other one; `seed_samples` is the action at several seeds on
    the IDENTICAL frame with the OWN instruction.

    The reference is the policy's own sampling spread. Nothing here is
    compared against a threshold from a paper about a different policy.
    """
    if len(swapped) < 2:
        raise NotComparable(
            "this dataset contains one distinct task string, so there is "
            "nothing to swap the instruction FOR. A distractor instruction "
            "made up here would not be a measurement of this policy, it "
            "would be a measurement of a sentence somebody invented. Most "
            "hobbyist recordings are single-task; this is expected rather "
            "than a fault."
        )
    if len(seed_samples) < 2:
        raise NotComparable(
            "fewer than two samples were drawn, so there is no sampling "
            "spread to compare the instruction against."
        )

    width = len(seed_samples[0])
    for sample in seed_samples:
        if len(sample) != width:
            raise NotComparable(
                "the sampled actions are not all the same width, so their "
                "spread is not a number."
            )

    seed_spread = _spread([_finite(s) for s in seed_samples])
    if seed_spread == 0.0:
        raise NotComparable(
            "re-rolling the sampler on the identical frame gave the identical "
            "action every time, so this policy is deterministic and its own "
            "sampling spread is exactly zero. The instruction effect is "
            "measured AGAINST that spread, and a ratio against zero is not a "
            "number — this refuses rather than reporting an infinity or "
            "silently substituting a threshold from somewhere else."
        )

    own = next((a for i, a in swapped if i == own_instruction), None)
    if own is None:
        raise BadRequest(
            "the policy's own instruction is not among the swapped runs, so "
            "there is no baseline to measure the others against."
        )
    own = _finite(own)

    arms = []
    for instruction, action in swapped:
        values = _finite(action)
        if len(values) != len(own):
            raise NotComparable(
                "two instructions produced action vectors of different "
                "widths, which means they are not the same measurement."
            )
        distance = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(values, own, strict=True))
        )
        arms.append(
            {
                # Empty is "no instruction", a CONDITION that #50 runs on
                # purpose — never "the instruction did not matter".
                "instruction": instruction or "(no instruction)",
                "is_own": instruction == own_instruction,
                "action": values,
                "distance_from_own": round(distance, 6),
                "ratio_to_sampling": round(distance / seed_spread, 4),
            }
        )

    instruction_spread = _spread([_finite(a) for _, a in swapped])
    ratio = instruction_spread / seed_spread
    return {
        "arms": arms,
        "instruction_spread": round(instruction_spread, 6),
        "sampling_spread": round(seed_spread, 6),
        "ratio": round(ratio, 4),
        "listens": bool(ratio > 1.0),
        "seeds": len(seed_samples),
        "instructions_tried": len(swapped),
        "instructions_dropped": dropped_instructions,
        "means": _swap_means(
            ratio=ratio,
            instruction_spread=instruction_spread,
            seed_spread=seed_spread,
            n_instructions=len(swapped),
            n_seeds=len(seed_samples),
            dropped=dropped_instructions,
            policy_repo=policy_repo,
        ),
    }


def _swap_means(
    *,
    ratio,
    instruction_spread,
    seed_spread,
    n_instructions,
    n_seeds,
    dropped,
    policy_repo,
) -> str:
    verdict = (
        f"Swapping the instruction moves this policy's action {ratio:,.2f}x "
        f"as much as re-rolling the sampler does, so on this frame the "
        f"instruction is doing more than noise."
        if ratio > 1.0
        else (
            f"SWAPPING THE INSTRUCTION MOVES THE ACTION LESS THAN RE-ROLLING "
            f"THE SAMPLER DOES ({ratio:,.2f}x). On this frame, changing what "
            f"the policy was told mattered less than changing the random seed."
        )
    )
    lost = (
        f" {dropped} further distinct instructions in this dataset were not "
        f"tried, so the spread across instructions is a lower bound."
        if dropped
        else ""
    )
    return (
        # `seed_spread` is the DENOMINATOR of the whole comparison, and ninety
        # lines above this the code refuses to divide by it when it is exactly
        # zero — "a ratio against zero is not a number". A near-deterministic
        # policy spreading 3e-05 then had that same denominator printed as
        # "0.0000": the reader is shown the one value the function just
        # refused to accept, beside a ratio computed from it.
        f"{verdict} Measured on ONE frame: {n_instructions} distinct "
        f"instructions spread {fmt.measured(instruction_spread, 4)}, against "
        f"{n_seeds} seeds spreading {fmt.measured(seed_spread, 4)}, on "
        f"{policy_repo or 'this policy'}.{lost} The reference is this "
        f"policy's own sampling variance rather than a threshold from "
        f"anywhere else, so the comparison holds for this policy and is not "
        f"transferable to another. One frame is a sample, not a property of "
        f"the policy."
    )


def knockout(
    *,
    baseline: list[float],
    arms: list[tuple[str, str, list[float]]],
    policy_repo: str = "",
    sampling_spread: float | None = None,
) -> dict:
    """How far does the action move when ONE input is replaced by its mean?

    `arms` is `(stream_id, label, action)`. The baseline is the action with
    every input intact.
    """
    if not arms:
        raise BadRequest(
            "no input streams were knocked out, so there is nothing to "
            "report. A policy consuming no inputs is not a policy."
        )
    base = _finite(baseline)

    rows = []
    for stream, label, action in arms:
        values = _finite(action)
        if len(values) != len(base):
            raise NotComparable(
                f"knocking out {stream} produced {len(values)} action "
                f"dimensions against the baseline's {len(base)}, so the two "
                f"cannot be differenced."
            )
        distance = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(values, base, strict=True))
        )
        rows.append(
            {
                "stream": stream,
                "label": label,
                "action": values,
                "distance": round(distance, 6),
                # Against the policy's own sampling spread where it is known.
                # None where it is not -- a bar whose height cannot be
                # compared to noise is still a real measurement, and inventing
                # a denominator would be worse than leaving it out.
                "ratio_to_sampling": (
                    round(distance / sampling_spread, 4) if sampling_spread else None
                ),
                "above_noise": (
                    bool(distance > sampling_spread) if sampling_spread else None
                ),
            }
        )
    rows.sort(key=lambda r: r["distance"], reverse=True)

    total = sum(r["distance"] for r in rows)
    return {
        "rows": rows,
        "baseline": base,
        "streams": len(rows),
        "sampling_spread": sampling_spread,
        "means": (
            f"Each bar is how far {policy_repo or 'the policy'}'s action moved "
            f"when that ONE input was replaced by its episode mean, on this "
            f"frame. {rows[0]['label']} moved it furthest "
            f"({fmt.measured(rows[0]['distance'], 4)}).\n\n"
            f"MEAN SUBSTITUTION IS A SPECIFIC BASELINE, NOT REMOVAL. The mean "
            f"frame of an episode is a real image the policy has opinions "
            f"about; it is not absence, and a different baseline gives "
            f"different bars.\n\n"
            f"THESE DO NOT ADD UP. The bars sum to {fmt.measured(total, 4)}, "
            f"and that "
            f"number means nothing: the policy's inputs interact, so removing "
            f"two streams is not the sum of removing each. Read them as "
            f"separate one-at-a-time measurements."
            + (
                ""
                if sampling_spread
                else "\n\nNo sampling spread was measured, so nothing here "
                "says whether a bar is larger than this policy's own noise."
            )
        ),
    }


def _spread(vectors: list[list[float]]) -> float:
    """Mean distance from the centroid — the spread of a set of actions.

    Mean rather than max: one outlying sample should widen the reference, not
    define it. Zero for a single vector, and every caller treats zero as a
    refusal rather than as a very small number.
    """
    if len(vectors) < 2:
        return 0.0
    width = len(vectors[0])
    centre = [sum(v[d] for v in vectors) / len(vectors) for d in range(width)]
    return sum(
        math.sqrt(sum((v[d] - centre[d]) ** 2 for d in range(width))) for v in vectors
    ) / len(vectors)


def plan_frames(length: int, *, stride: int = 0) -> tuple[list[int], int]:
    """Which frames to measure, and the stride that got there.

    Strided rather than truncated. Measuring the first 64 frames of a
    200-frame episode answers a question about the beginning of the episode
    and labels it as the episode; an even stride at least samples the whole
    thing, and `frames_skipped` says what is missing either way.
    """
    if length <= 0:
        raise BadRequest("this episode has no frames.")
    # A NEGATIVE stride is refused rather than clamped, and the reason is what
    # the caller does with the answer. MEASURED on a 161-frame episode:
    # `stride=0` priced 54 forward passes; `stride=-1` ran 161 and reported
    # `stride: 1` — 2.98x the cost this route exists to quote, with the
    # response naming a stride nobody asked for. `-1` is truthy, so it took
    # the "the caller set one" branch and `max(1, -1)` quietly made it a 1.
    #
    # Bool first, because `isinstance(True, int)` is True and `stride=True`
    # came out of the same branch as a deliberate 1.
    if isinstance(stride, bool) or not isinstance(stride, int):
        raise BadRequest(
            f"a stride of {stride!r} is not a number of frames. Send 0 to let "
            f"this choose one that fits the work budget, or a whole number of "
            f"frames to set it yourself."
        )
    if stride < 0:
        raise BadRequest(
            f"a stride of {stride} would step backwards through the episode. "
            f"Send 0 to let this choose one that fits the work budget, or a "
            f"positive number of frames to set it yourself."
        )
    chosen = stride if stride else max(1, math.ceil(length / MAX_FRAMES_PER_RUN))
    return list(range(0, length, chosen)), chosen
