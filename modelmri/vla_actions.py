"""What a policy would DO, next to what a human actually did.

Three measurements, and all three are shaped by the same worry: an action
curve is the most persuasive-looking thing this whole project draws. Two lines
on one axis, one labelled "policy" and one "recorded", read as a score — and
a reader will conclude something about a robot from the gap between them.

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
) -> dict:
    """Predicted against recorded, per dimension, across an episode.

    `frames` is `(t, predicted, recorded)` already measured by the caller —
    this module does no I/O and holds no model, which is what lets it be
    tested without a policy, a GPU or a dataset.
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
