# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""When the denoiser commits, and what the steps after that bought.

Somebody asks for 50 steps because a slider said 50. Somewhere in that run the
latent stops moving materially and every step after it buys a texture nobody
can see — and nothing local tells them where that step was, so they keep paying
for the whole slider on every image.

This measures it. Per denoising step: how far the latent moved from where it
was at the end of the previous step, and what share of the whole run's movement
had happened by then. `imaging.py` already lists `step_commit` and
`latent_trace` as capabilities of both diffusion families; this is them.

The capture follows `image_attention.capture` exactly — one real pipeline call,
`callback_on_step_end`, bounded memory, released in a `finally`-shaped handler —
because that module already solved the same problem and two different shapes for
one job is one shape too many.

## Latents, not decoded pixels

The obvious version decodes every step through the VAE and compares images. It
is the expensive wrong thing twice over.

Expensive: a decode is a full pass through the decoder, once per step, on top of
the denoising this is supposed to be measuring. Fifty extra decodes to answer a
question about the UNet.

Wrong: the answer would then be a property of the VAE as much as of the
denoiser, so the same denoiser behind two different decoders would appear to
commit at two different steps.

So nothing here is ever decoded, and the run is asked for `output_type="latent"`
so the pipeline skips even its own final decode. The response carries
`vae_decodes: 0` — a checkable claim rather than a promise.

The cost of that choice ships in the response rather than sitting here: LATENT
DISTANCE IS NOT VISIBLE DIFFERENCE. The decoder is non-linear, so a small late
change in the latent can still be a visible change in texture, and a large early
one can vanish. This says when the LATENT stopped moving. That is a different
sentence from when the picture stopped changing, and only one of them was
measured.

## Watching it form: a separate, opt-in companion

The section above argues against decoding inside the MEASUREMENT, and it stands
unchanged. It was never an argument against ever looking: "show me the picture
appearing" is a different and perfectly good question, and `filmstrip()` is the
function that answers it.

A separate function returning a separate object, never a flag on `trace()`.
`vae_decodes: 0` up there is checkable by counting calls to the VAE, and a flag
that could turn that 0 into a 12 would make the claim conditional on an argument
nobody reads in the response. So the trace still decodes nothing, and the strip
holds itself to the same standard from the other end: it wraps `pipe.vae.decode`
for the duration of the call and reports what it was actually called, rather
than what it intended.

The strip decodes a SUBSET the caller names — `every=N`, or `at=[...]` — and
never the whole run, because eight decoded frames fit beside the pipeline on an
8 GB card and fifty do not. Which steps were decoded, and which ones ran and
were never looked at, travel in every response: a strip of 8 frames is a picture
of a 50-step run only while it says which 8.

## The step axis is kept

Never collapsed to a single number. A commit step without the curve underneath
it is a claim nobody can check, and the curve is the part that shows whether the
movement tailed off smoothly or fell off a cliff at one step — two very
different runs with the same commit step.

## The threshold is an argument, not a magic constant

"Committed at step 19" is meaningless. "Committed at step 19 of 30, at the 95%
threshold" is a measurement. So the threshold is a parameter with a stated
default of 0.95, it travels in every response beside the step it produced, and
the step at three other thresholds is reported next to it so the number is never
read alone.

The default is a CONVENTION, not a finding. Nothing calibrated 0.95, no paper is
being cited for it, and this project does not borrow a threshold from a paper
about a different model — the same rule `vla_actions.instruction_swap` follows
when it uses a policy's own sampling spread as its reference.

## What step 0 cannot say

diffusers hands the latent to `callback_on_step_end` at the END of a step, so
the noise the run started from is never seen here. The movement during step 0 is
therefore unmeasured, and it travels as `None` rather than as `0.0`: zero would
claim that the largest single move most runs make never happened.

Recovering the starting latent would mean assuming how this pipeline scales its
initial noise — `init_noise_sigma` for Stable Diffusion, something else for a
packed-latent transformer — and `imaging.py` sets the rule that nothing here
assumes the model. An honest gap, stated in the response, beats a number that is
right for one family.

## Why too many steps is REFUSED here and merely capped in image_attention

`image_attention.capture` caps at 50 steps and reports what it dropped, and that
is correct there: the maps from the first 50 steps of a 100-step run are still
real maps of that run.

Truncating here would be a lie. A scheduler asked for 50 steps places them at
different timesteps from one asked for 100, so running 50 of a requested 100
does not measure the first half of the 100-step run — it measures a different
run and labels it with the number the caller asked for. So a step count past the
ceiling is refused by name, with the ceiling in the sentence.

## Memory

One latent per step, on the CPU, in float32. An SD 512x512 latent is
(4, 64, 64) — 64 KiB — so a 50-step trace holds 3.2 MiB, and that is the design
point rather than the limit.

The limit is `MAX_TRACE_BYTES` and it exists for the shape this was not written
for: a latent with a time axis, say (16, 13, 60, 104), is 4.95 MiB a step and
248 MiB over 50 steps, which all but fills the budget — a longer clip or a
higher resolution does not fit. Past the limit this refuses BEFORE the second
step, with the arithmetic and the two parameters that would fix it, and `plan()`
prices the same thing without running anything at all.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field

from . import fmt
from .errors import BadRequest, Refusal

# The share of a run's measured movement that counts as "committed". A
# convention with a stated default, never a magic number: it is an argument to
# every function that uses it and it travels in every response beside the step
# it produced. Nothing calibrated it — see the module docstring.
DEFAULT_COMMIT_THRESHOLD = 0.95

# Reported beside whichever threshold the caller chose, so one number is never
# read alone. Spread deliberately: 0.5 is where half the movement is done, 0.99
# is where the run is arguably finished, and the distance between those two step
# numbers is the actual finding.
REPORTED_THRESHOLDS = (0.5, 0.9, 0.95, 0.99)

# Below this there is nothing to find. Two steps yield exactly ONE measured
# change, so every threshold returns step 1 and the answer looks like a
# measurement without being one. Distilled few-step models (LCM, Turbo) do run
# at one or two steps; for those the honest answer is that there is no commit
# step to find, and the refusal says so rather than reporting a placeholder.
MIN_STEPS = 3

# The most steps this will run in one trace. A REFUSAL rather than a cap — see
# the module docstring for why truncating a schedule is a different run rather
# than a shorter one. Set well past any real slider so it bites on typos and
# runaway loops rather than on work somebody meant.
MAX_STEPS = 200

# The most host RAM one trace's held latents may occupy. Checked against the
# real latent at the first step, so a run that cannot be held is refused after
# one step rather than after forty-nine.
MAX_TRACE_BYTES = 256 * 1024 * 1024

# Every held latent is cast to float32, so this is what a value costs. Stated as
# a constant because `plan()` prices a run on a machine that has not loaded a
# model and must use the same arithmetic the run will.
BYTES_PER_VALUE = 4


class NotSupported(Refusal):
    """This pipeline does not expose what the measurement needs.

    Its own class because the caller's next move is to point at a different
    pipeline, not to change a parameter of this call — which is exactly the
    distinction `errors.py` draws between `Refusal` and `BadRequest`.
    """


class NotMeasurable(Refusal):
    """The run happened and there is still no number in it.

    Separate from `NotSupported` because the pipeline was fine: the trajectory
    itself has no measurable movement, or it diverged into non-finite values.
    Nothing the caller changes about the pipeline fixes either, and reporting a
    zero or a NaN would draw a flat line that reads as "the model committed
    immediately".
    """


@dataclass
class StepChange:
    """One denoising step: where the latent went, and how far it still had."""

    step: int
    # The scheduler's timestep. Carried for the same reason `image_attention`
    # carries it: "step 12" means nothing across schedulers that place their
    # steps differently, and the timestep is where the step actually was.
    timestep: float | None = None
    # RMS distance from the latent at the end of the previous step — the
    # movement DURING this step. `None` at the first captured step, where the
    # run's starting noise was never observed. None and 0.0 are different
    # answers and this is the case that makes the difference matter.
    rms_change: float | None = None
    # Running share of the run's total MEASURED movement. `None` wherever
    # `rms_change` is None, for the same reason.
    cumulative: float | None = None
    # RMS distance from this latent to the final one. The honest counterweight
    # to `cumulative`: the fractions are path length, and a trajectory that
    # wanders can be 95% of the way along its path and still a long way from
    # where it ends.
    rms_to_final: float | None = None
    # RMS of the latent itself, so a change of 0.4 can be read against the thing
    # it changed. A distance with no scale beside it is not a size.
    latent_rms: float | None = None

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "timestep": self.timestep,
            "rms_change": self.rms_change,
            "cumulative": self.cumulative,
            "rms_to_final": self.rms_to_final,
            "latent_rms": self.latent_rms,
        }


@dataclass
class LatentTrace:
    """Every step's movement, and everything one run of it cannot claim."""

    prompt: str = ""
    seed: int | None = None
    model: str = ""
    # The scheduler class. It decides WHERE the steps are placed, so it changes
    # the commit step more than most things a user would think to vary — which
    # is why it is a field rather than a footnote.
    scheduler: str = ""
    #: "callback" or "hook". NOT cosmetic: a callback latent is what the
    #: pipeline carried at the END of a step, and a hooked one is what went
    #: IN to the denoiser with the scheduler's scaling applied. Each is
    #: internally consistent; they must never be compared against each other,
    #: so every response says which produced it.
    captured_by: str = "callback"
    steps: list[StepChange] = field(default_factory=list)
    steps_requested: int = 0
    # Without the batch axis. Reported so the memory claim is checkable against
    # the run that made it.
    latent_shape: tuple[int, ...] = ()
    threshold: float = DEFAULT_COMMIT_THRESHOLD
    # The denominator every fraction here was divided by. Published because a
    # fraction whose denominator is hidden is not a number anybody can check.
    total_change: float = 0.0
    vae_decodes: int = 0
    bytes_held: int = 0

    def commit_at(self, threshold: float) -> int | None:
        """The first step by which `threshold` of the movement had happened."""
        return commit_step([s.cumulative for s in self.steps], threshold)

    def commits(self) -> list[dict]:
        """The commit step at several thresholds, so none is read alone."""
        return [
            {"threshold": t, "step": self.commit_at(t)} for t in REPORTED_THRESHOLDS
        ]

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "seed": self.seed,
            "model": self.model,
            "scheduler": self.scheduler,
            "captured_by": self.captured_by,
            "steps": [s.to_dict() for s in self.steps],
            "steps_requested": self.steps_requested,
            "steps_measured": len(self.steps),
            "latent_shape": list(self.latent_shape),
            "threshold": self.threshold,
            "commit_step": self.commit_at(self.threshold),
            "commits": self.commits(),
            "total_change": self.total_change,
            "vae_decodes": self.vae_decodes,
            "bytes_held": self.bytes_held,
            "means": self.means(),
        }

    def means(self) -> str:
        n = len(self.steps)
        committed = self.commit_at(self.threshold)
        if committed is None:
            headline = (
                f"No step in this run reached the {_pct(self.threshold)} "
                f"threshold, so this run has no commit step to report at it."
            )
            distance = ""
        else:
            after = n - 1 - committed
            headline = (
                f"At the {_pct(self.threshold)} threshold this run committed at "
                f"step {committed} of {n}: {_pct(self.threshold)} of the "
                f"movement this could measure had happened by the end of that "
                f"step, and the {after} step(s) after it did the rest."
            )
            at_commit = self.steps[committed].rms_to_final
            started = next(
                (s.rms_to_final for s in self.steps if s.rms_to_final is not None),
                None,
            )
            distance = (
                f" At that step the latent was still "
                f"{fmt.measured(at_commit, 4)} RMS from where it finished, "
                f"against {fmt.measured(started, 4)} at the first step "
                f"measured."
                if at_commit is not None and started is not None
                else ""
            )

        # WHICH QUANTITY. A hooked run measures the denoiser's input rather
        # than the latent the pipeline carried out of a step, and a reader
        # comparing a hooked commit step against a callback one would be
        # comparing two different things that share a name.
        route = (
            ""
            if self.captured_by != "hook"
            else (
                f" This pipeline does not offer `callback_on_step_end`, so the "
                f"trajectory was read with a forward hook on the denoiser: "
                f"what is plotted is {HOOK_QUANTITY}, not the latent the "
                f"pipeline carried out of each step. The two differ by the "
                f"scheduler's own scaling, so this curve is comparable step to "
                f"step and against another hooked run — and NOT against a run "
                f"of a pipeline that does offer the callback."
            )
        )

        seeded = (
            f"Seed {self.seed}."
            if self.seed is not None
            else (
                "NO SEED WAS FIXED, so this trajectory cannot be reproduced and "
                "the commit step cannot be checked or compared against another "
                "run — the next run starts from different noise and commits "
                "somewhere else."
            )
        )

        return (
            f"Per-step RMS movement of the latent across {n} denoising steps of "
            f"{self.model or 'this model'} on {self.scheduler or 'its scheduler'}"
            f", at the prompt given. {seeded}{route}\n\n"
            f"{headline}{distance} The step at other thresholds: "
            f"{_commit_line(self.commits())}.\n\n"
            f"THE THRESHOLD IS A CONVENTION, NOT A FINDING. Nothing calibrated "
            f"{_pct(self.threshold)} and no paper is being cited for it, so "
            f"'committed at step {committed}' is only a measurement while the "
            f"threshold is attached to it. A different threshold gives a "
            f"different step, which is what the line above is for.\n\n"
            f"STEP 0 CARRIES NO CHANGE. The pipeline hands its latent over at "
            f"the END of each step, so the noise this run started from was "
            f"never seen and the movement during step 0 is unmeasured — "
            f"reported as null rather than as zero, because zero would claim "
            f"the largest move of the run never happened. Every fraction here "
            f"is a share of the {self.total_change:,.4f} of movement from step "
            f"1 onward.\n\n"
            f"THESE ARE PATH LENGTHS, NOT DISTANCES. The fractions add up "
            f"step-to-step distances, and the latent does not travel in a "
            f"straight line, so {_pct(self.threshold)} of the movement is not "
            f"{_pct(self.threshold)} of the way to the final latent. "
            f"`rms_to_final` is the column that answers that.\n\n"
            f"LATENT DISTANCE IS NOT VISIBLE DIFFERENCE. Nothing here was "
            f"decoded — {self.vae_decodes} VAE decodes for the whole run — "
            f"because decoding every step would cost a full pass through the "
            f"decoder per step and would make the answer a property of the "
            f"decoder too. So this says when the latent stopped moving, which "
            f"is not the same sentence as when the picture stopped changing.\n\n"
            f"ONE TRAJECTORY, NOT A PROPERTY OF THE MODEL. One prompt, one "
            f"seed, one scheduler and one step count. The same model on another "
            f"scheduler redistributes the movement entirely, because a "
            f"scheduler is a decision about where to put the steps. A number "
            f"measured once is a sample, not a property."
        )


def _pct(value: float) -> str:
    """0.95 -> '95%', 0.995 -> '99.5%'. Never a rounded-away difference."""
    return f"{value * 100:g}%"


def _commit_line(commits: list[dict]) -> str:
    return ", ".join(
        f"{_pct(c['threshold'])} at step {c['step']}"
        if c["step"] is not None
        else f"{_pct(c['threshold'])} never reached"
        for c in commits
    )


# ------------------------------------------------------------- the arithmetic
#
# Pure, and deliberately so: no torch, no pipeline, no I/O. This is the half of
# the module whose correctness can be checked on a machine with no accelerator,
# which is the same reason `vla_actions.py` holds no model.


def cumulative_fractions(
    changes: list[float | None],
) -> tuple[list[float | None], float]:
    """Running share of the total movement, aligned to the step axis.

    Returns `(fractions, total)`. `fractions[i]` is `None` wherever
    `changes[i]` is — an unmeasured step stays unmeasured all the way through
    rather than becoming a zero somewhere in the middle of the arithmetic.

    The total is the LAST running sum rather than a separate `sum()`, so the
    final fraction is exactly 1.0 instead of 0.9999999999999999 — which a
    threshold of 1.0 would otherwise never reach.
    """
    measured: list[tuple[int, float]] = []
    for i, change in enumerate(changes):
        if change is None:
            continue
        # isinstance(True, int) is True, and `True` here would sail through as
        # a change of 1.0.
        if isinstance(change, bool) or not isinstance(change, (int, float)):
            raise BadRequest(
                f"the change at step {i} is not a number, so there is no "
                f"distance to take a fraction of."
            )
        value = float(change)
        if not math.isfinite(value):
            raise NotMeasurable(
                f"the change at step {i} is not finite, so this trajectory has "
                f"no measurable length. A diverged run usually means the "
                f"latent overflowed — plotting it would draw a break in the "
                f"curve that reads as the denoiser holding still."
            )
        if value < 0:
            raise BadRequest(
                f"the change at step {i} is negative, and an RMS distance "
                f"cannot be. Whatever produced this is not a distance."
            )
        measured.append((i, value))

    if not measured:
        raise BadRequest(
            "no step in this run carried a measured change, so there is "
            "nothing to take a fraction of. This is not a run that did not "
            "move — it is a run nothing looked at."
        )

    running = 0.0
    sums: list[tuple[int, float]] = []
    for i, value in measured:
        running += value
        sums.append((i, running))

    total = running
    if total == 0.0:
        raise NotMeasurable(
            "the latent is identical at every step this measured, so the total "
            "movement is exactly zero and a share of zero is not a number. "
            "This refuses rather than reporting a commit step of 0, which "
            "would read as the model deciding everything immediately."
        )

    fractions: list[float | None] = [None] * len(changes)
    for i, value in sums:
        fractions[i] = value / total
    return fractions, total


def commit_step(fractions: list[float | None], threshold: float) -> int | None:
    """The first step whose cumulative share has reached `threshold`.

    `None` when no step reaches it. `None` rather than the last step, because
    "the run never got there" and "the run got there at the end" are different
    answers about a trajectory.
    """
    check_threshold(threshold)
    for i, fraction in enumerate(fractions):
        if fraction is None:
            continue
        if fraction >= threshold:
            return i
    return None


def check_threshold(threshold: float) -> float:
    """A threshold is a share, and the two ends are both wrong for a reason."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise BadRequest(
            "the commit threshold must be a number between 0 and 1 — 0.95 "
            "means 'the step by which 95% of the movement had happened'."
        )
    value = float(threshold)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise BadRequest(
            f"a commit threshold of {threshold} is not a share of the "
            f"movement. It must be greater than 0 and at most 1: at 0 every "
            f"run 'commits' at its first measured step, which is a fact about "
            f"the threshold rather than about the model, and above 1 no run "
            f"ever commits."
        )
    return value


def check_steps(steps: int) -> int:
    """The step count, refused by name at both ends rather than clamped."""
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise BadRequest("`steps` must be a whole number of denoising steps.")
    if steps < MIN_STEPS:
        raise BadRequest(
            f"{steps} step(s) is too few to find a commit step in. A run of n "
            f"steps yields n-1 measured changes, so at two steps there is "
            f"exactly one and EVERY threshold reports step 1 — an answer that "
            f"looks like a measurement and is not. Ask for at least "
            f"{MIN_STEPS}. If this is a distilled few-step model, the honest "
            f"answer is that there is no commit step to find in two steps."
        )
    if steps > MAX_STEPS:
        raise BadRequest(
            f"{steps} steps is past the {MAX_STEPS} this will trace in one "
            f"run, and this refuses rather than running {MAX_STEPS} of them. A "
            f"scheduler asked for {MAX_STEPS} steps places them at different "
            f"timesteps from one asked for {steps}, so the shorter run is not "
            f"the first part of the longer one — it is a different run wearing "
            f"the number you asked for. Lower `steps` to at most {MAX_STEPS}."
        )
    return steps


def plan(
    steps: int,
    *,
    latent_shape: tuple[int, ...] | None = None,
    threshold: float = DEFAULT_COMMIT_THRESHOLD,
) -> dict:
    """What a trace will cost, before it costs it.

    Needs no pipeline and no model, which is the point: the memory a trace will
    hold is arithmetic on the latent shape, and `latent_shape_of` reads that off
    a pipeline without running anything.

    `latent_bytes` is `None` — never 0 — when the shape was not given. A run
    whose memory could not be priced is not a run that costs nothing.
    """
    check_steps(steps)
    check_threshold(threshold)

    latent_bytes: int | None = None
    total_bytes: int | None = None
    fits: bool | None = None
    if latent_shape:
        values = _positive_ints(latent_shape)
        if values is not None:
            latent_bytes = math.prod(values) * BYTES_PER_VALUE
            total_bytes = latent_bytes * steps
            fits = total_bytes <= MAX_TRACE_BYTES

    if latent_bytes is None:
        memory = (
            "The latent shape was not given, so the memory this holds cannot "
            "be priced — reported as unknown rather than as zero. "
            "`latent_shape_of(pipe)` reads it off a loaded pipeline."
        )
    else:
        memory = (
            f"One {latent_shape} float32 latent per step on the CPU: "
            f"{latent_bytes / 1024:,.1f} KiB each, "
            f"{total_bytes / 1024 / 1024:,.2f} MiB over the run"
            + (
                f" — within the {MAX_TRACE_BYTES / 1024 / 1024:,.0f} MiB this "
                f"will hold."
                if fits
                else (
                    f", which is past the {MAX_TRACE_BYTES / 1024 / 1024:,.0f} "
                    f"MiB this will hold. Lower `steps`, or generate at a "
                    f"smaller height and width; the run would refuse after its "
                    f"first step."
                )
            )
        )

    return {
        "steps": steps,
        "denoiser_passes": steps,
        "vae_decodes": 0,
        "latents_kept": steps,
        "latent_bytes": latent_bytes,
        "total_bytes": total_bytes,
        "fits": fits,
        "threshold": threshold,
        "means": (
            f"{steps} denoising steps, one denoiser pass each — two when "
            f"classifier-free guidance is on, which nothing here can know "
            f"without the pipeline. {memory}\n\n"
            f"NO VAE DECODES AT ALL. Nothing is turned back into pixels: a "
            f"decode is a full pass through the decoder, and doing one per step "
            f"would add {steps} of them to answer a question about the "
            f"denoiser — and would make the answer a property of the decoder "
            f"as well.\n\n"
            f"No seconds are quoted because this machine has not been timed on "
            f"this pipeline."
        ),
    }


def _positive_ints(shape) -> list[int] | None:
    """A shape, or `None` if any axis is not a positive whole number."""
    try:
        values = list(shape)
    except TypeError:
        return None
    if not values:
        return None
    out = []
    for value in values:
        # isinstance(True, int) is True, and a shape of (True, 64, 64) would
        # otherwise price a 1-channel latent from a typo.
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return None
        out.append(value)
    return out


# --------------------------------------------------------------- reading a pipe


def latent_shape_of(
    pipe,
    *,
    height: int | None = None,
    width: int | None = None,
) -> tuple[int, ...] | None:
    """The latent shape this pipeline will produce, without producing one.

    `None` when it could not be read — every source here is a config field, and
    a field that is absent is reported as unread rather than guessed. `plan()`
    then prices the memory as unknown, which is the honest answer.

    The channel count comes from the VAE, not from the denoiser. An inpainting
    UNet's `in_channels` is 9 — the latent, the mask and the masked latent
    concatenated — so reading the denoiser first would price a nine-channel
    latent that never exists.
    """
    denoiser = _denoiser_of(pipe)
    if denoiser is None:
        return None
    config = getattr(denoiser, "config", None)

    vae_config = getattr(getattr(pipe, "vae", None), "config", None)
    channels = _whole(getattr(vae_config, "latent_channels", None))
    if channels is None:
        channels = _whole(getattr(config, "in_channels", None))

    scale = _whole(getattr(pipe, "vae_scale_factor", None))
    sample = _whole(getattr(config, "sample_size", None))
    if channels is None or scale is None or scale < 1:
        return None

    # The pipeline's own default is `sample_size * vae_scale_factor`, so this
    # reproduces the call it is pricing rather than inventing a size.
    default = sample * scale if sample is not None else None
    pixels_h = _whole(height) if height is not None else default
    pixels_w = _whole(width) if width is not None else default
    if pixels_h is None or pixels_w is None:
        return None
    return (channels, pixels_h // scale, pixels_w // scale)


def _whole(value) -> int | None:
    """A positive whole number, or `None`. `True` is not a whole number here."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


#: Parameter names a denoiser's `forward` uses for the diffusion timestep.
#: This is the STRUCTURAL test — what the module takes, not what it is called.
_TIMESTEP_PARAMS = ("timestep", "timesteps", "t", "sigma")


def _denoiser_of(pipe):
    """The module that denoises, found by what it TAKES.

    The three names below — `unet`, `transformer`, `denoiser` — are right for
    everything diffusers ships today, so they stay as the fast path. They are
    a guess about naming, though, and a pipeline that calls its denoiser
    something else used to come back as "this pipeline has no `unet` or
    `transformer`, so there is no denoiser whose steps could be traced" —
    a refusal about a vocabulary, dressed as a fact about the model.

    So the fallback asks a question about the architecture instead: which
    component is an `nn.Module` whose `forward` accepts a diffusion timestep?
    That is what makes a denoiser a denoiser. A VAE, a text encoder and a
    safety checker all fail it; a UNet, a DiT and anything shaped like one
    passes it whatever its author named it.
    """
    for name in ("unet", "transformer", "denoiser"):
        found = getattr(pipe, name, None)
        if found is not None:
            return found

    import inspect

    try:
        import torch
    except Exception:  # pragma: no cover - torch is a hard dependency here
        return None

    # `components` is diffusers' own registry of what a pipeline holds, so this
    # walks the pipeline's declared parts rather than every attribute on it.
    parts = getattr(pipe, "components", None)
    if not isinstance(parts, dict):
        return None
    for part in parts.values():
        if not isinstance(part, torch.nn.Module):
            continue
        try:
            params = inspect.signature(part.forward).parameters
        except (TypeError, ValueError):
            continue
        if any(p in params for p in _TIMESTEP_PARAMS):
            return part
    return None


def _call_parameters(pipe) -> set[str]:
    """The named parameters of this pipeline's `__call__`.

    Read rather than assumed, and `**kwargs` is deliberately NOT treated as
    acceptance: a pipeline that swallows `output_type` into `**kwargs` would
    leave this believing it had skipped a decode it actually paid for.
    """
    import inspect

    try:
        signature = inspect.signature(pipe.__call__)
    except (TypeError, ValueError):
        return set()
    return {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        not in (
            parameter.VAR_KEYWORD,
            parameter.VAR_POSITIONAL,
            parameter.POSITIONAL_ONLY,
        )
    }


# -------------------------------------------------------------------- the run


def _label_table(pipe) -> tuple[dict, int | None]:
    """`({name: index}, n_classes)` for a class-conditioned pipeline.

    THREE SOURCES, because `pipe.labels` is `DiTPipeline`'s attribute and not
    a diffusers convention. Reading only that made the label lookup work for
    exactly one checkpoint and refuse every other class-conditioned model with
    "publishes no label list" — a statement about where this looked, not about
    what the model carries.

      pipe.labels            DiTPipeline's own {name: id}
      denoiser.config        `id2label`, the transformers convention, which is
                             also what `imaging` reads to count classes
      num_class_embeds       no names, but it fixes the VALID RANGE, which is
                             what a bare number has to be checked against

    Either half may be empty. Names with no count still validate a name; a
    count with no names still validates a number; neither is a reason to
    accept anything.
    """
    labels: dict = {}
    n_classes: int | None = None

    own = getattr(pipe, "labels", None)
    if isinstance(own, dict) and own:
        labels = {str(k): int(v) for k, v in own.items() if _whole(v) is not None}

    denoiser = _denoiser_of(pipe)
    config = getattr(denoiser, "config", None)
    id2label = getattr(config, "id2label", None)
    if isinstance(id2label, dict) and id2label and not labels:
        for key, name in id2label.items():
            index = _whole(key)
            if index is not None:
                labels[str(name)] = index

    for attr in ("num_class_embeds", "num_classes"):
        found = _whole(getattr(config, attr, None))
        if found:
            n_classes = found
            break
    if n_classes is None and labels:
        n_classes = max(labels.values()) + 1
    return labels, n_classes


def _class_label_of(pipe, prompt: str) -> int:
    """The class number a class-conditioned pipeline should denoise.

    A DiT is steered by a number from its own label list, not by words, so the
    prompt box means something different here and this is where that is
    resolved rather than guessed. Three forms are accepted, in order:

      "207"              the number itself, checked against the label list
      "golden retriever" a name, matched against the pipeline's own labels
      "a golden retriever on grass"
                         a name appearing inside a sentence, because the box
                         says "prompt" and people write prompts in it

    A prompt naming nothing is REFUSED with examples rather than defaulted to
    class 0. Filing every unmatched prompt under one class would produce a
    trajectory that looks like a measurement of the words and is a measurement
    of the tench.
    """
    labels, n_classes = _label_table(pipe)
    text = (prompt or "").strip()

    if text.isdigit():
        want = int(text)
        # The RANGE, from wherever it could be read — a name table, or a bare
        # `num_class_embeds` on a model that publishes no names at all.
        if n_classes is not None and not 0 <= want < n_classes:
            raise BadRequest(
                f"this model has classes 0..{n_classes - 1} and {want} is outside that."
            )
        return want

    if labels:
        lowered = text.lower()
        # Exact name first, then a name appearing inside a sentence — longest
        # match wins, so "golden retriever" beats "retriever".
        for name, index in labels.items():
            if name.lower() == lowered:
                return int(index)
        hits = sorted(
            ((name, idx) for name, idx in labels.items() if name.lower() in lowered),
            key=lambda pair: -len(pair[0]),
        )
        if hits:
            return int(hits[0][1])
        examples = ", ".join(sorted(labels)[:5])
        raise BadRequest(
            f"this model is class-conditioned: it is steered by a number from "
            f"its own label list, not by words, and nothing in {text!r} names "
            f"one of its {len(labels)} classes. Give a class number, or one of "
            f"its names — for example {examples}."
        )

    known = (
        f" It has {n_classes} classes, numbered 0..{n_classes - 1}."
        if n_classes
        else ""
    )
    raise BadRequest(
        f"this model is class-conditioned and publishes no label names, so a "
        f"class can only be given as a number.{known} Try a whole number in "
        f"place of the prompt."
    )


#: What a hook-captured latent IS, in one phrase, for every sentence that has
#: to name it. Not the same quantity as the callback's, and the difference is
#: not cosmetic — see `_capture_by_hook`.
HOOK_QUANTITY = "the denoiser's input at each step, as the scheduler scaled it"


def _capture_by_hook(pipe, store, steps: int, on_step=None):
    """Film a pipeline that will not hand its latents to a callback.

    WHY THIS EXISTS. `callback_on_step_end` is diffusers' convenience API, and
    a pipeline whose `__call__` does not take it was refused outright: "there
    is nothing here to measure between the steps". That was a claim about the
    API, not about the model. The denoiser is an `nn.Module`, and a forward
    hook on it fires once per denoising step with the latent as its first
    argument — no pipeline cooperation required at all.

    MEASURED on facebook/DiT-XL-2-256, the checkpoint that prompted this:
    `DiTPipeline.__call__` accepts `class_labels, guidance_scale, generator,
    num_inference_steps, output_type, return_dict` and no callback, and
    `DiTTransformer2DModel` has no `set_attn_processor`. Six requested steps,
    six hook calls, latents of (2, 4, 32, 32) whose mean absolute value fell
    0.766 -> 0.516 across the run. The trajectory was always there.

    IT IS NOT THE SAME QUANTITY, and that matters enough to travel in the
    response. The callback hands over the latent at the END of a step, after
    the scheduler has applied its update. A hook sees what goes IN to the
    denoiser, which the scheduler has scaled first — `scale_model_input` is
    applied on the way. So a hook-captured curve is internally consistent and
    comparable step to step, and a number from it must never be compared
    against a number from the callback path. Every result says which produced
    it.

    THE BATCH IS NOT IMAGES. Classifier-free guidance stacks the unconditional
    and conditional passes into one tensor, so the hook sees (2, ...) for a
    single image. `_Trace.add` refuses a leading dimension above 1 — correctly,
    because averaging two images' movement reports a commit step belonging to
    neither — so the conditional half is taken here. It is the second: diffusers
    concatenates [uncond, cond] throughout.
    """
    import torch

    denoiser = _denoiser_of(pipe)
    if denoiser is None:
        raise NotSupported(
            "this pipeline has no `unet` or `transformer`, so there is no "
            "denoiser to hook."
        )

    seen = {"n": 0}

    def _hook(_module, args, kwargs, _output):
        # `is not None`, never `or`. `a or b` evaluates `bool(a)`, and
        # `Tensor.__bool__` raises for anything with more than one element:
        #
        #   RuntimeError: Boolean value of Tensor with more than one value is
        #   ambiguous
        #
        # So a denoiser called with `hidden_states=` — rather than positionally
        # or as `sample=` — crashed the trace instead of being read. Installed
        # diffusers has pipelines that do exactly that AND lack
        # `callback_on_step_end`, which is precisely the combination that sends
        # `trace()` down this hook path. The `sample=` spelling worked, which is
        # why it stayed hidden.
        latent = (
            args[0]
            if args
            else next(
                (
                    kwargs[name]
                    for name in ("hidden_states", "sample")
                    if kwargs.get(name) is not None
                ),
                None,
            )
        )
        if not isinstance(latent, torch.Tensor) or latent.ndim < 2:
            return
        index = seen["n"]
        seen["n"] = index + 1
        if index >= steps:
            # A pipeline that calls its denoiser more than once per step — some
            # do for guidance variants — would otherwise overrun the trace it
            # was asked for. Kept to what was requested rather than reporting a
            # step count nobody asked about.
            return

        # The timestep, where the denoiser was told one. Positional for a UNet
        # (`sample, timestep, ...`) and for a DiT (`hidden_states, timestep`),
        # and a keyword for anything that passes it by name.
        raw = args[1] if len(args) > 1 else kwargs.get("timestep")
        if isinstance(raw, torch.Tensor):
            raw = raw.flatten()[0] if raw.numel() else None
        try:
            timestep = float(raw) if raw is not None else float(index)
        except (TypeError, ValueError):
            timestep = float(index)

        one = latent
        if int(latent.shape[0]) > 1:
            # Guidance, not a batch of images: take the conditional half.
            one = latent[latent.shape[0] // 2 :][:1]
        store.add(index, timestep, one)
        if on_step is not None:
            on_step(index, steps)

    # `with_kwargs=True` so a pipeline that calls its denoiser by keyword is
    # filmed too. Older torch has no such parameter; the positional form still
    # covers every diffusers pipeline in the wild.
    try:
        handle = denoiser.register_forward_hook(_hook, with_kwargs=True)
    except TypeError:
        handle = denoiser.register_forward_hook(lambda m, a, o: _hook(m, a, {}, o))
    return handle


def trace(
    pipe,
    prompt: str,
    *,
    seed: int | None,
    steps: int = 20,
    height: int | None = None,
    width: int | None = None,
    threshold: float = DEFAULT_COMMIT_THRESHOLD,
    on_step=None,
    # The repo id the caller loaded. See `public_model_name`: a pipeline
    # resolved from cache reports its snapshot DIRECTORY, and that path must
    # not travel in a response or inside a shared `.mri`.
    model_name: str = "",
) -> LatentTrace:
    """Run the pipeline once, keeping every step's latent, and measure the move.

    `pipe` is a loaded diffusers pipeline. This module never loads one: the
    caller decides what to hold in memory, and `imaging.detect` has already said
    whether this is a denoising architecture at all.

    `seed` is keyword-only and has NO DEFAULT, so a run cannot be left unseeded
    by forgetting. `seed=None` is allowed and is a decision — the response then
    says, in words, that the trajectory cannot be reproduced and the commit step
    cannot be compared against another run.
    """
    import torch

    if _denoiser_of(pipe) is None:
        raise NotSupported(
            "this pipeline has no `unet` or `transformer`, so there is no "
            "denoiser whose steps could be traced."
        )
    check_steps(steps)
    threshold = check_threshold(threshold)
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise BadRequest("a seed must be a whole number, or None for unseeded.")

    generator = None
    if seed is not None:
        # CPU generator, moved by the pipeline — the same choice
        # `image_attention.capture` makes and for the same reason: CUDA's stream
        # differs for the same seed, so "seed 7" would not be the same
        # trajectory on a machine without a GPU, and a seed that means different
        # things on different machines is not a seed.
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

    store = _Trace(steps=steps)

    def _tick(_pipe, step_index, timestep, kwargs):
        latents = kwargs.get("latents")
        if latents is None:
            raise NotSupported(
                "this pipeline does not hand its latents to "
                "`callback_on_step_end`, so there is nothing here whose change "
                "could be measured. Reconstructing them from the denoiser's "
                "input would mean undoing whatever scaling the scheduler "
                "applied, which is a different quantity from the one the "
                "pipeline carried."
            )
        store.add(int(step_index), float(timestep), latents)
        if on_step is not None:
            on_step(int(step_index), steps)
        return kwargs

    accepted = _call_parameters(pipe)
    call_kwargs: dict = {}
    if height is not None:
        call_kwargs["height"] = int(height)
    if width is not None:
        call_kwargs["width"] = int(width)
    decodes = 1
    if "output_type" in accepted:
        # The pipeline's own final decode, skipped. The per-step decodes were
        # never going to happen; this is the one that would have, and it is free
        # to avoid because the latents are what this measures.
        call_kwargs["output_type"] = "latent"
        decodes = 0

    # TWO WAYS IN, and the pipeline decides which. `callback_on_step_end` is
    # the one diffusers documents; a hook on the denoiser is the one that works
    # regardless. See `_capture_by_hook` for what the hook captures and why it
    # is not interchangeable with the callback's latent.
    by_hook = "callback_on_step_end" not in accepted
    handle = None
    try:
        if by_hook:
            handle = _capture_by_hook(pipe, store, steps, on_step)
            # A class-conditioned pipeline takes no prompt at all, and passing
            # one positionally lands it in `class_labels`. The caller's prompt
            # is carried into the response either way, so a reader can see what
            # was — and was not — given to the model.
            if "prompt" in accepted:
                call_kwargs["prompt"] = prompt
            elif "class_labels" in accepted:
                call_kwargs["class_labels"] = [_class_label_of(pipe, prompt)]
        with torch.inference_mode():
            if by_hook:
                pipe(
                    num_inference_steps=steps,
                    generator=generator,
                    **call_kwargs,
                )
            else:
                pipe(
                    prompt,
                    num_inference_steps=steps,
                    generator=generator,
                    callback_on_step_end=_tick,
                    **call_kwargs,
                )
    except BaseException:
        # Nothing was attached to the pipeline — `callback_on_step_end` is an
        # argument to the call rather than state on the model, so unlike
        # `image_attention.capture` there is no processor to put back. What has
        # to be released is this function's own buffer: a traceback keeps every
        # frame's locals alive for as long as the exception is being handled, so
        # a 50-step trace that failed at step 49 would otherwise hold 49 latents
        # through every handler above this one. "It will be collected
        # eventually" is not a memory bound.
        store.release()
        raise
    finally:
        # A hook left on the denoiser would film every later run too, and
        # `image_attention.capture` learned that lesson for attention
        # processors. Removed on every path, including the raising one.
        if handle is not None:
            handle.remove()

    if len(store.latents) < 2:
        raise NotSupported(
            f"the run finished having handed over {len(store.latents)} "
            f"latent(s), and measuring how far a latent moved needs at least "
            f"two. This pipeline may drive its scheduler somewhere the "
            f"step-end callback does not reach, which is a gap in coverage "
            f"rather than a property of the model."
        )

    rows, total = _rows(store, threshold=threshold)
    return LatentTrace(
        prompt=prompt,
        seed=seed,
        model=public_model_name(pipe, model_name),
        scheduler=type(getattr(pipe, "scheduler", None)).__name__,
        steps=rows,
        steps_requested=steps,
        latent_shape=store.shape,
        threshold=threshold,
        total_change=round(total, 6),
        vae_decodes=decodes,
        bytes_held=store.bytes_held,
        captured_by="hook" if by_hook else "callback",
    )


class _Trace:
    """The kept latents, bounded, on the CPU.

    One (channels x h x w) float32 array per step. The bound is checked against
    the FIRST real latent rather than against a guess, so a shape this cannot
    hold is refused after one step instead of after all of them.
    """

    def __init__(self, *, steps: int) -> None:
        self.steps = steps
        self.indices: list[int] = []
        self.timesteps: list[float] = []
        self.latents: list = []
        self.shape: tuple[int, ...] = ()
        self.bytes_held = 0

    def add(self, index: int, timestep: float, latent) -> None:
        import torch

        if getattr(latent, "ndim", 0) < 2:
            raise NotSupported(
                f"this pipeline's latent has {getattr(latent, 'ndim', 0)} "
                f"dimension(s), so there is no batch axis to read and nothing "
                f"here can say what it is looking at."
            )
        if int(latent.shape[0]) != 1:
            raise NotSupported(
                f"this run is denoising {int(latent.shape[0])} images at once "
                f"and their latents travel stacked in one tensor. Averaging "
                f"their movement into one curve would report a commit step "
                f"that belongs to none of them. Trace one image at a time."
            )

        # float32 before differencing, not after. The pipeline's latents are
        # often float16, and a float16 difference between two nearly identical
        # late-step latents rounds away exactly the small values a commit
        # reading depends on.
        held = latent.detach().to("cpu", torch.float32)
        # `.copy()` unconditionally. `.to()` returns the SAME storage when
        # nothing needs converting — which is what happens on a CPU run — and
        # `.numpy()` would then share memory with the live latent. A scheduler
        # that writes its result in place would rewrite every step already held,
        # and the symptom would be every change reading as zero: a chart saying
        # the model committed at step 1.
        array = held.numpy().copy()

        if not self.latents:
            self.shape = tuple(int(v) for v in array.shape[1:])
            projected = int(array.nbytes) * self.steps
            if projected > MAX_TRACE_BYTES:
                raise BadRequest(
                    f"a {self.shape} latent is "
                    f"{array.nbytes / 1024 / 1024:,.2f} MiB, so holding one per "
                    f"step for {self.steps} steps would need "
                    f"{projected / 1024 / 1024:,.0f} MiB — past the "
                    f"{MAX_TRACE_BYTES / 1024 / 1024:,.0f} MiB this will hold. "
                    f"Lower `steps` to at most "
                    f"{max(1, MAX_TRACE_BYTES // max(1, int(array.nbytes)))}, "
                    f"or generate at a smaller height and width. `plan()` "
                    f"prices this before the run rather than one step into it."
                )

        self.indices.append(index)
        self.timesteps.append(timestep)
        self.latents.append(array)
        self.bytes_held += int(array.nbytes)

    def release(self) -> None:
        """Drop the held latents. Idempotent, and safe to call mid-failure."""
        self.latents = []
        self.bytes_held = 0


def _rows(store: _Trace, *, threshold: float) -> tuple[list[StepChange], float]:
    """Turn the kept latents into the step axis, then release them."""
    latents = store.latents
    final = latents[-1]
    measured = len(latents)

    changes: list[float | None] = [None]
    for i in range(1, measured):
        changes.append(_rms(latents[i] - latents[i - 1], where=store.indices[i]))
    to_final = [
        _rms(latent - final, where=index)
        for index, latent in zip(store.indices, latents, strict=True)
    ]
    scale = [
        _rms(latent, where=index)
        for index, latent in zip(store.indices, latents, strict=True)
    ]

    # Released before the fractions are computed rather than after, and the two
    # local references are dropped with it — clearing the store while this
    # function still holds the list would release nothing at all. The
    # arithmetic below needs floats, not latents, and the caller is holding a
    # model on a machine that may be about to want this memory back.
    #
    # `del` rather than rebinding to `[]` and `None`. The old form read as two
    # dead assignments — CodeQL called them exactly that — because nothing
    # downstream reads either name. Dropping a reference IS the intent, and
    # `del` is the statement that says so; the rebinding also allocated a fresh
    # list on the way to freeing memory.
    del latents, final
    store.release()

    fractions, total = cumulative_fractions(changes)
    check_threshold(threshold)

    rows = [
        StepChange(
            step=store.indices[i],
            timestep=store.timesteps[i],
            rms_change=None if changes[i] is None else round(changes[i], 6),
            cumulative=None if fractions[i] is None else round(fractions[i], 6),
            rms_to_final=round(to_final[i], 6),
            latent_rms=round(scale[i], 6),
        )
        for i in range(measured)
    ]
    return rows, total


def _rms(array, *, where: int) -> float:
    """Root mean square of an array, refused when it is not finite.

    The squares are accumulated in float64 while the array stays float32: a
    float32 sum over sixteen thousand elements of a diverging latent can reach
    infinity on its own, and reporting that as "this run diverged" would blame
    the model for a summation.
    """
    import numpy as np

    value = float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))
    if not math.isfinite(value):
        raise NotMeasurable(
            f"the latent at step {where} holds values that are not finite, so "
            f"there is no distance to measure from it. A run that overflows "
            f"this way is usually float16 arithmetic that diverged; a NaN drawn "
            f"on this curve would be a break in the line, which reads as the "
            f"denoiser holding still."
        )
    return value


# -------------------------------------------------------------- the filmstrip
#
# Everything below decodes, which is exactly what the measurement above refuses
# to do. Both are true at once: the trace stays undecoded so its number belongs
# to the denoiser, and this decodes on purpose so somebody can watch the picture
# arrive. Two functions, two objects, and neither is a mode of the other — see
# the module docstring for why a flag would have cost `trace()` its checkable
# `vae_decodes: 0`.


# The longest side a frame is EMITTED at. This bounds the PAYLOAD and nothing
# else: the decoder still runs at whatever the pipeline generated — 1024x1024 on
# SDXL — and the picture is resized afterwards. Said out loud wherever it is
# reported, because "bounded resolution" reads as "cheaper decode" and is not.
def public_model_name(pipe, given: str = "") -> str:
    """A model NAME, never a path off this machine.

    `given` wins: the server knows the repo id it loaded, and an id somebody
    typed is the answer a reader wants. Otherwise `name_or_path`, which
    diffusers sets to the snapshot DIRECTORY when it resolved from cache --
    recoverable, because the Hub's cache encodes the repo id in the directory
    name (`models--org--repo`). Anything else that still looks like a path is
    dropped rather than published: an empty model field is a gap, and a gap is
    better than somebody's home directory in a file they shared.
    """
    if given:
        return given
    raw = str(getattr(pipe, "name_or_path", "") or "")
    if not raw:
        return ""
    marker = "models--"
    if marker in raw:
        tail = raw.split(marker, 1)[1]
        # `models--org--repo/snapshots/<sha>` -> `org/repo`
        repo = tail.replace("\\", "/").split("/", 1)[0]
        return repo.replace("--", "/")
    if "/" in raw or "\\" in raw or ":" in raw:
        return ""
    return raw


DEFAULT_FRAME_PIXELS = 384
MIN_FRAME_PIXELS = 32
MAX_FRAME_PIXELS = 768

# The most frames one strip will decode. Each is a full pass through the decoder
# on the card already holding the pipeline, and each becomes a PNG in the
# response, so this bounds device memory and payload at once. A selection past
# it is REFUSED, naming the `every` that would fit, rather than truncated: a
# strip silently shortened is the eight frames wearing a fifty-step label this
# whole feature exists to make impossible.
MAX_FRAMES = 12


@dataclass
class Frame:
    """One decoded step: the picture, and the size it is really at."""

    # The step-end index this latent was handed over at, NOT the frame's
    # position in the strip. A gap in these numbers is the whole point.
    step: int
    timestep: float | None = None
    # PNG bytes rather than a string. `to_dict` base64s them for the API; an
    # in-process caller writing a file wants bytes, not a data URL to undo.
    png: bytes = b""
    width: int | None = None
    height: int | None = None
    # What the decoder produced, before any resize. Carried beside the emitted
    # size for the rule `session.py` holds for a robot frame: a picture silently
    # shrunk is a picture of a resolution the model never worked at.
    decoded_width: int | None = None
    decoded_height: int | None = None
    # RMS of the latent this was decoded from — the same arithmetic and the same
    # name as the trace's column, so the two can be read side by side.
    latent_rms: float | None = None

    @property
    def downsampled(self) -> bool:
        if self.width is None or self.decoded_width is None:
            return False
        return (self.width, self.height) != (self.decoded_width, self.decoded_height)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "timestep": self.timestep,
            # `None`, never "", when there are no bytes: an empty data URL is a
            # broken image in a browser and looks like a decode that produced
            # black rather than one that never happened.
            "png": (
                "data:image/png;base64," + base64.b64encode(self.png).decode("ascii")
                if self.png
                else None
            ),
            "png_bytes": len(self.png),
            "width": self.width,
            "height": self.height,
            "decoded_width": self.decoded_width,
            "decoded_height": self.decoded_height,
            "downsampled": self.downsampled,
            "latent_rms": self.latent_rms,
        }


@dataclass
class Filmstrip:
    """The decoded frames, which steps they are, and what they are not."""

    prompt: str = ""
    seed: int | None = None
    model: str = ""
    scheduler: str = ""
    frames: list[Frame] = field(default_factory=list)
    steps_requested: int = 0
    # How many steps the pipeline actually handed over. Separate from
    # `steps_requested` because a pipeline that runs its schedule differently is
    # a fact about the run, not a rounding of the request.
    steps_run: int = 0
    decoded_steps: list[int] = field(default_factory=list)
    # Steps that RAN and were never looked at. The field that makes an 8-frame
    # strip impossible to read as an 8-step run.
    skipped_steps: list[int] = field(default_factory=list)
    # Steps that were selected and never arrived. Distinct from `skipped_steps`:
    # one is a choice, the other is a gap, and folding them together would hide
    # a pipeline whose callback does not reach every step.
    steps_never_reached: list[int] = field(default_factory=list)
    selection: dict = field(default_factory=dict)
    latent_shape: tuple[int, ...] = ()
    frame_pixels: int = DEFAULT_FRAME_PIXELS
    # MEASURED by wrapping `pipe.vae.decode` for the duration of the call, not
    # counted from intent. The sibling's 0 is checkable; so is this.
    vae_decodes: int = 0
    vae_decodes_for_frames: int = 0
    final_decode_skipped: bool | None = None
    # The numbers the latents were put through on the way to the decoder, as
    # read off this VAE's own config. Published because they change what the
    # frame LOOKS like, and `None` when the config did not carry them.
    latent_scaling: float | None = None
    latent_shift: float | None = None
    # The most the held latents ever occupied on the host at once. A real number
    # rather than the zero this object holds after releasing them.
    host_latent_bytes: int = 0
    peak_device_bytes: int | None = None
    peak_device: str | None = None
    peak_source: str | None = None
    # Why there is no peak, when there is none. `None` and 0 are different
    # answers and an unmeasurable peak is not a peak of zero.
    peak_unmeasured: str | None = None

    @property
    def png_bytes_total(self) -> int:
        return sum(len(f.png) for f in self.frames)

    def save(self, directory) -> list[str]:
        """Write every frame as a PNG under `directory`. Returns the paths.

        Named `step_<index>.png` by the STEP the frame came from rather than by
        its position in the strip, so a directory listing shows the gaps: eight
        files numbered 000, 006, 012 … are visibly eight of fifty steps, and
        eight files numbered 000 to 007 would not be.
        """
        from pathlib import Path

        root = Path(directory)
        if root.exists() and not root.is_dir():
            raise BadRequest(
                f"the frames have nowhere to go: {root} is a file rather than a "
                f"directory. Name a directory — this writes one PNG per frame."
            )
        root.mkdir(parents=True, exist_ok=True)
        written = []
        for frame in self.frames:
            path = root / f"step_{frame.step:03d}.png"
            path.write_bytes(frame.png)
            written.append(str(path))
        return written

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "seed": self.seed,
            "model": self.model,
            "scheduler": self.scheduler,
            "frames": [f.to_dict() for f in self.frames],
            "frames_decoded": len(self.frames),
            "steps_requested": self.steps_requested,
            "steps_run": self.steps_run,
            "decoded_steps": list(self.decoded_steps),
            "skipped_steps": list(self.skipped_steps),
            "steps_never_reached": list(self.steps_never_reached),
            "selection": dict(self.selection),
            "latent_shape": list(self.latent_shape),
            "frame_pixels": self.frame_pixels,
            "vae_decodes": self.vae_decodes,
            "vae_decodes_for_frames": self.vae_decodes_for_frames,
            "final_decode_skipped": self.final_decode_skipped,
            "latent_scaling": self.latent_scaling,
            "latent_shift": self.latent_shift,
            "host_latent_bytes": self.host_latent_bytes,
            "png_bytes_total": self.png_bytes_total,
            "peak_device_bytes": self.peak_device_bytes,
            "peak_device": self.peak_device,
            "peak_source": self.peak_source,
            "peak_unmeasured": self.peak_unmeasured,
            "means": self.means(),
        }

    def means(self) -> str:
        seeded = (
            f"Seed {self.seed}."
            if self.seed is not None
            else (
                "NO SEED WAS FIXED, so this strip cannot be reproduced and no "
                "other run will pass through these pictures — the next one "
                "starts from different noise."
            )
        )
        ran = (
            f" The pipeline handed over {self.steps_run} step(s) against the "
            f"{self.steps_requested} asked for."
            if self.steps_run != self.steps_requested
            else ""
        )
        never = (
            f"\n\n{len(self.steps_never_reached)} SELECTED STEP(S) NEVER "
            f"ARRIVED: {_ranges(self.steps_never_reached)}. They were asked for "
            f"and the run's callback never reached them, which is a gap in what "
            f"this could see rather than a choice about what to show."
            if self.steps_never_reached
            else ""
        )
        # Built from the COUNT rather than from what was asked for. "The
        # pipeline also decoded its own image" is a claim, and this only ever
        # says it when the counter saw it happen.
        asked = (
            "The pipeline was asked for `output_type='latent'` so it would skip "
            "its own final decode"
            if self.final_decode_skipped
            else (
                "This pipeline does not accept `output_type='latent'`, so its "
                "own final decode was never skipped"
            )
        )
        extra = self.vae_decodes - self.vae_decodes_for_frames
        if extra > 0:
            pipeline_decode = (
                f"{asked}, and {extra} decode(s) beyond the frames were counted "
                f"anyway — they are inside the total above."
            )
        elif extra == 0:
            pipeline_decode = (
                f"{asked}, and nothing beyond the frames was counted: the total "
                f"is exactly the frames."
            )
        else:
            pipeline_decode = (
                f"{asked}. The counter recorded FEWER decodes than there are "
                f"frames, which should not be possible — reported rather than "
                f"smoothed over, because a count that disagrees with itself is "
                f"the one thing worth knowing here."
            )
        scaled = (
            f"through this VAE's own scaling factor {self.latent_scaling:g}"
            if self.latent_scaling
            else (
                "with NO scaling factor applied, because this VAE's config did "
                "not publish one — so the contrast of these frames may not "
                "match what the pipeline's own output looks like"
            )
        )
        shifted = (
            f" and its shift factor {self.latent_shift:g}" if self.latent_shift else ""
        )
        # Only frames whose four sizes are all known: a size half-reported is
        # not a size, and it would sort against a None.
        sizes = sorted(
            {
                (f.decoded_width, f.decoded_height, f.width, f.height)
                for f in self.frames
                if None not in (f.decoded_width, f.decoded_height, f.width, f.height)
            }
        )
        if not sizes:
            resolution = (
                "RESOLUTION: no frame carried a size, so there is none to report here."
            )
        else:
            resized = [s for s in sizes if (s[0], s[1]) != (s[2], s[3])]
            resolution = (
                f"RESOLUTION: the decoder produced "
                f"{'; '.join(f'{a}x{b}' for a, b, _c, _d in sizes)} and the "
                f"frames are emitted at "
                f"{'; '.join(f'{c}x{d}' for _a, _b, c, d in sizes)}, the "
                f"longest side bounded to {self.frame_pixels} px. "
                + (
                    "That resize bounds this RESPONSE and not the decode: the "
                    "VAE still ran at full size and cost full memory, so a "
                    "smaller `frame_pixels` will not make a decode fit that did "
                    "not."
                    if resized
                    else "Nothing was resized — the decode was already within it."
                )
            )
        peak = (
            f"PEAK DEVICE MEMORY: {self.peak_device_bytes / 1024 / 1024:,.0f} "
            f"MiB, the allocator's high-water mark on {self.peak_device} across "
            f"this whole call — the denoising and every decode together, with "
            f"this pipeline's weights already resident. Read through "
            f"{self.peak_source}, so it is the allocator's number and not the "
            f"driver's: the driver reserves more than torch allocates. Measured "
            f"once, on this machine, for this run."
            if self.peak_device_bytes is not None
            else (
                f"PEAK DEVICE MEMORY: not measured — "
                f"{self.peak_unmeasured or 'no allocator counter was available'} "
                f"Reported as null rather than as zero, which would claim the "
                f"decodes were free."
            )
        )

        return (
            f"{len(self.frames)} decoded frame(s) from a {self.steps_run}-step "
            f"run of {self.model or 'this model'} on "
            f"{self.scheduler or 'its scheduler'}, at the prompt given. "
            f"{seeded}{ran}\n\n"
            f"THIS IS A SUBSET, AND THIS IS EXACTLY WHICH. Decoded: step(s) "
            f"{_ranges(self.decoded_steps)}. Run and never looked at: "
            f"{len(self.skipped_steps)} step(s) — {_ranges(self.skipped_steps)}. "
            f"So the difference between two neighbouring frames is everything "
            f"that happened across the gap between their step numbers, not one "
            f"step's work, and the strip is not a {len(self.frames)}-step run."
            f"{never}\n\n"
            f"{self.vae_decodes} VAE DECODE(S), COUNTED RATHER THAN CLAIMED. "
            f"`pipe.vae.decode` was wrapped for the duration of this call and "
            f"this is the number of times it was really called; "
            f"{self.vae_decodes_for_frames} of them made the frames above. "
            f"{pipeline_decode} Its sibling `trace()` decodes nothing at all and "
            f"reports 0 by the same counting, which is the claim this had to "
            f"leave intact.\n\n"
            f"WHAT A FRAME AT STEP k IS. The latent the pipeline handed over at "
            f"the END of step k, put through this run's own VAE {scaled}"
            f"{shifted}, denormalised the standard way — (x / 2 + 0.5), clamped "
            f"— and written as PNG. At the last decoded step that is the picture "
            f"this run would have produced, before whatever watermark or safety "
            f"pass the pipeline applies after its own decode.\n\n"
            f"WHAT A FRAME AT STEP k IS NOT. It is not the model's guess at the "
            f"finished image at that point: schedulers carry a prediction of the "
            f"clean sample and that is a different tensor, while this decodes "
            f"the RUNNING latent with the run's noise still in it. An early "
            f"frame that looks like noise is the latent, not the model failing "
            f"and not a bad prompt.\n\n"
            f"LATENT DISTANCE IS NOT VISIBLE DIFFERENCE, AND NOW BOTH EXIST. "
            f"`trace()` measures how far the LATENT moved per step; these frames "
            f"show what one decoder makes of it. They will disagree — the "
            f"decoder is non-linear, so a small late move can be a visible "
            f"change in texture and a large early one can vanish — and neither "
            f"is the other being wrong. Anything you measure BETWEEN two of "
            f"these pictures is a property of this VAE as much as of the "
            f"denoiser, which is the whole reason `trace()` refuses to decode; "
            f"per-step movement is its job and is not recomputed here.\n\n"
            f"{resolution}\n\n"
            f"{peak}\n\n"
            f"ONE TRAJECTORY, NOT A PROPERTY OF THE MODEL. One prompt, one seed, "
            f"one scheduler and one step count. The same model on another "
            f"scheduler forms its picture somewhere else entirely, because a "
            f"scheduler is a decision about where to put the steps."
        )


def _ranges(values) -> str:
    """[1,2,3,5] -> '1-3, 5'. Compressed for prose, and nothing is dropped.

    Groups rather than a truncated list, so a 190-step gap reads as one range
    instead of being cut off with an ellipsis. The payload carries every index
    either way — this is only how the sentence says it.
    """
    ordered = sorted(set(int(v) for v in values))
    if not ordered:
        return "none"
    groups: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        groups.append((start, previous))
        start = previous = value
    groups.append((start, previous))
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in groups)


# ------------------------------------------------------- choosing the subset
#
# Pure, like the arithmetic above and for the same reason: which steps get
# decoded is the caller's decision and the most important thing this feature
# reports, so it is checkable without a pipeline, a GPU or a download.


def select_steps(
    steps: int,
    *,
    every: int | None = None,
    at=None,
    include_final: bool = True,
) -> list[int]:
    """Which steps get decoded. The caller decides; nothing here picks for them.

    Exactly one of `every` and `at`. There is deliberately no default subset:
    "decode the whole run" is the cost this exists not to pay, and a subset
    chosen here would be this module deciding what a reader gets to see.

    Indices are the step-end callback's own, 0-based — index 0 is the END of the
    first denoising step, the earliest thing any of this can observe, and there
    is no index for the noise the run started from.

    `include_final` adds the last step to either mode, because a strip that
    stops one step short of the finished picture is the one frame everybody
    assumes is there. It is an argument rather than a rule, and it travels in
    the response.
    """
    _filmstrip_steps(steps)
    if not isinstance(include_final, bool):
        raise BadRequest(
            "`include_final` is a yes or no: whether the run's last step is "
            "decoded even when the pattern you gave would miss it."
        )
    if (every is None) == (at is None):
        raise BadRequest(
            "a filmstrip decodes a SUBSET of the steps, and this call gave "
            "either both ways of choosing it or neither. Pass `every=N` for "
            "every Nth step, or `at=[...]` for exactly the steps you want. "
            "Nothing here chooses for you: decoding every step is the cost this "
            "exists not to pay, and a default subset would be this module "
            "deciding what you get to look at."
        )

    if every is not None:
        if isinstance(every, bool) or not isinstance(every, int) or every < 1:
            raise BadRequest(
                "`every` is a whole number of steps between frames — `every=4` "
                "decodes steps 0, 4, 8 and so on. Zero or negative is not a "
                "spacing."
            )
        chosen = list(range(0, steps, every))
    else:
        chosen = _explicit_steps(at, steps)

    if include_final:
        chosen.append(steps - 1)
    chosen = sorted(set(chosen))

    if len(chosen) > MAX_FRAMES:
        raise BadRequest(
            _too_many_frames(steps, chosen, every=every, include_final=include_final)
        )
    return chosen


def _explicit_steps(at, steps: int) -> list[int]:
    """`at=[...]` checked one index at a time, refused by name at both ends."""
    try:
        values = list(at)
    except TypeError:
        raise BadRequest(
            "`at` is a list of step indices to decode, like `at=[0, 5, 19]`."
        ) from None
    if not values:
        raise BadRequest(
            "`at` named no steps, so there is no frame to decode. An empty "
            "strip is not a cheaper strip — it is a call with nothing in it."
        )
    out = []
    for value in values:
        # isinstance(True, int) is True, and `at=[True]` would decode step 1.
        if isinstance(value, bool) or not isinstance(value, int):
            raise BadRequest(
                "every entry in `at` is a step index — a whole number of the "
                "run's own steps."
            )
        if not 0 <= value < steps:
            raise BadRequest(
                f"step {value} is not in a {steps}-step run, whose steps are 0 "
                f"to {steps - 1}. Step 0 is the END of the first denoising "
                f"step; the noise the run started from has no index here, "
                f"because the pipeline never hands it over."
            )
        out.append(value)
    return out


def _too_many_frames(steps: int, chosen: list[int], *, every, include_final) -> str:
    """The refusal for a selection past `MAX_FRAMES`, with the fix in it."""
    if every is None:
        return (
            f"`at` names {len(chosen)} steps and this decodes at most "
            f"{MAX_FRAMES} in one strip — each frame is a full pass through the "
            f"decoder on the card already holding the pipeline. Name at most "
            f"{MAX_FRAMES}. This refuses rather than dropping the extras: a "
            f"strip quietly shortened is a strip whose gaps nobody was told "
            f"about, which is the one thing this feature must never produce."
        )
    fits = None
    for candidate in range(every + 1, steps + 1):
        count = len(range(0, steps, candidate))
        if include_final and (steps - 1) % candidate:
            count += 1
        if count <= MAX_FRAMES:
            fits = candidate
            break
    advice = (
        f"Raise `every` to at least {fits}"
        if fits is not None
        else "Ask for fewer steps"
    )
    return (
        f"`every={every}` over {steps} steps selects {len(chosen)} frames, past "
        f"the {MAX_FRAMES} this decodes in one strip — each is a full pass "
        f"through the decoder on the card already holding the pipeline. "
        f"{advice}, or name the steps you want in `at`. This refuses rather "
        f"than dropping the extras: a strip quietly shortened is a strip whose "
        f"gaps nobody was told about, which is the one thing this feature must "
        f"never produce."
    )


def _filmstrip_steps(steps: int) -> int:
    """The step count for a strip: the same ceiling, a different floor.

    `check_steps` refuses fewer than `MIN_STEPS` because a commit step needs at
    least two measured changes before it means anything. A filmstrip has no such
    arithmetic — a two-step distilled run is a perfectly watchable two frames —
    so the floor here is one step. The CEILING is the same constant for the same
    reason as there: a schedule truncated to fit is a different run.
    """
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise BadRequest("`steps` must be a whole number of denoising steps.")
    if steps < 1:
        raise BadRequest("a run needs at least one denoising step to film.")
    if steps > MAX_STEPS:
        raise BadRequest(
            f"{steps} steps is past the {MAX_STEPS} this will run in one "
            f"filmstrip, and it refuses rather than running {MAX_STEPS} of "
            f"them: a scheduler asked for {MAX_STEPS} steps places them at "
            f"different timesteps from one asked for {steps}, so the shorter "
            f"run is a different run wearing the number you asked for. Lower "
            f"`steps` to at most {MAX_STEPS}."
        )
    return steps


def check_frame_pixels(pixels: int) -> int:
    """The emitted resolution bound, refused by name at both ends."""
    if isinstance(pixels, bool) or not isinstance(pixels, int):
        raise BadRequest(
            "`frame_pixels` is a whole number of pixels — the longest side each "
            "frame is emitted at."
        )
    if not MIN_FRAME_PIXELS <= pixels <= MAX_FRAME_PIXELS:
        raise BadRequest(
            f"`frame_pixels={pixels}` is outside the {MIN_FRAME_PIXELS} to "
            f"{MAX_FRAME_PIXELS} this emits. Below {MIN_FRAME_PIXELS} a frame "
            f"is a thumbnail nobody can read a picture off; above "
            f"{MAX_FRAME_PIXELS} a strip of {MAX_FRAMES} base64 PNGs is a "
            f"response measured in tens of megabytes. It bounds the response "
            f"either way and never the decode."
        )
    return pixels


def filmstrip_plan(
    steps: int,
    *,
    every: int | None = None,
    at=None,
    include_final: bool = True,
    frame_pixels: int = DEFAULT_FRAME_PIXELS,
    latent_shape: tuple[int, ...] | None = None,
) -> dict:
    """What a strip will cost, before it costs it.

    The same shape as `plan()` and refusing exactly what `filmstrip()` refuses,
    in the same order — a preflight that accepts a call the call rejects is a
    promise the next request breaks.

    Needs no pipeline. `latent_shape_of(pipe)` reads the shape off a loaded one
    without generating anything, and without it the memory half is `None` rather
    than zero.
    """
    _filmstrip_steps(steps)
    chosen = select_steps(steps, every=every, at=at, include_final=include_final)
    frame_pixels = check_frame_pixels(frame_pixels)
    skipped = [i for i in range(steps) if i not in set(chosen)]

    latent_bytes: int | None = None
    total_bytes: int | None = None
    fits: bool | None = None
    if latent_shape:
        values = _positive_ints(latent_shape)
        if values is not None:
            latent_bytes = math.prod(values) * BYTES_PER_VALUE
            total_bytes = latent_bytes * len(chosen)
            fits = total_bytes <= MAX_TRACE_BYTES

    if latent_bytes is None:
        memory = (
            "The latent shape was not given, so the host memory this holds "
            "cannot be priced — reported as unknown rather than as zero. "
            "`latent_shape_of(pipe)` reads it off a loaded pipeline."
        )
    else:
        memory = (
            f"One {latent_shape} float32 latent held per FRAME on the host — "
            f"{latent_bytes / 1024:,.1f} KiB each, "
            f"{total_bytes / 1024 / 1024:,.2f} MiB for the "
            f"{len(chosen)} of them, released one at a time as they decode"
            + (
                f" — within the {MAX_TRACE_BYTES / 1024 / 1024:,.0f} MiB this "
                f"will hold."
                if fits
                else (
                    f", which is past the {MAX_TRACE_BYTES / 1024 / 1024:,.0f} "
                    f"MiB this will hold. Ask for fewer frames, or generate at "
                    f"a smaller height and width; the run would refuse at its "
                    f"first selected step."
                )
            )
        )

    return {
        "steps": steps,
        "denoiser_passes": steps,
        "frames": len(chosen),
        "decoded_steps": chosen,
        "skipped_steps": skipped,
        "vae_decodes": len(chosen),
        "vae_decodes_if_pipeline_also_decodes": len(chosen) + 1,
        "latents_kept": len(chosen),
        "latent_bytes": latent_bytes,
        "total_bytes": total_bytes,
        "fits": fits,
        "frame_pixels": frame_pixels,
        "png_bytes": None,
        "peak_device_bytes": None,
        "selection": _selection(every, at, include_final),
        "means": (
            f"{steps} denoising steps, one denoiser pass each — two when "
            f"classifier-free guidance is on, which nothing here can know "
            f"without the pipeline. {len(chosen)} of those steps get decoded "
            f"(step {_ranges(chosen)}); the other {len(skipped)} run and are "
            f"never looked at.\n\n"
            f"{len(chosen)} VAE DECODES, one per frame — plus ONE MORE if this "
            f"pipeline does not accept `output_type='latent'` and decodes its "
            f"own final image, which cannot be known without the pipeline and "
            f"so is stated as a maybe rather than folded into the number. The "
            f"run counts the real total by wrapping `pipe.vae.decode`, and "
            f"reports what it counted.\n\n"
            f"{memory}\n\n"
            f"`frame_pixels={frame_pixels}` BOUNDS THE RESPONSE, NOT THE "
            f"DECODE. The decoder still runs at whatever the pipeline "
            f"generates and the picture is resized afterwards, so this makes "
            f"the payload smaller and the decode no cheaper. If it is the "
            f"decode that runs out of memory, generate at a smaller height and "
            f"width, or turn on the decoder's own tiling before calling.\n\n"
            f"PNG SIZE AND PEAK MEMORY ARE NOT PRICED HERE. PNG size depends on "
            f"the picture and the peak depends on this machine and this "
            f"decoder — both are null rather than invented, and the run reports "
            f"the bytes and the peak it actually observed.\n\n"
            f"No seconds are quoted because this machine has not been timed on "
            f"this pipeline."
        ),
    }


def _selection(every, at, include_final: bool) -> dict:
    """How the subset was asked for, carried beside which steps it produced."""
    return {
        "mode": "every" if every is not None else "at",
        "every": every,
        "at": None if at is None else [int(v) for v in at],
        "include_final": bool(include_final),
    }


def _call_surface(pipe) -> set[str] | None:
    """This pipeline's named `__call__` parameters, or `None` for "cannot say".

    Deliberately three-valued, and it is the `None` that earns the function.
    `_call_parameters` above answers "does this accept `output_type`", where a
    wrong "no" costs one avoidable decode. Here the answer decides whether to
    REFUSE, so absence has to mean absence: a `__call__` that takes `**kwargs`,
    or one nothing can introspect at all, proves nothing about what it accepts
    and gets `None` rather than an empty set. Refusing on that would turn every
    wrapper around a working pipeline into a "not supported".
    """
    import inspect

    if not callable(pipe):
        return None
    try:
        signature = inspect.signature(pipe.__call__)
    except (TypeError, ValueError, AttributeError):
        return None
    parameters = signature.parameters.values()
    if any(p.kind is p.VAR_KEYWORD for p in parameters):
        return None
    return {
        p.name
        for p in parameters
        if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL, p.POSITIONAL_ONLY)
    }


def _check_call_surface(pipe) -> None:
    """Refuse a pipeline this cannot film, by name, before it runs anything.

    Both refusals are about the CALL, not about the family. A text-conditioned
    DiT — PixArt, SD3 — takes a prompt, exposes the step-end callback and films
    exactly like a UNet; a class-conditioned one takes ImageNet ids and exposes
    no callback at all. Reading the signature says which pipeline this is,
    instead of assuming from the name in the `transformer` slot, and the
    alternative is a `TypeError` from somebody else's library reaching the
    reader as "something broke".
    """
    accepted = _call_surface(pipe)
    if accepted is None:
        return
    if "callback_on_step_end" not in accepted:
        raise NotSupported(
            "this pipeline's `__call__` does not accept "
            "`callback_on_step_end`, which is the only place the intermediate "
            "latents are ever exposed. There is nothing here to film: the run "
            "would produce its final image and nothing in between. This is the "
            "call surface rather than the architecture — a text-conditioned DiT "
            "exposes the callback and films exactly like a UNet."
        )
    if "prompt" not in accepted:
        raise NotSupported(
            "this pipeline is not conditioned on text — its `__call__` takes no "
            "`prompt`, and a class-conditioned model takes label ids instead. "
            "Passing a prompt to it would condition on nothing this call chose, "
            "so the frames would be of some other image entirely."
        )


# ------------------------------------------------------------ the decoded run


def filmstrip(
    pipe,
    prompt: str,
    *,
    seed: int | None,
    steps: int = 20,
    every: int | None = None,
    at=None,
    include_final: bool = True,
    frame_pixels: int = DEFAULT_FRAME_PIXELS,
    height: int | None = None,
    width: int | None = None,
    on_step=None,
    # Same reason as `trace` above: the pipeline's own `name_or_path` is a
    # local snapshot directory, and a filmstrip is exactly the kind of result
    # somebody shares.
    model_name: str = "",
) -> Filmstrip:
    """Run the pipeline once and decode the steps the caller named.

    The opt-in companion to `trace()`: same one real pipeline call, same
    step-end callback, same released-in-a-`finally` memory — and this one
    decodes, on purpose, a subset chosen by the caller.

    `seed` is keyword-only with NO DEFAULT for the same reason it is there: a
    strip nobody can reproduce is a strip nobody can compare against another.

    Latents are held for the selected steps only, on the host, and decoded AFTER
    the run rather than inside the callback — a decode wedged between two
    denoising steps competes with the denoiser's own activations for the card,
    and an offloaded VAE may not even be resident yet.
    """
    import torch

    if _denoiser_of(pipe) is None:
        raise NotSupported(
            "this pipeline has no `unet` or `transformer`, so there is no "
            "denoiser whose steps could be filmed."
        )
    _filmstrip_steps(steps)
    selection = select_steps(steps, every=every, at=at, include_final=include_final)
    frame_pixels = check_frame_pixels(frame_pixels)
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise BadRequest("a seed must be a whole number, or None for unseeded.")

    _check_call_surface(pipe)

    vae = getattr(pipe, "vae", None)
    if vae is None or not callable(getattr(vae, "decode", None)):
        raise NotSupported(
            "this pipeline has no VAE to decode with, so there is no picture to "
            "make of an intermediate latent. A pixel-space diffusion model "
            "denoises the image directly and would need a different reader; "
            "`trace()` still measures this run's steps without decoding them."
        )

    generator = None
    if seed is not None:
        # CPU generator, moved by the pipeline — the same choice `trace` and
        # `image_attention.capture` make, and for the same reason: a seed that
        # means different things on different machines is not a seed.
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

    wanted = set(selection)
    store = _Frames(wanted=wanted)

    def _tick(_pipe, step_index, timestep, kwargs):
        index = int(step_index)
        store.seen(index)
        if index in wanted:
            latents = kwargs.get("latents")
            if latents is None:
                raise NotSupported(
                    "this pipeline does not hand its latents to "
                    "`callback_on_step_end`, so there is nothing here to "
                    "decode. Reconstructing them from the denoiser's input "
                    "would mean undoing whatever scaling the scheduler applied, "
                    "and a picture made of the wrong tensor still looks like a "
                    "picture."
                )
            store.add(index, float(timestep), latents)
        if on_step is not None:
            on_step(index, steps)
        return kwargs

    accepted = _call_parameters(pipe)
    call_kwargs: dict = {}
    if height is not None:
        call_kwargs["height"] = int(height)
    if width is not None:
        call_kwargs["width"] = int(width)
    skipped_final = False
    if "output_type" in accepted:
        # The pipeline's own final decode, skipped: the last selected step is
        # decoded here from the same latent, and paying for both would be one
        # decode this strip did not need and would still have had to report.
        call_kwargs["output_type"] = "latent"
        skipped_final = True

    scaling, shift = _vae_scaling(vae)
    peak = _Peak(pipe)
    decodes = _Decodes(vae)
    frames: list[Frame] = []

    peak.reset()
    decodes.install()
    try:
        with torch.inference_mode():
            pipe(
                prompt,
                num_inference_steps=steps,
                generator=generator,
                callback_on_step_end=_tick,
                **call_kwargs,
            )
            if store.steps_seen == 0:
                raise NotSupported(
                    "the run finished without handing over a single step, so "
                    "there was never a latent to decode. This pipeline may "
                    "drive its scheduler somewhere the step-end callback does "
                    "not reach, which is a gap in coverage rather than a "
                    "property of the model."
                )
            if not store.indices:
                raise NotSupported(
                    f"the run handed over {store.steps_seen} step(s) and none "
                    f"of them were the {len(wanted)} this was asked to decode. "
                    f"A strip of nothing is not a cheaper strip — ask for steps "
                    f"inside the range this run actually reports."
                )
            frames = _decode_frames(
                pipe, store, frame_pixels=frame_pixels, scaling=scaling, shift=shift
            )
    finally:
        # Always, on both paths. The wrapper is an attribute on somebody else's
        # VAE and a strip that raised must leave the pipeline exactly as usable
        # as it found it; the held latents go for the reason `trace` gives — a
        # traceback keeps every frame's locals alive for as long as the
        # exception is being handled, and "collected eventually" is not a bound.
        decodes.remove()
        store.release()

    reached = set(store.indices)
    return Filmstrip(
        prompt=prompt,
        seed=seed,
        model=public_model_name(pipe, model_name),
        scheduler=type(getattr(pipe, "scheduler", None)).__name__,
        frames=frames,
        steps_requested=steps,
        steps_run=store.steps_seen,
        decoded_steps=list(store.indices),
        skipped_steps=[i for i in store.seen_indices if i not in reached],
        steps_never_reached=sorted(wanted - reached),
        selection=_selection(every, at, include_final),
        latent_shape=store.shape,
        frame_pixels=frame_pixels,
        vae_decodes=decodes.count,
        vae_decodes_for_frames=len(frames),
        final_decode_skipped=skipped_final,
        latent_scaling=scaling,
        latent_shift=shift,
        host_latent_bytes=store.peak_bytes,
        peak_device_bytes=peak.read(),
        peak_device=peak.device_name,
        peak_source=peak.source,
        peak_unmeasured=peak.unmeasured,
    )


class _Frames:
    """The SELECTED latents, held on the host until they are decoded.

    Only the steps the caller named are kept and every other one is handed over
    and dropped, so this holds a dozen latents rather than a run's worth. The
    bound is checked against the FIRST real latent exactly as `_Trace` checks
    its own, so a shape this cannot hold is refused at the first selected step
    rather than at the last.
    """

    def __init__(self, *, wanted: set[int]) -> None:
        self.wanted = wanted
        self.indices: list[int] = []
        self.timesteps: list[float] = []
        self.latent_rms: list[float] = []
        self.latents: list = []
        self.shape: tuple[int, ...] = ()
        # Every step the run handed over, decoded or not. Kept as the indices
        # THIS RUN reported rather than as a count, so "which steps were
        # skipped" is answered from what happened instead of from `range(n)` —
        # a pipeline that numbers its steps differently would otherwise have its
        # gaps described with numbers it never used.
        self.seen_indices: list[int] = []
        self.bytes_held = 0
        # The most ever held at once. `bytes_held` is zero by the time anybody
        # reads this object, and a memory claim of zero is not a memory claim.
        self.peak_bytes = 0

    @property
    def steps_seen(self) -> int:
        return len(self.seen_indices)

    def seen(self, index: int) -> None:
        """One more step handed over, decoded or not."""
        self.seen_indices.append(index)

    def add(self, index: int, timestep: float, latent) -> None:
        import torch

        ndim = int(getattr(latent, "ndim", 0))
        if ndim == 3:
            raise NotSupported(
                "this pipeline carries its latents packed into a sequence "
                "(batch, tokens, channels) rather than as a picture, so there "
                "is no height and width here to decode. Unpacking them needs "
                "the pipeline's own arithmetic and the size it chose, and a "
                "wrong guess decodes to a scrambled picture that still looks "
                "like a frame — which is worse than no frame. `trace()` "
                "measures this run's steps without decoding them."
            )
        if ndim == 5:
            raise NotSupported(
                "this pipeline's latent has a time axis (batch, channels, "
                "frames, height, width), so 'the picture at step k' is a whole "
                "clip rather than a picture and there is no single frame to "
                "decode. `trace()` measures how far that latent moved per step, "
                "which is the question that still has one answer here."
            )
        if ndim != 4:
            raise NotSupported(
                f"this pipeline's latent has {ndim} dimension(s) where a "
                f"decodable one has four (batch, channels, height, width), so "
                f"nothing here can say what it is looking at."
            )
        if int(latent.shape[0]) != 1:
            raise NotSupported(
                f"this run is denoising {int(latent.shape[0])} images at once "
                f"and their latents travel stacked in one tensor, so every step "
                f"would decode to {int(latent.shape[0])} pictures. A filmstrip "
                f"has one axis, not two — generate one image at a time."
            )

        # `copy=True` unconditionally, for the hazard `_Trace` documents at
        # length: `.to()` hands back the SAME storage when nothing needs
        # converting, and a scheduler that writes its result in place would then
        # rewrite every frame already held — a strip where every picture is the
        # last one, which looks like a model that decided everything at step 0.
        held = latent.detach().to("cpu", torch.float32, copy=True)
        nbytes = int(held.numel() * held.element_size())

        if not self.indices:
            self.shape = tuple(int(v) for v in held.shape[1:])
            projected = nbytes * max(1, len(self.wanted))
            if projected > MAX_TRACE_BYTES:
                raise BadRequest(
                    f"a {self.shape} latent is {nbytes / 1024 / 1024:,.2f} MiB, "
                    f"so holding one for each of the {len(self.wanted)} frames "
                    f"asked for would need {projected / 1024 / 1024:,.0f} MiB — "
                    f"past the {MAX_TRACE_BYTES / 1024 / 1024:,.0f} MiB this "
                    f"will hold. Ask for fewer frames, or generate at a smaller "
                    f"height and width. `filmstrip_plan()` prices this before "
                    f"the run rather than one step into it."
                )

        self.indices.append(index)
        self.timesteps.append(timestep)
        self.latents.append(held)
        self.bytes_held += nbytes
        self.peak_bytes = max(self.peak_bytes, self.bytes_held)
        # Measured here rather than after decoding, and deliberately: a latent
        # that has gone non-finite refuses HERE, before it is decoded into a
        # picture of noise that a reader would take for the model's own.
        self.latent_rms.append(round(_rms(held.numpy(), where=index), 6))

    def take(self, position: int):
        """Hand over one held latent and drop this object's reference to it."""
        latent = self.latents[position]
        self.latents[position] = None
        if latent is not None:
            self.bytes_held -= int(latent.numel() * latent.element_size())
        return latent

    def release(self) -> None:
        """Drop the held latents. Idempotent, and safe to call mid-failure."""
        self.latents = []
        self.bytes_held = 0


class _Decodes:
    """Counts what `pipe.vae.decode` was really called, then puts it back.

    Measured rather than intended, which is the only version of this number
    worth publishing beside a sibling that claims a zero — `trace`'s test counts
    decodes exactly this way, and a claim that is only checkable from the test
    suite is not checkable by the reader.
    """

    def __init__(self, vae) -> None:
        self.vae = vae
        self.count = 0
        self._installed = False
        self._had_own = False
        self._previous = None

    def install(self) -> None:
        # Whether the VAE already carried its own `decode` attribute, and what
        # it was. Something else may already be wrapping this method — a
        # profiler, a test counting decodes — and putting the state back means
        # putting THAT back, not deleting it.
        self._had_own = "decode" in vars(self.vae)
        self._previous = vars(self.vae).get("decode")
        inner = self.vae.decode

        def counting(*args, **kwargs):
            self.count += 1
            return inner(*args, **kwargs)

        self.vae.decode = counting
        self._installed = True

    def remove(self) -> None:
        """Exactly the state that was there before, on every path.

        `del` when the VAE carried no `decode` of its own: re-assigning the
        bound method back would leave an instance attribute shadowing the
        class's own method on somebody else's pipeline forever — the hazard
        `test_image_steps.py` already names where it counts decodes.
        """
        if not self._installed:
            return
        self._installed = False
        if self._had_own:
            self.vae.decode = self._previous
            return
        try:
            del self.vae.decode
        except AttributeError:
            # Something else removed it first. There is nothing to restore and
            # nothing to report: the count is still the count.
            pass


class _Peak:
    """The allocator's high-water mark across one call, or why there is none.

    `budget.py` reads the same three counters for the text side; they are read
    again here rather than imported because that module answers a different
    question (what would this analysis NEED) through private helpers, and a
    private name that only happens to fit today is not an interface.

    Every failure path lands on `unmeasured` with a sentence. A peak that could
    not be read is null, never 0 — "the decodes cost nothing" is a claim, and it
    would be a false one.
    """

    def __init__(self, pipe) -> None:
        self.backend = None
        self.source: str | None = None
        self.device = None
        self.device_name: str | None = None
        self.unmeasured: str | None = None

        device = _device_of(pipe)
        kind = getattr(device, "type", None)
        if kind is None:
            self.unmeasured = (
                "the device this pipeline sits on could not be read, so no "
                "allocator counter could be chosen."
            )
            return
        self.device = device
        self.device_name = str(device)
        if kind in ("cuda", "xpu"):
            import torch

            backend = getattr(torch, kind, None)
            if backend is not None and hasattr(backend, "max_memory_allocated"):
                self.backend = backend
                self.source = f"torch.{kind}.max_memory_allocated"
                return
        self.unmeasured = (
            f"this pipeline is on {device}, where torch publishes no allocator "
            f"high-water mark."
        )

    def reset(self) -> None:
        if self.backend is None:
            return
        fn = getattr(self.backend, "reset_peak_memory_stats", None)
        try:
            fn(self.device)
        except Exception:
            # A peak read from a counter that would not zero includes whatever
            # this process did before the call, which is not this run's number.
            self.backend = None
            self.unmeasured = (
                "this machine's allocator would not reset its high-water mark, "
                "so anything read from it would include memory this call never "
                "asked for."
            )

    def read(self) -> int | None:
        if self.backend is None:
            return None
        try:
            return int(self.backend.max_memory_allocated(self.device))
        except Exception:
            self.unmeasured = (
                "this machine's allocator did not answer for its high-water "
                "mark after the run."
            )
            return None


def _device_of(pipe):
    """The device the denoiser's weights are on, or None.

    Read off a real parameter rather than off `pipe.device`, which some
    pipelines report as the CPU while their weights are on an accelerator under
    an offload hook.
    """
    denoiser = _denoiser_of(pipe)
    try:
        return next(denoiser.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        pass
    device = getattr(pipe, "device", None)
    return device if getattr(device, "type", None) else None


def _vae_scaling(vae) -> tuple[float | None, float | None]:
    """This VAE's own latent scaling and shift, or `None` for either.

    Read off the config rather than assumed: 0.18215 is Stable Diffusion's and
    hard-coding it would silently mis-decode every other VAE. `None` means the
    config did not carry one, which the response says out loud rather than
    quietly substituting a 1.
    """
    config = getattr(vae, "config", None)
    return (
        _finite(getattr(config, "scaling_factor", None)),
        _finite(getattr(config, "shift_factor", None)),
    )


def _finite(value) -> float | None:
    """A finite float, or None. `True` is not a scaling factor."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _decode_frames(pipe, store: _Frames, *, frame_pixels, scaling, shift):
    """Decode the held latents one at a time, releasing each as it goes.

    One latent on the accelerator and one decoded picture in memory at a time.
    That is the whole reason the selected latents were parked on the host: a
    dozen decoder outputs resident at once is the shape that does not fit beside
    a pipeline on 8 GB, and what survives each iteration is kilobytes of PNG.
    """
    frames: list[Frame] = []
    for position, index in enumerate(store.indices):
        latent = store.take(position)
        try:
            png, emitted, decoded = _decode_frame(
                pipe, latent, frame_pixels=frame_pixels, scaling=scaling, shift=shift
            )
        except (Refusal, BadRequest):
            raise
        except Exception as err:
            # The library's own message never reaches the reader — only the type
            # name, which is the one part that is ours to quote. Half a strip is
            # not returned as a whole one: the frames that did decode would look
            # exactly like a selection somebody asked for.
            raise Refusal(
                f"the decoder failed at step {index}, after "
                f"{len(frames)} frame(s) had decoded ({type(err).__name__}). On "
                f"a card this size a full-resolution decode is usually what "
                f"runs out: ask for fewer frames, generate at a smaller height "
                f"and width, or turn on the decoder's own tiling before calling "
                f"— `frame_pixels` bounds only what is EMITTED and does not make "
                f"the decode itself any smaller."
            ) from err
        finally:
            # The traceback would otherwise hold this latent alive through every
            # handler above, on the exact path where memory is already short.
            del latent

        frames.append(
            Frame(
                step=index,
                timestep=store.timesteps[position],
                png=png,
                width=emitted[0],
                height=emitted[1],
                decoded_width=decoded[0],
                decoded_height=decoded[1],
                latent_rms=store.latent_rms[position],
            )
        )
    return frames


def _decode_frame(pipe, latent, *, frame_pixels, scaling, shift):
    """One latent through this pipeline's own VAE, out as PNG bytes."""
    import torch

    vae = pipe.vae
    device, dtype = _vae_target(vae)
    z = latent if device is None and dtype is None else latent.to(device, dtype)
    # diffusers' own order: divide by the scaling factor, then add the shift.
    # A scaling factor of 0 is not a scaling factor and is left alone rather
    # than dividing by it.
    if scaling:
        z = z / scaling
    if shift is not None:
        z = z + shift

    decoded = vae.decode(z)
    image = getattr(decoded, "sample", None)
    if image is None:
        image = (
            decoded[0] if isinstance(decoded, (tuple, list)) and decoded else decoded
        )
    if int(getattr(image, "ndim", 0)) != 4:
        raise NotSupported(
            f"this VAE returned {int(getattr(image, 'ndim', 0))} dimension(s) "
            f"where a decoded picture is (batch, channels, height, width). "
            f"Nothing here will guess which axis is which and call the result a "
            f"frame."
        )

    # (x / 2 + 0.5), clamped — the same denormalisation diffusers' own image
    # processor applies to a pipeline's final output, written out here rather
    # than borrowed so one arithmetic answers for every family. A VAE configured
    # to skip it would look wrong, which the response says is possible.
    pixels = (image.to(torch.float32) / 2 + 0.5).clamp(0, 1)
    array = pixels[0].permute(1, 2, 0).mul(255).round().to(torch.uint8).cpu().numpy()
    # Dropped before the PNG is made, not after: this is the largest thing in
    # memory and the encoder is about to allocate too.
    del pixels, image, decoded, z
    return _png(array, frame_pixels=frame_pixels)


def _vae_target(vae):
    """(device, dtype) to hand the decoder, or (None, None) to hand it as-is.

    Read off the decoder's own first parameter rather than assumed. A VAE whose
    weights sit on `meta` is under an accelerate offload hook that moves its
    inputs itself, so the device is left alone — a tensor pushed to `meta` would
    decode nothing at all.
    """
    try:
        param = next(vae.parameters())
    except (AttributeError, StopIteration, TypeError):
        return None, None
    device = getattr(param, "device", None)
    dtype = getattr(param, "dtype", None)
    if device is None or getattr(device, "type", "") == "meta":
        return None, dtype
    return device, dtype


def _png(array, *, frame_pixels: int):
    """A uint8 (height, width, channels) array as PNG bytes.

    Returns `(png, (width, height) emitted, (width, height) decoded)` so the
    response can carry both. The resize is LANCZOS and it happens after the
    decode, which is why it saves payload and not memory.
    """
    try:
        from PIL import Image
    except ImportError as err:
        raise Refusal(
            "Writing a frame as PNG needs Pillow, which normally arrives with "
            "diffusers. Install it with `pip install pillow` and ask again."
        ) from err

    if array.ndim != 3:
        raise NotSupported(
            f"this decoder produced a {array.ndim}-dimensional picture, and a "
            f"frame here is (height, width, channels)."
        )
    channels = int(array.shape[2])
    if channels == 1:
        picture = Image.fromarray(array[:, :, 0], mode="L").convert("RGB")
    elif channels == 3:
        picture = Image.fromarray(array, mode="RGB")
    else:
        raise NotSupported(
            f"this decoder produced {channels} channels per pixel, and a frame "
            f"here is greyscale or RGB. Picking three of them would be this "
            f"module deciding what the other channels meant."
        )

    decoded = (int(picture.width), int(picture.height))
    longest = max(decoded)
    if longest > frame_pixels:
        scale = frame_pixels / longest
        size = (
            max(1, round(picture.width * scale)),
            max(1, round(picture.height * scale)),
        )
        resample = getattr(Image, "Resampling", Image).LANCZOS
        resized = picture.resize(size, resample)
        picture.close()
        picture = resized

    buffer = io.BytesIO()
    picture.save(buffer, format="PNG", optimize=True)
    emitted = (int(picture.width), int(picture.height))
    picture.close()
    return buffer.getvalue(), emitted, decoded
