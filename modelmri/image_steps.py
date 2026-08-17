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
for: a latent with a time axis, say (16, 13, 60, 104), is 5.2 MiB a step and
260 MiB over 50 steps. Past the limit this refuses BEFORE the second step, with
the arithmetic and the two parameters that would fix it, and `plan()` prices the
same thing without running anything at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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
                f" At that step the latent was still {at_commit:,.4f} RMS from "
                f"where it finished, against {started:,.4f} at the first step "
                f"measured."
                if at_commit is not None and started is not None
                else ""
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
            f", at the prompt given. {seeded}\n\n"
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


def _denoiser_of(pipe):
    """The module that denoises, whatever this pipeline calls it.

    Deliberately the same three names `image_attention._denoiser_of` looks for.
    Copied rather than imported: reaching across modules for a private name to
    save five lines couples two files that only happen to agree today.
    """
    for name in ("unet", "transformer", "denoiser"):
        found = getattr(pipe, name, None)
        if found is not None:
            return found
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
        not in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL, parameter.POSITIONAL_ONLY)
    }


# -------------------------------------------------------------------- the run


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

    try:
        with torch.inference_mode():
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
        model=getattr(pipe, "name_or_path", "") or "",
        scheduler=type(getattr(pipe, "scheduler", None)).__name__,
        steps=rows,
        steps_requested=steps,
        latent_shape=store.shape,
        threshold=threshold,
        total_change=round(total, 6),
        vae_decodes=decodes,
        bytes_held=store.bytes_held,
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
    latents = []
    final = None
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
