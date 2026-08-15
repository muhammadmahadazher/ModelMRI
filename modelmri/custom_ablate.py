"""What matters in the network YOU trained.

`custom.py` maps one forward pass: shapes, activation statistics, dead units,
anything non-finite. All of it descriptive. It can tell you a layer's output
has collapsed and it cannot tell you whether the answer would change if the
layer were not there.

This asks the causal question, on the one surface in this project that nothing
else in the category will ever cover. Every platform ModelMRI competes with is
a fixed catalogue of transformers; none of them will look at the CNN somebody
trained last week. So the standard here is the same one the LLM panels are
held to, not a lower one because the model is small:

    THE CONTROL IS NOT OPTIONAL, AND IT IS NOT THE ONE IN patch.py. The draw
    count and the seed are copied from that file; the construction is not, and
    copying it wholesale produced a null that nothing could ever beat.

    patch.py replaces an activation with a random vector OF THE SAME NORM,
    which is apples to apples there because its treatment is also a full
    replacement by an activation of comparable norm. Here the treatment is
    replacement by the MEAN -- the centre of the distribution, and therefore
    the gentlest same-norm replacement there is, while a random vector of that
    norm is nearly orthogonal to the data and among the harshest. MEASURED on
    a 20->64->3 net over 64 samples: the mean edit moves the activation by
    2.6735 and the same-norm random control moves it by 3.7624, so the control
    was a 1.41x larger intervention than the thing it was the null for. Every
    site lost to its own control, and a null that cannot be beaten is as
    useless as no null.

    So the control here matches THE SIZE OF THE EDIT rather than the norm of
    the replacement: the same distance the mean moves this sample's
    activation, in a random direction. That asks the question a reader thinks
    is being asked -- is it this layer and this direction, or would any
    perturbation of that size in that place do as much?

    THE METRIC COMES FROM THE ADAPTER, NOT FROM A DEFAULT. KL is right for a
    classifier and meaningless for a regressor; both still produce a plausible
    ordering, which is exactly why a default would be dangerous rather than
    convenient. An adapter that does not declare `TASK` is refused.

    A BATCH MEAN OVER ONE SAMPLE IS THE SAMPLE. Mean ablation replaces a
    layer's output with its average over your inputs, so with one input it
    replaces the layer with itself and every score is zero. `MIN_SAMPLES` is
    enforced rather than documented.

WHAT IT MEASURES

    layer ablation   replace one module's output with its mean over your
                     samples, and see how far the model's answer moves
    occlusion        replace one input feature, or one patch of an image,
                     with its mean over your samples, and the same

Both report in the task's own unit, and both put every score next to a null.
The two nulls are NOT the same construction, because the two sweeps ask
different questions and one control copied across both was measurably wrong
for one of them -- see `sweep_inputs`.

ONE MORE THING MEAN ABLATION CANNOT DO, and it is worth knowing before reading
a layer sweep. In a purely sequential model, replacing ANY module's output
with a constant makes every module after it constant too, so the final output
is the same constant wherever the chain was cut. MEASURED: ablating `act1` and
ablating `fc2` of a 20->64->32->3 net produced final logits differing by
1.19e-07, with zero variance across samples in both cases. The layer sweep
therefore separates what is WIRED IN from what is not -- a dead branch scores
exactly 0.0 while the live path scores 0.649 -- and it cannot rank depth along
a chain. `degenerate` says so when the numbers show it.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .errors import BadRequest

# The draw count and the seed are verbatim from patch.py, so a control is a
# measurement somebody else can repeat rather than a number that moves every
# time the page is refreshed. The CONSTRUCTION is deliberately not verbatim --
# see the module docstring, and `_jitter_like` below.
CONTROL_DRAWS = 8
CONTROL_SEED = 0

# Mean ablation over one sample replaces a layer with itself. Two is not
# better in any way that matters. Eight is the smallest batch where the mean
# is about the distribution rather than about one point, and it is the same
# floor `probe.py` sets on a class for the same reason.
MIN_SAMPLES = 8

# A model with a thousand leaves produces a table nobody reads and a sweep
# nobody waits for. `custom.MAX_LAYERS` caps the map at 512; this caps the
# SWEEP, which costs a forward pass each, far lower — and says what it cut.
MAX_SITES = 128

# How many of the strongest sites get the eight control draws. Same shape as
# `patch.py`'s `MAX_CONTROLLED_EDGES`: controlling everything multiplies the
# cost by nine, and a site that did not place is not one anybody is about to
# call hot.
MAX_CONTROLLED = 12

# Occluding one pixel of a 224x224 image measures nothing and costs a forward
# pass. Patches are the unit for anything with spatial dimensions, and this is
# the default grid along each spatial axis.
DEFAULT_PATCH_GRID = 8

# The tasks this knows how to score, and what it reports for each.
TASKS = {
    "classification": "KL divergence in nats, over the softmax of your output",
    "regression": (
        "shift in units of your model's own output spread across these samples"
    ),
}


class AblationError(BadRequest):
    """This measurement cannot be taken honestly, and we say why."""


@dataclass
class Site:
    """One thing that was ablated, and what happened to the answer."""

    name: str
    kind: str
    # The task's unit. Nats for a classifier, output-sigma for a regressor.
    effect: float
    # The strongest of `CONTROL_DRAWS` random edits OF THE SAME SIZE at this
    # same site. None when this site was not among the strongest and therefore
    # was never controlled -- NOT 0.0, which would read as "random edits here
    # do nothing" when nothing was tried.
    control_max: float | None = None
    control_draws: int = 0
    # True only when the real edit beat every control draw. None when it was
    # not tested.
    beats_control: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Ablation:
    kind: str
    task: str
    unit: str
    sites: list[Site] = field(default_factory=list)
    n_sites: int = 0
    n_controlled: int = 0
    n_samples: int = 0
    passes: int = 0
    seconds: float = 0.0
    truncated: int = 0
    # The largest effect any control draw reached anywhere. A reader comparing
    # two sites needs the scale that noise reaches on THIS model, once, at the
    # top -- not only per row.
    control_ceiling: float | None = None
    # How many sites clear by chance alone.
    #
    # Each tested site is compared against the strongest of `draws` control
    # draws, so under a null where every site is equivalent the real edit wins
    # with probability 1/(draws+1) -- 1 in 9 at the default. Sweep twelve
    # sites and 1.3 of them clear having done nothing.
    #
    # MEASURED on a trained net whose label depends only on features 0 and 1:
    # five of twenty features cleared, and the two real ones cleared by 590x
    # and 305x while the other three sat just above a control drawn from
    # equally unimportant regions. Without this line those three read exactly
    # like the first two.
    expected_false_positives: float = 0.0

    def to_dict(self) -> dict:
        out = asdict(self)
        out["means"] = self.means()
        return out

    def means(self) -> str:
        cleared = [s for s in self.sites if s.beats_control]
        tested = [s for s in self.sites if s.beats_control is not None]
        what = "module" if self.kind == "layers" else "input region"
        # Two sweeps, two different nulls, because they answer two different
        # questions -- and a summary that named one construction for both
        # would be describing a control half its readers did not get.
        null = (
            "a random edit of the same size at the same module"
            if self.kind == "layers"
            else "occluding a different region the same way"
        )
        parts = [
            f"Each {what} was replaced by its mean over your {self.n_samples} "
            f"samples and the model re-run, {self.passes} forward passes in "
            f"total. Scores are {self.unit}."
        ]
        if not tested:
            parts.append(
                "NOTHING WAS TESTED AGAINST CHANCE on this run, so every "
                "number above is an effect size with no null beside it."
            )
        elif not cleared:
            parts.append(
                f"NONE of the {len(tested)} strongest sites beat its own "
                f"control: {null} did as much or more. On this evidence "
                f"nothing here is distinguished from that, which on an "
                f"untrained or fragile model is the honest answer rather than "
                f"a failure of the sweep."
            )
        else:
            names = ", ".join(s.name for s in cleared[:4])
            if len(cleared) <= self.expected_false_positives:
                parts.append(
                    f"{len(cleared)} of the {len(tested)} sites tested beat "
                    f"their control, and about "
                    f"{self.expected_false_positives:.1f} would do that by "
                    f"chance alone — each site is compared against the "
                    f"strongest of its draws, so the real edit wins one time "
                    f"in {(self.sites[0].control_draws or 1) + 1} having done "
                    f"nothing. THIS SWEEP FOUND NOTHING ABOVE THAT."
                )
                parts.append(
                    "MEAN ABLATION IS OFF-DISTRIBUTION. The mean of your "
                    "samples is not a value this layer ever produces, so a "
                    "large effect can mean the layer matters or that the "
                    "model has never seen an input like the one you just "
                    "built. That is why the control is here."
                )
                return " ".join(parts)
            parts.append(
                f"{len(cleared)} of the {len(tested)} sites tested against "
                f"chance beat every one of their control draws — {null} "
                f"({names}). The other {self.n_sites - len(tested)} sites "
                f"carry a score and NO verdict — they were not tested. "
                f"About {self.expected_false_positives:.1f} of the "
                f"{len(tested)} would clear by chance, so read the margin "
                f"rather than the flag: a site clearing by a hair is what "
                f"that number looks like."
            )
        if self.truncated:
            parts.append(
                f"{self.truncated} further sites were not swept: this runs one "
                f"forward pass each and the cap is {MAX_SITES}. They are "
                f"missing from the list above, not measured as zero."
            )
        parts.append(
            "MEAN ABLATION IS OFF-DISTRIBUTION. The mean of your samples is "
            "not a value this layer ever produces, so a large effect can mean "
            "the layer matters or that the model has never seen an input like "
            "the one you just built. That is why the control is here."
        )
        return " ".join(parts)


# ------------------------------------------------------------ the contract


# Triple-quoted with REAL line breaks, not "\n" escapes.
#
# These are code samples the reader is meant to copy, so the newlines have to
# survive into the message. Written with escapes they went through two rounds
# of tooling and arrived first as a literal backslash-n printed on screen and
# then, after a careless fix, as a line continuation Python swallows entirely
# — a hint that read "beside load():    TASK = ..." on one line. A string with
# the line breaks already in it cannot be got wrong by anything downstream.
TASK_HINT = """add a module-level TASK to your adapter, beside load():

    TASK = "classification"   # or "regression"

KL over a softmax is the right measure for a classifier and meaningless for a \
regressor, and both still produce a plausible ordering — which is why ModelMRI \
will not pick one for you."""

SAMPLES_HINT = """add sample_inputs() to your adapter, beside load():

    def sample_inputs():
        # a batch of REAL inputs, or a loader you already have
        return torch.stack([x for x, _ in list(my_dataset)[:64]])

Mean ablation replaces a layer's output with its average over your inputs, so \
the average has to be over inputs your model actually sees. Random noise would \
make every score a statement about noise."""


def read_task(module: Any) -> str:
    """The adapter's declared task, or a refusal that says how to declare it.

    NO DEFAULT. A classifier scored as a regressor and a regressor scored as a
    classifier both produce a ranked list that looks exactly like a finding,
    and the reader has no way to tell from the output which happened.
    """
    task = getattr(module, "TASK", None)
    if task is None:
        raise AblationError(
            "this adapter does not say what kind of model it is, so there is "
            "no right way to measure how far its answer moved. " + TASK_HINT
        )
    if not isinstance(task, str) or task.strip().lower() not in TASKS:
        known = ", ".join(sorted(TASKS))
        raise AblationError(
            f"TASK is {task!r}, which is not one of: {known}. " + TASK_HINT
        )
    return task.strip().lower()


def read_samples(module: Any, *, min_samples: int = MIN_SAMPLES):
    """The adapter's sample batch, or a refusal that says how to provide one."""
    import torch

    maker = getattr(module, "sample_inputs", None)
    if maker is None:
        raise AblationError(
            "this adapter has no sample_inputs(), so there is nothing to take "
            "a mean over. " + SAMPLES_HINT
        )
    if not callable(maker):
        raise AblationError("sample_inputs is not callable. " + SAMPLES_HINT)
    try:
        samples = maker()
    except Exception as err:
        # AdapterError, NOT AblationError, and the distinction is load-bearing.
        #
        # `AblationError` is one of four types `runtime.py` re-publishes with
        # `raise Refusal(str(err))`, so the invariant on it is that it is
        # ALWAYS built from authored text -- and a leak test walks every raise
        # site to enforce that. This message relays the reader's own adapter
        # exception, which is the entire content of "why did my sweep not
        # run", so it belongs in the type `custom.py` already uses for exactly
        # that: `load()` raising, `example_input()` raising, and now this.
        #
        # Caught by that test rather than by review.
        from .custom import AdapterError

        raise AdapterError(
            f"sample_inputs() raised {type(err).__name__}: {err}"  # leak-ok: the reader's own adapter code
        ) from err

    if isinstance(samples, (list, tuple)):
        try:
            samples = torch.stack([torch.as_tensor(x) for x in samples])
        except Exception as err:
            raise AblationError(
                f"sample_inputs() returned a sequence ModelMRI could not "
                f"stack into a batch ({type(err).__name__}). Return one "
                f"tensor whose first dimension is the batch."
            ) from err
    if not isinstance(samples, torch.Tensor):
        raise AblationError(
            f"sample_inputs() returned {type(samples).__name__}, not a tensor "
            "or a list of them. " + SAMPLES_HINT
        )
    if samples.ndim < 1 or int(samples.shape[0]) < min_samples:
        have = int(samples.shape[0]) if samples.ndim else 0
        raise AblationError(
            f"sample_inputs() returned {have} sample(s) and mean ablation "
            f"needs at least {min_samples}. The mean of one sample IS that "
            f"sample, so every layer would be replaced by itself and every "
            f"score would be zero — a clean-looking result from a measurement "
            f"that did not happen."
        )
    return samples


# --------------------------------------------------------------- the metric


def _outputs(model, samples, batch: int = 64):
    """The model's answer for every sample, as one [N, ...] tensor."""
    import torch

    chunks = []
    with torch.no_grad():
        for start in range(0, int(samples.shape[0]), batch):
            piece = samples[start : start + batch]
            out = model(piece)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if not isinstance(out, torch.Tensor):
                raise AblationError(
                    f"this model returned {type(out).__name__} rather than a "
                    "tensor, so there is nothing to measure a shift in."
                )
            chunks.append(out.detach().float().reshape(int(piece.shape[0]), -1))
    return torch.cat(chunks, dim=0)


def _scale(task: str, base) -> float:
    """The denominator a regression shift is reported in.

    The spread of THIS model's own answers across YOUR samples. A raw L2 shift
    is in whatever units the model happens to emit — a model predicting house
    prices moves by thousands and one predicting probabilities by hundredths,
    and neither number means anything without knowing which. Dividing by the
    spread makes the score "this many times the variation the model produces
    on its own", which is comparable across sites, models and runs.
    """
    import torch

    if task != "regression":
        return 1.0
    centred = base - base.mean(dim=0, keepdim=True)
    spread = float(torch.linalg.vector_norm(centred, dim=1).pow(2).mean().sqrt())
    if not math.isfinite(spread) or spread <= 0.0:
        raise AblationError(
            "this model returns the same answer for every one of your "
            "samples, so there is no spread to measure a shift against. Give "
            "it inputs it responds to differently, or the ordering below "
            "would be division by nothing."
        )
    return spread


def _effect(task: str, base, moved, scale: float) -> float:
    """How far the answer moved, in the task's own unit."""
    import torch

    if task == "classification":
        # KL(base || moved), meaned over the batch. `log_softmax` on both
        # sides rather than `softmax` then `log`: the fused form is what keeps
        # a confident row from underflowing to log(0) and reporting inf for a
        # model that is merely certain.
        p = torch.log_softmax(base, dim=-1)
        q = torch.log_softmax(moved, dim=-1)
        kl = (p.exp() * (p - q)).sum(dim=-1)
        return float(kl.mean())
    shift = torch.linalg.vector_norm(moved - base, dim=1)
    return float(shift.mean()) / scale


# ------------------------------------------------------------ layer sweep


def _mean_outputs(model, module, samples, batch: int = 64):
    """One module's output, averaged over the whole sample batch.

    Averaged over the BATCH DIMENSION ONLY: a convolution's mean over its
    spatial dimensions as well would replace the feature map with a constant
    image, which is a much larger intervention than the one being claimed.
    """
    import torch

    total = None
    count = 0
    captured: list = []

    def hook(_m, _inp, output):
        t = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(t, torch.Tensor):
            captured.append(t.detach())

    handle = module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for start in range(0, int(samples.shape[0]), batch):
                captured.clear()
                model(samples[start : start + batch])
                if not captured:
                    return None
                # A module that fired more than once in a pass is averaged
                # over its firings too: it is one module, and the ablation
                # replaces it at every call.
                for t in captured:
                    summed = t.sum(dim=0)
                    total = summed if total is None else total + summed
                    count += int(t.shape[0])
    finally:
        handle.remove()
    if total is None or count == 0:
        return None
    return total / count


def _jitter_like(module, mean, generator):
    """Hook that moves the output as FAR as the mean would, in a random direction.

    The null for mean ablation, and the reason it is not the one in patch.py.
    Per sample: take how far this sample's activation sits from the batch
    mean, and move it that distance somewhere random instead. Same
    intervention size, no relationship to the data -- which is exactly the
    alternative explanation a reader needs ruled out before "this layer
    matters" means anything.

    Per SAMPLE and not per batch: the distance to the mean is not the same for
    every input, and one batch-wide magnitude would over-perturb the typical
    sample and under-perturb the outlier, both in the same direction.
    """
    import torch

    def hook(_m, _inp, output):
        t = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(t, torch.Tensor):
            return output
        flat = t.reshape(int(t.shape[0]), -1)
        centre = mean.to(t.device, t.dtype).reshape(1, -1)
        distance = torch.linalg.vector_norm(flat - centre, dim=1, keepdim=True)
        noise = torch.randn(flat.shape, generator=generator).to(t.device, t.dtype)
        # clamp_min on the DIVISOR, not on the result: a zero-norm draw is
        # astronomically unlikely and exactly what would put a silent nan in
        # the one place nobody would look for it.
        unit = noise / torch.linalg.vector_norm(
            noise, dim=1, keepdim=True
        ).clamp_min(1e-12)
        swapped = (flat + unit * distance).reshape(t.shape)
        if isinstance(output, tuple):
            return (swapped,) + tuple(output[1:])
        if isinstance(output, list):
            return [swapped] + list(output[1:])
        return swapped

    return module.register_forward_hook(hook)


def _replace_with(module, value):
    """Hook that swaps a module's output for `value`, broadcast over the batch."""
    import torch

    def hook(_m, _inp, output):
        t = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(t, torch.Tensor):
            return output
        swapped = value.to(t.device, t.dtype).unsqueeze(0).expand_as(t)
        if isinstance(output, tuple):
            return (swapped,) + tuple(output[1:])
        if isinstance(output, list):
            return [swapped] + list(output[1:])
        return swapped

    return module.register_forward_hook(hook)


def sweep_layers(
    model,
    samples,
    *,
    task: str,
    max_sites: int = MAX_SITES,
    max_controlled: int = MAX_CONTROLLED,
    draws: int = CONTROL_DRAWS,
) -> Ablation:
    """Mean-ablate every leaf module and report how far the answer moved."""
    import torch

    from .custom import leaf_modules

    if task not in TASKS:
        raise AblationError(f"unknown task {task!r}")
    leaves = leaf_modules(model)
    if not leaves:
        raise AblationError(
            "this model has no leaf modules, so there is nothing to ablate."
        )
    truncated = max(0, len(leaves) - max_sites)
    leaves = leaves[:max_sites]

    started = time.perf_counter()
    was_training = model.training
    model.eval()
    passes = 0
    try:
        base = _outputs(model, samples)
        passes += 1
        scale = _scale(task, base)

        rows: list[Site] = []
        for name, module in leaves:
            mean = _mean_outputs(model, module, samples)
            passes += 1
            if mean is None:
                # A module that produced no tensor, or never fired on this
                # input. Skipped rather than scored zero: "did not run" and
                # "ran and did not matter" are different facts.
                continue
            handle = _replace_with(module, mean)
            try:
                moved = _outputs(model, samples)
                passes += 1
            finally:
                handle.remove()
            rows.append(
                Site(
                    name=name,
                    kind=type(module).__name__,
                    effect=round(_effect(task, base, moved, scale), 6),
                )
            )

        # The controls, on the strongest sites only. Nine passes each is the
        # whole cost of this feature, and a site that did not place is not one
        # anybody is about to call hot.
        rows.sort(key=lambda s: -s.effect)
        by_name = dict(leaves)
        ceiling = None
        for site in rows[:max_controlled]:
            module = by_name[site.name]
            mean = _mean_outputs(model, module, samples)
            passes += 1
            if mean is None:
                continue
            gen = torch.Generator().manual_seed(CONTROL_SEED)
            worst = 0.0
            for _ in range(max(1, draws)):
                handle = _jitter_like(module, mean, gen)
                try:
                    moved = _outputs(model, samples)
                    passes += 1
                finally:
                    handle.remove()
                worst = max(worst, _effect(task, base, moved, scale))
            site.control_max = round(worst, 6)
            site.control_draws = max(1, draws)
            site.beats_control = site.effect > worst
            ceiling = worst if ceiling is None else max(ceiling, worst)
    finally:
        if was_training:
            model.train()

    return Ablation(
        kind="layers",
        task=task,
        unit=TASKS[task],
        sites=rows,
        n_sites=len(rows),
        n_controlled=sum(1 for s in rows if s.beats_control is not None),
        n_samples=int(samples.shape[0]),
        passes=passes,
        seconds=round(time.perf_counter() - started, 2),
        truncated=truncated,
        control_ceiling=None if ceiling is None else round(ceiling, 6),
        expected_false_positives=round(
            sum(1 for s in rows if s.beats_control is not None)
            / (max(1, draws) + 1),
            3,
        ),
    )


# --------------------------------------------------------------- occlusion


def _regions(shape: list[int], grid: int) -> list[tuple[str, tuple]]:
    """Which slices of one input get occluded, and what to call them.

    A flat feature vector occludes per feature. Anything with spatial
    dimensions occludes per PATCH: one pixel of a 224x224 image is 0.002% of
    the input, costs a forward pass, and measures nothing.
    """
    if len(shape) <= 1:
        return [(f"feature {i}", (i,)) for i in range(int(shape[0]))]

    # Channels are not spatial. Occluding "the red channel" is a different
    # question from occluding a region, and mixing them into one list would
    # put two kinds of claim under one heading.
    spatial = shape[1:]
    if not spatial:
        return [(f"channel {i}", (i,)) for i in range(int(shape[0]))]

    steps = []
    for size in spatial:
        size = int(size)
        n = min(grid, size)
        edges = [round(k * size / n) for k in range(n + 1)]
        steps.append([(edges[k], edges[k + 1]) for k in range(n) if edges[k + 1] > edges[k]])

    out: list[tuple[str, tuple]] = []

    def walk(prefix: list[tuple[int, int]], depth: int):
        if depth == len(steps):
            label = "patch " + "x".join(f"{a}:{b}" for a, b in prefix)
            out.append((label, tuple(slice(a, b) for a, b in prefix)))
            return
        for span in steps[depth]:
            walk(prefix + [span], depth + 1)

    walk([], 0)
    return out


def sweep_inputs(
    model,
    samples,
    *,
    task: str,
    grid: int = DEFAULT_PATCH_GRID,
    max_sites: int = MAX_SITES,
    max_controlled: int = MAX_CONTROLLED,
    draws: int = CONTROL_DRAWS,
) -> Ablation:
    """Occlude each input feature or patch and report how far the answer moved."""
    import torch

    if task not in TASKS:
        raise AblationError(f"unknown task {task!r}")
    if samples.ndim < 2:
        raise AblationError(
            "these samples have no feature dimension to occlude — "
            f"sample_inputs() returned shape {list(samples.shape)}."
        )
    if not samples.is_floating_point():
        raise AblationError(
            "these inputs are integers, which are token ids or indices rather "
            "than values, and the mean of two token ids is not a token. "
            "Occlusion is for continuous inputs; ablate the embedding layer "
            "instead."
        )

    per_sample = list(samples.shape[1:])
    regions = _regions(per_sample, grid)
    truncated = max(0, len(regions) - max_sites)
    regions = regions[:max_sites]

    started = time.perf_counter()
    was_training = model.training
    model.eval()
    passes = 0
    try:
        base = _outputs(model, samples)
        passes += 1
        scale = _scale(task, base)
        # The occlusion baseline: the same region's mean across your samples.
        mean_input = samples.mean(dim=0)

        rows: list[Site] = []
        for label, where in regions:
            edited = samples.clone()
            edited[(slice(None),) + where] = mean_input[where]
            moved = _outputs(model, edited)
            passes += 1
            rows.append(
                Site(
                    name=label,
                    kind="input",
                    effect=round(_effect(task, base, moved, scale), 6),
                )
            )

        rows.sort(key=lambda s: -s.effect)
        by_name = dict(regions)
        ceiling = None
        for site in rows[:max_controlled]:
            # A DIFFERENT REGION, not a random direction in this one. The
            # layer control jitters the activation because a layer output has
            # hundreds of dimensions and a random direction in that space is
            # genuinely unrelated to the data. A single input feature has ONE
            # dimension: a "random direction" there is +1 or -1, so the
            # control performs the same edit as the treatment up to a sign and
            # the comparison becomes a coin flip.
            #
            # MEASURED on a trained net whose label depends only on features 0
            # and 1: with the jitter control both came back beats=False at
            # effects of 4.7596 and 2.4738 against controls of 4.8044 and
            # 2.5052, while noise features 3, 6, 13 and 16 came back
            # beats=True at 0.1278 and below. The null was labelling the
            # finding as noise and the noise as findings.
            #
            # So the control is `patch.py`'s shifted-position control: occlude
            # a DIFFERENT region the same way and see whether this one did
            # more. That answers the question the reader is actually asking --
            # is it this region, or does occluding anything here do this?
            where = by_name[site.name]
            others = [w for name, w in regions if name != site.name]
            if not others:
                # One region is the whole input. There is nowhere else to
                # occlude, so there is no control -- which stays None rather
                # than becoming a zero nobody measured.
                continue
            gen = torch.Generator().manual_seed(CONTROL_SEED)
            picks = torch.randperm(len(others), generator=gen)[: max(1, draws)]
            worst = 0.0
            for index in picks.tolist():
                elsewhere = others[index]
                edited = samples.clone()
                edited[(slice(None),) + elsewhere] = mean_input[elsewhere]
                moved = _outputs(model, edited)
                passes += 1
                worst = max(worst, _effect(task, base, moved, scale))
            site.control_max = round(worst, 6)
            site.control_draws = int(picks.numel())
            site.beats_control = site.effect > worst
            ceiling = worst if ceiling is None else max(ceiling, worst)
    finally:
        if was_training:
            model.train()

    return Ablation(
        kind="inputs",
        task=task,
        unit=TASKS[task],
        sites=rows,
        n_sites=len(rows),
        n_controlled=sum(1 for s in rows if s.beats_control is not None),
        n_samples=int(samples.shape[0]),
        passes=passes,
        seconds=round(time.perf_counter() - started, 2),
        truncated=truncated,
        control_ceiling=None if ceiling is None else round(ceiling, 6),
        expected_false_positives=round(
            sum(1 for s in rows if s.beats_control is not None)
            / (max(1, draws) + 1),
            3,
        ),
    )
