"""Where does this information appear — and is that curve any better than noise?

Give it labelled examples and it fits a linear probe at every layer, producing
the curve the field draws constantly: accuracy against depth, showing where a
concept becomes linearly readable.

That curve is easy to draw and easy to be fooled by. A probe reading 62% on a
set that is 60% one class has found almost nothing, and a probe reading 85% on
30 examples of a 4096-dimensional residual stream can fit noise perfectly. Both
produce a confident-looking line that goes up.

SO THE CURVE IS NEVER DRAWN ALONE

Two references are measured beside it and neither is optional:

  the majority-class rate   what you get by ignoring the input entirely and
                            always guessing the commoner label
  a permutation null band   K refits on SHUFFLED labels, at the same layer,
                            with the same examples and the same fit. Whatever
                            accuracy that reaches is what this setup produces
                            from information that is not there.

A layer whose probe lands inside the null band is reported as inside it, not as
a finding. That is the whole feature: "we have probes" is not worth building,
"we show you when your curve is inside the null" is.

WHAT A PROBE DOES NOT ESTABLISH

**Finding information is not the model using it.** A direction can be linearly
readable at layer 8 and play no part in the answer — the residual stream
carries much more than any one forward pass consumes. The only thing that
upgrades the claim is intervening: the fitted direction exports in the same
shape the steering store reads, so it can be pushed through the ablation
harness, and until that has been done the reading stands as "readable here",
never "used here".
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .errors import BadRequest, Refusal
from .feature_ablate import CONTROL_SEED
from .fmt import ordinal as _ordinal

# Enforced, not documented. A probe on four examples of a 768-dimensional
# stream separates them perfectly and means nothing; a held-out set of two
# scores 0%, 50% or 100% and nothing else, so its "accuracy" has a resolution
# of 50 percentage points.
MIN_PER_CLASS = 8

# MEASURED, not chosen. At six held-out examples the permutation null reached
# 1.00 at several of a model's layers on a concept the probe read perfectly
# everywhere -- so whether a layer came out READABLE or "inside the null" was
# decided by which shuffles happened to fit, not by the model. Accuracy on n
# examples has a resolution of 100/n percentage points, and a null with that
# granularity saturates.
#
# Twelve keeps the resolution under ten points. It is still small, which is
# why saturation is DETECTED as well as bounded -- see `null_saturated`.
MIN_TEST = 12

# Refits on shuffled labels. Enough that the band has shape rather than being
# two points, and few enough that a layer sweep stays a button rather than a
# job. Seeded from CONTROL_SEED so the same examples give the same band.
N_PERMUTATIONS = 20

# The band is the middle of the null, not its extremes: with 20 refits the
# single best shuffle is the best of 20 draws and using it as the ceiling
# would make the null wider every time more permutations were run.
NULL_LOW, NULL_HIGH = 5, 95

# Fitting. Full-batch on at most a few hundred examples of d_model floats,
# which is a fraction of a second and needs no schedule.
STEPS = 300
LR = 0.05
L2 = 1e-3


@dataclass
class LayerProbe:
    layer: int
    accuracy: float
    null_low: float
    null_high: float
    # THE FIELD THE WHOLE MODULE EXISTS FOR. True when this layer's probe did
    # not beat what the same fit achieves on shuffled labels.
    inside_null: bool
    beats_majority: bool
    # The null reached the top of the scale, so NO accuracy could have cleared
    # it. That is not a finding about this layer, it is the test set being too
    # small to distinguish one -- a third state, and reporting it as "inside
    # the null" would blame the model for the design.
    null_saturated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProbeReport:
    layers: list[LayerProbe] = field(default_factory=list)
    majority: float = 0.0
    n_train: int = 0
    n_test: int = 0
    n_permutations: int = 0
    label_names: list[str] = field(default_factory=list)
    position: int = -1
    counts: dict = field(default_factory=dict)

    @property
    def expected_false_positives(self) -> float:
        """How many layers should clear the band by chance alone.

        THE LAYER SWEEP IS A MULTIPLE COMPARISON and nothing else in this
        module accounted for it. The band is a 95th percentile, so each layer
        has about a 1-in-20 chance of clearing it on information that is not
        there -- and the sweep asks every layer. On a 12-layer model that is
        0.6 expected false positives, which means ONE readable layer is close
        to what noise produces and is not a finding on its own.

        Measured while testing this module: a sweep over five layers of pure
        noise with shuffled labels returned a readable layer often enough to
        make a test that asserted zero fail.
        """
        return round(len(self.layers) * (100 - NULL_HIGH) / 100, 2)

    @property
    def underpowered(self) -> list[LayerProbe]:
        """Layers whose null saturated, so nothing could have cleared it."""
        return [row for row in self.layers if row.null_saturated]

    @property
    def readable(self) -> list[LayerProbe]:
        """Layers that cleared both references. Possibly none, which is a
        result and the one this module is built to be able to report."""
        return [
            row for row in self.layers if not row.inside_null and row.beats_majority
        ]

    @property
    def best_layer(self) -> int | None:
        rows = self.readable
        return max(rows, key=lambda r: r.accuracy).layer if rows else None

    def to_dict(self) -> dict:
        return {
            **{k: v for k, v in asdict(self).items() if k != "layers"},
            "layers": [row.to_dict() for row in self.layers],
            "best_layer": self.best_layer,
            "n_readable_layers": len(self.readable),
            "n_underpowered_layers": len(self.underpowered),
            "expected_false_positives": self.expected_false_positives,
            "means": self.means(),
        }

    def means(self) -> str:
        # Said FIRST when it applies, because it changes how every other
        # number below should be read.
        warning = (
            f"THE NULL SATURATED AT {len(self.underpowered)} OF "
            f"{len(self.layers)} LAYERS: a fit on shuffled labels reached the "
            f"top of the scale there, so no accuracy could have cleared it and "
            f"those layers are untestable with {self.n_test} held-out "
            f"examples rather than uninformative. Add examples. "
            if self.underpowered
            else ""
        )
        if not self.readable:
            return warning + (
                f"NO LAYER read this concept better than noise. Every probe "
                f"landed inside the permutation null — the accuracy the same "
                f"fit reaches on shuffled labels — or failed to beat the "
                f"majority-class rate of {self.majority:.1%}, which is what "
                f"you get by ignoring the input and always guessing the "
                f"commoner label. That is a result: on these examples, at this "
                f"position, this concept is not linearly readable anywhere in "
                f"the model."
            )
        best = max(self.readable, key=lambda r: r.accuracy)
        # Said whenever the count is small enough for chance to explain it.
        # A run of adjacent layers is evidence; one isolated layer at the
        # expected false-positive rate is not.
        chance_warning = (
            f"ONLY {len(self.readable)} LAYER CLEARED, and sweeping "
            f"{len(self.layers)} layers against a {_ordinal(NULL_HIGH)}-percentile "
            f"band produces {self.expected_false_positives} by chance — read "
            f"this as noise unless it repeats on more examples. "
            if len(self.readable) <= self.expected_false_positives + 1
            else ""
        )
        return (
            warning
            + chance_warning
            + (
                f"{len(self.readable)} of {len(self.layers)} layers read this "
                f"concept above both references; layer {best.layer} is highest at "
                f"{best.accuracy:.1%}, against a majority-class rate of "
                f"{self.majority:.1%} and a permutation null reaching "
                f"{best.null_high:.1%} at that layer. Held out on {self.n_test} "
                f"examples. READABLE IS NOT USED — a direction can be linearly "
                f"present and play no part in the answer. Export it and ablate it "
                f"to find out; until then this says where the information is, not "
                f"whether the model reads it."
            )
        )


def _split(vectors, labels, *, seed: int):
    """(train, test) indices, stratified and deterministic.

    Stratified because an unstratified split of an imbalanced set can put
    every example of the rarer class on one side, and the probe then scores
    the majority rate while looking like it was tested.
    """
    import torch

    generator = torch.Generator().manual_seed(seed)
    train, test = [], []
    for value in sorted(set(labels.tolist())):
        idx = (labels == value).nonzero(as_tuple=True)[0]
        shuffled = idx[torch.randperm(len(idx), generator=generator)]
        cut = max(1, round(len(shuffled) * 0.3))
        test.extend(shuffled[:cut].tolist())
        train.extend(shuffled[cut:].tolist())
    return torch.tensor(sorted(train)), torch.tensor(sorted(test))


def _percentile(ordered: list[float], p: float) -> float:
    """Nearest-rank percentile of an already-sorted list.

    The ceiling used to be `band[min(n - 1, round(n * p / 100))]`, one rank
    too high at every n -- and at the default 20 permutations that lands on
    index 19 of 0..19, which is the MAXIMUM. The comment beside NULL_HIGH
    rules that out in as many words: "with 20 refits the single best shuffle
    is the best of 20 draws and using it as the ceiling would make the null
    wider every time more permutations were run." The code did exactly what
    the comment says it must not.

    It is not only mislabelling. `expected_false_positives` is derived as
    `n_layers * (100 - NULL_HIGH) / 100`, which assumes the ceiling really is
    the 95th percentile; against the max of 20 draws the true rate is 1/21,
    and the two only agree by coincidence at this one value of
    `n_permutations`. Raising it would have moved the real rate and left the
    reported one alone.
    """
    if not ordered:
        raise ValueError("a percentile of nothing is not a number")
    rank = math.ceil(p / 100 * len(ordered)) - 1
    return float(ordered[min(len(ordered) - 1, max(0, rank))])


def _fit(x, y, *, steps: int = STEPS, lr: float = LR, l2: float = L2):
    """A logistic probe, in torch. Returns (weights, bias).

    Pure torch rather than scikit-learn on purpose: the runtime dependencies
    are torch, transformers and fastapi, and a probe panel is not worth adding
    a fourth to a package people install to look at a model.
    """
    import torch

    # Standardised per feature. Residual dimensions differ in scale by orders
    # of magnitude, and without this the fit spends its capacity on whichever
    # dimension happens to be largest rather than on the one that separates.
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp(min=1e-6)
    xs = (x - mean) / std

    weights = torch.zeros(xs.shape[1], dtype=torch.float32, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    optimiser = torch.optim.Adam([weights, bias], lr=lr)
    target = y.float()
    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        logits = xs @ weights + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss = loss + l2 * weights.pow(2).sum()
        loss.backward()
        optimiser.step()
    return weights.detach(), bias.detach(), mean, std


def _accuracy(weights, bias, mean, std, x, y) -> float:
    import torch

    with torch.no_grad():
        logits = ((x - mean) / std) @ weights + bias
        return float(((logits > 0).long() == y).float().mean())


def _binary(labels):
    """Labels as 0 and 1, whichever two values the caller actually used.

    `sweep` did this and `direction_at` did not, so one label vector got two
    answers. `runtime.probe_layers` only checks that a label is an int, so
    examples labelled 1 and 2 are legal: `sweep` remapped them and scored the
    layer correctly, while `direction_at` handed the raw 1s and 2s to `_fit`,
    which floats them into `binary_cross_entropy_with_logits`. That does not
    validate its target, so a target of 2.0 gives a finite loss whose gradient
    `sigmoid(z) - t` never changes sign -- the fit runs to completion and
    returns the class centroid direction rather than the separating one. It
    was then written into the steering store carrying the accuracy `sweep`
    measured, with no exception and nothing on screen.

    Shared rather than copied, and that matters beyond the remap itself: both
    sides must send the SAME class to 1, or the exported direction points
    opposite to the accuracy reported beside it.
    """
    import torch

    labels = torch.as_tensor(labels).long().cpu()
    values = sorted(set(labels.tolist()))
    if len(values) != 2:
        raise BadRequest(
            f"a probe separates two classes and these examples carry "
            f"{len(values)}. Label them 0 and 1."
        )
    if labels.min() != 0 or labels.max() != 1:
        labels = (labels == values[1]).long()
    return labels


def sweep(
    states: dict,
    labels,
    *,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = CONTROL_SEED,
) -> ProbeReport:
    """Fit a probe at every layer, with its null and the majority line.

    `states` is {layer: [n_examples, d_model]} — captured by the caller at the
    same pre-hook point `steer_vectors` and `patch` use, so a direction fitted
    here lives in the space those already measure.
    """
    import torch

    # ON THE CPU, deliberately. The states arrive on whatever device the model
    # runs on, and the fit is a few hundred examples of d_model floats -- but
    # it is run n_layers x (1 + n_permutations) times, and every one of those
    # allocations would be competing with the model for VRAM on a machine
    # chosen for having 8 GB of it. Moving them here also frees the capture.
    states = {layer: x.detach().float().cpu() for layer, x in states.items()}
    labels = _binary(labels)

    for value in (0, 1):
        count = int((labels == value).sum())
        if count < MIN_PER_CLASS:
            raise BadRequest(
                f"class {value} has {count} examples and a probe needs at "
                f"least {MIN_PER_CLASS}. Below that a linear fit separates the "
                f"examples perfectly and the number means nothing — this is "
                f"enforced rather than warned about."
            )

    train_idx, test_idx = _split(next(iter(states.values())), labels, seed=seed)
    if len(test_idx) < MIN_TEST:
        raise BadRequest(
            f"the held-out set would be {len(test_idx)} examples, and an "
            f"accuracy measured on that few has a resolution of "
            f"{100 / max(1, len(test_idx)):.0f} percentage points. At least "
            f"{MIN_TEST} are needed for the number to mean anything."
        )

    y_train, y_test = labels[train_idx], labels[test_idx]
    # What you get by ignoring the input entirely. Measured on the SAME
    # held-out set the probe is scored on, or the two would not be comparable.
    majority = float(max((y_test == v).float().mean() for v in (0, 1)))

    rows: list[LayerProbe] = []
    for layer in sorted(states):
        x = states[layer].float()
        x_train, x_test = x[train_idx], x[test_idx]

        weights, bias, mean, std = _fit(x_train, y_train)
        accuracy = _accuracy(weights, bias, mean, std, x_test, y_test)

        # THE NULL, refit from scratch on shuffled labels. Not a formula, not
        # a chi-square: the same fit, the same examples, the same number of
        # steps, on labels that carry no information. Whatever that reaches is
        # what this setup produces from nothing.
        generator = torch.Generator().manual_seed(seed + layer)
        null_scores = []
        for _ in range(n_permutations):
            shuffled = y_train[torch.randperm(len(y_train), generator=generator)]
            n_weights, n_bias, n_mean, n_std = _fit(x_train, shuffled)
            null_scores.append(
                _accuracy(n_weights, n_bias, n_mean, n_std, x_test, y_test)
            )
        band = sorted(null_scores)
        low = _percentile(band, NULL_LOW)
        high = _percentile(band, NULL_HIGH)

        rows.append(
            LayerProbe(
                layer=layer,
                accuracy=round(accuracy, 4),
                null_low=round(low, 4),
                null_high=round(high, 4),
                inside_null=accuracy <= high,
                beats_majority=accuracy > majority,
                null_saturated=high >= 1.0,
            )
        )

    return ProbeReport(
        layers=rows,
        majority=round(majority, 4),
        n_train=len(train_idx),
        n_test=len(test_idx),
        n_permutations=n_permutations,
        counts={str(v): int((labels == v).sum()) for v in (0, 1)},
    )


def direction_at(states: dict, labels, layer: int, *, seed: int = CONTROL_SEED):
    """The fitted direction at one layer, for export into the steering store.

    Returned in the residual stream's own units rather than the standardised
    space the fit works in, so it can be added to a stream directly — which is
    what the steering harness does with it.
    """
    if layer not in states:
        raise Refusal(f"no residual stream was captured at layer {layer}.")
    # The SAME remap `sweep` applies. Without it a 1/2 labelling fitted
    # against targets of 1.0 and 2.0 and exported a direction that separates
    # nothing, under the accuracy sweep had measured on the 0/1 version.
    labels = _binary(labels)
    train_idx, _ = _split(next(iter(states.values())), labels, seed=seed)
    weights, _, _, std = _fit(
        states[layer].detach().float().cpu()[train_idx], labels[train_idx]
    )
    # Undo the standardisation: a unit step along `weights` in standardised
    # space is a step of `weights / std` in the stream's own space.
    direction = weights / std.squeeze(0)
    return direction / direction.norm().clamp(min=1e-9)
