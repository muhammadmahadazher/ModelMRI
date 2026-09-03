# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""What a vision model actually looked at, by covering things up.

Every saliency picture this project could have drawn for a classifier is a
gradient or an attention weight, and both are correlational: they say what the
output was *sensitive to* in the limit of an infinitesimal nudge, or what the
architecture *routed*, and neither is an answer to "would it still say cat if
that were not there". `patch.py` makes the same argument on the text side and
draws the same conclusion — ablation says "this mattered", and only an
intervention says it about the thing you actually removed. `vla_occlude.py`
already applies it to a robot's camera. This is that measurement for a plain
image model.

So: slide an occluder over the image, re-run, and report how far the output
moved. The model is a black box throughout — anything that maps an image
tensor to logits qualifies, ViT or CNN or detector — because nothing here
reads a weight, hooks a module, or knows an architecture's name. That is a
deliberate trade: a black box cannot say WHERE inside the model the decision
lives (that is `patch.py`'s job), only where in the IMAGE the evidence was.

## The map is coarser than the image, and it must say so

One cell per occluded window, not one per pixel. A 224x224 image at patch 16
stride 16 gives a 14x14 map, and a feature smaller than 16 pixels cannot be
located more precisely than the window containing it. Every result carries its
resolution and that sentence, because a 14x14 array upsampled into a 224x224
heatmap is the most convincing-looking lie this module could tell.

Coverage is exact rather than approximate. Windows are placed at `stride` and
the final one on each axis is CLAMPED to the edge, so a 30-pixel axis at patch
8 stride 8 ends at 22 rather than 16 — otherwise the last six pixel columns
are never occluded and the map is silent about a strip of the image. A stride
wider than the patch is refused for the same reason: it leaves pixels no
window ever covers, and a map with holes in it still looks like a map.

## The fill is a choice that changes the answer

Grey, black, white and the image's own mean are four different baselines and
they give four different maps. There is no neutral one: occlusion is out of
distribution by construction — the model has never seen a flat grey square —
so part of every score is the square rather than the missing content.

The default is `grey`, the midpoint of the value range, because it is the
furthest from both extremes and therefore the least likely to be read as
content in its own right. It is still a choice, it is still named on every
result, and the honest use is to run two and keep what survives both.

MEAN SUBSTITUTION IS A SPECIFIC BASELINE, NOT "REMOVAL". `vla_actions.py`
carries this sentence for a robot's camera streams and it is the same fact
here: the mean colour of an image is a real colour the model has opinions
about, not absence. Nothing in this module can delete a region; it can only
replace it with something else, and the something else is in the result.

And the range those fills come from is READ, not assumed. A tensor may be
[0,1], [-1,1], or ImageNet-normalised to roughly [-2.1, 2.6], and "grey =
0.5" is wrong for two of the three. Where the caller does not state the range
this infers it from the image's own extremes and says that it did — one
image's observed minimum is a lower bound on the model's input range, not the
range itself.

## Cost is priced before it is spent, and refused above a ceiling

The sweep is `ceil(H/stride) x ceil(W/stride)` forward passes plus one
reference. That is 196 at the default on a 224 image, and 43,681 at stride 1
— which is not a slower run, it is a different afternoon. `estimate()` answers
the number without running anything and NEVER refuses, because refusing to
tell you the cost of the run you are about to be refused is useless. `sweep()`
refuses above `MAX_PASSES` and the refusal names the stride that would fit.

## Batched, because one pass per patch is the naive version

Occluded variants are stacked and run `batch` at a time under `no_grad`, so
the default 196-window sweep is 8 forward calls — seven batches and the
unoccluded reference — rather than 197. Measured on a 318,696-parameter
convolutional classifier at 224x224 on CPU, median of three: 0.402 s
unbatched against 0.124 s at batch 32, so the batching is worth 3.2x and the
whole default sweep is an eighth of a second on a small model. It is not free
on a large one, which is what `estimate()` is for.

The bound is `MAX_BATCH` and it exists in memory, not in taste: the batch
holds that many full-size copies of the image resident at once — 38.5 MB at
64 for a 224x224x3 float32 image — and the model's activations behind them are
some unknown multiple of that. The requested and the used batch size both
travel in the result, because a silently reduced batch is a silently different
run time and somebody will time it.

BATCHING IS NOT BIT-IDENTICAL, and the honest version of that claim was worth
measuring rather than asserting. The windows do not move — an exact-arithmetic
model gives the same map at batch 1 and batch 64, which is the test that would
catch a batching bug shuffling regions. But a *convolution* blocks a batch of
32 differently from a batch of one, and the measured discrepancy on that same
CNN was 1e-6 logits: exactly one unit in the last decimal these scores are
reported at, against a map spanning 1.23e-4. So the last reported digit is not
reproducible across batch sizes, and a map whose whole spread sits near
`10 ** -SCORE_DECIMALS` is a map made of rounding. `means()` says so when that
happens.

## Softmax confidence is not the probability of being right

Any result here that reports a class probability carries that sentence. A
softmax is a normalisation over the classes this head happens to have; it is
high when the model is confidently wrong, it moves when an unrelated class
moves, and it has no relationship to correctness that was not put there by
calibration nobody in this module has done. The primary score is therefore the
TARGET LOGIT's movement, which at least cannot be changed by what a different
class did — the probability drop rides alongside, labelled.

Scores are SIGNED. A positive drop means covering that window cost the class
evidence; a negative one means covering it HELPED, which is a region arguing
against the class and is a finding rather than an error. An absolute value
would erase the difference, exactly as it would for `vla_actions.compare`.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

from . import fmt
from .errors import BadRequest, Refusal
from .fmt import ordinal as _ordinal

# The four fills, named on every result. There is no neutral one — see the
# module docstring — so the point of shipping several is that a finding which
# only appears under one of them is a finding about the fill.
FILLS = ("grey", "black", "white", "image_mean")

# The default, and it is a choice rather than an absence of one: the midpoint
# of the value range is the furthest point from both extremes, so it is the
# fill least likely to read as content in its own right.
DEFAULT_FILL = "grey"

# Patch and stride defaults. 16 is the patch size of the ViT family this will
# most often be pointed at, so the default map lines up with the model's own
# token grid rather than cutting across it — and a stride equal to the patch
# is the cheapest schedule that covers every pixel exactly once.
DEFAULT_PATCH = 16

# The most forward passes a sweep will run before it refuses. A 224x224 image
# at patch 16 stride 1 is 43,681 windows; at stride 16 it is 196. The ceiling
# sits above every sensible tiling and below every accidental one, and the
# refusal names the stride that would fit rather than merely saying no.
MAX_PASSES = 4096

#: The largest axis this will do window arithmetic over.
#:
#: Not a bound on picture quality and not a second `MAX_PASSES`:
#: `_axis` MATERIALISES every window start as a list before anything
#: counts them, so a height nobody could photograph builds a
#: multi-gigabyte list to answer a question the pass ceiling was always
#: going to refuse. `/api/image/attribution/cost` and
#: `/api/image/cv/cost` take height and width straight from the query
#: string, so this is reachable by one request.
#:
#: Checked in the one function `estimate` and `plan_windows` share, so
#: the price and the run cannot disagree about which geometries exist.
MAX_AXIS = 16_384

# Occluded copies per forward call. The batch holds this many full-size images
# resident at once -- at 224x224x3 float32 that is 38.5 MB at 64, and the
# activations behind it are an unknown multiple -- so the bound is memory
# rather than preference, and both the requested and the used value travel in
# the result.
DEFAULT_BATCH = 32
MAX_BATCH = 64

# Decimals every score is reported at. Named rather than sprinkled through the
# rounding calls, because it is the map's precision floor and two sentences in
# `means()` are about it: a map whose entire spread fits inside one unit of
# this is a map made of rounding, and the measured batch-to-batch discrepancy
# on a real convolution is exactly one unit of it.
SCORE_DECIMALS = 6


class NotAttributable(Refusal):
    """No honest map can be drawn from this model on this image.

    Its own class because the caller's fix differs from a `BadRequest`: not "a
    parameter is wrong" but "the run itself would not measure what the picture
    would claim". A model in training mode, an image with no contrast, a head
    that returns something this cannot read as logits.
    """


def _as_int(value, name: str) -> int:
    """An integer parameter, with `bool` refused rather than silently accepted.

    `isinstance(True, int)` is True in Python, so `patch=True` would sail
    through an `isinstance` check and become a patch size of 1 — a 224x224 map
    of single pixels, which is 50,176 forward passes and looks like somebody
    asked for it.
    """
    if isinstance(value, bool):
        raise BadRequest(
            f"{name} was given as the boolean {value!r}. `isinstance(True, "
            f"int)` is True in Python, so this would quietly become {name}="
            f"{int(value)}. Pass the number you meant."
        )
    if not isinstance(value, int):
        raise BadRequest(f"{name} must be a whole number of pixels, not {value!r}.")
    return int(value)


def _axis(length: int, patch: int, stride: int) -> tuple[list[int], bool]:
    """Window starts along one axis, with the last one clamped to the edge.

    Returns `(starts, clamped)`. Without the clamp a 30-pixel axis at patch 8
    stride 8 stops at 16, and pixels 24-29 are never occluded by anything —
    the map would be silent about a strip of the image while still being
    presented as a map OF the image. The clamp costs one extra window and
    makes the final pair overlap, which is reported rather than hidden.
    """
    starts = list(range(0, length - patch + 1, stride))
    if not starts:
        starts = [0]
    clamped = False
    if starts[-1] + patch < length:
        starts.append(length - patch)
        clamped = True
    return starts, clamped


def _count_windows(height: int, width: int, patch: int, stride: int) -> tuple[int, int]:
    """How many rows and columns the map will have. Runs nothing."""
    tops, _ = _axis(height, patch, stride)
    lefts, _ = _axis(width, patch, stride)
    return len(tops), len(lefts)


@dataclass
class Grid:
    """Where the occluder goes, and what the resulting map's resolution is."""

    height: int
    width: int
    patch: int
    stride: int
    tops: list[int] = field(default_factory=list)
    lefts: list[int] = field(default_factory=list)
    # Whether a final window had to be pulled back to the edge. True means the
    # last row (or column) overlaps its neighbour by more than `patch -
    # stride`, which is a fact about this map and not a defect.
    edge_row_clamped: bool = False
    edge_col_clamped: bool = False

    @property
    def rows(self) -> int:
        return len(self.tops)

    @property
    def cols(self) -> int:
        return len(self.lefts)

    @property
    def n_windows(self) -> int:
        return self.rows * self.cols

    @property
    def passes(self) -> int:
        """Every window plus the one unoccluded reference run."""
        return self.n_windows + 1

    @property
    def overlap(self) -> int:
        """Pixels shared by neighbouring windows, from the stride alone."""
        return max(0, self.patch - self.stride)

    def windows(self) -> list[tuple[int, int, int, int]]:
        """`(row, col, top, left)` in row-major order."""
        return [
            (r, c, top, left)
            for r, top in enumerate(self.tops)
            for c, left in enumerate(self.lefts)
        ]

    def to_dict(self) -> dict:
        return {
            "height": self.height,
            "width": self.width,
            "patch": self.patch,
            "stride": self.stride,
            "map_rows": self.rows,
            "map_cols": self.cols,
            "n_windows": self.n_windows,
            "passes": self.passes,
            "overlap": self.overlap,
            "edge_row_clamped": self.edge_row_clamped,
            "edge_col_clamped": self.edge_col_clamped,
        }


def _validate_geometry(height: int, width: int, patch: int, stride: int) -> int:
    """Shared checks, so `estimate` and `plan_windows` cannot disagree."""
    patch = _as_int(patch, "patch")
    stride = _as_int(stride, "stride")
    if patch < 1:
        raise BadRequest("patch must be at least 1 pixel.")
    if stride < 1:
        raise BadRequest(
            "stride must be at least 1 pixel. A stride of 0 would place every "
            "window at the same place forever."
        )
    if height < 1 or width < 1:
        raise BadRequest(f"an image of {height}x{width} pixels has nothing to occlude.")
    if height > MAX_AXIS or width > MAX_AXIS:
        raise BadRequest(
            f"an image of {height:,}x{width:,} is past the {MAX_AXIS:,}-pixel "
            f"axis this plans over. The bound is on the ARITHMETIC rather "
            f"than on the picture: every window start is materialised "
            f"before any of them is counted, so a size like this builds a "
            f"list of billions of positions to price a sweep the "
            f"{MAX_PASSES}-pass ceiling was always going to refuse. Price "
            f"the checkpoint's own input size — the model resizes to it "
            f"before it sees anything."
        )
    if patch > min(height, width):
        raise BadRequest(
            f"a patch of {patch} does not fit inside a {height}x{width} image. "
            f"The occluder has to be smaller than what it is occluding."
        )
    if stride > patch:
        # The substantive geometric refusal. Stride 16 with patch 8 skips
        # every other block, so half the pixels are never covered by any
        # window — and the resulting map has holes in it while still being
        # shaped like a complete map of the image.
        covered = patch / stride
        raise BadRequest(
            f"a stride of {stride} with a patch of {patch} leaves "
            f"{(1 - covered) * 100:.0f}% of the pixels under no window at all, "
            f"so the map would have holes in it and still look like a map of "
            f"the whole image. Set the stride to {patch} or less."
        )
    return stride


def plan_windows(
    height: int,
    width: int,
    *,
    patch: int = DEFAULT_PATCH,
    stride: int | None = None,
    max_passes: int = MAX_PASSES,
) -> Grid:
    """Where every occluder goes. Refuses above the ceiling rather than running.

    `stride` defaults to `patch`: a plain tiling, which is the cheapest
    schedule that covers every pixel exactly once. A smaller stride overlaps
    and gives a smoother map for a quadratic price.
    """
    height = _as_int(height, "height")
    width = _as_int(width, "width")
    stride = _validate_geometry(
        height, width, patch, patch if stride is None else stride
    )
    patch = int(patch)

    tops, row_clamped = _axis(height, patch, stride)
    lefts, col_clamped = _axis(width, patch, stride)
    grid = Grid(
        height=height,
        width=width,
        patch=patch,
        stride=stride,
        tops=tops,
        lefts=lefts,
        edge_row_clamped=row_clamped,
        edge_col_clamped=col_clamped,
    )

    if grid.n_windows < 2:
        raise BadRequest(
            f"a patch of {patch} on a {height}x{width} image gives one window, "
            f"which covers the whole picture. Occluding everything measures "
            f"whether the model can see at all, not where in the image it was "
            f"looking. Use a smaller patch."
        )

    if grid.passes > max_passes:
        raise BadRequest(
            f"this is {grid.n_windows} windows and {grid.passes} forward "
            f"passes, past the ceiling of {max_passes}. {_stride_that_fits(height, width, patch, max_passes)} "
            f"Nothing was run: a sweep this size is a job rather than a click, "
            f"and finding out by waiting is the failure this refusal exists to "
            f"prevent."
        )
    return grid


def _stride_that_fits(height: int, width: int, patch: int, max_passes: int) -> str:
    """The sentence naming a stride that would fit, or why none does.

    A refusal that only says no leaves the caller guessing at the parameter,
    and the arithmetic to un-guess it is right here.
    """
    for stride in range(1, patch + 1):
        rows, cols = _count_windows(height, width, patch, stride)
        if rows * cols + 1 <= max_passes:
            return (
                f"A stride of {stride} would be {rows * cols} windows "
                f"({rows}x{cols}) and fits."
            )
    rows, cols = _count_windows(height, width, patch, patch)
    return (
        f"Even a plain tiling at stride {patch} is {rows * cols} windows, so "
        f"this image needs a larger patch or a raised ceiling — and a larger "
        f"patch means a coarser map, which is the trade being made."
    )


def estimate(
    height: int,
    width: int,
    *,
    patch: int = DEFAULT_PATCH,
    stride: int | None = None,
    batch: int = DEFAULT_BATCH,
    channels: int = 3,
    bytes_per_value: int = 4,
    seconds_per_pass: float | None = None,
    max_passes: int = MAX_PASSES,
) -> dict:
    """What the sweep will cost, before a single pass is taken.

    This NEVER refuses on the ceiling. A caller who is about to be refused
    needs the number that got them refused, and an `estimate` that declined to
    produce it would leave them guessing at the stride.

    `seconds_per_pass` is the caller's own measurement of this model on this
    machine, and there is no default: a time here that came from a constant
    somebody typed would be a fabricated forecast. Absent, `seconds` is None —
    which is "nobody measured", not "instant".
    """
    height = _as_int(height, "height")
    width = _as_int(width, "width")
    stride = _validate_geometry(
        height, width, patch, patch if stride is None else stride
    )
    patch = int(patch)
    # BOTH, which is what `_clamp_batch` returns and what its docstring asks
    # for: "Both travel, because a silent cap is a defect." This took `[1]`
    # and threw the request away, so the cost estimate quoted a batch the
    # caller never asked for with nothing saying it had been reduced — and
    # `sweep`, the run this estimates, reports the pair. One question, two
    # answers, and the estimate is the one read BEFORE committing to the cost.
    batch_requested, batch = _clamp_batch(batch)

    rows, cols = _count_windows(height, width, patch, stride)
    n_windows = rows * cols
    passes = n_windows + 1
    # The reference pass is its own call; the rest go out in full batches.
    calls = 1 + math.ceil(n_windows / batch)
    input_bytes = batch * channels * height * width * bytes_per_value

    seconds = None
    if seconds_per_pass is not None:
        value = float(seconds_per_pass)
        if not math.isfinite(value) or value < 0:
            raise BadRequest(
                f"seconds_per_pass must be a finite, non-negative measurement, "
                f"not {seconds_per_pass!r}."
            )
        seconds = round(passes * value, 3)

    within = passes <= max_passes
    return {
        "map_rows": rows,
        "map_cols": cols,
        "n_windows": n_windows,
        "passes": passes,
        "forward_calls": calls,
        "batch": batch,
        "batch_requested": batch_requested,
        "patch": patch,
        "stride": stride,
        # Only the occluded copies of the input, which is the one number this
        # can compute exactly. Activations are a multiple of it that nothing
        # here can know without running the model, so they are absent rather
        # than estimated.
        "input_bytes_per_call": input_bytes,
        "seconds": seconds,
        "within_ceiling": within,
        "ceiling": max_passes,
        "means": (
            f"A {patch}x{patch} occluder at stride {stride} over a "
            f"{height}x{width} image is {n_windows} windows — a {rows}x{cols} "
            f"map — and {passes} forward passes, sent {batch} at a time in "
            f"{calls} calls."
            + (
                f" A batch of {batch_requested} was asked for and {batch} is "
                f"this module's bound on how many full-size copies of the "
                f"image it will hold at once, so the figures here are for "
                f"{batch}."
                if batch_requested != batch
                else ""
            )
            + f" The occluded copies alone are "
            f"{fmt.bytes_si(input_bytes)} per call; the activations behind "
            f"them are a multiple of that which nothing here can know without "
            f"running the model."
            + (
                f" At the {float(seconds_per_pass):,.4f}s per pass you "
                f"measured, that is {seconds:,.1f}s."
                if seconds is not None
                else " No per-pass time was measured, so there is no forecast "
                "here — an invented one would be a number this tool made up."
            )
            + (
                ""
                if within
                else f" THIS IS PAST THE CEILING OF {max_passes} PASSES and "
                f"`sweep` will refuse it. "
                + _stride_that_fits(height, width, patch, max_passes)
            )
        ),
    }


def _clamp_batch(batch) -> tuple[int, int]:
    """`(requested, used)`. Both travel, because a silent cap is a defect."""
    requested = _as_int(batch, "batch")
    if requested < 1:
        raise BadRequest("batch must be at least 1 occluded copy per call.")
    return requested, min(requested, MAX_BATCH)


@dataclass
class Window:
    """One occluded region, and what covering it did to the target class."""

    row: int
    col: int
    top: int
    left: int
    height: int
    width: int
    # Signed: positive means covering this cost the class evidence, negative
    # means covering it HELPED. An absolute value would print the same number
    # for a region that supports the class and one that argues against it.
    logit_drop: float = 0.0
    # The same movement in softmax probability, which is a different quantity
    # and not a better one -- it moves when any other class moves. None when
    # the head has a single output, where a softmax is 1.0 by construction.
    prob_drop: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Attribution:
    """One occlusion map, and everything it is not allowed to claim."""

    grid: Grid
    windows: list[Window] = field(default_factory=list)
    fill: str = DEFAULT_FILL
    # The value the occluder was actually filled with, per channel where it
    # varies. Reported because "grey" is a word and this is the number.
    fill_value: list[float] = field(default_factory=list)
    value_range: tuple[float, float] = (0.0, 1.0)
    # Whether that range was read from the caller or inferred from this one
    # image. One image's extremes are a lower bound on the model's input
    # range, not the range, and the distinction is in the sentence.
    value_range_inferred: bool = True
    target: int = 0
    target_label: str = ""
    # Whether the class was the model's own top prediction or one the caller
    # named. Attributing the model's own answer and auditing a label you
    # supplied are different questions with the same picture.
    target_chosen_by_model: bool = True
    classes: int = 0
    base_logit: float = 0.0
    # None for a single-output head, where softmax is 1.0 by construction.
    base_prob: float | None = None
    passes: int = 0
    forward_calls: int = 0
    batch_requested: int = 0
    batch_used: int = 0
    seconds: float = 0.0
    model_name: str = ""
    # True when class names were supplied and did not match the head's width,
    # so they were dropped entirely rather than applied to the wrong classes.
    class_names_dropped: bool = False

    @property
    def spread(self) -> float:
        """Largest drop minus smallest, over the whole map.

        The scale the peak has to be read against. A map whose spread is
        0.0001 has a "strongest window" in the same sense that a flat field
        has a highest blade of grass.
        """
        if not self.windows:
            return 0.0
        drops = [w.logit_drop for w in self.windows]
        return max(drops) - min(drops)

    @property
    def below_precision(self) -> bool:
        """Does the whole map fit inside one unit of its own last digit?

        Not a threshold somebody chose: `SCORE_DECIMALS` is the precision
        these scores are reported at, and the measured batch-to-batch
        discrepancy on a real convolution is exactly one unit of it. A map
        whose entire span is that small is made of rounding, and ranking its
        windows is ranking rounding.
        """
        return bool(self.windows) and 0.0 < self.spread <= 10.0**-SCORE_DECIMALS

    @property
    def strongest(self) -> Window | None:
        """The window whose occlusion cost the class the most evidence.

        None -- not the first window -- when the map is exactly flat. A model
        that returns the same logits for every occlusion has told you it did
        not use the image, and naming a peak in that map would be reading rank
        order out of a tie.
        """
        if not self.windows or self.spread == 0.0:
            return None
        return max(self.windows, key=lambda w: w.logit_drop)

    @property
    def most_negative(self) -> Window | None:
        """The window that most INCREASED the class when covered, or None.

        None when nothing did, which is a different answer from a drop of 0.0:
        it means no region of this image was arguing against the class.
        """
        if not self.windows:
            return None
        low = min(self.windows, key=lambda w: w.logit_drop)
        return low if low.logit_drop < 0 else None

    def map_rows(self) -> list[list[float]]:
        """The map as `rows x cols` of signed logit drops, for drawing.

        Every cell is filled — the grid covers the image completely by
        construction — so a caller never has to decide what a missing cell
        means.
        """
        table = [[0.0] * self.grid.cols for _ in range(self.grid.rows)]
        for w in self.windows:
            table[w.row][w.col] = w.logit_drop
        return table

    def to_dict(self) -> dict:
        return {
            "grid": self.grid.to_dict(),
            "windows": [w.to_dict() for w in self.windows],
            "map": self.map_rows(),
            "fill": self.fill,
            "fill_value": self.fill_value,
            "value_range": list(self.value_range),
            "value_range_inferred": self.value_range_inferred,
            "target": self.target,
            "target_label": self.target_label,
            "target_chosen_by_model": self.target_chosen_by_model,
            "classes": self.classes,
            "base_logit": self.base_logit,
            "base_prob": self.base_prob,
            "spread": round(self.spread, SCORE_DECIMALS),
            "strongest": self.strongest.to_dict() if self.strongest else None,
            "most_negative": (
                self.most_negative.to_dict() if self.most_negative else None
            ),
            "passes": self.passes,
            "forward_calls": self.forward_calls,
            "batch_requested": self.batch_requested,
            "batch_used": self.batch_used,
            "seconds": self.seconds,
            "model_name": self.model_name,
            "class_names_dropped": self.class_names_dropped,
            "means": self.means(),
        }

    def means(self) -> str:
        parts: list[str] = []
        who = self.target_label or f"class {self.target}"
        parts.append(
            f"Every {self.grid.patch}x{self.grid.patch} window of this "
            f"{self.grid.height}x{self.grid.width} image was replaced by the "
            f"{self.fill} fill and {self.model_name or 'the model'} re-run — "
            f"{self.grid.n_windows} windows at stride {self.grid.stride}, "
            f"{self.passes} forward passes in {self.forward_calls} batched "
            f"calls of at most {self.batch_used}."
        )
        parts.append(
            f"THE MAP IS {self.grid.rows}x{self.grid.cols}, ONE CELL PER "
            f"WINDOW, so it is coarser than the image by a factor of "
            f"{self.grid.patch}: anything smaller than {self.grid.patch} "
            f"pixels cannot be located more precisely than the window it sits "
            f"in, whatever an upsampled heatmap appears to show."
        )

        if self.target_chosen_by_model:
            parts.append(
                f"The class is {who}, which the model chose itself as its top "
                f"prediction — nobody said it is the right answer, and this "
                f"map explains the answer given rather than a correct one."
            )
        else:
            parts.append(
                f"The class is {who}, which you named. This is where the "
                f"evidence for THAT class is, whether or not the model "
                f"predicted it."
            )

        peak = self.strongest
        if peak is None:
            parts.append(
                "NOTHING IN THIS IMAGE MOVED THE OUTPUT: every window scored "
                "identically, so there is no strongest region and none is "
                "named. A flat map is a result — it says this model's answer "
                "did not depend on any part of this picture — and picking a "
                "peak out of a tie would invent one."
            )
        else:
            parts.append(
                f"The strongest window is row {peak.row}, column {peak.col} "
                f"(pixels {peak.top}-{peak.top + peak.height} by "
                f"{peak.left}-{peak.left + peak.width}), where covering it "
                f"moved the {who} logit by "
                f"{peak.logit_drop:+,.{SCORE_DECIMALS}f}. That is only a peak "
                f"relative to the other {self.grid.n_windows - 1} windows OF "
                f"THIS IMAGE: the whole map spans "
                f"{self.spread:,.{SCORE_DECIMALS}f} logits, and if that span "
                f"is small then nothing here is distinguished from anything "
                f"else."
            )
            if self.below_precision:
                # Not a chosen threshold: `SCORE_DECIMALS` is the precision
                # these numbers are printed at, and the measured batch-to-batch
                # discrepancy through a real convolution is one unit of it. A
                # map narrower than that is being ranked by its own rounding.
                parts.append(
                    f"AND THAT SPAN IS SMALLER THAN THE PRECISION THIS IS "
                    f"REPORTED AT. The whole map fits inside one unit of the "
                    f"{_ordinal(SCORE_DECIMALS)} decimal, which is also how much the "
                    f"same sweep moves between batch sizes on a convolution — "
                    f"so the ranking above is a ranking of rounding, and no "
                    f"window here is distinguishable from any other."
                )
            low = self.most_negative
            if low is not None:
                parts.append(
                    f"Scores are signed, and row {low.row} column {low.col} is "
                    f"NEGATIVE at {low.logit_drop:+,.{SCORE_DECIMALS}f}: "
                    f"covering it RAISED the {who} logit, so that region was "
                    f"arguing against the class. An absolute value would have "
                    f"drawn it as evidence for."
                )

        fill_note = (
            "MEAN SUBSTITUTION IS A SPECIFIC BASELINE, NOT REMOVAL. The mean "
            "colour of this image is a real colour the model has opinions "
            "about; it is not absence, and nothing here can delete a region — "
            "only replace it."
            if self.fill == "image_mean"
            else (
                f"A flat {self.fill} square is a specific baseline, not "
                f"removal: nothing here can delete a region, only replace it "
                f"with something else, and a different fill gives a different "
                f"map."
            )
        )
        parts.append(
            f"{fill_note} OCCLUSION IS OUT OF DISTRIBUTION — the model has "
            f"never seen a flat {self.fill} patch — so part of every score is "
            f"the patch rather than the missing content. Run another of "
            f"{', '.join(FILLS)} and keep what survives both."
        )

        if self.base_prob is not None:
            parts.append(
                f"SOFTMAX CONFIDENCE IS NOT THE PROBABILITY OF BEING RIGHT. "
                f"The {self.base_prob:.4f} reported for {who} is a "
                f"normalisation over this head's {self.classes} classes: it "
                f"moves when an unrelated class moves, it is just as high when "
                f"the model is confidently wrong, and nothing in this tool has "
                f"calibrated it. The logit movement above is the primary score "
                f"for that reason."
            )
        else:
            parts.append(
                f"This head has a single output, so there is no class "
                f"probability to report — a softmax over one number is 1.0 by "
                f"construction and would say nothing. Only the raw output "
                f"movement is here, in {self.model_name or 'the model'}'s own "
                f"units, which are not comparable with another model's."
            )

        if self.value_range_inferred:
            parts.append(
                f"The fill was drawn from a value range of "
                f"[{self.value_range[0]:,.4f}, {self.value_range[1]:,.4f}] "
                f"INFERRED FROM THIS ONE IMAGE, because none was stated. One "
                f"image's extremes are a lower bound on what the model accepts, "
                f"not its input range — pass `value_range` if you know the "
                f"preprocessing."
            )

        if self.grid.overlap:
            parts.append(
                f"Neighbouring windows overlap by {self.grid.overlap} pixels, "
                f"so a pixel is occluded in several windows and their scores "
                f"are not independent of one another."
            )
        if self.grid.edge_row_clamped or self.grid.edge_col_clamped:
            which = " and ".join(
                w
                for w, on in (
                    ("row", self.grid.edge_row_clamped),
                    ("column", self.grid.edge_col_clamped),
                )
                if on
            )
            parts.append(
                f"The last {which} of windows was pulled back to the edge so "
                f"that no strip of the image went unmeasured; it therefore "
                f"overlaps its neighbour by more than the stride."
            )
        if self.batch_requested != self.batch_used:
            parts.append(
                f"A batch of {self.batch_requested} was asked for and "
                f"{self.batch_used} was used, which is this module's bound on "
                f"how many full-size copies of the image it will hold at once."
            )
        if self.class_names_dropped:
            parts.append(
                f"The class names supplied did not match this head's "
                f"{self.classes} outputs, so ALL of them were dropped rather "
                f"than applied by position — a mislabelled class is worse than "
                f"an unlabelled one."
            )

        parts.append(
            "ONE IMAGE IS A SAMPLE, NOT A PROPERTY OF THE MODEL. This says "
            "where the evidence was in this picture; it does not say what the "
            "model attends to in general, and the next picture can rank "
            "differently."
        )
        return " ".join(parts)


# ------------------------------------------------------------------ the sweep


def _one_image(image):
    """Validate the input and return it as `[1, C, H, W]` floats.

    Refuses rather than reshaping anything ambiguous. A batch of images with
    one attribution map drawn over it would be a map of whichever image the
    indexing happened to pick.
    """
    import torch

    if not isinstance(image, torch.Tensor):
        raise BadRequest(
            f"image must be a torch tensor shaped [C, H, W] or [1, C, H, W], "
            f"not {type(image).__name__}. This module does no image loading of "
            f"its own, so what the model is shown is exactly what you built."
        )
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise BadRequest(
            f"image has shape {tuple(image.shape)}. This expects [C, H, W] or "
            f"[1, C, H, W]."
        )
    if int(image.shape[0]) != 1:
        raise BadRequest(
            f"image carries a batch of {int(image.shape[0])}. Attribution is "
            f"about ONE image — a single map drawn over several would be a map "
            f"of whichever one the indexing happened to pick."
        )
    if not image.is_floating_point():
        raise BadRequest(
            f"image is of dtype {image.dtype}, and an integer tensor cannot "
            f"hold the fill: a grey of 0.5 truncates to 0, which is black. "
            f"Convert to float first."
        )
    if not bool(torch.isfinite(image).all()):
        raise NotAttributable(
            "this image contains a NaN or an infinity, so every occluded copy "
            "of it would carry one too and every score would be a difference "
            "between things that are not numbers."
        )
    return image


def _as_logits(output, where: str):
    """Pull a `[N, K]` tensor out of whatever the model returned.

    Four shapes are read because four are common: a bare tensor, a
    transformers `ModelOutput` with `.logits`, a dict, and a tuple whose first
    element is the tensor. Anything else refuses and names `forward=`, which
    is the hook a detector or a multi-head model is meant to use — guessing
    which of several returned tensors is "the" output would be picking the
    quantity the map is about by accident.
    """
    import torch

    found = output
    if hasattr(found, "logits"):
        found = found.logits
    elif isinstance(found, dict):
        if "logits" not in found:
            raise NotAttributable(
                f"{where} returned a dict with keys "
                f"{sorted(str(k) for k in found)} and no `logits`. Pass "
                f"`forward=` to say which of them the map should be about."
            )
        found = found["logits"]
    elif isinstance(found, (tuple, list)):
        if not found:
            raise NotAttributable(f"{where} returned an empty sequence.")
        found = found[0]

    if not isinstance(found, torch.Tensor):
        raise NotAttributable(
            f"{where} returned {type(output).__name__}, which this cannot read "
            f"as logits. Pass `forward=lambda model, x: ...` returning a "
            f"[batch, classes] tensor — a detector's box scores or one head of "
            f"several is exactly what that hook is for."
        )
    if found.ndim != 2:
        raise NotAttributable(
            f"{where} returned logits shaped {tuple(found.shape)}. This needs "
            f"[batch, classes]; pass `forward=` to reduce a richer output — "
            f"choosing an axis here would pick what the map is about by "
            f"accident."
        )
    return found.float()


def _fill_tensor(image, fill: str, lo: float, hi: float):
    """What goes inside the occluder, in the image's own value space."""
    import torch

    if fill == "image_mean":
        # Mean over the SPATIAL axes only, giving one colour per channel:
        # [1, C, 1, 1], which broadcasts into any window. `vla_occlude` was
        # corrected for averaging over the channel axis instead and producing
        # a "colour" that varied along the width; the same mistake is
        # invisible here too, because a scalar fill would broadcast anyway.
        return image.mean(dim=(-2, -1), keepdim=True)
    if fill == "black":
        value = lo
    elif fill == "white":
        value = hi
    elif fill == "grey":
        value = (lo + hi) / 2.0
    else:
        raise BadRequest(
            f"unknown fill {fill!r} — expected one of {FILLS}. The fill "
            f"changes the answer, so this refuses rather than quietly using "
            f"the default."
        )
    return torch.tensor(value, dtype=image.dtype, device=image.device)


def sweep(
    model,
    image,
    *,
    target: int | None = None,
    patch: int = DEFAULT_PATCH,
    stride: int | None = None,
    fill: str = DEFAULT_FILL,
    value_range: tuple[float, float] | None = None,
    batch: int = DEFAULT_BATCH,
    forward=None,
    class_names: list[str] | None = None,
    max_passes: int = MAX_PASSES,
    model_name: str = "",
) -> Attribution:
    """Occlude every window of one image and report what the output did.

    `model` is treated as a black box: it is called with a `[N, C, H, W]`
    tensor and expected to return `[N, K]` logits, so a ViT, a CNN and a
    detector are all the same thing here. `forward(model, x)` overrides that
    call for anything whose output needs reducing first.

    Everything is under `torch.no_grad()`. No graph is retained, which is what
    makes a 64-wide batch of full-size images affordable — and it is also why
    this cannot be turned into a gradient method by accident.
    """
    import torch

    started = time.perf_counter()

    # A model left in training mode has live dropout and batch-norm running
    # statistics, so the SAME input twice is two different answers and every
    # score in the map would be that noise plus the occlusion. Refused rather
    # than silently switched: flipping somebody's model to eval as a side
    # effect of drawing a picture is a change to their object that they did
    # not ask for, and the fix is one call they can see.
    if getattr(model, "training", False) is True:
        raise NotAttributable(
            "this model is in training mode, where dropout and batch-norm make "
            "the same input give a different answer each time — every score in "
            "the map would be that noise as much as the occlusion. Call "
            "`model.eval()` first. This will not do it for you, because "
            "changing your model as a side effect of drawing a picture is not "
            "this function's business."
        )

    image = _one_image(image)
    height, width = int(image.shape[-2]), int(image.shape[-1])
    grid = plan_windows(
        height, width, patch=patch, stride=stride, max_passes=max_passes
    )
    batch_requested, batch_used = _clamp_batch(batch)

    inferred = value_range is None
    if inferred:
        lo, hi = float(image.min()), float(image.max())
        if hi <= lo:
            raise NotAttributable(
                f"every pixel of this image is {lo:,.4f}, so it has no range to "
                f"draw a fill from: grey, black and white would all be the "
                f"pixel that is already there and the whole map would read as "
                f"a flat zero, which looks exactly like a model that ignores "
                f"its input. Pass `value_range` if the image really is "
                f"constant and you know the model's input range."
            )
    else:
        lo, hi = float(value_range[0]), float(value_range[1])
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            raise BadRequest(
                f"value_range must be (low, high) with high greater than low, "
                f"not ({lo!r}, {hi!r})."
            )

    if fill not in FILLS:
        raise BadRequest(
            f"unknown fill {fill!r} — expected one of {FILLS}. The fill "
            f"changes the answer, so this refuses rather than quietly using "
            f"the default."
        )
    fill_value = _fill_tensor(image, fill, lo, hi)

    def run(x):
        with torch.no_grad():
            out = forward(model, x) if forward is not None else model(x)
        return _as_logits(out, "the model" if forward is None else "`forward`")

    base = run(image)
    calls = 1
    classes = int(base.shape[1])
    if classes < 1:
        raise NotAttributable(
            "the model returned zero classes, so there is nothing to attribute."
        )
    if not bool(torch.isfinite(base).all()):
        raise NotAttributable(
            "the unoccluded run produced a non-finite logit, so there is no "
            "reference for anything else to be measured against."
        )

    if target is None:
        chosen_by_model = True
        target = int(base[0].argmax())
    else:
        chosen_by_model = False
        target = _as_int(target, "target")
        if not 0 <= target < classes:
            raise BadRequest(
                f"target {target} is outside this head's {classes} classes "
                f"(0 to {classes - 1}). Leave it out to attribute the model's "
                f"own top prediction instead."
            )

    names = list(class_names or [])
    names_dropped = bool(names) and len(names) != classes
    if names_dropped:
        # Dropped entirely rather than indexed into: a name list of the wrong
        # length mislabels at least one class, and a map captioned with the
        # wrong class name is worse than one captioned with a number.
        names = []
    label = str(names[target]) if names else ""

    base_logit = float(base[0, target])
    # A softmax over a single output is 1.0 by construction and says nothing,
    # so it is None here rather than 1.0 -- reporting "confidence 1.00" for a
    # one-output head would be a number with no information in it.
    base_prob = float(torch.softmax(base[0], dim=-1)[target]) if classes > 1 else None

    plan = grid.windows()
    rows: list[Window] = []
    for start in range(0, len(plan), batch_used):
        chunk = plan[start : start + batch_used]
        # One allocation per call, `len(chunk)` copies of the image, released
        # when the loop moves on. That is the memory this module was priced
        # at, and it is why the batch is bounded rather than "all of them".
        stack = image.repeat(len(chunk), 1, 1, 1)
        for i, (_r, _c, top, left) in enumerate(chunk):
            stack[i, :, top : top + grid.patch, left : left + grid.patch] = fill_value
        moved = run(stack)
        calls += 1
        if not bool(torch.isfinite(moved).all()):
            raise NotAttributable(
                "an occluded run produced a non-finite logit, so part of this "
                "map would be a hole. A gap in a heatmap reads as 'this region "
                "did nothing', which is not what happened."
            )
        probs = torch.softmax(moved, dim=-1) if classes > 1 else None
        for i, (r, c, top, left) in enumerate(chunk):
            rows.append(
                Window(
                    row=r,
                    col=c,
                    top=top,
                    left=left,
                    height=grid.patch,
                    width=grid.patch,
                    logit_drop=round(
                        base_logit - float(moved[i, target]), SCORE_DECIMALS
                    ),
                    prob_drop=(
                        None
                        if probs is None or base_prob is None
                        else round(base_prob - float(probs[i, target]), 8)
                    ),
                )
            )
        del stack, moved

    return Attribution(
        grid=grid,
        windows=rows,
        fill=fill,
        fill_value=[
            round(float(v), SCORE_DECIMALS) for v in fill_value.reshape(-1).tolist()
        ],
        value_range=(round(lo, SCORE_DECIMALS), round(hi, SCORE_DECIMALS)),
        value_range_inferred=inferred,
        target=target,
        target_label=label,
        target_chosen_by_model=chosen_by_model,
        classes=classes,
        base_logit=round(base_logit, SCORE_DECIMALS),
        base_prob=None if base_prob is None else round(base_prob, 8),
        passes=grid.passes,
        forward_calls=calls,
        batch_requested=batch_requested,
        batch_used=batch_used,
        seconds=round(time.perf_counter() - started, 3),
        model_name=str(model_name or ""),
        class_names_dropped=names_dropped,
    )
