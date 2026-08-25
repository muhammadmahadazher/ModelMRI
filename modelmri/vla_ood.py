"""How far outside its own dataset each frame of an episode sits — against a named reference set.

"Frame 40 is unusual" is not a finding. Unusual *compared to what* is the whole
measurement, and the reference set is therefore the first field in every payload
this module produces rather than a footnote under the chart.

WHICH DISTRIBUTION, AND WHY THIS ONE
------------------------------------
Four candidates were on the table and only one of them is the thing a policy was
actually trained on:

  the recorded VIDEO FRAMES     pixels. A distance in pixel space is dominated
                                by lighting and by the camera's own noise, and
                                two visually identical scenes recorded an hour
                                apart are far apart in it. It also needs the
                                video decoded, which is the one part of a
                                LeRobot snapshot that is expensive and the one
                                part that fails (see `frames_readable`).

  the POLICY'S EMBEDDING        the tower's geometry, not the dataset's. It
  of those frames               answers "unusual to this encoder", which is a
                                real question and is `vla_occlude`'s and
                                `checkpoints.py`'s question, not this one. It
                                also needs a loaded policy, and `pip install
                                modelmri` does not bring the sidecar — so the
                                measurement would be unavailable on exactly the
                                machines that can still read the dataset.

  the recorded ACTIONS          the labels. "This commanded action is unusual"
                                is a fair question and this module will answer
                                it on request (`space="action"`), but it is not
                                the same question: an action is what the
                                demonstrator did, not what the policy was
                                conditioned on.

  the recorded STATE VECTORS    what the policy SEES, in the units the robot
  <- this one, by default       reported them in, for every frame of every
                                episode in the snapshot. It is the training
                                distribution in the most literal sense
                                available: these rows are the ones that went
                                through the dataloader.

So the reference set is **the dataset's own per-frame `observation.state` rows**,
and any other per-frame numeric column can be named instead. Every payload
carries `reference.space`, `reference.repo_id`, `reference.rows_read` and
`reference.rows_eligible`, because a distance without those four is a number
about nothing.

WHAT THIS COSTS: ZERO FORWARD PASSES
------------------------------------
`estimate()` prices a run the way `budget.py` and `vla_occlude.estimate` do, and
the honest figure is zero — nothing here runs a model. NO PART of this module
needs a loaded policy, a vision tower, a GPU, `av`, or the lerobot sidecar. It
needs pyarrow (the `vla-lite` extra) and torch (a base dependency, used here only
for a d x d eigendecomposition and some batched arithmetic). The real cost is
parquet rows read, and `estimate()` reports it in that unit rather than dressing
it up as passes it does not take.

The consequence is a stated blind spot rather than a free lunch: a frame that is
visually bizarre while its joints sit in an ordinary configuration is at an
ordinary distance here. This measures one recorded column and says so in every
sentence it writes.

A DISTANCE IS NOT A VERDICT
---------------------------
There is no `is_ood` boolean anywhere in this module and there will not be one: a
boolean is a threshold somebody chose, and this project does not ship those. What
travels instead is

  the DISTANCE            Mahalanobis, in the reference set's own units of
                          spread — so a joint recorded in pixel coordinates
                          around 512 and a gripper flag in [0,1] contribute on
                          the same scale.
  its PERCENTILE          the share of the reference rows AT OR BELOW this
                          frame's distance from the reference mean — ties
                          included, which is what makes a frame sitting exactly
                          on the reference maximum read 100.0. Exact, read off
                          the reference distances themselves, with a resolution
                          of 100/rows_read percentage points.
  the REFERENCE'S OWN     50th through 100th, so a reader can see the shape of
  distance percentiles    the distribution the percentile was taken in.

Mahalanobis rather than a per-dimension z-score, and the difference is the point
rather than a refinement. A robot's state dimensions are strongly correlated: an
arm pose can be inside +/-1 standard deviation on every single joint and still be
a configuration the arm never once held, because the joints never held those
values TOGETHER. A diagonal metric is blind to exactly that case, which is the
case worth finding. MEASURED on the fixture `tests/test_vla_ood.py` builds for
it — two joints on a tight ridge, 400 reference rows, and one frame off it:

    per-dimension z-score      0.8639 sigma on dim 0, 0.8639 on dim 1
    this metric                141.59
    largest reference distance   2.21

Both readings are of the same frame and the same rows. One of them says it is
unremarkable.

AND ONE FLAG, GATED ON A MEASURED NULL
--------------------------------------
`head_types.py` refuses to attach a label until the head clears a null measured
on this model, and `nullmodel.py` exists because a pipeline that produces a
confident ordered list on an untrained network produces one on anything. The same
question applies here: **would this frame have looked far away anyway?** Some
share of any distribution's own rows sit in its tail — that is what a tail is —
so a frame at the 99.4th percentile of the reference distances is not remarkable,
it is the 99.4th percentile.

So the null is measured, from the dataset itself: rows are held back OUT of the
reference set entirely, scored against the reference built from the others, and
the LARGEST distance any of them reached is `reference.null_max`. A frame's
`clears_null` is `distance > null_max` and nothing else. It is `None` — never
`False` — when no null could be drawn, which happens whenever the reference set is
every eligible row and nothing is left over; the payload then carries
`null_reason` saying so and what to change.

TWO numbers travel with that null, and they are different quantities:

  null_covers_percentile   100*K/(K+1), the percentile a maximum of K draws is
                           EXPECTED near. It is computed from the draw count and
                           from nothing else, so it is the null's RESOLUTION —
                           at 8 draws it is the 88.9th, which is very little to
                           clear, and saying so is what stops "it beat the null"
                           carrying the same weight at every K.
  null_max_percentile      where `null_max` actually landed among the reference
                           distances, read off them exactly by the same
                           `percentile_of` every frame uses.

They are reported separately because they disagree, and the disagreement is
informative rather than noise: on the fixture `tests/test_vla_ood.py` builds, at
100 draws, the expected figure is the 99.01th percentile and the measured one is
the 100th and past every reference row. The reference distances are IN-SAMPLE —
the mean and the covariance were fitted on those very rows — while every null row
is out of that fit, so the null's distances run a little larger for that reason
alone. Quoting the expected figure as though it were the measured one, which this
module did until it was checked, understates how far outside the fit the null
already sits.

THE EPISODE UNDER EXAMINATION IS HELD OUT OF ITS OWN REFERENCE
--------------------------------------------------------------
By default every row of the scored episode is excluded from the reference set,
from the covariance, and from the null. Otherwise a frame is being compared
against a distribution it helped define, and a long episode in a small dataset
partly defines the mean it is then measured from. A caller may pass a reference
built with a different exclusion, or none — `reference.excluded_episode` says
which, and `means()` names the contamination when the two disagree.

A FRAME THAT COULD NOT BE SCORED IS ABSENT, NOT ZERO
-----------------------------------------------------
`vla_sweep.run` established the pattern and this is the same one: a frame whose
row is unreadable, the wrong width, or non-finite does not get a distance of 0.0
— a zero would sit at the bottom of the ranking looking like a measurement of an
extremely ordinary frame. It goes into `unscored` with a sentence, the listing is
capped at `MAX_UNSCORED_LISTED`, and `n_unscored` carries the true count beside
the truncated list.

MEMORY
------
Bounded and MEASURED, and the two are not the same claim. Two streaming passes
over the frame shards plus one targeted read of the shards the scored episode
lives in, one arrow batch resident at a time, plus a d x d covariance
accumulator, one float per sampled reference row, and the scored episode's own
vectors.

The figures below are tracemalloc peaks of a whole `score_episode`, taken with
the lazy imports already warmed so that only the run is counted. tracemalloc
counts PYTHON allocations; the arrow buffer under each batch and every torch
tensor are outside it and sit on top.

    14-dimension state, 512-frame episode, the default 20,000 reference
    rows and 256 null draws                             5,927,140 B (5.93 MB)
    the same, priced by `estimate()` beforehand         6,800,480 B (6.80 MB)

FLAT IN THE ROW COUNT, which is the whole streaming claim and is therefore
measured rather than asserted — a 14-wide column, `max_reference_rows=None`:

    20,000 reference rows                               5,915,843 B
    60,000 reference rows                               5,915,224 B
    100,000 reference rows                              5,917,044 B

WIDTH is what moves it, because one batch of 8,192 rows is 8,192 rows of Python
floats: 1.79 MB at one dimension, 5.92 MB at fourteen, 22.04 MB at sixty-four.
`estimate()` returns a ceiling at every one of those (1.03x-1.09x) and says so.
At the hard caps — 500,000 reference rows and 5,000 frames — `estimate()` returns
17.8 MB for a 14-wide column and 44.3 MB for a 64-wide one; those two are
computed by `_peak_bytes` rather than measured, and a figure quoted without its
width is the reason the sentence that used to sit here said 30 MB.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import fmt
from .errors import BadRequest, Refusal
from .fmt import ordinal as _ordinal

# THREE THINGS BORROWED FROM `vla_data`, EACH SO THERE IS ONE OF IT RATHER
# THAN TWO.
#
#   ACTION_ROW_BATCH   its comment carries a tracemalloc measurement showing the
#                      peak is flat in the row count at this batch size, on the
#                      identical operation: pyarrow hands back one batch of one
#                      column, it is folded into accumulators, it is dropped.
#   _as_row            what counts as one readable recorded vector. An empty
#                      list is a row with nothing in it rather than a
#                      zero-dimensional observation; a dict is not a vector; a
#                      bare scalar is a one-dimensional one. A row that
#                      `dataset_action_stats` calls malformed has to be
#                      malformed here too or the two disagree about the same
#                      dataset.
#   _read_all          concatenates every parquet shard rather than reading the
#                      first, which is the trap its own docstring names.
from .vla_data import ACTION_ROW_BATCH, _as_row, _read_all

# The column whose distribution this measures unless told otherwise. The state is
# what the policy is conditioned on; `action` is a label and a different
# question, available by name.
DEFAULT_SPACE = "observation.state"

# How many reference rows are read before the sample is strided. 20,000 rows of a
# 14-dimensional state is 1,428 rows per dimension, which is a covariance
# estimate nobody has to caveat; `rows_per_dimension` travels in the payload so a
# reader can judge a thinner one.
DEFAULT_MAX_REFERENCE_ROWS = 20_000

# The ceiling on that. 500,000 float64 distances is 4 MB, the largest single
# allocation this module will make, and it is reported.
MAX_REFERENCE_ROWS = 500_000

# Rows held back OUT of the reference set to measure the null with. 256 draws put
# the null at about the 99.6th percentile of the in-distribution distances;
# `null_covers_percentile` states it exactly for the run that was made.
DEFAULT_NULL_DRAWS = 256
MAX_NULL_DRAWS = 4_096

# Bins in the reference distance histogram. For DISPLAY only — every percentile
# in this module is read off the exact sorted distances rather than off these
# bins, so the bin width is nobody's resolution here. That is the one difference
# from `dataset_action_stats`, which cannot hold its values and says so.
DEFAULT_HISTOGRAM_BINS = 32
MAX_HISTOGRAM_BINS = 512

# Frames scored in one call. Each is a vector held in memory and a row in the
# payload; past this the answer is refused with the number rather than cut short,
# for the reason `vla_sweep.plan` gives — a ranking missing its tail looks
# exactly like a ranking.
MAX_FRAMES = 5_000

# How many unscored frames are LISTED. `n_unscored` carries the true count.
MAX_UNSCORED_LISTED = 20

# How many rows the `ranked` list holds. `n_ranked_total` carries the true count.
DEFAULT_RANKED = 20

# Which percentiles of the REFERENCE distances travel, so a reader can see the
# distribution a frame's percentile was taken in. 100 is included deliberately:
# it is the largest distance any reference row reached, and it is the number a
# frame has to beat to be beyond everything the reference set contains.
REFERENCE_PERCENTILES: tuple[float, ...] = (50.0, 75.0, 90.0, 95.0, 99.0, 100.0)

# A direction of the reference covariance is treated as degenerate below this
# share of the largest eigenvalue. Not a statistical threshold — a conditioning
# one: float64 carries about 16 significant digits, `eigh` returns eigenvalues
# with a relative error near machine epsilon times the condition number, and
# dividing by an eigenvalue 1e-8 of the largest amplifies that error by 1e4 while
# still leaving eight digits. Below it the "distance" along that direction is
# reporting rounding. Both the ratio and the resulting absolute floor travel in
# the payload, with the count of directions it dropped.
CONDITION_FLOOR = 1e-8


class OODError(BadRequest):
    """This scoring cannot be done honestly on these parameters, and we say why.

    A sibling of `vla_occlude.OcclusionError` and `vla_sweep.SweepError`, and the
    same 422: it is about the call that was just made. Conditions that are about
    the DATASET rather than the request — no such column anywhere, no rows under
    it, every row identical — raise `Refusal` (409) directly, the way `vla_data`
    does for the same class of fact.
    """


# ------------------------------------------------------------------ the frames


@dataclass
class FrameScore:
    """One frame of the episode, scored against the reference set."""

    t: int
    # Mahalanobis distance from the reference mean, over the directions the
    # reference set actually varies in. Units: that reference set's own spread.
    distance: float
    # Share of the reference rows AT OR BELOW this frame's distance from the
    # reference mean — the empirical CDF over the reference distances, exact by
    # construction and computed by `Reference.percentile_of`.
    #
    # "At or below", not "strictly closer", and the two differ by exactly the
    # rows that tie: a distance equal to the 11th smallest of 100 reference
    # distances reads 11.0 here and 10.0 under the strict reading. This line said
    # "closer than" while `percentile_of`'s own docstring said "at or below",
    # which is one number described two ways in one payload.
    percentile: float
    # 100 / rows_read, in percentage points. A percentile taken in a sample of N
    # cannot be resolved finer than one row, and this says what one row is worth.
    # Not a bin width — there are no bins in this number.
    percentile_resolution: float
    # True when this frame is further out than every row of the reference set.
    # The percentile then reads 100.0, and a reader needs to know that means
    # "beyond all of them" rather than "measured at exactly the top".
    beyond_reference_max: bool
    # Movement along the directions the reference set NEVER varied in, in the
    # column's own raw units. There is no spread to divide by there, so it is not
    # folded into `distance` — it is a different quantity on a different scale
    # and it is reported beside. `None` when the reference varied everywhere.
    off_manifold: float | None
    # `distance` beat the largest distance any held-out in-distribution row
    # reached. `None` — never False — when no null could be drawn.
    clears_null: bool | None

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------- the reference


@dataclass
class Reference:
    """The distribution every distance is measured against, and the metric itself.

    The published half is what `to_dict` returns. The unpublished half is four
    torch tensors — the mean, the whitening basis, its degenerate complement, and
    the sorted reference distances — which is why `to_dict` names its fields one
    by one instead of calling `asdict`. `asdict` deep-copies what it walks, so it
    would both put a d x d basis and a 4 MB distance array into a JSON payload
    and copy them to do it. They live on this object rather than beside it
    because a report and a transform that can be paired with the wrong report is
    a shape this codebase has already been bitten by.
    """

    repo_id: str
    #: The column these rows came from. In the payload because the whole
    #: measurement is meaningless without it.
    space: str
    #: Width taken from the first readable row, beside what `info.json` claims.
    #: Both, for the reason `dataset_action_stats` gives: when they disagree one
    #: of the two is wrong and nothing here can tell which.
    dimensions: int
    dimensions_declared: int | None
    rows_total: int
    rows_with_column: int
    #: Rows carrying the column and NOT belonging to the excluded episode.
    rows_eligible: int
    #: Rows actually folded into the mean and covariance.
    rows_read: int
    rows_malformed: int
    #: 1 when every eligible row was read. Above 1 this is a SAMPLE, and
    #: `sampled` says so in one field rather than making a reader compare two.
    row_stride: int
    sampled: bool
    excluded_episode: int | None
    excluded_rows: int
    #: Where the episode row spans came from: the dataset's own
    #: `dataset_from_index`, or summed episode lengths when it publishes none.
    #: The second is an assumption and is named as one.
    row_span_from: str
    rows_per_dimension: float
    directions_kept: int
    directions_dropped: int
    variance_floor: float
    condition_ratio: float
    metric: str
    #: min / max / mean of the reference distances, their percentiles, and a
    #: histogram for drawing. The percentiles are exact.
    distances: dict = field(default_factory=dict)
    percentile_resolution: float = 0.0
    null_max: float | None = None
    null_draws: int = 0
    #: The percentile the maximum of `null_draws` draws is EXPECTED to sit at,
    #: 100*K/(K+1), computed from the draw COUNT and from nothing else. It is the
    #: null's RESOLUTION — what raising `null_draws` buys — and it is not a
    #: statement about where `null_max` actually landed. `None` when no null was
    #: drawn.
    null_covers_percentile: float | None = None
    #: Where `null_max` ACTUALLY landed, read off the reference distances this
    #: object holds exactly, by the same `percentile_of` every frame uses. These
    #: two used to be conflated in one sentence, and on this module's own
    #: flagship fixture they disagreed by the whole top percentile: expected
    #: 99.01, actual 100.0 and past every reference row.
    null_max_percentile: float | None = None
    null_max_beyond_reference_max: bool | None = None
    #: Why the two differ by more than sampling noise, stated rather than left
    #: for a reader to rediscover.
    null_position_caveat: str = ""
    null_reason: str = ""
    null_description: str = ""

    # ---- the unpublished half. Set by `build_reference`, never serialised.
    mean_vector: object = None
    kept_basis: object = None
    inv_sqrt: object = None
    dropped_basis: object = None
    sorted_distances: object = None

    # ---- the metric

    def distances_of(self, matrix):
        """`(mahalanobis, off_manifold_or_None)` for an `n x d` float64 tensor.

        Batched because the reference pass calls it with thousands of rows at a
        time, and a per-row Python loop over torch scalars costs more than the
        arithmetic it is wrapping.
        """
        import torch

        centred = matrix - self.mean_vector
        projected = centred @ self.kept_basis
        mahalanobis = torch.linalg.vector_norm(projected * self.inv_sqrt, dim=1)
        off = None
        if self.dropped_basis is not None and int(self.dropped_basis.shape[1]):
            off = torch.linalg.vector_norm(centred @ self.dropped_basis, dim=1)
        return mahalanobis, off

    def score(self, values: Sequence[float]) -> tuple[float, float | None]:
        """One vector's `(distance, off_manifold_or_None)`."""
        import torch

        mahalanobis, off = self.distances_of(
            torch.tensor([list(values)], dtype=torch.float64)
        )
        return float(mahalanobis[0]), (None if off is None else float(off[0]))

    def percentile_of(self, distance: float) -> tuple[float, bool]:
        """`(percentile, beyond_the_reference_maximum)` for one distance.

        The empirical CDF over the reference distances: the share of them at or
        below `distance`. Exact — the distances are held, sorted and searched, so
        there is no interpolation and no bin to be inside of. Its resolution is
        `percentile_resolution`, which is what one row of the sample is worth in
        percentage points.

        `beyond` travels beside it because the CDF genuinely reads 100.0 for a
        frame further out than everything in the reference set, and 100.0 with no
        flag beside it reads as "measured at exactly the top" rather than "past
        the end of what was measured".
        """
        import torch

        found = torch.searchsorted(
            self.sorted_distances,
            torch.tensor([float(distance)], dtype=torch.float64),
            right=True,
        )
        n = int(self.sorted_distances.numel())
        beyond = float(distance) > float(self.sorted_distances[-1])
        return round(100.0 * int(found[0]) / n, 6), beyond

    def clears(self, distance: float) -> bool | None:
        """Did this distance beat the measured null? `None` when there was none.

        `None` rather than `False`, and the distinction is the reason the null is
        reported separately at all: "this frame did not beat the null" and "no
        null was drawn, so nothing was tested" are opposite statements about how
        much is known, and a `False` in both places erases the difference.
        """
        return None if self.null_max is None else float(distance) > self.null_max

    def to_dict(self) -> dict:
        """The described half. The four tensors above never travel."""
        return {
            "repo_id": self.repo_id,
            "space": self.space,
            "dimensions": self.dimensions,
            "dimensions_declared": self.dimensions_declared,
            "rows_total": self.rows_total,
            "rows_with_column": self.rows_with_column,
            "rows_eligible": self.rows_eligible,
            "rows_read": self.rows_read,
            "rows_malformed": self.rows_malformed,
            "row_stride": self.row_stride,
            "sampled": self.sampled,
            "excluded_episode": self.excluded_episode,
            "excluded_rows": self.excluded_rows,
            "row_span_from": self.row_span_from,
            "rows_per_dimension": self.rows_per_dimension,
            "directions_kept": self.directions_kept,
            "directions_dropped": self.directions_dropped,
            "variance_floor": self.variance_floor,
            "condition_ratio": self.condition_ratio,
            "metric": self.metric,
            "distances": self.distances,
            "percentile_resolution": self.percentile_resolution,
            "null_max": self.null_max,
            "null_draws": self.null_draws,
            "null_covers_percentile": self.null_covers_percentile,
            "null_max_percentile": self.null_max_percentile,
            "null_max_beyond_reference_max": self.null_max_beyond_reference_max,
            "null_position_caveat": self.null_position_caveat,
            "null_reason": self.null_reason,
            "null_description": self.null_description,
        }


# ------------------------------------------------------------------ the report


@dataclass
class EpisodeOOD:
    """One episode, frame by frame, against a named reference set."""

    repo_id: str
    episode: int
    space: str
    frame_stride: int
    reference: Reference
    #: Every scored frame, in TIME order — this is what a per-frame chart plots.
    frames: list[FrameScore] = field(default_factory=list)
    #: The most distant frames, capped. `n_ranked_total` is how many were scored.
    ranked: list[FrameScore] = field(default_factory=list)
    n_ranked_total: int = 0
    n_frames: int = 0
    #: Frames in the episode, before the stride and before anything failed.
    frames_total: int = 0
    #: A SAMPLE of what could not be scored, capped at `MAX_UNSCORED_LISTED`.
    unscored: list[dict] = field(default_factory=list)
    n_unscored: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "episode": self.episode,
            "space": self.space,
            "frame_stride": self.frame_stride,
            "reference": self.reference.to_dict(),
            "frames": [f.to_dict() for f in self.frames],
            "ranked": [f.to_dict() for f in self.ranked],
            "n_ranked_total": self.n_ranked_total,
            "n_frames": self.n_frames,
            "frames_total": self.frames_total,
            "unscored": self.unscored,
            "n_unscored": self.n_unscored,
            "seconds": self.seconds,
            "means": self.means(),
        }

    def means(self) -> str:
        ref = self.reference
        sampling = (
            f"every {_ordinal(ref.row_stride)} eligible row"
            if ref.sampled
            else "every eligible row"
        )
        parts = [
            f"Episode {self.episode} of {self.repo_id or 'this dataset'}, "
            f"{self.n_frames} of its {self.frames_total} frames scored at stride "
            f"{self.frame_stride}. THE REFERENCE SET IS {ref.rows_read} ROWS of "
            f"`{ref.space}` from this same dataset — {sampling} of "
            f"{ref.rows_eligible} eligible, out of {ref.rows_with_column} "
            f"carrying the column and {ref.rows_total} in the snapshot. The "
            f"distance is {ref.metric}, so it is in units of that reference "
            f"set's own spread and nothing else."
        ]
        if ref.excluded_episode == self.episode:
            parts.append(
                f"This episode's own {ref.excluded_rows} rows were held OUT of "
                f"the reference set, its mean, its covariance and its null — a "
                f"frame compared against a distribution it helped define is "
                f"partly measuring itself."
            )
        elif ref.excluded_episode is None:
            parts.append(
                f"NOTHING WAS EXCLUDED from this reference set, so episode "
                f"{self.episode}'s own rows are inside the distribution it is "
                f"being measured against. Build the reference with "
                f"`exclude_episode={self.episode}` to take them out."
            )
        else:
            parts.append(
                f"The reference set excluded episode {ref.excluded_episode}, NOT "
                f"episode {self.episode} — so this episode's own rows are inside "
                f"the distribution it is being measured against."
            )
        if ref.directions_dropped:
            parts.append(
                f"{ref.directions_dropped} of {ref.dimensions} directions were "
                f"DROPPED from the distance: the reference set does not move "
                f"along them at all (eigenvalue at or below "
                f"{fmt.measured(ref.variance_floor, 4)}), and there is no spread "
                f"there to divide by. Movement along them is reported separately "
                f"as `off_manifold`, in the column's raw units, because it is "
                f"not on the same scale as the rest."
            )
        if self.frames:
            top = max(self.frames, key=lambda f: f.distance)
            parts.append(
                f"The furthest frame is t={top.t} at "
                f"{fmt.measured(top.distance, 4)}, which is the "
                f"{top.percentile:.3f}th percentile of the reference distances "
                f"(one row of the sample is {top.percentile_resolution:.3f} of a "
                f"point, and that is the resolution)"
                + (
                    " — further out than every row in the reference set."
                    if top.beyond_reference_max
                    else "."
                )
            )
        if ref.null_max is None:
            parts.append(
                f"NO NULL WAS MEASURED on this run, so nothing is flagged and "
                f"`clears_null` is null for every frame rather than false. "
                f"{ref.null_reason}"
            )
        else:
            cleared = [f for f in self.frames if f.clears_null]
            landed = (
                "past every one of the reference distances"
                if ref.null_max_beyond_reference_max
                else f"at the {ref.null_max_percentile:.3f}th percentile of them"
            )
            parts.append(
                f"THE NULL: {ref.null_description} It reached "
                f"{fmt.measured(ref.null_max, 4)}, which sits {landed} "
                f"({ref.null_max_percentile:.3f}, read off the "
                f"{ref.distances.get('count', ref.rows_read)} reference distances "
                f"themselves rather than assumed). The maximum of "
                f"{ref.null_draws} draws is EXPECTED near the "
                f"{ref.null_covers_percentile:.2f}th percentile — that is the "
                f"null's own resolution, and raising `null_draws` is what "
                f"sharpens it, but it is a figure from the draw count and not a "
                f"reading of this null. "
                + (
                    f"{len(cleared)} of {self.n_frames} scored frames beat it."
                    if cleared
                    else f"NONE of the {self.n_frames} scored frames beat it: on "
                    f"this evidence no frame of this episode is further from the "
                    f"dataset's centre than rows drawn from the dataset itself "
                    f"get."
                )
            )
        if self.n_ranked_total > len(self.ranked):
            # The third cap in this payload, and the only one that was applied
            # without a sentence. `n_ranked_total` carried the true count, but a
            # reader looking at a chart is reading this and not the JSON — and a
            # capped answer that cannot say what it capped is indistinguishable
            # from a complete one.
            parts.append(
                f"THE RANKING IS CAPPED: `ranked` holds the "
                f"{len(self.ranked)} furthest of {self.n_ranked_total} scored "
                f"frames. Every one of the {self.n_frames} is in `frames`, in "
                f"time order — the cap is on the ranked listing only."
            )
        if self.n_unscored:
            listed = (
                ""
                if self.n_unscored <= len(self.unscored)
                else f" ({len(self.unscored)} of them listed)"
            )
            parts.append(
                f"{self.n_unscored} frame(s) could not be scored and are ABSENT "
                f"from the ranking rather than scored zero — a row that could not "
                f"be read is not a frame at distance nothing{listed}."
            )
        parts.append(
            f"A DISTANCE IS NOT A VERDICT. Nothing here is labelled "
            f"out-of-distribution, because that word is a threshold and this "
            f"reports a distance, its percentile in a named reference set, and "
            f"whether it beat a measured null. And it is a distance in "
            f"`{ref.space}` ONLY: a frame that is visually bizarre while its "
            f"recorded vector sits in an ordinary place reads as ordinary here, "
            f"and no vision model was run."
        )
        return " ".join(parts)


# ----------------------------------------------------------------- the plumbing


def _files(reader) -> list[Path]:
    """Every frame shard, in the order that defines the global row index.

    Sorted, because `dataset_from_index` and `_frame_table` both number rows by
    the concatenation of the shards in exactly this order. A different order here
    would map an episode's span onto somebody else's rows, silently.
    """
    return sorted((Path(reader.snapshot) / "data").rglob("*.parquet"))


def _shard_map(
    files: list[Path], column: str
) -> tuple[list[tuple[int, int, bool]], list[str]]:
    """`[(base_row, n_rows, has_column)]` per shard, from the parquet FOOTERS.

    Footers only, the way `dataset_action_stats` reads them: the trailing schema
    block gives the row count and the column names without touching a row group,
    so the true totals cost one seek per shard even for shards this run will never
    open again. A capped answer that cannot say what it capped is
    indistinguishable from a complete one.

    Shards that do not carry the column still advance the base index. Skipping
    them would shift every row number after them, and episode row spans are
    absolute — the frames of episode 12 would come from wherever the shift landed.
    """
    import pyarrow.parquet as pq

    out: list[tuple[int, int, bool]] = []
    first_columns: list[str] = []
    base = 0
    for path in files:
        with pq.ParquetFile(path) as handle:
            names = list(handle.schema_arrow.names)
            if not first_columns:
                first_columns = names
            n = int(handle.metadata.num_rows)
        out.append((base, n, column in names))
        base += n
    return out, first_columns


def _stream_column(
    files: list[Path],
    shards: list[tuple[int, int, bool]],
    column: str,
    *,
    first_row: int = 0,
    stop_row: int | None = None,
) -> Iterator[tuple[int, list]]:
    """Yield `(absolute_index_of_row_0, rows)`, one arrow batch at a time.

    Deliberately NOT `vla_data._stream_action_rows`, which is the same shape and
    a different function: that one yields rows and nothing else, because a
    dataset-wide statistic does not care where a row sits. Everything this module
    does is positional — which rows belong to the scored episode, which ordinal in
    the eligible sequence a row holds, and therefore whether it lands in the
    reference sample or in the held-out null. Recovering the index by counting
    yielded rows would work only while every shard carries the column, and shards
    that do not are exactly the case the index exists to survive.

    `first_row`/`stop_row` skip whole shards from their footers, without opening
    them: scoring one episode of `lerobot/droid` should not read 26 million rows
    to find 400 of them.

    One batch is resident at a time, and the `del` below is what makes that
    sentence true rather than merely intended. A generator resumes INSIDE its own
    frame: without the `del`, this function's `rows` still names the batch it
    just yielded while `to_pylist()` builds the next one, so two full batches of
    Python floats overlap on every read. MEASURED with tracemalloc at width 14,
    8,192 rows to a batch: 10,241,959 bytes peak before, 5,901,168 after — the
    extra 4.3 MB was exactly one more materialised batch. The caller has to drop
    its own reference too, and each of the three loops below ends with a `del`
    saying so.
    """
    import pyarrow.parquet as pq

    for path, (base, n, has) in zip(files, shards, strict=True):
        if not has:
            continue
        if base + n <= first_row or (stop_row is not None and base >= stop_row):
            continue
        at = base
        with pq.ParquetFile(path) as handle:
            for batch in handle.iter_batches(
                batch_size=ACTION_ROW_BATCH, columns=[column]
            ):
                length = batch.num_rows
                if at + length > first_row and (stop_row is None or at < stop_row):
                    rows = batch.column(0).to_pylist()
                    if rows:
                        yield at, rows
                    del rows
                at += length
                if stop_row is not None and at >= stop_row:
                    break


def _episode_spans(reader) -> tuple[list[tuple[int, int, int]], str]:
    """`[(episode_index, length, first_row)]`, and where the first row came from.

    NOT `reader.episodes()`, for the reason `vla_data._episode_count` gives for
    not calling it either: `episodes()` builds video routing for the currently
    selected camera and refuses outright when `videos/<camera>/from_timestamp` is
    missing. That refusal is right there and wrong here — the recorded vectors
    live in the data shards and are perfectly measurable on a dataset whose video
    routing is broken, absent, or simply not downloaded. This reads the three
    columns that have nothing to do with any camera and would give the same spans
    on all of them.
    """
    files = sorted((Path(reader.snapshot) / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise Refusal(
            f"{getattr(reader, 'repo_id', None) or 'this dataset'} has no episode "
            f"metadata under {Path(reader.snapshot) / 'meta' / 'episodes'}, so "
            f"there is no way to say which rows belong to which episode. A "
            f"LeRobot v3.0 snapshot keeps it in "
            f"meta/episodes/chunk-*/file-*.parquet; if the download was "
            f"interrupted, re-run it."
        )
    table = _read_all(files)
    indices = table.get("episode_index")
    lengths = table.get("length")
    if indices is None or lengths is None:
        missing = "episode_index" if indices is None else "length"
        raise Refusal(
            f"{getattr(reader, 'repo_id', None) or 'this dataset'} has no "
            f"`{missing}` column in its episode metadata, so an episode's rows "
            f"cannot be located. Scoring against the wrong rows would produce a "
            f"full table of numbers about a different episode, which is worse "
            f"than this refusal."
        )
    starts = table.get("dataset_from_index")
    if starts is not None:
        source = "meta/episodes `dataset_from_index`, the dataset's own row map"
        firsts = [int(v or 0) for v in starts]
    else:
        # The same fallback `_episodes_locked` takes, and it is an ASSUMPTION: it
        # holds only while the rows are contiguous and in episode order. Named in
        # the payload rather than hidden, so a reader whose numbers look wrong
        # has somewhere to start.
        source = (
            "summed episode lengths — this snapshot publishes no "
            "`dataset_from_index`, so the rows are assumed contiguous and in "
            "episode order"
        )
        firsts = []
        at = 0
        for length in lengths:
            firsts.append(at)
            at += int(length)
    return (
        [
            (int(indices[i]), int(lengths[i]), int(firsts[i]))
            for i in range(len(indices))
        ],
        source,
    )


class _Covariance:
    """Mean and co-moment over streamed batches, in float64.

    WELFORD, in its batch form, for the reason `vla_data._ActionDim.add` gives
    for using Welford at all — carried up into d dimensions, where the failure is
    worse than a wrong number. The cheap `E[xx^T] - E[x]E[x]^T` form subtracts two
    large nearly-equal matrices; the difference loses its significant digits, and
    what comes back is a matrix that is not positive semi-definite. Its
    eigendecomposition then returns a NEGATIVE eigenvalue, whose reciprocal square
    root — which is exactly what the whitener is built from — is NaN. A whitener
    holding one NaN makes every distance NaN, and a chart of NaNs looks like an
    episode that was never recorded rather than like a broken measurement.

    MEASURED here rather than assumed, both forms in float64 over the identical
    4,000 rows, three correlated dimensions, spread 0.3, smallest true eigenvalue
    +4.9960e-05 (`tests/test_vla_ood.py` runs the same scan):

        offset 0      naive +4.9960e-05    this +4.9960e-05
        offset 1e4    naive +4.9931e-05    this +4.9960e-05
        offset 1e5    naive +5.4317e-05    this +4.9960e-05   (8.7% high)
        offset 1e6    naive -1.2301e-04    this +4.9960e-05   (rsqrt -> NaN)
        offset 1e7    naive -2.4399e-02    this +4.9960e-05

    So float64 buys real headroom and does not buy immunity: the naive form is
    still fine at the 512-ish magnitudes `vla_data` names and gives way five
    orders of magnitude further up, which a dataset recording millimetres, encoder
    counts or microseconds reaches without doing anything unusual. This form
    subtracts before it squares and never builds either large intermediate.

    The batch form (Chan, Golub and LeVeque's parallel update) rather than the
    per-row one because the per-row loop is thousands of torch scalar operations
    per batch and the combination is exact either way.
    """

    def __init__(self, width: int) -> None:
        import torch

        self.n = 0
        self.width = width
        self.mean = torch.zeros(width, dtype=torch.float64)
        self.m2 = torch.zeros((width, width), dtype=torch.float64)

    def add(self, block) -> None:
        import torch

        m = int(block.shape[0])
        if m == 0:
            return
        block_mean = block.mean(dim=0)
        centred = block - block_mean
        block_m2 = centred.T @ centred
        if self.n == 0:
            self.n, self.mean, self.m2 = m, block_mean, block_m2
            return
        total = self.n + m
        delta = block_mean - self.mean
        self.mean = self.mean + delta * (m / total)
        self.m2 = self.m2 + block_m2 + torch.outer(delta, delta) * (self.n * m / total)
        self.n = total

    def covariance(self):
        """Population covariance — divided by n, not n-1.

        The convention `vla_data` already publishes, and here the choice cannot
        bias a reading either way: every distance in the report is computed with
        THIS estimate, and each frame's percentile is taken against reference
        distances computed with the same one. A constant factor on every distance
        leaves every percentile exactly where it was.
        """
        if self.n < 1:
            raise Refusal(
                "no readable row reached the covariance, so there is no "
                "reference distribution to measure anything against."
            )
        return self.m2 / self.n


def _phase(ordinal: int, stride: int) -> str:
    """Which half of the strided sample an eligible row falls in.

    Deterministic and seedless on purpose: a reference set that depends on a seed
    is one a reader cannot reproduce without being told the seed, and the two
    passes over the data have to agree on the same rows or the distances would
    describe a different set from the mean they were measured against.

    The null takes the rows halfway between the reference's, which is disjoint
    from it by construction for every stride of 2 or more. At stride 1 the
    reference is every eligible row and there is nothing left over — the caller is
    told that rather than handed a null drawn from rows it has already seen.

    A STRIDE CAN ALIAS. If the rows have a period sharing a factor with the
    stride, a strided sample systematically favours one phase of that period.
    `row_stride` is in the payload for that reason; a run whose reference looks
    suspiciously narrow is worth repeating at a different `max_reference_rows`.
    """
    if stride < 2:
        return "reference"
    position = ordinal % stride
    if position == 0:
        return "reference"
    return "null" if position == stride // 2 else "unused"


def _declared_width(reader, space: str) -> int | None:
    """What `info.json` says this column's width is, or `None` when it says none."""
    feature = (getattr(reader, "info", None) or {}).get("features") or {}
    shape = (feature.get(space) or {}).get("shape")
    if isinstance(shape, (list, tuple)) and len(shape) == 1:
        try:
            return int(shape[0])
        except (TypeError, ValueError):
            return None
    return None


def _find_span(spans, episode: int, repo_id: str) -> tuple[int, int, int]:
    for span in spans:
        if span[0] == episode:
            return span
    known = sorted(s[0] for s in spans)
    shown = known[:12]
    # The cap is REPORTED rather than merely applied: a reader shown twelve
    # indices out of two hundred needs to know the list was cut, or the episode
    # they wanted looks absent from a dataset that has it.
    tail = "" if len(known) == len(shown) else f", ... ({len(known)} in total)"
    raise OODError(
        f"{repo_id} has no episode {episode}. It has "
        f"{', '.join(str(k) for k in shown)}{tail}."
    )


def _overlap(shards, first_row: int, length: int) -> int:
    """How many of an episode's rows sit in shards that carry the column."""
    stop = first_row + length
    total = 0
    for base, n, has in shards:
        if not has:
            continue
        total += max(0, min(stop, base + n) - max(first_row, base))
    return total


def _shard_rows_touching(shards, first_row: int, length: int) -> int:
    """Rows in the shards an episode overlaps — what a targeted read costs.

    Not the episode's own length: parquet is read by row group inside a shard, and
    the honest unit for "what will this read" is the shards it has to open.
    """
    stop = first_row + length
    return sum(
        n for base, n, has in shards if has and base < stop and base + n > first_row
    )


def _reference_stride(eligible: int, max_reference_rows: int | None) -> int:
    if max_reference_rows is None or eligible <= max_reference_rows:
        return 1
    return math.ceil(eligible / max_reference_rows)


def _null_plan(eligible: int, stride: int, null_draws: int) -> tuple[int, int, str]:
    """`(draws, stride_over_the_held-out_phase, why_there_are_none)`.

    The held-out rows are strided too. Taking the first K of them would draw the
    whole null from the first episodes of the dataset, which is a null about the
    beginning of the recording rather than about the recording — and a null that
    samples one corner of the distribution is the failure this module exists to
    guard against, arrived at from the inside.
    """
    if null_draws < 1:
        return (
            0,
            1,
            (
                "no null was asked for (`null_draws` is 0), so nothing is flagged and "
                "`clears_null` is null rather than false on every frame."
            ),
        )
    if stride < 2:
        return (
            0,
            1,
            (
                "the reference set is EVERY eligible row of this dataset, so there is "
                "no row left over to draw a null from. Lower `max_reference_rows` to "
                "hold rows back — the null is rows the reference has not seen, and "
                "there is no way to have both out of the same rows."
            ),
        )
    available = len(range(stride // 2, eligible, stride))
    if available == 0:
        return (
            0,
            1,
            (
                f"this dataset has {eligible} eligible row(s) at stride {stride}, "
                f"which leaves none in the held-out phase. There is nothing to draw a "
                f"null from."
            ),
        )
    null_stride = max(1, math.ceil(available / null_draws))
    return len(range(0, available, null_stride)), null_stride, ""


# ------------------------------------------------- the parameter checks, ONCE
#
# These live here, as functions, because `estimate` shipped without them and
# quoted firm figures for four parameter values the run then refused — including
# a `row_stride: -80` with `sampled: False`, a preflight describing a run that
# cannot exist. A preflight that validates less than the run is not a preflight,
# it is a second opinion nobody asked for. One copy of each rule, called from
# both, so they cannot drift.


def _check_int(name: str, value, *, allow_none: bool = False) -> None:
    """Reject a bool where an int belongs, BEFORE any range check sees it.

    `isinstance(True, int)` is True, so `True` sails through `2 <= v <= 500_000`
    as the integer 1 and through `v >= 1` as a stride of 1. It then travels into
    the payload as a number nobody typed. A bool is a flag that arrived in the
    wrong argument, and saying so is cheaper than the run it would misdescribe.
    """
    if value is None and allow_none:
        return
    if isinstance(value, bool):
        raise OODError(
            f"{name} must be a whole number, and {value} is a boolean. "
            f"`isinstance(True, int)` is True in Python, so this would have been "
            f"read as {int(value)} and reported as a figure nobody asked for."
        )
    if not isinstance(value, int):
        raise OODError(
            f"{name} must be a whole number; a {type(value).__name__} was passed."
        )


def _check_frame_stride(frame_stride) -> None:
    _check_int("frame_stride", frame_stride)
    if frame_stride < 1:
        raise OODError(
            f"frame_stride must be at least 1; {frame_stride} was asked for. A "
            f"stride of zero is not a coarser sample, it is no sample at all."
        )


def _check_max_reference_rows(max_reference_rows) -> None:
    _check_int("max_reference_rows", max_reference_rows, allow_none=True)
    if max_reference_rows is not None and (
        max_reference_rows < 2 or max_reference_rows > MAX_REFERENCE_ROWS
    ):
        raise OODError(
            f"max_reference_rows must be between 2 and {MAX_REFERENCE_ROWS:,}, or "
            f"omitted to read every eligible row; {max_reference_rows} was asked "
            f"for. One row has no spread and is not a distribution."
        )


def _check_null_draws(null_draws) -> None:
    _check_int("null_draws", null_draws)
    if null_draws < 0 or null_draws > MAX_NULL_DRAWS:
        raise OODError(
            f"null_draws must be between 0 and {MAX_NULL_DRAWS:,}; {null_draws} "
            f"was asked for. 0 means no null is measured, and then nothing is "
            f"flagged at all."
        )


def _check_bins(bins) -> None:
    _check_int("bins", bins)
    if bins < 1 or bins > MAX_HISTOGRAM_BINS:
        raise OODError(
            f"bins must be between 1 and {MAX_HISTOGRAM_BINS}; {bins} was asked "
            f"for. Every bin travels in the response, so this is a payload limit. "
            f"It is a display histogram only — the percentiles in this report are "
            f"exact and do not come from it."
        )


# The Python cost of ONE row of width `w` between `to_pylist()` and the tensor,
# MEASURED with tracemalloc on the real objects rather than reasoned about — one
# arrow batch of 8,192 rows through `to_pylist` and then through `_readable`:
#
#   width    to_pylist alone    + the _readable copy    this formula
#       1          120.3 B                  216.7 B          224 B
#       2          144.3 B                  240.4 B          264 B
#      14          528.3 B                  720.4 B          744 B
#      64         2112.3 B                 2688.7 B         2744 B
#
# The slope is 40 bytes a value — a 24-byte float object, an 8-byte slot in the
# arrow row's list, an 8-byte slot in `_readable`'s copy — on fixed per-row
# overhead of two list objects and their item arrays. 184 rather than the ~160
# a least-squares fit gives, deliberately: this is quoted as a CEILING, and it is
# at or above every measured width above (by 3% at 14, by 10% at 2). A memory
# figure a caller plans around must not come in under what they then measure.
#
# The old figure here was a flat `32 * ACTION_ROW_BATCH` — 262,144 bytes at EVERY
# width, for the one term that is not flat in the width at all. It understated
# the measured peak by 5x at two dimensions and 57x at sixty-four.
PY_BYTES_PER_VALUE = 40
PY_BYTES_PER_ROW = 184


def _row_bytes(width: int) -> int:
    return PY_BYTES_PER_VALUE * width + PY_BYTES_PER_ROW


def _peak_bytes(
    width: int | None, reference_rows: int, frames: int
) -> tuple[int | None, str]:
    """The largest this run should hold, or `None` when the width is unknown.

    `None` rather than a figure computed from a guessed width. A memory estimate
    that silently assumed a dimension count would be wrong by exactly the factor
    nobody could see, and `budget.py` holds the line that unknown is never zero in
    either direction.

    EVERY held row is priced at `_row_bytes(width)` — 40 bytes a value on 176
    bytes of per-row overhead — rather than at the tensor rate of 8. Between
    `to_pylist()` and the tensor these are Python float objects in Python lists,
    twice over (arrow's row and `_readable`'s copy of it), and that is the term a
    caller can actually make large. It is the batch term, not the episode term,
    that dominates: one batch of 8,192 rows of a 64-wide column is 22 MB of
    Python objects, which is 32x the whole figure this function used to return.

    THIS IS A CEILING, and it is quoted as one. MEASURED end to end with
    tracemalloc against what this returns, at 20,000 reference rows with
    `null_draws=0` and the lazy imports warmed first so only the run is counted.
    The `distances` term is a torch tensor, which tracemalloc cannot see, so it
    is subtracted for the like-for-like column:

        width      this returns    its Python part      measured    ratio
            1         2,156,376          1,836,376     1,787,562     1.03
            2         2,484,368          2,164,368     1,983,598     1.09
           14         6,424,016          6,104,016     5,916,180     1.03
           64        22,913,616         22,593,616    22,039,416     1.03

    It runs further above the measurement when the EPISODE is the large object
    rather than the reference set, because the two never peak together: 5,000
    frames of a 64-wide column returns 36,297,152 against a measured 22,212,485
    (1.63x). The episode's vectors are read after both reference passes are done
    with their batch, so a run holds one or the other and this adds them. High
    and stated beats low: the figure a caller plans around must not come in under
    what they then measure.

    tracemalloc sees Python allocations only. The arrow buffer under each batch
    and every torch tensor — the covariance, the whitener, the distances — are
    allocated outside it and sit on top of the measured column.
    """
    if width is None or width < 1:
        return None, (
            "this snapshot does not declare a width for this column, so the "
            "per-row cost is unknown until the first row is read. It is measured "
            "on the run and reported as `dimensions`."
        )
    per_row = _row_bytes(width)
    covariance = 8 * width * width * 3
    batch = ACTION_ROW_BATCH * per_row
    distances = 8 * reference_rows * 2
    episode = frames * per_row
    return (covariance + batch + distances + episode), (
        f"one arrow batch of {ACTION_ROW_BATCH} rows at {per_row} bytes a row "
        f"(a {width}-value row is {PY_BYTES_PER_VALUE * width} bytes of Python "
        f"floats and list slots on {PY_BYTES_PER_ROW} bytes of list overhead, "
        f"MEASURED with tracemalloc rather than assumed), a {width}x{width} "
        f"float64 covariance accumulator, {reference_rows} float64 reference "
        f"distances (doubled for the concatenation that sorts them), and "
        f"{frames} episode vectors at the same per-row rate. The arrow buffer "
        f"under the batch and the torch tensors are outside the Python heap and "
        f"sit on top of this."
    )


# --------------------------------------------------------------------- the cost


def estimate(
    reader,
    episode: int,
    *,
    space: str = DEFAULT_SPACE,
    frame_stride: int = 1,
    max_reference_rows: int | None = DEFAULT_MAX_REFERENCE_ROWS,
    null_draws: int = DEFAULT_NULL_DRAWS,
) -> dict:
    """What this will cost, before it is spent. Footers only — no row is read.

    FORWARD PASSES: ZERO, and that is the honest figure rather than a modest one.
    `budget.py` prices analyses in forward passes because every ranking in this
    package is a loop of them; this one is not. Reporting a pass count of zero and
    then naming the unit that does apply is the only way the number means
    anything — a cost function quoting the wrong unit is worse than one admitting
    its unit does not apply.

    No seconds unless somebody measures them, for the reason `vla_sweep.estimate`
    states: a duration from another machine is a number people plan around.

    REFUSES WHAT THE RUN REFUSES, which is the whole reason a preflight exists.
    `vla_occlude.estimate` shipped without that and quoted a firm figure for a
    run the very next click turned down; a missing column and a bad stride are
    this module's version of the same gap, so both are checked here on the same
    footers the run would check them on.

    That claim was false for three of the four bounds until it was measured.
    `max_reference_rows=0` crashed here with a ZeroDivisionError out of
    `_reference_stride`; `-5` returned `row_stride: -80` beside `sampled: False`,
    which is a preflight describing a run that cannot exist; `1` and `10**9`
    returned confident figures the run turns down on its next line; and
    `null_draws=99999` quoted `null_draws: 0` rather than the refusal. All four
    now go through the SAME `_check_*` functions `build_reference` calls, so the
    two cannot disagree about what is askable.
    """
    _check_frame_stride(frame_stride)
    _check_max_reference_rows(max_reference_rows)
    _check_null_draws(null_draws)
    files = _files(reader)
    if not files:
        raise Refusal(
            f"{getattr(reader, 'repo_id', None) or 'this dataset'} has no frame "
            f"data under {Path(reader.snapshot) / 'data'}, so there are no "
            f"recorded vectors to build a reference set from."
        )
    shards, first_columns = _shard_map(files, space)
    if not any(has for _, _, has in shards):
        raise Refusal(
            f"{getattr(reader, 'repo_id', None) or 'this dataset'} has "
            f"{len(files)} frame shard(s) and none of them has a `{space}` "
            f"column, so there is nothing here to price a measurement of. The "
            f"first shard's columns are: "
            f"{', '.join(first_columns) or '(none)'}. Name one of those as "
            f"`space`, or point at a dataset that records this one."
        )
    spans, span_source = _episode_spans(reader)
    _, length, first_row = _find_span(
        spans, episode, getattr(reader, "repo_id", None) or "this dataset"
    )

    rows_total = sum(n for _, n, _ in shards)
    rows_with_column = sum(n for _, n, has in shards if has)
    episode_rows = _overlap(shards, first_row, length)
    eligible = rows_with_column - episode_rows
    stride = _reference_stride(eligible, max_reference_rows)
    reference_rows = len(range(0, eligible, stride)) if eligible > 0 else 0
    draws, _, null_why = _null_plan(eligible, stride, null_draws)

    frames = len(range(0, length, frame_stride))
    declared = _declared_width(reader, space)
    peak, peak_basis = _peak_bytes(declared, reference_rows, frames)
    targeted = _shard_rows_touching(shards, first_row, length)

    return {
        "repo_id": getattr(reader, "repo_id", ""),
        "episode": episode,
        "space": space,
        # The headline, and it is zero. Named beside the unit that does apply.
        "forward_passes": 0,
        "forward_passes_why": (
            "nothing here runs a model. No vision tower, no action expert, no "
            "GPU, no `av`, and no lerobot sidecar — the reference set is the "
            "dataset's own recorded vectors and the distance is arithmetic over "
            "them. The cost is parquet rows read, below."
        ),
        "cost_unit": "parquet rows read",
        "passes_over_the_data": 2,
        "passes_why": (
            "the first pass measures the reference mean and covariance, the "
            "second measures every reference and held-out null row's distance "
            "against them — the metric has to exist before a distance in it can. "
            "Twice the I/O for a constant amount of memory, the same trade "
            "`dataset_action_stats` makes. The episode's own rows are then read "
            "once more, from the shards it lives in only."
        ),
        "rows_to_read": 2 * rows_with_column + targeted,
        "rows_in_episode_shards": targeted,
        "rows_total": rows_total,
        "rows_with_column": rows_with_column,
        "files_total": len(files),
        "files_with_column": sum(1 for _, _, has in shards if has),
        "rows_in_episode": episode_rows,
        "rows_eligible": eligible,
        "reference_rows": reference_rows,
        # A CEILING, and it says so rather than letting the run quietly come in
        # under it. A footer carries row counts and column names; it does not
        # carry row CONTENTS, so a row that is not a list, is the wrong width, or
        # holds a NaN is invisible from here and is dropped by the run. On a
        # fixture with 20 of 400 reference rows written as None this figure is
        # 100 and the run reads 80.
        "reference_rows_is_a_ceiling": True,
        "null_draws_is_a_ceiling": True,
        "readability_why": (
            "`reference_rows` and `null_draws` are counted from the parquet "
            "footers, which carry row counts and column names but not row "
            "contents. A malformed row — not a list, the wrong width, or holding "
            "a non-finite value — cannot be seen from a footer and is dropped by "
            "the run, so both figures are ceilings rather than predictions. The "
            "run reports what it actually folded in as `rows_read` and what it "
            "threw away as `rows_malformed`; if those two do not add up to this "
            "number, the difference is malformed rows and is named there."
        ),
        "row_stride": stride,
        "sampled": stride > 1,
        "row_span_from": span_source,
        "null_draws": draws,
        "null_reason": null_why,
        "frames": frames,
        "frames_total": length,
        "frame_stride": frame_stride,
        "dimensions_declared": declared,
        "peak_bytes": peak,
        "peak_basis": peak_basis,
        "seconds": None,
        "seconds_from": (
            "not estimated — this is disk-bound, nobody has timed this machine's "
            "parquet reads, and a duration from somebody else's disk is a number "
            "people plan around"
        ),
    }


# ----------------------------------------------------------------- the building


def build_reference(
    reader,
    *,
    space: str = DEFAULT_SPACE,
    exclude_episode: int | None = None,
    max_reference_rows: int | None = DEFAULT_MAX_REFERENCE_ROWS,
    bins: int = DEFAULT_HISTOGRAM_BINS,
    null_draws: int = DEFAULT_NULL_DRAWS,
) -> Reference:
    """Measure the distribution, and its null, from the dataset's own rows.

    Two streaming passes. The first folds the reference-phase rows into a mean and
    a covariance; the second measures every reference row's distance under the
    metric that produced, and every held-out null row's distance under the same
    one. Nothing but the accumulators and one arrow batch is resident during
    either.
    """
    import torch

    if not isinstance(space, str) or not space.strip():
        raise OODError(
            "a column name is required to say what the reference set is made of; "
            f"the default is `{DEFAULT_SPACE}`."
        )
    _check_bins(bins)
    _check_max_reference_rows(max_reference_rows)
    _check_null_draws(null_draws)

    repo_id = getattr(reader, "repo_id", "") or ""
    files = _files(reader)
    if not files:
        raise Refusal(
            f"{repo_id or 'this dataset'} has no frame data under "
            f"{Path(reader.snapshot) / 'data'}, so there are no recorded vectors "
            f"to build a reference set from. A LeRobot v3.0 snapshot keeps them "
            f"in data/chunk-*/file-*.parquet; if the download was interrupted, "
            f"re-run it."
        )
    shards, first_columns = _shard_map(files, space)
    if not any(has for _, _, has in shards):
        raise Refusal(
            f"{repo_id or 'this dataset'} has {len(files)} frame shard(s) and none "
            f"of them has a `{space}` column, so there is no distribution here to "
            f"measure against. The first shard's columns are: "
            f"{', '.join(first_columns) or '(none)'}. Name one of those as "
            f"`space`, or point at a dataset that records this one."
        )

    excluded_from, excluded_len = 0, 0
    span_source = ""
    if exclude_episode is not None:
        spans, span_source = _episode_spans(reader)
        _, excluded_len, excluded_from = _find_span(
            spans, exclude_episode, repo_id or "this dataset"
        )
    exclude_stop = excluded_from + excluded_len

    rows_total = sum(n for _, n, _ in shards)
    rows_with_column = sum(n for _, n, has in shards if has)
    excluded_rows = (
        _overlap(shards, excluded_from, excluded_len)
        if exclude_episode is not None
        else 0
    )
    eligible = rows_with_column - excluded_rows
    if eligible < 2:
        raise Refusal(
            f"{repo_id or 'this dataset'} leaves {eligible} row(s) of `{space}` "
            f"outside episode {exclude_episode} to build a reference set from. A "
            f"distribution needs at least two, and this is a recording with a "
            f"single episode in it rather than a broken snapshot — pass "
            f"`exclude_episode=None` to measure against every row including this "
            f"episode's own, knowing what that means."
        )
    stride = _reference_stride(eligible, max_reference_rows)
    wanted_nulls, null_stride, null_reason = _null_plan(eligible, stride, null_draws)

    def excluded(index: int) -> bool:
        return exclude_episode is not None and excluded_from <= index < exclude_stop

    # ---- pass 1: the mean and the covariance, over the reference phase only
    accumulator: _Covariance | None = None
    width = 0
    rows_read = 0
    rows_malformed = 0
    ordinal = 0
    for base, rows in _stream_column(files, shards, space):
        block: list[list[float]] = []
        for offset, raw in enumerate(rows):
            if excluded(base + offset):
                continue
            phase = _phase(ordinal, stride)
            ordinal += 1
            if phase != "reference":
                continue
            values = _readable(raw, width if accumulator is not None else None)
            if values is None:
                rows_malformed += 1
                continue
            if accumulator is None:
                # The width is the dataset's own, taken from its first readable
                # row — the same rule `dataset_action_stats` follows, so the two
                # agree about what this dataset is.
                width = len(values)
                accumulator = _Covariance(width)
            block.append(values)
        if block and accumulator is not None:
            accumulator.add(torch.tensor(block, dtype=torch.float64))
            rows_read += len(block)
        # Both references dropped before the generator is asked for the next
        # batch — see `_stream_column`. A `for` target is rebound only AFTER
        # `next()` returns, so without this the previous batch is still reachable
        # while the next one is being built.
        del rows, block
    if accumulator is None or rows_read < 2:
        raise Refusal(
            f"{repo_id or 'this dataset'} gave {rows_read} readable row(s) of "
            f"`{space}` out of {eligible} eligible ({rows_malformed} could not be "
            f"read as a list of finite numbers). A covariance needs at least two "
            f"rows and a single row has no spread — this column holds something "
            f"this reader does not understand, or the recording is empty."
        )
    if rows_read <= width:
        raise Refusal(
            f"a {width}x{width} covariance estimated from {rows_read} row(s) is "
            f"singular by construction: with no more rows than dimensions the "
            f"sample cannot span the space, and every direction it missed would "
            f"read as infinitely far away. Read more rows (raise "
            f"`max_reference_rows`), or measure a narrower column."
        )

    covariance = accumulator.covariance()
    # `eigh`, not `eig`: a covariance is symmetric, and the symmetric routine
    # returns real eigenvalues in ascending order rather than complex ones that
    # would then have to be argued about.
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    # A covariance is positive semi-definite, and in float64 `eigh` returns the
    # directions the sample never moved along as a few ulps either side of zero.
    # Clamped rather than kept: a negative variance has no square root, and the
    # floor below is what actually decides whether a direction is usable.
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    largest = float(eigenvalues.max())
    if largest <= 0.0:
        raise Refusal(
            f"every one of the {rows_read} sampled rows of `{space}` is identical, "
            f"so there is no spread to measure a distance against — every frame "
            f"would be at distance zero from the only point this dataset "
            f"contains. The recording may be static, or this column may be a "
            f"constant the robot never varied."
        )
    floor = largest * CONDITION_FLOOR
    keep = eigenvalues > floor

    reference = Reference(
        repo_id=repo_id,
        space=space,
        dimensions=width,
        dimensions_declared=_declared_width(reader, space),
        rows_total=rows_total,
        rows_with_column=rows_with_column,
        rows_eligible=eligible,
        rows_read=rows_read,
        rows_malformed=rows_malformed,
        row_stride=stride,
        sampled=stride > 1,
        excluded_episode=exclude_episode,
        excluded_rows=excluded_rows,
        row_span_from=span_source,
        rows_per_dimension=round(rows_read / width, 3),
        directions_kept=int(keep.sum()),
        directions_dropped=int((~keep).sum()),
        variance_floor=float(floor),
        condition_ratio=CONDITION_FLOOR,
        metric=(
            f"Mahalanobis distance from the mean of {rows_read} rows of "
            f"`{space}`, over the {int(keep.sum())} of {width} directions that "
            f"reference set varies in"
        ),
    )
    reference.mean_vector = accumulator.mean
    reference.kept_basis = vectors[:, keep]
    reference.inv_sqrt = eigenvalues[keep].rsqrt()
    reference.dropped_basis = vectors[:, ~keep]

    # ---- pass 2: every reference row's distance, and the held-out null's
    chunks: list = []
    null_max: float | None = None
    null_used = 0
    null_seen = 0
    ordinal = 0
    for base, rows in _stream_column(files, shards, space):
        block: list[list[float]] = []
        null_block: list[list[float]] = []
        for offset, raw in enumerate(rows):
            if excluded(base + offset):
                continue
            phase = _phase(ordinal, stride)
            ordinal += 1
            if phase == "unused":
                continue
            if phase == "null":
                # Strided over the held-out phase as well, so the null spans the
                # dataset instead of being drawn entirely from its first episodes.
                take = null_seen % null_stride == 0
                null_seen += 1
                if not take or null_used + len(null_block) >= wanted_nulls:
                    continue
            values = _readable(raw, width)
            if values is None:
                continue
            (block if phase == "reference" else null_block).append(values)
        if block:
            distances, _ = reference.distances_of(
                torch.tensor(block, dtype=torch.float64)
            )
            chunks.append(distances)
        if null_block:
            distances, _ = reference.distances_of(
                torch.tensor(null_block, dtype=torch.float64)
            )
            null_used += len(null_block)
            batch_max = float(distances.max())
            null_max = batch_max if null_max is None else max(null_max, batch_max)
        del rows, block, null_block

    if not chunks:
        raise Refusal(
            f"the second pass over `{space}` read none of the rows the first pass "
            f"had read, which means the shards changed underneath this "
            f"measurement. Re-run it."
        )
    sorted_distances = torch.sort(torch.cat(chunks)).values
    reference.sorted_distances = sorted_distances
    reference.percentile_resolution = round(100.0 / int(sorted_distances.numel()), 6)
    reference.distances = _distance_summary(sorted_distances, bins)
    if null_used and null_max is not None:
        reference.null_max = round(null_max, 6)
        reference.null_draws = null_used
        # The expected percentile of the maximum of K draws from a continuous
        # distribution is K/(K+1). That is the null's own RESOLUTION: at 8 draws
        # it is the 88.9th percentile, and a frame beating it has cleared very
        # little. Reported so "it beat the null" does not carry the same weight
        # at every K.
        reference.null_covers_percentile = round(100.0 * null_used / (null_used + 1), 4)
        # And where it ACTUALLY landed, which is a different number and is one
        # this object can answer exactly — it holds every reference distance. The
        # sentence used to quote the figure above as `null_max`'s position "of
        # in-distribution distances", which is a claim about the measurement made
        # without consulting the measurement. On the fixture in
        # `tests/test_vla_ood.py` the two read 99.01 and 100.0.
        position, past_max = reference.percentile_of(reference.null_max)
        reference.null_max_percentile = position
        reference.null_max_beyond_reference_max = past_max
        reference.null_position_caveat = (
            "these two percentiles are not the same quantity and are not "
            "expected to agree: `null_covers_percentile` is 100*K/(K+1) from the "
            "draw count alone, while `null_max_percentile` is where the measured "
            "maximum landed among the reference distances. They are also not "
            "measured under the same conditions — the reference distances are "
            "IN-SAMPLE (the mean and the covariance were fitted on those very "
            "rows) while every null row is OUT of that fit, so the null's "
            "distances run a little larger for that reason alone. Reading the "
            "expected figure as the measured one overstates how ordinary the "
            "null was."
        )
        reference.null_description = (
            f"the largest distance reached by {null_used} row(s) of `{space}` "
            f"drawn from this same dataset, none of them in the reference set"
            + (
                f" and none from episode {exclude_episode}"
                if exclude_episode is not None
                else ""
            )
            + "."
        )
    else:
        reference.null_reason = null_reason or (
            "no held-out row could be read, so nothing was tested against chance."
        )
    return reference


def _readable(raw, width: int | None) -> list[float] | None:
    """One recorded vector as finite floats, or `None`.

    Width is checked against the reference's own, taken from its first readable
    row, exactly as `dataset_action_stats` does: a row of a different length is
    not a narrower observation, it is a row nobody can pair with the others by
    position. Non-finite values disqualify the WHOLE row rather than being dropped
    from it, because a vector with a hole in it is not a point and a distance to
    it is not a distance.
    """
    values = _as_row(raw)
    if values is None:
        return None
    if width is not None and len(values) != width:
        return None
    out: list[float] = []
    for value in values:
        # `isinstance(True, int)` is True, so bool comes FIRST. A gripper flag
        # recorded as a boolean is a real value and 1.0/0.0 is what it means;
        # falling through the numeric arm would give the same answer by accident
        # and hide that a bool was relying on it.
        if isinstance(value, bool):
            out.append(1.0 if value else 0.0)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        out.append(number)
    return out


def _why_no_row(shards, index: int, space: str) -> str:
    """Why a frame's row never reached the reader — the TRUE cause, not one cause.

    There used to be a single sentence here, and it blamed the episode's declared
    span for running past the rows the shards carry. That is a dataset-shaped
    assertion published as a diagnosis, and for the commonest case it is false:
    `_shard_map` has already recorded `has=False` for a shard that does not carry
    the column, `_stream_column` has already skipped it by that flag, and the row
    sits squarely inside the data. The reference set was very likely built off
    the other shards in the same call, so the module knew.

    Three causes, three sentences, each naming what a reader would have to change.
    """
    total = sum(n for _, n, _ in shards)
    for base, n, has in shards:
        if base <= index < base + n:
            if not has:
                return (
                    f"the shard holding this row does not carry a `{space}` "
                    f"column at all — the row exists and is numbered (row {index} "
                    f"of the snapshot, in the shard covering rows {base} to "
                    f"{base + n - 1}), and it is the COLUMN that is absent there. "
                    f"The reference set was built from the shards that do carry "
                    f"it. This frame has no recorded vector to measure."
                )
            return (
                f"row {index} sits in a shard that does carry `{space}`, and no "
                f"value for it came back from the read. The shard may have been "
                f"rewritten underneath this measurement; re-run it."
            )
    return (
        f"the episode's declared span runs past the rows the shards actually "
        f"carry — this frame is row {index} and the snapshot has {total}. The "
        f"episode metadata and the data shards disagree about how long this "
        f"episode is."
    )


def _why_unreadable(raw, width: int) -> str:
    """The sentence that travels with a frame absent from the ranking."""
    values = _as_row(raw)
    if values is None:
        return (
            "this row is not a list of numbers — the column holds something with "
            "no vector in it for this frame"
        )
    if len(values) != width:
        return (
            f"this row has width {len(values)} and the reference set has {width}; "
            f"there is no way to pair the two by position"
        )
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            return (
                "this row holds a value that is not a number, so it is not a "
                "point in the reference space"
            )
        if not math.isfinite(number):
            return (
                "this row holds a non-finite value (NaN or infinity), and a "
                "vector with a hole in it is not a point to measure a distance to"
            )
    return "this row could not be read as a vector of finite numbers"


def _distance_summary(sorted_distances, bins: int) -> dict:
    """min / max / mean, the exact percentiles, and a histogram for drawing."""
    import torch

    n = int(sorted_distances.numel())
    low = float(sorted_distances[0])
    high = float(sorted_distances[-1])
    percentiles = []
    for q in REFERENCE_PERCENTILES:
        # Nearest rank over the exact sorted distances. No interpolation, because
        # there is nothing to interpolate between: these are the values
        # themselves rather than a histogram summarising them.
        rank = min(n - 1, max(0, math.ceil(q / 100.0 * n) - 1))
        percentiles.append({"q": q, "value": round(float(sorted_distances[rank]), 6)})
    if high > low:
        width = (high - low) / bins
        index = torch.clamp(((sorted_distances - low) / width).long(), 0, bins - 1)
        counts = torch.bincount(index, minlength=bins).tolist()
        edges = [round(low + i * width, 6) for i in range(bins + 1)]
        # The top edge is set from the measurement rather than accumulated to, for
        # the reason `_ActionDim.to_dict` gives: `bins` additions of (hi-lo)/bins
        # does not land on `hi`, and an edge a few ulps above the maximum reads as
        # a value nothing reached.
        #
        # Rounded to the same six places `max` below is, and that is not
        # cosmetic: unrounded, the last edge and the reported maximum disagreed
        # in the seventh digit (1.7148160424389374 against 1.714816), so a panel
        # drawing the histogram put its final bar past the largest distance the
        # payload said existed. One number, printed twice, contradicting itself.
        edges[-1] = round(high, 6)
    else:
        # Every reference row at the same distance. One bar holding all of them,
        # not `bins` empty ones around a single value — the same call
        # `_ActionDim.bin_plan` makes, and for the same reason: it is real data
        # rather than a degenerate case to skip.
        width = 0.0
        counts = [n]
        edges = [low, high]
    return {
        "min": round(low, 6),
        "max": round(high, 6),
        "mean": round(float(sorted_distances.mean()), 6),
        "count": n,
        "percentile_levels": list(REFERENCE_PERCENTILES),
        "percentiles": percentiles,
        "percentile_method": (
            "nearest rank over every reference distance, held exactly — these are "
            "not read off the histogram below, which exists for drawing"
        ),
        "histogram": {"bin_edges": edges, "counts": counts, "bin_width": width},
    }


# ------------------------------------------------------------------ the scoring


def score_episode(
    reader,
    episode: int,
    *,
    space: str = DEFAULT_SPACE,
    frame_stride: int = 1,
    max_frames: int = MAX_FRAMES,
    max_reference_rows: int | None = DEFAULT_MAX_REFERENCE_ROWS,
    bins: int = DEFAULT_HISTOGRAM_BINS,
    null_draws: int = DEFAULT_NULL_DRAWS,
    ranked: int = DEFAULT_RANKED,
    reference: Reference | None = None,
) -> EpisodeOOD:
    """Score every sampled frame of one episode against the dataset it came from.

    Builds its own reference set by default, with this episode's rows held out of
    it. A caller sweeping many episodes may pass one in — `reference` is then
    reported exactly as it was built, including which episode (if any) it
    excluded, and `means()` names the mismatch rather than quietly comparing this
    episode against a distribution containing it.
    """
    _check_frame_stride(frame_stride)
    _check_max_reference_rows(max_reference_rows)
    _check_null_draws(null_draws)
    _check_bins(bins)
    _check_int("max_frames", max_frames)
    if max_frames < 1 or max_frames > MAX_FRAMES:
        raise OODError(
            f"max_frames must be between 1 and {MAX_FRAMES:,}; {max_frames} was "
            f"asked for."
        )
    _check_int("ranked", ranked)
    if ranked < 1:
        raise OODError(
            f"ranked must be at least 1; {ranked} was asked for. The ranking is "
            f"the point of scoring an episode."
        )

    repo_id = getattr(reader, "repo_id", "") or ""
    spans, span_source = _episode_spans(reader)
    _, length, first_row = _find_span(spans, episode, repo_id or "this dataset")
    wanted = list(range(0, length, frame_stride))
    if len(wanted) > max_frames:
        raise OODError(
            f"episode {episode} is {length:,} frames and stride {frame_stride} "
            f"samples {len(wanted):,} of them; the cap is {max_frames:,}. Raise "
            f"the stride rather than having the list cut short — a ranking "
            f"missing its tail looks exactly like a ranking, and you would have "
            f"no way to tell."
        )

    started = time.perf_counter()
    if reference is None:
        reference = build_reference(
            reader,
            space=space,
            exclude_episode=episode,
            max_reference_rows=max_reference_rows,
            bins=bins,
            null_draws=null_draws,
        )
    elif reference.space != space:
        raise OODError(
            f"the reference set handed in is over `{reference.space}` and this "
            f"call asks for `{space}`. Two different columns are two different "
            f"distributions, and a distance between them is not a number about "
            f"anything."
        )
    if not reference.row_span_from:
        # A reference built with no exclusion never had to read the spans. The
        # report still owes the reader where the SCORED episode's rows came from,
        # and that is this call's answer rather than the reference's.
        reference.row_span_from = span_source

    # ---- the episode's own rows, read from the shards it lives in only
    files = _files(reader)
    shards, _ = _shard_map(files, space)
    stop = first_row + length
    keep = set(wanted)
    parsed: dict[int, list[float]] = {}
    refused: dict[int, str] = {}
    for base, rows in _stream_column(
        files, shards, space, first_row=first_row, stop_row=stop
    ):
        for offset, raw in enumerate(rows):
            index = base + offset
            if not first_row <= index < stop:
                continue
            t = index - first_row
            if t not in keep:
                continue
            values = _readable(raw, reference.dimensions)
            if values is None:
                # The reason is written while the row is still in hand. Holding
                # the raw row to explain it later would keep an arrow-derived
                # object alive for every unreadable frame in the episode.
                refused[t] = _why_unreadable(raw, reference.dimensions)
            else:
                parsed[t] = values
        del rows

    frames: list[FrameScore] = []
    unscored: list[dict] = []
    n_unscored = 0

    def cannot(t: int, why: str) -> None:
        # ABSENT from the ranking, not scored zero. `vla_sweep.run` set this
        # pattern: a frame that could not be measured is not a frame that
        # measured low, and a zero would sit at the bottom of the table looking
        # like the most ordinary frame in the episode. The listing is capped and
        # `n_unscored` carries the true count beside it.
        nonlocal n_unscored
        n_unscored += 1
        if len(unscored) < MAX_UNSCORED_LISTED:
            unscored.append({"t": t, "why": why})

    for t in wanted:
        if t in refused:
            cannot(t, refused[t])
            continue
        if t not in parsed:
            cannot(t, _why_no_row(shards, first_row + t, space))
            continue
        raw_distance, off = reference.score(parsed[t])
        if not math.isfinite(raw_distance):
            cannot(
                t,
                "the distance came out non-finite, which means the whitening of "
                "this vector overflowed rather than that the frame is far away",
            )
            continue
        # Rounded ONCE, here, and every comparison below uses the rounded value —
        # so the number in the payload is the number the null was beaten by and
        # the percentile was taken at. Publishing one value and deciding on
        # another is how a row comes to contradict the sentence under it.
        distance = round(raw_distance, 6)
        percentile, beyond = reference.percentile_of(distance)
        frames.append(
            FrameScore(
                t=t,
                distance=distance,
                percentile=percentile,
                percentile_resolution=reference.percentile_resolution,
                beyond_reference_max=beyond,
                off_manifold=None if off is None else round(off, 6),
                clears_null=reference.clears(distance),
            )
        )

    # Ranked by DISTANCE and nothing else — one stated measured quantity, the rule
    # `vla_sweep` holds. `off_manifold` and `clears_null` ride beside it and do
    # not reorder it: a ranking by two quantities is a ranking by neither.
    order = sorted(frames, key=lambda f: -f.distance)
    return EpisodeOOD(
        repo_id=repo_id,
        episode=episode,
        space=space,
        frame_stride=frame_stride,
        reference=reference,
        frames=frames,
        ranked=order[:ranked],
        n_ranked_total=len(order),
        n_frames=len(frames),
        frames_total=length,
        unscored=unscored,
        n_unscored=n_unscored,
        seconds=round(time.perf_counter() - started, 3),
    )
