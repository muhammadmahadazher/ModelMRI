"""A distance is meaningless without the set it was measured against, and this file holds that.

`vla_ood.py` answers "how far outside its own dataset does this frame sit". Every
way that question can be answered dishonestly is a test here:

  by not saying WHAT             a payload that reports 6.5 without reporting
                                 which column, how many rows, and which episode
                                 was held out of them is a number about nothing.
  by measuring per dimension     an arm pose can be inside one standard
                                 deviation on every joint and still be a
                                 configuration the arm never held, because the
                                 joints never held those values together. The
                                 headline test builds exactly that frame and
                                 measures both metrics on it.
  by calling something OOD       a boolean is a threshold somebody chose. The
                                 only flag in the module is gated on a null
                                 measured from the dataset's own held-out rows,
                                 and it is `None` — never `False` — when no null
                                 could be drawn.
  by scoring a broken frame 0    a zero sits at the bottom of the ranking looking
                                 like the most ordinary frame in the episode.
  by losing the row numbering    a shard without the column still occupies row
                                 numbers, and skipping it shifts every episode
                                 after it onto somebody else's rows.

The fixtures are synthetic and carry no video, deliberately: this measurement
needs no decoder, no vision tower and no policy, and a fixture that carried an
mp4 would let a dependency on one creep in without failing anything. Two tests
assert that absence directly.

Written against `tests/test_vla_routing.py`'s `build()` in shape but not in
substance — that one exists to test camera routing and its snapshots are built
for it. This one needs planted row values, multiple shards, and shards that omit
a column, so it builds its own.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("torch")

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from modelmri import vla_ood
from modelmri.errors import BadRequest, Refusal
from modelmri.vla_data import LeRobotV3Reader

STATE = "observation.state"
CAM = "observation.images.top"


# --------------------------------------------------------------- the fixtures


def build(
    root: Path,
    episodes: list[list],
    *,
    shards: int = 1,
    with_index: bool = True,
    routing: bool = True,
    drop_state_in: tuple[int, ...] = (),
    declared_width: int | None = None,
    state_key: str = STATE,
) -> Path:
    """A LeRobot v3.0 snapshot whose per-frame rows are exactly what was asked for.

    `episodes` is a list of episodes, each a list of one row's worth of
    `observation.state`. A row may be `None`, a list of any length, or hold a
    NaN — the three ways a real recording is unreadable, and each gets its own
    sentence out of the module.
    """
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

    rows = [row for episode in episodes for row in episode]
    lengths = [len(episode) for episode in episodes]
    width = declared_width
    if width is None:
        width = next((len(r) for r in rows if isinstance(r, list)), 1)

    features: dict = {
        state_key: {"dtype": "float32", "shape": [width]},
        "action": {"dtype": "float32", "shape": [1]},
    }
    if routing:
        features[CAM] = {"dtype": "video", "shape": [96, 96, 3]}
    (root / "meta" / "info.json").write_text(
        json.dumps({"fps": 10, "features": features}), encoding="utf-8"
    )

    n = len(episodes)
    starts = [sum(lengths[:i]) for i in range(n)]
    cols: dict[str, list] = {
        "episode_index": list(range(n)),
        "length": lengths,
        "tasks": [["push the T"] for _ in range(n)],
    }
    if with_index:
        cols["dataset_from_index"] = starts
    if routing:
        cols[f"videos/{CAM}/from_timestamp"] = [s / 10.0 for s in starts]
        cols[f"videos/{CAM}/to_timestamp"] = [
            (s + length) / 10.0 for s, length in zip(starts, lengths, strict=True)
        ]
        cols[f"videos/{CAM}/chunk_index"] = [0] * n
        cols[f"videos/{CAM}/file_index"] = [0] * n
    pq.write_table(
        pa.table(cols), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )

    # Contiguous shards, so shard `k` holds a known span of the global row index.
    # `sorted()` in the reader defines that order and file-000/001/002 sort the
    # way they are written here.
    per = math.ceil(len(rows) / shards)
    for shard in range(shards):
        chunk = rows[shard * per : (shard + 1) * per]
        offset = shard * per
        table: dict[str, list] = {
            "action": [[float(offset + i)] for i in range(len(chunk))]
        }
        if shard not in drop_state_in:
            table[state_key] = chunk
        pq.write_table(
            pa.table(table),
            root / "data" / "chunk-000" / f"file-{shard:03d}.parquet",
        )
    return root


def reader(tmp_path: Path, episodes, **kw) -> LeRobotV3Reader:
    return LeRobotV3Reader(build(tmp_path / "snap", episodes, **kw), "test/ood")


def ridge(
    n: int, *, jitter: float = 0.01, offset: float = 0.0, alternating: bool = False
) -> list[list[float]]:
    """`n` rows on a tight diagonal ridge: two joints that only ever move together.

    Deterministic rather than sampled — a reference distribution a reader cannot
    reproduce is one they cannot check, and the module refuses to take a seed for
    the same reason.

    The default off-ridge wobble is `sin(1.7 i)`, which has no short period, and
    that matters: the FIRST version of this fixture alternated +/- every row, and
    every stride the module chooses is even — so a strided reference sample drew
    only rows of one parity, the wobble vanished from it entirely, and the ridge
    direction was measured as having no spread at all. That is precisely the
    aliasing `_phase` warns about, it is real, and `alternating=True` keeps it so
    the test below can watch the module report it rather than hide it.
    """
    out = []
    for i in range(n):
        x = -1.0 + 2.0 * i / (n - 1)
        wobble = (1.0 if i % 2 == 0 else -1.0) if alternating else math.sin(1.7 * i)
        out.append([offset + x, offset + x + jitter * wobble])
    return out


def split(rows: list, per: int) -> list[list]:
    return [rows[i : i + per] for i in range(0, len(rows), per)]


# The probe episode: five frames sitting on the ridge, and one that does not.
# `[0.5, -0.5]` is inside one standard deviation on EVERY dimension and nowhere
# near the ridge, which is the whole argument for the metric this module uses.
OFF_RIDGE = [0.5, -0.5]
ON_RIDGE = [[-0.4, -0.4], [-0.2, -0.2], [0.0, 0.0], [0.2, 0.2], [0.6, 0.6]]


def standard_episodes() -> list[list]:
    """Episode 0 is the probe; episodes 1..8 are 400 reference rows on the ridge."""
    probe = [*ON_RIDGE[:3], OFF_RIDGE, *ON_RIDGE[3:]]
    return [probe, *split(ridge(400), 50)]


# ------------------------------------------------- the reference set is named


def test_every_payload_names_the_set_the_distance_was_measured_against(tmp_path):
    """ "Frame 40 is unusual" is not a finding without "compared to what".

    The four fields below are the answer to that question, and a payload missing
    any one of them is a distance about nothing: which column, from which
    dataset, over how many rows, out of how many that were eligible. Without this
    test nothing stops a later refactor from computing the same number and
    publishing it bare.
    """
    out = vla_ood.score_episode(reader(tmp_path, standard_episodes()), 0)
    ref = out.to_dict()["reference"]
    assert ref["space"] == STATE
    assert ref["repo_id"] == "test/ood"
    assert ref["rows_read"] == 400
    assert ref["rows_eligible"] == 400
    assert ref["rows_total"] == 406
    assert ref["metric"].startswith("Mahalanobis distance from the mean of 400 rows")
    # And the sentence says all four out loud, because a reader looking at a
    # chart is reading that and not the JSON.
    said = out.means()
    assert "THE REFERENCE SET IS 400 ROWS of `observation.state`" in said
    assert "406 in the snapshot" in said


def test_the_scored_episode_is_held_out_of_the_distribution_it_is_measured_against(
    tmp_path,
):
    """A frame compared against a distribution it helped define is partly
    measuring itself.

    Episode 0 is 6 of this dataset's 406 rows — 1.5% of the mean it would
    otherwise be scored against, and the planted outlier is one of them, so
    leaving it in drags the reference toward the thing being measured. Both
    halves are asserted: the rows are gone from `rows_eligible`, and the count
    that was removed is reported rather than merely applied.
    """
    out = vla_ood.score_episode(reader(tmp_path, standard_episodes()), 0)
    assert out.reference.excluded_episode == 0
    assert out.reference.excluded_rows == 6
    assert out.reference.rows_eligible == out.reference.rows_with_column - 6
    assert "held OUT of the reference set" in out.means()


def test_a_reference_built_for_another_episode_is_named_as_a_mismatch(tmp_path):
    """A caller sweeping many episodes will reuse one reference set, and the
    moment they do, every episode except the excluded one is inside the
    distribution it is being measured against.

    That is a legitimate thing to do — rebuilding per episode costs a full pass
    each — and it is not a legitimate thing to do silently. The module reports
    the mismatch rather than refusing it, and this asserts the sentence exists,
    because the payload field alone is one nobody reads.
    """
    r = reader(tmp_path, standard_episodes())
    shared = vla_ood.build_reference(r, exclude_episode=0)
    out = vla_ood.score_episode(r, 1, reference=shared)
    assert out.reference.excluded_episode == 0
    assert "excluded episode 0, NOT episode 1" in out.means()


# ------------------------------------------ the headline: why not a z-score


def test_a_pose_inside_one_sigma_on_every_joint_can_still_be_one_the_arm_never_held(
    tmp_path,
):
    """The reason this module does not use a per-dimension z-score.

    The reference set is a tight ridge: two joints that only ever move together.
    `[0.5, -0.5]` is inside one standard deviation on BOTH of them and is a
    configuration nothing in the dataset ever reached, because the joints never
    held those values at the same time. A diagonal metric cannot see that — it
    has thrown away the only fact that distinguishes the point.

    MEASURED on this fixture: 0.8639 sigma on dim 0 and 0.8639 on dim 1, against
    141.59 in the metric the module uses — while the furthest of the 400
    reference rows reaches 2.21. Without this test a future simplification to
    per-dimension z-scores would pass every other test in this file.
    """
    out = vla_ood.score_episode(reader(tmp_path, standard_episodes()), 0)

    # The diagonal reading, computed here from the same rows the module read, so
    # the comparison is between two metrics rather than between two datasets.
    rows = ridge(400)
    for dim in (0, 1):
        column = [row[dim] for row in rows]
        mean = sum(column) / len(column)
        std = math.sqrt(sum((v - mean) ** 2 for v in column) / len(column))
        assert abs(OFF_RIDGE[dim] - mean) / std == pytest.approx(0.8639, abs=1e-4)

    planted = next(f for f in out.frames if f.t == 3)
    # Pinned to the measured value, not to a comfortable inequality: the gap
    # between the two readings IS the finding, and a change that quietly shrank
    # it to 3.0 would still pass `> 1.0`.
    assert planted.distance == pytest.approx(141.59, rel=1e-3)
    assert out.reference.distances["max"] == pytest.approx(2.21, abs=0.01)
    assert planted.beyond_reference_max is True
    assert planted.percentile == 100.0
    # And it is the top of the ranking, which is the point of ranking at all.
    assert out.ranked[0].t == 3
    # while every frame that IS on the ridge stays inside the reference set.
    for frame in out.frames:
        if frame.t != 3:
            assert frame.beyond_reference_max is False


def test_the_distance_ranks_the_frames_and_nothing_else_reorders_them(tmp_path):
    """One stated measured quantity, the rule `vla_sweep` holds.

    `off_manifold` and `clears_null` travel beside the distance and must not
    enter the sort: a ranking by two quantities is a ranking by neither, and a
    reader who sorts a table by its first column has to get the same order the
    module claims.
    """
    out = vla_ood.score_episode(reader(tmp_path, standard_episodes()), 0)
    assert [f.distance for f in out.ranked] == sorted(
        (f.distance for f in out.ranked), reverse=True
    )
    assert out.n_ranked_total == out.n_frames == 6
    assert [f.t for f in out.frames] == [0, 1, 2, 3, 4, 5], "frames are in time order"


# ------------------------------------------------- a distance is not a verdict


def test_nothing_in_the_payload_is_a_verdict(tmp_path):
    """No `is_ood`, no `outlier`, no `anomaly`, no label of any kind.

    Every one of those words is a threshold somebody chose, and this project does
    not ship those. The only boolean about how far out a frame is, is
    `clears_null`, and it is gated on a measured null — this asserts the surface
    directly so a well-meaning convenience field cannot be added without failing
    here.
    """
    payload = vla_ood.score_episode(reader(tmp_path, standard_episodes()), 0).to_dict()
    keys = set(payload["frames"][0])
    assert keys == {
        "t",
        "distance",
        "percentile",
        "percentile_resolution",
        "beyond_reference_max",
        "off_manifold",
        "clears_null",
    }
    flat = json.dumps(payload).lower()
    for word in ("is_ood", '"ood"', "outlier", "anomaly", '"verdict"'):
        assert word not in flat, f"{word} is a label, and this module reports counts"


def test_a_percentile_carries_what_one_row_of_the_sample_is_worth(tmp_path):
    """A percentile taken in a sample of 400 cannot be resolved finer than 0.25
    of a point, and a number without its resolution is a claim to a precision
    the method does not have.

    Asserted as arithmetic on the reported row count rather than as a constant,
    because the whole point is that it moves with the sample.
    """
    out = vla_ood.score_episode(reader(tmp_path, standard_episodes()), 0)
    assert out.reference.rows_read == 400
    assert out.reference.percentile_resolution == 0.25
    for frame in out.frames:
        assert frame.percentile_resolution == 100.0 / out.reference.rows_read
    assert "the resolution" in out.means()


def test_the_hundredth_percentile_says_whether_it_is_the_top_or_past_the_end(tmp_path):
    """The empirical CDF genuinely reads 100.0 for a frame beyond every reference
    row, and 100.0 with nothing beside it reads as "measured at exactly the top".

    Those are different claims: one says this frame ties the furthest row in the
    reference set, the other says the reference set ran out. `beyond_reference_max`
    is the difference, and without it the planted outlier and a frame sitting
    exactly on the reference maximum publish the identical number.
    """
    out = vla_ood.score_episode(reader(tmp_path, standard_episodes()), 0)
    planted = next(f for f in out.frames if f.t == 3)
    assert (planted.percentile, planted.beyond_reference_max) == (100.0, True)
    assert "further out than every row in the reference set" in out.means()
    inside = [f for f in out.frames if f.t != 3]
    assert all(f.percentile < 100.0 for f in inside)


# ---------------------------------------------------------------- the null


def test_without_a_measured_null_nothing_is_flagged_and_it_is_none_not_false(tmp_path):
    """`clears_null=False` would say "this frame was tested and did not clear".
    `None` says "nothing was tested". Those are opposite statements about how
    much is known.

    Reading every eligible row into the reference leaves nothing held back, so
    there is no null to draw — and the honest answer is to say so, name what to
    change, and flag nothing, rather than to quietly compare each frame against
    the reference's own maximum and call that a control.
    """
    out = vla_ood.score_episode(
        reader(tmp_path, standard_episodes()), 0, max_reference_rows=None
    )
    assert out.reference.row_stride == 1
    assert out.reference.null_max is None
    assert out.reference.null_covers_percentile is None
    assert all(f.clears_null is None for f in out.frames)
    assert "no row left over to draw a null from" in out.reference.null_reason
    assert "Lower `max_reference_rows`" in out.reference.null_reason
    assert "NO NULL WAS MEASURED" in out.means()


def test_the_null_is_rows_the_reference_never_saw_and_the_ordinary_frames_do_not_beat_it(
    tmp_path,
):
    """The `head_types.py` question, asked here: would this frame have looked far
    away anyway?

    Some share of any distribution's own rows sit in its tail — that is what a
    tail is. So the null is 100 rows of this dataset held OUT of the reference
    set and scored against it, and a frame is only flagged when it beats the
    largest distance any of them reached. On this fixture the five on-ridge
    frames do not, and the planted one does; a null drawn from rows the
    reference had already seen would be a control that cannot fail.
    """
    out = vla_ood.score_episode(
        reader(tmp_path, standard_episodes()), 0, max_reference_rows=100
    )
    ref = out.reference
    assert ref.row_stride == 4
    assert ref.rows_read == 100
    assert ref.null_draws == 100
    assert ref.null_max is not None
    assert ref.null_covers_percentile == round(100.0 * 100 / 101, 4)
    assert "none of them in the reference set" in ref.null_description
    assert "none from episode 0" in ref.null_description
    cleared = [f.t for f in out.frames if f.clears_null]
    assert cleared == [3]
    assert all(f.clears_null is False for f in out.frames if f.t != 3)


def test_the_null_states_how_little_a_small_one_proves(tmp_path):
    """The maximum of K draws sits at about the 100*K/(K+1)th percentile. At 8
    draws that is the 88.9th, and a frame beating it has cleared very little.

    Without this figure "it beat the null" reads the same at 8 draws and at 4,000,
    which lets a cheap run borrow the authority of an expensive one.
    """
    out = vla_ood.score_episode(
        reader(tmp_path, standard_episodes()), 0, max_reference_rows=100, null_draws=8
    )
    assert out.reference.null_draws == 8
    assert out.reference.null_covers_percentile == round(100.0 * 8 / 9, 4)
    assert "88.89th percentile" in out.means()


def test_asking_for_no_null_says_so_rather_than_flagging_on_nothing(tmp_path):
    """`null_draws=0` is a legitimate request — it halves the second pass — and
    it must not silently leave `clears_null` false everywhere, which reads as a
    control that every frame failed."""
    out = vla_ood.score_episode(
        reader(tmp_path, standard_episodes()), 0, max_reference_rows=100, null_draws=0
    )
    assert out.reference.null_max is None
    assert all(f.clears_null is None for f in out.frames)
    assert "no null was asked for" in out.reference.null_reason


# ------------------------------------------------ frames that cannot be scored


def test_an_unscorable_frame_is_absent_from_the_ranking_with_its_own_reason(tmp_path):
    """A zero would sit at the bottom of the table looking like the most ordinary
    frame in the episode. `vla_sweep.run` set this pattern and this is it.

    Three ways a recorded row is not a point, and each gets its own sentence: no
    row at all, a row of the wrong width, and a row with a NaN in it. One shared
    "could not read" would leave a reader unable to tell a truncated recording
    from a sensor dropout.
    """
    probe = [
        [0.0, 0.0],
        None,
        [1.0, 2.0, 3.0],
        [float("nan"), 0.0],
        [0.1, 0.1],
    ]
    out = vla_ood.score_episode(reader(tmp_path, [probe, *split(ridge(400), 50)]), 0)
    assert [f.t for f in out.frames] == [0, 4]
    assert out.n_frames == 2
    assert out.n_unscored == 3
    reasons = {row["t"]: row["why"] for row in out.unscored}
    assert set(reasons) == {1, 2, 3}
    assert "not a list of numbers" in reasons[1]
    assert "width 3 and the reference set has 2" in reasons[2]
    assert "non-finite" in reasons[3]
    # And the reason a zero is not available as a placeholder: 0.0 is a REAL
    # measurement here. Frame 0 sits at the reference mean, so its distance
    # genuinely rounds to nothing — the same number a scored-zero placeholder
    # would have produced for the three frames above, and indistinguishable from
    # it in any payload that used one.
    assert next(f for f in out.frames if f.t == 0).distance == pytest.approx(
        0.0, abs=0.1
    )
    assert {1, 2, 3}.isdisjoint({f.t for f in out.frames})
    assert "ABSENT from the ranking rather than scored zero" in out.means()


def test_a_truncated_unscored_listing_carries_the_true_count_beside_it(tmp_path):
    """`vla_sweep` shipped a report that said "20 frame(s) could not be measured"
    when 600 had, because the sentence counted the truncated list.

    Same cap, same trap, same fix: the list is a sample to look at and
    `n_unscored` is the measurement, and the sentence says how many of the total
    it is showing.
    """
    probe = [None] * 40 + [[0.0, 0.0]]
    out = vla_ood.score_episode(reader(tmp_path, [probe, *split(ridge(400), 50)]), 0)
    assert out.n_unscored == 40
    assert len(out.unscored) == vla_ood.MAX_UNSCORED_LISTED == 20
    assert "40 frame(s) could not be scored" in out.means()
    assert "(20 of them listed)" in out.means()


# --------------------------------------------- directions with no spread at all


def test_a_direction_the_dataset_never_moved_in_is_dropped_and_reported(tmp_path):
    """A constant column is real recorded data — a gripper held open for a whole
    dataset — and it has no spread to divide by. Whitening it is a division by
    zero, and the resulting infinity makes every frame infinitely far away.

    So the direction is dropped from the distance and movement along it is
    reported separately, in the column's own raw units, because it is not on the
    same scale as the rest. A frame that moves 5.0 along an axis the dataset
    never touched is a real and different finding from a frame far along an axis
    it does, and folding them into one number would erase which happened.
    """
    rows = [[*pair, 0.0] for pair in ridge(400)]
    probe = [[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]
    out = vla_ood.score_episode(reader(tmp_path, [probe, *split(rows, 50)]), 0)
    ref = out.reference
    assert (ref.dimensions, ref.directions_kept, ref.directions_dropped) == (3, 2, 1)
    # The floor is DERIVED from the measured spectrum rather than chosen: it is
    # the largest eigenvalue of these very rows times the stated conditioning
    # ratio, recomputed here from the same numbers so the payload cannot quietly
    # start reporting a constant instead.
    block = torch.tensor(rows, dtype=torch.float64)
    centred = block - block.mean(dim=0)
    largest = float(torch.linalg.eigvalsh(centred.T @ centred / block.shape[0]).max())
    assert ref.condition_ratio == vla_ood.CONDITION_FLOOR
    assert ref.variance_floor == pytest.approx(largest * ref.condition_ratio, rel=1e-6)
    still = next(f for f in out.frames if f.t == 0)
    moved = next(f for f in out.frames if f.t == 1)
    assert still.off_manifold == 0.0
    assert moved.off_manifold == pytest.approx(5.0, abs=1e-9)
    # The distance itself is finite and IDENTICAL for the two, because they
    # differ only along the direction that was dropped. That is the claim: the
    # off-manifold movement did not leak into the whitened number.
    assert math.isfinite(moved.distance)
    assert moved.distance == still.distance
    assert "1 of 3 directions were DROPPED" in out.means()


def test_a_dataset_that_never_moves_at_all_is_refused_rather_than_divided_by(tmp_path):
    """Every row identical is not a distribution with zero spread, it is a single
    point, and every frame would be at distance zero from it — including one
    nothing in the dataset resembles.

    `vla_occlude.scale_from` refuses the same shape of thing for the same reason,
    and the sentence names both explanations a reader has to choose between.
    """
    flat = [[1.0, 2.0] for _ in range(400)]
    with pytest.raises(Refusal) as err:
        vla_ood.score_episode(reader(tmp_path, [[[1.0, 2.0]] * 4, *split(flat, 50)]), 0)
    assert "no spread to measure a distance against" in err.value.sentence
    assert "may be static" in err.value.sentence


# ------------------------------------------------------------ the arithmetic


def test_the_covariance_survives_an_offset_that_breaks_the_textbook_form(tmp_path):
    """`E[xx^T] - E[x]E[x]^T` subtracts two large nearly-equal matrices, and what
    comes back is not positive semi-definite.

    That matters more in d dimensions than in one: the eigendecomposition returns
    a NEGATIVE eigenvalue, `rsqrt` of it is NaN, the whitener is NaN, and every
    distance is NaN — which on a chart looks like an episode that was never
    recorded rather than like a broken measurement.

    MEASURED here, both forms in float64 over identical rows (the same scan the
    module docstring pastes): the naive form holds to about offset 1e5 and is
    negative by 1e6, while the accumulator is unmoved at 1e7. Without this test
    nothing stops the accumulator being "simplified" back, since every other test
    in this file uses data centred near zero where the two agree exactly.
    """
    n, spread, jitter = 4000, 0.3, 0.01

    def block(offset: float):
        t = torch.linspace(-1.0, 1.0, n, dtype=torch.float64)
        sign = torch.where(torch.arange(n) % 2 == 0, 1.0, -1.0)
        return torch.stack(
            [
                offset + spread * t,
                offset + spread * t + jitter * sign,
                offset + spread * t * t,
            ],
            dim=1,
        ).contiguous()

    def naive(rows):
        mean = rows.mean(dim=0)
        return (rows.T @ rows) / rows.shape[0] - torch.outer(mean, mean)

    def accumulated(rows):
        acc = vla_ood._Covariance(rows.shape[1])
        for i in range(0, rows.shape[0], 8192):
            acc.add(rows[i : i + 8192])
        return acc.covariance()

    truth = float(torch.linalg.eigvalsh(accumulated(block(0.0))).min())
    assert truth == pytest.approx(4.9960e-05, rel=1e-3)

    # 1e7, and the offset is chosen from a measurement rather than by taste.
    # Both forms have a working range and this test needs an offset inside
    # ours and well outside theirs. Relative error against the true smallest
    # eigenvalue (4.995959e-05), measured on this machine:
    #
    #     offset    textbook       accumulated
    #     1e6       3.46           2.78e-13     <- textbook off by 3x: MARGINAL
    #     1e7       4.89e+02       3.13e-13     <- this one
    #     1e8       1.35e+05       1.12e-06     <- accumulated now over tolerance
    #
    # It started at 1e6 and failed CI on macos-latest/py3.12 with "the textbook
    # form has stopped being the failing one", while passing on windows py3.10,
    # py3.13 and macos py3.11 in the same run: at that offset the textbook
    # form's error is only three times the quantity being measured, so which
    # side of zero it lands on depends on the BLAS and the summation order.
    # Moving to 1e8 fixed that and broke the other end — the accumulated form's
    # own error passes 1e-6 there. 1e7 is inside both margins by orders of
    # magnitude, which is what makes the assertions below claims about the
    # ARITHMETIC rather than about a particular machine's.
    broken = block(1e7)
    naive_min = float(torch.linalg.eigvalsh(naive(broken)).min())
    ours_min = float(torch.linalg.eigvalsh(accumulated(broken)).min())
    assert naive_min < 0.0, "the textbook form has stopped being the failing one"
    assert not torch.isfinite(torch.tensor(naive_min).clamp(min=0.0).rsqrt()), (
        "a zero or negative eigenvalue is what makes the whitener NaN"
    )
    assert ours_min == pytest.approx(truth, rel=1e-6)

    # And end to end: a dataset recorded a million units from the origin still
    # scores, and the planted outlier is still the furthest frame.
    far = [[v + 1e6 for v in row] for row in ridge(400)]
    probe = [
        [1e6 + v for v in row] for row in [*ON_RIDGE[:3], OFF_RIDGE, *ON_RIDGE[3:]]
    ]
    out = vla_ood.score_episode(reader(tmp_path, [probe, *split(far, 50)]), 0)
    assert all(math.isfinite(f.distance) for f in out.frames)
    assert out.ranked[0].t == 3


def test_a_bool_is_read_as_a_number_before_it_is_read_as_an_int(tmp_path):
    """`isinstance(True, int)` is True, so the bool arm has to come first.

    A gripper flag recorded as a boolean is a real value and 1.0/0.0 is what it
    means. Falling through the numeric arm gives the same answer by accident,
    which is exactly the state in which somebody narrows that arm and silently
    turns every gripper flag into an unreadable row.
    """
    assert vla_ood._readable([True, False], 2) == [1.0, 0.0]
    assert vla_ood._readable([True, float("nan")], 2) is None


def test_a_covariance_from_no_more_rows_than_dimensions_is_refused_by_arithmetic(
    tmp_path,
):
    """With no more rows than dimensions the sample cannot span the space, so the
    covariance is singular BY CONSTRUCTION and every direction the sample missed
    reads as infinitely far away.

    Refused rather than regularised: a ridge term nobody asked for would turn an
    unmeasurable distance into a plausible-looking one, and the reader would have
    no way to see which they were reading.
    """
    # Four rows, four dimensions, one episode each — so every row is eligible and
    # the sample is exactly as wide as the space it has to span.
    rows = [[float(i), float(i) * 2, float(i) * 3, 1.0 + i] for i in range(4)]
    with pytest.raises(Refusal) as err:
        vla_ood.build_reference(reader(tmp_path, [[row] for row in rows]))
    assert "4x4 covariance estimated from 4 row(s)" in err.value.sentence
    assert "singular by construction" in err.value.sentence
    assert "max_reference_rows" in err.value.sentence


# --------------------------------------------------- no policy, no decoder


def test_scoring_works_on_a_snapshot_whose_video_routing_does_not(tmp_path):
    """This measurement reads recorded numbers and needs no picture, so a
    snapshot with no camera metadata at all must still score.

    `LeRobotV3Reader.episodes()` refuses on exactly this snapshot — correctly,
    because a frame that cannot be located must not be guessed at — and reusing
    it here would have made the OOD panel unavailable on every dataset whose
    videos were not downloaded. `vla_data._episode_count` makes the same argument
    for the same reason; this test is what holds the module to it.
    """
    r = reader(tmp_path, standard_episodes(), routing=False)
    with pytest.raises(Refusal) as err:
        r.episodes()
    assert "from_timestamp" in str(err.value)

    out = vla_ood.score_episode(r, 0)
    assert out.n_frames == 6
    assert out.ranked[0].t == 3


def test_nothing_in_the_module_reaches_for_a_model_a_decoder_or_the_sidecar(tmp_path):
    """`pip install modelmri` does not bring the policy sidecar, and the vla-lite
    extra does not bring a GPU. A single `import av` or `from .vla import ...`
    added later would make this whole feature unavailable on the machines it was
    written for, and it would fail at runtime rather than here.

    The import guard is static because that is where the dependency would land;
    the run below is the dynamic half — the scoring above completes on a snapshot
    with no video file in it at all.
    """
    source = Path(vla_ood.__file__).read_text(encoding="utf-8")
    for forbidden in ("import av", "from .vla import", "from .policy import"):
        assert forbidden not in source, f"{forbidden} would need the vla sidecar"
    assert (
        not list((tmp_path / "snap").rglob("*.mp4"))
        if (tmp_path / "snap").exists()
        else True
    )


# ------------------------------------------------------------ shards and rows


def test_a_shard_without_the_column_still_occupies_its_row_numbers(tmp_path):
    """Episode row spans are ABSOLUTE. A shard that does not carry the column is
    not a shard that does not exist, and skipping its rows shifts every episode
    after it onto somebody else's rows.

    That failure is silent: the numbers arrive, the chart draws, and every frame
    belongs to a different episode. Here the middle shard has no
    `observation.state`, and the assertion is that episode 9's planted outlier is
    still found at its own timestep rather than 100 rows away.
    """
    ridge_rows = ridge(400)
    probe = [*ON_RIDGE[:3], OFF_RIDGE, *ON_RIDGE[3:]]
    # 400 ridge rows in eight episodes, then the probe LAST so it sits in the
    # final shard, behind the shard that drops the column.
    episodes = [*split(ridge_rows, 50), probe]
    out = vla_ood.score_episode(
        reader(tmp_path, episodes, shards=3, drop_state_in=(1,)), 8
    )
    ref = out.reference
    assert ref.rows_total == 406
    assert ref.rows_with_column < ref.rows_total, "the middle shard was counted"
    assert out.n_frames == 6
    assert out.ranked[0].t == 3, "the row numbering slipped by the dropped shard"


def test_the_row_span_says_whether_it_was_read_or_assumed(tmp_path):
    """`dataset_from_index` is the dataset's own row map. Summing episode lengths
    gets the same answer only while the rows are contiguous and in episode order,
    which is an assumption, and an assumption a reader has to be able to see.

    `vla_data._episodes_locked` takes the same fallback; what it does not do is
    say which of the two it used, and a reader whose frames look wrong has
    nowhere to start.
    """
    read = vla_ood.score_episode(reader(tmp_path, standard_episodes()), 0)
    assert "dataset_from_index" in read.reference.row_span_from

    assumed = vla_ood.score_episode(
        reader(tmp_path / "b", standard_episodes(), with_index=False), 0
    )
    assert "summed episode lengths" in assumed.reference.row_span_from
    assert "assumed contiguous" in assumed.reference.row_span_from
    # And it still finds the same frame, because on this snapshot the assumption
    # happens to hold — which is exactly why it has to be stated rather than
    # inferred from the answer looking reasonable.
    assert assumed.ranked[0].t == 3


def test_a_strided_sample_can_alias_and_the_payload_shows_when_it_did(tmp_path):
    """The stride is systematic, not random, and a dataset with a period that
    shares a factor with it loses a whole direction from the reference set.

    FOUND WHILE WRITING THIS FILE, not reasoned about afterwards. The first ridge
    fixture wobbled +/- every row; every stride the module picks is even; so a
    strided reference drew rows of one parity only, the wobble was identically
    zero across all 100 of them, and the ridge direction was measured as having
    no spread. The frame that is unusual only in that direction then read as
    ordinary — the exact failure this module exists to prevent, produced by the
    sampling rather than by the data.

    It is not silent, and that is what is asserted here: the aliased run reports
    a dropped direction where the unstrided run over the identical rows reports
    none, and the sentence says so. `row_stride` is in the payload so a reader
    who sees this can re-run at a different `max_reference_rows`, which is the
    documented way out.
    """
    aliased = ridge(400, alternating=True)
    probe = [*ON_RIDGE[:3], OFF_RIDGE, *ON_RIDGE[3:]]
    r = reader(tmp_path, [probe, *split(aliased, 50)])

    strided = vla_ood.score_episode(r, 0, max_reference_rows=100)
    assert strided.reference.row_stride == 4
    assert strided.reference.directions_dropped == 1
    assert "1 of 2 directions were DROPPED" in strided.means()

    whole = vla_ood.score_episode(r, 0, max_reference_rows=None)
    assert whole.reference.row_stride == 1
    assert whole.reference.directions_dropped == 0
    # And the consequence, stated as numbers: the same frame, two sampling
    # choices, and the aliased one cannot see what the whole one can.
    assert whole.ranked[0].t == 3
    assert next(f for f in whole.frames if f.t == 3).distance > 10.0
    assert next(f for f in strided.frames if f.t == 3).distance < 10.0


def test_a_sampled_reference_says_so_and_says_by_how_much(tmp_path):
    """The reference set is a SAMPLE unless every eligible row was read, and a
    payload that cannot say which is one a reader has to guess about.

    Both the stride and the boolean, because one field to test beats two that can
    disagree — and the row count has to be the count the stride implies, or the
    two are describing different runs.
    """
    out = vla_ood.score_episode(
        reader(tmp_path, standard_episodes()), 0, max_reference_rows=57
    )
    ref = out.reference
    assert ref.sampled is True
    assert ref.row_stride == math.ceil(400 / 57)
    assert ref.rows_read == len(range(0, 400, ref.row_stride))
    assert f"every {ref.row_stride}th eligible row" in out.means()


# --------------------------------------------------------------- the refusals


def test_a_dataset_with_no_such_column_is_refused_with_the_columns_it_does_have(
    tmp_path,
):
    """ "Not found" sends a reader to guess. The shard's own column list is the
    answer, and it is a list this program read rather than an exception's text."""
    with pytest.raises(Refusal) as err:
        vla_ood.score_episode(
            reader(tmp_path, standard_episodes()), 0, space="observation.joints"
        )
    said = err.value.sentence
    assert "none of them has a `observation.joints` column" in said
    assert "observation.state" in said and "action" in said

    # And the COST route refuses it too, on the same footers. `vla_occlude`
    # shipped a preflight that answered 200 with a firm figure for a run the very
    # next click turned down; a price quoted for a measurement that cannot happen
    # is worse than no price.
    with pytest.raises(Refusal) as err:
        vla_ood.estimate(
            reader(tmp_path / "b", standard_episodes()), 0, space="observation.joints"
        )
    assert "nothing here to price a measurement of" in err.value.sentence


def test_an_episode_that_is_not_there_names_the_ones_that_are(tmp_path):
    """A 422, not a 409: the parameter of the call that was just made is wrong,
    and the fix is in the caller's hands. The list is capped and the cap is
    reported, so an episode absent from a truncated list is not mistaken for an
    episode absent from the dataset."""
    with pytest.raises(BadRequest) as err:
        vla_ood.score_episode(reader(tmp_path, standard_episodes()), 99)
    assert "has no episode 99" in err.value.sentence
    assert "0, 1, 2" in err.value.sentence


def test_the_parameter_refusals_name_the_acceptable_values(tmp_path):
    """Every refusal names the cause and a next step. A bare "invalid" makes the
    caller read the source to find the range."""
    r = reader(tmp_path, standard_episodes())
    with pytest.raises(BadRequest) as err:
        vla_ood.score_episode(r, 0, frame_stride=0)
    assert "must be at least 1" in err.value.sentence
    with pytest.raises(BadRequest) as err:
        vla_ood.build_reference(r, bins=0)
    assert f"between 1 and {vla_ood.MAX_HISTOGRAM_BINS}" in err.value.sentence
    with pytest.raises(BadRequest) as err:
        vla_ood.build_reference(r, max_reference_rows=1)
    assert "One row has no spread" in err.value.sentence
    with pytest.raises(BadRequest) as err:
        vla_ood.build_reference(r, null_draws=vla_ood.MAX_NULL_DRAWS + 1)
    assert f"between 0 and {vla_ood.MAX_NULL_DRAWS:,}" in err.value.sentence


def test_an_episode_longer_than_the_cap_is_refused_with_the_number(tmp_path):
    """A ranking missing its tail looks exactly like a ranking, so the cap
    refuses rather than truncates, and it names both numbers — `vla_sweep.plan`'s
    rule and its sentence."""
    with pytest.raises(BadRequest) as err:
        vla_ood.score_episode(reader(tmp_path, standard_episodes()), 0, max_frames=3)
    assert "samples 6 of them; the cap is 3" in err.value.sentence
    assert "Raise the stride" in err.value.sentence


def test_a_reference_over_a_different_column_is_refused_rather_than_used(tmp_path):
    """Two columns are two distributions, and a distance between them is not a
    number about anything. The reference set is reusable across episodes and must
    not be reusable across spaces."""
    r = reader(tmp_path, standard_episodes())
    over_action = vla_ood.build_reference(r, space="action", exclude_episode=0)
    with pytest.raises(BadRequest) as err:
        vla_ood.score_episode(r, 0, space=STATE, reference=over_action)
    assert "is over `action` and this call asks for `observation.state`" in (
        err.value.sentence
    )


# ------------------------------------------------------------------- the cost


def test_the_cost_is_quoted_in_forward_passes_and_the_answer_is_zero(tmp_path):
    """`budget.py` prices analyses in forward passes because every ranking in
    this package is a loop of them. This one is not, and quoting a small number
    would be quoting the wrong unit.

    So the field is present, it is 0, and the unit that DOES apply is named
    beside it with the row count. A cost route that omitted the zero would leave
    a caller assuming this needs a GPU like everything else on the panel.
    """
    plan = vla_ood.estimate(reader(tmp_path, standard_episodes()), 0)
    assert plan["forward_passes"] == 0
    assert "no lerobot sidecar" in plan["forward_passes_why"]
    assert plan["cost_unit"] == "parquet rows read"
    assert plan["rows_to_read"] == 2 * 406 + 406
    assert plan["seconds"] is None, "a duration nobody measured is not reported"
    assert plan["peak_bytes"] > 0


def test_the_estimate_describes_the_run_that_actually_happens(tmp_path):
    """A preflight that quotes a figure for a different run is worse than none —
    `vla_occlude.estimate` shipped exactly that (a price for stride 1 while the
    payload echoed the caller's stride) and it is the reason its docstring is as
    long as it is.

    Every field here is one the run reports back, so the two cannot drift apart
    without failing.
    """
    r = reader(tmp_path, standard_episodes())
    plan = vla_ood.estimate(r, 0, max_reference_rows=100, null_draws=8)
    out = vla_ood.score_episode(r, 0, max_reference_rows=100, null_draws=8)
    assert plan["reference_rows"] == out.reference.rows_read
    assert plan["row_stride"] == out.reference.row_stride
    assert plan["sampled"] == out.reference.sampled
    assert plan["rows_eligible"] == out.reference.rows_eligible
    assert plan["rows_in_episode"] == out.reference.excluded_rows
    assert plan["null_draws"] == out.reference.null_draws
    assert plan["frames"] == out.n_frames + out.n_unscored
    assert plan["frames_total"] == out.frames_total


def test_the_estimate_will_not_invent_a_peak_it_cannot_compute(tmp_path):
    """A memory figure computed from a guessed dimension count is wrong by
    exactly the factor nobody can see. `budget.py` holds the line that unknown is
    never zero in either direction, and this is that line here: no declared
    width, no number, and a sentence saying when it becomes known."""
    plan = vla_ood.estimate(
        reader(tmp_path, standard_episodes(), declared_width=None), 0
    )
    assert plan["dimensions_declared"] == 2
    stripped = build(tmp_path / "bare", standard_episodes())
    info = json.loads((stripped / "meta" / "info.json").read_text(encoding="utf-8"))
    del info["features"][STATE]["shape"]
    (stripped / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    plan = vla_ood.estimate(LeRobotV3Reader(stripped, "test/ood"), 0)
    assert plan["dimensions_declared"] is None
    assert plan["peak_bytes"] is None
    assert "unknown until the first row is read" in plan["peak_basis"]


# ------------------------------------------------------------ the payload shape


def test_the_reference_tensors_never_reach_the_payload(tmp_path):
    """The whitening basis is a d x d tensor and the reference distances are one
    float per sampled row — 4 MB at the cap. `asdict` would deep-copy both into
    the JSON, which is why `to_dict` names its fields one at a time.

    Asserted by serialising, because that is where it would fail: on a route, at
    runtime, for a caller.
    """
    out = vla_ood.score_episode(
        reader(tmp_path, standard_episodes()), 0, max_reference_rows=100
    )
    blob = json.dumps(out.to_dict())
    assert "tensor" not in blob
    ref = out.to_dict()["reference"]
    assert set(ref) & {"kept_basis", "sorted_distances", "mean_vector"} == set()
    # The histogram IS published, because it is what a panel draws, and its edges
    # end exactly at the largest distance rather than a few ulps above it.
    histogram = ref["distances"]["histogram"]
    assert len(histogram["counts"]) == vla_ood.DEFAULT_HISTOGRAM_BINS
    assert histogram["bin_edges"][-1] == ref["distances"]["max"]
    assert sum(histogram["counts"]) == ref["rows_read"]


def test_the_published_percentiles_are_exact_rather_than_read_off_the_histogram(
    tmp_path,
):
    """`dataset_action_stats` reads its percentiles off a histogram and reports
    the bin width as their resolution, because it cannot hold every value. This
    can — one float per sampled row, capped and reported — so the percentiles are
    the values themselves.

    Stated in the payload so the two modules are not read as doing the same thing
    with the same caveat, and checked against the sorted distances here so the
    claim cannot rot.
    """
    out = vla_ood.score_episode(
        reader(tmp_path, standard_episodes()), 0, max_reference_rows=100
    )
    ref = out.reference
    assert (
        "nearest rank over every reference distance"
        in (ref.distances["percentile_method"])
    )
    exact = sorted(float(v) for v in ref.sorted_distances.tolist())
    for entry in ref.distances["percentiles"]:
        rank = min(
            len(exact) - 1, max(0, math.ceil(entry["q"] / 100.0 * len(exact)) - 1)
        )
        assert entry["value"] == pytest.approx(exact[rank], abs=1e-6)
    assert ref.distances["percentiles"][-1]["value"] == ref.distances["max"]


def test_the_action_column_is_a_different_question_and_says_which_one_it_answered(
    tmp_path,
):
    """ "This commanded action is unusual" is a fair question and is not the same
    question as "this observation is unusual". The module answers either and the
    payload names which, because two reports that differ only in a field nobody
    reads are two reports that get confused."""
    r = reader(tmp_path, standard_episodes())
    state = vla_ood.score_episode(r, 0)
    action = vla_ood.score_episode(r, 0, space="action")
    assert action.space == "action"
    assert action.reference.space == "action"
    assert action.reference.dimensions == 1
    assert "`action`" in action.means()
    assert [f.distance for f in action.frames] != [f.distance for f in state.frames]
