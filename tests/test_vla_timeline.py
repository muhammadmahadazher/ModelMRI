# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Several series on ONE axis, and the four ways that stops being true.

The robot panel had a frame and a scrubber, which answers "what did the camera
see at t" and nothing else. The questions people bring to a recorded episode
are about COINCIDENCE — the gripper closed here, what was the state doing, did
the reward move — and every one of those needs two series on one axis. Read
off two panels with two x-ranges, a coincidence gets asserted that is not
there.

So the shared axis is the product, and these are the ways it silently stops
being shared:

  a missing column drawn      an empty reward track at zero says the reward
  as zero                     WAS zero, which nobody measured
  a stride smoothed over      a line through dropped timesteps claims frames
                              nobody read
  a non-finite interpolated   corruption in a recording is not a value
  an invented dimension name  "dim 2" looks exactly like a published name and
                              the dataset never wrote it

The fixtures are synthetic and their numbers are planted, so every assertion
is against a value known before the code ran.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa
import pyarrow.parquet as pq

from modelmri import vla_data, vla_timeline
from modelmri.errors import BadRequest, Refusal

LENGTHS = [6, 4]
FPS = 10.0


def build(root, *, columns=("action", "observation.state"), rows=None, names=True):
    """A LeRobot v3.0 snapshot carrying exactly the columns named."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    feature = {"dtype": "float32", "shape": [2]}
    if names:
        feature = {**feature, "names": ["motor_a", "motor_b"]}
    (root / "meta" / "info.json").write_text(
        __import__("json").dumps(
            {
                "fps": FPS,
                "features": {
                    "action": dict(feature),
                    "observation.state": dict(feature),
                    "next.reward": {"dtype": "float32", "shape": [1], "unit": "points"},
                    # Declared so `_video_key` resolves to the camera the
                    # routing columns below are written for. Nothing here
                    # decodes a frame — the timeline reads the frame TABLE —
                    # but `episodes()` locates spans per camera, so the two
                    # have to agree on which camera exists.
                    "observation.images.top": {
                        "dtype": "video",
                        "shape": [96, 96, 3],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    n = len(LENGTHS)
    meta = {
        "episode_index": list(range(n)),
        "length": list(LENGTHS),
        "tasks": [["push"] for _ in range(n)],
        "dataset_from_index": [sum(LENGTHS[:i]) for i in range(n)],
    }
    for cam in ("observation.images.top",):
        starts, at = [], 0.0
        for i in range(n):
            starts.append(at)
            at += LENGTHS[i] / FPS
        meta[f"videos/{cam}/from_timestamp"] = starts
        meta[f"videos/{cam}/to_timestamp"] = [
            s + LENGTHS[i] / FPS for i, s in enumerate(starts)
        ]
        meta[f"videos/{cam}/chunk_index"] = [0] * n
        meta[f"videos/{cam}/file_index"] = [0] * n
    pq.write_table(
        pa.table(meta), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )

    total = sum(LENGTHS)
    table = rows or {}
    if "action" in columns and "action" not in table:
        table["action"] = [[float(i), float(-i)] for i in range(total)]
    if "observation.state" in columns and "observation.state" not in table:
        table["observation.state"] = [[float(i) * 2, 1.0] for i in range(total)]
    if "next.reward" in columns and "next.reward" not in table:
        table["next.reward"] = [float(i) / 10 for i in range(total)]
    table["timestamp"] = [round(i / FPS, 4) for i in range(total)]
    pq.write_table(pa.table(table), root / "data" / "chunk-000" / "file-000.parquet")
    return root


def reader(tmp_path, name="snap", **kw):
    return vla_data.LeRobotV3Reader(build(tmp_path / name, **kw), "test/timeline")


# ------------------------------------------------------------- one axis


def test_every_track_is_indexed_by_the_same_timesteps(tmp_path):
    """THE PRODUCT. Two series sampled on different grids and drawn on one
    chart is the failure this exists to prevent, and it is invisible once
    rendered — both lines look like lines."""
    out = vla_timeline.tracks(reader(tmp_path), 0)
    assert out.timesteps == list(range(LENGTHS[0]))
    for track in out.tracks:
        for dimension in track.series:
            assert len(dimension) == len(out.timesteps), track.column


def test_an_episode_reads_its_own_rows_and_not_the_one_before_it(tmp_path):
    """LeRobot concatenates every episode into one table, so an episode is a
    SPAN. Reading from row 0 for every one of them is the mistake
    `vla_data`'s own routing tests were written against, and it produces a
    perfectly plausible timeline of somebody else's episode."""
    out = vla_timeline.tracks(reader(tmp_path), 1)
    # Episode 1 starts at row 6, so its first action is [6.0, -6.0].
    action = next(t for t in out.tracks if t.column == "action")
    assert action.series[0][0] == 6.0
    assert action.series[1][0] == -6.0
    assert len(out.timesteps) == LENGTHS[1]


def test_seconds_start_at_this_episodes_own_beginning(tmp_path):
    """Absolute timestamps are seconds into the CONCATENATED file, so a reader
    shown them sees episode 1 start at 0.6 s. Relative to its own first frame
    is the only reading that means anything on one episode's axis."""
    out = vla_timeline.tracks(reader(tmp_path), 1)
    assert out.seconds is not None
    assert out.seconds[0] == 0.0
    assert out.seconds[1] == pytest.approx(1 / FPS, abs=1e-9)


# -------------------------------------------------- what it will not draw


def test_a_column_this_dataset_lacks_is_absent_rather_than_a_line_at_zero(tmp_path):
    """`next.reward` is optional in the format and most manipulation datasets
    have none. A track drawn at zero says the reward WAS zero for the whole
    episode — a measurement nobody took — where absence says the true thing."""
    out = vla_timeline.tracks(
        reader(tmp_path, columns=("action", "observation.state")), 0
    )
    drawn = {t.column for t in out.tracks}
    assert "next.reward" not in drawn
    gone = {a["column"] for a in out.absent}
    assert "next.reward" in gone
    why = next(a["why"] for a in out.absent if a["column"] == "next.reward")
    assert "nobody measured" in why
    assert "publishes no `next.reward`" in why
    assert "publishes no `next.reward`" in out.means()


def test_a_stride_leaves_gaps_rather_than_a_line_through_them(tmp_path):
    """A line drawn through dropped timesteps claims frames nobody read. The
    stride is in the payload and the sentence says a feature narrower than it
    can fall entirely between two points."""
    out = vla_timeline.tracks(reader(tmp_path), 0, max_points=3)
    assert out.strided is True
    assert out.stride == 2
    assert out.timesteps == [0, 2, 4]
    assert len(out.timesteps) < LENGTHS[0]
    assert "SAMPLED EVERY 2 FRAMES" in out.means()
    assert "fall entirely between two points" in out.means()

    whole = vla_timeline.tracks(reader(tmp_path, "b"), 0, max_points=1000)
    assert whole.strided is False and whole.stride == 1
    assert "SAMPLED EVERY" not in whole.means()


def test_a_non_finite_value_is_counted_and_left_out_not_drawn_as_zero(tmp_path):
    """A NaN in a recorded action is corruption, not a reading of zero. It is
    `None` in the series, counted per dimension, and named in the sentence."""
    total = sum(LENGTHS)
    rows = {
        "action": [
            [float("nan"), float(i)] if i == 2 else [float(i), float(i)]
            for i in range(total)
        ]
    }
    out = vla_timeline.tracks(reader(tmp_path, rows=rows), 0)
    action = next(t for t in out.tracks if t.column == "action")
    assert action.series[0][2] is None, "a NaN was drawn as a value"
    assert action.n_nonfinite == [1, 0]
    # And it is excluded from the axis rather than dragging it to zero.
    assert action.low[0] == 0.0 and action.high[0] == 5.0
    assert "non-finite value(s) were found and left out" in out.means()
    assert "not a zero" in out.means()


def test_a_dimension_name_is_the_datasets_own_or_nothing(tmp_path):
    """ "dim 2" looks exactly like a published name once rendered, and a reader
    who believes the dataset named its third joint that has been told
    something nobody wrote."""
    named = vla_timeline.tracks(reader(tmp_path, "named"), 0)
    action = next(t for t in named.tracks if t.column == "action")
    assert action.names == ["motor_a", "motor_b"]

    plain = vla_timeline.tracks(reader(tmp_path, "plain", names=False), 0)
    action = next(t for t in plain.tracks if t.column == "action")
    assert action.names is None, "a name was invented for an unnamed dimension"


def test_a_unit_is_the_datasets_own_or_null_and_the_sentence_says_which(tmp_path):
    """Most LeRobot datasets publish no units at all. Two tracks with no units
    cannot be compared to each other, and the sentence says so rather than
    leaving a reader to assume one axis explains another."""
    # Built WITH the reward column, because the point is comparing a track
    # that has a published unit against one that does not.
    r = reader(tmp_path, columns=("action", "observation.state", "next.reward"))
    out = vla_timeline.tracks(r, 0)
    reward = next(t for t in out.tracks if t.column == "next.reward")
    action = next(t for t in out.tracks if t.column == "action")
    assert reward.unit == "points"
    assert action.unit is None
    assert "publishes no unit for action" in out.means()
    assert "cannot be compared to each other" in out.means()


def test_no_verdict_is_attached_to_two_tracks_moving_together(tmp_path):
    """This aligns series. Whether two of them moving together means anything
    is the reader's call, and a correlation here would be a number nobody
    asked for attached to a claim nobody made."""
    said = vla_timeline.tracks(reader(tmp_path), 0).means().lower()
    for verdict in ("correlat", "caused", "because", "explains"):
        assert verdict not in said


# --------------------------------------------------------------- refusals


def test_an_episode_outside_the_dataset_is_refused_by_count(tmp_path):
    r = reader(tmp_path)
    with pytest.raises(BadRequest, match="is not in this dataset"):
        vla_timeline.tracks(r, 99)
    with pytest.raises(BadRequest, match="at least 0"):
        vla_timeline.tracks(r, -1)


def test_a_bool_is_not_an_index_or_a_budget(tmp_path):
    """`isinstance(True, int)` is True, so `episode=True` would have drawn
    episode 1 and `max_points=True` would have drawn one point."""
    r = reader(tmp_path)
    with pytest.raises(BadRequest, match="whole number"):
        vla_timeline.tracks(r, True)
    with pytest.raises(BadRequest, match="whole number"):
        vla_timeline.tracks(r, 0, max_points=True)


def test_a_dataset_with_none_of_the_columns_is_refused_with_what_it_has(tmp_path):
    """An empty chart is not an answer. The refusal points at the route that
    lists what the dataset does carry."""
    r = reader(tmp_path)
    with pytest.raises(Refusal, match="nothing to align"):
        vla_timeline.tracks(r, 0, columns=("no.such.column",))


def test_every_published_number_is_finite_or_absent(tmp_path):
    """The invariant a chart depends on: anything that is not `None` can be
    drawn. One NaN reaching a series is a chart with a hole that renders as a
    spike to zero in most libraries."""
    out = vla_timeline.tracks(reader(tmp_path), 0)
    for track in out.tracks:
        for dimension in track.series:
            for value in dimension:
                assert value is None or math.isfinite(value), track.column
        for bound in (*track.low, *track.high):
            assert bound is None or math.isfinite(bound), track.column
