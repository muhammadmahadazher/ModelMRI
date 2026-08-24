"""Dataset-wide action statistics, and the four ways they can lie.

`LeRobotV3Reader.action_stats()` has the matching name and computes nothing —
it parses the mean/std/min/max the PUBLISHER wrote into meta/stats.json. So
the first thing tested here is that the two are different facts and that both
travel: a publisher's normalisation constants disagreeing with the actions
recorded beside them breaks nothing visibly (training normalises with them and
the policy simply sees a distribution nobody intended), and this is the one
place both halves are on screen together.

The other three:

  memory      `_frame_table()` builds a dict of every row, which is right for
              PushT's 1.4 MB and impossible for a 26-million-row dataset.
              `test_the_whole_frame_table_is_never_materialised` asserts the
              streaming path never touches it, and the tracemalloc test below
              measures the difference rather than asserting it in prose.
  NaN         one NaN folded into a mean makes the mean, the std, the min and
              the max all NaN, and four NaNs read as a dimension that was
              never recorded — a much less alarming finding than a hole in one
              that was.
  caps        a capped answer that cannot say what it capped is
              indistinguishable from a complete one.

The fixtures are synthetic and carry no video, following
tests/test_vla_routing.py: none of this decodes a frame, and the arithmetic is
what is being checked. The builder is a copy of that file's rather than an
import, because these tests need control of the action column itself — a
constant dimension, a NaN, a short row — and that file's fixture is shaped for
routing.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa
import pyarrow.parquet as pq

from modelmri.errors import BadRequest, Refusal
from modelmri.vla_data import LeRobotV3Reader

CAM = "observation.images.top"
LENGTHS = [10, 7, 13]  # 30 rows over three episodes
WIDTH = 4
# Where the second data shard starts. Deliberately mid-episode: a reader that
# streams one shard and stops is wrong in a way that still looks like a whole
# dataset.
SHARD_SPLIT = 18

NAMES = ["x", "gripper", "flip", "wrist"]


def action_rows() -> list[list[float]]:
    """The recorded actions, with one deliberate hole.

    dim 0  a clean ramp, so min/max/mean are checkable by eye
    dim 1  never moves — a held gripper is real data, not a degenerate case
    dim 2  alternating, with a single NaN at row 5
    dim 3  a sawtooth whose published std will be wrong by 10x
    """
    rows = []
    for i in range(sum(LENGTHS)):
        rows.append(
            [
                float(i),
                1.5,
                float("nan") if i == 5 else (-1.0 if i % 2 else 1.0),
                float(i % 7) * 0.25,
            ]
        )
    return rows


def finite(dim: int, rows: list[list[float]] | None = None) -> list[float]:
    """Every value of one dimension that a statistic may legitimately use."""
    rows = rows if rows is not None else action_rows()
    return [r[dim] for r in rows if math.isfinite(r[dim])]


def published_stats(*, std_scale: float = 1.0) -> dict:
    """meta/stats.json, honest except where a test asks for a lie.

    `std_scale` multiplies dimension 3's published standard deviation. That is
    the corruption this whole file exists to make visible: it is arithmetically
    invisible — no shape changes, no column is missing, every file still
    loads — and it is what training normalises with.
    """
    rows = action_rows()
    means, stds, mins, maxs = [], [], [], []
    for d in range(WIDTH):
        values = finite(d, rows)
        means.append(statistics.fmean(values))
        stds.append(statistics.pstdev(values))
        mins.append(min(values))
        maxs.append(max(values))
    stds[3] *= std_scale
    return {"action": {"mean": means, "std": stds, "min": mins, "max": maxs}}


def build(
    root: Path,
    *,
    rows: list[list[float]] | None = None,
    stats: dict | None = None,
    routing: bool = True,
    units=None,
    names: list[str] | None = NAMES,
    declared_width: int = WIDTH,
    with_action: bool = True,
    shards: bool = True,
) -> Path:
    """A LeRobot v3.0 snapshot whose action column is the interesting part."""
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

    action_feature: dict = {"dtype": "float32", "shape": [declared_width]}
    if names is not None:
        action_feature["names"] = list(names)
    if units is not None:
        action_feature["unit"] = units
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 10,
                "features": {
                    CAM: {"dtype": "video", "shape": [96, 96, 3]},
                    "observation.state": {"dtype": "float32", "shape": [2]},
                    "action": action_feature,
                },
            }
        ),
        encoding="utf-8",
    )

    n = len(LENGTHS)
    cols: dict[str, list] = {
        "episode_index": list(range(n)),
        "length": list(LENGTHS),
        "tasks": [["push the T"] for _ in range(n)],
        "dataset_from_index": [sum(LENGTHS[:i]) for i in range(n)],
    }
    if routing:
        starts, at = [], 0.0
        for i in range(n):
            starts.append(at)
            at += LENGTHS[i] / 10.0
        cols[f"videos/{CAM}/from_timestamp"] = starts
        cols[f"videos/{CAM}/to_timestamp"] = [
            s + LENGTHS[i] / 10.0 for i, s in enumerate(starts)
        ]
        cols[f"videos/{CAM}/chunk_index"] = [0] * n
        cols[f"videos/{CAM}/file_index"] = [0] * n
    pq.write_table(
        pa.table(cols), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )

    rows = action_rows() if rows is None else rows
    state = [[float(i), 0.0] for i in range(len(rows))]
    spans = [(0, SHARD_SPLIT), (SHARD_SPLIT, len(rows))] if shards else [(0, len(rows))]
    for shard, (lo, hi) in enumerate(spans):
        table: dict = {"observation.state": state[lo:hi]}
        if with_action:
            table["action"] = rows[lo:hi]
        schema = pa.schema(
            [
                ("observation.state", pa.list_(pa.float64())),
                *([("action", pa.list_(pa.float64()))] if with_action else []),
            ]
        )
        pq.write_table(
            pa.table(table, schema=schema),
            root / "data" / "chunk-000" / f"file-{shard:03d}.parquet",
        )

    if stats is not None:
        (root / "meta" / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    return root


def reader(tmp_path: Path, sub: str = "snap", **kw) -> LeRobotV3Reader:
    """One reader over a freshly built snapshot.

    `sub` exists because several tests below build TWO datasets to compare a
    carried label against a dropped one, and a second `build()` into the same
    directory fails on Windows rather than overwriting.
    """
    return LeRobotV3Reader(build(tmp_path / sub, **kw), "test/actions")


# ------------------------------------------------------ the measured numbers


def test_every_dimension_is_summarised_over_every_shard(tmp_path):
    """The 30 rows live in two parquet files split mid-episode. A reader that
    streams the first shard and stops returns a prefix of the dataset that
    looks exactly like a whole one — the same trap `_read_all` was written
    for, one layer down."""
    out = reader(tmp_path).dataset_action_stats()

    assert out["rows_total"] == sum(LENGTHS)
    assert out["rows_read"] == sum(LENGTHS)
    assert out["files_read"] == 2
    assert out["dimensions"] == WIDTH
    assert len(out["per_dimension"]) == WIDTH

    ramp = out["per_dimension"][0]
    assert ramp["count"] == 30
    assert ramp["min"] == 0.0
    assert ramp["max"] == 29.0
    assert ramp["mean"] == pytest.approx(statistics.fmean(finite(0)))
    assert ramp["std"] == pytest.approx(statistics.pstdev(finite(0)))


def test_the_mean_matches_an_independent_computation_per_dimension(tmp_path):
    """Welford's update is not the obvious formula, and the obvious formula is
    the one that would have been written. Checked against `statistics.fmean`
    over the same values so a transcription slip in the incremental update
    cannot pass."""
    out = reader(tmp_path).dataset_action_stats()
    for d, dim in enumerate(out["per_dimension"]):
        values = finite(d)
        assert dim["mean"] == pytest.approx(statistics.fmean(values), abs=1e-12)
        assert dim["min"] == min(values)
        assert dim["max"] == max(values)


def test_the_episode_count_travels_with_the_row_count(tmp_path):
    """ "30 rows" says nothing about whether that is one long recording or
    thirty one-frame ones, and the sentence promises both."""
    out = reader(tmp_path).dataset_action_stats()
    assert out["n_episodes"] == len(LENGTHS)
    assert out["episodes_reason"] == ""
    assert "3 episode(s)" in out["means"]


def test_action_statistics_survive_a_dataset_whose_video_routing_is_missing(tmp_path):
    """`episodes()` refuses a snapshot with no `videos/<cam>/from_timestamp`
    column, and is right to — a frame that cannot be located must not be
    guessed at. The ACTIONS are in the data shards and are perfectly
    measurable, so counting episodes from parquet footers rather than through
    the routing builder is what keeps a broken-video dataset measurable at
    all."""
    r = reader(tmp_path, routing=False)
    with pytest.raises(Refusal):
        r.episodes()  # the routing really is gone
    out = r.dataset_action_stats()
    assert out["n_episodes"] == len(LENGTHS)
    assert out["rows_read"] == sum(LENGTHS)


# ------------------------------------------------------------- a still dimension


def test_a_constant_dimension_is_reported_not_skipped(tmp_path):
    """A gripper held open for a whole recording is real data. Its spread of
    zero is a MEASUREMENT — the one number in this payload that is allowed to
    be 0.0 without being a missing one — and its histogram is one bar holding
    every row rather than 32 empty ones around a single value."""
    out = reader(tmp_path).dataset_action_stats()
    dim = out["per_dimension"][1]
    assert "Dimension(s) [1] never varied" in out["means"]
    assert dim["constant"] is True
    assert dim["min"] == dim["max"] == 1.5
    assert dim["std"] == 0.0
    assert dim["histogram"]["counts"] == [30]
    assert dim["histogram"]["bin_edges"] == [1.5, 1.5]
    assert [p["value"] for p in dim["percentiles"]] == [1.5] * len(dim["percentiles"])
    # Exact, and said so: there is no interpolation error inside a bin that
    # holds one value.
    assert dim["percentile_resolution"] == 0.0


def test_a_dimension_that_moves_is_not_called_constant(tmp_path):
    """The other half of the same claim: `constant` has to be able to be
    False, or the field is decoration."""
    out = reader(tmp_path).dataset_action_stats()
    assert out["per_dimension"][0]["constant"] is False
    assert out["per_dimension"][3]["constant"] is False


# ------------------------------------------------------------------- NaN / inf


def test_a_nan_is_counted_and_excluded_rather_than_poisoning_the_mean(tmp_path):
    """One NaN folded into a running mean makes the mean, the std, the min and
    the max all NaN. Four NaNs on a chart read as a dimension that was never
    recorded, which is a much less alarming finding than a hole in one that
    was — so the value is excluded and the exclusion is counted."""
    out = reader(tmp_path).dataset_action_stats()
    dim = out["per_dimension"][2]
    assert dim["n_nan"] == 1
    assert dim["count"] == 29, "the NaN row must not be counted as a measurement"
    assert math.isfinite(dim["mean"]) and math.isfinite(dim["std"])
    assert dim["mean"] == pytest.approx(statistics.fmean(finite(2)))
    assert out["nonfinite_total"] == 1
    assert "NaN or infinite" in out["means"]
    # And the dimensions beside it are untouched: this is one dimension's hole,
    # not the row's.
    assert out["per_dimension"][0]["count"] == 30


def test_an_infinity_is_counted_separately_from_a_nan(tmp_path):
    """They arrive from different accidents — a divide by zero versus an
    uninitialised buffer — and collapsing them into one counter loses which
    one happened."""
    rows = action_rows()
    rows[3][0] = float("inf")
    rows[4][0] = float("-inf")
    out = reader(tmp_path, rows=rows).dataset_action_stats()
    dim = out["per_dimension"][0]
    assert (dim["n_inf"], dim["n_nan"]) == (2, 0)
    assert dim["count"] == 28
    assert dim["max"] == 29.0, "an infinity must not become the maximum"


def test_a_dimension_that_is_entirely_nan_is_unknown_not_zero(tmp_path):
    """`None`, never 0.0. "every value was NaN" and "every value was zero" are
    opposite findings and a zero is the one somebody would act on."""
    rows = action_rows()
    for row in rows:
        row[0] = float("nan")
    dim = reader(tmp_path, rows=rows).dataset_action_stats()["per_dimension"][0]
    assert dim["count"] == 0
    assert dim["n_nan"] == 30
    assert dim["mean"] is None and dim["std"] is None
    assert dim["min"] is None and dim["max"] is None
    assert dim["constant"] is None, "nothing was compared, so nothing is known"
    assert [p["value"] for p in dim["percentiles"]] == [None] * 7
    assert dim["percentile_resolution"] is None


# ------------------------------------------------- published against measured


def test_the_published_std_and_the_measured_one_both_travel(tmp_path):
    """The finding this module exists for. `action_stats()` parses what the
    publisher wrote; this measures the rows. A 10x disagreement on dimension 3
    breaks nothing visibly — every file still loads — and training normalises
    with the published half."""
    out = reader(tmp_path, stats=published_stats(std_scale=10.0)).dataset_action_stats()
    dim = out["per_dimension"][3]
    measured = statistics.pstdev(finite(3))

    assert dim["std"] == pytest.approx(measured)
    assert dim["published"]["std"] == pytest.approx(measured * 10.0)
    assert dim["measured_minus_published"]["std"] == pytest.approx(-9.0 * measured)
    # Neither was substituted for the other, and the honest dimensions agree.
    assert out["per_dimension"][0]["measured_minus_published"]["std"] == pytest.approx(
        0.0, abs=1e-9
    )
    assert out["published"]["std"][3] == pytest.approx(measured * 10.0)


def test_the_largest_gap_points_at_the_dimension_and_the_statistic(tmp_path):
    """Forty dimensions times four statistics is a table nobody scans. This
    says where to look first — and says in the payload that it is a pointer
    rather than a ranking, because absolute differences only compare inside
    one dimension."""
    out = reader(tmp_path, stats=published_stats(std_scale=10.0)).dataset_action_stats()
    gap = out["largest_published_gap"]
    assert gap["dimension"] == 3
    assert gap["name"] == "wrist"
    assert gap["statistic"] == "std"
    assert gap["difference"] == pytest.approx(-9.0 * statistics.pstdev(finite(3)))
    assert "not what is worst" in gap["caveat"]
    assert "`std` on dimension 3 (wrist)" in out["means"]


def test_there_is_no_largest_gap_when_nothing_was_published(tmp_path):
    """`None`, not a zero-difference row: "nothing to compare" and "they agree
    exactly" are opposite findings."""
    assert reader(tmp_path).dataset_action_stats()["largest_published_gap"] is None


def test_a_dataset_with_no_published_stats_says_empty_is_not_agreement(tmp_path):
    """Empty is load-bearing elsewhere in this project — `vla_actions` reads
    it as "do not overlay" rather than as identity scaling — so it must not
    read here as "the publisher agrees"."""
    out = reader(tmp_path).dataset_action_stats()
    assert out["published"] == {}
    assert out["published_dimensions"] is None
    for dim in out["per_dimension"]:
        assert dim["published"] == {"mean": None, "std": None, "min": None, "max": None}
        assert dim["measured_minus_published"]["mean"] is None
    assert "Empty is not agreement" in out["means"]


def test_published_stats_of_the_wrong_width_are_not_paired(tmp_path):
    """Pairing dimension 3 with dimension 3 across two different action spaces
    compares unrelated joints, which is the mistake `vla_actions.units_agree`
    refuses for policies. Same rule, same reason."""
    stats = published_stats()
    stats["action"] = {k: v[:2] for k, v in stats["action"].items()}
    out = reader(tmp_path, stats=stats).dataset_action_stats()
    assert out["published_dimensions"] == 2
    assert out["dimensions"] == WIDTH
    for dim in out["per_dimension"]:
        assert dim["published"]["mean"] is None
        assert dim["measured_minus_published"]["mean"] is None
    assert "NOT paired" in out["means"]


def test_one_published_statistic_of_the_wrong_width_does_not_drop_the_others(tmp_path):
    """All-or-nothing would hide the half that was fine."""
    stats = published_stats()
    stats["action"]["min"] = stats["action"]["min"][:2]
    dim = reader(tmp_path, stats=stats).dataset_action_stats()["per_dimension"][0]
    assert dim["published"]["min"] is None
    assert dim["published"]["mean"] == pytest.approx(statistics.fmean(finite(0)))


# ------------------------------------------------------------------- caps


def test_a_row_cap_reports_the_true_row_count_beside_it(tmp_path):
    """A capped answer that cannot say what it capped is indistinguishable
    from a complete one — and here the difference is a statistic over 12 rows
    presented as a statistic over a dataset."""
    out = reader(tmp_path).dataset_action_stats(max_rows=12)
    assert out["rows_read"] == 12
    assert out["rows_total"] == 30
    assert out["rows_with_action_column"] == 30
    assert out["rows_skipped"] == 18
    assert out["max_rows"] == 12
    assert out["per_dimension"][0]["max"] == 11.0, "row 12 onward was not read"
    assert "CAP OF 12 ROWS" in out["means"]
    assert "18 row(s) went unread" in out["means"]


def test_a_cap_larger_than_the_dataset_says_everything_was_read(tmp_path):
    """The other arm, because "capped" and "capped at more than exists" are
    different states and only one of them shortens the answer."""
    out = reader(tmp_path).dataset_action_stats(max_rows=10_000)
    assert out["rows_read"] == 30
    assert out["rows_skipped"] == 0
    assert "every row was read" in out["means"]


def test_the_cap_lands_mid_batch_and_both_passes_take_the_same_rows(tmp_path):
    """The histogram is a second pass. If it stopped at a different row from
    the first, the counts would describe a different set of rows from the mean
    printed beside them, and the sum would silently disagree with `count`."""
    out = reader(tmp_path).dataset_action_stats(max_rows=7)
    for dim in out["per_dimension"]:
        assert sum(dim["histogram"]["counts"]) == dim["count"]
    assert out["per_dimension"][0]["count"] == 7


def test_a_dimension_cap_is_reported_with_the_true_width(tmp_path):
    out = reader(tmp_path).dataset_action_stats(max_dimensions=2)
    assert out["dimensions"] == WIDTH
    assert out["dimensions_reported"] == 2
    assert len(out["per_dimension"]) == 2
    assert "ONLY THE FIRST 2 of 4" in out["means"]


def test_the_bin_count_is_reported_and_bounded(tmp_path):
    """Every bin travels in the response, once per dimension, so this is a
    payload limit rather than a statistical one — and it says so."""
    r = reader(tmp_path)
    out = r.dataset_action_stats(bins=8)
    assert out["bins"] == 8
    assert len(out["per_dimension"][0]["histogram"]["counts"]) == 8
    with pytest.raises(BadRequest) as err:
        r.dataset_action_stats(bins=0)
    assert "between 1 and 512" in str(err.value)
    with pytest.raises(BadRequest):
        r.dataset_action_stats(bins=100_000)


def test_a_cap_of_zero_rows_is_refused_rather_than_answered(tmp_path):
    """Zero rows is not an empty dataset, it is a request for no measurement,
    and answering it with a payload full of `None` would look like a finding
    about the data."""
    with pytest.raises(BadRequest) as err:
        reader(tmp_path).dataset_action_stats(max_rows=0)
    assert "at least 1" in str(err.value)


def test_percentiles_outside_zero_to_a_hundred_are_refused(tmp_path):
    with pytest.raises(BadRequest) as err:
        reader(tmp_path).dataset_action_stats(percentiles=(50.0, 101.0))
    assert "between 0 and 100" in str(err.value)


# ------------------------------------------------------- histogram and quantiles


def test_the_histogram_counts_every_finite_value_exactly_once(tmp_path):
    """The counts are exact even though the percentiles read off them are
    not — including the maximum, which lands on the top edge and would fall
    one past the last bin without a clamp. That row is exactly the one
    somebody looking for a joint limit came for."""
    out = reader(tmp_path).dataset_action_stats(bins=16)
    for d, dim in enumerate(out["per_dimension"]):
        assert sum(dim["histogram"]["counts"]) == dim["count"] == len(finite(d))
    ramp = out["per_dimension"][0]["histogram"]
    assert ramp["counts"][-1] >= 1, "the maximum fell off the end of the histogram"
    assert ramp["bin_edges"][0] == 0.0
    assert ramp["bin_edges"][-1] == 29.0, "the top edge is the measured maximum"


def test_the_percentile_resolution_is_the_bin_width_and_bounds_the_error(tmp_path):
    """These are estimates and the payload has to say by how much. The median
    of a 30-row ramp is known exactly, so the promise is checkable."""
    out = reader(tmp_path).dataset_action_stats(bins=32)
    dim = out["per_dimension"][0]
    width = (dim["max"] - dim["min"]) / 32
    assert dim["percentile_resolution"] == pytest.approx(width)
    median = next(p["value"] for p in dim["percentiles"] if p["q"] == 50.0)
    assert abs(median - statistics.median(finite(0))) <= width
    assert "percentile_resolution" in out["percentile_method"]
    assert "histogram COUNTS are exact" in out["means"]


def test_percentiles_never_leave_the_measured_range(tmp_path):
    """Interpolating inside a bin can otherwise produce a value larger than
    anything recorded, which is a fabricated measurement in the field whose
    job is describing the tail."""
    out = reader(tmp_path).dataset_action_stats(bins=4)
    for dim in out["per_dimension"]:
        if dim["count"] == 0:
            continue
        for point in dim["percentiles"]:
            assert dim["min"] <= point["value"] <= dim["max"]


# ------------------------------------------------------------------- units


def test_a_dataset_that_states_no_units_says_none_rather_than_guessing(tmp_path):
    """A number with no unit is not a measurement. LeRobot's feature schema
    has no unit field at all, so this is the common case — and inventing
    "radians" because most arms use them would invent it for every dataset
    that never said."""
    out = reader(tmp_path).dataset_action_stats()
    assert out["units"] == {"published": None, "source": None}
    assert all(dim["unit"] is None for dim in out["per_dimension"])
    assert "states no units" in out["means"]


def test_a_stated_unit_is_carried_onto_every_dimension(tmp_path):
    out = reader(tmp_path, units="rad").dataset_action_stats()
    assert out["units"]["published"] == "rad"
    assert "features.action.unit" in out["units"]["source"]
    assert [d["unit"] for d in out["per_dimension"]] == ["rad"] * WIDTH


def test_per_dimension_units_are_carried_and_a_mismatched_list_is_dropped(tmp_path):
    """Expanding a list of the wrong length would be this module inventing the
    label, which is the same rule that drops mismatched joint names."""
    good = reader(tmp_path, "a", units=["m", "m", "rad", "rad"])
    assert [d["unit"] for d in good.dataset_action_stats()["per_dimension"]] == [
        "m",
        "m",
        "rad",
        "rad",
    ]
    bad = reader(tmp_path, "b", units=["m", "rad"]).dataset_action_stats()
    assert [d["unit"] for d in bad["per_dimension"]] == [None] * WIDTH


def test_joint_names_are_carried_and_a_mismatched_list_is_dropped(tmp_path):
    named = reader(tmp_path, "a").dataset_action_stats()
    assert [d["name"] for d in named["per_dimension"]] == NAMES
    short = reader(tmp_path, "b", names=["x", "y"]).dataset_action_stats()
    assert short["action_names"] == []
    assert [d["name"] for d in short["per_dimension"]] == [None] * WIDTH


def test_a_declared_width_that_disagrees_with_the_rows_is_named(tmp_path):
    """One of the two is wrong and nothing here can tell which, so both are
    printed and the rows are what was measured."""
    out = reader(tmp_path, declared_width=7).dataset_action_stats()
    assert out["dimensions_declared"] == 7
    assert out["dimensions"] == WIDTH
    assert "declares 7 action dimension(s)" in out["means"]


# ---------------------------------------------------------------- malformed


def test_a_row_of_the_wrong_width_is_excluded_and_counted(tmp_path):
    """A short row is not a narrower action, it is a row nobody can pair with
    the others by position. Folding its first two values into dimensions 0 and
    1 would be an invented alignment."""
    rows = action_rows()
    rows[2] = [9.0, 9.0]
    out = reader(tmp_path, rows=rows).dataset_action_stats()
    assert out["rows_malformed"] == 1
    assert out["rows_read"] == 30, "it was read, then excluded — both are true"
    assert out["per_dimension"][0]["count"] == 29
    assert out["per_dimension"][0]["n_missing"] == 1
    assert out["per_dimension"][0]["max"] == 29.0
    assert "could not be read as an action vector" in out["means"]


# ---------------------------------------------------------------- refusals


def test_an_empty_dataset_is_refused_with_a_sentence_and_a_next_step(tmp_path):
    """There is no distribution over no actions. A payload of `None` would
    look like a finding about the data rather than an absence of data."""
    with pytest.raises(Refusal) as err:
        reader(tmp_path, rows=[], shards=False).dataset_action_stats()
    assert "zero rows" in err.value.sentence
    assert "empty recording" in err.value.sentence


def test_a_dataset_with_no_action_column_names_what_it_does_have(tmp_path):
    """ "Recorded actions are what this reads" and "this dataset is broken" are
    different sentences, and the columns that ARE there is the fact that tells
    the reader which one they are looking at."""
    with pytest.raises(Refusal) as err:
        reader(tmp_path, with_action=False).dataset_action_stats()
    assert "observation.state" in err.value.sentence
    assert "`action` column" in err.value.sentence


def test_a_snapshot_with_no_frame_data_at_all_is_refused(tmp_path):
    root = build(tmp_path / "snap")
    for parquet in (root / "data").rglob("*.parquet"):
        parquet.unlink()
    with pytest.raises(Refusal) as err:
        LeRobotV3Reader(root, "test/actions").dataset_action_stats()
    assert "data/chunk-*/file-*.parquet" in err.value.sentence


# ------------------------------------------------------------------- memory


def test_the_whole_frame_table_is_never_materialised(tmp_path):
    """`_frame_table()` concatenates every shard into one dict of every row.
    That is right for PushT's 1.4 MB and impossible for lerobot/droid's 26
    million rows, and this method is the one that has to survive the second
    case. `_frames` staying None is the direct evidence it did not take that
    path."""
    r = reader(tmp_path)
    r.dataset_action_stats()
    assert r._frames is None, "the streaming path fell back to the whole table"


def _peak(call) -> int:
    """Peak Python allocation during `call`, in bytes."""
    import tracemalloc

    tracemalloc.start()
    try:
        call()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def _ramp(n: int) -> list[list[float]]:
    return [[float(i), 1.5, float(i % 3), float(i % 7) * 0.25] for i in range(n)]


def test_the_peak_does_not_grow_with_the_row_count(tmp_path):
    """The claim in the module comment, measured rather than asserted in prose.

    The same fixture at 20,000 and at 80,000 rows: four times the data through
    a path whose peak is one batch plus a fixed set of accumulators, so the
    peak must barely move. Then the same 80,000 rows through `_frame_table()`,
    which builds a dict of every row — right for PushT's 1.4 MB and impossible
    for lerobot/droid's 26 million rows.

    MEASURED on this machine: 3.2 MB streaming at 20,000 rows, 3.2 MB at
    80,000 (and still 3.2 MB at 400,000, which is too slow to run here), against
    9.1 MB / 27.6 MB / 135.3 MB for the frame table over the same rows. The
    bounds below are deliberately loose — 1.5x for the growth, 4x for the
    ratio at 80,000 — so this fails on a regression to list-building rather
    than on a pyarrow release changing its batch allocation.
    """
    small = reader(tmp_path, "small", rows=_ramp(20_000))
    large = reader(tmp_path, "large", rows=_ramp(80_000))

    at_20k = _peak(small.dataset_action_stats)
    at_80k = _peak(large.dataset_action_stats)
    whole = _peak(large._frame_table)

    assert large.dataset_action_stats()["rows_read"] == 80_000
    assert at_80k < at_20k * 1.5, (
        f"four times the rows moved the peak from {at_20k / 1e6:.1f} MB to "
        f"{at_80k / 1e6:.1f} MB — something in the streaming path is keeping "
        f"rows"
    )
    assert at_80k * 4 < whole, (
        f"streaming peaked at {at_80k / 1e6:.1f} MB against the frame table's "
        f"{whole / 1e6:.1f} MB over the identical rows"
    )
