"""An episode is a span inside a file, and the file depends on the camera.

The robot panel showed episode 0's video for every episode. `episodes()` read
`video_from_timestamp`, which is not a column any LeRobot v3.0 dataset has —
the real name is `videos/<camera>/from_timestamp` — and `.get(name, 0.0)`
turned the miss into a timestamp of zero for all of them. Measured on
lerobot/pusht before the fix: episodes 0, 5 and 20 returned byte-identical
images, while the state vector printed underneath each one was correctly that
episode's. The picture and the numbers disagreed and nothing said so.

Three separate assumptions produced that, and all three are tested here:

  the timestamp   `video_from_timestamp` vs `videos/<cam>/from_timestamp`
  the file        `sorted(rglob("*.mp4"))[0]`, ignoring camera and chunk
  the row         summing earlier episodes' lengths instead of reading
                  `dataset_from_index`

The fixtures are synthetic and carry no video, because the arithmetic is what
went wrong: routing is decided entirely from the metadata parquet, and a
dataset with two cameras and two chunks is exactly the shape nobody had here
to try it on. The one test that needs a real decode is skipped unless
lerobot/pusht happens to be cached.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa
import pyarrow.parquet as pq

from modelmri.errors import BadRequest, Refusal
from modelmri.vla_data import LeRobotV3Reader

CAMS = ["observation.images.top", "observation.images.wrist"]
LENGTHS = [10, 7, 13]


def build(root: Path, cameras=CAMS, *, routing=True, chunks=(0, 1, 1)) -> Path:
    """A LeRobot v3.0 snapshot with two cameras and two video chunks."""
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 10,
                "features": {
                    **{c: {"dtype": "video", "shape": [96, 96, 3]} for c in cameras},
                    "observation.state": {"dtype": "float32", "shape": [2]},
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
        for ci, cam in enumerate(cameras):
            # Each camera keeps its own clock: the wrist file is offset, so a
            # reader that grabs the first mp4 it finds gets a timestamp that
            # belongs to a different view.
            base = 100.0 * ci
            starts, at = [], base
            for i in range(n):
                starts.append(at)
                at += LENGTHS[i] / 10.0
            cols[f"videos/{cam}/from_timestamp"] = starts
            cols[f"videos/{cam}/to_timestamp"] = [
                s + LENGTHS[i] / 10.0 for i, s in enumerate(starts)
            ]
            cols[f"videos/{cam}/chunk_index"] = list(chunks[:n])
            cols[f"videos/{cam}/file_index"] = [0] * n
    else:
        # The pre-fix world: the column the reader looked for, spelled the way
        # it looked for it, and therefore absent under its real name.
        cols["video_from_timestamp"] = [0.0] * n
        cols["video_to_timestamp"] = [1.0] * n

    pq.write_table(
        pa.table(cols), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    pq.write_table(
        pa.table(
            {
                "observation.state": [[float(i), 0.0] for i in range(sum(LENGTHS))],
                "action": [[float(i), 1.0] for i in range(sum(LENGTHS))],
            }
        ),
        root / "data" / "chunk-000" / "file-000.parquet",
    )

    for cam in cameras:
        for c in sorted(set(chunks[:n])):
            d = root / "videos" / cam / f"chunk-{c:03d}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "file-000.mp4").write_bytes(b"")
    return root


def reader(tmp_path: Path, **kw) -> LeRobotV3Reader:
    return LeRobotV3Reader(build(tmp_path / "snap", **kw), "test/two-cams")


# --------------------------------------------------------------- timestamps


def test_each_episode_starts_at_its_own_timestamp(tmp_path):
    """The bug, stated directly: three episodes, three different starts."""
    eps = reader(tmp_path).episodes()
    assert [e.from_ts for e in eps] == [0.0, 1.0, 1.7]
    assert len({e.from_ts for e in eps}) == 3, (
        "every episode decodes from the same point in the video again"
    )


def test_the_span_matches_the_episode_length(tmp_path):
    r = reader(tmp_path)
    for e in r.episodes():
        assert abs((e.to_ts - e.from_ts) - e.length / r.fps) < 1e-6


def test_a_dataset_without_routing_is_refused_not_defaulted(tmp_path):
    """`.get(name, 0.0)` is what made this silent. A missing routing column
    means frames cannot be located, and saying so is the whole fix."""
    r = reader(tmp_path, routing=False)
    with pytest.raises(Refusal) as err:
        r.episodes()
    assert "from_timestamp" in str(err.value)


# ------------------------------------------------------------------ cameras


def test_every_camera_is_listed_not_just_the_first(tmp_path):
    r = reader(tmp_path)
    assert r.cameras == CAMS
    assert r.summary()["cameras"] == CAMS


def test_switching_camera_rereads_the_routing(tmp_path):
    """Routing is stored per camera, so the timestamps must change with it —
    a cached episode list would keep the first camera's clock."""
    r = reader(tmp_path)
    first = [e.from_ts for e in r.episodes()]
    r.use_camera(CAMS[1])
    second = [e.from_ts for e in r.episodes()]
    assert r.camera == CAMS[1]
    assert first == [0.0, 1.0, 1.7]
    assert second == [100.0, 101.0, 101.7]


def test_an_unknown_camera_says_which_ones_exist(tmp_path):
    r = reader(tmp_path)
    with pytest.raises(BadRequest) as err:
        r.use_camera("observation.images.elbow")
    assert "observation.images.top" in str(err.value)


def test_naming_the_current_camera_again_is_a_no_op(tmp_path):
    r = reader(tmp_path)
    r.use_camera(CAMS[0])
    r.use_camera(None)
    assert r.camera == CAMS[0]


# --------------------------------------------------------------- video file


def test_the_file_follows_the_camera(tmp_path):
    """`sorted(rglob("*.mp4"))[0]` returned the same path for both views, so
    the panel could show the wrist camera while labelling it the overhead."""
    r = reader(tmp_path)
    ep = r.episodes()[0]
    top = r._video_file(ep)
    r.use_camera(CAMS[1])
    wrist = r._video_file(r.episodes()[0])
    assert top != wrist
    assert CAMS[0] in str(top) and CAMS[1] in str(wrist)


def test_the_file_follows_the_chunk(tmp_path):
    """Episodes 1 and 2 live in chunk-001. Reading chunk-000 for them decodes
    a frame from somewhere else entirely."""
    r = reader(tmp_path)
    eps = r.episodes()
    assert "chunk-000" in str(r._video_file(eps[0]))
    assert "chunk-001" in str(r._video_file(eps[1]))
    assert "chunk-001" in str(r._video_file(eps[2]))


# ------------------------------------------------------------------- rows


def test_the_row_offset_comes_from_the_dataset(tmp_path):
    r = reader(tmp_path)
    assert [e.data_from for e in r.episodes()] == [0, 10, 17]


def test_frames_read_their_own_episodes_rows(tmp_path):
    """State is row-indexed, so a wrong offset returns another episode's
    numbers under this episode's name."""
    r = reader(tmp_path)
    # frame() itself needs a decodable video; the row lookup is the part that
    # went wrong and it can be checked on its own.
    rows = r._frame_table()
    for e in r.episodes():
        assert rows["observation.state"][e.data_from][0] == float(e.data_from)


# ------------------------------------------------------- against the real one


@pytest.mark.skipif(
    not Path(__import__("modelmri.paths", fromlist=["x"]).hf_home())
    .joinpath("hub")
    .exists(),
    reason="no HF cache on this machine",
)
def test_a_real_dataset_gives_distinct_episodes_if_one_is_cached():
    """The measurement that started this. Skipped where pusht is not cached —
    the synthetic tests above carry the regression."""
    try:
        r = LeRobotV3Reader.discover()
    except Exception:
        pytest.skip("no LeRobot dataset cached")
    eps = r.episodes()
    if len(eps) < 21:
        pytest.skip("dataset too short to compare distant episodes")
    starts = {e.from_ts for e in eps}
    assert len(starts) > 1, "every episode still starts at the same timestamp"
