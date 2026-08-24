"""LeRobot v3.0 dataset reader — episodes, states, actions, and video frames.

Deliberately does NOT depend on the `lerobot` package: the v3.0 on-disk
layout is stable and reading it directly with pyarrow + pyav is both
faster and keeps ModelMRI installable next to any torch version (lerobot
pins torch<2.12 / numpy<2.3, which would downgrade the core runtime).

Layout (as cached by lerobot under $HF_HOME/lerobot/hub):
    datasets--<owner>--<name>/
      refs/<rev>                     -> snapshot hash  (PushT's ref is "v3.0", NOT "main")
      snapshots/<hash>/
        meta/info.json               fps, features, chunk layout
        meta/episodes/**.parquet     per-episode index + timestamps + task
        meta/stats.json              normalization statistics
        data/**/file-*.parquet       per-frame state/action rows
        videos/<key>/**/file-*.mp4   AV1-encoded frames
"""

from __future__ import annotations

import base64
import io
import json
import math
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import fmt
from .errors import BadRequest, Refusal

DEFAULT_DATASET = "lerobot/pusht"

# How many rows one streaming batch materialises in `dataset_action_stats`.
# A LeRobot dataset is millions of rows -- `lerobot/droid` is 26 million --
# and `_frame_table()` above deliberately loads the whole thing because PushT
# is 1.4 MB. Dataset-wide statistics cannot make that bet, so nothing there
# builds a list of every action: pyarrow hands back one batch, it is folded
# into per-dimension accumulators, and it is dropped before the next arrives.
#
# MEASURED with tracemalloc on synthetic 4-dimension snapshots, streaming
# against `_frame_table()` over the identical rows:
#
#      20,000 rows    3.2 MB  vs    9.1 MB
#      80,000 rows    3.2 MB  vs   27.6 MB
#     200,000 rows    3.2 MB  vs   67.8 MB
#     400,000 rows    3.2 MB  vs  135.3 MB
#
# The left column is flat and the right one is linear, which is the whole
# claim: the peak here is one batch plus the accumulators, so it is the same
# at 26 million rows as at 30. Time is linear on both — 2.35 s for 200,000
# rows and 12.25 s for 1,000,000 on this machine — which is what `max_rows`
# is for.
ACTION_ROW_BATCH = 8192

# Bins in a per-dimension action histogram, and the ceiling on what a caller
# may ask for. A resolution, not a measurement: it is reported in the payload,
# and the percentiles read off it carry the resulting bin width beside them so
# nobody has to infer their accuracy.
DEFAULT_ACTION_BINS = 32
MAX_ACTION_BINS = 512

# Which percentiles a dimension gets by default. The quartiles say where the
# bulk sits and the 1/99 pair says how far the tails reach -- a joint limit
# that is hit twice in a whole dataset is invisible in a mean and a standard
# deviation, and it is the number that decides whether a policy trained here
# will ever see the edge of its action space.
ACTION_PERCENTILES: tuple[float, ...] = (1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0)


@dataclass
class EpisodeInfo:
    index: int
    length: int
    task: str
    from_ts: float
    to_ts: float
    # Where this episode's frames actually live. LeRobot v3.0 concatenates
    # many episodes into one mp4 and one parquet, so an episode is a SPAN
    # inside a file, not a file -- and which file depends on the camera.
    video_chunk: int = 0
    video_file: int = 0
    data_from: int = 0


@dataclass
class FrameSample:
    episode: int
    t: int
    timestamp: float
    state: list[float]
    action: list[float]
    task: str
    image: str  # data:image/png;base64,...
    width: int
    height: int


def default_hf_home() -> Path:
    from . import paths

    return paths.hf_home()


def dataset_roots(hf_home: str | Path | None = None) -> list[Path]:
    """Every directory a cached LeRobot dataset can be sitting in.

    LeRobot keeps its own hub root; a dataset pulled with plain
    `huggingface-cli download --repo-type dataset` lands in the ordinary hub
    cache instead. Both are normal, so both are searched.

    This exists because the lister and the opener disagreed: `cached_datasets`
    looked in three places and `snapshot_path` in one, so the picker offered
    datasets that then failed to open with "not cached" — pointing at a
    directory the user could see was not where they had put it.
    """
    from . import paths

    root = Path(hf_home) if hf_home else default_hf_home()
    out = [root / "lerobot" / "hub", paths.hf_hub_cache()]
    if hf_home:
        out.insert(1, root / "hub")
    # The "not cached" refusal tells the reader to point HF_LEROBOT_HOME at the
    # cache that has it. Nothing read it. So a user who kept datasets on a
    # second drive did exactly what the message said, restarted, and got the
    # identical refusal listing the same directories — with the one they had
    # just configured still missing. The tool's own instructions were the dead
    # end. LEROBOT_HOME is the older spelling and costs nothing to honour.
    for var in ("HF_LEROBOT_HOME", "LEROBOT_HOME"):
        if env := paths._env_path(var):
            out.insert(0, env)
            out.insert(1, env / "hub")
    return [p for i, p in enumerate(out) if p not in out[:i]]


def _read_all(files: list) -> dict:
    """Concatenate every parquet shard into one column dict.

    `pq.read_table(files[0])` was the old shape of this and it is the kind of
    mistake that never raises: one shard reads perfectly, so a small dataset is
    correct and a large one is quietly truncated.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if len(files) == 1:
        return pq.read_table(files[0]).to_pydict()
    return pa.concat_tables([pq.read_table(f) for f in files]).to_pydict()


def snapshot_path(hf_home: str | Path | None, repo_id: str = DEFAULT_DATASET) -> Path:
    """Resolve the newest snapshot dir for a cached LeRobot dataset.

    THE ID IS CHECKED BEFORE IT IS SPLIT. `vla._snapshot` grew this guard and
    wrote the sentence for it; this copy never received one, so the same typo
    that gets a refusal there crashed here on the unpack below — `pusht` gave
    `ValueError: not enough values to unpack`, answered as HTTP 500 by
    `/api/vla/dataset` and as a raw traceback by `modelmri audit`. A `None`
    reaching it gave `AttributeError` instead, which is the same defect wearing
    a different exception.
    """
    # Imported here, the way every other `paths` use in this module is — the
    # file keeps it lazy so importing the reader costs nothing.
    from . import paths

    repo_id = paths.validate_repo_id(repo_id, kind="dataset")
    owner, name = repo_id.split("/", 1)
    roots = dataset_roots(hf_home)
    tried = [r / f"datasets--{owner}--{name}" for r in roots]
    base = next((b for b in tried if b.is_dir()), None)
    if base is None:
        where = "\n  ".join(str(t) for t in tried)
        # A Refusal, not a FileNotFoundError, and the six in this file changed
        # together. The distinction is not cosmetic: the /api/vla/* handlers
        # answered `except FileNotFoundError` with 409 and the exception's own
        # text, and that arm cannot tell this sentence apart from pyarrow
        # failing to open a parquet file or av failing to open a container —
        # so a library's errno message was published as though someone here
        # had written it for the reader. Measured: a reader raising
        # `FileNotFoundError(2, "No such file or directory", <abs path>)` came
        # back as 409 with that path in the body, on all four routes.
        #
        # The directories in this message are deliberate and are the answer
        # (see errors.py): they are the places that were searched, and the
        # reader is the person who has to put the dataset in one of them.
        raise Refusal(
            f"{repo_id} is not cached. Looked in:\n  {where}\nDownload it with "
            f"lerobot, or point HF_LEROBOT_HOME / HF_HUB_CACHE at the cache "
            f"that has it."
        )
    refs = base / "refs"
    # PushT's ref is literally "v3.0" — never assume "main" exists.
    candidates = sorted(refs.glob("*")) if refs.is_dir() else []
    if not candidates:
        raise Refusal(f"No snapshot ref under {refs}")
    digest = candidates[-1].read_text(encoding="utf-8").strip()
    snap = base / "snapshots" / digest
    if not snap.is_dir():
        raise Refusal(f"Snapshot {digest} missing under {base / 'snapshots'}")
    return snap


class LeRobotV3Reader:
    """Reads one cached LeRobot v3.0 dataset. Thread-safe per instance."""

    def __init__(self, snapshot: str | Path, repo_id: str = DEFAULT_DATASET) -> None:
        self.snapshot = Path(snapshot)
        self.repo_id = repo_id
        self._lock = threading.Lock()
        self._container = None  # lazily opened av container
        self._container_key: tuple[str, str] | None = None

        self.info: dict = json.loads(
            (self.snapshot / "meta" / "info.json").read_text(encoding="utf-8")
        )
        self.fps: int = int(self.info.get("fps", 10))
        self._episodes: list[EpisodeInfo] | None = None
        self._frames: dict | None = None
        # Every camera, not the first one. A single-arm SO-100 recording has
        # a wrist and an overhead view; an ALOHA recording has four. Keeping
        # only `next(...)` meant the other views were not merely unselectable,
        # they were invisible -- the panel reported one camera as though it
        # were the dataset.
        self._cameras = [
            k
            for k, v in self.info.get("features", {}).items()
            if v.get("dtype") in ("video", "image")
        ] or ["observation.image"]
        self._video_key = self._cameras[0]

    @property
    def cameras(self) -> list[str]:
        return list(self._cameras)

    @property
    def camera(self) -> str:
        return self._video_key

    def use_camera(self, name: str | None) -> None:
        """Choose which view the frames come from.

        UNDER THE LOCK, all of it. This mutates `_video_key` and `_episodes`
        and then frees the open PyAV container, while `frame()` and
        `raw_frame()` decode from that same container inside `self._lock`.
        Done outside it, a camera change landing between another request's
        `av.open` and its decode frees the container out from under a C
        extension that is mid-read.

        MEASURED on a two-camera LeRobot v3.0 snapshot: a switcher thread
        against a decode loop segfaults the process — exit 139, repeatable,
        and the same loop without the switcher survives 200 iterations. There
        is no catching it: `except Exception` never runs, and every other
        request in flight dies with the process. Reachable from the camera
        dropdown, whose effect cleanup does not abort the in-flight fetch.
        """
        if not name or name == self._video_key:
            return
        if name not in self._cameras:
            raise BadRequest(
                f"{name!r} is not a camera in {self.repo_id} — "
                f"this dataset has {', '.join(self._cameras)}"
            )
        with self._lock:
            self._video_key = name
            self._episodes = None  # routing is per camera
            self._close_locked()  # and so is the open container

    @classmethod
    def discover(
        cls, hf_home: str | Path | None = None, repo_id: str = DEFAULT_DATASET
    ) -> LeRobotV3Reader:
        return cls(snapshot_path(hf_home, repo_id), repo_id)

    # ---------- metadata ----------

    def episodes(self) -> list[EpisodeInfo]:
        """Episode routing for the CURRENT camera, built once and cached.

        Under the lock because `_episodes` is the cache `use_camera` clears and
        `_video_key` is what it swaps. `frame()` calls this before taking the
        lock for its decode, so without this a camera change landing in the gap
        hands back camera A's routing rows under camera B's key — a frame from
        the wrong view, with nothing anywhere saying so.
        """
        with self._lock:
            return self._episodes_locked()

    def _episodes_locked(self) -> list[EpisodeInfo]:
        if self._episodes is None:
            files = sorted((self.snapshot / "meta" / "episodes").rglob("*.parquet"))
            if not files:
                raise Refusal(f"No episode metadata under {self.snapshot}")
            table = _read_all(files)  # same shard trap as the frames
            out: list[EpisodeInfo] = []
            n = len(table["episode_index"])
            cam = self._video_key
            vid_from = self._column(table, f"videos/{cam}/from_timestamp", n, True)
            vid_to = self._column(table, f"videos/{cam}/to_timestamp", n, True)
            vid_chunk = self._column(table, f"videos/{cam}/chunk_index", n, True)
            vid_file = self._column(table, f"videos/{cam}/file_index", n, True)
            data_from = self._column(table, "dataset_from_index", n, False)
            for i in range(n):
                length = int(table["length"][i])
                tasks = table.get("tasks", [[]] * n)[i]
                task = (
                    tasks[0] if isinstance(tasks, list) and tasks else str(tasks or "")
                )
                # THE COLUMNS ARE NAMESPACED BY CAMERA. This read
                # `video_from_timestamp`, which is not a column any LeRobot
                # v3.0 dataset has -- the real name is
                # `videos/<camera>/from_timestamp`. `.get(name, default)`
                # turned that miss into 0.0 for every episode, so every
                # episode decoded from the start of the file: measured on
                # lerobot/pusht, episodes 0, 5 and 20 returned byte-identical
                # images while the state vector printed underneath them was
                # correctly episode 5's and episode 20's. The picture and the
                # numbers disagreed and nothing said so.
                #
                # Hence `_column`, which refuses instead of defaulting. A
                # missing routing column means frames cannot be located, and
                # saying so is the only honest answer.
                out.append(
                    EpisodeInfo(
                        index=int(table["episode_index"][i]),
                        length=length,
                        task=task,
                        from_ts=float(vid_from[i] or 0.0),
                        to_ts=float(vid_to[i] or 0.0),
                        video_chunk=int(vid_chunk[i] or 0),
                        video_file=int(vid_file[i] or 0),
                        # The dataset states the row range outright. Summing
                        # the lengths of earlier episodes gets the same answer
                        # only while the rows happen to be contiguous and in
                        # episode order.
                        data_from=int(data_from[i] or 0)
                        if data_from is not None
                        else sum(int(table["length"][j]) for j in range(i)),
                    )
                )
            self._episodes = out
        return self._episodes

    def _column(self, table: dict, name: str, n: int, required: bool):
        col = table.get(name)
        if col is None:
            if required:
                raise Refusal(
                    f"{self.repo_id} has no `{name}` column — this reader needs "
                    "LeRobot v3.0 episode metadata to locate frames, and "
                    "guessing the location silently is how the wrong episode "
                    "ends up on screen"
                )
            return None
        return col

    def _frame_table(self) -> dict:
        """All per-frame rows, loaded once (PushT is ~1.4 MB)."""
        if self._frames is None:
            files = sorted((self.snapshot / "data").rglob("*.parquet"))
            if not files:
                raise Refusal(f"No frame data under {self.snapshot}")
            # EVERY shard, not the first. LeRobot splits larger datasets across
            # data/chunk-000/file-000.parquet, file-001, ... and reading only
            # files[0] does not fail — it silently returns a prefix of the
            # dataset, so episodes past the first shard simply do not exist and
            # every index past it is wrong. A truncated answer that looks whole
            # is the worst shape a bug can take here.
            self._frames = _read_all(files)
        return self._frames

    def action_stats(self) -> dict:
        """The statistics this dataset publishes about its recorded actions.

        `{}` when it publishes none, and empty is load-bearing: `vla_actions`
        treats it as "do not overlay" rather than as identity scaling. A
        dataset's recorded actions and a policy's predicted ones are two lists
        of floats of the same length, which is exactly what makes drawing them
        on one axis look reasonable when their units differ.

        `meta/stats.json` has been in this module's own layout docstring since
        it was written and was never actually read — the file was described and
        ignored, so every comparison downstream would have had to assume units
        that were sitting on disk unread.
        """
        path = self.snapshot / "meta" / "stats.json"
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        action = blob.get("action")
        if not isinstance(action, dict):
            return {}
        out: dict = {}
        for key in ("mean", "std", "min", "max"):
            value = action.get(key)
            if isinstance(value, (list, tuple)) and value:
                try:
                    out[key] = [float(v) for v in value]
                except (TypeError, ValueError):
                    continue
        return {"action": out} if out else {}

    def action_units(self) -> tuple[object | None, str]:
        """What this dataset says its action dimensions are measured IN.

        Returns `(published, source)`. `published` is whatever the dataset
        wrote -- a string for the whole vector, or a list of one string per
        dimension -- and `None` when it wrote nothing, which is the common
        case: LeRobot's `info.json` feature schema has `dtype`, `shape` and
        `names`, and no unit field at all. `source` names the key it came from
        so a reader can go and look, and is `""` when there was none.

        `None` rather than a guess, and `vla_actions.units_agree` is the
        reason the distinction is worth a method: it already refuses to draw a
        policy's actions over a dataset's when either side publishes no
        normalisation statistics, on the grounds that two unlabelled axes of
        the same length are exactly the case that looks comparable and is not.
        A number with no unit is not a measurement, and inventing "radians"
        here because most arms use them would be inventing it for every
        dataset that never said.
        """
        feature = (self.info.get("features") or {}).get("action") or {}
        for key in ("unit", "units"):
            value = feature.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip(), f"meta/info.json features.action.{key}"
            if isinstance(value, (list, tuple)) and value:
                return [str(v) for v in value], (
                    f"meta/info.json features.action.{key}"
                )
        return None, ""

    def dataset_action_stats(
        self,
        *,
        max_rows: int | None = None,
        bins: int = DEFAULT_ACTION_BINS,
        max_dimensions: int | None = None,
        percentiles: Sequence[float] = ACTION_PERCENTILES,
    ) -> dict:
        """Per-dimension statistics over EVERY recorded action in the dataset.

        `action_stats()` above is a different fact wearing a similar name: it
        parses the mean/std/min/max the publisher wrote into `meta/stats.json`
        and computes nothing. This measures the actions themselves, and it
        returns the published numbers beside the measured ones without
        preferring either — a publisher's normalisation constants disagreeing
        with the recorded actions is the quietest corruption in a robot
        dataset (training normalises with them, so nothing breaks visibly and
        the policy simply sees a distribution nobody intended) and this is the
        one place both halves are on screen together.

        STREAMED, in `ACTION_ROW_BATCH`-row batches, one parquet shard open at
        a time. Two passes: the first measures count/min/max/mean/std, the
        second fills a histogram whose edges the first pass had to establish.
        That is twice the I/O and a constant amount of memory, and the
        alternative — binning against the PUBLISHED min/max — would make the
        histogram depend on the numbers this method exists to check.

        Nothing is judged. A dimension that never moves is reported with a
        standard deviation of 0.0, which is a measurement; whether that is a
        gripper held open on purpose or a sensor that came unplugged is the
        reader's call.
        """
        if bins < 1 or bins > MAX_ACTION_BINS:
            raise BadRequest(
                f"bins must be between 1 and {MAX_ACTION_BINS}; {bins} was "
                f"asked for. The histogram is per action dimension and every "
                f"bin travels in the response, so this is a payload limit "
                f"rather than a statistical one."
            )
        if max_rows is not None and max_rows < 1:
            raise BadRequest(
                f"max_rows must be at least 1, or omitted to read every row; "
                f"{max_rows} was asked for. A cap of zero is not an empty "
                f"dataset, it is a request for no measurement at all."
            )
        if max_dimensions is not None and max_dimensions < 1:
            raise BadRequest(
                f"max_dimensions must be at least 1, or omitted to summarise "
                f"every action dimension; {max_dimensions} was asked for."
            )
        levels = [float(q) for q in percentiles]
        if not levels or any(not 0.0 <= q <= 100.0 for q in levels):
            raise BadRequest(
                "every percentile must be between 0 and 100, and at least one "
                f"is required; got {levels}."
            )

        import pyarrow.parquet as pq

        files = sorted((self.snapshot / "data").rglob("*.parquet"))
        if not files:
            raise Refusal(
                f"{self.repo_id} has no frame data under {self.snapshot / 'data'}, "
                f"so there are no recorded actions to summarise. A LeRobot v3.0 "
                f"snapshot keeps them in data/chunk-*/file-*.parquet; if the "
                f"download was interrupted, re-run it."
            )

        # FOOTERS ONLY. `ParquetFile.metadata` reads the trailing schema block
        # and no row group, so the true row count costs one seek per shard
        # even when the cap below means most of them are never opened again --
        # which is the whole point: a capped answer that cannot say what it
        # capped is indistinguishable from a complete one.
        rows_total = 0
        rows_available = 0
        with_action: list[Path] = []
        first_columns: list[str] = []
        for path in files:
            with pq.ParquetFile(path) as handle:
                names = list(handle.schema_arrow.names)
                if not first_columns:
                    first_columns = names
                n = int(handle.metadata.num_rows)
            rows_total += n
            if "action" in names:
                with_action.append(path)
                rows_available += n
        if not with_action:
            raise Refusal(
                f"{self.repo_id} has {len(files)} frame shard(s) and none of "
                f"them has an `action` column, so there is nothing here to "
                f"summarise. The first shard's columns are: "
                f"{', '.join(first_columns) or '(none)'}. Recorded actions are "
                f"what this reads; a dataset of observations alone is a real "
                f"thing and is simply not what this measures."
            )
        if rows_available == 0:
            raise Refusal(
                f"{self.repo_id} has an `action` column and zero rows under it "
                f"({len(files)} shard(s), {rows_total} row(s) in total). There "
                f"is no distribution over no actions — this is an empty "
                f"recording rather than a broken one, and re-recording or "
                f"pointing at another dataset is the way forward."
            )

        limit = min(max_rows, rows_available) if max_rows is not None else None

        # ---- pass 1: count, min, max, mean, std, and the non-finite tally
        accs: list[_ActionDim] = []
        rows_read = 0
        rows_malformed = 0
        width = 0
        for batch in _stream_action_rows(with_action, ACTION_ROW_BATCH, limit):
            for row in batch:
                rows_read += 1
                values = _as_row(row)
                if values is None:
                    rows_malformed += 1
                    for acc in accs:
                        acc.n_missing += 1
                    continue
                if not accs:
                    width = len(values)
                    keep = (
                        width if max_dimensions is None else min(width, max_dimensions)
                    )
                    accs = [_ActionDim() for _ in range(keep)]
                elif len(values) != width:
                    # The width is the dataset's own, taken from its first
                    # readable row. A row of a different length is not a
                    # narrower action, it is a row nobody can pair with the
                    # others by position -- so it counts as malformed and its
                    # values are excluded rather than being folded into
                    # whichever dimensions happen to line up.
                    rows_malformed += 1
                    for acc in accs:
                        acc.n_missing += 1
                    continue
                for i, acc in enumerate(accs):
                    acc.add(values[i])
        if not accs:
            raise Refusal(
                f"{self.repo_id} has {rows_read} action row(s) and not one of "
                f"them could be read as a list of numbers, so no dimension "
                f"could even be counted. The `action` column holds something "
                f"this reader does not understand; open one shard under "
                f"{self.snapshot / 'data'} and look at what is in it."
            )

        # ---- pass 2: the histogram, now that the edges are known
        plans = [acc.bin_plan(bins) for acc in accs]
        counts = [[0] * plan[2] for plan in plans]
        if any(acc.count for acc in accs):
            for batch in _stream_action_rows(with_action, ACTION_ROW_BATCH, limit):
                for row in batch:
                    values = _as_row(row)
                    if values is None or len(values) != width:
                        continue
                    for i in range(len(accs)):
                        lo, bin_width, n_bins = plans[i]
                        if n_bins == 0:
                            continue
                        v = _as_float(values[i])
                        if v is None or not math.isfinite(v):
                            continue
                        counts[i][_bin_of(v, lo, bin_width, n_bins)] += 1

        names = self.action_names()
        # Names are dropped WHOLESALE when the count disagrees, the way
        # `vla_actions.compare` drops them: a table with seven rows and six
        # labels mislabels at least one, and a mislabelled joint is worse than
        # an unlabelled one.
        if len(names) != width:
            names = []
        unit_published, unit_source = self.action_units()
        published = self.action_stats().get("action") or {}
        pub_width = max((len(v) for v in published.values()), default=0)

        per_dimension = []
        for i, acc in enumerate(accs):
            lo, bin_width, _ = plans[i]
            per_dimension.append(
                acc.to_dict(
                    index=i,
                    name=names[i] if names else None,
                    unit=_unit_for(unit_published, i, width),
                    levels=levels,
                    bin_lo=lo,
                    bin_width=bin_width,
                    bin_counts=counts[i],
                    published=_published_for(published, i, width),
                )
            )

        n_episodes, episodes_reason = self._episode_count()
        out = {
            "repo_id": self.repo_id,
            "dimensions": width,
            # What `info.json` CLAIMS the action width is, beside what the
            # rows actually carry. Usually the same number; when it is not,
            # one of the two is wrong and neither this method nor the reader
            # can tell which without seeing both.
            "dimensions_declared": _declared_width(self.info),
            "dimensions_reported": len(accs),
            "max_dimensions": max_dimensions,
            "action_names": names,
            "units": {"published": unit_published, "source": unit_source or None},
            "n_episodes": n_episodes,
            "episodes_reason": episodes_reason,
            "rows_total": rows_total,
            "rows_with_action_column": rows_available,
            "rows_read": rows_read,
            "rows_skipped": rows_available - rows_read,
            "rows_malformed": rows_malformed,
            "max_rows": max_rows,
            "files_total": len(files),
            "files_read": len(with_action),
            "files_without_action": len(files) - len(with_action),
            "bins": bins,
            "percentile_levels": levels,
            # Said in the payload rather than only in the docstring: these are
            # read off the histogram, so each is accurate to the bin width
            # sitting beside it. The histogram COUNTS are exact.
            "percentile_method": (
                f"linear interpolation inside a histogram of {bins} bins over "
                f"the rows that were read; each dimension's "
                f"`percentile_resolution` is its bin width and bounds the error"
            ),
            "nonfinite_total": sum(a.n_nan + a.n_inf for a in accs),
            "published": published,
            "published_dimensions": pub_width or None,
            "per_dimension": per_dimension,
        }
        out["largest_published_gap"] = _largest_gap(per_dimension)
        out["means"] = _action_stats_means(out)
        return out

    def _episode_count(self) -> tuple[int | None, str]:
        """How many episodes the metadata declares, from parquet footers only.

        NOT `episodes()`. That builds routing for the current camera and
        refuses outright when a `videos/<camera>/from_timestamp` column is
        missing — correct there, because a frame that cannot be located must
        not be guessed at, and wrong here: the actions live in the data
        shards and are perfectly measurable on a dataset whose video routing
        is broken or absent. Counting rows in `meta/episodes/**.parquet` is
        the same number `episodes()` would return, one footer per shard, and
        it does not care which camera is selected.

        `(None, why)` when it cannot be counted — never `(0, ...)`, because
        "this dataset has no episodes" and "the episode table could not be
        read" are different answers and the sentence has to say which.
        """
        import pyarrow.parquet as pq

        files = sorted((self.snapshot / "meta" / "episodes").rglob("*.parquet"))
        if not files:
            return None, (
                f"no episode metadata under {self.snapshot / 'meta' / 'episodes'}, "
                f"so the row counts below are not divided by episode"
            )
        total = 0
        for path in files:
            try:
                with pq.ParquetFile(path) as handle:
                    total += int(handle.metadata.num_rows)
            except (OSError, ValueError) as err:
                return None, (
                    f"{path.name} could not be read as parquet "
                    f"({type(err).__name__}), so the episode count is unknown "
                    f"rather than zero"
                )
        return total, ""

    def action_names(self) -> list[str]:
        """What this dataset calls each action dimension, or `[]`.

        From `meta/info.json`'s `features["action"]["names"]`. Empty rather
        than invented indices: a chart with six curves and five labels
        mislabels at least one, and `vla_actions.compare` drops a name list
        whose length disagrees for that reason.
        """
        feature = (self.info.get("features") or {}).get("action") or {}
        names = feature.get("names")
        # LeRobot writes this either as a flat list or as {"motors": [...]}.
        if isinstance(names, dict):
            for value in names.values():
                if isinstance(value, (list, tuple)):
                    names = value
                    break
        if not isinstance(names, (list, tuple)):
            return []
        return [str(n) for n in names]

    def frames_readable(self) -> tuple[bool, str]:
        """Whether a picture can be produced at all, and why not when it cannot.

        Answered WITHOUT decoding anything, because it has to be answerable
        before the panel draws. Two things stop a frame from arriving and
        neither is visible from the episode table: `av` is imported the first
        time a frame is decoded, and the videos may simply not be in the
        snapshot.

        MEASURED on a machine with pyarrow and no av: `GET /api/vla/episodes`
        answered 200 with 206 episodes and nothing to say the pictures were
        unreachable, so the frontend swapped in the full panel — an episode
        picker, a frame scrubber, a "load vision tower" button — none of which
        can ever produce an image, while every actual frame request answered
        409 naming the missing package. The refusal WAS shown, at the bottom,
        under four sub-panels that should never have rendered.

        `""` for the reason when readable, so the caller has one field to
        test rather than two that can disagree.
        """
        try:
            import av  # noqa: F401
        except ImportError as err:
            return False, (
                f"Decoding this dataset's video needs `av`, which is not "
                f"installed on this machine ({err.name or 'av'} is missing). "
                f"Install it with `pip install modelmri[vla]`. The episode "
                f"table below was read from the parquet metadata and is real; "
                f"no picture from it can be shown until then."
            )
        try:
            self._video_file()
        except Refusal as err:
            return False, err.sentence
        return True, ""

    def summary(self) -> dict:
        eps = self.episodes()
        shape = (
            self.info.get("features", {})
            .get(self._video_key, {})
            .get("shape", [96, 96, 3])
        )
        readable, why = self.frames_readable()
        return {
            "repo_id": self.repo_id,
            "fps": self.fps,
            "video_key": self._video_key,
            "cameras": self.cameras,
            "image_shape": list(shape),
            "n_episodes": len(eps),
            "episodes": [e.__dict__ for e in eps],
            # Whether anything in this table can be SEEN. The episode list is
            # read from parquet and arrives fine on a machine that cannot
            # decode a single frame, so a caller that gates on the list alone
            # draws a picture panel that can only ever refuse.
            "frames_readable": readable,
            "frames_reason": why,
        }

    # ---------- frames ----------

    def _video_file(self, ep: EpisodeInfo | None = None) -> Path:
        """The mp4 holding `ep`, for the camera currently selected.

        This was `sorted(rglob("*.mp4"))[0]` -- the first mp4 anywhere under
        the snapshot. With one camera and one chunk that is the right file by
        luck. With two cameras it is whichever key sorts first, so the panel
        could show the wrist view while labelling it the overhead one; with
        two chunks every episode past the first chunk decoded from the wrong
        file entirely. The layout states where to look:
        videos/<camera>/chunk-000/file-000.mp4
        """
        root = self.snapshot / "videos" / self._video_key
        if ep is not None:
            exact = (
                root / f"chunk-{ep.video_chunk:03d}" / f"file-{ep.video_file:03d}.mp4"
            )
            if exact.is_file():
                return exact
        vids = sorted(root.rglob("*.mp4")) or sorted(
            (self.snapshot / "videos").rglob("*.mp4")
        )
        if not vids:
            raise Refusal(f"No videos under {self.snapshot}")
        return vids[0]

    def _decode(self, timestamp: float, ep: EpisodeInfo | None = None):
        """Decode the frame at `timestamp` seconds (keeps one container open)."""
        import av

        path = self._video_file(ep)
        if self._container is None or self._container_key != (str(path), "r"):
            # `_close_locked`, not `close`: every caller of `_decode` already
            # holds `self._lock`, and `close` now takes it.
            self._close_locked()
            self._container = av.open(str(path))
            self._container_key = (str(path), "r")
        container = self._container
        stream = container.streams.video[0]
        target = int(timestamp / float(stream.time_base))
        container.seek(max(target, 0), stream=stream, any_frame=False, backward=True)
        best = None
        for frame in container.decode(stream):
            best = frame
            if frame.pts is not None and frame.pts >= target:
                break
        if best is None:
            # Deliberately a plain RuntimeError, and the only raise in this
            # file that is not a BadRequest: PyAV yielded nothing at all for
            # this file — a truncated video, a codec this build cannot decode,
            # a container that failed to seek. That is something breaking
            # underneath, not a decision anyone made, so it belongs on the 500
            # path with its traceback in the terminal.
            #
            # Note it is NOT the out-of-range case: seeking past the end still
            # decodes frames and returns the last one (measured on
            # lerobot/pusht), which is a different problem and not this
            # branch's to report.
            raise RuntimeError(f"Could not decode a frame at t={timestamp:.3f}s")
        return best.to_ndarray(format="rgb24")

    def frame(self, episode: int, t: int) -> FrameSample:
        # ONE lock for the routing AND the decode. Resolving the episode first
        # and taking the lock afterwards leaves a gap a camera change fits
        # inside, and the rows resolved for the old camera would then be
        # decoded against the new one's container.
        with self._lock:
            eps = self._episodes_locked()
            match = next((e for e in eps if e.index == episode), None)
            if match is None:
                raise BadRequest(f"episode {episode} not in [0,{len(eps)})")
            if not 0 <= t < match.length:
                raise BadRequest(
                    f"t must be in [0,{match.length}) for episode {episode}"
                )
            rows = self._frame_table()
            # `data_from` comes from the dataset's own dataset_from_index
            # where that column exists, and falls back to the summed-length
            # assumption where it does not.
            i = match.data_from + t
            state = [float(v) for v in rows["observation.state"][i]]
            action = [float(v) for v in rows["action"][i]]
            timestamp = match.from_ts + t / self.fps
            rgb = self._decode(timestamp, match)

        return FrameSample(
            episode=episode,
            t=t,
            timestamp=round(timestamp, 3),
            state=state,
            action=action,
            task=match.task,
            image=encode_png(rgb),
            width=int(rgb.shape[1]),
            height=int(rgb.shape[0]),
        )

    def raw_frame(self, episode: int, t: int):
        """The decoded RGB ndarray for a frame (for model input)."""
        # One lock across routing and decode, for the reason `frame()` gives.
        with self._lock:
            eps = self._episodes_locked()
            match = next((e for e in eps if e.index == episode), None)
            if match is None:
                raise BadRequest(f"episode {episode} not in [0,{len(eps)})")
            # The same bound `frame()` above enforces, and
            # `HDF5Reader.raw_frame` enforces, and this one did not. LeRobot
            # v3.0 concatenates many episodes into ONE mp4, so an episode is a
            # span inside a file: a `t` past the end resolves to
            # `from_ts + t/fps`, which lands inside the NEXT episode and
            # decodes a real frame from it. No error, a picture on screen, and
            # it is the method the analysis and occlusion paths call -- so the
            # causal map, the attention comparison and the shared .mri would
            # all be of a frame belonging to another episode, labelled with
            # this one.
            if not 0 <= t < match.length:
                raise BadRequest(
                    f"t must be in [0,{match.length}) for episode {episode}"
                )
            return self._decode(match.from_ts + t / self.fps, match)

    def close(self) -> None:
        """Free the open container. Safe to call from another thread."""
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        """The same, for callers that already hold `self._lock`.

        Split because `threading.Lock` is not reentrant: `use_camera` and
        `_decode` both need to close while holding it, and calling the public
        `close` from there would deadlock rather than crash — which is a
        quieter failure and no better.
        """
        if self._container is not None:
            try:
                self._container.close()
            finally:
                self._container = None
                self._container_key = None


# ------------------------------ dataset-wide action statistics --------------
#
# Everything below is memory-bounded on purpose. The accumulator holds eight
# numbers per action dimension and the histogram holds `bins` integers, so the
# whole measurement costs the same on a 200-row recording and on a 26-million
# row one; the only thing that grows with the dataset is time.


@dataclass
class _ActionDim:
    """One action dimension's running totals over a streamed frame table."""

    count: int = 0  # finite values folded in
    n_nan: int = 0
    n_inf: int = 0
    n_missing: int = 0  # rows where this dimension had no readable value
    lo: float | None = None
    hi: float | None = None
    mean: float = 0.0
    m2: float = 0.0  # Welford's sum of squared deviations

    def add(self, raw) -> None:
        """Fold one recorded value in, or record why it could not be.

        WELFORD, not sum-and-sum-of-squares. The cheap form computes the
        variance as `E[x^2] - E[x]^2`, and a joint recorded in pixel
        coordinates around 512 with a spread of 0.3 is exactly the case where
        those two large numbers cancel: in float64 the difference loses most
        of its significant digits, and a variance can come out NEGATIVE, whose
        square root is a domain error rather than a wrong answer anybody
        notices. This form subtracts before it squares and never forms either
        large intermediate.
        """
        v = _as_float(raw)
        if v is None:
            self.n_missing += 1
            return
        # NaN and infinity are COUNTED and excluded, never folded in. One NaN
        # in a million rows makes the mean, the standard deviation, the min
        # and the max all NaN, and a chart of NaNs looks like a chart of a
        # dimension that was never recorded -- which is a different and much
        # less alarming finding than "your recording has a hole in it".
        if math.isnan(v):
            self.n_nan += 1
            return
        if math.isinf(v):
            self.n_inf += 1
            return
        self.count += 1
        if self.lo is None or v < self.lo:
            self.lo = v
        if self.hi is None or v > self.hi:
            self.hi = v
        delta = v - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (v - self.mean)

    def bin_plan(self, bins: int) -> tuple[float | None, float, int]:
        """`(low edge, bin width, number of bins)` for this dimension.

        A dimension that never varies gets ONE bin of zero width rather than
        `bins` empty ones around a single value. It is not a degenerate case
        to skip: a gripper held open for a whole dataset is real recorded
        data, its histogram is one bar holding every row, and its percentiles
        are all exactly that value with a resolution of zero.
        """
        if self.count == 0 or self.lo is None or self.hi is None:
            return None, 0.0, 0
        if self.hi == self.lo:
            return self.lo, 0.0, 1
        return self.lo, (self.hi - self.lo) / bins, bins

    def to_dict(
        self,
        *,
        index: int,
        name: str | None,
        unit: str | None,
        levels: list[float],
        bin_lo: float | None,
        bin_width: float,
        bin_counts: list[int],
        published: dict,
    ) -> dict:
        # POPULATION standard deviation (divide by N), which is what LeRobot
        # writes into meta/stats.json -- comparing a sample std against a
        # population one would manufacture a disagreement out of the
        # convention rather than out of the data. `None` under two values,
        # because the spread of a single sample is not zero, it is unmeasured.
        std = math.sqrt(self.m2 / self.count) if self.count >= 2 else None
        measured = {
            "mean": self.mean if self.count else None,
            "std": std,
            "min": self.lo,
            "max": self.hi,
        }
        edges: list[float] = []
        if bin_lo is not None and bin_counts:
            edges = [bin_lo + i * bin_width for i in range(len(bin_counts) + 1)]
            # The top edge is set from the measurement rather than accumulated
            # to: 32 additions of (hi-lo)/32 does not land on `hi`, and an
            # edge a few ulps above the maximum reads as a value nothing in
            # the dataset reaches.
            edges[-1] = self.hi if self.hi is not None else edges[-1]
        return {
            "index": index,
            "name": name,
            "unit": unit,
            "count": self.count,
            "n_nan": self.n_nan,
            "n_inf": self.n_inf,
            "n_missing": self.n_missing,
            **measured,
            # `min == max` over everything that was read. A fact, not a
            # verdict: whether a motionless dimension is a held gripper or an
            # unplugged sensor is the reader's call, and `None` when there was
            # nothing to compare.
            "constant": None if self.count == 0 else self.lo == self.hi,
            "percentiles": [
                {
                    "q": q,
                    "value": _percentile_from_histogram(
                        q, self.count, bin_lo, bin_width, bin_counts, self.lo, self.hi
                    ),
                }
                for q in levels
            ],
            # The bin width, which bounds the error on every percentile above.
            # Exactly 0.0 for a constant dimension, where they are exact.
            "percentile_resolution": bin_width if self.count else None,
            "histogram": {"bin_edges": edges, "counts": list(bin_counts)},
            "published": published,
            # Measured minus published, per statistic, and `None` wherever one
            # of the two does not exist. The subtraction is the whole reason
            # both halves are here: normalisation constants that do not
            # describe the data beside them break nothing visibly, and this is
            # the difference that shows it.
            "measured_minus_published": {
                key: (
                    measured[key] - published[key]
                    if measured[key] is not None and published.get(key) is not None
                    else None
                )
                for key in ("mean", "std", "min", "max")
            },
        }


def _stream_action_rows(
    files: list[Path], batch_size: int, limit: int | None
) -> Iterator[list]:
    """Yield the `action` column in batches, one shard open at a time.

    The generator is the memory bound: `to_pylist()` materialises exactly one
    batch, the caller folds it into accumulators, and it is unreachable before
    the next one is read. `_frame_table()` does the opposite by design (it
    concatenates every shard into one dict) and is right to for PushT's 1.4 MB
    of rows; it is not an option for a dataset-wide statistic.
    """
    import pyarrow.parquet as pq

    seen = 0
    for path in files:
        with pq.ParquetFile(path) as handle:
            for batch in handle.iter_batches(batch_size=batch_size, columns=["action"]):
                rows = batch.column(0).to_pylist()
                if limit is not None and seen + len(rows) > limit:
                    # The cap lands mid-batch rather than at a shard boundary,
                    # so `rows_read` is the number asked for instead of the
                    # nearest batch above it. Both passes take the identical
                    # prefix, so the histogram describes the rows the mean
                    # was computed over and not a different set.
                    rows = rows[: limit - seen]
                seen += len(rows)
                if rows:
                    yield rows
                if limit is not None and seen >= limit:
                    return


def _as_float(v) -> float | None:
    """One recorded number, or `None` when it is not one at all."""
    if isinstance(v, bool):
        # A gripper flag recorded as a boolean is a real value and 1.0/0.0 is
        # what it means. Explicit and FIRST, because `isinstance(True, int)`
        # is True: falling through the numeric arm below would give the same
        # answer by accident, and the next person to narrow that arm would
        # have no way to see that a bool had been relying on it.
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_row(row) -> list | None:
    """One frame's action as a list of values, or `None` if it is not usable."""
    if row is None:
        return None
    if isinstance(row, (list, tuple)):
        # An empty list is not a zero-dimensional action, it is a row with
        # nothing in it — counted as malformed rather than silently defining
        # the dataset's width as zero.
        return list(row) if row else None
    if isinstance(row, dict):
        return None
    value = _as_float(row)
    # A scalar action column is a one-dimensional action, which is a real
    # LeRobot shape (a single gripper channel) and not a malformed row.
    return None if value is None else [value]


def _bin_of(v: float, lo: float, bin_width: float, n_bins: int) -> int:
    if bin_width <= 0.0:
        return 0
    index = int((v - lo) / bin_width)
    if index < 0:
        return 0
    # The maximum lands exactly on the top edge, which is one past the last
    # bin. Clamped rather than dropped: the largest recorded value belongs in
    # the histogram, and an off-by-one there loses exactly the row a reader
    # looking for a joint limit came for.
    return n_bins - 1 if index >= n_bins else index


def _percentile_from_histogram(
    q: float,
    count: int,
    lo: float | None,
    bin_width: float,
    counts: list[int],
    vmin: float | None,
    vmax: float | None,
) -> float | None:
    """The q-th percentile, interpolated inside the bin that contains it.

    ESTIMATED, and the payload says so: the error is bounded by one bin width,
    which travels beside it as `percentile_resolution`. Exact percentiles
    would mean holding every value of every dimension in memory, which is the
    one thing this method exists not to do.
    """
    if count == 0 or lo is None or not counts:
        return None
    if bin_width == 0.0:
        # A constant dimension. Every percentile IS the value, exactly.
        return vmin
    target = q / 100.0 * count
    cumulative = 0
    for index, held in enumerate(counts):
        if held == 0:
            continue
        if cumulative + held >= target:
            fraction = (target - cumulative) / held
            value = lo + (index + fraction) * bin_width
            if vmin is not None and value < vmin:
                return vmin
            if vmax is not None and value > vmax:
                return vmax
            return value
        cumulative += held
    return vmax


def _declared_width(info: dict) -> int | None:
    """The action width `info.json` claims, or `None` when it claims none."""
    feature = (info.get("features") or {}).get("action") or {}
    shape = feature.get("shape")
    if isinstance(shape, (list, tuple)) and len(shape) == 1:
        head = shape[0]
        # bool before int, because `isinstance(True, int)` is True and a
        # shape of `[True]` is not a one-dimensional action.
        if not isinstance(head, bool) and isinstance(head, int):
            return int(head)
    if not isinstance(shape, bool) and isinstance(shape, int):
        return int(shape)
    return None


def _unit_for(published, index: int, width: int) -> str | None:
    """This dimension's stated unit, or `None` when nothing stated one."""
    if isinstance(published, str):
        # One unit for the whole action feature is the DATASET's statement
        # about every dimension in it, so it is carried on each. Expanding a
        # per-dimension list of the wrong length would be this module's
        # invention instead, which is why that case is dropped below.
        return published
    if isinstance(published, (list, tuple)) and len(published) == width:
        return str(published[index])
    return None


def _published_for(published: dict, index: int, width: int) -> dict:
    """What the dataset published for this dimension, `None` where it did not.

    Per statistic rather than all-or-nothing: a `meta/stats.json` can carry a
    mean over the right number of dimensions and a min over the wrong one, and
    dropping the pair because one of them disagrees would hide the half that
    was fine.
    """
    out: dict = {}
    for key in ("mean", "std", "min", "max"):
        column = published.get(key)
        out[key] = (
            float(column[index])
            if isinstance(column, (list, tuple)) and len(column) == width
            else None
        )
    return out


def _largest_gap(per_dimension: list[dict]) -> dict | None:
    """The biggest |measured - published| anywhere, or `None` if nothing paired.

    A POINTER, not a ranking, and the payload says so in `caveat`. It is
    chosen by absolute difference, which is only meaningful within one
    dimension: dimension 0 in pixels and dimension 5 in radians do not compare
    that way, and normalising them against each other's spread would mean
    inventing the tolerance this module refuses to invent. What it is for is
    the case where a reader has forty dimensions and a table too wide to scan
    — it says where to look first.
    """
    best: dict | None = None
    for dim in per_dimension:
        for key, gap in dim["measured_minus_published"].items():
            if gap is None:
                continue
            if best is None or abs(gap) > abs(best["difference"]):
                best = {
                    "dimension": dim["index"],
                    "name": dim["name"],
                    "statistic": key,
                    "measured": dim[key],
                    "published": dim["published"][key],
                    "difference": gap,
                }
    if best is not None:
        best["caveat"] = (
            "Chosen by absolute difference, which only compares within one "
            "dimension — dimensions in different units do not rank against "
            "each other. This says where to look first, not what is worst."
        )
    return best


def _action_stats_means(out: dict) -> str:
    """What was measured, over how much, and every cap that shaped it."""
    read, available = out["rows_read"], out["rows_with_action_column"]
    where = (
        f"across {out['n_episodes']} episode(s)"
        if out["n_episodes"] is not None
        else f"across an unknown number of episodes ({out['episodes_reason']})"
    )
    parts = [
        f"{read:,} of {available:,} recorded action row(s) in {out['repo_id']}, "
        f"{where}, read from {out['files_read']} parquet shard(s) in "
        f"{ACTION_ROW_BATCH:,}-row batches — no list of every action was ever "
        f"built, so this costs the same memory at 26 million rows as at 30."
    ]
    if out["max_rows"] is not None:
        parts.append(
            f"A CAP OF {out['max_rows']:,} ROWS was applied and "
            f"{out['rows_skipped']:,} row(s) went unread; every number here "
            f"describes the first {read:,} rows of the dataset in file order "
            f"and nothing else."
            if out["rows_skipped"]
            else f"A cap of {out['max_rows']:,} rows was asked for and this "
            f"dataset has {available:,}, so every row was read."
        )
    if out["dimensions_reported"] < out["dimensions"]:
        parts.append(
            f"ONLY THE FIRST {out['dimensions_reported']} of "
            f"{out['dimensions']} action dimension(s) are reported, because "
            f"max_dimensions was set; the rest were not measured."
        )
    declared = out["dimensions_declared"]
    if declared is not None and declared != out["dimensions"]:
        parts.append(
            f"meta/info.json declares {declared} action dimension(s) and the "
            f"rows carry {out['dimensions']}. One of those is wrong and "
            f"nothing here can tell which; the rows are what was measured."
        )
    if out["files_without_action"]:
        parts.append(
            f"{out['files_without_action']} of {out['files_total']} frame "
            f"shard(s) have no `action` column at all and were skipped, so "
            f"their {out['rows_total'] - available:,} row(s) are outside every "
            f"count above."
        )
    parts.append(
        f"Percentiles are read off a histogram of {out['bins']} bins per "
        f"dimension, so each is accurate to that dimension's "
        f"`percentile_resolution` (its bin width); the histogram COUNTS are "
        f"exact."
    )
    if out["nonfinite_total"]:
        parts.append(
            f"{out['nonfinite_total']} value(s) were NaN or infinite. They are "
            f"EXCLUDED from every statistic and counted per dimension as "
            f"`n_nan` and `n_inf` — one NaN folded in would have made a mean, "
            f"a min and a max all NaN, which reads as a dimension that was "
            f"never recorded rather than as a hole in one that was."
        )
    if out["rows_malformed"]:
        parts.append(
            f"{out['rows_malformed']:,} row(s) could not be read as an action "
            f"vector of {out['dimensions']} number(s) and were excluded; they "
            f"are counted in each dimension's `n_missing`."
        )
    frozen = [d["index"] for d in out["per_dimension"] if d["constant"] is True]
    if frozen:
        parts.append(
            f"Dimension(s) {frozen} never varied. That is a measurement, not a "
            f"defect — a gripper held open for a whole recording is real data "
            f"— and their standard deviation of 0.0 is the spread that was "
            f"measured rather than a missing one."
        )
    published_width = out["published_dimensions"]
    if not out["published"]:
        parts.append(
            "This dataset publishes no action statistics of its own in "
            "meta/stats.json, so `published` is empty and there is nothing to "
            "compare the measured numbers against. Empty is not agreement."
        )
    elif published_width is not None and published_width != out["dimensions"]:
        parts.append(
            f"meta/stats.json publishes statistics over {published_width} "
            f"dimension(s) and the rows carry {out['dimensions']}, so the two "
            f"were NOT paired — matching dimension 3 to dimension 3 across "
            f"different action spaces would compare unrelated joints."
        )
    else:
        parts.append(
            "The mean/std/min/max this dataset publishes in meta/stats.json "
            "travel beside the measured ones with the difference computed, "
            "and neither was substituted for the other: training normalises "
            "with the published pair, so if they do not describe the rows "
            "here, nothing raises and the policy simply sees a distribution "
            "nobody intended."
        )
        gap = out["largest_published_gap"]
        if gap is not None:
            named = f"dimension {gap['dimension']}" + (
                f" ({gap['name']})" if gap["name"] else ""
            )
            # The number, not a word for it. Whether a gap of this size
            # matters depends on the units, which this dataset may not even
            # have stated, so the reader gets both sides and decides.
            parts.append(
                f"The largest of those differences is `{gap['statistic']}` on "
                f"{named}: published {fmt.measured(gap['published'], 6)}, "
                f"measured {fmt.measured(gap['measured'], 6)}, a difference of "
                f"{fmt.measured(gap['difference'], 6)}. That is where to look "
                f"first, not a ranking — dimensions in different units do not "
                f"compare by absolute difference."
            )
    units = out["units"]["published"]
    parts.append(
        f"Units come from {out['units']['source']}: {units!r}."
        if units is not None
        else "This dataset states no units for its action dimensions, so every "
        "number here is unitless as far as the dataset is concerned — which "
        "is why `unit` is None rather than a guess."
    )
    parts.append(
        "NOTHING HERE IS GRADED. These are counts, ranges and spreads; which "
        "of them is disqualifying for your run is your call."
    )
    return " ".join(parts)


def encode_png(rgb) -> str:
    """RGB ndarray -> data URL (96x96 PNG is ~5 KB, fine for JSON).

    Pillow lives in the vla-lite extra, so a base install reaching this raised
    a bare ModuleNotFoundError naming a module the user never asked for. It is
    reached from three places -- the frame server, `VLA.share_payload` and the
    HDF5 reader -- and every one of them is somebody looking at a robot frame.
    """
    try:
        from PIL import Image
    except ImportError as err:
        raise Refusal(
            "Encoding a robot frame as PNG needs Pillow. Install it with "
            "`pip install modelmri[vla-lite]`."
        ) from err

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def cached_datasets(hf_home: str | Path | None = None) -> list[dict]:
    """LeRobot datasets already on this machine.

    The panel was pinned to one repository, so "the robot section only has one
    dataset" was literally true. Anything cached is openable — the reader has
    always taken a repo_id, nothing ever offered a choice.

    Cheap by construction: directory names and one refs file per dataset. It
    never opens a parquet or decodes a frame, so listing costs nothing even
    when a dataset is tens of gigabytes.
    """
    out: list[dict] = []
    # Exactly the roots snapshot_path will open from. Listing a superset is
    # how the picker came to advertise datasets that could not be opened.
    for hub in dataset_roots(hf_home):
        if not hub.is_dir():
            continue
        for entry in sorted(hub.glob("datasets--*")):
            try:
                owner, name = entry.name.removeprefix("datasets--").split("--", 1)
            except ValueError:
                continue
            repo = f"{owner}/{name.replace('--', '/')}"
            if any(d["repo_id"] == repo for d in out):
                continue
            refs = (
                sorted((entry / "refs").glob("*")) if (entry / "refs").is_dir() else []
            )
            # Per file, not one `sum(...)` over a generator. As one expression
            # a single unreadable blob aborted the whole sum and left `size`
            # at 0 — so the picker advertised "0.0 GB" for a dataset that may
            # be 40 GB, which is a fabricated number in the one field whose
            # job is telling you how big a thing is before you open it. One
            # bad blob now costs its own bytes and nothing else.
            #
            # OSError is the expected failure and continuing is right: a blob
            # removed mid-scan (FileNotFoundError), a permission the walk does
            # not have, or — this is the common one on a synced drive — an
            # unmaterialised OneDrive/Google Drive placeholder, which raises
            # WinError 1920 rather than being quietly skipped by is_file().
            size = 0
            unreadable = 0
            try:
                for f in (entry / "blobs").rglob("*"):
                    try:
                        if f.is_file():
                            size += f.stat().st_size
                    except OSError:
                        unreadable += 1
            except OSError:
                # rglob's own directory walk, which can raise on Python 3.10
                # and 3.11 (this project supports both) where the pathlib
                # selector re-raises anything that is not PermissionError.
                unreadable += 1
            note = "" if refs else "no snapshot ref — the download is incomplete"
            if unreadable:
                # Said out loud rather than folded into the number: the size
                # below is now a lower bound, and a reader who is deciding
                # whether they have room for this needs to know that.
                note = (
                    f"{note + '; ' if note else ''}{unreadable} file(s) could not "
                    f"be read, so this size is a lower bound"
                ).strip()
            out.append(
                {
                    "repo_id": repo,
                    "ref": refs[-1].name if refs else None,
                    "size_gb": round(size / 1e9, 2),
                    "usable": bool(refs),
                    "note": note,
                }
            )
    return out
