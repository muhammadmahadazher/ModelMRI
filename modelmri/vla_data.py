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
import threading
from dataclasses import dataclass
from pathlib import Path

from .errors import BadRequest, Refusal

DEFAULT_DATASET = "lerobot/pusht"


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
    """Resolve the newest snapshot dir for a cached LeRobot dataset."""
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
        """Choose which view the frames come from."""
        if not name or name == self._video_key:
            return
        if name not in self._cameras:
            raise BadRequest(
                f"{name!r} is not a camera in {self.repo_id} — "
                f"this dataset has {', '.join(self._cameras)}"
            )
        self._video_key = name
        self._episodes = None  # routing is per camera
        self.close()  # and so is the open container

    @classmethod
    def discover(
        cls, hf_home: str | Path | None = None, repo_id: str = DEFAULT_DATASET
    ) -> LeRobotV3Reader:
        return cls(snapshot_path(hf_home, repo_id), repo_id)

    # ---------- metadata ----------

    def episodes(self) -> list[EpisodeInfo]:
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

    def summary(self) -> dict:
        eps = self.episodes()
        shape = (
            self.info.get("features", {})
            .get(self._video_key, {})
            .get("shape", [96, 96, 3])
        )
        return {
            "repo_id": self.repo_id,
            "fps": self.fps,
            "video_key": self._video_key,
            "cameras": self.cameras,
            "image_shape": list(shape),
            "n_episodes": len(eps),
            "episodes": [e.__dict__ for e in eps],
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
            self.close()
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
        eps = self.episodes()
        match = next((e for e in eps if e.index == episode), None)
        if match is None:
            raise BadRequest(f"episode {episode} not in [0,{len(eps)})")
        if not 0 <= t < match.length:
            raise BadRequest(f"t must be in [0,{match.length}) for episode {episode}")

        with self._lock:
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
        eps = self.episodes()
        match = next((e for e in eps if e.index == episode), None)
        if match is None:
            raise BadRequest(f"episode {episode} not in [0,{len(eps)})")
        # The same bound `frame()` above enforces, and `HDF5Reader.raw_frame`
        # enforces, and this one did not. LeRobot v3.0 concatenates many
        # episodes into ONE mp4, so an episode is a span inside a file: a `t`
        # past the end resolves to `from_ts + t/fps`, which lands inside the
        # NEXT episode and decodes a real frame from it. No error, a picture
        # on screen, and it is the method the analysis and occlusion paths
        # call -- so the causal map, the attention comparison and the shared
        # .mri would all be of a frame belonging to another episode, labelled
        # with this one.
        if not 0 <= t < match.length:
            raise BadRequest(f"t must be in [0,{match.length}) for episode {episode}")
        with self._lock:
            return self._decode(match.from_ts + t / self.fps, match)

    def close(self) -> None:
        if self._container is not None:
            try:
                self._container.close()
            finally:
                self._container = None
                self._container_key = None


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
