# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""ALOHA and robomimic HDF5, through the interface the LeRobot reader presents.

`vla_data.py` reads LeRobot v3.0 and nothing else, and "LeRobot format" does
not mean one thing: GR00T still ships a v3-to-v2 downconverter and Rerun could
not load v3.0 at all until a patch this year. Meanwhile the ALOHA and
robomimic layouts are what a large amount of real robot data is actually
stored in.

So this presents EXACTLY the surface `LeRobotV3Reader` presents — `episodes()`,
`frame()`, `raw_frame()`, `cameras`, `use_camera()`, `summary()`, `close()` —
which is why `VLAPanel.tsx` and every `/api/vla/*` route is untouched by it.
A reader that needed its own routes would be a second robot panel.

IT REFUSES AN UNFAMILIAR LAYOUT RATHER THAN GUESSING
----------------------------------------------------
HDF5 layouts vary between labs, and the guess that goes wrong is the
expensive one: picking the wrong dataset as "the action" produces a panel full
of numbers that are all real and all about the wrong thing. An unrecognised
file raises with the top-level keys it actually found, so the reader can see
what their file has and say what it means.

RLDS IS DELIBERATELY EXCLUDED. Hand-rolling a TFRecord framing reader plus a
`tf.Example` protobuf parser is well beyond the effort this was scoped at, and
the plan for it skipped masked CRC32C validation — accepting a silently corrupt
record in the same release as a dataset-integrity auditor is an incoherent
posture.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from .errors import BadRequest, Refusal
from .vla_data import EpisodeInfo, FrameSample, encode_png

# The two layouts this knows. Anything else is refused by name.
#
#   aloha      one episode per file: /observations/images/<cam>, /action
#   robomimic  many per file: /data/demo_N/obs/<cam>, /data/demo_N/actions
ALOHA = "aloha"
ROBOMIMIC = "robomimic"

# Where each layout keeps its pieces. Read from the file rather than assumed:
# `detect` walks these and the first that matches wins, and no match refuses.
_ALOHA_IMAGES = "observations/images"
_ALOHA_STATE = ("observations/qpos", "observations/state")
_ALOHA_ACTION = ("action", "actions")

_ROBOMIMIC_ROOT = "data"
_ROBOMIMIC_OBS = ("obs", "observations")
_ROBOMIMIC_ACTION = ("actions", "action")

# A frame this large is not a camera frame anybody trained on, and decoding
# one into the panel would be a browser hang rather than a picture.
MAX_FRAME_PIXELS = 8_000_000


class Hdf5Error(BadRequest):
    """This file cannot be read as a robot dataset, and we say why."""


@dataclass
class Layout:
    """Which convention this file follows, and where its pieces are."""

    kind: str
    # group path -> episode. ALOHA has one; robomimic has one per demo.
    episodes: list[str]
    images: str
    state: str
    action: str

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "n_episodes": len(self.episodes),
            "images": self.images,
            "state": self.state,
            "action": self.action,
        }


def _first_present(group, names: tuple) -> str:
    for name in names:
        if name in group:
            return name
    return ""


def detect(handle) -> Layout:
    """Which layout this file follows, or a refusal listing what it has.

    The refusal carries the TOP-LEVEL KEYS of the reader's own file, which is
    the one thing that lets them work out what to do next. It is their data on
    their machine, so there is nothing to withhold.
    """
    top = sorted(handle.keys())

    # ALOHA: one episode per file, images under a fixed group.
    if _ALOHA_IMAGES in handle:
        state = _first_present(handle, _ALOHA_STATE)
        action = _first_present(handle, _ALOHA_ACTION)
        if not action:
            raise Hdf5Error(
                f"this looks like an ALOHA file — it has "
                f"`{_ALOHA_IMAGES}` — but no action dataset at any of "
                f"{list(_ALOHA_ACTION)}. Top-level keys: {top}. Guessing "
                f"which dataset is the action would fill the panel with real "
                f"numbers about the wrong thing."
            )
        return Layout(
            kind=ALOHA,
            episodes=[""],
            images=_ALOHA_IMAGES,
            state=state,
            action=action,
        )

    # robomimic: many demos under /data.
    if _ROBOMIMIC_ROOT in handle:
        root = handle[_ROBOMIMIC_ROOT]
        demos = sorted(
            (k for k in root.keys() if hasattr(root[k], "keys")),
            key=_demo_order,
        )
        if not demos:
            raise Hdf5Error(
                f"this file has a `{_ROBOMIMIC_ROOT}` group with no episodes "
                f"in it. Top-level keys: {top}."
            )
        first = root[demos[0]]
        obs = _first_present(first, _ROBOMIMIC_OBS)
        action = _first_present(first, _ROBOMIMIC_ACTION)
        if not obs or not action:
            raise Hdf5Error(
                f"this looks like a robomimic file, but its first episode "
                f"`{demos[0]}` has no "
                f"{'observations' if not obs else 'action'} group — it "
                f"contains {sorted(first.keys())}. Guessing which dataset is "
                f"the action would fill the panel with real numbers about the "
                f"wrong thing."
            )
        return Layout(
            kind=ROBOMIMIC,
            episodes=[f"{_ROBOMIMIC_ROOT}/{d}" for d in demos],
            images=obs,
            state=obs,
            action=action,
        )

    raise Hdf5Error(
        f"ModelMRI does not recognise this HDF5 layout. It knows the ALOHA "
        f"convention (a `{_ALOHA_IMAGES}` group and an `action` dataset) and "
        f"the robomimic one (a `{_ROBOMIMIC_ROOT}` group of demos). This file's "
        f"top-level keys are: {top}. Refusing rather than guessing — picking "
        f"the wrong dataset as the action produces a panel of numbers that are "
        f"all real and all about the wrong thing."
    )


def _demo_order(name: str) -> tuple:
    """`demo_10` after `demo_9`, not between `demo_1` and `demo_2`.

    Sorting demo names as strings puts demo_10 before demo_2, so the episode
    indices in the panel would not match the order in the file — and a reader
    comparing episode 2 here against episode 2 in their training log would be
    looking at different data.
    """
    digits = "".join(c for c in name if c.isdigit())
    return (int(digits) if digits else 0, name)


class Hdf5Reader:
    """An ALOHA or robomimic file, behind the LeRobot reader's interface."""

    def __init__(self, path: str | Path) -> None:
        try:
            import h5py
        except ImportError as err:
            raise Refusal(
                "Reading an HDF5 robot dataset needs h5py. Install it with "
                "`pip install modelmri[vla-lite]`."
            ) from err

        target = Path(path)
        if not target.is_file():
            # No path in the message. `errors.py` names this exactly: a
            # library talking about this machine to somebody who was not
            # asking about this machine.
            raise Hdf5Error("that HDF5 file does not exist, or is not a file.")

        self._lock = threading.RLock()
        try:
            self._h5 = h5py.File(str(target), "r")
        except OSError as err:
            raise Hdf5Error(
                f"that file could not be opened as HDF5 "
                f"({type(err).__name__}). An .hdf5 extension is not a format."
            ) from err

        self.repo_id = target.stem
        self.layout = detect(self._h5)
        self._cameras = self._discover_cameras()
        if not self._cameras:
            raise Hdf5Error(
                f"this file has no camera images under "
                f"`{self.layout.images}`, so there is nothing for the vision "
                f"tower to look at."
            )
        self._camera = self._cameras[0]
        self._episodes: list[EpisodeInfo] | None = None

    # ---------- the LeRobot reader's surface ----------

    @property
    def cameras(self) -> list[str]:
        return list(self._cameras)

    @property
    def camera(self) -> str:
        return self._camera

    def use_camera(self, name: str | None) -> None:
        if name and name not in self._cameras:
            raise BadRequest(
                f"{name!r} is not a camera in this file — it has {self._cameras}."
            )
        if name:
            self._camera = name
            # The frame cache is per camera, so switching invalidates it.
            self._episodes = None

    @property
    def fps(self) -> float:
        """The control frequency, when the file states one.

        ALOHA files usually carry `sim` or a `/observations` attribute; many
        carry nothing. 0.0 means UNKNOWN, and every caller treats it as such —
        `vla_audit.check_action_lag` refuses outright rather than assuming 30,
        because a lag in frames means nothing without a real frequency.
        """
        for source in (
            self._h5.attrs,
            self._h5[self.layout.episodes[0]].attrs
            if self.layout.episodes[0]
            else self._h5.attrs,
        ):
            for key in ("fps", "frame_rate", "control_hz"):
                if key in source:
                    try:
                        return float(source[key])
                    except (TypeError, ValueError):
                        continue
        return 0.0

    @property
    def info(self) -> dict:
        """The shape `vla_audit` reads `fps` from. Empty when unknown."""
        rate = self.fps
        return {"fps": rate} if rate else {}

    def episodes(self) -> list[EpisodeInfo]:
        if self._episodes is None:
            out: list[EpisodeInfo] = []
            cursor = 0
            for index, group in enumerate(self.layout.episodes):
                images = self._images_for(group)
                length = int(images.shape[0])
                out.append(
                    EpisodeInfo(
                        index=index,
                        length=length,
                        task=self._task_for(group),
                        # HDF5 frames are arrays, not a video span. The
                        # timestamps are real seconds when the file states a
                        # rate and 0.0 when it does not — and `vla_audit`'s
                        # routing check reads a zero span as a fault, so this
                        # reader is not put through that check at all.
                        from_ts=0.0,
                        to_ts=length / self.fps if self.fps else 0.0,
                        data_from=cursor,
                    )
                )
                cursor += length
            self._episodes = out
        return self._episodes

    def summary(self) -> dict:
        eps = self.episodes()
        sample = self._images_for(self.layout.episodes[0])
        shape = list(sample.shape[1:]) if sample.ndim > 1 else []
        return {
            "repo_id": self.repo_id,
            "fps": self.fps,
            "video_key": self._camera,
            "cameras": self.cameras,
            "image_shape": shape,
            "n_episodes": len(eps),
            "episodes": [e.__dict__ for e in eps],
            # Named so a reader can see which convention was detected rather
            # than wondering why the episode count is what it is.
            "layout": self.layout.to_dict(),
        }

    def frame(self, episode: int, t: int) -> FrameSample:
        rgb = self.raw_frame(episode, t)
        info = self._episode_at(episode)
        group = self.layout.episodes[episode]
        return FrameSample(
            episode=episode,
            t=t,
            timestamp=round(t / self.fps, 3) if self.fps else 0.0,
            state=self._row(group, self.layout.state, t, images_ok=False),
            action=self._row(group, self.layout.action, t, images_ok=False),
            task=info.task,
            image=encode_png(rgb),
            width=int(rgb.shape[1]),
            height=int(rgb.shape[0]),
        )

    def raw_frame(self, episode: int, t: int):
        """The RGB ndarray for a frame, for model input."""
        import numpy as np

        info = self._episode_at(episode)
        if not 0 <= t < info.length:
            raise BadRequest(f"t must be in [0,{info.length}) for episode {episode}")
        with self._lock:
            images = self._images_for(self.layout.episodes[episode])
            rgb = np.asarray(images[t])
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        if rgb.ndim != 3 or rgb.shape[-1] not in (1, 3, 4):
            raise Hdf5Error(
                f"the frames under `{self._camera}` are shaped "
                f"{list(rgb.shape)}, which is not an image this can show."
            )
        if rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        rgb = rgb[..., :3]
        if rgb.shape[0] * rgb.shape[1] > MAX_FRAME_PIXELS:
            raise Hdf5Error(
                f"a {rgb.shape[1]}x{rgb.shape[0]} frame is larger than "
                f"anything a policy was trained on, and decoding it would "
                f"hang the panel rather than show a picture."
            )
        if rgb.dtype != np.uint8:
            # Float frames are stored in [0,1] by every convention this
            # reads. Scaled rather than cast: a plain cast turns 0.87 into 0
            # and the panel shows black.
            top = float(rgb.max()) if rgb.size else 1.0
            rgb = (rgb * (255.0 if top <= 1.0 else 1.0)).clip(0, 255).astype(np.uint8)
        return rgb

    def close(self) -> None:
        with self._lock:
            try:
                self._h5.close()
            except Exception:  # noqa: S110 - closing an already-closed file
                # Deliberately silent: `close()` is idempotent by contract here
                # and `test_closing_twice_is_not_an_error` pins that. There is
                # nothing to report and nowhere useful to report it.
                pass

    # ---------- internals ----------

    def _discover_cameras(self) -> list[str]:
        group = self._group_for(self.layout.episodes[0], self.layout.images)
        if group is None:
            return []
        out = []
        for key in sorted(group.keys()):
            node = group[key]
            # A camera is a dataset with a frame axis and two spatial ones.
            # `qpos` is also a dataset under robomimic's obs group and is not
            # a camera, which is why this checks the shape rather than the
            # name.
            if hasattr(node, "shape") and len(node.shape) >= 3:
                out.append(key)
        return out

    def _group_for(self, episode_path: str, relative: str):
        node = self._h5[episode_path] if episode_path else self._h5
        for part in relative.split("/"):
            if part not in node:
                return None
            node = node[part]
        return node

    def _images_for(self, episode_path: str):
        group = self._group_for(episode_path, self.layout.images)
        if group is None or self._camera not in group:
            raise Hdf5Error(
                f"episode `{episode_path or 'root'}` has no camera "
                f"`{self._camera}`. Datasets whose episodes carry different "
                f"cameras are refused rather than silently skipped."
            )
        return group[self._camera]

    def _row(self, episode_path: str, relative: str, t: int, images_ok=True) -> list:
        if not relative:
            return []
        node = self._group_for(episode_path, relative)
        if node is None:
            return []
        # Under robomimic the state lives inside the obs group beside the
        # cameras, so `state` may resolve to the group rather than an array.
        if not hasattr(node, "shape"):
            for key in ("qpos", "state", "robot0_eef_pos", "ee_pos"):
                if key in node and hasattr(node[key], "shape"):
                    node = node[key]
                    break
            else:
                return []
        try:
            row = node[t]
        except (IndexError, ValueError):
            return []
        try:
            return [float(v) for v in row]
        except TypeError:
            return [float(row)]

    def _task_for(self, episode_path: str) -> str:
        node = self._h5[episode_path] if episode_path else self._h5
        for key in ("task", "language_instruction", "instruction"):
            if key in node.attrs:
                value = node.attrs[key]
                return value.decode() if isinstance(value, bytes) else str(value)
        return ""

    def _episode_at(self, episode: int) -> EpisodeInfo:
        eps = self.episodes()
        if not 0 <= episode < len(eps):
            raise BadRequest(f"episode {episode} not in [0,{len(eps)})")
        return eps[episode]
