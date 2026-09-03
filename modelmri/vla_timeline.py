# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""One episode, several aligned tracks, one time axis.

The robot panel shows a frame and a scrubber. That answers "what did the
camera see at t" and nothing else, and the questions people actually bring to
a recorded episode are about COINCIDENCE: the gripper closed here — what was
the state doing, did the reward move, was this frame unlike the rest of the
dataset? Every one of those needs two series on one axis, and reading them off
two panels with two x-ranges is how a coincidence gets asserted that is not
there.

So this returns several tracks sampled on THE SAME timesteps, and the sameness
is the product. `t` is the frame index within the episode, every track is
indexed by it, and a track with nothing at a timestep carries `None` there
rather than a zero.

## What a track is allowed to claim

**A unit or nothing.** LeRobot datasets publish per-feature metadata and most
of them publish no units at all. A track whose unit is unknown says `null`,
and `vla_audit` already refuses to overlay a policy's actions on a dataset's
for exactly this reason — a number with no unit is not a measurement, and two
of them on one axis is a picture of nothing.

**A missing column is ABSENT, not empty.** `next.reward` is optional in the
format and most manipulation datasets have no reward at all. An empty track
drawn at zero would say the reward was zero the whole way, which is a
measurement nobody took; a track that says "this dataset publishes no
`next.reward`" says the true thing.

**Non-finite is counted, never plotted as a value.** A NaN in a recorded
action is real data corruption and it is reported per track with its count,
rather than being interpolated away or drawn as zero.

## What it does not do

No smoothing, no interpolation, no resampling onto a prettier grid. If a
stride drops timesteps, the dropped ones are ABSENT from the series and the
stride is in the payload — a line drawn through gaps as though the frames
between it had been measured is the same lie as a zero for a missing one.

And no verdicts. This aligns series; whether two of them moving together means
anything is the reader's call, and a "correlation" field here would be a
number nobody asked for attached to a claim nobody made.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .errors import BadRequest, Refusal

#: Points per track before a stride is applied. A 3,000-frame episode drawn at
#: one point per pixel is wider than any screen, and the stride is REPORTED
#: rather than applied quietly — "the spike is not in this data" and "the spike
#: fell between two sampled frames" are different answers.
MAX_POINTS = 600

#: Columns worth a track, in the order a reader wants them. `action` and
#: `observation.state` are in every LeRobot dataset; the rest are optional in
#: the format and absent from most manipulation sets.
TRACK_COLUMNS = (
    "action",
    "observation.state",
    "next.reward",
    "next.success",
    "next.done",
)


@dataclass
class Track:
    """One series over the episode's timesteps, with what it is a series OF."""

    column: str
    #: Per-dimension series. A scalar column has one. `None` at a timestep
    #: means "not measured there" — a dropped stride point, or a non-finite
    #: value — and never zero.
    series: list[list[float | None]]
    #: What the dataset calls each dimension, when it publishes names.
    #: `None` rather than "dim 0", which would be this module inventing a
    #: label the dataset never wrote.
    names: list[str] | None
    #: The dataset's own unit for this column, or `None`. Most publish none.
    unit: str | None
    #: Min and max over what was actually read, per dimension — the axis a
    #: caller should draw against. `None` for a dimension with no finite value.
    low: list[float | None]
    high: list[float | None]
    #: Non-finite values found and excluded, per dimension. Real corruption in
    #: a recording, reported rather than interpolated away.
    n_nonfinite: list[int]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Timeline:
    episode: int
    repo_id: str
    #: The timesteps every track is indexed by. One axis, shared.
    timesteps: list[int]
    #: Seconds from the start of the episode, when the dataset records them.
    seconds: list[float] | None
    length: int
    stride: int
    #: True when `stride > 1` — some timesteps are ABSENT from every series
    #: rather than smoothed over.
    strided: bool
    tracks: list[Track]
    #: Columns asked for that this dataset does not publish, with the reason.
    #: Absent, never an empty track: a reward track drawn at zero says the
    #: reward was zero, which nobody measured.
    absent: list[dict]

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "tracks": [t.to_dict() for t in self.tracks],
            "means": self.means(),
        }

    def means(self) -> str:
        drawn = ", ".join(t.column for t in self.tracks) or "nothing"
        parts = [
            f"Episode {self.episode} of {self.repo_id}, {len(self.timesteps)} of "
            f"its {self.length} timesteps, on one axis. Tracks: {drawn}. Every "
            f"series is indexed by the SAME `t`, which is the point — two "
            f"series read off two panels with two x-ranges is how a "
            f"coincidence gets asserted that is not there."
        ]
        if self.strided:
            parts.append(
                f"SAMPLED EVERY {self.stride} FRAMES. The timesteps between "
                f"are absent from every series rather than interpolated: a "
                f"line drawn through them would claim frames nobody read. A "
                f"feature narrower than {self.stride} frames can fall entirely "
                f"between two points."
            )
        broken = sum(sum(t.n_nonfinite) for t in self.tracks)
        if broken:
            parts.append(
                f"{broken} non-finite value(s) were found and left out — that "
                f"is corruption in the recording, not a zero, and each track "
                f"says how many of its own were affected."
            )
        for gone in self.absent:
            parts.append(gone["why"])
        units = [t.column for t in self.tracks if t.unit is None]
        if units:
            parts.append(
                f"This dataset publishes no unit for {', '.join(units)}, so "
                f"those axes are in whatever the recording used. Two tracks "
                f"with no units cannot be compared to each other, only to "
                f"themselves over time."
            )
        return " ".join(parts)


def _whole(name: str, value, *, low: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BadRequest(f"`{name}` is a whole number and this was {value!r}.")
    if value < low:
        raise BadRequest(f"`{name}` is at least {low} and this was {value}.")
    return value


def _dimension_names(info: dict, column: str, width: int) -> list[str] | None:
    """The dataset's own names for a column's dimensions, or `None`.

    NOT "dim 0", "dim 1". A generated label looks exactly like a published one
    on screen, and a reader who believes the dataset named its third joint
    `dim 2` has been told something nobody wrote.
    """
    feature = ((info or {}).get("features") or {}).get(column) or {}
    names = feature.get("names")
    # LeRobot nests these one level deep on some datasets and not others.
    if isinstance(names, dict):
        names = names.get("motors") or names.get("axes") or None
    if not isinstance(names, list) or len(names) != width:
        return None
    return [str(n) for n in names]


def _unit_of(info: dict, column: str) -> str | None:
    """Whatever the dataset says this column is measured in. Usually nothing."""
    feature = ((info or {}).get("features") or {}).get(column) or {}
    for key in ("unit", "units"):
        value = feature.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _as_rows(raw, start: int, stop: int) -> list:
    """The slice of a column belonging to one episode."""
    if raw is None:
        return []
    return list(raw[start:stop])


def _widen(value) -> list:
    """A row as a list, whether the column is scalar or vector."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def tracks(
    reader,
    episode: int,
    *,
    columns: tuple[str, ...] = TRACK_COLUMNS,
    max_points: int = MAX_POINTS,
) -> Timeline:
    """Several aligned series for one episode, on one shared time axis."""
    episode = _whole("episode", episode, low=0)
    max_points = _whole("max_points", max_points, low=1)

    episodes = reader.episodes()
    match = next((e for e in episodes if e.index == episode), None)
    if match is None:
        raise BadRequest(
            f"episode {episode} is not in this dataset — it has "
            f"{len(episodes)}, numbered from {episodes[0].index if episodes else 0}."
        )
    length = int(match.length)
    if length <= 0:
        raise Refusal(
            f"episode {episode} has no frames, so there is no axis to draw "
            f"anything against."
        )

    table = reader._frame_table()
    start = int(getattr(match, "data_from", 0))
    stop = start + length

    stride = max(1, math.ceil(length / max_points))
    steps = list(range(0, length, stride))

    stamps = _as_rows(table.get("timestamp"), start, stop)
    seconds = None
    if stamps:
        picked = [stamps[t] for t in steps if t < len(stamps)]
        if len(picked) == len(steps) and all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in picked
        ):
            # Relative to the episode's own first frame. Absolute timestamps in
            # a LeRobot dataset are seconds into the CONCATENATED file, so a
            # reader shown them would see episode 40 start at 700 seconds.
            base = float(picked[0])
            seconds = [round(float(v) - base, 6) for v in picked]

    built: list[Track] = []
    absent: list[dict] = []
    for column in columns:
        raw = table.get(column)
        if raw is None:
            absent.append(
                {
                    "column": column,
                    "why": (
                        f"This dataset publishes no `{column}`, so there is no "
                        f"track for it. That is a fact about the recording — "
                        f"drawing an empty one at zero would say the value WAS "
                        f"zero, which nobody measured."
                    ),
                }
            )
            continue
        rows = _as_rows(raw, start, stop)
        if not rows:
            absent.append(
                {
                    "column": column,
                    "why": (
                        f"`{column}` exists in this dataset but carries no rows "
                        f"for episode {episode}."
                    ),
                }
            )
            continue

        width = max((len(_widen(r)) for r in rows[:64]), default=0)
        if width <= 0:
            continue
        series: list[list[float | None]] = [[] for _ in range(width)]
        broken = [0] * width
        for t in steps:
            row = _widen(rows[t]) if t < len(rows) else []
            for d in range(width):
                value = row[d] if d < len(row) else None
                if value is None or isinstance(value, bool):
                    # A bool IS a real value for `next.done`, so it is widened
                    # to 0/1 rather than dropped — but only where the column is
                    # actually boolean, never as a silent int coercion.
                    series[d].append(None if value is None else float(bool(value)))
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    broken[d] += 1
                    series[d].append(None)
                    continue
                if not math.isfinite(number):
                    broken[d] += 1
                    series[d].append(None)
                    continue
                series[d].append(round(number, 6))

        low: list[float | None] = []
        high: list[float | None] = []
        for d in range(width):
            finite = [v for v in series[d] if v is not None]
            low.append(min(finite) if finite else None)
            high.append(max(finite) if finite else None)

        built.append(
            Track(
                column=column,
                series=series,
                names=_dimension_names(reader.info, column, width),
                unit=_unit_of(reader.info, column),
                low=low,
                high=high,
                n_nonfinite=broken,
            )
        )

    if not built:
        raise Refusal(
            f"none of {', '.join(columns)} is in this dataset, so there is "
            f"nothing to align. `GET /api/vla/episodes` lists what it does "
            f"carry."
        )

    return Timeline(
        episode=episode,
        repo_id=getattr(reader, "repo_id", ""),
        timesteps=steps,
        seconds=seconds,
        length=length,
        stride=stride,
        strided=stride > 1,
        tracks=built,
        absent=absent,
    )
