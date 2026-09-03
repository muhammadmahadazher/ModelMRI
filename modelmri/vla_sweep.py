# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""One measurement, over every episode, so you can find the frame worth looking at.

The VLA panel shows one episode at a time. That is fine for looking and
useless for finding: a dataset has two hundred episodes and the frame you
want is in one of them, and clicking through is not a method.

So this runs a chosen measurement over a strided sample of every episode and
ranks the result. RoboLab's dashboard does cross-episode ranking over
simulator-emitted event labels and Event-SAE clusters kinematic keyframes;
ranking over MEASURED INTERNALS is the version nobody else can do, because
nobody else holds the policy and the dataset in one process.

THE SWEEP ADDS ITERATION, NOT MEASUREMENT. Every metric here is a callable
that already exists — the attention path and `vla_occlude` — so a number in
this table and the same number in the panel come from the same code. A sweep
with its own copy of a metric is a second implementation that drifts.

TWO THINGS IT WILL NOT DO
-------------------------
NO FAILURE-MODE CLUSTERING WITH NAMES. A cluster labelled "dropped the object"
that ModelMRI never verified is exactly the fabrication this project refuses.
The ranking is by one stated measured quantity and the output says which,
every time.

NO SILENT STRIDE. A strided ranking may miss the worst frame entirely — that
is not a caveat to bury, it is the first thing a reader needs, because the
whole point of a ranking is that the top of it is the worst thing there is.
Every row and every summary carries the stride.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import asdict, dataclass, field

from . import fmt, paths
from .errors import BadRequest

# Cap on frames per sweep. Every frame is at least one tower pass and the
# occlusion metric is dozens, so this is what decides whether a sweep is a
# coffee break or an afternoon. REFUSED past it with the number, never
# truncated: a ranking silently missing its tail looks exactly like a ranking.
MAX_FRAMES = 5_000

# How many undecodable frames are LISTED. The full count travels beside them as
# `Sweep.n_failed` — the list is a sample to look at, not the measurement, and
# reporting its length as the measurement is how a 600-frame failure was
# published as twenty.
MAX_FAILED_LISTED = 20

# Default steps. Episodes and frames stride independently: you usually want
# every episode and a few frames of each, not the reverse.
DEFAULT_EPISODE_STRIDE = 1
DEFAULT_FRAME_STRIDE = 25

# TWO TABLES, BECAUSE A ROW IS NOT A SWEEP. `vla_sweep` holds the
# measurements; `vla_sweep_run` holds what they ARE — the unit they were
# measured in, the two strides that say which frames were never looked at, and
# the totals every coverage sentence is computed against. None of that is
# derivable from the rows: a column of floats does not know it is nats over the
# patch grid, and `robot_export.Timeline` refuses a track with no unit for
# exactly the reason nobody may supply one from memory.
#
# The run table is keyed by the same four columns the row key starts with, so
# one run record describes one set of rows, and `save()` writes both together.
#
# MIGRATING A LIVE DATABASE. Every statement here is `IF NOT EXISTS` and the
# whole script runs on every `_db()`, against the same file this project has
# been writing rows into since the sweep shipped. An install that already holds
# rows gains an EMPTY `vla_sweep_run` and keeps working — `stored()` reads the
# row table alone and did not change. What those older rows do not gain is a
# run record, and `retrieve()` refuses them by name rather than reconstructing
# one; see the refusal there for why a defaulted unit is the single thing this
# table must never hand out.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS vla_sweep (
    dataset   TEXT NOT NULL,
    policy    TEXT NOT NULL,
    metric    TEXT NOT NULL,
    camera    TEXT NOT NULL,
    episode   INTEGER NOT NULL,
    timestep  INTEGER NOT NULL,
    value     REAL NOT NULL,
    stride    INTEGER NOT NULL,
    taken_at  TEXT NOT NULL,
    PRIMARY KEY (dataset, policy, metric, camera, episode, timestep)
);
CREATE INDEX IF NOT EXISTS vla_sweep_rank
    ON vla_sweep (dataset, policy, metric, value DESC);
CREATE TABLE IF NOT EXISTS vla_sweep_run (
    dataset        TEXT NOT NULL,
    policy         TEXT NOT NULL,
    metric         TEXT NOT NULL,
    camera         TEXT NOT NULL,
    unit           TEXT NOT NULL,
    episode_stride INTEGER NOT NULL,
    frame_stride   INTEGER NOT NULL,
    n_frames       INTEGER NOT NULL,
    n_episodes     INTEGER NOT NULL,
    frames_total   INTEGER NOT NULL,
    seconds        REAL NOT NULL,
    n_failed       INTEGER NOT NULL,
    failed         TEXT NOT NULL,
    taken_at       TEXT NOT NULL,
    PRIMARY KEY (dataset, policy, metric, camera)
);
"""


class SweepError(BadRequest):
    """This sweep cannot be run honestly, and we say why."""


@dataclass
class Row:
    episode: int
    timestep: int
    value: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Sweep:
    metric: str
    unit: str
    dataset: str
    #: `None` when no policy was resident, never `""`. The sibling field on
    #: `/api/vla` says `null` for the same fact, and a sweep read back out of
    #: sqlite months later has only this to say what produced it.
    policy: str | None
    camera: str
    episode_stride: int
    frame_stride: int
    rows: list[Row] = field(default_factory=list)
    n_frames: int = 0
    n_episodes: int = 0
    frames_total: int = 0
    seconds: float = 0.0
    #: A SAMPLE of what could not be measured, capped at `MAX_FAILED_LISTED`.
    failed: list[dict] = field(default_factory=list)
    #: How many failed in total. Separate from `len(failed)` because the list
    #: is truncated and the sentence below was counting the truncated list:
    #: measured with PyAV absent over six episodes of a hundred frames, every
    #: one of the 600 failed and the report said "20 frame(s) could not be
    #: measured". The true figure was not derivable from the payload at all.
    n_failed: int = 0

    def to_dict(self) -> dict:
        out = asdict(self)
        out["means"] = self.means()
        return out

    def means(self) -> str:
        share = (
            f"{self.n_frames} of {self.frames_total} frames "
            f"({100.0 * self.n_frames / self.frames_total:.1f}%)"
            if self.frames_total
            else f"{self.n_frames} frames"
        )
        parts = [
            f"{share} across {self.n_episodes} episodes, measured by "
            f"{self.metric.upper()} and ranked by that and nothing else. "
            f"The unit is {self.unit}."
        ]
        parts.append(
            f"THE STRIDE IS {self.frame_stride} FRAMES (and every "
            f"{self.episode_stride} episode). A strided ranking can miss the "
            f"worst frame entirely — the top of this list is the worst frame "
            f"THAT WAS SAMPLED, which is not the same claim. Drop the stride "
            f"to narrow the gap, and it costs proportionally."
        )
        if self.rows:
            top = self.rows[0]
            parts.append(
                f"The highest is episode {top.episode} at timestep "
                f"{top.timestep} ({fmt.measured(top.value, 4)}). That is a ranking by "
                f"{self.metric}, not a diagnosis: nothing here has verified "
                f"what happens in that frame, and no failure mode has been "
                f"named for it."
            )
        if self.n_failed:
            listed = (
                ""
                if self.n_failed <= len(self.failed)
                else f" ({len(self.failed)} of them listed below)"
            )
            parts.append(
                f"{self.n_failed} frame(s) could not be measured and are "
                f"ABSENT from the ranking rather than scored zero — a frame "
                f"that failed to decode is not a frame with a low score"
                f"{listed}."
            )
        return " ".join(parts)


# ------------------------------------------------------------- the metrics


def attention_entropy(handle, rgb, **_) -> float:
    """How spread out the tower's attention is on this frame, in nats.

    Low entropy means the model concentrated on a few patches; high means it
    spread across the image. NOT a quality score in either direction — a
    policy staring at one patch may be locked onto the object or may be stuck
    — which is why this ranks and does not judge.
    """
    handle.analyse(rgb, ("sweep", 0, 0))
    # The LAST layer that was actually captured, not `status().n_layers - 1`:
    # the two agree on every model this has seen and the capture is the thing
    # `attention()` indexes into, so one of them is the truth and the other is
    # a config field that happens to match.
    layers = len(handle._attn)
    if not layers:
        raise SweepError("this frame produced no attention maps")
    # THE RAW ATTENTION, not `attention()["heat"]`. That heatmap is min-max
    # normalised to [0,1] for display, which subtracts the frame's own
    # minimum -- and an entropy computed after subtracting a constant is not
    # the entropy of the distribution. It drives the least-attended patch to
    # exactly zero probability and makes every frame look more concentrated
    # than it was, by an amount that depends on that frame's minimum. Since
    # this metric exists to RANK frames against each other, a per-frame
    # distortion is the one error it cannot absorb.
    #
    # Worse than a distortion at the top end: a perfectly UNIFORM map is the
    # most spread-out frame there is, and min-max normalising it gives all
    # zeros -- so the sweep raised "this frame produced an empty attention
    # map" for the maximum-entropy case.
    flat = [
        max(0.0, float(v))
        for row in handle.attention(layers - 1, -1)["values"]
        for v in row
    ]
    total = sum(flat)
    if total <= 0:
        raise SweepError("this frame produced an empty attention map")
    entropy = 0.0
    for v in flat:
        p = v / total
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


def occlusion_peak(handle, rgb, *, scale_frames, stride=0, **_) -> float:
    """The strongest single-block causal shift on this frame.

    The expensive one — dozens of tower passes per frame — and the reason the
    cost estimate exists. Perception only, like everything `vla_occlude`
    reports: a shift in an embedding, never an effect on an action.
    """
    from . import vla_occlude

    out = handle.occlude(rgb, scale_frames, stride=stride or vla_occlude.DEFAULT_STRIDE)
    blocks = out.get("blocks") or []
    if not blocks:
        raise SweepError("this frame produced no occlusion blocks")
    return max(float(b["shift"]) for b in blocks)


METRICS = {
    "attention_entropy": (attention_entropy, "nats over the patch grid"),
    "occlusion_peak": (
        occlusion_peak,
        "the tower's own embedding spread, perception only",
    ),
}


# ------------------------------------------------------------- the planning


def plan(
    reader,
    *,
    episode_stride: int = DEFAULT_EPISODE_STRIDE,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    max_frames: int = MAX_FRAMES,
) -> tuple[list[tuple[int, int]], int]:
    """Which (episode, timestep) pairs get measured, and how many exist.

    Returns the pairs and the TOTAL frame count, so the caller can say what
    share of the dataset a ranking covers rather than implying it covers all
    of it.
    """
    if episode_stride < 1 or frame_stride < 1:
        raise SweepError("both strides must be at least 1")
    episodes = reader.episodes()
    if not episodes:
        raise SweepError("this dataset has no episodes to sweep")

    pairs: list[tuple[int, int]] = []
    total = 0
    for ep in episodes:
        total += int(ep.length)
    for ep in episodes[::episode_stride]:
        for t in range(0, int(ep.length), frame_stride):
            pairs.append((ep.index, t))
    if len(pairs) > max_frames:
        raise SweepError(
            f"that is {len(pairs):,} frames and the cap is {max_frames:,}. "
            f"Raise the stride rather than having the sweep cut short — a "
            f"ranking missing its tail looks exactly like a ranking, and you "
            f"would have no way to tell."
        )
    return pairs, total


def estimate(
    reader,
    metric: str,
    *,
    episode_stride: int = DEFAULT_EPISODE_STRIDE,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    grid: list[int] | None = None,
    occlusion_stride: int = 0,
    seconds_per_pass: float | None = None,
) -> dict:
    """Frames and passes before the run, and seconds only if measured.

    `seconds_per_pass` comes from the caller's own machine — a warmup time or
    a previous run. When it is absent the estimate reports passes and NO
    seconds, because a duration guessed from somebody else's hardware is the
    kind of number people plan around and should not.
    """
    if metric not in METRICS:
        raise SweepError(
            f"unknown metric {metric!r} — expected one of {sorted(METRICS)}"
        )
    pairs, total = plan(
        reader, episode_stride=episode_stride, frame_stride=frame_stride
    )
    if metric == "occlusion_peak":
        from . import vla_occlude

        per_frame = vla_occlude.estimate(
            grid or [32, 32], occlusion_stride or vla_occlude.DEFAULT_STRIDE
        )["passes"]
    else:
        per_frame = 1
    passes = len(pairs) * per_frame
    out = {
        "metric": metric,
        "frames": len(pairs),
        "frames_total": total,
        "passes_per_frame": per_frame,
        "passes": passes,
        "episode_stride": episode_stride,
        "frame_stride": frame_stride,
        "coverage": round(len(pairs) / total, 4) if total else 0.0,
    }
    if seconds_per_pass:
        out["seconds"] = round(passes * float(seconds_per_pass), 1)
        out["seconds_from"] = "measured on this machine"
    else:
        out["seconds"] = None
        out["seconds_from"] = (
            "not estimated — this machine has not been timed yet, and a "
            "duration from somebody else's hardware is a number people plan "
            "around"
        )
    return out


# ------------------------------------------------------------------ the run


def run(
    handle,
    reader,
    metric: str,
    *,
    episode_stride: int = DEFAULT_EPISODE_STRIDE,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    occlusion_stride: int = 0,
    max_frames: int = MAX_FRAMES,
    on_progress=None,
    should_stop=None,
) -> Sweep:
    """Measure every sampled frame and rank the result.

    `should_stop()` is polled between frames so a sweep can be cancelled
    without killing the process — a partial sweep is still a ranking over what
    it covered, and it says how much that was.
    """
    if metric not in METRICS:
        raise SweepError(
            f"unknown metric {metric!r} — expected one of {sorted(METRICS)}"
        )
    fn, unit = METRICS[metric]
    pairs, total = plan(
        reader,
        episode_stride=episode_stride,
        frame_stride=frame_stride,
        max_frames=max_frames,
    )

    started = time.perf_counter()
    scale_frames: list = []
    if metric == "occlusion_peak":
        from . import vla_occlude

        # The scale is set ONCE from the first episode and reused, so every
        # row is in the same unit. Re-deriving it per episode would make the
        # ranking a comparison of numbers measured against different rulers.
        first = reader.episodes()[0]
        step = max(1, int(first.length) // vla_occlude.SCALE_FRAMES)
        scale_frames = [
            reader.raw_frame(first.index, t) for t in range(0, int(first.length), step)
        ][: vla_occlude.SCALE_FRAMES]

    rows: list[Row] = []
    failed: list[dict] = []
    seen_episodes = set()
    for index, (episode, timestep) in enumerate(pairs):
        if should_stop and should_stop():
            break
        if on_progress:
            on_progress(index, len(pairs), episode, timestep)
        try:
            rgb = reader.raw_frame(episode, timestep)
            value = fn(
                handle,
                rgb,
                scale_frames=scale_frames,
                stride=occlusion_stride,
            )
        except ImportError:
            # NOT a per-frame failure, so it does not get the per-frame
            # handling below. `av` is imported the first time a frame is
            # actually decoded, so a machine without it fails EVERY frame for
            # the same reason — and this handler turned that into `rows: []`
            # with 200 OK and a `failed` table of `why: ModuleNotFoundError`.
            # MEASURED on this machine (pyarrow present, av absent):
            # `POST /api/vla/sweep {"frame_stride": 1e12}` came back 200 with
            # 0 rows and a summary sentence reading "0 of 25650 frames (0.0%)
            # across 0 episodes, measured by ATTENTION_ENTROPY" — a completed
            # measurement of nothing.
            #
            # The route above has carried the sentence naming the missing
            # package all along (`_missing_reader_dep`, 409); it was simply
            # unreachable from inside this loop.
            raise
        except Exception as err:
            # ABSENT from the ranking, not scored zero. A frame that failed to
            # decode is not a frame with a low score, and a zero would sit at
            # the bottom of the table looking like a measurement.
            failed.append(
                {
                    "episode": episode,
                    "timestep": timestep,
                    "why": type(err).__name__,
                }
            )
            continue
        if not math.isfinite(value):
            failed.append(
                {"episode": episode, "timestep": timestep, "why": "non-finite"}
            )
            continue
        rows.append(Row(episode=episode, timestep=timestep, value=round(value, 6)))
        seen_episodes.add(episode)

    rows.sort(key=lambda r: -r.value)
    return Sweep(
        metric=metric,
        unit=unit,
        dataset=getattr(reader, "repo_id", ""),
        policy=getattr(handle.status(), "repo", None) or None,
        camera=getattr(reader, "camera", ""),
        episode_stride=episode_stride,
        frame_stride=frame_stride,
        rows=rows,
        n_frames=len(rows),
        n_episodes=len(seen_episodes),
        frames_total=total,
        seconds=round(time.perf_counter() - started, 2),
        failed=failed[:MAX_FAILED_LISTED],
        n_failed=len(failed),
    )


# ---------------------------------------------------------------- storage


def _db() -> sqlite3.Connection:
    path = paths.trace_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path))
    db.executescript(_SCHEMA)
    return db


def save(sweep: Sweep) -> int:
    """Persist a sweep beside the traces, so it survives the process.

    The stride is stored PER ROW rather than once for the sweep: two runs at
    different strides land in the same table, and a row that did not carry its
    own stride would be indistinguishable from one taken at a finer step.

    The RUN-LEVEL facts — the unit, the episode stride, the counts, the
    duration and the failure sample — go to `vla_sweep_run` in the same
    transaction, because rows with no run record beside them are the state
    `retrieve()` has to refuse. Written together or not at all: a commit that
    stored four thousand rows and lost the unit would produce a ranking nobody
    can export and nobody can label, which is a worse outcome than storing
    nothing.
    """
    from datetime import datetime, timezone

    taken = datetime.now(timezone.utc).isoformat()
    db = _db()
    try:
        db.execute(
            "INSERT OR REPLACE INTO vla_sweep_run "
            "(dataset, policy, metric, camera, unit, episode_stride, "
            " frame_stride, n_frames, n_episodes, frames_total, seconds, "
            " n_failed, failed, taken_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                sweep.dataset,
                # `""` in the COLUMN for the same reason as the rows below: it
                # is half the lookup key, and a NULL there would not match the
                # `""` a caller sends. `retrieve()` turns it back into `None`,
                # which is lossless because `Sweep.policy` is documented as
                # never being `""`.
                sweep.policy or "",
                sweep.metric,
                sweep.camera,
                sweep.unit,
                sweep.episode_stride,
                sweep.frame_stride,
                sweep.n_frames,
                sweep.n_episodes,
                sweep.frames_total,
                sweep.seconds,
                sweep.n_failed,
                # The SAMPLE, stored as the cap left it, with `n_failed` beside
                # it as the real count. The two are separate columns for the
                # same reason the dataclass keeps them separate: a report that
                # counted the truncated list published 600 failures as 20.
                json.dumps(sweep.failed),
                taken,
            ),
        )
        db.executemany(
            "INSERT OR REPLACE INTO vla_sweep "
            "(dataset, policy, metric, camera, episode, timestep, value, "
            " stride, taken_at) VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (
                    sweep.dataset,
                    # `""` in the COLUMN, not in the payload. This is half of
                    # the lookup key `stored()` queries on, and a NULL there
                    # would not match the `""` a caller sends.
                    sweep.policy or "",
                    sweep.metric,
                    sweep.camera,
                    r.episode,
                    r.timestep,
                    r.value,
                    sweep.frame_stride,
                    taken,
                )
                for r in sweep.rows
            ],
        )
        db.commit()
    finally:
        db.close()
    return len(sweep.rows)


def stored(dataset: str, policy: str, metric: str, *, limit: int = 200) -> list[dict]:
    """The strongest stored rows for this dataset, policy and metric."""
    db = _db()
    try:
        cur = db.execute(
            "SELECT episode, timestep, value, stride, taken_at FROM vla_sweep "
            "WHERE dataset=? AND policy=? AND metric=? "
            "ORDER BY value DESC LIMIT ?",
            (dataset, policy, metric, int(limit)),
        )
        return [
            {
                "episode": r[0],
                "timestep": r[1],
                "value": r[2],
                # Carried out of the table, not dropped: a ranking whose rows
                # were taken at different strides is not one ranking.
                "stride": r[3],
                "taken_at": r[4],
            }
            for r in cur.fetchall()
        ]
    finally:
        db.close()


def _listed(values: list[str]) -> str:
    """`a, b and c` for a sentence, rather than a Python list repr."""
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + f" and {values[-1]}"


def retrieve(
    dataset: str, policy: str, metric: str, *, camera: str | None = None
) -> Sweep:
    """The stored sweep, whole — or a refusal naming what is missing.

    `stored()` answers "which rows are strongest"; this answers "what was the
    sweep", and they are not the same question. A `Sweep` carries the unit its
    numbers are in, the two strides that say which frames were never opened,
    and the totals a coverage claim is computed against — facts that live in
    `vla_sweep_run` and are NOT recoverable from a column of floats. Every one
    of them is required by `robot_export.timeline_from_sweep`, which is why
    this function exists: without it a route exporting a stored sweep would
    have to invent a unit, and a number in Foxglove under an invented unit is
    indistinguishable from one that was measured in it.

    NOTHING IS DEFAULTED HERE. Rows saved before this database grew a run
    table are refused by name — they are not a sweep in nats with a frame
    total of zero — and so is a row set that two runs have superimposed. The
    refusals are the deliverable as much as the return value is.

    `camera=None` means the caller did not name one. With a single stored
    camera that is unambiguous and it is resolved; with two it is refused
    rather than picked, because a `Sweep` names ONE camera and two cameras
    watched the same episodes through different lenses.
    """
    if metric not in METRICS:
        raise SweepError(
            f"unknown metric {metric!r} — expected one of {sorted(METRICS)}. "
            f"Nothing was ever measured under that name, so there is no stored "
            f"sweep to read back."
        )
    key = (dataset, policy or "", metric)
    columns = (
        "camera, unit, episode_stride, frame_stride, n_frames, n_episodes, "
        "frames_total, seconds, n_failed, failed, taken_at"
    )
    db = _db()
    try:
        if camera is None:
            runs = db.execute(
                f"SELECT {columns} FROM vla_sweep_run "
                "WHERE dataset=? AND policy=? AND metric=? ORDER BY camera",
                key,
            ).fetchall()
        else:
            runs = db.execute(
                f"SELECT {columns} FROM vla_sweep_run "
                "WHERE dataset=? AND policy=? AND metric=? AND camera=?",
                (*key, camera),
            ).fetchall()

        if len(runs) > 1:
            names = _listed([repr(r[0]) for r in runs])
            raise SweepError(
                f"{len(runs)} sweeps of {metric} are stored for {dataset}, one "
                f"per camera ({names}), and a sweep names ONE camera. Taking "
                f"the first would rank frames through a lens the reader did "
                f"not ask for, and nothing downstream would say which. Name "
                f"the camera."
            )

        if not runs:
            raise _nothing_stored(db, key, metric, dataset, camera)

        (
            camera_name,
            unit,
            episode_stride,
            frame_stride,
            n_frames,
            n_episodes,
            frames_total,
            seconds,
            n_failed,
            failed_json,
            _taken_at,
        ) = runs[0]

        # Filtered by CAMERA as well, which `stored()` is not: that function
        # answers a ranking query and carries each row's own stride out so a
        # reader can see a mixture, while this one is building a single object
        # that states one camera and one stride for all of its rows.
        rows = db.execute(
            "SELECT episode, timestep, value, stride FROM vla_sweep "
            "WHERE dataset=? AND policy=? AND metric=? AND camera=? "
            "ORDER BY value DESC",
            (*key, camera_name),
        ).fetchall()
    finally:
        db.close()

    n_frames = int(n_frames)
    n_failed = int(n_failed)
    if not rows:
        why = (
            f"every one of its {n_failed:,} sampled frame(s) failed to measure"
            if n_failed
            else "it measured no frames at all"
        )
        raise SweepError(
            f"the stored sweep of {metric} on {dataset} has no rows: {why}. "
            f"There is nothing to rank and nothing to export — a timeline with "
            f"no samples in it reads as a policy that produced no "
            f"measurements, which is a different claim from one nobody could "
            f"take. Fix what stopped the frames decoding (`pip install "
            f"modelmri[vla]` installs the reader's `av` and `pyarrow`) and run "
            f"the sweep again."
        )

    strides = sorted({int(r[3]) for r in rows})
    if strides != [int(frame_stride)]:
        found = _listed([f"{s}" for s in strides])
        raise SweepError(
            f"the stored rows of {metric} on {dataset} were taken at "
            f"{len(strides)} different frame stride(s) ({found}), and the run "
            f"record beside them says {frame_stride}. Rows are keyed by "
            f"(episode, timestep), so a second run at a coarser step REPLACES "
            f"the frames it re-measured and leaves the finer run's extra "
            f"frames sitting under the same keys — the two samplings are "
            f"superimposed in the table, and handing them back as one Sweep "
            f"would publish points from two runs under a single stated stride. "
            f"`stored()` shows you every row with the stride it was taken at; "
            f"`forget({dataset!r}, {policy!r}, {metric!r}, "
            f"camera={camera_name!r})` drops this key so the next run is the "
            f"only one in it."
        )

    if len(rows) != n_frames:
        raise SweepError(
            f"the run record for {metric} on {dataset} counted {n_frames:,} "
            f"measured frame(s) and the table holds {len(rows):,} row(s) under "
            f"the same dataset, policy, metric and camera. At one stride that "
            f"is an earlier run at a different EPISODE stride still sitting in "
            f"the table, and its frames would be counted into this sweep's "
            f"coverage as though this run had opened them. Nothing here will "
            f"decide which rows belong to which run: "
            f"`forget({dataset!r}, {policy!r}, {metric!r}, "
            f"camera={camera_name!r})` drops the key, and the next sweep is "
            f"the only one under it."
        )

    try:
        failed = json.loads(failed_json)
        if not isinstance(failed, list):
            raise ValueError("the failure sample is not a list")
    except ValueError as err:
        raise SweepError(
            f"the run record for {metric} on {dataset} carries a failure "
            f"sample that is not readable, so nothing here can say which "
            f"frames were absent from the ranking rather than scored low. An "
            f"empty list in its place would claim every frame measured "
            f"cleanly. Re-run the sweep to rewrite the record."
        ) from err

    return Sweep(
        metric=metric,
        unit=unit,
        dataset=dataset,
        # `""` back to `None` — the encoding `save()` writes, reversed. The
        # dataclass documents this field as never `""`, and
        # `robot_export.Provenance` writes "no policy was resident for this
        # measurement" for the `None`; an empty string there would print as a
        # blank a reader takes for an oversight.
        policy=(policy or "") or None,
        camera=camera_name,
        episode_stride=int(episode_stride),
        frame_stride=int(frame_stride),
        rows=[Row(episode=r[0], timestep=r[1], value=r[2]) for r in rows],
        n_frames=n_frames,
        n_episodes=int(n_episodes),
        frames_total=int(frames_total),
        seconds=float(seconds),
        failed=failed,
        n_failed=n_failed,
    )


def _nothing_stored(
    db: sqlite3.Connection,
    key: tuple[str, str, str],
    metric: str,
    dataset: str,
    camera: str | None,
) -> SweepError:
    """Two different absences, told apart, because the fix differs.

    "You have not run this sweep" is a different sentence from "you ran it
    before this database recorded what a sweep IS", and only the second one is
    a migration. Collapsing them would tell somebody with four thousand
    measured rows on disk that they had never measured anything.
    """
    if camera is None:
        orphans = db.execute(
            "SELECT COUNT(*), MIN(stride), MAX(stride), MIN(taken_at), "
            "MAX(taken_at) FROM vla_sweep "
            "WHERE dataset=? AND policy=? AND metric=?",
            key,
        ).fetchone()
        which = ""
    else:
        orphans = db.execute(
            "SELECT COUNT(*), MIN(stride), MAX(stride), MIN(taken_at), "
            "MAX(taken_at) FROM vla_sweep "
            "WHERE dataset=? AND policy=? AND metric=? AND camera=?",
            (*key, camera),
        ).fetchone()
        which = f" through {camera}"

    count = int(orphans[0] or 0)
    if not count:
        return SweepError(
            f"no sweep of {metric} is stored for {dataset}{which} under policy "
            f"{key[1] or '(none resident)'}. Run one — `POST /api/vla/sweep` "
            f"measures it and saves it — or check the three keys: a sweep is "
            f"stored against the dataset, the policy that was resident and the "
            f"metric, and all three have to match what was run."
        )

    when = (
        f"taken {orphans[3]}"
        if orphans[3] == orphans[4]
        else f"taken between {orphans[3]} and {orphans[4]}"
    )
    return SweepError(
        f"{count:,} measured row(s) of {metric} on {dataset}{which} are stored "
        f"({when}), and nothing says what they ARE: they were saved before "
        f"this database kept a run record beside the rows, so the unit they "
        f"are in, the episode stride and the frame total their coverage is "
        f"measured against are not in it. The rows are intact and `stored()` "
        f"still returns them with their own per-row stride. What cannot happen "
        f"is a Sweep built from them: the unit would have to come from this "
        f"code rather than from the run, and a number exported into another "
        f"tool's timeline under a unit ModelMRI supplied from memory is "
        f"indistinguishable from one that was measured in it. Re-run the "
        f"sweep — the rows are replaced in place and the run record is written "
        f"beside them."
    )


def forget(
    dataset: str, policy: str, metric: str, *, camera: str | None = None
) -> dict:
    """Drop one stored sweep — its rows and its run record — and say how much.

    The remedy `retrieve()` names when two runs are superimposed under one set
    of keys. It is never automatic: `save()` has always been INSERT OR REPLACE
    and leaves behind whatever it did not overwrite, and this module does not
    delete a reader's measurements to make its own invariant hold. The counts
    come back so the caller can report what actually went rather than assuming
    something did.
    """
    key = (dataset, policy or "", metric)
    db = _db()
    try:
        if camera is None:
            rows = db.execute(
                "DELETE FROM vla_sweep WHERE dataset=? AND policy=? AND metric=?",
                key,
            ).rowcount
            runs = db.execute(
                "DELETE FROM vla_sweep_run WHERE dataset=? AND policy=? AND metric=?",
                key,
            ).rowcount
        else:
            rows = db.execute(
                "DELETE FROM vla_sweep "
                "WHERE dataset=? AND policy=? AND metric=? AND camera=?",
                (*key, camera),
            ).rowcount
            runs = db.execute(
                "DELETE FROM vla_sweep_run "
                "WHERE dataset=? AND policy=? AND metric=? AND camera=?",
                (*key, camera),
            ).rowcount
        db.commit()
    finally:
        db.close()
    # `max(0, ...)`: sqlite3 reports -1 for a statement whose row count it did
    # not track, and a -1 rendered as "removed -1 rows" is worse than useless.
    return {"rows": max(0, rows), "runs": max(0, runs)}


def heat_strip(sweep: Sweep) -> dict:
    """Episodes down, sampled time across — the shape the panel draws.

    Ragged by construction: episodes have different lengths, so the rows have
    different widths and the strip says so rather than padding to a rectangle
    with zeros that would read as measured lows.
    """
    by_episode: dict[int, list] = {}
    for row in sweep.rows:
        by_episode.setdefault(row.episode, []).append((row.timestep, row.value))
    strip = []
    for episode in sorted(by_episode):
        cells = sorted(by_episode[episode])
        strip.append(
            {
                "episode": episode,
                "timesteps": [t for t, _ in cells],
                "values": [v for _, v in cells],
            }
        )
    values = [r.value for r in sweep.rows]
    return {
        "rows": strip,
        "metric": sweep.metric,
        "unit": sweep.unit,
        "frame_stride": sweep.frame_stride,
        # `None`, not 0.0. These are the measured RANGE of the metric, and
        # with no rows nothing measured anything — "0.0 to 0.0" reads as a
        # flat result in nats over the patch grid, which is a finding, when
        # the truth is that every sampled frame failed to decode. MEASURED
        # with `av` absent: rows [], n_frames 0, six entries in `failed`, and
        # a strip claiming a range.
        "low": min(values) if values else None,
        "high": max(values) if values else None,
        # Named so the panel cannot pad. See the docstring.
        "ragged": len({len(r["values"]) for r in strip}) > 1,
    }
