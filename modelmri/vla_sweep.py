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

from . import paths
from .errors import BadRequest

# Cap on frames per sweep. Every frame is at least one tower pass and the
# occlusion metric is dozens, so this is what decides whether a sweep is a
# coffee break or an afternoon. REFUSED past it with the number, never
# truncated: a ranking silently missing its tail looks exactly like a ranking.
MAX_FRAMES = 5_000

# Default steps. Episodes and frames stride independently: you usually want
# every episode and a few frames of each, not the reverse.
DEFAULT_EPISODE_STRIDE = 1
DEFAULT_FRAME_STRIDE = 25

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
    policy: str
    camera: str
    episode_stride: int
    frame_stride: int
    rows: list[Row] = field(default_factory=list)
    n_frames: int = 0
    n_episodes: int = 0
    frames_total: int = 0
    seconds: float = 0.0
    failed: list[dict] = field(default_factory=list)

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
                f"{top.timestep} ({top.value:.4f}). That is a ranking by "
                f"{self.metric}, not a diagnosis: nothing here has verified "
                f"what happens in that frame, and no failure mode has been "
                f"named for it."
            )
        if self.failed:
            parts.append(
                f"{len(self.failed)} frame(s) could not be measured and are "
                f"ABSENT from the ranking rather than scored zero — a frame "
                f"that failed to decode is not a frame with a low score."
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
    heat = handle.attention(layers - 1, -1)["heat"]
    flat = [max(0.0, float(v)) for row in heat for v in row]
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
        raise SweepError(f"unknown metric {metric!r} — expected one of {sorted(METRICS)}")
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
            reader.raw_frame(first.index, t)
            for t in range(0, int(first.length), step)
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
        except Exception as err:  # noqa: BLE001 - recorded, never scored
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
        policy=getattr(handle.status(), "repo", "") or "",
        camera=getattr(reader, "camera", ""),
        episode_stride=episode_stride,
        frame_stride=frame_stride,
        rows=rows,
        n_frames=len(rows),
        n_episodes=len(seen_episodes),
        frames_total=total,
        seconds=round(time.perf_counter() - started, 2),
        failed=failed[:20],
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
    """
    from datetime import datetime, timezone

    taken = datetime.now(timezone.utc).isoformat()
    db = _db()
    try:
        db.executemany(
            "INSERT OR REPLACE INTO vla_sweep "
            "(dataset, policy, metric, camera, episode, timestep, value, "
            " stride, taken_at) VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (
                    sweep.dataset,
                    sweep.policy,
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
        "low": min(values) if values else 0.0,
        "high": max(values) if values else 0.0,
        # Named so the panel cannot pad. See the docstring.
        "ragged": len({len(r["values"]) for r in strip}) > 1,
    }
