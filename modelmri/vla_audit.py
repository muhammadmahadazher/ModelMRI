"""Prove a robot dataset is intact — or say exactly where it is not.

THE BUG CLASS THIS EXISTS FOR IS ONE THIS PROJECT SHIPPED. `vla_data.py`
read `video_from_timestamp`, which is not a column any LeRobot v3.0 dataset
has; the real name is `videos/<camera>/from_timestamp`. A `.get(name, 0.0)`
turned that miss into zero for every episode, so all 206 episodes decoded from
the start of the file — episodes 0, 5 and 20 returned byte-identical images
while the state vector printed underneath them was correctly episode 5's and
episode 20's. The picture and the numbers disagreed and nothing said so.

None of it crashed. That is the point: a dataset can be thoroughly broken and
still load, train and evaluate, and the damage shows up as a policy that will
not learn rather than as an error anybody can search for.

WHAT IT WILL NOT DO
-------------------
It does not grade. ORBIT ships an A-to-F readiness score calibrated on eighty
other people's datasets, and a letter is a summary of somebody else's opinion
about what matters. Every check here reports what it measured and what it
compared against, and the reader decides what is disqualifying for their run.

It does not count defects in the contradiction search. Two similar states with
different actions is usually legitimate multimodality — a human demonstrator
solving the same situation two ways — so those come back as PAIRS TO INSPECT
with both thresholds printed, never as a number that looks like a verdict.

NOTHING IS DOWNLOADED, no policy is loaded, no GPU is touched and lerobot is
not imported. This reads the files already on disk.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field

from .errors import BadRequest, Refusal

# Verdicts. Deliberately not a score: a check either proved the thing, proved
# the opposite, or could not be run here, and collapsing those three into a
# number is what a grade does.
OK = "ok"
BROKEN = "broken"
UNCHECKED = "unchecked"

# Nearest-neighbour over every frame pair is quadratic. This caps what one
# audit reads, and a capped scan is a SAMPLE -- which the output says.
MAX_CONTRADICTION_FRAMES = 2_000

# How many contradictory pairs the search will find before it stops. The scan
# is quadratic in the frame count above — 2,000 frames is two million pairs —
# so it stops once it has enough to show rather than pricing every audit at
# that. REPORTED as `pairs_complete`, because the stopping point is not the
# number of pairs that exist and was being published as though it were.
MAX_PAIRS_KEPT = 40

# How close two states have to be to count as "the same situation", and how
# far apart their actions have to be to count as disagreeing. Both are
# fractions of the data's own spread rather than absolute numbers: a state
# vector in metres and one in pixels have nothing in common but their spread.
STATE_EPSILON = 0.02
ACTION_DELTA = 0.25

# Episodes whose first and last frames are decoded and hashed. A full decode
# of every episode is minutes; this is the direct generalisation of the
# 206-episodes bug and a handful of episodes catches it.
DECODE_SAMPLE = 6


class AuditError(BadRequest):
    """This dataset cannot be audited honestly, and we say why."""


@dataclass
class Check:
    """One proof, its verdict, and the numbers behind it."""

    name: str
    verdict: str
    detail: str
    measured: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Report:
    repo_id: str
    # None when the dataset would not tell us. NOT 0: "this dataset has no
    # frames" and "the frame table could not be read" are different answers,
    # and the second one is the whole reason somebody opened an audit.
    n_episodes: int | None
    n_frames: int | None
    checks: list[Check] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        out = asdict(self)
        out["means"] = self.means()
        return out

    @property
    def broken(self) -> list[Check]:
        return [c for c in self.checks if c.verdict == BROKEN]

    @property
    def unchecked(self) -> list[Check]:
        return [c for c in self.checks if c.verdict == UNCHECKED]

    def means(self) -> str:
        parts = [
            f"{len(self.checks)} checks on {self.repo_id}: "
            f"{len(self.checks) - len(self.broken) - len(self.unchecked)} "
            f"proved, {len(self.broken)} failed, {len(self.unchecked)} could "
            f"not be run here."
        ]
        if self.broken:
            names = ", ".join(c.name for c in self.broken)
            parts.append(
                f"WHAT FAILED: {names}. Each says what it measured and what it "
                f"compared against — none of this crashes anything, which is "
                f"why it is worth checking: a dataset can be thoroughly broken "
                f"and still load, train and evaluate."
            )
        else:
            parts.append(
                "Nothing failed. That is not a certificate — it means these "
                "checks passed, and the ones that could not run here are "
                "listed rather than counted as passes."
            )
        if self.unchecked:
            parts.append(
                "COULD NOT BE RUN: "
                + ", ".join(f"{c.name} ({c.detail})" for c in self.unchecked[:3])
                + "."
            )
        parts.append(
            "THERE IS NO GRADE. A letter would be a summary of somebody else's "
            "opinion about what matters in your data; every number above is "
            "here so you can decide what is disqualifying for your run."
        )
        return " ".join(parts)


# --------------------------------------------------------------- the checks


def check_tiling(reader) -> Check:
    """Do the episodes tile the frame table exactly — no gaps, no overlaps?

    The failure this catches is a metadata boundary bug in a v2.1-to-v3.0
    conversion: episode spans that overlap silently hand two episodes the same
    frames, and spans with a gap leave frames belonging to nobody. Neither
    raises. Training on either produces a policy that will not learn and no
    error anybody can search for.
    """
    episodes = reader.episodes()
    table = reader._frame_table()
    n_frames = len(table.get("episode_index") or [])
    spans = sorted((e.data_from, e.data_from + e.length, e.index) for e in episodes)

    gaps, overlaps = [], []
    cursor = 0
    for start, end, index in spans:
        if start > cursor:
            gaps.append(
                {"after_row": cursor, "before_episode": index, "rows": start - cursor}
            )
        elif start < cursor:
            overlaps.append(
                {"episode": index, "starts_at": start, "previous_ends_at": cursor}
            )
        cursor = max(cursor, end)

    total = sum(e.length for e in episodes)
    measured = {
        "episodes": len(episodes),
        "frames_in_table": n_frames,
        "frames_claimed": total,
        "gaps": gaps[:8],
        "overlaps": overlaps[:8],
        "n_gaps": len(gaps),
        "n_overlaps": len(overlaps),
    }
    problems = []
    if gaps:
        problems.append(f"{len(gaps)} gap(s) — rows no episode claims")
    if overlaps:
        problems.append(f"{len(overlaps)} overlap(s) — rows two episodes both claim")
    if cursor != n_frames:
        problems.append(
            f"the episodes cover {cursor} rows and the frame table has {n_frames}"
        )
    if problems:
        return Check(
            "episode tiling",
            BROKEN,
            "; ".join(problems)
            + ". A frame that belongs to two episodes, or to none, is not "
            "something any loader will complain about.",
            measured,
        )
    return Check(
        "episode tiling",
        OK,
        f"{len(episodes)} episodes tile all {n_frames} frames exactly — every "
        f"row belongs to exactly one episode, with no gaps and no overlaps.",
        measured,
    )


def check_routing(reader) -> Check:
    """Does every episode's video routing land in a file that exists and covers it?

    The other half of the 206-episodes bug. A routing column that is missing,
    zero or points past the end of a container does not raise — it decodes
    SOMETHING, and something is a frame from the wrong episode.
    """
    episodes = reader.episodes()
    missing, out_of_range, zeroed = [], [], []
    durations: dict[str, float] = {}

    for ep in episodes:
        try:
            path = reader._video_file(ep)
        except Exception as err:
            missing.append({"episode": ep.index, "why": type(err).__name__})
            continue
        if not path.exists():
            missing.append({"episode": ep.index, "path": path.name})
            continue
        # from == to == 0 for every episode is exactly what the `.get(name,
        # 0.0)` bug produced, and it is worth naming as its own shape rather
        # than folding into "out of range".
        if ep.from_ts == 0.0 and ep.to_ts == 0.0:
            zeroed.append(ep.index)
            continue
        key = str(path)
        if key not in durations:
            durations[key] = _duration(path)
        limit = durations[key]
        if limit and ep.to_ts > limit + 1e-3:
            out_of_range.append(
                {
                    "episode": ep.index,
                    "to_ts": ep.to_ts,
                    "file_seconds": round(limit, 3),
                }
            )

    measured = {
        "episodes": len(episodes),
        "files": len(durations),
        "missing": missing[:8],
        "out_of_range": out_of_range[:8],
        "n_missing": len(missing),
        "n_out_of_range": len(out_of_range),
        "n_zero_span": len(zeroed),
    }
    if len(zeroed) == len(episodes) and episodes:
        return Check(
            "video routing",
            BROKEN,
            f"every one of {len(episodes)} episodes routes to timestamp 0.0 — "
            f"they would all decode the same frames from the start of the "
            f"file. This is the exact shape of the bug this tool shipped in "
            f"0.10: a missing routing column defaulted to zero and 206 "
            f"episodes showed one video while the state vector underneath was "
            f"correct.",
            measured,
        )
    problems = []
    if missing:
        problems.append(f"{len(missing)} episode(s) route to a file that is not there")
    if out_of_range:
        problems.append(
            f"{len(out_of_range)} episode(s) end past their file's duration"
        )
    if zeroed:
        problems.append(f"{len(zeroed)} episode(s) have a zero-length span")
    if problems:
        return Check("video routing", BROKEN, "; ".join(problems) + ".", measured)
    if not durations:
        return Check(
            "video routing",
            UNCHECKED,
            "no video file durations could be read — PyAV is not installed, or "
            "these files are not decodable here",
            measured,
        )
    return Check(
        "video routing",
        OK,
        f"all {len(episodes)} episodes route into {len(durations)} file(s) "
        f"that exist, and every span ends inside its file's duration.",
        measured,
    )


def _duration(path) -> float:
    try:
        import av
    except ImportError:
        return 0.0
    try:
        with av.open(str(path)) as container:
            if container.duration:
                return float(container.duration) / 1_000_000.0
    except Exception:
        return 0.0
    return 0.0


def check_distinct_frames(reader, sample: int = DECODE_SAMPLE) -> Check:
    """Do different episodes actually decode to different pictures?

    THE DIRECT GENERALISATION of the 206-episodes bug, and the only check here
    that looks at pixels. Every structural check above can pass while the
    decoder still hands back the same frame for every episode — routing that
    is present and wrong looks exactly like routing that is right.
    """
    episodes = reader.episodes()
    if len(episodes) < 2:
        return Check(
            "distinct frames",
            UNCHECKED,
            "this dataset has fewer than two episodes, so there is nothing to "
            "tell apart",
            {"episodes": len(episodes)},
        )

    step = max(1, len(episodes) // max(1, sample))
    picked = episodes[::step][:sample]
    digests: dict[str, list[int]] = {}
    failed = []
    for ep in picked:
        try:
            rgb = reader.raw_frame(ep.index, 0)
        except Exception as err:
            failed.append({"episode": ep.index, "why": type(err).__name__})
            continue
        digest = hashlib.sha256(bytes(memoryview(rgb).tobytes())).hexdigest()[:16]
        digests.setdefault(digest, []).append(ep.index)

    collisions = [v for v in digests.values() if len(v) > 1]
    measured = {
        "episodes_sampled": len(picked),
        "distinct_images": len(digests),
        "failed": failed[:4],
        "collisions": collisions[:4],
        # Both lists above are capped at 4, so both need their true length
        # beside them the way every other capped list in this file does.
        # MEASURED on a machine without PyAV, where DECODE_SAMPLE is 6 against
        # a cap of 4: the payload read `episodes_sampled: 6, failed: [4
        # entries], distinct_images: 0` and contradicted itself -- counting the
        # list said 4 of 6 failed and 2 decoded, `distinct_images: 0` said none
        # did. The frontend derives "showing 4 of N" from `n_<key>` generically,
        # so with these absent it rendered the capped list as the whole story.
        "n_failed": len(failed),
        "n_collisions": len(collisions),
    }
    if failed and not digests:
        return Check(
            "distinct frames",
            UNCHECKED,
            f"no frame could be decoded ({failed[0].get('why')}) — PyAV is "
            f"missing, or these videos are not readable here",
            measured,
        )
    if len(digests) == 1 and len(picked) > 1:
        return Check(
            "distinct frames",
            BROKEN,
            f"all {len(picked)} sampled episodes decoded to the SAME image. "
            f"Different episodes are showing the same video while their state "
            f"vectors differ — the picture and the numbers disagree, and "
            f"nothing else in this audit would have said so.",
            measured,
        )
    if collisions:
        return Check(
            "distinct frames",
            BROKEN,
            f"{len(collisions)} group(s) of episodes decoded to identical "
            f"first frames — e.g. episodes {collisions[0]}. Two episodes can "
            f"legitimately start from the same pose, so check these by eye; "
            f"what is not legitimate is routing that sends them to the same "
            f"place in the file.",
            measured,
        )
    return Check(
        "distinct frames",
        OK,
        f"{len(picked)} sampled episodes decoded to {len(digests)} distinct "
        f"first frames. SAMPLED, not exhaustive: a full decode of every "
        f"episode is minutes, and this is the shape the 0.10 bug took.",
        measured,
    )


def check_normalisation(reader) -> Check:
    """Do the recorded statistics describe the data they sit beside?

    Stale stats are the quietest corruption in this list: training normalises
    with them, so wrong numbers do not break anything visibly — the policy
    simply sees a distribution nobody intended.
    """
    path = reader.snapshot / "meta" / "stats.json"
    if not path.exists():
        return Check(
            "normalisation stats",
            UNCHECKED,
            "this dataset carries no meta/stats.json to check against",
            {},
        )
    try:
        stats = json.loads(path.read_text(encoding="utf-8"))
    except Exception as err:
        return Check(
            "normalisation stats",
            BROKEN,
            f"meta/stats.json could not be read as JSON ({type(err).__name__})",
            {},
        )

    table = reader._frame_table()
    drifted, checked = [], []
    for key, recorded in stats.items():
        column = table.get(key)
        if column is None or not isinstance(recorded, dict):
            continue
        wanted = recorded.get("mean")
        if wanted is None:
            continue
        actual = _column_mean(column)
        if actual is None:
            continue
        want = wanted if isinstance(wanted, list) else [wanted]
        if len(want) != len(actual):
            drifted.append(
                {"field": key, "recorded_dims": len(want), "actual_dims": len(actual)}
            )
            continue
        checked.append(key)
        worst = max(
            (abs(float(w) - a) for w, a in zip(want, actual, strict=True)),
            default=0.0,
        )
        spread = max((abs(a) for a in actual), default=1.0) or 1.0
        # A RELATIVE tolerance. The fields here are in different units --
        # pixels, metres, radians -- and one absolute number would be strict
        # for one and meaningless for another.
        if worst > 0.01 * spread:
            drifted.append(
                {
                    "field": key,
                    "worst_abs_diff": round(worst, 6),
                    "scale": round(spread, 6),
                }
            )

    measured = {
        "fields_checked": checked,
        "drifted": drifted[:8],
        "n_drifted": len(drifted),
    }
    if not checked and not drifted:
        return Check(
            "normalisation stats",
            UNCHECKED,
            "no field in meta/stats.json matched a column in the frame table",
            measured,
        )
    if drifted:
        return Check(
            "normalisation stats",
            BROKEN,
            f"{len(drifted)} field(s) have recorded statistics that do not "
            f"describe the data beside them, e.g. {drifted[0]}. Training "
            f"normalises with these, so the policy sees a distribution nobody "
            f"intended and nothing raises.",
            measured,
        )
    return Check(
        "normalisation stats",
        OK,
        f"the recorded mean matches the data for {len(checked)} field(s), "
        f"within 1% of each field's own scale.",
        measured,
    )


def _column_mean(column) -> list[float] | None:
    """Per-dimension mean of a column that may be scalars or vectors."""
    rows = list(column)
    if not rows:
        return None
    first = rows[0]
    if isinstance(first, (list, tuple)):
        width = len(first)
        if width == 0:
            return None
        totals = [0.0] * width
        used = 0
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) != width:
                continue
            for i, v in enumerate(row):
                totals[i] += float(v)
            used += 1
        return [t / used for t in totals] if used else None
    try:
        return [sum(float(v) for v in rows) / len(rows)]
    except (TypeError, ValueError):
        return None


def check_constant_dimensions(reader) -> Check:
    """Which state and action dimensions never change?

    Not a defect on its own -- a gripper that is open for a whole dataset is
    real data -- which is why this reports the indices and does not judge
    them. A dimension that never moves is a dimension a policy cannot learn
    from, and knowing which ones they are before training is the point.
    """
    table = reader._frame_table()
    frozen: dict[str, list[int]] = {}
    widths: dict[str, int] = {}
    for key in ("observation.state", "action"):
        column = table.get(key)
        if column is None:
            continue
        rows = [r for r in column if isinstance(r, (list, tuple))]
        if not rows:
            continue
        width = len(rows[0])
        widths[key] = width
        still = []
        for i in range(width):
            first = float(rows[0][i])
            if all(float(r[i]) == first for r in rows if len(r) == width):
                still.append(i)
        if still:
            frozen[key] = still

    measured = {"widths": widths, "constant": frozen}
    if not widths:
        return Check(
            "constant dimensions",
            UNCHECKED,
            "no observation.state or action column found to inspect",
            measured,
        )
    if not frozen:
        return Check(
            "constant dimensions",
            OK,
            "every state and action dimension varies somewhere in this dataset.",
            measured,
        )
    where = "; ".join(f"{k} dims {v}" for k, v in frozen.items())
    return Check(
        "constant dimensions",
        OK,
        f"these never change: {where}. NOT A DEFECT — a gripper held open for "
        f"a whole dataset is real data — but a dimension that never moves is "
        f"one a policy cannot learn from, and it is better known now.",
        measured,
    )


def check_action_lag(reader, max_lag: int = 10) -> Check:
    """Does the action lead or lag the state, and by how many frames?

    Assumes a FIXED control frequency, which `info.json` states. Refused when
    it does not: a lag in frames is meaningless if the frames are not evenly
    spaced in time, and reporting one anyway would be a number about nothing.
    """
    fps = reader.info.get("fps")
    if not isinstance(fps, (int, float)) or fps <= 0:
        return Check(
            "action lag",
            UNCHECKED,
            "this dataset's info.json does not state a control frequency, and "
            "a lag measured in frames means nothing without one",
            {},
        )

    table = reader._frame_table()
    state = table.get("observation.state")
    action = table.get("action")
    if state is None or action is None:
        return Check(
            "action lag",
            UNCHECKED,
            "this dataset has no observation.state or no action column",
            {},
        )

    xs = _first_varying(state)
    ys = _first_varying(action)
    if xs is None or ys is None:
        return Check(
            "action lag",
            UNCHECKED,
            "no state or action dimension varies enough to correlate",
            {},
        )

    best_lag, best_r = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        r = _correlate(xs, ys, lag)
        if r is not None and r > best_r:
            best_lag, best_r = lag, r

    measured = {
        "fps": float(fps),
        "best_lag_frames": best_lag,
        "best_correlation": round(best_r, 4),
        "ms_per_frame": round(1000.0 / float(fps), 2),
        "searched": [-max_lag, max_lag],
    }
    if best_r < 0.2:
        return Check(
            "action lag",
            UNCHECKED,
            f"the strongest correlation at any lag is {best_r:.3f}, too weak "
            f"to place a lag. That is a fact about this data, not a failure: "
            f"state and action need not be correlated at all.",
            measured,
        )
    ms = abs(best_lag) * 1000.0 / float(fps)
    if best_lag == 0:
        detail = f"action and state line up with no lag (r={best_r:.3f})."
    elif best_lag > 0:
        detail = (
            f"the action LAGS the state by {best_lag} frame(s) ({ms:.0f} ms at "
            f"{fps} fps, r={best_r:.3f}) — the recorded action follows the "
            f"observation it responds to."
        )
    else:
        # TWO READINGS, and this check cannot tell them apart. Measured on
        # lerobot/pusht: the action leads the state by one frame at r=0.986,
        # and that is CORRECT there — pusht's action is a target position, so
        # it necessarily describes where the arm is going rather than where it
        # has been. Calling that an off-by-one, as the first version did,
        # would have reported a healthy dataset as damaged.
        detail = (
            f"the action LEADS the state by {abs(best_lag)} frame(s) "
            f"({ms:.0f} ms at {fps} fps, r={best_r:.3f}). Two things look like "
            f"this and this check cannot separate them: an action space where "
            f"the action is a TARGET rather than a delta, which is normal and "
            f"is what lerobot/pusht does — or an off-by-one in the recorder, "
            f"where a policy would learn to predict the past. Which one it is "
            f"depends on what your action column means."
        )
    return Check("action lag", OK, detail, measured)


def _first_varying(column) -> list[float] | None:
    rows = [r for r in column if isinstance(r, (list, tuple))]
    if not rows:
        try:
            values = [float(v) for v in column]
        except (TypeError, ValueError):
            return None
        return values if len(set(values)) > 1 else None
    width = len(rows[0])
    for i in range(width):
        values = [float(r[i]) for r in rows if len(r) == width]
        if len(set(values)) > 1:
            return values
    return None


def _correlate(xs: list[float], ys: list[float], lag: int) -> float | None:
    if lag > 0:
        a, b = xs[: len(xs) - lag], ys[lag:]
    elif lag < 0:
        a, b = xs[-lag:], ys[: len(ys) + lag]
    else:
        a, b = xs, ys
    n = min(len(a), len(b))
    if n < 8:
        return None
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((v - ma) ** 2 for v in a)
    vb = sum((v - mb) ** 2 for v in b)
    if va <= 0 or vb <= 0:
        return None
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    return cov / math.sqrt(va * vb)


def check_contradictions(
    reader,
    *,
    max_frames: int = MAX_CONTRADICTION_FRAMES,
    epsilon: float = STATE_EPSILON,
    delta: float = ACTION_DELTA,
) -> Check:
    """Frames whose states are close and whose actions are far apart.

    PAIRS TO INSPECT, NEVER A DEFECT COUNT. Two similar states with different
    actions is usually legitimate multimodality — a human demonstrator solving
    the same situation two ways, or a symmetric task with two valid routes —
    and presenting that as corruption would be the tool inventing a problem.

    Both thresholds are fractions of the data's own spread and both are
    printed, because they are the entire content of the claim: a different
    epsilon gives a different list.
    """
    table = reader._frame_table()
    state = table.get("observation.state")
    action = table.get("action")
    if state is None or action is None:
        return Check(
            "contradictory demonstrations",
            UNCHECKED,
            "this dataset has no observation.state or no action column",
            {},
        )
    states = [list(map(float, r)) for r in state if isinstance(r, (list, tuple))]
    actions = [list(map(float, r)) for r in action if isinstance(r, (list, tuple))]
    n = min(len(states), len(actions))
    if n < 2:
        return Check(
            "contradictory demonstrations",
            UNCHECKED,
            "fewer than two frames carry both a state and an action",
            {"frames": n},
        )

    step = max(1, n // max_frames)
    idx = list(range(0, n, step))[:max_frames]
    truncated = len(idx) < n

    s_scale = _spread([states[i] for i in idx])
    a_scale = _spread([actions[i] for i in idx])
    if s_scale <= 0 or a_scale <= 0:
        return Check(
            "contradictory demonstrations",
            UNCHECKED,
            "the state or action has no spread to measure closeness against",
            {"frames": len(idx)},
        )

    near = epsilon * s_scale
    far = delta * a_scale
    pairs = []
    # Whether the search STOPPED rather than finished. The pair scan is
    # quadratic in `max_frames` (2,000 frames is two million pairs), so it
    # stops once it has enough to show — and that stopping point was being
    # published as the count of what exists.
    stopped_early = False
    for a_i in range(len(idx)):
        for b_i in range(a_i + 1, len(idx)):
            i, j = idx[a_i], idx[b_i]
            ds = _dist(states[i], states[j])
            if ds > near:
                continue
            da = _dist(actions[i], actions[j])
            if da > far:
                pairs.append(
                    {
                        "frame_a": i,
                        "frame_b": j,
                        "state_distance": round(ds, 6),
                        "action_distance": round(da, 6),
                    }
                )
                if len(pairs) >= MAX_PAIRS_KEPT:
                    stopped_early = True
                    break
        if stopped_early:
            break

    measured = {
        "frames_scanned": len(idx),
        "frames_total": n,
        "stride": step,
        "truncated": truncated,
        "state_epsilon": round(near, 6),
        "action_delta": round(far, 6),
        "epsilon_fraction": epsilon,
        "delta_fraction": delta,
        "pairs": pairs[:12],
        # `n_pairs` is what was FOUND, and `pairs_complete` says whether that
        # is all there is. The search stops at `MAX_PAIRS_KEPT` because it is
        # quadratic, and this published the stopping point as the total:
        # measured against this repo's own fixture, a set with 400 qualifying
        # pairs reported `n_pairs: 40, truncated: False, "Scanned all 80
        # frames"` — a tenfold under-report presented as a complete count,
        # with the one field that could have contradicted it saying the scan
        # was complete. It was: the FRAME scan finished, the PAIR search did
        # not, and those are two different truncations that needed two fields.
        "n_pairs": len(pairs),
        "pairs_complete": not stopped_early,
        "pairs_cap": MAX_PAIRS_KEPT,
    }
    tail = (
        f" Scanned {len(idx)} of {n} frames at stride {step}, so this is a "
        f"SAMPLE and a pair outside it would not appear."
        if truncated
        else f" Scanned all {n} frames."
    )
    if not pairs:
        return Check(
            "contradictory demonstrations",
            OK,
            f"no pair of frames sits within {near:.4g} in state and further "
            f"than {far:.4g} apart in action." + tail,
            measured,
        )
    how_many = (
        f"{len(pairs)} pair(s)"
        if not stopped_early
        else f"at least {len(pairs)} pair(s) — the search stopped there"
    )
    return Check(
        "contradictory demonstrations",
        OK,
        f"{how_many} of frames are within {near:.4g} in state and "
        f"more than {far:.4g} apart in action. THESE ARE PAIRS TO INSPECT, "
        f"NOT DEFECTS: two similar states with different actions is usually "
        f"legitimate multimodality, and a different epsilon gives a different "
        f"list — both thresholds are above so you can move them." + tail,
        measured,
    )


def _spread(rows: list[list[float]]) -> float:
    """Root-mean-square distance of the rows from their own centre."""
    if not rows:
        return 0.0
    width = len(rows[0])
    centre = [
        sum(r[i] for r in rows if len(r) == width) / len(rows) for i in range(width)
    ]
    total = sum(_dist(r, centre) ** 2 for r in rows if len(r) == width)
    return math.sqrt(total / len(rows))


def _dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


# ------------------------------------------------------------------ the run

CHECKS = (
    check_tiling,
    check_routing,
    check_distinct_frames,
    check_normalisation,
    check_constant_dimensions,
    check_action_lag,
    check_contradictions,
)


def audit(reader) -> Report:
    """Run every check. One failing check never stops the others.

    Deliberately: an audit that stops at the first problem tells you about one
    problem, and the reader is trying to decide whether to train on this data
    at all. A check that raises becomes an `unchecked` row naming the
    exception, which is a fact about this machine rather than about the data.
    """
    started = time.perf_counter()

    # INSIDE the guard, like every check below. These two sat outside it, so
    # a dataset whose parquet will not open took the whole audit down with an
    # unhandled exception -- from the one tool whose stated purpose is telling
    # you "whether to train on this data at all". The case where the data is
    # broken is exactly the case it must survive.
    preamble: list[Check] = []
    try:
        episodes = reader.episodes()
        n_episodes = len(episodes)
    except Exception as err:
        n_episodes = None
        preamble.append(
            Check(
                "episode index",
                UNCHECKED,
                f"the episode index raised {type(err).__name__} on this "
                f"machine, so nothing below could count episodes.",
            )
        )
    try:
        table = reader._frame_table()
        n_frames = len(table.get("episode_index") or [])
    except Exception as err:
        n_frames = None
        preamble.append(
            Check(
                "frame table",
                UNCHECKED,
                f"the frame table raised {type(err).__name__} on this "
                f"machine, so the per-frame checks below have nothing to read.",
            )
        )

    report = Report(
        repo_id=getattr(reader, "repo_id", "?"),
        n_episodes=n_episodes,
        n_frames=n_frames,
    )
    report.checks.extend(preamble)
    for check in CHECKS:
        name = check.__name__.replace("check_", "").replace("_", " ")
        try:
            report.checks.append(check(reader))
        except Refusal as err:
            report.checks.append(Check(name, UNCHECKED, str(err)))
        except Exception as err:
            report.checks.append(
                Check(
                    name,
                    UNCHECKED,
                    f"this check raised {type(err).__name__} on this machine, "
                    f"which is a fact about the run rather than about the data",
                )
            )
    report.seconds = round(time.perf_counter() - started, 2)
    return report
