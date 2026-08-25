import { useEffect, useMemo, useState } from "react";
import {
  EpisodeOod as Ood,
  EpisodeOodCost,
  errorText,
  getEpisodeOod,
  getEpisodeOodCost,
} from "./api";
import { measured, ordinal } from "./measured";

/** How unusual each frame of an episode is, against the rest of the dataset.
 *
 *  A DISTANCE AND A PERCENTILE, never a verdict. "OOD" as a boolean would be
 *  a threshold somebody chose, and this project does not ship those — so the
 *  reference set is named on screen, the null it is gated on is drawn as a
 *  line rather than folded into a label, and what a reader does with the
 *  number is theirs.
 *
 *  The chart is deliberately the timeline's lane, at the timeline's width, on
 *  the timeline's `t`. This is the same episode measured a second way, and
 *  two charts of one episode that do not line up are the exact failure the
 *  panel above exists to prevent.
 *
 *  THE TWO HORIZONTAL RULES ARE THE WHOLE READING. One is the largest
 *  distance rows drawn from this same dataset reached — so a frame under it
 *  is no further out than ordinary data gets — and the other is the furthest
 *  any reference row sat. Neither is a threshold: they are two measured
 *  quantities drawn where they fall, and a frame's position between them is
 *  the answer.
 */

/** The columns worth scoring. Both are vectors present in every LeRobot
 *  dataset; a distance in one says nothing whatever about the other, which is
 *  why the choice is on screen and travels in the payload. */
const SPACES = ["observation.state", "action"];

/** Rows of the ranked list drawn. The route's own cap is 20 and both numbers
 *  are on screen, because a list capped twice with one cap reported is a list
 *  that lied about the other. */
const RANKED_SHOWN = 8;

const LANE_W = 720;
const LANE_H = 96;
const PAD_Y = 8;

export default function EpisodeOod({
  episode,
  ready,
}: {
  episode: number;
  ready: boolean;
}) {
  const [space, setSpace] = useState(SPACES[0]);
  const [cost, setCost] = useState<EpisodeOodCost | null>(null);
  const [data, setData] = useState<Ood | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [at, setAt] = useState<number | null>(null);
  // The column this answer is OF, snapshotted with it. The dial above is live
  // state and a distance in `observation.state` says nothing about `action`.
  const [readSpace, setReadSpace] = useState(SPACES[0]);

  // Cost is cheap — it reads file metadata, not rows — so it leads, and the
  // score itself waits to be asked for. Two passes over 25,650 parquet rows
  // is not something to spend on a panel scrolling past.
  useEffect(() => {
    let live = true;
    setData(null);
    setAt(null);
    setErr("");
    if (!ready) {
      setCost(null);
      return;
    }
    getEpisodeOodCost(episode, space)
      .then((c) => live && setCost(c))
      .catch((e) => {
        if (!live) return;
        setCost(null);
        setErr(errorText(e));
      });
    return () => {
      live = false;
    };
  }, [episode, space, ready]);

  async function score() {
    if (busy) return;
    setBusy(true);
    setErr("");
    // THIS ROUTE READS EVERY PARQUET ROW OF THE DATASET TWICE, so the window
    // between asking and answering is seconds wide, not milliseconds. Without
    // this, moving the episode dial mid-score repainted episode 3's distances
    // under episode 5's label — and moving the SPACE dial was worse, because
    // the "raw {space} units" line reads live state beside a payload measured
    // in the other column.
    const forEpisode = episode;
    const forSpace = space;
    try {
      const got = await getEpisodeOod(forEpisode, forSpace);
      if (forEpisode !== episode || forSpace !== space) return;
      setData(got);
      setReadSpace(forSpace);
    } catch (e) {
      if (forEpisode !== episode || forSpace !== space) return;
      setData(null);
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  /** Chart geometry. The y-axis spans every quantity DRAWN on it — the
   *  frames, the null and the reference maximum — because a rule that falls
   *  off the top of its own chart says the frames beat something they did
   *  not. */
  const chart = useMemo(() => {
    if (!data || data.frames.length === 0) return null;
    const ref = data.reference;
    const marks = [
      ref.null_max,
      ref.distances?.max ?? null,
      ...data.frames.map((f) => f.distance),
    ].filter((v): v is number => typeof v === "number" && Number.isFinite(v));
    const hi = Math.max(...marks);
    const lo = Math.min(0, ...marks);
    const n = data.frames.length;
    const x = (i: number) => (n < 2 ? LANE_W / 2 : (i / (n - 1)) * LANE_W);
    const y = (v: number) =>
      hi > lo ? LANE_H - PAD_Y - ((v - lo) / (hi - lo)) * (LANE_H - 2 * PAD_Y) : LANE_H / 2;
    return {
      x,
      y,
      hi,
      lo,
      points: data.frames.map((f, i) => `${x(i).toFixed(2)},${y(f.distance).toFixed(2)}`).join(" "),
    };
  }, [data]);

  /** THE PLAYHEAD IS THE MEASUREMENT, so it cannot be pointer-only.
   *
   *  Every per-dimension value on this panel is gated on it — with no
   *  playhead they all read "—" — and the SVGs are `aria-hidden` because the
   *  text readout beside them is the accessible version. That is only true if
   *  the readout can be reached, and it could not: a keyboard user saw em
   *  dashes and nothing else, forever.
   *
   *  `role="slider"` with arrow keys is the right shape here: it IS a value
   *  along one axis, and Home/End are the ends of the episode. */
  function onKey(e: React.KeyboardEvent<HTMLDivElement>, n: number) {
    if (n === 0) return;
    const step = e.shiftKey ? Math.max(1, Math.round(n / 10)) : 1;
    const from = at ?? 0;
    let next: number | null = null;
    if (e.key === "ArrowRight" || e.key === "ArrowUp") next = from + step;
    else if (e.key === "ArrowLeft" || e.key === "ArrowDown") next = from - step;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = n - 1;
    else if (e.key === "Escape") {
      setAt(null);
      return;
    }
    if (next === null) return;
    e.preventDefault();
    setAt(Math.max(0, Math.min(n - 1, next)));
  }

  if (!ready) return null;

  const here =
    at !== null && data && at >= 0 && at < data.frames.length
      ? (data.frames[at] ?? null)
      : null;

  return (
    <div className="episode-ood">
      <div className="sect sub">
        <span className="dot d-vla" />
        <h3>HOW UNUSUAL EACH FRAME IS</h3>
        <span className="rule" />
      </div>
      <p className="meta">
        Every frame of this episode against the rest of the dataset, in the
        column you pick. A distance and a percentile — nothing here is labelled
        out-of-distribution, because that word is a threshold and this reports a
        measurement.
      </p>

      <div className="row ood-controls">
        <label className="meta" htmlFor="ood-space">
          measured in
        </label>
        <select
          id="ood-space"
          className="combo"
          value={space}
          onChange={(e) => setSpace(e.target.value)}
          disabled={busy}
        >
          {SPACES.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <button className="ghost sm" onClick={() => void score()} disabled={busy}>
          {busy ? "scoring…" : "score this episode"}
        </button>
      </div>

      {/* WHAT IT COSTS, BEFORE IT COSTS IT. `forward_passes: 0` is the
          headline and it is not a technicality — every other measurement in
          this tool is priced in forward passes, and a reader who assumes this
          one is too will over-estimate it by orders of magnitude. */}
      {cost && !data && (
        <div className="hint ood-cost">
          <b>{cost.forward_passes} forward passes.</b> {cost.forward_passes_why}{" "}
          {cost.passes_over_the_data} passes over the data ·{" "}
          {cost.rows_to_read.toLocaleString()} {cost.cost_unit} ·{" "}
          {cost.reference_rows.toLocaleString()} reference rows
          {cost.reference_rows_is_a_ceiling ? " at most" : ""} ·{" "}
          {/* NOT "about — s". The route sends `seconds: null` when nothing has
              timed this machine's disk, and printing a dash into a sentence
              shaped like an estimate makes the absence look like a formatting
              fault rather than the deliberate refusal it is. */}
          {cost.seconds === null || !Number.isFinite(cost.seconds)
            ? cost.seconds_from
            : `about ${measured(cost.seconds, 1)} s (${cost.seconds_from})`}
        </div>
      )}

      {err && <div className="hint err">{err}</div>}

      {data && chart && (
        <>
          <div className="row ood-chips">
            <span className="pill">
              {data.n_frames === data.frames_total
                ? `${data.n_frames} frames`
                : `${data.n_frames} of ${data.frames_total} frames`}
            </span>
            <span className="pill">
              {data.reference.rows_read.toLocaleString()} reference rows
              {data.reference.sampled
                ? ` · every ${ordinal(data.reference.row_stride)} eligible`
                : ""}
            </span>
            {data.reference.excluded_episode !== null && (
              <span className="meta">
                episode {data.reference.excluded_episode}'s own{" "}
                {data.reference.excluded_rows} rows were held OUT of it
              </span>
            )}
            {data.reference.directions_dropped > 0 && (
              <span className="meta warn">
                {data.reference.directions_dropped} of{" "}
                {data.reference.dimensions} directions had no spread to measure
                in — movement along those is reported separately, not folded in
              </span>
            )}
            {here !== null && (
              <span className="pill tl-now">
                t = {here.t} · {measured(here.distance, 3)} ·{" "}
                {measured(here.percentile, 2)}th pct
              </span>
            )}
          </div>

          <div
            className="ood-chart"
            tabIndex={0}
            role="slider"
            aria-label={`playhead over ${data.frames.length} scored frames of episode ${data.episode}`}
            aria-valuemin={0}
            aria-valuemax={Math.max(0, data.frames.length - 1)}
            aria-valuenow={here ? at ?? undefined : undefined}
            aria-valuetext={
              here
                ? `frame ${here.t}, distance ${measured(here.distance, 3)}, ${measured(here.percentile, 2)}th percentile`
                : "no frame selected"
            }
            onKeyDown={(e) => onKey(e, data.frames.length)}
            onPointerMove={(e) => {
              const box = e.currentTarget.getBoundingClientRect();
              if (!(box.width > 0) || !data.frames.length) return;
              const i = Math.round(
                ((e.clientX - box.left) / box.width) * (data.frames.length - 1),
              );
              setAt(Math.max(0, Math.min(data.frames.length - 1, i)));
            }}
            onPointerLeave={() => setAt(null)}
          >
            <svg viewBox={`0 0 ${LANE_W} ${LANE_H}`} preserveAspectRatio="none" aria-hidden="true">
              {/* The reference maximum, then the null, in that order so the
                  null draws on top where they nearly coincide. */}
              {Number.isFinite(data.reference.distances?.max) && (
                <line
                  className="ood-rule ref"
                  x1="0"
                  x2={LANE_W}
                  y1={chart.y(data.reference.distances.max)}
                  y2={chart.y(data.reference.distances.max)}
                  vectorEffect="non-scaling-stroke"
                />
              )}
              {data.reference.null_max !== null && (
                <line
                  className="ood-rule null"
                  x1="0"
                  x2={LANE_W}
                  y1={chart.y(data.reference.null_max)}
                  y2={chart.y(data.reference.null_max)}
                  vectorEffect="non-scaling-stroke"
                />
              )}
              <polyline
                className="ood-line"
                points={chart.points}
                fill="none"
                vectorEffect="non-scaling-stroke"
              />
              {/* A frame past every reference row is not a taller point on the
                  same scale — the percentile saturates at 100 there and stops
                  distinguishing them. Marked, so it is not read as one. */}
              {data.frames.map((f, i) =>
                f.beyond_reference_max ? (
                  <circle
                    key={i}
                    className="ood-beyond"
                    cx={chart.x(i)}
                    cy={chart.y(f.distance)}
                    r={2.5}
                    vectorEffect="non-scaling-stroke"
                  />
                ) : null,
              )}
              {at !== null && (
                <line
                  className="tl-head"
                  x1={chart.x(at)}
                  x2={chart.x(at)}
                  y1="0"
                  y2={LANE_H}
                  vectorEffect="non-scaling-stroke"
                />
              )}
            </svg>
          </div>

          <ul className="ood-legend">
            {data.reference.null_max !== null && (
              <li>
                <i className="ood-swatch null" />
                <span className="meta">
                  {measured(data.reference.null_max, 3)} — {data.reference.null_description}
                </span>
              </li>
            )}
            {data.reference.null_max === null && data.reference.null_reason && (
              <li>
                <span className="meta warn">
                  No null was drawn: {data.reference.null_reason}
                </span>
              </li>
            )}
            {Number.isFinite(data.reference.distances?.max) && (
              <li>
                <i className="ood-swatch ref" />
                <span className="meta">
                  {measured(data.reference.distances.max, 3)} — the furthest any
                  of the {data.reference.distances.count.toLocaleString()}{" "}
                  reference rows sat from their own centre
                </span>
              </li>
            )}
          </ul>

          {/* The frame under the pointer, in full. The chart carries the
              shape; this carries the three numbers that qualify it. */}
          {here && (
            <div className="ood-readout">
              <span>
                <span className="meta">distance</span>
                <b>{measured(here.distance, 4)}</b>
              </span>
              <span>
                <span className="meta">percentile</span>
                <b>
                  {measured(here.percentile, 2)}
                  {here.beyond_reference_max ? " (beyond every reference row)" : ""}
                </b>
                <span className="meta">
                  ±{measured(here.percentile_resolution, 3)}, one row of the sample
                </span>
              </span>
              <span>
                <span className="meta">past the null</span>
                <b>
                  {here.clears_null === null
                    ? "no null to compare with"
                    : here.clears_null
                      ? "yes"
                      : "no"}
                </b>
              </span>
              {here.off_manifold !== null && (
                <span>
                  <span className="meta">off the manifold</span>
                  <b>{measured(here.off_manifold, 4)}</b>
                  <span className="meta">
                    raw {readSpace} units — no spread to divide by there
                  </span>
                </span>
              )}
            </div>
          )}

          {data.ranked.length > 0 && (
            <div className="ood-ranked">
              {/* TWO caps, and only one was reported. The route ranks the
                  furthest 20; this list then sliced to 8 while the sentence
                  printed `ranked.length` — so it read "furthest 20 of 159"
                  above eight rows, which is exactly the silent truncation the
                  sentence promises is not happening. */}
              <span className="meta">
                furthest {Math.min(RANKED_SHOWN, data.ranked.length)} of{" "}
                {data.n_ranked_total} scored — the cap is on this list only,
                every frame is in the chart
              </span>
              <ol>
                {data.ranked.slice(0, RANKED_SHOWN).map((f) => (
                  <li key={f.t}>
                    <button
                      className="ghost sm"
                      onClick={() => {
                        // `findIndex` returns -1 for a miss, and -1 passes
                        // `at < frames.length` — so `frames[-1]` is
                        // `undefined`, and `here !== null` is TRUE for
                        // undefined, which takes the panel down on `here.t`.
                        const i = data.frames.findIndex((g) => g.t === f.t);
                        if (i >= 0) setAt(i);
                      }}
                    >
                      t {f.t}
                    </button>
                    <span className="meta">
                      {measured(f.distance, 3)} · {measured(f.percentile, 2)}th pct
                      {f.clears_null === true ? " · past the null" : ""}
                    </span>
                  </li>
                ))}
              </ol>
            </div>
          )}

          {data.n_unscored > 0 && (
            <div className="hint">
              {data.n_unscored} frame(s) could not be scored
              {data.unscored.length < data.n_unscored
                ? `, ${data.unscored.length} listed`
                : ""}
              : {data.unscored.map((u) => `t ${u.t} — ${u.why}`).join("; ")}
            </div>
          )}

          <div className="hint">{data.means}</div>
        </>
      )}
    </div>
  );
}
