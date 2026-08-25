import { useEffect, useId, useMemo, useRef, useState } from "react";
import { EpisodeTimeline as Timeline, errorText, getEpisodeTimeline } from "./api";
import { measured } from "./measured";

/** Several of an episode's series on ONE time axis, with one playhead.
 *
 *  The frame and the scrubber above answer "what did the camera see at t" and
 *  nothing else. The questions people bring to a recorded episode are about
 *  COINCIDENCE — the gripper closed here, what was the state doing, did the
 *  reward move — and every one of those needs two series on one axis. Read
 *  off two panels with two x-ranges, a coincidence gets asserted that is not
 *  there, so the shared axis is the product and everything here defends it.
 *
 *  WHAT THE MOTION IS FOR. Two moving things, and neither is decoration:
 *
 *    the playhead   ONE vertical line through every lane at once. Moving it
 *                   is the measurement — it is the only way to read "at this
 *                   t, these three values" without trusting your eye to hold
 *                   a horizontal position across four charts.
 *    the reveal     the series draw in left-to-right through a single clip
 *                   that spans every lane, so they arrive in step. A stagger
 *                   would say the lanes are on different axes, which is the
 *                   one thing this panel exists to deny.
 *
 *  And `play` runs the playhead at the RECORDING'S own rate, off the
 *  `seconds` the dataset published — a 159-frame pusht episode at 10 fps
 *  takes 15.9 seconds here because it took 15.9 seconds there. When the
 *  dataset publishes no timestamps there is no rate to play at, and the
 *  button says so rather than inventing one.
 */

/** Distinguishable within a lane, which is where it matters — a 7-DoF arm
 *  puts seven of these on one chart. All are contrast-checked tokens. */
const DIM_COLOURS = [
  "var(--sem-vla)",
  "var(--sem-scope)",
  "var(--sem-agent)",
  "var(--sem-base)",
  "var(--sem-feat)",
  "var(--sem-ground)",
  "var(--sem-error)",
];

const LANE_W = 720;
const LANE_H = 56;
const PAD_Y = 6;

/** Break a series into the runs that were actually measured.
 *
 *  A `null` is a hole — a non-finite value in the recording, or a timestep a
 *  stride dropped — and a polyline drawn straight across one claims frames
 *  nobody read. Each run becomes its own path, so a hole is a gap on screen.
 */
function runs(series: (number | null)[]): { i: number; v: number }[][] {
  const out: { i: number; v: number }[][] = [];
  let current: { i: number; v: number }[] = [];
  series.forEach((v, i) => {
    if (v === null || !Number.isFinite(v)) {
      if (current.length) out.push(current);
      current = [];
      return;
    }
    current.push({ i, v });
  });
  if (current.length) out.push(current);
  return out;
}

export default function EpisodeTimeline({
  episode,
  ready,
  onSeek,
}: {
  episode: number;
  ready: boolean;
  /** Move the frame scrubber above to a timestep. Called on a CLICK only —
   *  never on hover and never per tick while playing, because the frame it
   *  moves costs a video decode on the server and a 159-frame playback would
   *  order 159 of them. Hovering is free and stays free. */
  onSeek?: (t: number) => void;
}) {
  const [data, setData] = useState<Timeline | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [at, setAt] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const clip = useId().replace(/:/g, "");
  const frame = useRef(0);

  // The payload is of the episode it was asked for, so changing the dial has
  // to drop it rather than leave one episode's series under another's label.
  useEffect(() => {
    let live = true;
    if (!ready) {
      setData(null);
      setErr("");
      return;
    }
    setBusy(true);
    setErr("");
    setAt(null);
    setPlaying(false);
    getEpisodeTimeline(episode)
      .then((d) => {
        if (!live) return;
        setData(d);
      })
      .catch((e) => {
        if (!live) return;
        setData(null);
        setErr(errorText(e));
      })
      .finally(() => live && setBusy(false));
    return () => {
      live = false;
    };
  }, [episode, ready]);

  // Real time, off the dataset's own timestamps. `seconds` is relative to
  // this episode's first frame, so the elapsed clock and the axis agree.
  useEffect(() => {
    if (!playing || !data?.seconds) return;
    const seconds = data.seconds;
    const span = seconds[seconds.length - 1] - seconds[0];
    if (!(span > 0)) {
      setPlaying(false);
      return;
    }
    const started = performance.now();
    const from = at !== null && at < seconds.length - 1 ? seconds[at] : 0;
    const step = () => {
      const elapsed = from + (performance.now() - started) / 1000;
      if (elapsed >= span) {
        setAt(seconds.length - 1);
        setPlaying(false);
        return;
      }
      // The frame whose timestamp this moment has reached — a search, not a
      // multiplication, because a dataset may have dropped frames and an
      // index derived from fps would then point at the wrong one.
      let i = 0;
      while (i + 1 < seconds.length && seconds[i + 1] <= elapsed) i += 1;
      setAt(i);
      frame.current = requestAnimationFrame(step);
    };
    frame.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame.current);
    // `at` is the resume point, read once when play starts. Following it
    // would restart the clock on every tick.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, data]);

  /** Lane geometry, per track. Dimensions of one column share a y-scale —
   *  they are the same quantity, and separate scales would make two motors
   *  look alike when one moved ten times as far. */
  const lanes = useMemo(() => {
    if (!data) return [];
    const n = data.timesteps.length;
    const x = (i: number) => (n < 2 ? LANE_W / 2 : (i / (n - 1)) * LANE_W);
    return data.tracks.map((track) => {
      const lows = track.low.filter((v): v is number => v !== null);
      const highs = track.high.filter((v): v is number => v !== null);
      const lo = lows.length ? Math.min(...lows) : 0;
      const hi = highs.length ? Math.max(...highs) : 0;
      // A series that never moved has no range to scale against. Drawn down
      // the middle of its lane and SAID, rather than stretched to fill a
      // height it never reached.
      const flat = !(hi > lo);
      const y = (v: number) =>
        flat
          ? LANE_H / 2
          : LANE_H - PAD_Y - ((v - lo) / (hi - lo)) * (LANE_H - 2 * PAD_Y);
      return {
        track,
        lo,
        hi,
        flat,
        measured: lows.length > 0,
        paths: track.series.map((dim) =>
          runs(dim).map((run) =>
            run.map((p) => `${x(p.i).toFixed(2)},${y(p.v).toFixed(2)}`).join(" "),
          ),
        ),
        x,
      };
    });
  }, [data]);

  if (!ready) return null;

  const n = data?.timesteps.length ?? 0;
  const cursor = at !== null && at < n ? at : null;
  const clock =
    data?.seconds && cursor !== null ? `${data.seconds[cursor].toFixed(2)} s` : null;

  /** The timestep under the pointer, or `null` off the ends. */
  function under(e: React.PointerEvent<HTMLDivElement>): number | null {
    if (!data || n === 0) return null;
    const box = e.currentTarget.getBoundingClientRect();
    if (!(box.width > 0)) return null;
    const share = (e.clientX - box.left) / box.width;
    const i = Math.round(share * (n - 1));
    return Math.max(0, Math.min(n - 1, i));
  }

  function moveHead(e: React.PointerEvent<HTMLDivElement>) {
    const i = under(e);
    if (i === null) return;
    setPlaying(false);
    setAt(i);
  }

  // A CLICK moves the picture; hovering does not. The frame above costs a
  // video decode, so hovering 159 timesteps at pointer speed would order 159
  // of them — and the reading this panel is for does not need the picture,
  // only the aligned series.
  function commit(e: React.PointerEvent<HTMLDivElement>) {
    const i = under(e);
    if (i === null) return;
    setPlaying(false);
    setAt(i);
    if (onSeek && data) onSeek(data.timesteps[i]);
  }

  return (
    <div className="episode-timeline">
      <div className="sect sub">
        <span className="dot d-vla" />
        <h3>EVERY SERIES ON ONE AXIS</h3>
        <span className="rule" />
      </div>
      {/* Two mounts, two true sentences. Under the scrubber there IS a frame
          to point at; on a machine that cannot decode this dataset's video
          there is not, and claiming one would be describing a picture the
          reader cannot see. */}
      <p className="meta">
        {onSeek ? (
          <>
            The frame above answers what the camera saw at <code>t</code>. These
            answer what everything else was doing at the same <code>t</code> —
            one playhead, one axis, so a coincidence you read here is one the
            recording actually holds. Click a lane to move the frame to it.
          </>
        ) : (
          <>
            No picture here — this dataset's video will not decode on this
            machine. These are its recorded columns, which need no decoder: one
            playhead, one axis, so a coincidence you read here is one the
            recording actually holds.
          </>
        )}
      </p>

      {busy && <div className="hint">aligning the tracks…</div>}
      {err && <div className="hint err">{err}</div>}

      {data && lanes.length > 0 && (
        <>
          <div className="row tl-controls">
            <button
              className="ghost sm"
              onClick={() => setPlaying((p) => !p)}
              disabled={!data.seconds}
              title={
                data.seconds
                  ? "Run the playhead at the rate the episode was recorded at"
                  : "This dataset publishes no timestamps, so there is no rate to play at"
              }
            >
              {playing ? "pause" : "play at recorded rate"}
            </button>
            {!data.seconds && (
              <span className="meta warn">
                no timestamps published — the axis is frame index only
              </span>
            )}
            <span className="pill">
              {n === data.length
                ? `${data.length} timesteps`
                : `${n} of ${data.length} timesteps`}
            </span>
            {data.strided && (
              <span className="meta warn">
                sampled every {data.stride} frames — the ones between are absent,
                not smoothed
              </span>
            )}
            {cursor !== null && (
              <span className="pill tl-now">
                t = {data.timesteps[cursor]}
                {clock && <span className="meta"> · {clock}</span>}
              </span>
            )}
          </div>

          <div
            className={`tl-lanes${playing ? " playing" : ""}`}
            onPointerMove={moveHead}
            onPointerDown={commit}
            onPointerLeave={() => !playing && setAt(null)}
          >
            {lanes.map((lane) => (
              <div className="tl-lane" key={lane.track.column}>
                <div className="tl-label">
                  <code>{lane.track.column}</code>
                  <span className="meta">
                    {lane.track.unit ? lane.track.unit : "no published unit"}
                  </span>
                </div>
                <div className="tl-chart">
                  <svg
                    viewBox={`0 0 ${LANE_W} ${LANE_H}`}
                    preserveAspectRatio="none"
                    aria-hidden="true"
                  >
                    <defs>
                      <clipPath id={`tl-${clip}-${lane.track.column}`}>
                        <rect
                          className="tl-reveal"
                          x="0"
                          y="0"
                          width={LANE_W}
                          height={LANE_H}
                        />
                      </clipPath>
                    </defs>
                    <g clipPath={`url(#tl-${clip}-${lane.track.column})`}>
                      {lane.paths.map((segments, d) =>
                        segments.map((points, s) => (
                          <polyline
                            key={`${d}-${s}`}
                            points={points}
                            fill="none"
                            stroke={DIM_COLOURS[d % DIM_COLOURS.length]}
                            strokeWidth={1.5}
                            strokeLinejoin="round"
                            vectorEffect="non-scaling-stroke"
                          />
                        )),
                      )}
                    </g>
                    {cursor !== null && (
                      <line
                        className="tl-head"
                        x1={lane.x(cursor)}
                        x2={lane.x(cursor)}
                        y1="0"
                        y2={LANE_H}
                        vectorEffect="non-scaling-stroke"
                      />
                    )}
                  </svg>
                  {lane.flat && lane.measured && (
                    <span className="tl-flat meta">
                      <span>
                        constant at {measured(lane.lo, 3)} for every timestep read
                      </span>
                    </span>
                  )}
                </div>
                <div className="tl-values">
                  {lane.track.series.map((dim, d) => {
                    const value = cursor === null ? null : dim[cursor];
                    const named = lane.track.names?.[d];
                    return (
                      <span className="tl-value" key={d}>
                        <i
                          className="tl-swatch"
                          style={{
                            background: DIM_COLOURS[d % DIM_COLOURS.length],
                          }}
                        />
                        <span className={named ? "meta" : "meta unnamed"}>
                          {named ?? `[${d}]`}
                        </span>
                        <b>
                          {cursor === null
                            ? "—"
                            : value === null
                              ? "not measured"
                              : measured(value, 3)}
                        </b>
                        {lane.track.n_nonfinite[d] > 0 && (
                          <span className="meta warn">
                            {lane.track.n_nonfinite[d]} non-finite left out
                          </span>
                        )}
                      </span>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>

          {/* A positional index is not a name. The module refuses to invent
              "dim 0" for exactly this reason, and printing one here would put
              the invention back one layer further out. */}
          {lanes.some((l) => !l.track.names) && (
            <p className="meta">
              <code>[0]</code>, <code>[1]</code> are positions in the recorded
              vector, not names — this dataset publishes none for those columns.
            </p>
          )}

          {data.absent.length > 0 && (
            <ul className="tl-absent">
              {data.absent.map((gone) => (
                <li key={gone.column}>
                  <code>{gone.column}</code>
                  <span className="meta">{gone.why}</span>
                </li>
              ))}
            </ul>
          )}

          <div className="hint">{data.means}</div>
        </>
      )}
    </div>
  );
}
