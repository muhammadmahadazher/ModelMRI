import { CSSProperties, useEffect, useState } from "react";
import { measured, ordinal, percent } from "./measured";
import {
  errorText,
  occludeFrame,
  occlusionCost,
  OcclusionMap,
  runVlaSweep,
  shareVlaFinding,
  VLASweep,
  vlaSweepCost,
} from "./api";

/**
 * The causal half of the robot panel: what the vision DEPENDED on.
 *
 * The panel above this paints attention, and the field has already settled
 * that attention is the weak version — interventional masking beats attention
 * weights on explanation fidelity. So this sits directly underneath it, on
 * purpose: the two maps are meant to be read together, and the number that
 * matters most is how far apart they rank the same blocks.
 *
 * Three things it draws that an attention map cannot:
 *
 *   - THE DISAGREEMENT, as a number. Spearman between the causal map and the
 *     attention map for this frame. On somebody's own checkpoint the two
 *     disagreeing is the finding, and it is invisible from either alone.
 *   - THE CONTROL, per block. A block is drawn as clearing or not clearing
 *     same-area occlusions at random locations, never as a shorter bar.
 *   - THE COST, before the run. A fine grid is over a thousand tower passes
 *     and nobody should discover that by waiting.
 */

const BASELINES = [
  ["episode_mean", "the average pixel of this episode"],
  ["midpoint", "the tower's own normalisation centre"],
] as const;

const METRICS = [
  ["attention_entropy", "attention entropy · 1 pass per frame"],
  ["occlusion_peak", "strongest causal block · dozens of passes per frame"],
] as const;

export default function VLACausal({
  episode,
  timestep,
  layer,
  ready,
}: {
  episode: number;
  timestep: number;
  layer: number;
  /** The tower has to be loaded and a frame analysed; without both there is
   *  nothing to occlude and no attention map to compare against. */
  ready: boolean;
}) {
  const [baseline, setBaseline] = useState<string>("episode_mean");
  const [stride, setStride] = useState(4);
  const [cost, setCost] = useState<{ blocks: number; passes: number } | null>(null);
  const [map, setMap] = useState<OcclusionMap | null>(null);
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");

  const [metric, setMetric] = useState<string>("attention_entropy");
  const [frameStride, setFrameStride] = useState(25);
  const [sweepCost, setSweepCost] = useState<{
    frames: number;
    frames_total: number;
    passes: number;
    coverage: number;
    seconds: number | null;
    seconds_from: string;
  } | null>(null);
  const [sweep, setSweep] = useState<VLASweep | null>(null);

  // A causal map belongs to ONE frame. Moving the slider makes it a map of a
  // picture that is no longer on screen, which is exactly the confusion the
  // attention panel above already guards against with its stale badge.
  //
  // `layer` belongs in here too: the agreement is measured against ONE layer's
  // attention and changes with it — measured on a real SmolVLA checkpoint,
  // -0.053 at layer 0 against -0.103 at layer 11. Leaving the old number up
  // under a new layer picker states something that was never measured.
  useEffect(() => {
    setMap(null);
    setErr("");
  }, [episode, timestep, layer]);

  // The cost is fetched whenever the knob moves, so it is on screen BEFORE
  // the button rather than after the wait.
  useEffect(() => {
    if (!ready) return;
    let live = true;
    void occlusionCost(stride)
      .then((c) => live && setCost(c))
      .catch(() => live && setCost(null));
    return () => {
      live = false;
    };
  }, [stride, ready]);

  useEffect(() => {
    if (!ready) return;
    let live = true;
    void vlaSweepCost(metric, frameStride)
      .then((c) => live && setSweepCost(c))
      .catch(() => live && setSweepCost(null));
    return () => {
      live = false;
    };
  }, [metric, frameStride, ready]);

  async function occlude() {
    setBusy("occlude");
    setErr("");
    setMap(null);
    try {
      setMap(await occludeFrame({ episode, t: timestep, baseline, stride, layer }));
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }

  async function runSweep() {
    setBusy("sweep");
    setErr("");
    setSweep(null);
    try {
      setSweep(await runVlaSweep({ metric, frame_stride: frameStride }));
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }

  async function share() {
    setBusy("share");
    setErr("");
    try {
      const blob = await shareVlaFinding({
        episode,
        t: timestep,
        layer,
        occlusion: map ?? undefined,
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `robot-finding-ep${episode}-t${timestep}.mri`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }

  if (!ready) return null;

  const widest = map
    ? map.blocks.reduce((m, b) => Math.max(m, b.shift, b.control_max ?? 0), 0) || 1
    : 1;
  const cleared = map ? map.blocks.filter((b) => b.clears_control) : [];
  const agreement = map?.attention_agreement ?? null;

  return (
    <div className="vla-causal">
      <div className="row">
        <span className="meta">
          what it <b>depended on</b> — the map above is where the policy
          looked, which the field has measured is the weaker explanation
        </span>
      </div>

      {/* ── occlusion ───────────────────────────────────────────────── */}
      <div className="row vla-opts">
        <label className="meta" htmlFor="vc-fill">
          fill
        </label>
        <select
          id="vc-fill"
          value={baseline}
          onChange={(e) => setBaseline(e.target.value)}
        >
          {BASELINES.map(([id, label]) => (
            <option key={id} value={id}>
              {label}
            </option>
          ))}
        </select>
        <label className="meta" htmlFor="vc-stride">
          block size
        </label>
        <select
          id="vc-stride"
          value={stride}
          onChange={(e) => setStride(Number(e.target.value))}
        >
          {[1, 2, 4, 8].map((s) => (
            <option key={s} value={s}>
              {s} patch{s === 1 ? "" : "es"}
            </option>
          ))}
        </select>
        {/* The cost, BEFORE the button. A stride of 1 on a 32x32 grid is over
            a thousand tower passes and nobody should find that out by
            waiting. */}
        {cost && (
          <span className="meta vla-cost">
            {cost.blocks} blocks · {cost.passes} tower passes
          </span>
        )}
      </div>

      <div className="row" style={{ margin: "8px 0" }}>
        <button className="cta" onClick={() => void occlude()} disabled={busy !== ""}>
          {busy === "occlude" ? "Occluding every block…" : "Occlude the frame"}
        </button>
        <button
          className="ghost sm"
          onClick={() => void share()}
          disabled={busy !== ""}
          title="Writes a .mri holding this frame, its attention and its causal map"
        >
          {busy === "share" ? "Writing…" : "Share this finding (.mri)"}
        </button>
      </div>

      {err && <div className="hint err">{err}</div>}

      {map && (
        <>
          {/* The headline. Where the model LOOKED and what it DEPENDED ON
              ranking blocks differently is the whole reason this exists. */}
          <div
            className={`vla-verdict ${
              agreement === null
                ? "none"
                : agreement > 0.6
                  ? "agree"
                  : agreement < -0.6
                    ? "inverted"
                    : "differ"
            }`}
          >
            {agreement === null ? (
              <>
                No attention map for this frame, so the two cannot be compared.
                Run the policy on it first.
              </>
            ) : (
              <>
                The attention map and the causal map agree at Spearman{" "}
                <b>{agreement >= 0 ? "+" : ""}{agreement.toFixed(3)}</b>
                {/* Which layer, always. The agreement changes with it, so the
                    bare number is not a reportable one. */}
                {map.compared_layer !== null && (
                  <>
                    {" "}
                    against <b>layer {map.compared_layer}</b>
                    {map.compared_head === null || map.compared_head < 0
                      ? " averaged over its heads"
                      : ` head ${map.compared_head}`}
                  </>
                )}
                .{" "}
                {/* Three readings. A strong NEGATIVE correlation says the
                    blocks attention ranked highest are the ones the
                    representation depended on least — a relationship, not the
                    absence of one, and the same sentence must not serve both
                    -0.9 and -0.05. */}
                {agreement > 0.6 ? (
                  <>They largely rank the same blocks — worth knowing, and not the usual result.</>
                ) : agreement < -0.6 ? (
                  <b>
                    The rankings are inverted: the blocks this frame&rsquo;s
                    attention ranked highest are the ones its representation
                    depended on least.
                  </b>
                ) : (
                  <b>
                    Where the policy looked and what its representation
                    depended on are ranking these blocks differently.
                  </b>
                )}
              </>
            )}
          </div>

          <p className="meta">
            {map.n_blocks} blocks at stride {map.stride}, {map.passes} passes in{" "}
            {map.seconds}s ·{" "}
            {cleared.length
              ? `${cleared.length} of ${map.n_controlled} tested beat every same-area occlusion elsewhere`
              : `none of the ${map.n_controlled} tested beat a same-area occlusion elsewhere`}
          </p>

          <ol className="vla-blocks stagger">
            {map.blocks.slice(0, 10).map((b, i) => (
              <li
                key={`${b.row}-${b.col}`}
                className={
                  b.clears_control === null
                    ? "untested"
                    : b.clears_control
                      ? "clears"
                      : "null"
                }
                style={{ "--i": i } as CSSProperties}
              >
                <span className="mid">
                  r{b.row} c{b.col}
                </span>
                <span className="vla-track">
                  <span
                    className="vla-bar"
                    style={{ width: `${Math.min(100, (b.shift / widest) * 100)}%` }}
                  />
                  {b.control_max !== null && (
                    <span
                      className="vla-ctl"
                      style={{
                        left: `${Math.min(100, (b.control_max / widest) * 100)}%`,
                      }}
                      title={`the strongest of ${b.control_draws} same-area occlusions elsewhere reached ${b.control_max}`}
                    />
                  )}
                </span>
                <span className="mid">{b.shift.toFixed(4)}</span>
                <span className="meta">
                  {b.clears_control === null
                    ? "not tested"
                    : b.clears_control
                      ? "beats its control"
                      : "a block elsewhere did as much"}
                </span>
              </li>
            ))}
          </ol>

          {/* The list is capped, so it SAYS it is capped. The top of a list is
              read as "these are the ones", and 10 of 64 silently shown is the
              same silent-truncation defect the ablation panel carries a
              `truncated` field to avoid. */}
          {map.blocks.length > 10 && (
            <p className="meta">
              Showing the 10 strongest of {map.blocks.length}. The rest were
              measured and are in the .mri, not dropped.
            </p>
          )}

          <p className="meta vla-means">{map.means}</p>
        </>
      )}

      {/* ── cross-episode sweep ─────────────────────────────────────── */}
      <div className="row vla-opts" style={{ marginTop: 14 }}>
        <span className="meta">
          <b>across every episode</b> — one measurement, ranked, so the frame
          worth looking at can be found rather than clicked to
        </span>
      </div>
      <div className="row vla-opts">
        <select value={metric} onChange={(e) => setMetric(e.target.value)}>
          {METRICS.map(([id, label]) => (
            <option key={id} value={id}>
              {label}
            </option>
          ))}
        </select>
        <label className="meta" htmlFor="vc-fs">
          every
        </label>
        <input
          id="vc-fs"
          className="vla-num"
          type="number"
          min={1}
          value={frameStride}
          onChange={(e) => setFrameStride(Math.max(1, Number(e.target.value)))}
        />
        <span className="meta">frames</span>
        {sweepCost && (
          <span className="meta vla-cost">
            {sweepCost.frames} frames of {sweepCost.frames_total} (
            {percent(sweepCost.coverage, 1)}) · {sweepCost.passes} passes
            {/* No seconds unless this machine has been timed. A duration from
                somebody else's hardware is a number people plan around. */}
            {sweepCost.seconds !== null ? ` · ~${sweepCost.seconds}s` : ""}
          </span>
        )}
      </div>
      <div className="row" style={{ margin: "8px 0" }}>
        <button className="cta" onClick={() => void runSweep()} disabled={busy !== ""}>
          {busy === "sweep" ? "Sweeping every episode…" : "Sweep the dataset"}
        </button>
      </div>

      {sweep && (
        <>
          {/* The stride, first. A strided ranking can miss the worst frame
              entirely, and the top of a list is read as the worst thing there
              is. */}
          <div className="vla-verdict differ">
            <b>
              {sweep.frame_stride === 1
                ? "Every frame."
                : `Every ${ordinal(sweep.frame_stride)} frame.`}
            </b>{" "}
            This ranks{" "}
            {sweep.n_frames} of {sweep.frames_total} frames by{" "}
            <b>{sweep.metric}</b> and nothing else — the top of it is the worst
            frame <i>that was sampled</i>, which is not the same claim.
          </div>
          <ol className="vla-rank">
            {/* Eight of however many the sweep measured. The count is
                rendered under the list; a strip that simply stops reads as
                the whole ranking. */}
            {sweep.rows.slice(0, 8).map((r) => (
              <li key={`${r.episode}-${r.timestep}`}>
                <span className="mid">
                  ep {r.episode} · t {r.timestep}
                </span>
                <span className="vla-track">
                  <span
                    className="vla-bar"
                    style={{
                      width: `${
                        sweep.strip.high != null &&
                        sweep.strip.low != null &&
                        sweep.strip.high > sweep.strip.low
                          ? ((r.value - sweep.strip.low) /
                              (sweep.strip.high - sweep.strip.low)) *
                            100
                          : 100
                      }%`,
                    }}
                  />
                </span>
                <span className="mid">{measured(r.value, 4)}</span>
              </li>
            ))}
          </ol>
          {sweep.rows.length > 8 && (
            <p className="meta">
              {sweep.rows.length - 8} more frame(s) were measured and ranked
              lower. The strip shows the top eight; the sweep read them all.
            </p>
          )}
          {sweep.rows.length > 8 && (
            <p className="meta">
              Showing the 8 highest of {sweep.rows.length} sampled frames.
            </p>
          )}
          <p className="meta vla-means">{sweep.means}</p>
        </>
      )}
    </div>
  );
}
