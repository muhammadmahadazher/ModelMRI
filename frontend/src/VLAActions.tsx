import { CSSProperties, useEffect, useState } from "react";
import {
  ApiError,
  errorText,
  getPolicy,
  PolicyStatus,
  VLAActionCost,
  VLAChunkConsistency,
  VLACompare,
  vlaActionCost,
  vlaCompareActions,
  VLAKnockout,
  vlaKnockoutInputs,
  VLASwap,
  vlaSwapInstruction,
} from "./api";
import { measured, signed } from "./measured";

/**
 * What the policy would DO, next to what the human actually did.
 *
 * The panel above says where a policy LOOKED. This one needs the other half —
 * the action expert, which lives in a second process because lerobot's pins
 * cannot share an environment with ModelMRI's — and every measurement here is
 * shaped by the same worry: an action curve is the most persuasive-looking
 * thing this project draws, and a reader will conclude something about a robot
 * from the gap between two lines.
 *
 * So each of the three refuses more than it reports, and this panel's job is
 * to draw the refusals as findings rather than as failures:
 *
 *   - **Compare** refuses when the two sides' units disagree, and it refuses
 *     BEFORE spending a single forward pass, because the answer never depended
 *     on them. "The policy emits 6 action dimensions and this dataset recorded
 *     2" is the result of the measurement.
 *   - **Instruction swap** refuses a single-task dataset by name — a distractor
 *     instruction invented here would measure a sentence somebody wrote — and
 *     refuses a deterministic policy, whose own sampling spread is zero and
 *     cannot be a denominator.
 *   - **Knockout** reports `ratio_to_sampling: null` when the policy's noise
 *     could not be measured. That bar is still a real number; it simply has no
 *     denominator, and this panel says so rather than drawing a zero.
 *
 * The cost is on screen before the button, from `/api/vla/actions/cost`, which
 * quotes frames and passes and deliberately never quotes seconds.
 */

type Outcome<T> = { data: T | null; refusal: string; err: string };

function blank<T>(): Outcome<T> {
  return { data: null, refusal: "", err: "" };
}

/** Enough digits to distinguish two readings, whatever the units are.
 *
 *  Chosen once per VECTOR rather than per element — deciding element by
 *  element prints two axes of one measurement in two different formats, which
 *  reads as though they are different kinds of number.
 */
function vec(xs: number[]): string {
  if (!xs.length) return "—";
  const peak = Math.max(...xs.map(Math.abs), 0);
  const dp = peak >= 100 ? 1 : peak >= 1 ? 3 : 5;
  return xs.map((v) => v.toFixed(dp)).join(", ");
}

/** A ratio the server could not compute. `null` is UNKNOWN, never 0 — a bar
 *  with no denominator is not a bar that tied with the noise. */
function ratio(v: number | null): string {
  return v === null ? "no denominator" : `${v.toFixed(2)}× noise`;
}

/**
 * The picture the chunk-consistency numbers are of: one chunk, the chunk
 * predicted `ahead` frames later, and the timesteps both of them describe.
 *
 * DRAWN FROM THE MEASUREMENT, not from constants. `horizon` is how many steps
 * this policy actually returned and `ahead` is the nearest gap between two
 * frames this run sampled, so the shaded region is the real overlap rather
 * than an illustration of a typical one. A legend with a baked-in H would be a
 * picture of some other policy sitting above these numbers.
 *
 * The bar length is a fixed 200 user units across `ahead + horizon` frames, so
 * a 50-step chunk and a 3-step chunk draw at the same size — this says WHICH
 * PART overlaps, and a diagram that grew with H would be unreadable for the
 * ACT-shaped policies that are the common case.
 */
function OverlapLegend({
  horizon,
  ahead,
}: {
  horizon: number;
  ahead: number;
}) {
  const span = ahead + horizon;
  const unit = 200 / span;
  const gutter = 78;
  const at = (frame: number) => gutter + frame * unit;
  const shared = Math.max(0, horizon - ahead);
  return (
    <svg
      className="vla-legend"
      viewBox={`0 0 ${gutter + 200 + 4} 52`}
      role="img"
      aria-label={
        `The chunk predicted at a frame covers ${horizon} steps. The chunk ` +
        `predicted ${ahead} frame${ahead === 1 ? "" : "s"} later starts ` +
        `${ahead} step${ahead === 1 ? "" : "s"} into it, so the two describe ` +
        `${shared} of the same timestep${shared === 1 ? "" : "s"}.`
      }
    >
      {shared > 0 && (
        <rect
          className="vla-legend-shared"
          x={at(ahead)}
          y={10}
          width={at(horizon) - at(ahead)}
          height={36}
          rx="2"
        />
      )}
      <text className="vla-legend-tag" x={at(ahead)} y={8}>
        {shared} shared
      </text>
      <text x={gutter - 6} y={24} textAnchor="end">
        chunk at t
      </text>
      <rect
        className="vla-legend-a"
        x={at(0)}
        y={16}
        width={horizon * unit}
        height={9}
        rx="2"
      />
      <text x={gutter - 6} y={42} textAnchor="end">
        chunk at t+{ahead}
      </text>
      <rect
        className="vla-legend-b"
        x={at(ahead)}
        y={34}
        width={horizon * unit}
        height={9}
        rx="2"
      />
    </svg>
  );
}

/**
 * How far the policy's own successive chunks disagree about frames they share.
 *
 * A SEPARATE measurement from the one above it, and not a second opinion on
 * it. Both operands here are predictions — step `dt+k` of the chunk from frame
 * `t` against step `k` of the chunk from frame `t+dt`, two claims about ONE
 * absolute timestep made an observation apart — and the recorded action is on
 * neither side. A policy that predicts the same wrong action every time scores
 * perfectly here, which is why the block carries its own sentence rather than
 * borrowing the "NOT GROUND TRUTH" one above.
 *
 * `measurable: false` is a STATE, drawn as the server's sentence verbatim. It
 * names the stride, the horizon and the stride that would have worked, and
 * paraphrasing it into "not available" would throw away the only actionable
 * half. A zero would be worse still: 0.0 is what a policy that agreed with
 * itself perfectly scores.
 */
function ChunkConsistency({ block }: { block: VLAChunkConsistency }) {
  if (!block.measurable) {
    return (
      <div className="vla-consist">
        <span className="judge-tag">chunk consistency</span>
        <div className="vla-refusal">
          <p>{block.means}</p>
        </div>
      </div>
    );
  }

  const rows = block.by_steps_ahead;
  // ONE scale across every small multiple. Scaling each row to its own widest
  // value would draw the tightest bucket and the loosest identically, which is
  // exactly the comparison this strip exists to make.
  const widest = rows.reduce((m, r) => Math.max(m, r.p75, r.median), 0) || 1;
  // `horizon` is null when the chunks were not all the same length; the legend
  // then draws the longest of them, and the sentence below says it varied.
  const horizon = block.horizon ?? block.horizon_max ?? 0;
  const nearest = rows[0]?.steps_ahead ?? block.stride;

  return (
    <div className="vla-consist">
      <span className="judge-tag">chunk consistency</span>
      <p className="meta">
        The same policy, one observation later, about a frame it had already
        committed to. Free: these chunks were computed for the rows above and
        everything past their first step used to be discarded.
      </p>
      <div className="vla-consist-key">
        {horizon > 0 && nearest > 0 && (
          <OverlapLegend horizon={horizon} ahead={nearest} />
        )}
        <p className="meta">
          {/* `measured`, not `toFixed`. A near-consistent policy diverges by
              3e-7 and four fixed places print that as 0.0000 — the reading of
              a policy that never revised anything, on one that just did. */}
          median <b>{measured(block.median, 4)}</b> over{" "}
          <b>{block.overlapping_steps}</b> shared timestep
          {block.overlapping_steps === 1 ? "" : "s"} from <b>{block.pairs}</b>{" "}
          chunk pair{block.pairs === 1 ? "" : "s"} · middle half{" "}
          {measured(block.p25, 4)}–{measured(block.p75, 4)} · stride{" "}
          {block.stride} ·{" "}
          {block.horizon === null ? (
            <span className="aud-unknown">
              horizon varied, {block.horizon_min}–{block.horizon_max} steps
            </span>
          ) : (
            <>{block.horizon}-step horizon</>
          )}
        </p>
      </div>
      {/* The band is the middle half and the rule is the median: one line on
          its own would be a median pretending to be a distribution. The same
          two primitives the model-diff panel draws its per-layer spread with. */}
      <ol className="vla-ahead stagger">
        {rows.map((r, i) => (
          <li key={r.steps_ahead} style={{ "--i": i } as CSSProperties}>
            <span className="mid">
              +{r.steps_ahead} frame{r.steps_ahead === 1 ? "" : "s"}
            </span>
            <span className="vla-track">
              <span
                className="vla-band"
                style={{
                  left: `${Math.min(100, (r.p25 / widest) * 100)}%`,
                  width: `${Math.max(1, Math.min(100, ((r.p75 - r.p25) / widest) * 100))}%`,
                }}
              />
              <span
                className="vla-median"
                style={{ left: `${Math.min(100, (r.median / widest) * 100)}%` }}
              />
            </span>
            <span className="mid">{measured(r.median, 4)}</span>
            {/* `n` beside every median. A median over one shared timestep and
                a median over eighty are different claims. */}
            <span className="meta">
              {r.overlapping_steps} shared · {r.pairs} pair
              {r.pairs === 1 ? "" : "s"}
            </span>
          </li>
        ))}
      </ol>
      {rows.length > 1 && nearest > 0 && (
        <p className="meta vla-cap">
          Rows past +{nearest} compare NON-ADJACENT chunks. The published
          detectors compare consecutive inference steps only, so that part of
          this curve is this project&rsquo;s extension of them rather than
          something borrowed.
        </p>
      )}
      <ul className="vla-bias">
        {block.by_dimension.map((d) => (
          <li key={d.dimension}>
            <span className="mid">{d.name ?? `dim ${d.dimension}`}</span>
            {/* Signed, because "it keeps revising this joint upward" and "this
                joint jitters" are different findings and the absolute value
                beside it cannot tell them apart on its own. */}
            <span className="mid">{signed(d.revision_bias, 4)}</span>
            <span className="meta">±{measured(d.disagreement, 4)}</span>
          </li>
        ))}
      </ul>
      <p className="meta">
        Signed mean of the revision per dimension, with the median absolute
        beside it. A policy that keeps pulling one joint the same way as it
        gets closer looks nothing like one that jitters around it, and only the
        sign tells them apart.
      </p>
      {block.worst_pair && (
        <p className="meta">
          widest single disagreement{" "}
          <b>{measured(block.worst_pair.distance, 4)}</b> — the chunk from frame{" "}
          {block.worst_pair.t_earlier} against the chunk from frame{" "}
          {block.worst_pair.t_later}, both about frame{" "}
          <b>{block.worst_pair.t_later + block.worst_pair.step}</b>
        </p>
      )}
      {block.pairs_skipped_same_frame ? (
        <p className="meta vla-cap">
          <b>{block.pairs_skipped_same_frame}</b> chunk pair
          {block.pairs_skipped_same_frame === 1 ? " was" : "s were"} predicted
          at the identical frame and skipped. Two chunks at one frame differ by
          sampling noise, which the instruction swap measures and this does not.
        </p>
      ) : null}
      <p className="meta vla-means">{block.means}</p>
    </div>
  );
}

export default function VLAActions({
  episode,
  timestep,
}: {
  episode: number;
  timestep: number;
}) {
  const [policy, setPolicy] = useState<PolicyStatus | null>(null);
  const [asked, setAsked] = useState(false);
  const [stride, setStride] = useState(0);
  // Blank is not seed 0. Blank means "do not fix the sampler", which is a
  // different request and a different claim about the result.
  const [seed, setSeed] = useState("");
  const [cost, setCost] = useState<VLAActionCost | null>(null);
  const [costErr, setCostErr] = useState("");
  const [busy, setBusy] = useState("");

  const [cmp, setCmp] = useState<Outcome<VLACompare>>(blank<VLACompare>());
  const [swap, setSwap] = useState<Outcome<VLASwap>>(blank<VLASwap>());
  const [knock, setKnock] = useState<Outcome<VLAKnockout>>(blank<VLAKnockout>());

  const seedValue = seed.trim() === "" ? null : Number(seed);

  useEffect(() => {
    let live = true;
    void getPolicy()
      .then((p) => live && setPolicy(p))
      .catch(() => undefined)
      .finally(() => live && setAsked(true));
    return () => {
      live = false;
    };
  }, []);

  // The cost follows the knobs, so it is on screen BEFORE the button rather
  // than after the wait. It opens no policy — it counts frames.
  useEffect(() => {
    let live = true;
    void vlaActionCost(episode, stride)
      .then((c) => {
        if (!live) return;
        setCost(c);
        setCostErr("");
      })
      .catch((e) => {
        if (!live) return;
        setCost(null);
        setCostErr(errorText(e));
      });
    return () => {
      live = false;
    };
  }, [episode, stride]);

  // A swap and a knockout belong to ONE frame, and a comparison to one episode.
  // Leaving either up while the scrubber moves states something that was never
  // measured — the same staleness the attention panel guards with a badge.
  useEffect(() => {
    setSwap(blank<VLASwap>());
    setKnock(blank<VLAKnockout>());
  }, [episode, timestep]);

  useEffect(() => {
    setCmp(blank<VLACompare>());
  }, [episode, stride]);

  async function attempt<T>(
    tag: string,
    call: () => Promise<T>,
    set: (o: Outcome<T>) => void,
  ) {
    setBusy(tag);
    set(blank<T>());
    try {
      set({ data: await call(), refusal: "", err: "" });
    } catch (e) {
      // 409 is a deliberate no with a sentence behind it — different units, a
      // single-task dataset, a deterministic head, no sidecar. Every one of
      // those is the measurement's answer. Anything else is a fault.
      if (e instanceof ApiError && e.status === 409)
        set({ data: null, refusal: errorText(e), err: "" });
      else set({ data: null, refusal: "", err: errorText(e) });
    } finally {
      setBusy("");
    }
  }

  const running = policy?.running ?? false;
  const ranked = cmp.data
    ? [...cmp.data.rows].sort((a, b) => b.distance - a.distance)
    : [];
  const shownRows = ranked.slice(0, 10);
  const widestKnock = knock.data
    ? knock.data.rows.reduce((m, r) => Math.max(m, r.distance), 0) || 1
    : 1;
  const widestArm = swap.data
    ? swap.data.arms.reduce((m, a) => Math.max(m, a.distance_from_own), 0) || 1
    : 1;

  return (
    <div className="vla-actions">
      <div className="row vla-opts">
        <span className="meta">
          what it would <b>do</b> — the action expert answers from its own
          process, and a recorded action is one human demonstration rather than
          ground truth
        </span>
      </div>

      {/* The sidecar's own sentence, verbatim. It names the checkpoint, the
          revision and the device the actions came from, or says why there are
          none — and either way it is the provenance of every number below. */}
      {asked && (
        <p className={`meta ${running ? "vla-means" : "vla-idle"}`}>
          {policy
            ? policy.means
            : "The policy sidecar could not be asked, so nothing here knows whether an action expert is available."}
        </p>
      )}

      <div className="row vla-opts">
        <label className="meta" htmlFor="va-stride">
          stride
        </label>
        <input
          id="va-stride"
          className="vla-num"
          type="number"
          min={0}
          value={stride}
          onChange={(e) => setStride(Math.max(0, Number(e.target.value)))}
        />
        <span className="meta">0 lets the server pick one that fits</span>
        <label className="meta" htmlFor="va-seed">
          seed
        </label>
        <input
          id="va-seed"
          className="vla-num"
          type="number"
          min={0}
          placeholder="none"
          value={seed}
          onChange={(e) => setSeed(e.target.value)}
        />
        <span className="meta">
          blank does not fix the sampler, which is not the same as seed 0
        </span>
      </div>

      {/* THE PRICE, BEFORE THE SPEND. Frames and passes, never seconds — the
          server refuses to quote a duration it has not measured on this
          machine, and the one timing in its sentence says where it came from. */}
      {cost && (
        <div className="vla-plan">
          <p className="meta">
            <b>{cost.passes}</b> forward pass{cost.passes === 1 ? "" : "es"} ·{" "}
            {cost.frames_measured} of {cost.frames_in_episode} frames at stride{" "}
            {cost.stride} · <b>{cost.frames_skipped}</b> frame
            {cost.frames_skipped === 1 ? "" : "s"} skipped
          </p>
          <p className="meta vla-means">{cost.means}</p>
        </div>
      )}
      {costErr && <p className="hint">{costErr}</p>}

      <div className="row" style={{ margin: "8px 0" }}>
        <button
          className="cta"
          disabled={busy !== "" || !running || !cost}
          onClick={() =>
            void attempt(
              "compare",
              () => vlaCompareActions({ episode, stride, seed: seedValue }),
              setCmp,
            )
          }
        >
          {busy === "compare"
            ? "Running the episode…"
            : cost
              ? `Predicted vs recorded · ${cost.passes} passes`
              : "Predicted vs recorded"}
        </button>
        <button
          className="ghost sm"
          disabled={busy !== "" || !running}
          onClick={() =>
            void attempt(
              "swap",
              () => vlaSwapInstruction({ episode, t: timestep, seed: seedValue }),
              setSwap,
            )
          }
          title="one frame, every distinct task string this dataset contains, against this policy's own sampling spread"
        >
          {busy === "swap" ? "Swapping…" : "Swap the instruction (this frame)"}
        </button>
        <button
          className="ghost sm"
          disabled={busy !== "" || !running}
          onClick={() =>
            void attempt(
              "knockout",
              () => vlaKnockoutInputs({ episode, t: timestep, seed: seedValue }),
              setKnock,
            )
          }
          title="one bar per input, each replaced alone by its episode mean"
        >
          {busy === "knockout" ? "Knocking out…" : "Knock out each input (this frame)"}
        </button>
      </div>

      {/* ── predicted against recorded ──────────────────────────────── */}
      {cmp.refusal && (
        <div className="vla-refusal">
          <span className="judge-tag">
            refused before a single pass was spent
          </span>
          <p>{cmp.refusal}</p>
        </div>
      )}
      {cmp.err && <div className="hint err">{cmp.err}</div>}
      {cmp.data && (
        <div className="vla-result">
          <p className="meta">
            {cmp.data.frames_measured} of {cmp.data.frames_in_episode} frames at
            stride {cmp.data.stride} · <b>{cmp.data.frames_skipped}</b> skipped ·{" "}
            {cmp.data.dimensions} action dimension
            {cmp.data.dimensions === 1 ? "" : "s"} ·{" "}
            {/* `null` is "no seed was fixed", and re-running then gives a
                different curve. It is not seed 0. */}
            {cmp.data.seed === null ? (
              <span className="aud-unknown">no seed was fixed</span>
            ) : (
              <>seed {cmp.data.seed}</>
            )}
          </p>
          {cmp.data.joint_names.length === 0 && (
            <p className="meta aud-unknown">
              This dataset named no usable dimensions — or named a count that
              disagreed with the policy&rsquo;s width, in which case the server
              drops the whole list rather than mislabelling one joint. The
              dimensions below are numbered, not named.
            </p>
          )}
          <p className="meta">
            largest gap at frame <b>{cmp.data.worst_frame}</b> (
            {cmp.data.worst_distance.toFixed(4)})
          </p>
          <ul className="vla-bias">
            {cmp.data.bias.map((b, d) => (
              <li key={d}>
                <span className="mid">{cmp.data?.joint_names[d] ?? `dim ${d}`}</span>
                <span className="mid">
                  {b >= 0 ? "+" : ""}
                  {b.toFixed(4)}
                </span>
              </li>
            ))}
          </ul>
          <p className="meta">
            Signed mean of the per-dimension difference — bias, not error. A
            policy that reaches the same amount further every frame looks
            exactly like this; one that is randomly wrong in both directions
            averages to nothing here.
          </p>
          {/* Under the per-dimension chart, and deliberately not inside it.
              Everything above this line differences a PREDICTION against a
              RECORDING; everything below it differences two predictions
              against each other and never touches the demonstrator. Mixing
              them into one table would be the exact reading the comment in
              `_run_compare` exists to prevent, one section over. */}
          <ChunkConsistency block={cmp.data.chunk_consistency} />
          <ol className="vla-rank stagger">
            {shownRows.map((r, i) => (
              <li key={r.t} style={{ "--i": i } as CSSProperties}>
                <span className="mid">frame {r.t}</span>
                <span className="vla-track">
                  <span
                    className="vla-bar"
                    style={{
                      width: `${Math.min(100, (r.distance / (ranked[0]?.distance || 1)) * 100)}%`,
                    }}
                  />
                </span>
                <span className="mid">{r.distance.toFixed(4)}</span>
              </li>
            ))}
          </ol>
          {/* The list is capped, so it says it is capped. The top of a list is
              read as "these are the ones".

              THERE WERE TWO OF THESE, on the identical predicate, and the
              first described a measurement this request never ran: "N more
              input(s) were knocked out and moved the action less". Nothing was
              knocked out — `ranked` is `cmp.data.rows`, which is the
              predicted-versus-recorded comparison, and its rows are FRAMES.
              The sentence was copied from the knockout block below, where it
              is true, and it fired on the ordinary path: the server plans up
              to MAX_FRAMES_PER_RUN = 64 frames, so any episode past ten
              printed a fabricated experiment above the correct sentence. */}
          {ranked.length > shownRows.length && (
            <p className="meta">
              Showing the {shownRows.length} largest gaps of{" "}
              {ranked.length} frames measured, ranked by distance rather than by
              time. The rest were measured and are not on screen.
            </p>
          )}
          <p className="meta vla-means">{cmp.data.means}</p>
        </div>
      )}

      {/* ── instruction swap ────────────────────────────────────────── */}
      {swap.refusal && (
        <div className="vla-refusal">
          <span className="judge-tag">refused, and that is the reading</span>
          <p>{swap.refusal}</p>
        </div>
      )}
      {swap.err && <div className="hint err">{swap.err}</div>}
      {swap.data && (
        <div className="vla-result">
          {/* The reference is this policy's OWN sampling variance, so the two
              readings are opposites of one measurement rather than a pass and a
              fail against somebody else's threshold. */}
          <div className={`vla-verdict ${swap.data.listens ? "agree" : "inverted"}`}>
            {swap.data.listens ? (
              <>
                Swapping the instruction moves this policy&rsquo;s action{" "}
                <b>{swap.data.ratio.toFixed(2)}×</b> as much as re-rolling the
                sampler does, on this frame.
              </>
            ) : (
              <b>
                Swapping the instruction moves the action LESS than re-rolling
                the sampler does ({swap.data.ratio.toFixed(2)}×). On this frame,
                changing what the policy was told mattered less than changing
                the random seed.
              </b>
            )}
          </div>
          <p className="meta">
            {swap.data.instructions_tried} distinct instruction
            {swap.data.instructions_tried === 1 ? "" : "s"} spreading{" "}
            {swap.data.instruction_spread.toFixed(4)}, against{" "}
            {swap.data.seeds} seed{swap.data.seeds === 1 ? "" : "s"} spreading{" "}
            {swap.data.sampling_spread.toFixed(4)}
          </p>
          {/* A CAP. Above zero, the instruction spread is a lower bound. */}
          {swap.data.instructions_dropped > 0 && (
            <p className="meta vla-cap">
              <b>{swap.data.instructions_dropped}</b> further distinct
              instruction
              {swap.data.instructions_dropped === 1 ? "" : "s"} in this dataset
              were not tried, so the spread above is a lower bound.
            </p>
          )}
          <ol className="vla-arms stagger">
            {swap.data.arms.map((a, i) => (
              <li
                key={a.instruction}
                className={a.is_own ? "own" : ""}
                style={{ "--i": i } as CSSProperties}
              >
                <span className="mid vla-arm-name">
                  {a.instruction}
                  {a.is_own && <span className="pill tiny">its own</span>}
                </span>
                <span className="vla-track">
                  <span
                    className="vla-bar"
                    style={{
                      width: `${Math.min(100, (a.distance_from_own / widestArm) * 100)}%`,
                    }}
                  />
                </span>
                <span className="mid">{a.distance_from_own.toFixed(4)}</span>
                <span className="meta">{a.ratio_to_sampling.toFixed(2)}× noise</span>
              </li>
            ))}
          </ol>
          <p className="meta vla-means">{swap.data.means}</p>
        </div>
      )}

      {/* ── input-stream knockout ───────────────────────────────────── */}
      {knock.refusal && (
        <div className="vla-refusal">
          <span className="judge-tag">refused, and that is the reading</span>
          <p>{knock.refusal}</p>
        </div>
      )}
      {knock.err && <div className="hint err">{knock.err}</div>}
      {knock.data && (
        <div className="vla-result">
          <p className="meta">
            {knock.data.streams} input stream
            {knock.data.streams === 1 ? "" : "s"} ·{" "}
            {/* `null` is "it could not be measured", NOT a spread of zero. */}
            {knock.data.sampling_spread === null ? (
              <span className="aud-unknown">
                this policy&rsquo;s sampling spread could not be measured, so
                nothing below says whether a bar beats its own noise
              </span>
            ) : (
              <>sampling spread {knock.data.sampling_spread.toFixed(4)}</>
            )}
          </p>
          {/* `vla-blocks` for the three bar states it already draws — clears,
              does not clear, and untested — and `vla-knock` only to widen the
              first column: these labels are "camera1 → episode mean", not
              "r3 c7", and 5rem clips every one of them. */}
          <ol className="vla-blocks vla-knock stagger">
            {knock.data.rows.map((r, i) => (
              <li
                key={r.stream}
                className={
                  r.above_noise === null
                    ? "untested"
                    : r.above_noise
                      ? "clears"
                      : "null"
                }
                style={{ "--i": i } as CSSProperties}
              >
                <span className="mid">{r.label}</span>
                <span className="vla-track">
                  <span
                    className="vla-bar"
                    style={{
                      width: `${Math.min(100, (r.distance / widestKnock) * 100)}%`,
                    }}
                  />
                </span>
                <span className="mid">{r.distance.toFixed(4)}</span>
                <span className="meta">
                  {/* Three states, never two: above its noise, below it, or no
                      denominator at all. */}
                  {r.above_noise === null
                    ? ratio(r.ratio_to_sampling)
                    : r.above_noise
                      ? `${ratio(r.ratio_to_sampling)} — beats it`
                      : `${ratio(r.ratio_to_sampling)} — inside it`}
                </span>
              </li>
            ))}
          </ol>
          <p className="meta">
            baseline action, every input intact: [{vec(knock.data.baseline)}]
          </p>
          {/* VERBATIM, and it is written with paragraph breaks: mean
              substitution is a specific baseline rather than removal, and the
              bars do not add up. Both sentences are the point. */}
          <p className="meta vla-means vla-pre">{knock.data.means}</p>
        </div>
      )}
    </div>
  );
}
