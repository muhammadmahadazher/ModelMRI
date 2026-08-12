import { useEffect, useState } from "react";
import { useScanOnData } from "./useScanOnData";
import {
  Ablation,
  AttentionData,
  AttentionDiff,
  TokenAttribution,
  TokenScore,
  attributeTokens,
  errorText,
  exportSession,
  getAttention,
  getAttentionDiff,
  getAttentionMeta,
  rankHeads,
} from "./api";
import ArcCanvas from "./ArcCanvas";
import { DEMO } from "./demo";
import { VIEWER } from "./viewer";

export default function AttentionPanel({
  epoch,
  replay,
}: {
  epoch: number;
  replay?: boolean;
}) {
  const scanRef = useScanOnData(epoch);
  const [layers, setLayers] = useState(0);
  const [heads, setHeads] = useState(0);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [data, setData] = useState<AttentionData | null>(null);
  const [info, setInfo] = useState("");

  /** An instrument says where it is *within a range*, and pads so the row
   *  does not reflow as the number crosses ten. "L 09/18" beats "layer 9". */
  const dial = (label: string, i: number, n: number) => {
    const w = String(Math.max(n - 1, 0)).length;
    return `${label} ${String(i).padStart(w, "0")}/${String(Math.max(n - 1, 0))}`;
  };

  const [err, setErr] = useState("");
  const [ranked, setRanked] = useState<Ablation | null>(null);
  const [ranking, setRanking] = useState(false);
  const [baseline, setBaseline] = useState<"zero" | "mean">("zero");
  // Seconds per forward pass, measured on THIS model. Null until one
  // ranking has run — an estimate before that would be a number we made up.
  const [secPerPass, setSecPerPass] = useState<number | null>(null);
  // A comparison against the same generation with one head removed. Null
  // means we are showing the run itself rather than a difference.
  const [diff, setDiff] = useState<AttentionDiff | null>(null);
  // Which of the tokens moved the answer, and at which position. The position
  // is not decoration around the measurement, it IS the measurement's scope —
  // there is no version of this where the position is fixed and the claim
  // stays true — so `pinned` is the strip's answer to "attribute where next"
  // and `attr.position` is where the result on screen was actually taken.
  const [attr, setAttr] = useState<TokenAttribution | null>(null);
  const [attributing, setAttributing] = useState(false);
  const [attrErr, setAttrErr] = useState("");
  const [pinned, setPinned] = useState(-1);

  /** Rank the tokens at the pinned position, or at the server's own default.
   *
   *  Failures land in their own slot rather than the panel's `err`. The
   *  refusals this endpoint can answer with — "the token being attributed is
   *  a control token", "position N has read nothing but index 0" — are the
   *  feature, and the shared error branch would wrap them in "Could not
   *  compute this attention map … pick a different layer", which is advice
   *  that cannot work and about a control the reader did not touch.
   */
  async function attribute() {
    setAttributing(true);
    setAttrErr("");
    try {
      setAttr(await attributeTokens(pinned >= 0 ? pinned : undefined));
    } catch (e) {
      setAttrErr(errorText(e));
      setAttr(null);
    } finally {
      setAttributing(false);
    }
  }

  /** Show what removing one head changes — at the first layer where it can.
   *
   *  Removing a head zeroes its OUTPUT, so the layer it lives in is computed
   *  from an unchanged input and its attention is bit-identical. Wired
   *  naively, this button compared the viewed layer against an ablation in
   *  that same layer and was therefore guaranteed to show nothing at all.
   *  The first layer that can differ is the next one, so go there.
   */
  async function compare(cutLayer: number, cutHead: number) {
    const at = Math.min(cutLayer + 1, layers - 1);
    setErr("");
    setLayer(at);
    try {
      const result = await getAttentionDiff(
        at,
        head,
        "live",
        `ablate:${cutLayer}.${cutHead}`,
      );
      setDiff(result);
    } catch (e) {
      setErr(errorText(e));
      setDiff(null);
    }
  }

  async function rank(scope: "layer" | "all" = "layer") {
    setRanking(true);
    setErr("");
    try {
      const result = await rankHeads(layer, baseline, scope);
      setRanked(result);
      // What a ranking costs is dominated by the forward-pass count, so the
      // whole-model button extrapolates from a measured layer rather than
      // guessing — and it only exists once there is a measurement.
      //
      // Keep the FASTEST rate seen for this generation, not the latest. The
      // first ranking after a load pays for CUDA warm-up and runs several
      // times slower: measured on an RTX 4060, Qwen3-0.6B's first layer took
      // 3.05 s and the next two 0.80 and 0.78; its first whole-model sweep
      // 51.9 s against 19.9 and 20.0 after. Since the button appears only
      // after that first ranking, the latest-rate version quoted its worst
      // possible number — 46.8% over on that run. Warm-up only ever inflates,
      // so the minimum is the honest estimator, and once warm the
      // extrapolation held to within 2.5% across repeats on both models.
      if (result.passes > 0) {
        const rate = result.elapsed_s / result.passes;
        setSecPerPass((prev) => (prev === null ? rate : Math.min(prev, rate)));
      }
      // Open the head that moved the answer most — the whole point is to
      // stop the user picking blind. Which head that is depends on what was
      // asked: ranking one layer answers "which head here", so stay here;
      // ranking the model answers "which head anywhere", so go there. Staying
      // put after a whole-model sweep leaves the winner named in the list and
      // the arcs showing something else.
      const scoped =
        scope === "all"
          ? result.ranked
          : result.ranked.filter((r) => r.layer === layer);
      const best = (scoped.length ? scoped : result.ranked)[0];
      if (best) {
        setHead(best.head);
        if (best.layer !== layer) setLayer(best.layer);
      }
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setRanking(false);
    }
  }

  /** "11s" / "2m 17s" — a wait the user can decide about. */
  const humanSeconds = (s: number) =>
    s < 90 ? `${Math.max(1, Math.round(s))}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;

  const layerPasses = heads + 2;
  const allPasses = layers * heads + 2;

  // Where a token ranking would run. The pin wins; failing that, wherever the
  // last result was taken (which is the server telling us its own default);
  // failing that, the last prompt token, which is what the strip already
  // rests on. -1 means we cannot say, and then no estimate is offered.
  const lastPrompt = data?.n_prompt ? data.n_prompt - 1 : -1;
  const attrTarget = pinned >= 0 ? pinned : attr ? attr.position : lastPrompt;
  // Mirrors modelmri/attribute.py: candidates are range(1, position), capped
  // at MAX_CANDIDATES = 64, plus the base, the noise floor, the plain pass,
  // index 0, one joint mask and one check that masking works — and the one
  // runtime.py adds in front to read the model's own answer at the position.
  // An estimate; the result carries the count it actually spent and a
  // sentence breaking it down, and both are shown.
  const attrPasses = attrTarget > 1 ? Math.min(attrTarget - 1, 64) + 7 : 0;
  // Built here rather than inline so the gate on the button below sits right
  // next to the label it gates — a 500-character title string wedged between
  // the two is how a static check of that gate goes blind.
  const attrTitle =
    (attrPasses
      ? `About ${attrPasses} forward passes — one per candidate token, plus ` +
        `the answer, a baseline, a noise-floor pass, a plain pass, index 0, ` +
        `one joint mask and one check that masking really empties the column`
      : "Masks one token at a time and measures how far the answer moves") +
    (attrPasses && secPerPass
      ? `. About ${humanSeconds(attrPasses * secPerPass)} on this model.`
      : "") +
    (attrTarget >= 0
      ? ` Runs at token ${attrTarget}; click a token in the strip to move it.`
      : "");
  // A result covering more than one layer came from a whole-model sweep, and
  // reads differently: the list is ranked across layers, not within one.
  const wholeModel =
    !!ranked && new Set(ranked.ranked.map((r) => r.layer)).size > 1;

  // A ranking is about specific layers under one generation, so moving to a
  // layer it does not cover makes it an answer to a question nobody asked.
  // It must NOT be discarded for a layer it does cover: a whole-model sweep
  // ranks every layer and then jumps to the winning one, and clearing on any
  // layer change would delete the result the moment it arrived.
  useEffect(() => {
    setRanked((r) => (r && r.ranked.some((x) => x.layer === layer) ? r : null));
  }, [layer]);

  // A new generation invalidates everything derived from the old one, and
  // the per-pass rate belongs to whichever model produced it.
  useEffect(() => {
    setRanked(null);
    setDiff(null);
    setSecPerPass(null);
    // A token ranking is a claim about a position in ONE sequence. Index 14
    // of the next generation is a different word, so keeping the result would
    // put the old scores under the new strip.
    setAttr(null);
    setAttrErr("");
  }, [epoch]);

  useEffect(() => {
    let live = true;
    void (async () => {
      const meta = await getAttentionMeta().catch(() => null);
      if (!live || !meta?.available) return;
      setErr("");
      setLayers(meta.n_layers!);
      setHeads(meta.n_heads!);
      setLayer(Math.floor(meta.n_layers! / 2));
      setHead(0);
      setInfo(
        `${meta.n_layers}L × ${meta.n_heads}H · ${meta.n_tokens} tok`,
      );
    })();
    return () => {
      live = false;
    };
  }, [epoch]);

  useEffect(() => {
    if (layers === 0) return;
    let live = true;
    const t = performance.now();
    // The first fetch for a generation runs a full output_attentions forward
    // pass server-side and can take seconds, so say so rather than showing
    // controls above an empty space.
    setInfo(`${dial("L", layer, layers)} · ${dial("H", head, heads)} · computing…`);
    void getAttention(layer, head)
      .then((d) => {
        if (!live) return;
        setErr("");
        setData(d);
        setInfo(
          `${dial("L", d.layer, layers)} · ${dial("H", d.head, heads)} · ` +
            `${d.tokens.length} tok · ${((performance.now() - t) / 1000).toFixed(2)}s`,
        );
      })
      .catch((e) => {
        if (!live) return;
        setErr(errorText(e));
        setInfo("");
      });
    return () => {
      live = false;
    };
  }, [layers, layer, head, epoch]);

  if (layers === 0) return null;

  const options = (n: number) =>
    Array.from({ length: n }, (_, i) => (
      <option key={i} value={i}>
        {i}
      </option>
    ));

  /** Head options, ordered by a ranking when there is one.
   *
   *  The dropdown is 144 numbers and no reason to pick any of them. Once
   *  heads are ranked, the label carries the score and the order carries the
   *  answer — so the first entry is the head worth opening.
   */
  const headOptions = () => {
    if (!ranked) return options(heads);
    const byHead = new Map(
      ranked.ranked.filter((r) => r.layer === layer).map((r) => [r.head, r]),
    );
    return [...byHead.entries()]
      .sort((a, b) => b[1].kl - a[1].kl)
      .map(([h, score]) => (
        <option key={h} value={h}>
          {h} · KL {score.kl < ranked.noise_floor_kl ? "—" : score.kl.toFixed(3)}
          {score.flips_top ? " · flips" : ""}
        </option>
      ));
  };

  return (
    <div ref={scanRef} className="panel attn">
      <div className="sect">
        <span className="dot d-attn" />
        <h2 className="h-attn">ATTENTION — WHERE EACH TOKEN LOOKED</h2>
        <span className="rule" />
      </div>
      <div className="row" style={{ margin: "10px 0" }}>
        {/* `htmlFor`/`id`, not adjacency. These read as "layer" and "head" to
            anyone looking at them, and as two unnamed combo boxes to a screen
            reader — the visual label was never connected to the control. */}
        <label className="meta" htmlFor="attn-layer">
          layer
        </label>
        <select
          id="attn-layer"
          value={layer}
          onChange={(e) => setLayer(Number(e.target.value))}
        >
          {options(layers)}
        </select>
        <label className="meta" htmlFor="attn-head">
          head
        </label>
        <select
          id="attn-head"
          value={head}
          onChange={(e) => setHead(Number(e.target.value))}
        >
          {headOptions()}
        </select>
        <span className="meta">{info}</span>
        {/* Needs a forward pass per head, so it needs a model — a recording
            does not carry one. */}
        {!replay && (
          <>
            <button
              className="ghost sm"
              onClick={() => void rank("layer")}
              disabled={ranking}
              title={
                `${layerPasses} forward passes — one per head, plus a baseline ` +
                `and a noise-floor pass` +
                (secPerPass
                  ? `. About ${humanSeconds(layerPasses * secPerPass)} on this model.`
                  : "")
              }
            >
              {ranking ? "ranking…" : ranked ? "Rank heads again" : "Rank heads"}
              <span className="meta">
                {" "}
                {secPerPass
                  ? `≈ ${humanSeconds(layerPasses * secPerPass)}`
                  : `${layerPasses} passes`}
              </span>
            </button>
            {/* Only offered once one layer has been timed on THIS model. A
                whole-model sweep is seconds on gpt2 and can be minutes on a
                28-layer model, and the difference is not something the user
                can guess — so the button does not exist until it can state
                which one this is, from a measurement on this machine. */}
            {secPerPass !== null && layers > 1 && (
              <button
                className="ghost sm"
                onClick={() => void rank("all")}
                disabled={ranking}
                title={`Rank every head in the model — ${allPasses} forward passes`}
              >
                all {layers} layers
                <span className="meta">
                  {" "}
                  ≈ {humanSeconds(allPasses * secPerPass)}
                </span>
              </button>
            )}
            {/* The panel already tells the user this changes the order. It
                should let them look. */}
            <select
              className="sm"
              value={baseline}
              onChange={(e) => setBaseline(e.target.value as "zero" | "mean")}
              disabled={ranking}
              title="What a removed head is replaced with — this changes the ranking"
            >
              <option value="zero">zero-ablation</option>
              <option value="mean">mean-ablation</option>
            </select>
          </>
        )}
        {/* Beside "Rank heads", and gated harder than it is.

            A recording, the static demo and the `.mri` viewer all have no
            model, so this button could do nothing there but produce an error
            — and a control that only ever errors does not teach a visitor
            that the page has no model behind it. It teaches them the
            measurement does not work, which is the one impression this
            project cannot afford on the surface most people ever touch.
            Gating it is also what stops the call existing at all: no button,
            no unhandled path to fall through to demo.ts's catch-all.
            tests/demo_check.py asserts this gate. */}
        {!replay && !DEMO && !VIEWER && (
          <button
            className="ghost sm"
            onClick={() => void attribute()}
            disabled={attributing}
            title={attrTitle}
          >
            {attributing ? "attributing…" : attr ? "Rank tokens again" : "Rank tokens"}
            {attrPasses > 0 && (
              <span className="meta">
                {" "}
                {secPerPass
                  ? `≈ ${humanSeconds(attrPasses * secPerPass)}`
                  : `${attrPasses} passes`}
              </span>
            )}
          </button>
        )}
        {!replay && <ShareButton layer={layer} head={head} />}
        <span className="spacer" />
        {/* Arc thickness encodes weight, which cannot be read without a
            key. Three stops is enough to calibrate the eye. */}
        <span className="weight-key" aria-hidden="true">
          <span><i style={{ height: 1 }} />0.05</span>
          <span><i style={{ height: 3 }} />0.20</span>
          <span><i style={{ height: 6 }} />0.50</span>
        </span>
      </div>
      {ranked && (
        <div className="ranking">
          {/* A WAY BACK. Ranking replaces the arcs with a table and there was
              no control that returned you to them -- the only exits were a
              page reload or generating again, neither of which is obviously
              "close this". A result you cannot dismiss is a mode, and this
              panel is not supposed to have modes. */}
          <button
            className="ghost sm ranking-close"
            onClick={() => setRanked(null)}
            title="Back to the attention arcs"
          >
            ← back to the arcs
          </button>
          <div className="ranking-head">
            <strong>
              Removing one head from{" "}
              {wholeModel ? "anywhere in the model" : `layer ${layer}`}, and
              measuring how far the answer to{" "}
              {JSON.stringify(ranked.target_token)} moves
            </strong>
            <span className="meta">
              {ranked.passes} forward passes · {ranked.elapsed_s}s ·{" "}
              {ranked.baseline}-ablation
            </span>
          </div>
          <ol className="ranking-list">
            {/* A whole-model sweep is ranked across layers; a single-layer
                one only has this layer to show. */}
            {(wholeModel
              ? ranked.ranked
              : ranked.ranked.filter((r) => r.layer === layer)
            )
              .slice(0, 5)
              .map((r) => {
                const noise = r.kl <= ranked.noise_floor_kl;
                return (
                  <li key={`${r.layer}.${r.head}`} className={noise ? "faint" : ""}>
                    <button
                      className="ghost sm"
                      onClick={() => {
                        setLayer(r.layer);
                        setHead(r.head);
                      }}
                    >
                      {wholeModel ? `L${r.layer} H${r.head}` : `H${r.head}`}
                    </button>
                    <span className="mid">
                      {noise ? "below the noise floor" : `KL ${r.kl.toFixed(3)}`}
                    </span>
                    <span className="meta">
                      p({JSON.stringify(ranked.target_token)}){" "}
                      {r.p_top_before.toFixed(3)} → {r.p_top_after.toFixed(3)}
                      {r.flips_top && " · changes the top token"}
                    </span>
                    <span className="spacer" />
                    <button
                      className="ghost sm"
                      onClick={() => void compare(r.layer, r.head)}
                      title={`Show what changes at layer ${r.layer + 1} when L${r.layer} H${r.head} is removed`}
                    >
                      what changes?
                    </button>
                  </li>
                );
              })}
          </ol>
          {/* The caveat travels with the numbers. An ordered list reads as
              truth, and two of these scores are not what a reader assumes. */}
          <div className="hint">
            These are <em>not</em> each head's share of the prediction — they do
            not add up, and are not meant to. Each says only: removing this one
            head, on its own, moves the answer this much. The ranking also
            depends on what a removed head is replaced with; this used{" "}
            <code>{ranked.baseline}</code>, and on some layers the mean baseline
            gives a different order.
          </div>
        </div>
      )}
      {/* The server's own sentence, unwrapped. Several of the things this
          endpoint can answer are refusals rather than failures — a control
          token at the attribution position, a position with nothing before it
          but the sink — and they already say what would make the measurement
          work. Anything added in front of them is the client guessing. */}
      {attrErr && <div className="hint err">{attrErr}</div>}
      {attr && <TokenRanking a={attr} />}
      {err ? (
        <div className="hint err">
          Could not compute this attention map — {err}. Generate again, or pick
          a different layer.
        </div>
      ) : (
        <>
          {diff ? (
            <>
              <div className="diffbar">
                <strong>
                  Layer {diff.layer}, head {diff.head} — what changed when{" "}
                  {diff.b.replace("ablate:", "L").replace(".", " H")} was removed
                </strong>
                <span className="meta">
                  {diff.moved} of {diff.cells} cells moved · largest{" "}
                  {diff.max_abs.toFixed(3)}
                </span>
                <span className="spacer" />
                <span className="diff-key" aria-hidden="true">
                  <i className="up" /> attends more <i className="down" /> less
                </span>
                <button className="ghost sm" onClick={() => setDiff(null)}>
                  Back to the map
                </button>
              </div>
              {diff.note ? (
                <div className="hint">{diff.note}</div>
              ) : (
                <ArcCanvas tokens={diff.tokens} matrix={diff.matrix} signed />
              )}
              <div className="hint">
                A difference between two forward passes over the{" "}
                <em>same</em> tokens — so position {"i"} is the same token on
                both sides. Comparing two separate generations would not be:
                different sampling, or a different chat template, shifts every
                index and the subtraction becomes fiction.
              </div>
            </>
          ) : (
            <>
              {data && (
                <ArcCanvas
                  tokens={data.tokens}
                  matrix={data.matrix}
                  nPrompt={data.n_prompt}
                  onPin={setPinned}
                  attrPos={attr ? attr.position : undefined}
                  scores={attr ? strip(attr, data.tokens.length) : undefined}
                  testedFrom={attr ? attr.tested_span[0] : undefined}
                />
              )}
              <div className="hint">
                hover or focus a token → arcs show what it attended to · click
                or Enter to pin · arc thickness = attention weight
                {!replay && !DEMO && !VIEWER && " · click a token to attribute there"}
                {replay && " · recorded, not live"}
              </div>
              {/* Two claims, and the second used to be a closed list of two
                  reasons that did not cover the commonest one. When the
                  64-candidate cap bites, every candidate below the tested
                  window renders bar-less as well — measured on gpt2 with a
                  73-token prompt, 64 of 71 candidates were tested and indices
                  1..7 came back unmarked while the sentence below told the
                  reader they must be outside the causal cone. They are now
                  dashed like `after-attr` and enumerated here only when there
                  actually are any. */}
              {attr && (
                <div className="hint">
                  The bar on a chip's left edge is that token's score against
                  the largest one measured in this run — index 0 included, so
                  on some models the tallest bar is the sink rather than a
                  word, and its row above says so. The ramp is linear in nats
                  and the scores span orders of magnitude, so most bars sit on
                  the floor: measured on a 73-token gpt2 prompt, 60 of the 65
                  bars were under 5% of the tallest and 34 under 2%. Read the
                  strip for the ordering and the lists for the nats.
                  <br />A chip with NO bar was never tested — everything after
                  token {attr.position} is outside the causal cone, and token{" "}
                  {attr.position} itself is excluded by rule rather than by its
                  size
                  {attr.truncated &&
                    `, and so is everything before token ${attr.tested_span[0]}, which fell outside the ${attr.n_tested} nearest candidates this run had budget for`}
                  .
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}

/** Per-chip ramp values for the token strip, 0..1, or null for "not tested".
 *
 *  Normalised against the largest score in the run, INDEX 0 INCLUDED. The
 *  alternative — normalising over the ranked rows only and leaving index 0
 *  blank — would have the strip and the list disagreeing about what was
 *  measured, and a blank chip is this component's word for "never asked". On
 *  gpt2 that means the first chip carries the tallest bar; the row above the
 *  lists says why that is a property of the position rather than of the word.
 */
function strip(a: TokenAttribution, n: number): (number | null)[] {
  const rows = [a.index0, ...a.ranked];
  const peak = rows.reduce((m, r) => Math.max(m, r.kl), 0);
  const out: (number | null)[] = Array.from({ length: n }, () => null);
  for (const r of rows) {
    if (r.index >= 0 && r.index < n) out[r.index] = peak > 0 ? r.kl / peak : 0;
  }
  return out;
}

/** A KL small enough that fixed decimals would print it as zero.
 *
 *  Not cosmetic. On Qwen3-0.6B every token the user typed scores between
 *  3.1e-05 and 7.9e-05 nats, and five decimal places render most of that list
 *  as 0.00003 or 0.00000 — a measured value displayed as nothing. The
 *  exponent keeps the reader looking at what was measured; whether a number
 *  that small MEANS anything is what the caveat below the lists is for. */
const fmtKL = (kl: number) =>
  kl !== 0 && Math.abs(kl) < 0.001 ? kl.toExponential(2) : kl.toFixed(5);

/** How many rows of each list are printed. Every tested token still carries
 *  its bar in the strip above, so nothing measured is hidden — this only
 *  caps how far down a 64-row list the panel reads out loud. */
const SHOWN_PER_LIST = 10;

/** The token ranking: two lists, never interleaved.
 *
 *  Separating them is not tidiness. Measured on Qwen3-0.6B at the end of a
 *  templated prompt, the top scores are the template's own '\n' (6.24429),
 *  'assistant' (2.02161) and '<|im_start|>' (0.32266), while every word the
 *  user typed sits between 3.1e-05 and 7.9e-05 — four to five orders of
 *  magnitude down. One merged list therefore answers "the chat template"
 *  every time and renders the user's own words invisible. Both lists are
 *  shown, because the scaffold dominating is a real and useful finding; what
 *  is not acceptable is presenting the two as one ranking.
 */
function TokenRanking({ a }: { a: TokenAttribution }) {
  const rows = (list: TokenScore[]) => (
    <ol className="ranking-list">
      {list.slice(0, SHOWN_PER_LIST).map((r) => {
        const noise = r.kl <= a.noise_floor_kl;
        return (
          <li key={r.index} className={noise ? "faint" : ""}>
            <span className="attr-tok">{JSON.stringify(r.token)}</span>
            <span className="mid">
              {noise ? "below the noise floor" : `KL ${fmtKL(r.kl)}`}
            </span>
            <span className="meta">
              #{r.index} · p({JSON.stringify(a.target_token)}){" "}
              {r.p_top_before.toFixed(3)} → {r.p_top_after.toFixed(3)}
              {r.flips_top && " · changes the top token"}
            </span>
          </li>
        );
      })}
      {list.length > SHOWN_PER_LIST && (
        <li className="meta">
          {list.length - SHOWN_PER_LIST} more were tested and scored lower;
          every one of them carries its bar in the strip below.
        </li>
      )}
    </ol>
  );

  // One list per group the server named, and never a heading for a group it
  // did not. A span the server could not locate is not a span of zero length
  // and is not "all of it is yours" either, so `unknown` rows go under a
  // heading that claims nothing; and rows past the prompt are the model's own
  // output, which used to be printed under "chat template scaffold" on gpt2 —
  // a model whose span_note says two lines above that it has no chat
  // template, and whose own words were the three highest scores in the run.
  const byGroup = (g: TokenScore["group"]) => a.ranked.filter((r) => r.group === g);
  // `typed`/`template` and `unknown` are mutually exclusive by construction —
  // the server emits the first pair when it located the user's words and the
  // second when it could not — so the heading pair follows from typed_span
  // and never from whether a list came back empty. Empty is a finding there
  // and keeps its heading. `generated` is orthogonal to both and is shown only
  // when there is something in it: attributing inside the prompt is the
  // ordinary case, and an always-present empty heading would be noise.
  // A "chat template scaffold" heading is only a finding on a model that HAS
  // one. gpt2's span is the whole prompt — the server's own note says so in
  // the same panel — and an empty heading there asserts a template exists and
  // contributed nothing, which is a different and false statement.
  const hasTemplate =
    a.typed_span != null && (a.typed_span[0] > 0 || a.typed_span[1] < a.n_prompt);
  const lists: [string, TokenScore[], boolean][] =
    a.typed_span != null
      ? [
          ["what you typed", byGroup("typed"), true],
          ["chat template scaffold", byGroup("template"), hasTemplate],
        ]
      : [["every token that was tested", byGroup("unknown"), true]];
  const generated = byGroup("generated");
  if (generated.length)
    lists.push(["the model's own output, not yours", generated, false]);
  const ratio = a.joint_kl > 0 ? a.sum_of_singles / a.joint_kl : null;

  /** Why a group is empty — and never "they were not candidates" when they
   *  were candidates that the cap simply did not reach. That distinction is
   *  the whole job of `coverage`, and the sentence here used to contradict
   *  it: measured on gpt2 at position 100 of a 125-token generation, the
   *  typed span was [0,5], all five were candidates, none was tested because
   *  the window started at 36 — and the panel said they were not candidates. */
  const whyEmpty = (heading: string) => {
    if (heading === "what you typed" && a.typed_span) {
      const [lo, hi] = a.typed_span;
      const anyCandidate = hi > 1 && lo < a.position;
      if (anyCandidate && a.truncated && lo < a.tested_span[0]) {
        return `Tokens ${lo}-${hi - 1} are yours and were candidates, but this run tested only the ${a.n_tested} nearest token ${a.position} — from token ${a.tested_span[0]} on. They were not asked, not found unimportant.`;
      }
      return `None of the tokens in that span were candidates at token ${a.position}.`;
    }
    if (a.truncated) {
      return `No token in this group was among the ${a.n_tested} tested at token ${a.position}; the run reached back only to token ${a.tested_span[0]}.`;
    }
    return `No token in this group was a candidate at token ${a.position}.`;
  };

  return (
    <div className="ranking">
      <div className="ranking-head">
        {/* The target token belongs here and not in a footnote. Without it
            this panel reads as a general claim about the prompt, which is the
            README's Paris sentence all over again — the scores are about ONE
            next-token distribution at ONE position, and that is the sentence
            that says so. */}
        <strong>
          Masking one token from every later position, and measuring how far
          the answer at token {a.position} — {JSON.stringify(a.target_token)} —
          moves.
        </strong>
        <span className="meta">
          {a.passes} forward passes · {a.elapsed_s}s · {a.baseline} · noise
          floor {a.noise_floor_kl}
        </span>
      </div>
      {/* The breakdown of that count, in the server's words — including the
          part the seconds beside it need: the pass count transfers between
          machines, the duration does not. */}
      <div className="hint">{a.passes_note}</div>
      {/* Where the user's own words are, or why the server could not say.
          Shown whichever answer it is: on a model with no chat template this
          is the sentence that stops an empty scaffold list reading as a
          measurement that went missing. */}
      <div className="hint">{a.span_note}</div>

      <div className="attr-index0">
        <span className="attr-tok">{JSON.stringify(a.index0.token)}</span>{" "}
        <span className="mid">KL {fmtKL(a.index0.kl)}</span>{" "}
        <span className="meta">
          #0 · p({JSON.stringify(a.target_token)}){" "}
          {a.index0.p_top_before.toFixed(3)} →{" "}
          {a.index0.p_top_after.toFixed(3)}
          {a.index0.flips_top && " · changes the top token"}
        </span>
        <div className="hint">{a.index0.note}</div>
      </div>

      {/* A heading appears when the server put rows in that group OR when the
          group is one the server's own labelling implies exists — "none of my
          words were candidates here" is a finding, and a heading that
          vanishes turns it into an absence the reader has to notice on their
          own. A heading for a group the server never uses is the opposite
          error and is what put gpt2's own output under "chat template
          scaffold", so `unknown` and `typed`/`template` are mutually
          exclusive by construction and only the applicable pair is shown. */}
      {lists.map(([heading, list, keepWhenEmpty]) =>
        list.length === 0 && !keepWhenEmpty ? null : (
          <div key={heading}>
            <div className="attr-head">{heading}</div>
            {list.length ? rows(list) : <div className="hint">{whyEmpty(heading)}</div>}
          </div>
        ),
      )}

      {/* In the server's words, because "not listed" and "not important" are
          the two things a truncated leaderboard is read as meaning. */}
      {a.truncated && <div className="hint">{a.coverage}</div>}

      {/* The caveat travels with the numbers, and the third clause is the one
          that cannot be copied from the head ranking above. Head ablation
          over-counts 8x on gpt2 and under-counts on gemma; token masking is a
          different phenomenon and misses in a direction that depends on the
          model AND on which tokens you sum. So the panel prints this run's own
          two numbers rather than a factor — the reader can see which way it
          goes on THEIR model instead of transferring a rule that does not
          hold. */}
      <div className="hint">
        These are <em>not</em> each token's share of the answer. They do not
        add up, and the direction of the error is not fixed — read it off this
        run: all {a.n_tested} tested tokens sum to{" "}
        <b>{fmtKL(a.sum_of_singles)}</b> nats, while one joint mask of those
        same tokens gives <b>{fmtKL(a.joint_kl)}</b>
        {ratio !== null && <> — a ratio of <b>{ratio.toFixed(2)}x</b></>}. Above
        1 the singles over-state the joint, below 1 they under-state it, and
        which one happens depends on the model and on which tokens you sum.
        There is no correction factor here, and the head ranking's version of
        this caveat does not carry over in either direction.
      </div>
      <div className="hint">{a.means}</div>
      {/* Whether the mask did what the whole measurement assumes. Red when it
          could not be confirmed: every score above would then be describing
          something other than a removed token. */}
      <div className={a.mask_verified ? "hint" : "hint err"}>{a.mask_check}</div>
    </div>
  );
}

/** Save this analysis as a `.mri`, optionally with a note.
 *
 *  The note is the reason the format exists. "L14 H3 moves the subject token"
 *  is what you want to say when you send it; without somewhere to put that,
 *  the file arrives as a heat map with no claim attached and the recipient
 *  has to guess what they are meant to be seeing.
 */
function ShareButton({ layer, head }: { layer: number; head: number }) {
  const [open, setOpen] = useState(false);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const save = async () => {
    setBusy(true);
    setErr("");
    try {
      const { blob, filename } = await exportSession(layer, head, note);
      // Object URL rather than a data: URI — a session is megabytes, and a
      // data: URI that large is refused by the browser without saying why.
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
      setOpen(false);
      setNote("");
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  if (!open) {
    return (
      <button
        className="ghost sm"
        onClick={() => setOpen(true)}
        title="Save this analysis as a .mri anyone can open without the model"
      >
        Share this view
      </button>
    );
  }

  return (
    <span className="share-row">
      <input
        autoFocus
        className="share-note"
        placeholder="what did you find? (optional)"
        value={note}
        maxLength={200}
        onChange={(e) => setNote(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") void save();
          if (e.key === "Escape") setOpen(false);
        }}
      />
      <button className="ghost sm" onClick={() => void save()} disabled={busy}>
        {busy ? "packing…" : "Save .mri"}
      </button>
      <button className="ghost sm" onClick={() => setOpen(false)}>
        Cancel
      </button>
      {err && <span className="hint err">{err}</span>}
    </span>
  );
}
