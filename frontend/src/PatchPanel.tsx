import { CSSProperties, useEffect, useState } from "react";
import { measured } from "./measured";
import RunsOn, { useModelIdentity, useModelReady } from "./RunsOn";
import {
  errorText,
  pathTrace,
  PathTrace,
  patchTrace,
  PatchTrace,
} from "./api";
import ReceiptLine from "./ReceiptLine";
import PatchScreenPanel from "./PatchScreenPanel";
import { DEMO } from "./demo";
import { VIEWER } from "./viewer";
import { useScanOnData } from "./useScanOnData";

/** Activation patching — where in the model the answer is decided.
 *
 *  Every other panel here removes something from one prompt. This one needs
 *  two, and the pair is the unit of meaning: the grid is the share of the gap
 *  between their two answers that comes back when one activation is moved
 *  from the run that knows into the run that does not.
 *
 *  The colour scale is DIVERGING, not sequential, and that is the whole
 *  reason this component does not reuse the attention heatmap. Recovery is
 *  signed: on the reference pair 5 of 132 sites pushed the answer further
 *  from the clean run, the worst by -0.157, and a one-sided ramp would paint
 *  those the same colour as a site that did nothing.
 */

const LABEL: Record<string, string> = {
  resid: "residual stream",
  attn: "attention",
  mlp: "MLP",
};

/** What each grid is actually claiming, in the reader's terms. */
const BLURB: Record<string, string> = {
  resid: "where the answer is, at the input to each block",
  attn: "what attention moved in — usually late, at the last token",
  mlp: "what the MLP wrote — usually early, at the subject",
};

const CLEAN_DEFAULT = "The Eiffel Tower is located in the city of";
const CORRUPT_DEFAULT = "The Colosseum is located in the city of";

/** Which senders are worth a row.
 *
 *  Most of what a path trace returns sits at one representable step above
 *  zero, untested. Printing all of them buries the one that beat its controls
 *  under rows that all say the same thing, and dropping them silently would
 *  claim the list is complete. So the default is the ones that carry a claim
 *  — anything tested against chance, plus anything the resolution can
 *  actually separate from zero — and the rest are folded behind a COUNTED
 *  button.
 */
function shownSenders(path: PathTrace, all: boolean): PathTrace["senders"] {
  if (all) return path.senders;
  return path.senders.filter(
    (s) =>
      s.control_max !== undefined ||
      Math.abs(s.recovery) >= path.recovery_resolution,
  );
}

/** Blue for recovered, red for pushed away, transparent at zero. */
function cell(v: number): string {
  const a = Math.min(1, Math.abs(v));
  // Named tokens only, and ones this stylesheet defines: `var(--accent)` does
  // not exist here, and color-mix with an undefined custom property yields
  // transparent rather than an error — so the entire positive half of the
  // scale rendered as nothing while the negative half worked.
  return v >= 0
    ? `color-mix(in oklab, var(--color-cobalt) ${a * 100}%, transparent)`
    : `color-mix(in oklab, var(--crimson-500) ${a * 100}%, transparent)`;
}

export default function PatchPanel({
  epoch,
  recorded,
}: {
  epoch: number;
  /** Set when a `.mri` is open and carries a trace: the prompts it was
   *  measured on, so the panel shows the recording's own pair rather than
   *  the defaults and offers to draw it instead of a button that can only
   *  refuse. */
  recorded?: { clean: string; corrupt: string };
}) {
  // Nothing loaded means every button here can only be refused. Shares
  // `RunsOn`'s cached session, so the badge and the control it disables
  // read one answer rather than two requests that can disagree.
  const ready = useModelReady(epoch);
  const model = useModelIdentity(epoch);
  const [clean, setClean] = useState(recorded?.clean || CLEAN_DEFAULT);
  const [corrupt, setCorrupt] = useState(recorded?.corrupt || CORRUPT_DEFAULT);
  const [data, setData] = useState<PatchTrace | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [hover, setHover] = useState<{ l: number; p: number } | null>(null);
  // Which grid is on screen. Three are computed in one call because the
  // comparison IS the finding — the residual grid says where, the two
  // sublayer grids say through what, and they do not agree.
  const [comp, setComp] = useState("resid");
  // The follow-up to a bright cell. Patching a residual stream restores
  // EVERYTHING that ever wrote into it at once, so the grid says where the
  // answer is carried and cannot say what put it there. Clicking a cell asks
  // that second question at that site.
  const [pinned, setPinned] = useState<{ l: number; p: number } | null>(null);
  const [path, setPath] = useState<PathTrace | null>(null);
  const [tracing, setTracing] = useState(false);
  const [pathErr, setPathErr] = useState("");

  // `key={epoch}` on the element this returns is NOT this. A key remounts the
  // DOM subtree; it does not reset the state of the component that returns
  // it, whose fiber identity comes from its position in the PARENT. So the
  // recorded trace survived a model swap and sat under the new model's name,
  // which is the one thing a causal result must never do.
  useEffect(() => {
    setData(null);
    setErr("");
    setHover(null);
    setPinned(null);
    setPath(null);
    setPathErr("");
  }, [epoch, model]);
  // Folded by default, and the fold is counted rather than silent.
  const [allSenders, setAllSenders] = useState(false);
  // The specular scan, on the same terms as every other panel: keyed on the
  // payload, so it fires when a trace ARRIVES and not on every re-render.
  // Switching tabs re-reads a grid that was already here, and deliberately
  // does not flash — the data did not change, only which of it you are
  // looking at.
  const scanRef = useScanOnData(data);

  async function run() {
    setBusy(true);
    setErr("");
    setData(null);
    // A sender list is about one site of one pair. A new pair makes it a
    // claim about a grid that is no longer on screen.
    setPinned(null);
    setPath(null);
    setPathErr("");
    try {
      setData(await patchTrace(clean, corrupt));
    } catch (e) {
      // The refusals are the point of this panel's error path, not an
      // afterthought: two prompts of different token lengths both run fine on
      // their own, so without the message there is nothing to see.
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  const controlled = new Map(
    (data?.sites ?? [])
      .filter((s) => s.component === comp)
      .map((s) => [`${s.layer}:${s.position}`, s]),
  );
  const hovered = hover ? controlled.get(`${hover.l}:${hover.p}`) : undefined;

  async function trace(l: number, p: number) {
    setPinned({ l, p });
    setPath(null);
    setPathErr("");
    setAllSenders(false);
    setTracing(true);
    try {
      setPath(await pathTrace({ clean, corrupt, layer: l, position: p }));
    } catch (e) {
      // Layer 0 has nothing upstream of it, and the refusal says so — which
      // is a better answer than an empty list that reads as "no component
      // mattered".
      setPathErr(errorText(e));
    } finally {
      setTracing(false);
    }
  }
  // `?.` on grids, not just on data. A built bundle can be older or newer
  // than the server it is served by — during this feature it was, and
  // `data.grids[comp]` on a response that still carried the old single `grid`
  // threw during render, which in React unmounts the whole tree: every panel
  // on the page went blank because one of them read a missing key.
  const grid = data?.grids?.[comp] ?? [];

  return (
    <div className="panel patch" key={epoch} ref={scanRef}>
      {/* The house header, which this panel was not using: a colour-coded dot,
          letterspaced mono caps and the ruler rule that runs to the edge — and
          that recolours when a cell is pinned, same as the attention panel's
          does when a token is. Without it this panel read as a different
          application bolted onto the page. */}
      <div className="sect">
        <span className="dot d-patch" />
        <h2 className="h-patch">PATCHING — WHERE THE ANSWER IS DECIDED</h2>
        <span className="rule" />
      </div>
      <RunsOn epoch={epoch} />
      <p className="meta">
        Two prompts that differ in one fact. Every other panel asks what
        mattered by taking something away; this moves an activation from the
        run that knows the answer into the run that does not, and reports how
        much of the difference comes back.
      </p>

      <div className="patch-inputs">
        <label>
          <span className="meta">clean — the run that knows</span>
          <input
            value={clean}
            onChange={(e) => setClean(e.target.value)}
            spellCheck={false}
          />
        </label>
        <label>
          <span className="meta">corrupt — one fact changed</span>
          <input
            value={corrupt}
            onChange={(e) => setCorrupt(e.target.value)}
            spellCheck={false}
          />
        </label>
      </div>

      <div className="row" style={{ marginBottom: 10 }}>
        <button
          className="cta"
          onClick={() => void run()}
          // `ready === false` and not `!ready`: null means the session could
          // not be read, and taking a working control away over a network
          // blip is worse than letting the route refuse in its own words.
          //
          // AND NOT WHEN THERE IS A RECORDING. Showing one runs nothing: the
          // grid is already in the file. This button read "Show the recorded
          // trace" and was disabled underneath that label in every viewer
          // build, because no model is loaded there and none ever will be —
          // so the one audience the format exists for could see the sentence
          // naming their measurement and not open it. `GroundPanel` had this
          // right; this panel did not.
          disabled={busy || (!recorded && ready === false)}
          title={
            !recorded && ready === false
              ? "Load a model in Run at the top of the page first — this measurement runs it."
              : undefined
          }
        >
          {busy ? "Patching every site…" : recorded ? "Show the recorded trace" : "Trace it"}
        </button>
        <span className="meta">
          the two prompts have to split into the same number of tokens
        </span>
      </div>

      {err && <div className="hint err">{err}</div>}

      {data && !data.grids && (
        <div className="hint err">
          The server answered in a shape this page does not know — it is
          probably running a different version of ModelMRI than the one that
          built this bundle. Restart `modelmri serve`.
        </div>
      )}

      {data && data.grids && (
        <>
          <div className="patch-answers meta">
            <span>
              clean <b>{data.clean.answer.text}</b> ({data.clean.answer.p})
            </span>
            <span>
              corrupt <b>{data.corrupt.answer.text}</b> ({data.corrupt.answer.p}
              )
            </span>
            <span className="spacer" />
            <span>
              {data.passes} passes · {data.seconds}s · {data.dtype}
            </span>
          </div>

          <div className="patch-tabs">
            {data.components.map((c) => (
              <button
                key={c}
                className={`pill sm ${c === comp ? "on" : ""}`}
                onClick={() => {
                  setComp(c);
                  setHover(null);
                }}
              >
                {LABEL[c] ?? c}
              </button>
            ))}
            <span className="meta">{BLURB[comp]}</span>
          </div>

          {/* WHAT THIS ARCHITECTURE DID NOT EXPOSE. The trace catches a
              PatchError per component and carries on so the rest is still
              measured — a Mixtral or an OLMoE names its sublayer
              `block_sparse_moe` and has no `mlp` to patch. `patch.py` put
              `skipped` in the payload with a comment saying why: without it
              "two grids would have arrived looking like the whole answer".
              The payload was fixed and the panel still did not read it, so
              two grids still arrived looking like the whole answer. */}
          {data.skipped && data.skipped.length > 0 && (
            <p className="hint">
              {data.skipped.length} component
              {data.skipped.length === 1 ? " was" : "s were"} not traced on this
              model, so the tabs above are not the whole picture:
              <ul className="withheld">
                {data.skipped.map((why) => (
                  <li key={why}>{why}</li>
                ))}
              </ul>
            </p>
          )}

          <div className="patch-grid-wrap">
            <table className="patch-grid" aria-label="recovery by layer and position">
              <thead>
                <tr>
                  <th />
                  {data.corrupt.tokens.map((t, i) => (
                    <th key={i} className="mid" title={t}>
                      {t.trim() || "␣"}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="stagger">
                {/* Deepest layer at the top, so the grid reads the way the
                    model runs down the page. */}
                {[...grid].reverse().map((row, ri) => {
                  const li = data.n_layers - 1 - ri;
                  return (
                    <tr key={li} style={{ "--i": ri } as CSSProperties}>
                      <th className="mid">L{li}</th>
                      {row.map((v, pi) => {
                        const site = controlled.get(`${li}:${pi}`);
                        return (
                          <td
                            key={pi}
                            style={{ background: cell(v) }}
                            className={[
                              site && site.clears_control && site.clears_position
                                ? "clears"
                                : "",
                              pinned && pinned.l === li && pinned.p === pi
                                ? "pinned"
                                : "",
                            ]
                              .filter(Boolean)
                              .join(" ") || undefined}
                            onMouseEnter={() => setHover({ l: li, p: pi })}
                            onMouseLeave={() => setHover(null)}
                            /* A cell is a button, not a decoration: keyboard
                               reachable and announced, because the follow-up
                               measurement is only available through it. */
                            role="button"
                            tabIndex={0}
                            onClick={() => void trace(li, pi)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter" || e.key === " ") {
                                e.preventDefault();
                                void trace(li, pi);
                              }
                            }}
                            aria-label={`layer ${li}, token ${data.corrupt.tokens[pi]}, recovery ${measured(v, 3)} — trace what wrote here`}
                            title={`layer ${li}, ${data.corrupt.tokens[pi]} — ${measured(v, 3)}. Click to trace what wrote here.`}
                          >
                            {Math.abs(v) >= 0.1 ? v.toFixed(2) : ""}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="patch-read meta">
            {hovered ? (
              <>
                layer {hovered.layer}, token{" "}
                <b>{data.corrupt.tokens[hovered.position]}</b> — recovers{" "}
                <b>{hovered.recovery.toFixed(3)}</b> of the gap. The best of{" "}
                {hovered.control_draws} random vectors of the same size at the
                same site got {hovered.control_max.toFixed(3)}, and this layer's
                activation from the next token over got{" "}
                {hovered.shifted_position.toFixed(3)}.{" "}
                {hovered.clears_control && hovered.clears_position
                  ? "It beats both, so it is this place and not the size of the edit."
                  : "It does not beat both, so it is not distinguished from an edit of that size at that layer."}
              </>
            ) : (
              <>
                Blue recovered the clean answer, red pushed it further away. 1.0
                is the clean answer and 0.0 the corrupted one. Outlined cells are
                the {controlled.size} strongest sites in THIS grid, and they are
                the only ones that were tested against chance — hover for the
                verdict.
                {/* `controlled.size`, not `data.controlled`. The outlines are
                    filtered to the component whose tab is open; the sentence
                    quoted the total across ALL components, so the `attn` tab
                    could show eight outlines under a sentence claiming
                    twenty-four. The total is still worth saying — it just has
                    to say that it is the total. */}
                {data.controlled > controlled.size &&
                  ` ${data.controlled} were controlled across all ` +
                    `${data.components.length} components.`}{" "}
                <b>Click any cell</b> to ask what wrote into it: a residual
                stream carries everything that ever wrote there, so this grid
                cannot say which component put it there.
              </>
            )}
          </div>

          {/* ─── what wrote into that site ─────────────────────────────────
              The grid patches a residual stream, which restores everything
              that ever wrote into it AT ONCE. This splits that apart: one
              component's contribution at a time, on the same recovery scale
              and against the same controls, so an edge number and a node
              number can be read together. */}
          {pinned && (
            <div className="patch-path">
              <div className="row">
                <span className="meta">
                  what wrote into <b>layer {pinned.l}</b> at{" "}
                  <b>{data.corrupt.tokens[pinned.p]?.trim() || "␣"}</b>
                </span>
                <span className="spacer" />
                <button
                  className="ghost sm"
                  onClick={() => {
                    setPinned(null);
                    setPath(null);
                    setPathErr("");
                  }}
                >
                  close
                </button>
              </div>

              {tracing && (
                <p className="meta">
                  Patching every earlier head and MLP into this site, one at a
                  time…
                </p>
              )}
              {pathErr && <div className="hint err">{pathErr}</div>}

              {path && !path.senders.length && (
                <p className="meta">
                  Nothing upstream writes into this site — it is at the top of
                  the model's first block, so there is no earlier component to
                  test.
                </p>
              )}

              {path && path.senders.length > 0 && (
                <>
                  {/* Ties, not a ranking. Recovery is quantised by the
                      model's dtype, and senders come back within one
                      representable step of the top, which as a ranked list
                      would have read as a 1st, 2nd and 3rd place that the
                      numbers cannot support. */}
                  <p className="meta">
                    {path.n_senders} components tested, {path.n_controlled}{" "}
                    against chance, in {path.passes} passes ({path.seconds}s).
                    Two senders closer than{" "}
                    {/* Through `measured`, not `toFixed(3)`. This is one step
                        of the model's number format on the recovery scale, and
                        in float32 it is around 1e-6 — so the sentence that
                        exists to publish the tie threshold printed it as
                        "0.000", which says nothing is tied. bfloat16 was the
                        only dtype it read correctly on. */}
                    <b>{measured(path.recovery_resolution, 3)}</b> are{" "}
                    <b>tied</b>, not ranked — that is what one step of this
                    model's number format is worth on this scale.
                  </p>

                  <ol className="path-list stagger">
                    {shownSenders(path, allSenders).map((s, i) => {
                      const top = path.senders[0].recovery;
                      const tied =
                        i > 0 &&
                        Math.abs(top - s.recovery) < path.recovery_resolution;
                      const tested = s.control_max !== undefined;
                      const clears = s.clears_control && s.clears_position;
                      return (
                        <li
                          key={s.name}
                          className={tied ? "tied" : undefined}
                          style={{ "--i": i } as CSSProperties}
                        >
                          <span className="mid path-name">{s.name}</span>
                          <span className="path-track">
                            <span
                              className="path-bar"
                              style={{
                                width: `${Math.min(100, Math.abs(s.recovery) * 100)}%`,
                                background:
                                  s.recovery >= 0
                                    ? "var(--color-cobalt)"
                                    : "var(--crimson-500)",
                              }}
                            />
                          </span>
                          <span className="mid path-val">
                            {s.recovery.toFixed(3)}
                          </span>
                          <span className="meta path-verd">
                            {/* Short enough to fit its column. The full form
                                — "not distinguished from an edit of that size
                                at that layer" — ellipsised at "an edit of t…"
                                on a 1000px viewport, which says nothing at
                                all. What the controls ARE is spelled out in
                                the paragraph below, once, rather than
                                truncated on every row. */}
                            {tied
                              ? "tied with the top"
                              : !tested
                                ? "not tested against chance"
                                : clears
                                  ? "beats both controls"
                                  : "no better than its controls"}
                          </span>
                        </li>
                      );
                    })}
                  </ol>

                  {/* NOT a silent cap. Every sender is counted, the reason
                      the rest are folded away is stated, and one click brings
                      them back — a list that quietly stopped at twelve would
                      read as "these are all of them". */}
                  {path.senders.length > shownSenders(path, allSenders).length && (
                    <button
                      className="ghost sm"
                      onClick={() => setAllSenders(true)}
                    >
                      show the other {path.senders.length -
                        shownSenders(path, allSenders).length}{" "}
                      — all below the{" "}
                      {measured(path.recovery_resolution, 3)} resolution and
                      none tested against chance
                    </button>
                  )}
                  {allSenders && (
                    <button className="ghost sm" onClick={() => setAllSenders(false)}>
                      fold the untested ones away again
                    </button>
                  )}

                  <p className="meta">
                    {path.seeding} {path.scope}
                  </p>
                  <p className="meta">{path.means}</p>
                  <ReceiptLine receipt={path.receipt} />
                </>
              )}
            </div>
          )}

          <ul className="patch-notes meta stagger">
            {data.notes.map((n, i) => (
              <li key={i} style={{ "--i": i } as CSSProperties}>
                {n}
              </li>
            ))}
          </ul>
          <ReceiptLine receipt={data.receipt} />
        </>
      )}

      {/* THE SAME GRID, RANKED IN TWO PASSES INSTEAD OF HUNDREDS — and then
          checked against this one on the few sites it shortlists. It lives
          here, under the exact answer, because that is the order the claim
          has to be read in: the screen is an approximation OF the measurement
          above, and its numbers are deliberately named differently so a
          reader cannot copy one out believing it is the other.

          Gated as every live-model control here is: a static page has no
          model to run a gradient pass against. */}
      {!DEMO && !VIEWER && <PatchScreenPanel disabled={busy} />}
    </div>
  );
}
