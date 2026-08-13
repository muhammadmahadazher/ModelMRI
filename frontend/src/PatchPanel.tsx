import { CSSProperties, useState } from "react";
import { errorText, patchTrace, PatchTrace } from "./api";
import ReceiptLine from "./ReceiptLine";
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
        <button className="cta" onClick={() => void run()} disabled={busy}>
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
                            className={
                              site && site.clears_control && site.clears_position
                                ? "clears"
                                : undefined
                            }
                            onMouseEnter={() => setHover({ l: li, p: pi })}
                            onMouseLeave={() => setHover(null)}
                            title={`layer ${li}, ${data.corrupt.tokens[pi]} — ${v.toFixed(3)}`}
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
                the {data.controlled} strongest sites, and they are the only ones
                that were tested against chance — hover for the verdict.
              </>
            )}
          </div>

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
    </div>
  );
}
