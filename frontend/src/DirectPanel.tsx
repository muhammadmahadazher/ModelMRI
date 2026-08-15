import { CSSProperties, useEffect, useState } from "react";
import { DirectAttribution, errorText, getDirectAttribution } from "./api";
import ReceiptLine from "./ReceiptLine";

/**
 * How many logits each head and MLP put behind the token the model chose.
 *
 * Sited inside the ablation panel on purpose. The two measurements answer
 * related questions and DISAGREE — the ranking says what breaks when a head is
 * removed, this says what a head put into the logit directly, and a head can be
 * near zero here and still decide the answer by feeding a later head. Putting
 * this in its own panel would let a reader take either number as the whole
 * story.
 *
 * Two things this renders that a normal bar chart would not:
 *
 *   - The RECONSTRUCTION RESIDUAL, always. Direct attribution is exact only if
 *     the final normalisation is linear, and it is not. On GPT-2 the pieces
 *     sum to about 99% of the real logit, and a chart that did not say so
 *     would be claiming a decomposition it does not have.
 *   - Components under that residual as UNREADABLE rather than small. A bar
 *     shorter than the error the approximation already makes cannot be told
 *     from that error.
 */
export default function DirectPanel({ epoch }: { epoch: number }) {
  const [data, setData] = useState<DirectAttribution | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [showUnreadable, setShowUnreadable] = useState(false);

  async function run() {
    setBusy(true);
    setErr("");
    try {
      setData(await getDirectAttribution(60));
    } catch (e) {
      setErr(errorText(e));
      setData(null);
    } finally {
      setBusy(false);
    }
  }

  // Cleared on a new generation rather than compared against an epoch the
  // response does not carry. The attribution is about one token at one
  // position of ONE run, and the runtime's epoch deliberately does not move
  // on generation, so it could not have answered this question anyway.
  useEffect(() => {
    setData(null);
    setErr("");
  }, [epoch]);

  const shown = data
    ? data.components.filter((c) => showUnreadable || !c.unreadable)
    : [];
  // Scaled against the strongest bar rather than the total: the pieces are
  // signed and do not sum to the whole, so a percentage-of-total width would
  // be arithmetic about nothing.
  const widest = shown.reduce((m, c) => Math.max(m, Math.abs(c.logits)), 0.0001);
  const hidden = data ? data.components.length - shown.length : 0;

  return (
    <div className="direct">
      <div className="row" style={{ marginBottom: 10 }}>
        <button className="ghost" onClick={() => void run()} disabled={busy}>
          {busy ? "Decomposing the logit…" : "Where did this logit come from?"}
        </button>
        <span className="meta">
          direct contribution per head and MLP — a different question from the
          ranking above
        </span>
      </div>

      {err && <div className="hint err">{err}</div>}
      {data && (
        <>
          <div className="direct-head">
            <b>{data.token.replace(/ /g, "·") || "␀"}</b> at position{" "}
            {data.position} — logit <b>{data.real_logit.toFixed(3)}</b>
          </div>

          <div className="dla-bars stagger">
            {shown.map((c, i) => (
              <div
                className={`dla-row${c.unreadable ? " faint" : ""}`}
                key={c.name}
                style={{ "--i": i } as CSSProperties}
                title={
                  c.unreadable
                    ? "under the reconstruction residual — direct-path attribution cannot separate this from the approximation's own error"
                    : `${c.logits > 0 ? "+" : ""}${c.logits} logits`
                }
              >
                <span className="dla-name">{c.name}</span>
                <span className="dla-track">
                  {/* Signed, from a centre line. A component pushing AGAINST
                      the chosen token is half the mechanism and folding it
                      into a magnitude would hide it. */}
                  <i
                    className={c.logits >= 0 ? "pos" : "neg"}
                    style={{
                      width: `${(Math.abs(c.logits) / widest) * 50}%`,
                      [c.logits >= 0 ? "left" : "right"]: "50%",
                    } as CSSProperties}
                  />
                </span>
                <span className="dla-val">
                  {c.logits > 0 ? "+" : ""}
                  {c.logits.toFixed(3)}
                </span>
              </div>
            ))}
          </div>

          {hidden > 0 && (
            <button className="linkish" onClick={() => setShowUnreadable(true)}>
              show {hidden} components under the residual
            </button>
          )}
          {showUnreadable && hidden === 0 && (
            <button className="linkish" onClick={() => setShowUnreadable(false)}>
              hide components under the residual
            </button>
          )}

          {/* MANDATORY, not an option. Without it the chart claims a
              decomposition it does not have. */}
          <div className="hint residual">
            The pieces sum to{" "}
            <b>{(data.real_logit - data.residual).toFixed(3)}</b> against the
            model's real <b>{data.real_logit.toFixed(3)}</b> — a reconstruction
            residual of <b>{data.residual.toFixed(4)}</b> (
            {(data.residual_share * 100).toFixed(2)}%), which is what freezing
            the {data.norm_kind} scale cost on this run. The model was not
            modified to make this exact.
          </div>
          <div className="hint">
            <b>Direct path only.</b> {data.n_unreadable} of{" "}
            {data.components.length} components fall below that residual: their
            direct effect cannot be told from the approximation's own error,
            which is <em>not</em> the same as their being unimportant. A head
            that feeds a later head shows near zero here and can still decide
            the answer — which is what the ablation ranking above measures
            instead.
          </div>
          <ReceiptLine receipt={data.receipt} />
        </>
      )}
    </div>
  );
}
