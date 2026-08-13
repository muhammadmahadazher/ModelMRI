import { CSSProperties, useEffect, useState } from "react";
import { errorText, getLens, LensRow, Receipt } from "./api";
import ReceiptLine from "./ReceiptLine";

/** Logit lens — the answer for every model that has no sparse autoencoder.
 *
 *  Most models have none and never will: an SAE is trained per model, and
 *  public ones exist for only a handful of models. This asks the other
 *  question you can put to a residual stream, using nothing but the model —
 *  what token would it have emitted if it had stopped at layer N.
 *
 *  Deliberately labelled as *not* features. It is a coarser probe, and saying
 *  so is the difference between a fallback and a substitute.
 */
export default function LensPanel({ epoch }: { epoch: number }) {
  const [rows, setRows] = useState<LensRow[] | null>(null);
  const [final, setFinal] = useState("");
  const [settled, setSettled] = useState<number | null>(null);
  const [receipt, setReceipt] = useState<Receipt | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    setRows(null);
    setErr("");
    // The receipt names the prompt this lens was read on, so it goes when the
    // rows do -- leaving it would caption a cleared table with the setup of a
    // run that is no longer on screen.
    setReceipt(null);
  }, [epoch]);

  async function run() {
    setBusy(true);
    setErr("");
    try {
      const d = await getLens(4);
      setRows(d.layers);
      setFinal(d.final);
      setSettled(d.settled_at);
      setReceipt(d.receipt ?? null);
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  const maxH = rows ? Math.max(...rows.map((r) => r.entropy), 0.001) : 1;

  return (
    <div className="lens">
      <div className="row" style={{ marginBottom: 10 }}>
        <button className="violet" onClick={() => void run()} disabled={busy}>
          {busy ? "Reading every layer…" : "Run the logit lens"}
        </button>
        <span className="meta">
          what this model would have said if it stopped at each layer
        </span>
      </div>

      {err && <div className="hint err">{err}</div>}

      {rows && (
        <>
          {/* Rows settle in order, so the eye reads the stack the way the
              model runs it. No scan here: this panel is mounted inside the
              features panel, which already flashes on arrival, and two
              flashes for one piece of news is noise. */}
          <div className="lens-table stagger" role="table" aria-label="logit lens">
            <div className="lens-row head" role="row">
              <span>layer</span>
              <span>entropy</span>
              <span>would say</span>
            </div>
            {rows.map((r, ri) => {
              const agrees = r.tokens[0] === final;
              return (
                <div
                  className={`lens-row ${agrees ? "agrees" : ""} ${
                    settled !== null && r.layer === settled ? "settle" : ""
                  }`}
                  key={r.layer}
                  role="row"
                  style={{ "--i": ri } as CSSProperties}
                >
                  <span className="l-name">
                    {r.layer === 0 ? "embed" : `L ${String(r.layer).padStart(2, "0")}`}
                  </span>
                  <span className="lens-h">
                    <i style={{ width: `${(r.entropy / maxH) * 100}%` }} />
                    {r.entropy.toFixed(2)}
                  </span>
                  <span className="lens-toks">
                    {r.tokens.map((t, i) => (
                      <span
                        className={`lens-tok ${i === 0 ? "top" : ""}`}
                        key={i}
                        title={`p = ${r.probs[i]}`}
                      >
                        {t.replace(/ /g, "·") || "␀"}
                        <em>{(r.probs[i] * 100).toFixed(0)}%</em>
                      </span>
                    ))}
                  </span>
                </div>
              );
            })}
          </div>
          <div className="hint">
            {settled !== null && settled > 0 ? (
              <>
                {final.trim() || "the answer"} first stays on top at layer{" "}
                <b>{settled}</b> and never loses it — everything below that is
                the model still deciding.
              </>
            ) : (
              <>
                The answer only arrives at the last layer here, so this
                generation gives no earlier settling point.
              </>
            )}{" "}
            · a probe, not features: it reads which token the stream points at,
            not which concepts are active · entropy falls as the model commits
          </div>
          <ReceiptLine receipt={receipt} />
        </>
      )}
    </div>
  );
}
