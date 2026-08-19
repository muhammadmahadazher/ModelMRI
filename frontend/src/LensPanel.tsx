import { CSSProperties, useEffect, useState } from "react";
import { percent } from "./measured";
import {
  errorText,
  getLens,
  LensRow,
  Receipt,
  trainTunedLens,
  tunedLensStatus,
  TunedLensInfo,
} from "./api";
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
  // The tuned column, when one has been trained. NEVER replaces `rows`: a
  // translator fitted to minimise disagreement with the final distribution
  // will reduce disagreement with the final distribution, so a reader shown
  // only the tuned rows has no way to tell the model from the fit.
  const [tuned, setTuned] = useState<LensRow[] | null>(null);
  const [tunedInfo, setTunedInfo] = useState<TunedLensInfo | null>(null);
  const [training, setTraining] = useState(false);
  const [corpus, setCorpus] = useState("");

  useEffect(() => {
    setRows(null);
    setErr("");
    // The receipt names the prompt this lens was read on, so it goes when the
    // rows do -- leaving it would caption a cleared table with the setup of a
    // run that is no longer on screen.
    setReceipt(null);
    // The tuned READING goes with the rows -- it describes this generation.
    // The trained translator itself does not: it belongs to the model and
    // survives a new prompt, which is the whole point of caching it.
    setTuned(null);
  }, [epoch]);

  // A trained translator lives on the server and is cached on disk, so it
  // survives a reload while this component's state does not. Without asking,
  // the panel came back after any refresh believing no lens existed and
  // quietly requested `kind=plain` — the tuned column simply never appeared
  // again, with nothing on screen saying why.
  useEffect(() => {
    let live = true;
    void tunedLensStatus()
      .then((s) => {
        if (live && s.trained && s.info) setTunedInfo(s.info);
      })
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [epoch]);

  async function run() {
    setBusy(true);
    setErr("");
    try {
      const d = await getLens(4, tunedInfo ? "both" : "plain");
      setRows(d.layers);
      setFinal(d.final);
      setSettled(d.settled_at);
      setReceipt(d.receipt ?? null);
      setTuned(d.tuned ?? null);
      if (d.tuned_info) setTunedInfo(d.tuned_info);
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  // One sequence per line, which is the same shape `modelmri sweep` reads, so
  // a corpus file that works for one works for the other. Derived once: the
  // count is shown, the button gates on it, and the request sends it, and
  // three copies of the same split is three chances for them to disagree.
  const lines = corpus
    .split("\n")
    .map((t) => t.trim())
    .filter(Boolean);

  async function train() {
    setTraining(true);
    setErr("");
    try {
      const info = await trainTunedLens({
        // One sequence per line, which is the same shape `modelmri sweep`
        // reads, so a corpus file that works for one works for the other.
        texts: lines,
        steps: 250,
      });
      setTunedInfo(info);
      const d = await getLens(4, "both");
      setRows(d.layers);
      setFinal(d.final);
      setSettled(d.settled_at);
      setTuned(d.tuned ?? null);
      setReceipt(d.receipt ?? null);
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setTraining(false);
    }
  }

  const maxH = rows ? Math.max(...rows.map((r) => r.entropy), 0.001) : 1;
  // BY LAYER, never by index. The plain lens has one more row than the tuned
  // one -- the model's own final state, which needs no translator because it
  // is the answer -- so zipping the two arrays positionally would put every
  // tuned row one layer out from the plain row beside it.
  const tunedAt = new Map((tuned ?? []).map((r) => [r.layer, r]));
  const gainAt = new Map(
    (tunedInfo?.layers ?? []).map((r) => [r.layer, r.gain]),
  );

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

      <details className="tune">
        <summary>
          {tunedInfo
            ? `tuned lens: ${tunedInfo.n_layers_improved} of ${tunedInfo.n_layers} layers improved on held-out text`
            : "train a tuned lens on your own text"}
        </summary>
        <p className="hint">
          A tuned lens learns a per-layer map so each layer is read through a
          transform fitted to <em>it</em>, instead of the one fitted to the
          last layer. It is trained here, on this machine.{" "}
          <b>Nothing is downloaded</b> — pretrained lenses exist, and fetching
          one would break the offline promise the rest of this tool keeps.
        </p>
        <textarea
          className="tune-corpus"
          rows={4}
          placeholder={
            "One sequence per line. Your own logs, your own prompts, your own " +
            "documents — at least 8 lines, because some are held back to " +
            "measure the result on text the translator never saw."
          }
          value={corpus}
          onChange={(e) => setCorpus(e.target.value)}
        />
        <div className="row">
          <button
            className="ghost"
            onClick={() => void train()}
            disabled={training || lines.length < 8}
          >
            {training ? "Fitting a translator per layer…" : "Train"}
          </button>
          <span className="meta">
            {lines.length} sequences
          </span>
        </div>
        {tunedInfo?.caution && (
          /* The size of the fit relative to the corpus, which decides how
             seriously to take the column. Shown whenever it applies rather
             than only on request. */
          <div className="hint warn">{tunedInfo.caution}</div>
        )}
        {tunedInfo?.means && <div className="hint">{tunedInfo.means}</div>}
      </details>

      {err && <div className="hint err">{err}</div>}

      {rows && (
        <>
          {/* Rows settle in order, so the eye reads the stack the way the
              model runs it. No scan here: this panel is mounted inside the
              features panel, which already flashes on arrival, and two
              flashes for one piece of news is noise. */}
          <div className="lens-table stagger" role="table" aria-label="logit lens">
            <div className={`lens-row head${tuned ? " twin" : ""}`} role="row">
              <span>layer</span>
              <span>entropy</span>
              <span>would say{tuned ? " (plain)" : ""}</span>
              {tuned && <span>tuned · held-out KL change</span>}
            </div>
            {rows.map((r, ri) => {
              const agrees = r.tokens[0] === final;
              return (
                <div
                  className={`lens-row ${tuned ? "twin " : ""}${
                    agrees ? "agrees" : ""
                  } ${settled !== null && r.layer === settled ? "settle" : ""}`}
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
                        <em>{percent(r.probs[i], 0)}</em>
                      </span>
                    ))}
                  </span>
                  {tuned && (
                    <span className="lens-toks tuned-col">
                      {tunedAt.has(r.layer) ? (
                        <>
                          {tunedAt.get(r.layer)!.tokens.map((t, i) => (
                            <span
                              className={`lens-tok ${i === 0 ? "top" : ""}`}
                              key={i}
                              title={`p = ${tunedAt.get(r.layer)!.probs[i]}`}
                            >
                              {t.replace(/ /g, "·") || "␀"}
                              <em>
                                {percent(tunedAt.get(r.layer)!.probs[i], 0)}
                              </em>
                            </span>
                          ))}
                          {gainAt.has(r.layer) && (
                            /* Signed on purpose. A translator that made a
                               layer WORSE on held-out text is a finding, and
                               clamping it at zero would hide the one row that
                               says the fit did not help. */
                            <span
                              className={`lens-gain ${
                                (gainAt.get(r.layer) ?? 0) > 0 ? "up" : "down"
                              }`}
                            >
                              {/* The sign is the DIRECTION OF THE KL, not of
                                  a score: a positive gain means the
                                  translator moved this layer closer to the
                                  model, which shows as the KL going down.
                                  Labelling the column "gain" while printing a
                                  minus read as a loss. */}
                              {(gainAt.get(r.layer) ?? 0) > 0 ? "−" : "+"}
                              {Math.abs(gainAt.get(r.layer) ?? 0).toFixed(2)}{" "}
                              nats KL
                            </span>
                          )}
                        </>
                      ) : (
                        /* The final row has no translator. Said, not left
                           blank: an empty cell reads as a missing measurement
                           rather than as one that would be meaningless. */
                        <span className="meta">
                          no translator — this row is the model
                        </span>
                      )}
                    </span>
                  )}
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
