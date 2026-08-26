import { CSSProperties, useEffect, useState } from "react";
import { measured, percent } from "./measured";
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
export default function LensPanel({
  epoch,
  recorded,
}: {
  epoch: number;
  /** Set when a `.mri` is open and carries a trajectory: the answer it ended
   *  on, so the panel offers the recording instead of a button whose only
   *  outcome is a refusal.
   *
   *  There is nothing here to re-run. A lens means walking the residual
   *  stream through the unembedding at every depth, and a `.mri` holds
   *  activations rather than weights — so in this mode every control that
   *  needs the model is gone rather than disabled, which is `PatchPanel`'s
   *  rule after the same bug: a button labelled "Show the recorded trace"
   *  that was disabled underneath the label in every viewer build. */
  recorded?: { available: boolean; n_rows: number; final: string };
}) {
  const [rows, setRows] = useState<LensRow[] | null>(null);
  const [final, setFinal] = useState("");
  // THREE STATES, not two. A number is the layer the answer first stays on
  // top at; `null` is the finding "it never gets on top before the last
  // layer"; `undefined` is a file that never said. `session._lens` keeps
  // `settled_at: null` deliberately and drops the key when it is absent, so
  // collapsing the two here would print a finding over a silence.
  const [settled, setSettled] = useState<number | null | undefined>(undefined);
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
  //
  // NOT ASKED FOR A RECORDING. A translator is fitted to the LIVE model, and
  // a `.mri` was recorded on somebody else's — so a trained lens found here
  // would caption a stranger's trajectory with "N of M layers improved" about
  // a model that never produced it.
  useEffect(() => {
    if (recorded) return;
    let live = true;
    void tunedLensStatus()
      .then((s) => {
        if (live && s.trained && s.info) setTunedInfo(s.info);
      })
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [epoch, recorded]);

  async function run() {
    setBusy(true);
    setErr("");
    try {
      const d = await getLens(4, !recorded && tunedInfo ? "both" : "plain");
      setRows(d.layers);
      // `?? ""` and `d.settled_at` LEFT ALONE. A recording's scalars are
      // whatever `lens_info` carried: the live route always sends both, a
      // `.mri` need not, and `final.trim()` on an absent one is a TypeError
      // that takes the whole page down with it. "" is what this panel already
      // reads as "the answer was not named" — see the sentence below, which
      // falls back to "the answer" — while `settled_at` keeps its third state.
      setFinal(d.final ?? "");
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

  // Defensive by TYPE, not by presence, and for GraphPanel's reason: this
  // renders a file a stranger forwarded, `parse()` in viewer.ts does not walk
  // the lens section, and `frontend/src` has no error boundary — so a row
  // carrying `entropy: "high"` would take the whole page white rather than
  // one cell. Python validates its side; this is the side that must not
  // trust Python having run.
  const nats = (v: unknown): number | null =>
    typeof v === "number" && Number.isFinite(v) ? v : null;
  // The widest bar is the largest entropy ANYBODY MEASURED. `reduce`, not
  // `Math.max(...spread)`, so a long trajectory cannot throw RangeError; and
  // no 0.001 floor, because the floor was standing in for the divide-by-zero
  // that only happens when there is nothing to scale — which is now answered
  // by drawing no bar at all.
  const peakH = rows
    ? rows.reduce((m, r) => Math.max(m, nats(r.entropy) ?? 0), 0)
    : 0;
  // BY LAYER, never by index. The plain lens has one more row than the tuned
  // one -- the model's own final state, which needs no translator because it
  // is the answer -- so zipping the two arrays positionally would put every
  // tuned row one layer out from the plain row beside it.
  const tunedAt = new Map((tuned ?? []).map((r) => [r.layer, r]));
  const gainAt = new Map(
    (tunedInfo?.layers ?? []).map((r) => [r.layer, r.gain]),
  );

  const body = (
    <div className="lens">
      <div className="row" style={{ marginBottom: 10 }}>
        <button className="violet" onClick={() => void run()} disabled={busy}>
          {busy
            ? "Reading every layer…"
            : recorded
              ? "Show the recorded lens"
              : "Run the logit lens"}
        </button>
        <span className="meta">
          {recorded
            ? "what that model would have said if it stopped at each layer — read on the sender's machine"
            : "what this model would have said if it stopped at each layer"}
        </span>
      </div>

      {recorded && (
        <p className="meta">
          This is a recording. The trajectory is already in the file and
          nothing here re-reads it: a lens means walking the residual stream
          through the unembedding at every depth, and a `.mri` holds
          activations rather than weights.
          {recorded.final.trim() && (
            <>
              {" "}
              The model ended on <b>{recorded.final.trim()}</b>.
            </>
          )}
        </p>
      )}

      {/* GONE, not disabled, on a recording. A translator is fitted to a LIVE
          model over a corpus — minutes of compute — and the model this
          trajectory came off is not on this machine and never will be. */}
      {!recorded && (
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
      )}

      {err && <div className="hint err">{err}</div>}

      {/* AN EMPTY ARRAY IS TRUTHY. `rows = []` rendered the table header with
          no body and no error — the reader clicked "Run the logit lens",
          watched the label change and change back, and saw nothing appear.
          A result with no rows is a finding and gets a sentence. */}
      {rows && rows.length === 0 && (
        <div className="hint">
          The lens ran and read no layers. That is not an error and not an
          empty answer to your prompt — it means this model exposes no
          decoder blocks this can walk, so there was nothing to unembed at
          each depth.
        </div>
      )}

      {rows && rows.length > 0 && (
        <>
          {/* Rows settle in order, so the eye reads the stack the way the
              model runs it. No scan here: this panel is mounted inside the
              features panel, which already flashes on arrival, and two
              flashes for one piece of news is noise. */}
          <div className="lens-table stagger" role="table" aria-label="logit lens">
            <div className={`lens-row head${tuned ? " twin" : ""}`} role="row">
              <span>layer</span>
              <span>entropy</span>
              {/* The lens's own error, in the same units as a head score —
                  which is why `lens.py` computes it in that direction. It was
                  measured on every row and rendered nowhere. */}
              <span title="KL(truth ‖ lens), in nats: how much is lost by reading this layer instead of the model's answer.">
                lost
              </span>
              <span>would say{tuned ? " (plain)" : ""}</span>
              {tuned && <span>tuned · held-out KL change</span>}
            </div>
            {rows.map((r, ri) => {
              const agrees = r.tokens[0] === final;
              const h = nats(r.entropy);
              return (
                <div
                  className={`lens-row ${tuned ? "twin " : ""}${
                    agrees ? "agrees" : ""
                  } ${typeof settled === "number" && r.layer === settled ? "settle" : ""}`}
                  key={r.layer}
                  role="row"
                  style={{ "--i": ri } as CSSProperties}
                >
                  <span className="l-name">
                    {r.layer === 0 ? "embed" : `L ${String(r.layer).padStart(2, "0")}`}
                  </span>
                  {/* NO BAR AT ALL when nobody measured it. A zero-width bar
                      beside "0.00" is a reading — "this layer was already
                      certain" — and that is the opposite of what an absent
                      entropy says. `measured` answers "—" for an unknown,
                      which is the same mark `lost` uses one column along. */}
                  <span className="lens-h">
                    {h !== null && peakH > 0 && (
                      <i style={{ width: `${(h / peakH) * 100}%` }} />
                    )}
                    {measured(h, 2)}
                  </span>
                  <span className="lens-kl mid">
                    {r.kl_to_final === undefined ? "—" : measured(r.kl_to_final, 3)}
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
                  {/* `length > 0` for the same reason as the table above:
                      an empty array is truthy, so a tuned lens that read no
                      layers drew a second column with a header and nothing
                      under it. */}
                  {tuned && tuned.length > 0 && (
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
            {typeof settled === "number" && settled > 0 ? (
              <>
                {final.trim() || "the answer"} first stays on top at layer{" "}
                <b>{settled}</b> and never loses it — everything below that is
                the model still deciding.
              </>
            ) : settled === undefined ? (
              /* THE THIRD STATE. A `.mri` need not carry `settled_at`, and
                 the sentence below is a FINDING about the model — saying it
                 over a file that never took that reading would be inventing
                 the measurement this tool exists to demand. */
              <>
                This recording does not say where the answer settled, so the
                trajectory above is the whole of what it carries.
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

  // Inside the features panel this is a SECTION of one — it has a heading
  // above it already, and `.lens`'s hairline is what separates it from the
  // SAE half. A recording has no features panel to sit in (that one carries
  // live-model controls and is `!replay` by construction), so on that path it
  // becomes a panel of its own, with the features accent because the lens is
  // the answer for every model that has no sparse autoencoder.
  if (!recorded) return body;
  return (
    <div className="panel lensp">
      <div className="sect">
        <span className="dot d-feat" />
        <h2 className="h-feat">LOGIT LENS — WHAT IT WOULD HAVE SAID, LAYER BY LAYER</h2>
        <span className="rule" />
      </div>
      {body}
    </div>
  );
}
