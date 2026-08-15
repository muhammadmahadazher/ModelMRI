import { useEffect, useState } from "react";
import {
  BundlePreview,
  bundlePreview,
  errorText,
  exportSession,
  TraceStep,
} from "./api";

/**
 * Ship an agent failure as one file.
 *
 * Every competitor's share artefact is a link into their hosted trace UI,
 * which dies when the account lapses — Helicone went into maintenance mode in
 * March 2026, Langfuse changed owners in January, migration checklists are
 * circulating. This is a gzipped file that opens in a browser with nothing
 * installed: the recipient sees the failing tool call, clicks it, and lands in
 * the attention view of the generation that produced the bad argument, on a
 * machine with no GPU.
 *
 * **The preview is not decoration.** This is the one path in the project where
 * data leaves the machine, so what is about to leave is on screen before the
 * button is. Redaction runs at export — the recorder's own pass happens at
 * delivery, which is long behind us by the time steps come out of the store —
 * and the count of what it replaced is shown, along with the honest caveat
 * that "none found" is not "none there".
 */
export default function ShareRun({
  traceId,
  selected,
  ready,
}: {
  traceId: string;
  /** The step to open the bundle on, when the reader has picked one. */
  selected: TraceStep | null;
  /** A mechanistic snapshot exists to bundle the run with. Without it there
   *  is still a file worth sharing, but it is a timeline and nothing more. */
  ready: boolean;
}) {
  const [prev, setPrev] = useState<BundlePreview | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [withStep, setWithStep] = useState(true);

  // Refetched whenever the run changes, so the numbers on screen belong to
  // the file the button would write.
  useEffect(() => {
    if (!traceId) return;
    let live = true;
    setErr("");
    void bundlePreview(traceId)
      .then((p) => live && setPrev(p))
      .catch((e) => {
        if (!live) return;
        setPrev(null);
        setErr(errorText(e));
      });
    return () => {
      live = false;
    };
  }, [traceId]);

  async function save() {
    setBusy(true);
    setErr("");
    try {
      const step = withStep && selected ? selected.id : "";
      const { blob, filename } = await exportSession(0, 0, "", {
        trace_id: traceId,
        step_ref: step,
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  if (!traceId) return null;

  return (
    <div className="share-run">
      <div className="row">
        <span className="meta">
          <b>send this run to somebody</b> — one file, opens in a browser with
          nothing installed, no account to lapse
        </span>
      </div>

      {prev && (
        <div className="share-prev">
          <div className="meta share-title">what will be in the file</div>
          <dl className="share-grid">
            <div>
              <dt className="meta">steps</dt>
              <dd className="mid">{prev.n_steps.toLocaleString()}</dd>
            </div>
            <div>
              <dt className="meta">fields scanned</dt>
              <dd className="mid">{prev.fields_scanned.toLocaleString()}</dd>
            </div>
            <div>
              <dt className="meta">replaced</dt>
              <dd className={prev.n_redactions ? "mid share-hit" : "mid"}>
                {prev.n_redactions.toLocaleString()}
              </dd>
            </div>
          </dl>
          {/* The sentence is authored server-side beside the redaction, so
              the panel cannot claim something the pass did not do. */}
          <p className="meta share-means">{prev.means}</p>
          {prev.redactions.length > 0 && (
            <ul className="share-kinds">
              {prev.redactions.map((r) => (
                <li key={r.label} className="meta">
                  <b className="mid">{r.count}x</b> {r.label}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <div className="row" style={{ marginTop: 8 }}>
        <label className="meta share-opt">
          <input
            type="checkbox"
            checked={withStep && Boolean(selected)}
            disabled={!selected}
            onChange={(e) => setWithStep(e.target.checked)}
          />{" "}
          {selected
            ? `open it on step ${selected.seq} (${selected.name || selected.kind})`
            : "open it on a step — pick one on the timeline first"}
        </label>
      </div>

      <div className="row" style={{ marginTop: 6 }}>
        <button className="cta" onClick={() => void save()} disabled={busy}>
          {busy ? "writing…" : "Save this run (.mri)"}
        </button>
        {/* A sentence, not a disabled button. The file is still worth having
            without a snapshot — it is a timeline somebody else can read. */}
        {!ready && (
          <span className="meta">
            no generation is loaded, so this carries the run and not the
            attention behind it
          </span>
        )}
      </div>

      {err && <div className="hint err">{err}</div>}
    </div>
  );
}
