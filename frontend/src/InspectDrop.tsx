import { useRef, useState } from "react";
import { errorText, importInspect, InspectImport } from "./api";

/**
 * Drop a UK AISI Inspect `.eval` log onto this timeline.
 *
 * Inspect is where eval interop is consolidating — Docent integrates natively,
 * Apollo publicly adopted it. Reading it turns this panel into a second viewer
 * for logs people already have, and stops `.mri` being a private dialect.
 *
 * Two things this draws that a quiet importer would not:
 *
 *   - **WHAT WAS DROPPED.** Inspect's schema is not frozen, so any event kind
 *     with no step kind here is counted and named. Showing 40 steps from a
 *     90-event sample without saying so is the failure this avoids.
 *   - **WHY THIS SAMPLE.** The reader opens the FAILING sample by default, and
 *     the banner says that is why it is on screen — otherwise the reader is
 *     looking at one row of a 4,000-row eval with no idea how it was chosen.
 *
 * Reader only. There is no writer and there will not be one: tracking an
 * unfrozen schema in both directions forever is not solo-maintainer work, and
 * somebody with Inspect logs already has Inspect's viewer.
 */
export default function InspectDrop({
  onImported,
}: {
  /** Hand the panel the trace id it should select. */
  onImported: (traceId: string) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [out, setOut] = useState<InspectImport | null>(null);
  // The log itself, kept so a different sample can be read without asking for
  // the file again. A dropped file never reaches `input.files` — it arrives
  // through `dataTransfer` — so reading the input would leave the sample
  // picker dead for exactly the interaction this component is named after.
  const [file, setFile] = useState<File | null>(null);

  async function take(chosen: File | undefined, sampleId = "") {
    if (!chosen) return;
    setBusy(true);
    setErr("");
    try {
      const result = await importInspect(chosen, sampleId);
      setOut(result);
      setFile(chosen);
      onImported(result.trace_id);
    } catch (e) {
      setOut(null);
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  const dropped = out ? Object.entries(out.mapping.dropped) : [];

  return (
    <div
      className={`ins-drop ${dragging ? "over" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        void take(e.dataTransfer.files?.[0]);
      }}
    >
      <input
        ref={input}
        type="file"
        accept=".eval"
        hidden
        onChange={(e) => {
          void take(e.target.files?.[0]);
          e.target.value = ""; // so re-picking the same file fires again
        }}
      />

      <button
        className="ins-target"
        onClick={() => input.current?.click()}
        disabled={busy}
      >
        {busy
          ? "reading…"
          : "Drop an Inspect .eval log here, or click to choose one"}
      </button>
      <span className="meta">
        read-only — the samples, messages and tool calls render on the timeline
        above
      </span>

      {err && <div className="hint err">{err}</div>}

      {out && (
        <div className="ins-result">
          <div className="meta">
            <b>{out.header.task || "an eval"}</b> · {out.header.model || "model not stated"}{" "}
            · log format v{out.header.version} · {out.samples_total} sample
            {out.samples_total === 1 ? "" : "s"}
            {/* `samples_total`, not `samples.length`. The list is capped
                server-side, and printing its length stated the cap as a fact
                about the reader's own file: a 6,000-sample log read "5000
                samples". The dropdown below holds the listed subset, so when
                the two differ that has to be said rather than left to be
                discovered by a sample being unselectable. */}
            {out.samples_truncated && (
              <>
                {" "}
                — the first {out.samples.length} are listed below
              </>
            )}
          </div>

          {/* Why THIS sample. Otherwise the reader is looking at one row of a
              4,000-row eval with no idea how it was chosen. */}
          <div className={`ins-why ${out.failed ? "failed" : ""}`}>
            {out.failed ? (
              <>
                Showing <b>{out.trace.name}</b> — the first sample this log
                marks as failed.
                {out.error && <> The log records: {out.error}</>}
              </>
            ) : (
              <>
                Showing <b>{out.trace.name}</b> — no sample in this log is
                marked failed, so this is the first one.
              </>
            )}
          </div>

          {/* What was NOT mapped. The sentence is authored server-side beside
              the mapping, so the panel cannot claim coverage it did not have. */}
          {dropped.length > 0 && (
            <div className="ins-dropped">
              <div className="meta">
                <b>not shown on the timeline</b>
              </div>
              <ul className="ins-kinds">
                {dropped.map(([kind, n]) => (
                  <li key={kind} className="meta">
                    <b className="mid">{n}x</b> {kind}
                  </li>
                ))}
              </ul>
            </div>
          )}
          <p className="meta ins-means">{out.mapping.means}</p>

          {out.samples.length > 1 && (
            <div className="row ins-pick">
              <label className="meta" htmlFor="ins-sample">
                open another sample
              </label>
              <select
                id="ins-sample"
                value={out.trace.name.split(" ")[0]}
                onChange={(e) => {
                  if (file && e.target.value) void take(file, e.target.value);
                }}
                disabled={busy || !file}
                title="read a different sample from the same log"
              >
                {out.samples.map((s) => (
                  <option key={`${s.id}-${s.epoch}`} value={s.id}>
                    {s.id}
                    {s.epoch > 1 ? ` (epoch ${s.epoch})` : ""}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
