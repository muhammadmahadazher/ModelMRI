// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

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
 *   - **WHAT IT SCORED.** An eval log's whole output is its scores, and this
 *     panel drew the timeline without them: `out.scores` arrived in the
 *     payload, typed in `api.ts`, and nothing read it. The unscored state is
 *     a sentence rather than a blank or a zero, because a sample nobody
 *     scored and a sample scored zero are different facts.
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
  const scores = out ? Object.entries(out.scores) : [];
  const unreadable = out ? Object.entries(out.skipped_scores) : [];

  // ONE entry per sample id, not one per (id, epoch).
  //
  // The import route takes a `sample_id` and nothing else, and the reader
  // answers with the LOWEST epoch of that id. So an option labelled
  // "a (epoch 3)" opened epoch 1 and said nothing — a cosmetic timeline swap
  // while the panel drew only steps, and a wrong measurement now that it
  // draws the scores: epoch 1's `match=I` on screen under the epoch 3 the
  // reader chose. Offering only what can actually be opened is the honest
  // half of that; the other half is a route that takes an epoch, and it is
  // in a file this lane does not own.
  const pickable: { id: string; epochs: number[] }[] = [];
  for (const s of out?.samples ?? []) {
    const seen = pickable.find((p) => p.id === s.id);
    if (seen) seen.epochs.push(s.epoch);
    else pickable.push({ id: s.id, epochs: [s.epoch] });
  }
  for (const p of pickable) p.epochs.sort((a, b) => a - b);

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
                samples". "were read", not "are listed below": the dropdown
                holds one entry per sample ID and the listed subset counts
                epochs, so the two numbers differ on a multi-epoch eval and
                only the first is a fact about the reader's file. */}
            {out.samples_truncated && (
              <>
                {" "}
                — the first {out.samples.length} were read
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

          {/* WHAT THE EVAL SCORED — the one thing an eval produces, and the
              thing this panel used to leave on the floor. Never a table of
              numbers: Inspect's canonical value is the string "C" or "I", and
              a panel that coerced those to 1 and 0 would be showing a number
              the log never wrote. */}
          <div className="ins-scores">
            <div className="meta">
              <b>scored</b>
            </div>
            {scores.length > 0 ? (
              <ul className="ins-facts">
                {scores.map(([name, value], i) => {
                  const marker = String(value).trim().toUpperCase();
                  // Only C and I, Inspect's own correct/incorrect markers,
                  // get a colour. Tinting a rubric's 7 would be this panel
                  // deciding what a good score is.
                  const tone =
                    marker === "C" ? "ok" : marker === "I" ? "no" : "";
                  return (
                    <li
                      key={name}
                      className={`ins-score ${tone}`}
                      // Staggered so several scorers read as a row arriving
                      // rather than a block appearing. Capped, so a sample
                      // with twenty scorers does not animate for a second.
                      style={{ animationDelay: `${Math.min(i, 8) * 28}ms` }}
                      title={
                        tone === "ok"
                          ? "C — Inspect's marker for a correct answer"
                          : tone === "no"
                            ? "I — Inspect's marker for an incorrect answer"
                            : `${name}, as the log recorded it`
                      }
                    >
                      <b>{name}</b>
                      <span className="mid">{String(value)}</span>
                    </li>
                  );
                })}
              </ul>
            ) : (
              // NOT a blank and not a 0. House rule: an unknown is not a
              // zero, and "this sample scored nothing" is a claim about the
              // sample that would be false.
              <p className="meta" style={{ margin: "3px 0 0" }}>
                this log records no score for this sample — which is not a
                score of zero, it is nobody having measured it
              </p>
            )}
            {unreadable.length > 0 && (
              // Counted rather than dropped, the same discipline the event
              // mapping keeps. A rubric whose value is a nested object used
              // to disappear and read as a sample with no rubric.
              <ul className="ins-kinds" style={{ marginTop: 6 }}>
                {unreadable.map(([name, why]) => (
                  <li key={name} className="meta">
                    <b className="mid">{name}</b> not read — {why}
                  </li>
                ))}
              </ul>
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
          {/* `out.mapping.means`, deliberately, and NOT `out.means`.
              `Imported.means()` is one sentence covering the mapping AND the
              scores AND the unreadable entries AND why this sample is on
              screen — which is right for the terminal and for an API caller
              with no panel, and wrong here, because this panel now renders
              every one of those clauses itself, above. Rendering both put
              each of them on screen twice: the no-score sentence, the
              NOT-SCORED-HERE list, and "this sample is marked as failed",
              which the banner at the top already says. The mapping's own
              sentence is the one clause nothing else here draws. */}
          <p className="meta ins-means">{out.mapping.means}</p>

          {pickable.length > 1 && (
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
                {pickable.map(({ id, epochs }) => (
                  <option
                    key={id}
                    value={id}
                    title={
                      epochs.length > 1
                        ? `${epochs.length} epochs of this sample are in the log; this reader opens the lowest`
                        : `sample ${id}`
                    }
                  >
                    {id}
                    {epochs.length > 1
                      ? ` (epoch ${epochs[0]} of ${epochs.length} listed)`
                      : ""}
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
