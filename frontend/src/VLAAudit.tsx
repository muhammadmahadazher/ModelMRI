import { useState } from "react";
import { ApiError, AuditReport, errorText, vlaAudit } from "./api";

/**
 * Prove the loaded robot dataset is intact — or say exactly where it is not.
 *
 * NOTHING IS DOWNLOADED, no policy is loaded, no GPU is touched and lerobot is
 * not imported: this reads the files already on disk. That is worth a line in
 * the panel because every other button on this page costs a model.
 *
 * **There is deliberately no grade, and this panel must not invent one.** The
 * server ships three verdicts — proved, failed, could-not-be-run — and refuses
 * to collapse them into a letter, because a letter is a summary of somebody
 * else's opinion about what matters in the reader's data. So each check is
 * drawn with its own verdict, its own sentence and its own numbers, and the
 * only summary on screen is the server's own `means`.
 *
 * The bug class it exists for is one this project shipped: a `.get(name, 0.0)`
 * that made 206 episodes decode from timestamp zero, so every episode drew the
 * same video while the state vector under it was correct. Nothing crashed.
 */

/** A number as itself, and a `null` as UNKNOWN.
 *
 *  `null` here is "the frame table could not be read", which is the whole
 *  reason somebody opens an audit. Rendering it as 0 would claim an empty
 *  dataset — a different answer, and a confident one.
 */
function count(v: number | null): string {
  return v === null ? "unknown" : v.toLocaleString();
}

function fmtNum(n: number): string {
  if (!Number.isFinite(n)) return String(n);
  if (Number.isInteger(n)) return n.toLocaleString();
  return String(Number(n.toFixed(6)));
}

/** One `measured` value, flattened far enough to read in a row.
 *
 *  Generic on purpose: the keys belong to each check, not to this panel, so a
 *  check added to the server appears here with its numbers rather than being
 *  silently dropped by a renderer that only knew the old ones.
 */
function compact(v: unknown): string {
  if (v === null || v === undefined) return "unknown";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return fmtNum(v);
  if (typeof v === "string") return v;
  if (Array.isArray(v)) return v.length ? v.map(compact).join(" / ") : "none";
  if (typeof v === "object") {
    const entries = Object.entries(v as Record<string, unknown>);
    if (!entries.length) return "none";
    return entries.map(([k, x]) => `${k} ${compact(x)}`).join(", ");
  }
  return String(v);
}

/** Every item of a list value, so a capped list can be seen as a list. */
function items(v: unknown): string[] {
  if (Array.isArray(v)) return v.map(compact);
  if (v && typeof v === "object") {
    return Object.entries(v as Record<string, unknown>).map(
      ([k, x]) => `${k} — ${compact(x)}`,
    );
  }
  return [];
}

export default function VLAAudit() {
  const [report, setReport] = useState<AuditReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [refusal, setRefusal] = useState("");
  const [err, setErr] = useState("");

  async function run() {
    setBusy(true);
    setErr("");
    setRefusal("");
    setReport(null);
    try {
      setReport(await vlaAudit());
    } catch (e) {
      // A refusal here names a dataset this machine cannot prove anything
      // about — a missing reader dependency, a parquet that will not open —
      // and that is an answer about the data, not a broken button.
      if (e instanceof ApiError && e.status === 409) setRefusal(errorText(e));
      else setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="vla-audit">
      <div className="row vla-opts">
        <span className="meta">
          <b>is this dataset intact?</b> — nothing is downloaded, no policy is
          loaded and no GPU is touched; this reads the files already on disk
        </span>
      </div>

      <div className="row" style={{ margin: "8px 0" }}>
        <button className="cta" onClick={() => void run()} disabled={busy}>
          {busy ? "Reading the files…" : "Audit this dataset"}
        </button>
      </div>

      {refusal && (
        <div className="vla-refusal">
          <span className="judge-tag">refused, and that is the reading</span>
          <p>{refusal}</p>
        </div>
      )}
      {err && <div className="hint err">{err}</div>}

      {report && (
        <>
          <p className="meta">
            <b>{report.repo_id}</b> ·{" "}
            {/* `null` is UNKNOWN, never 0 — the two are different answers and
                the second is why an audit was opened. */}
            <span className={report.n_episodes === null ? "aud-unknown" : ""}>
              {count(report.n_episodes)} episodes
            </span>{" "}
            ·{" "}
            <span className={report.n_frames === null ? "aud-unknown" : ""}>
              {count(report.n_frames)} frames
            </span>{" "}
            · read in {report.seconds}s
          </p>

          <ol className="aud-checks">
            {report.checks.map((c) => {
              // The verdict word is the server's, and so is the class derived
              // from it: a verdict added later styles as the neutral base
              // rather than vanishing behind a hardcoded list of three.
              const kind = c.verdict.replace(/[^a-z]/gi, "").toLowerCase();
              const keys = Object.entries(c.measured);
              return (
                <li key={c.name} className={`aud-check aud-${kind}`}>
                  <div className="row aud-head">
                    <span className="aud-verdict">{c.verdict}</span>
                    <span className="mid aud-name">{c.name}</span>
                  </div>
                  <p className="meta aud-detail">{c.detail}</p>
                  {keys.length > 0 && (
                    <dl className="aud-measured">
                      {keys.map(([key, value]) => {
                        // A cap, stated. Every list in a `measured` blob is
                        // truncated by the server and carries its true length
                        // beside it as `n_<key>`; without this the reader sees
                        // eight gaps and believes there are eight.
                        const total = c.measured[`n_${key}`];
                        const list = items(value);
                        const capped =
                          Array.isArray(value) &&
                          typeof total === "number" &&
                          total > value.length;
                        return (
                          <div className="aud-row" key={key}>
                            <dt className="meta">{key.replace(/_/g, " ")}</dt>
                            <dd className={value === null ? "aud-unknown" : ""}>
                              {Array.isArray(value) || (value && typeof value === "object")
                                ? list.length
                                  ? list.map((entry, i) => (
                                      <span className="aud-item" key={i}>
                                        {entry}
                                      </span>
                                    ))
                                  : "none"
                                : compact(value)}
                              {capped && (
                                <span className="aud-cap">
                                  showing {(value as unknown[]).length} of{" "}
                                  {fmtNum(total as number)} — the rest were
                                  found and are not on screen
                                </span>
                              )}
                            </dd>
                          </div>
                        );
                      })}
                    </dl>
                  )}
                </li>
              );
            })}
          </ol>

          {/* VERBATIM, and it is the only summary here. It carries what passed,
              what failed, what could not be run, and the sentence saying there
              is no grade — which is the point of the whole route. */}
          <p className="meta vla-means">{report.means}</p>
        </>
      )}
    </div>
  );
}
