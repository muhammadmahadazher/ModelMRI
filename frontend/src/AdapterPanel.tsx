import { useState } from "react";
import { measured, percent } from "./measured";
import { AdapterReport, errorText, readAdapter } from "./api";

/**
 * What a LoRA changes, before you merge it into anything.
 *
 * WHY A PATH RATHER THAN A FILE PICKER
 *
 * A browser file input hands back a `File`, not a path — the page never learns
 * where on disk the thing lives, by design. The reader needs a path because it
 * opens a safetensors header and multiplies two matrices out of the data
 * segment; uploading 400 MB through the page to do that would be absurd when
 * the file is already on the machine the server is running on. So this takes a
 * path, exactly as the custom-model panel does, and the route behind it
 * refuses any request that did not come from this machine.
 *
 * WHY THE GROUPS COME FIRST
 *
 * The individual modules are a long tail — 788 of them in the adapter this was
 * built against — and the question people arrive with is not "which single
 * matrix moved most" but "did it touch the part that reads my prompt". That is
 * a question about GROUPS, so groups lead and the top modules follow.
 *
 * WHAT THIS PANEL MUST NEVER IMPLY
 *
 * That a bigger bar means a bigger visible change. It does not: a large delta
 * in a layer the sampler barely exercises can matter less than a small one in
 * a layer it leans on. The server says so in `means` and that sentence is
 * rendered verbatim rather than summarised away.
 */
export default function AdapterPanel() {
  const [path, setPath] = useState("");
  const [report, setReport] = useState<AdapterReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  async function run() {
    const p = path.trim();
    if (!p) return;
    setBusy(true);
    setErr("");
    try {
      setReport(await readAdapter(p));
    } catch (e) {
      setErr(errorText(e));
      // `null` is the honest terminal state. Leaving the previous report up
      // would attribute it to the adapter that just failed to read.
      setReport(null);
    } finally {
      setBusy(false);
    }
  }

  /** The widest group, so the bars have a scale. Groups only — mixing the two
   *  scales would make every individual module look negligible beside a sum
   *  of 280 of them. */
  const widest = report
    ? report.groups.reduce((m, g) => Math.max(m, g.delta_norm), 0) || 1
    : 1;
  const topWidest = report
    ? report.top.reduce((m, t) => Math.max(m, t.delta_norm), 0) || 1
    : 1;

  return (
    <div className="isect adapter">
      <h3 className="mid isect-head">fine-tune — what a LoRA changes</h3>
      <p className="meta">
        Reads the adapter itself: which modules it targets, at what rank, and
        how far it moves each one. Nothing is loaded and nothing is merged.
      </p>

      <div className="row cand-manual">
        <input
          className="combo grow"
          placeholder="a path to a .safetensors adapter, or the folder holding it"
          value={path}
          onChange={(e) => setPath(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && void run()}
          spellCheck={false}
          aria-label="path to a LoRA adapter on this machine"
        />
        <button className="green" onClick={() => void run()} disabled={busy || !path.trim()}>
          {busy ? "Reading…" : "Read it"}
        </button>
      </div>

      {err && <p className="err">{err}</p>}

      {report && (
        <>
          <div className="row adapter-head">
            <span className="pill">
              {report.modules_total} module
              {report.modules_total === 1 ? "" : "s"}
            </span>
            {/* Ranks is a LIST because an adapter may mix them per module, and
                reporting one number would be picking which. */}
            <span className="pill">
              rank {report.ranks.length ? report.ranks.join(", ") : "unstated"}
            </span>
            {/* Not decoration: an unscaled norm is not comparable with a
                scaled one, and the two appear in the same table. */}
            {!report.all_scaled && <span className="pill warn">some unscaled</span>}
            {report.base_model && (
              <span className="meta">built for {report.base_model}</span>
            )}
          </div>

          <table className="adapter-groups">
            <caption className="meta">
              Where it lands. Cross-attention is where the prompt enters — an
              adapter that never touches it is not changing how words are read.
            </caption>
            <tbody>
              {report.groups.map((g) => (
                <tr key={`${g.component}:${g.role}`}>
                  <th scope="row" className="mid">
                    {g.component}
                  </th>
                  <td className="meta">{g.role}</td>
                  <td className="meta adapter-n">
                    {g.modules} module{g.modules === 1 ? "" : "s"}
                  </td>
                  <td className="adapter-track">
                    <span
                      className="adapter-bar"
                      style={{ width: `${Math.max((g.delta_norm / widest) * 100, 1)}%` }}
                    />
                  </td>
                  <td className="mid adapter-norm">{g.delta_norm.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <details className="adapter-modules">
            <summary className="meta">
              the {report.modules_listed} largest modules individually
            </summary>
            <ol className="adapter-list">
              {report.top.map((m) => (
                <li key={m.name}>
                  <span className="mid adapter-mod" title={m.name}>
                    {m.name}
                  </span>
                  <span className="meta">{m.role}</span>
                  <span className="adapter-track">
                    <span
                      className="adapter-bar"
                      style={{
                        width: `${Math.max((m.delta_norm / topWidest) * 100, 1)}%`,
                      }}
                    />
                  </span>
                  <span className="mid adapter-norm">
                    {measured(m.delta_norm, 3)}
                    {/* Shown only when it exists. Null means the base weights
                        were not resident, and a ratio against a stand-in
                        denominator is worse than no ratio. */}
                    {m.relative !== null && (
                      <em className="meta"> · {percent(m.relative, 1)} of base</em>
                    )}
                  </span>
                </li>
              ))}
            </ol>
          </details>

          {/* The server's own sentence, verbatim. It carries the caveat that a
              norm is a magnitude and not an effect, and a summary written here
              would be the place that caveat got dropped. */}
          <p className="meta adapter-means">{report.means}</p>
          {report.notes.map((n) => (
            <p key={n} className="meta icv-note">
              {n}
            </p>
          ))}
        </>
      )}
    </div>
  );
}
