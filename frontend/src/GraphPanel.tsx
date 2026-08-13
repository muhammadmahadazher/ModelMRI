import { useEffect, useState } from "react";
import { GraphView, getGraph } from "./api";

/** An attribution graph somebody else computed.
 *
 *  The provenance banner is the feature, not chrome. Every other panel on this
 *  page shows something ModelMRI measured; this one does not, and a reader who
 *  cannot tell the difference has been misled by the page rather than informed
 *  by it. So the banner is the first thing drawn, it names the file, the tool
 *  and the model, and it carries the sentence the reader sends — the component
 *  never composes its own disclaimer, because a disclaimer assembled in the UI
 *  is one a future refactor can drop.
 *
 *  Both the server and the viewer refuse to report a graph whose provenance is
 *  missing, so `available` implies it is here. The guard below is the third
 *  copy on purpose: this is the one running in the recipient's browser.
 */
export default function GraphPanel() {
  const [g, setG] = useState<GraphView | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    let live = true;
    getGraph()
      .then((r) => live && setG(r))
      .catch((e) => live && setErr(String(e?.message || e)));
    return () => {
      live = false;
    };
  }, []);

  if (err) return null; // a missing endpoint is not this panel's problem
  if (!g || !g.available) return null;
  const p = g.provenance;
  // Non-empty string. See the shim in viewer.ts: a boolean passes truthiness
  // and renders as nothing, which is a graph shown with no disclaimer.
  if (typeof p?.measured_by !== "string" || !p.measured_by.trim()) return null;

  const s = g.summary ?? {};
  // Defensive by type, not by presence. This component renders a file a
  // stranger forwarded, `frontend/src` has no error boundary, and every one of
  // these was a `!= null` check that a string or an object walks straight
  // through into `.toFixed` / `.toExponential` / `.map` — a TypeError there
  // unmounts the whole viewer and the reader sees a white screen with no
  // explanation. Python validates its side; this is the side that must not
  // trust Python having run.
  const num = (v: unknown): number | null =>
    typeof v === "number" && Number.isFinite(v) ? v : null;
  const density = num(s.density);
  const peakWeight = num(s.max_abs_weight);
  const nonzero = num(s.nonzero_edges) ?? 0;
  const possible = num(s.possible_edges) ?? 0;
  const nodes = num(g.n_nodes) ?? 0;
  const notes = Array.isArray(g.notes) ? g.notes.filter((n) => typeof n === "string") : [];

  const edges = (Array.isArray(g.edges) ? g.edges : []).filter(
    (e) =>
      e &&
      typeof e.source === "number" &&
      typeof e.target === "number" &&
      num(e.weight) !== null,
  );
  // `reduce`, not `Math.max(...spread)`: 50,000 arguments is within a hair of
  // the engine's limit and throws RangeError rather than returning a number.
  const peak = edges.reduce((m, e) => Math.max(m, Math.abs(e.weight)), Number.MIN_VALUE);

  return (
    <section className="panel">
      <div className="panel-head">
        <span className="dot d-graph" />
        <h2>ATTRIBUTION GRAPH — SOMEBODY ELSE'S MEASUREMENT</h2>
        <span className="rule" />
      </div>

      {/* First, before any number. */}
      <div className="graph-provenance">
        <div className="row">
          <span className="pill warn">not measured here</span>
          <span className="meta">
            <code>{p.file}</code> · produced by <strong>{p.producer}</strong>
            {p.model ? (
              <>
                {" "}
                on <code>{p.model}</code>
              </>
            ) : (
              " · the file does not name a model"
            )}
            {p.scan ? (
              <>
                {" "}
                · transcoders <code>{p.scan}</code>
              </>
            ) : null}
          </span>
        </div>
        {/* Rendered from the payload, never composed here. */}
        <div className="hint">{p.measured_by}</div>
      </div>

      <div className="gguf-head">
        <span className="gguf-stat">
          <b>{nodes.toLocaleString()}</b>
          <span className="meta">nodes</span>
        </span>
        <span className="gguf-stat">
          <b>{nonzero.toLocaleString()}</b>
          <span className="meta">non-zero of {possible.toLocaleString()}</span>
        </span>
        {density !== null && (
          <span className="gguf-stat">
            <b>{density.toExponential(2)}</b>
            <span className="meta">density</span>
          </span>
        )}
        {peakWeight !== null && (
          <span className="gguf-stat gguf-bpw">
            <b>{peakWeight.toFixed(4)}</b>
            <span className="meta">strongest edge</span>
          </span>
        )}
      </div>

      {typeof g.prompt === "string" && g.prompt && (
        <div className="hint">
          computed on the prompt <code>{g.prompt}</code>
        </div>
      )}

      {edges.length > 0 && (
        <>
          <div className="meta cand-head">
            strongest {edges.length.toLocaleString()} edges
            {s.truncated ? " — pruned, this is not the whole graph" : ""}
          </div>
          <div className="graph-edges">
            {edges.slice(0, 60).map((e) => (
              <div className="graph-edge" key={`${e.source}-${e.target}`}>
                <span className="graph-node">{e.source}</span>
                <span className="graph-arrow" aria-hidden="true">
                  →
                </span>
                <span className="graph-node">{e.target}</span>
                {/* Signed: an edge that suppresses is not a weak edge that
                    promotes, and a bar drawn from |w| alone loses that. */}
                <span className="graph-bar">
                  <span
                    className={e.weight < 0 ? "neg" : "pos"}
                    style={{
                      ["--w" as string]: `${(Math.abs(e.weight) / peak) * 100}%`,
                    }}
                  />
                </span>
                <span className="graph-w">{e.weight.toFixed(4)}</span>
              </div>
            ))}
          </div>
          {edges.length > 60 && (
            <div className="hint">
              showing the strongest 60 of {edges.length.toLocaleString()} carried
              in this file.
            </div>
          )}
        </>
      )}

      {notes.map((n) => (
        <div className="hint" key={n}>
          {n}
        </div>
      ))}
      {typeof s.means === "string" && <div className="hint">{s.means}</div>}
    </section>
  );
}
