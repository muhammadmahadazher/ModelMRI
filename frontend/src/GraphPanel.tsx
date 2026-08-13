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
  if (!p?.measured_by) return null;

  const s = g.summary ?? {};
  const edges = g.edges ?? [];
  const peak = Math.max(...edges.map((e) => Math.abs(e.weight)), Number.MIN_VALUE);

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
          <b>{(g.n_nodes ?? 0).toLocaleString()}</b>
          <span className="meta">nodes</span>
        </span>
        <span className="gguf-stat">
          <b>{(s.nonzero_edges ?? 0).toLocaleString()}</b>
          <span className="meta">
            non-zero of {(s.possible_edges ?? 0).toLocaleString()}
          </span>
        </span>
        {s.density != null && (
          <span className="gguf-stat">
            <b>{s.density.toExponential(2)}</b>
            <span className="meta">density</span>
          </span>
        )}
        {s.max_abs_weight != null && (
          <span className="gguf-stat gguf-bpw">
            <b>{s.max_abs_weight.toFixed(4)}</b>
            <span className="meta">strongest edge</span>
          </span>
        )}
      </div>

      {g.prompt && (
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

      {(g.notes ?? []).map((n) => (
        <div className="hint" key={n}>
          {n}
        </div>
      ))}
      {s.means && <div className="hint">{s.means}</div>}
    </section>
  );
}
