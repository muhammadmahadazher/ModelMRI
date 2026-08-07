import { useEffect, useMemo, useState } from "react";
import { getTrace, getTraces, TraceDoc, TraceStep, TraceSummary } from "./api";

const KIND_COLOR: Record<TraceStep["kind"], string> = {
  llm_call: "var(--color-agent)",
  tool_call: "var(--color-attn)",
  subagent: "var(--color-feat)",
  mcp_call: "var(--color-vla)",
  user_turn: "var(--color-mute)",
  error: "var(--color-pop)",
};

/** Agent Mode: recorded runs -> lanes timeline -> step inspector. */
export default function AgentsPanel() {
  const [list, setList] = useState<TraceSummary[] | null>(null);
  const [doc, setDoc] = useState<TraceDoc | null>(null);
  const [sel, setSel] = useState<TraceStep | null>(null);

  useEffect(() => {
    void getTraces().then((l) => {
      setList(l);
      if (l.length) void getTrace(l[0].id).then(setDoc);
    });
  }, []);

  const lanes = useMemo(() => {
    if (!doc) return [];
    const depth = new Map<string, number>();
    for (const s of doc.steps) {
      depth.set(s.id, s.parent_id ? (depth.get(s.parent_id) ?? 0) + 1 : 0);
    }
    return doc.steps.map((s) => ({ step: s, lane: depth.get(s.id) ?? 0 }));
  }, [doc]);

  if (!list || list.length === 0) {
    return (
      <div className="panel">
        <div className="sect">
          <span className="dot d-agent" />
          <h2 className="h-agent">AGENTS — RECORDED RUNS</h2>
          <span className="rule" />
        </div>
        <div className="hint">
          no traces yet — record one:&nbsp; <b>uv run python examples/record_demo.py</b>
          &nbsp; · or instrument your agent with <b>modelmri.record</b>
        </div>
      </div>
    );
  }

  const maxMs = doc
    ? Math.max(...doc.steps.map((s) => s.started_ms + s.duration_ms), 1)
    : 1;
  const nLanes = Math.max(...lanes.map((l) => l.lane), 0) + 1;

  return (
    <div className="panel agents">
      <div className="sect">
        <span className="dot d-agent" />
        <h2 className="h-agent">AGENTS — RECORDED RUNS</h2>
        <span className="rule" />
      </div>

      <div className="trace-list">
        {list.map((t) => (
          <button
            key={t.id}
            className={`trace-row ${doc?.id === t.id ? "sel" : ""}`}
            onClick={() => {
              setSel(null);
              void getTrace(t.id).then(setDoc);
            }}
          >
            <span className="tname">{t.name}</span>
            <span className="tmeta">
              {t.n_steps} steps · {(t.total_ms / 1000).toFixed(1)}s
              {t.n_errors > 0 && <em> · {t.n_errors} error</em>}
            </span>
          </button>
        ))}
      </div>

      {doc && (
        <>
          <div className="timeline" style={{ height: nLanes * 36 + 8 }}>
            {lanes.map(({ step, lane }) => (
              <button
                key={step.id}
                className={`tl-block ${step.error ? "err" : ""} ${sel?.id === step.id ? "sel" : ""}`}
                style={{
                  left: `${(step.started_ms / maxMs) * 100}%`,
                  width: `${Math.max((step.duration_ms / maxMs) * 100, 0.8)}%`,
                  top: lane * 36 + 4,
                  background: KIND_COLOR[step.kind],
                }}
                title={`${step.kind} · ${step.name}`}
                onClick={() => setSel(step)}
              />
            ))}
          </div>
          <div className="tl-legend meta">
            {(Object.keys(KIND_COLOR) as TraceStep["kind"][]).map((k) => (
              <span key={k}>
                <i style={{ background: KIND_COLOR[k] }} /> {k}
              </span>
            ))}
          </div>

          {sel && (
            <div className={`inspector ${sel.error ? "err" : ""}`}>
              <div className="row" style={{ marginBottom: 8 }}>
                <span className="lbl-chip" style={{ borderColor: KIND_COLOR[sel.kind], color: KIND_COLOR[sel.kind] }}>
                  {sel.kind}
                </span>
                <span className="meta">
                  {sel.name} · step {sel.seq} · {sel.duration_ms}ms
                  {sel.tokens_in != null && ` · ${sel.tokens_in}→${sel.tokens_out} tok`}
                  {sel.error && " · FAILED"}
                </span>
              </div>
              {sel.input && (
                <pre className="io">
                  <span className="io-l">IN</span>
                  {sel.input}
                </pre>
              )}
              {sel.output && (
                <pre className="io">
                  <span className="io-l">OUT</span>
                  {sel.output}
                </pre>
              )}
            </div>
          )}
          {!sel && <div className="hint">click any block to inspect the step</div>}
        </>
      )}
    </div>
  );
}
