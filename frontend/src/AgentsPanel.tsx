import { useEffect, useMemo, useState } from "react";
import { useScanOnData } from "./useScanOnData";
import {
  clearTraces,
  getTrace,
  getTraces,
  TraceDoc,
  TraceStep,
  TraceSummary,
} from "./api";

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
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [clearing, setClearing] = useState(false);
  // Above every conditional return. This panel returns early for its empty
  // and loading states, so a hook placed after that ran on some renders and
  // not others — React error #310, and the whole page blank. Hooks are not
  // allowed to be conditional, and an early return makes everything below it
  // conditional.
  const scanRef = useScanOnData(doc?.id ?? null);

  useEffect(() => {
    void getTraces().then((l) => {
      setList(l);
      if (l.length) void getTrace(l[0].id).then(setDoc);
    });
  }, []);

  /** One row per agent, not per run.
   *
   *  A recorder is something you leave switched on, so the list fills with
   *  repeats of whatever you run most: 16 of 21 traces here were the same
   *  `e2e-check-run`, which buried every other agent below the fold. Group by
   *  name, show the newest, and let the count expand.
   */
  const groups = useMemo(() => {
    if (!list) return [];
    const by = new Map<string, TraceSummary[]>();
    for (const t of list) {
      const runs = by.get(t.name);
      if (runs) runs.push(t);
      else by.set(t.name, [t]);
    }
    return [...by.entries()].map(([name, runs]) => ({ name, runs }));
  }, [list]);

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

  // Bundled samples in the list, so the panel can name them and offer to
  // remove them rather than leaving a red "1 error" nobody recognises.
  const demos = list.filter((t) => t.demo).length;

  async function wipe(keepDemo: boolean) {
    setClearing(true);
    try {
      await clearTraces(keepDemo);
      setList(await getTraces());
      setDoc(null);
    } catch {
      /* the list is refetched either way */
    } finally {
      setClearing(false);
    }
  }

  const maxMs = doc
    ? Math.max(...doc.steps.map((s) => s.started_ms + s.duration_ms), 1)
    : 1;
  const nLanes = Math.max(...lanes.map((l) => l.lane), 0) + 1;

  return (
    // A trace arriving from somebody's agent run IS new data — it is the only
    // panel here whose content can change without the reader doing anything,
    // which is exactly what the scan is for. Keyed on the opened document so
    // it fires when a run is loaded, not on every re-render of the list.
    <div className="panel agents" ref={scanRef}>
      <div className="sect">
        <span className="dot d-agent" />
        <h2 className="h-agent">AGENTS — RECORDED RUNS</h2>
        <span className="rule" />
      </div>

      {/* What this panel is, which nothing said.
          It records YOUR agent's steps — an external program you instrumented
          with `modelmri-record` — and has no connection to the model selected
          in the playground above. Reading it as "the calls that model made"
          is the obvious guess, and it is wrong, so say so once, here. */}
      <p className="agents-what meta">
        A flight recorder for agent runs: your program calls{" "}
        <b>modelmri-record</b> and its steps land here. Independent of the model
        loaded above — an agent usually calls a hosted API, not this process.
      </p>

      <div className="row" style={{ marginBottom: 10 }}>
        <span className="meta">
          {list.length} recording{list.length === 1 ? "" : "s"}
          {demos > 0 && ` · ${demos} bundled sample${demos === 1 ? "" : "s"}`}
        </span>
        <span className="spacer" />
        <button
          className="ghost sm"
          disabled={clearing}
          title="Removes runs you recorded, and keeps the bundled sample"
          onClick={() => void wipe(true)}
        >
          {clearing ? "Clearing…" : "Clear my runs"}
        </button>
        {/* The sample kept reappearing as "1 error" in a list of your work,
            and the only clear button deliberately spared it. If you have seen
            it, you should be able to be rid of it. */}
        {demos > 0 && (
          <button
            className="ghost sm"
            disabled={clearing}
            title="Removes the bundled sample too"
            onClick={() => void wipe(false)}
          >
            Remove sample
          </button>
        )}
      </div>

      {demos > 0 && (
        <p className="hint">
          The run marked <b>demo</b> is sample data shipped with ModelMRI
          (<code>examples/record_demo.py</code>). Its failing step is
          deliberate — a timeline needs an error to show what one looks like.
          It is not your agent failing.
        </p>
      )}

      <div className="trace-list">
        {groups.map(({ name, runs }) => {
          const open = expanded.has(name);
          const shown = open ? runs : runs.slice(0, 1);
          // Count errors among the runs this button is hiding, not across
          // the whole group: "15 more runs · 16 with errors" describes two
          // different sets in one sentence.
          const hiddenErrors = runs.slice(1).filter((r) => r.n_errors > 0).length;
          return (
            <div className="trace-group" key={name}>
              {shown.map((t, i) => (
                <button
                  key={t.id}
                  className={`trace-row ${doc?.id === t.id ? "sel" : ""} ${
                    i > 0 ? "repeat" : ""
                  }`}
                  onClick={() => {
                    setSel(null);
                    void getTrace(t.id).then(setDoc);
                  }}
                >
                  <span className="tname">
                    {i > 0 ? <span className="tdim">run {runs.length - i}</span> : name}
                    {i === 0 && t.demo && (
                      // Sample data shipped with the tool. Without this label
                      // its deliberately failed `git push` read as your agent
                      // failing.
                      <span className="pill tiny" title="shipped sample, not a run you recorded">
                        demo
                      </span>
                    )}
                  </span>
                  <span className="tmeta">
                    {t.n_steps} steps · {(t.total_ms / 1000).toFixed(1)}s
                    {t.n_errors > 0 && <em> · {t.n_errors} error</em>}
                  </span>
                </button>
              ))}
              {runs.length > 1 && (
                <button
                  className="trace-more"
                  aria-expanded={open}
                  onClick={() =>
                    setExpanded((prev) => {
                      const next = new Set(prev);
                      if (next.has(name)) next.delete(name);
                      else next.add(name);
                      return next;
                    })
                  }
                >
                  {open
                    ? "fewer"
                    : `${runs.length - 1} more run${runs.length === 2 ? "" : "s"}` +
                      (hiddenErrors ? ` · ${hiddenErrors} with errors` : "")}
                </button>
              )}
            </div>
          );
        })}
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
