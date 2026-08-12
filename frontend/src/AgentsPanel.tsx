import { useEffect, useMemo, useState, useRef } from "react";
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

  // THE EMPTY STATE USED TO BE A DEAD END. This ran once on mount with `[]`
  // deps and there was no other path to `setList` reachable from the empty
  // branch -- so the panel told you to go and run `record_demo.py`, you ran
  // it, the trace was delivered and stored correctly, and the panel kept
  // rendering "no traces yet" until the whole page was reloaded. Nothing in
  // the UI said so. It was instructing you to do the one thing whose result
  // it could not display.
  //
  // A recorder is something you leave switched on in another terminal, so the
  // list has to be able to arrive later. Polled while there is nothing to
  // show, and on regaining focus -- which is the actual gesture, since you
  // ran the command in a different window and came back.
  // Tracked in a ref so the poll can ask "is it still empty?" without the
  // effect depending on the value it sets.
  const empty = useRef(true);
  empty.current = !list || list.length === 0;

  useEffect(() => {
    let live = true;
    const load = () =>
      void getTraces()
        .then((l) => {
          if (!live) return;
          setList(l);
          // Only auto-open the newest when nothing is open, so a refresh
          // never yanks the reader off the trace they are reading.
          setDoc((current) => {
            if (current || !l.length) return current;
            void getTrace(l[0].id).then((d) => live && setDoc(d));
            return current;
          });
        })
        // A failed poll is not an empty store. Keeping the last list is the
        // difference between "nothing recorded" and "the server blinked".
        .catch(() => undefined);

    load();

    // FOCUS IS THE REAL SIGNAL. You ran `record_demo.py` in another terminal
    // and came back to this window; that is the gesture, and it costs one
    // request at the exact moment something might have changed.
    const again = () => {
      if (document.visibilityState === "visible") load();
    };
    window.addEventListener("focus", again);
    document.addEventListener("visibilitychange", again);

    // The backstop is for a reader watching both windows at once, and it has
    // to STOP -- soon. An unbounded poll on an empty panel is a request every
    // few seconds forever AND it means the page never reaches network idle,
    // which broke Playwright's `networkidle` wait outright.
    //
    // Six tries at a widening gap totalled ~31s of activity, which is longer
    // than the 30s `goto` budget, so the "bounded" version failed CI in
    // exactly the same way as the unbounded one. Three tries, ~6s: long
    // enough to catch a trace that lands just after the page does, short
    // enough that the network genuinely goes quiet. Focus does the rest, and
    // focus was always the real signal.
    let tries = 0;
    let timer = 0;
    const tick = () => {
      if (!live || tries >= 3) return;
      tries += 1;
      timer = window.setTimeout(() => {
        // `empty.current`, not `list`. Depending on `list` here is what made
        // the previous two attempts at bounding this fail: the effect both
        // READ and SET it, so every fetch produced a new array reference,
        // re-ran the effect, and reset the counter -- an unbounded poll
        // wearing a bound. The page then never reached network idle and
        // Playwright's `goto` timed out, twice, at two different intervals.
        if (empty.current) load();
        tick();
      }, 1000 * tries);
    };
    tick();

    return () => {
      live = false;
      window.removeEventListener("focus", again);
      document.removeEventListener("visibilitychange", again);
      window.clearTimeout(timer);
    };
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
        <div className="agents-empty">
          {/* WHAT THIS PANEL IS, before what to type into it.
              It said "no traces yet" and gave a command, which reads as an
              empty list of something you already have. It is not: this panel
              has nothing to do with the model loaded above, and loading a
              bigger one or a reasoning one will never fill it. It shows runs
              of YOUR OWN agent code, recorded by a separate library you add
              to that code. Somebody who does not know that reasonably
              concludes the feature is broken. */}
          <p>
            <b>This is a flight recorder for agents you write</b> — not for the
            model loaded above. Nothing you run in this page appears here, and
            no model will fill it.
          </p>
          <p className="meta">
            Add three lines to your own agent and every run it makes shows up
            here as a timeline: each LLM call, tool call and sub-agent, nested,
            with inputs, outputs, timings and failures.
          </p>
          <pre className="agents-snippet">
{`from modelmri.record import trace, step

with trace("my-agent"):
    step("llm_call", name="plan", input=prompt, output=reply)
    step("tool_call", name="search", input=query, output=hits)`}
          </pre>
          <div className="hint">
            Try it without writing anything:&nbsp;
            <b>uv run python examples/record_demo.py</b>
            &nbsp;— this panel picks it up on its own.
          </div>
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
