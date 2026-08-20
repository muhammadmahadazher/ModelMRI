import { useCallback, useEffect, useMemo, useState, useRef } from "react";
import { howLong } from "./measured";
import { VIEWER } from "./viewer";
import { useScanOnData } from "./useScanOnData";
import { CostBanner, StepTokens, TokenTable } from "./TokenLedger";
import ShareRun from "./ShareRun";
import InspectDrop from "./InspectDrop";
import PatternsAcross from "./PatternsAcross";
import RubricPanel from "./RubricPanel";
import JudgePanel from "./JudgePanel";
import {
  Adopted,
  adoptStep,
  clearTraces,
  errorText,
  getSessionState,
  getSessionTrace,
  getTrace,
  getTraces,
  SearchResult,
  searchTraces,
  SessionState,
  SessionTraceDoc,
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

interface Props {
  /**
   * How many generations the playground above has finished this session.
   * Every one of them is a recorded run now, so the panel has to refetch —
   * see the effect below for why nothing else here would notice.
   */
  runs?: number;
  /**
   * Fires after a step has been opened in the mechanistic panels. Those are
   * mounted by Playground, which asks the server what it can answer — so the
   * parent remounts it and the attention, lens, ablation and patching views
   * come up on the adopted generation.
   */
  onAdopted?: () => void;
}

/** Agent Mode: recorded runs -> lanes timeline -> step inspector. */
export default function AgentsPanel({ runs = 0, onAdopted }: Props) {
  const [list, setList] = useState<TraceSummary[] | null>(null);
  const [doc, setDoc] = useState<TraceDoc | null>(null);
  const [sel, setSel] = useState<TraceStep | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [clearing, setClearing] = useState(false);
  // The adopt result, and its refusal. Both belong to the selected step, so
  // both are cleared whenever the selection changes — a refusal left hanging
  // beside a different step reads as being about that one.
  const [adopted, setAdopted] = useState<Adopted | null>(null);
  const [adoptErr, setAdoptErr] = useState("");
  const [adopting, setAdopting] = useState(false);
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<SearchResult | null>(null);
  const [searchErr, setSearchErr] = useState("");
  // The run carried INSIDE the open `.mri`, which is a different source from
  // the store this panel lists. A bundle built around a failing step used to
  // open here to "0 recordings" with the run sitting in the file.
  const [carried, setCarried] = useState<SessionState["trace"] | null>(null);
  // Whether what is on screen came from the file rather than the store. It
  // changes what a refusal is allowed to say: a carried step is not adoptable
  // for a reason that has nothing to do with where it ran.
  const [fromSession, setFromSession] = useState(false);
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

  // Hoisted out of the mount effect so the generation effect below can call
  // the same fetch. StrictMode mounts twice in dev, so this is set on the way
  // IN as well as cleared on the way out — otherwise the second mount runs
  // with a flag the first mount's cleanup already turned off, and the panel
  // never renders a list at all.
  const live = useRef(true);
  useEffect(() => {
    live.current = true;
    return () => {
      live.current = false;
    };
  }, []);

  // A carried run belongs to the file that is open, so it must not outlive
  // it. Close the session — or open a different one — while its run is on
  // screen and the timeline used to stay exactly as it was, under a sentence
  // reading "the session file you have open", about a file that is not.
  useEffect(() => {
    if (!fromSession) return;
    if (!carried?.available) {
      setDoc(null);
      setSel(null);
      setFromSession(false);
      return;
    }
    // A DIFFERENT file, not merely a re-read of the same one. Compared by the
    // run's id rather than by object identity, because the state is refetched
    // on a timer and every poll produces a new object.
    if (doc && doc.id !== carried.id) {
      setDoc(null);
      setSel(null);
    }
    // `doc` is in the deps and this sets it, which is only stable because the
    // condition is false once it has run: the next pass sees `doc === null`.
  }, [carried, fromSession, doc]);

  const load = useCallback(() => {
    // Asked on the same schedule as the store list, because opening a `.mri`
    // is exactly as much of an arrival as recording a run is.
    void getSessionState()
      .then((st) => live.current && setCarried(st.trace ?? null))
      .catch(() => undefined);
    void getTraces()
      .then((l) => {
        if (!live.current) return;
        setList(l);
        // Only auto-open the newest when nothing is open, so a refresh
        // never yanks the reader off the trace they are reading.
        setDoc((current) => {
          if (current || !l.length) return current;
          void getTrace(l[0].id).then((d) => live.current && setDoc(d));
          return current;
        });
      })
      // A failed poll is not an empty store. Keeping the last list is the
      // difference between "nothing recorded" and "the server blinked".
      .catch(() => undefined);
  }, []);

  // GENERATING IN THIS TAB IS THE ONE ARRIVAL NOTHING ELSE HERE SEES.
  //
  // Every other refresh path assumes the run happened somewhere else: the
  // focus listener fires when you come back from the terminal you ran your
  // agent in, and the three-try poll is long finished by the time anyone has
  // loaded a model. Now that a generation in the playground above IS a
  // recorded run, it lands with the window already focused and the poll long
  // over — so without this the panel keeps saying it has nothing while the
  // run sits in the store, which is the exact dead end the empty state used
  // to be.
  useEffect(() => {
    if (runs > 0) load();
  }, [runs, load]);

  useEffect(() => {
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
      if (!live.current || tries >= 3) return;
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
      window.removeEventListener("focus", again);
      document.removeEventListener("visibilitychange", again);
      window.clearTimeout(timer);
    };
  }, [load]);

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

  // The empty branch is for "there is nothing to look at", and a `.mri`
  // carrying an agent run means there IS. Returning early on the store alone
  // is the same mistake the drop zone below already had to be fixed for: this
  // panel returns before the list renders, so anything mounted only in the
  // populated branch is invisible to exactly the reader who has no store
  // entries — here, somebody who was sent a bundle and opened it.
  const empties = !list || list.length === 0;
  // A DIFFERENT EMPTY IN THE VIEWER. The branch below tells a reader how to
  // fill this panel — generate above, or add three lines to their agent — and
  // neither is available to somebody reading a file on a machine with nothing
  // installed. The honest answer there is that this particular bundle carries
  // no run, which is an ordinary thing for a bundle to do.
  if (empties && !carried?.available && VIEWER) {
    return (
      <div className="panel">
        <div className="sect">
          <span className="dot d-agent" />
          <h2 className="h-agent">AGENTS — RECORDED RUNS</h2>
          <span className="rule" />
        </div>
        <div className="agents-empty">
          <p>
            <b>This file carries no agent run.</b> A `.mri` holds whatever was
            captured when it was written — attention, a logit lens, a patching
            trace, an agent run, or any combination — and this one was
            exported without one.
          </p>
          <p className="meta">
            A bundle that does carry a run opens here with its timeline, every
            step's input and output, and what the run cost in tokens. Nothing
            is installed to read it: the file is the whole of it.
          </p>
        </div>
      </div>
    );
  }
  if (empties && !carried?.available) {
    return (
      <div className="panel">
        <div className="sect">
          <span className="dot d-agent" />
          <h2 className="h-agent">AGENTS — RECORDED RUNS</h2>
          <span className="rule" />
        </div>
        <div className="agents-empty">
          {/* WHAT FILLS THIS PANEL, both ways, before what to type into it.
              This used to say the opposite — "not for the model loaded above.
              Nothing you run in this page appears here, and no model will
              fill it" — which was true of the panel and a dead end for the
              person reading it, who had come here after loading a model and
              generating. Generations made in this page are now recorded, so
              that sentence is false as well as unhelpful. Both sources get a
              line, in the order somebody meets them. */}
          <p>
            <b>A flight recorder for model runs.</b> Two things fill it:
            generations you make in the playground above, recorded with their
            prompt, output, timing and token count — and runs of your own agent
            code, with every LLM call, tool call and sub-agent nested as a
            timeline.
          </p>
          <p className="meta">
            Nothing yet. Generate something above and the run appears here. For
            your own agent, add three lines to it:
          </p>
          <pre className="agents-snippet">
{`from modelmri.record import trace, step

with trace("my-agent"):
    step("llm_call", name="plan", input=prompt, output=reply)
    step("tool_call", name="search", input=query, output=hits)`}
          </pre>
          <div className="hint">
            Or see what an agent run looks like without writing one:&nbsp;
            <b>uv run python examples/record_demo.py</b>
            &nbsp;— this panel picks it up on its own.
          </div>

          {/* HERE TOO, not only in the populated branch. This panel returns
              early when the list is empty, so mounting the drop zone below
              the list alone made it invisible to the one reader who needs it
              most: somebody whose only data is an eval log they already have.
              An empty flight recorder was a dead end for them. */}
          <InspectDrop
            onImported={(id) => {
              void getTraces().then(setList);
              setFromSession(false);
          void getTrace(id).then(setDoc);
            }}
          />
        </div>
      </div>
    );
  }

  // Bundled samples in the list, so the panel can name them and offer to
  // remove them rather than leaving a red "1 error" nobody recognises.
  const rows = list ?? [];
  const demos = rows.filter((t) => t.demo).length;

  async function search(text: string) {
    setQ(text);
    setSearchErr("");
    if (!text.trim()) {
      setHits(null);
      return;
    }
    try {
      setHits(await searchTraces(text));
    } catch (e) {
      // The parser's own sentence — it names the bad filter and the values it
      // accepts, which is more use than "search failed".
      setSearchErr(errorText(e));
      setHits(null);
    }
  }

  /** Open the run a hit belongs to, with that step selected. */
  async function openHit(traceId: string, stepId: string) {
    const d = await getTrace(traceId);
    setDoc(d);
    setFromSession(false);
    setSel(d.steps.find((s) => s.id === stepId) ?? null);
    setAdopted(null);
    setAdoptErr("");
  }

  /** Open the run the `.mri` carries, on the step it was built around. */
  async function openCarried() {
    const d = await getSessionTrace();
    if (!d.available) {
      // The file was closed between the list refresh and this click.
      setCarried(null);
      return;
    }
    setDoc(d);
    setFromSession(true);
    setAdopted(null);
    setAdoptErr("");
    // `step_ref` is the reason the bundle was sent. Landing on step one and
    // making the reader hunt for it wastes the one thing the sender said.
    setSel(d.step_ref ? (d.steps.find((x) => x.id === d.step_ref) ?? null) : null);
  }

  async function adopt(step: TraceStep) {
    if (!doc) return;
    setAdopting(true);
    setAdoptErr("");
    try {
      const result = await adoptStep(doc.id, step.id);
      setAdopted(result);
      // Tell the parent to remount the playground. Nothing is re-run — the
      // server is already holding this step's token ids as its current
      // generation, and Playground's restore path mounts the panels on
      // whatever the server says it can answer.
      onAdopted?.();
    } catch (e) {
      // The server's own sentence, unwrapped. Every refusal here already says
      // what would make it work — wrong model, weights not on this machine, a
      // tokenisation that no longer matches — and anything added in front of
      // those is the client guessing.
      setAdoptErr(errorText(e));
      setAdopted(null);
    } finally {
      setAdopting(false);
    }
  }

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
    ? Math.max(...doc.steps.map((s) => s.started_ms + (s.duration_ms ?? 0)), 1)
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

      {/* What this panel is, which nothing said — and which changed.
          It used to record only YOUR agent's steps, so this line's job was to
          say it had no connection to the model above. It now records that
          too, and a paragraph still denying it would be the worse error: the
          reader would look for their own generation and be told it is not
          here, while it sits in the list underneath. Both sources, one line,
          and the `this app` pill on the rows says which is which. */}
      {/* Both sources name something the viewer does not have: there is no
          playground above it to generate in, and `modelmri-record` is a
          package on a machine with ModelMRI installed. A reader who was sent
          a file has neither, and telling them how to fill a panel that is
          already full is the same dead end the empty state was fixed for. */}
      <p className="agents-what meta">
        {VIEWER ? (
          <>
            The agent run this file carries, with every step's input and
            output and what the run cost in tokens. Nothing is installed to
            read it — the file is the whole of it.
          </>
        ) : (
          <>
            A flight recorder for runs: each generation you make above lands
            here as one <b>llm_call</b>, and a program of your own calling{" "}
            <b>modelmri-record</b> lands here with its tool calls and
            sub-agents nested underneath it.
          </>
        )}
      </p>

      {/* Search is over STEPS, not runs — what somebody is looking for is the
          tool call that failed, not the hour it happened in. Across the
          STORE, though, which the viewer does not have: one run needs no
          search across it, and the box would only ever refuse. */}
      {!VIEWER && (
        <>
      <div className="row trace-search" style={{ marginBottom: 10 }}>
        <input
          className="sm"
          value={q}
          placeholder="search every step — try  error:true  or  kind:tool_call  or  duration>2000"
          onChange={(e) => void search(e.target.value)}
          spellCheck={false}
        />
        {hits && (
          <span className="meta">
            {hits.results.length} step{hits.results.length === 1 ? "" : "s"} ·{" "}
            {/* Which engine answered. A feature that quietly becomes a
                different feature is worse than one that says it degraded. */}
            <code>{hits.engine}</code>
          </span>
        )}
      </div>
      {searchErr && <div className="hint err">{searchErr}</div>}
      {hits && (
        <div className="search-hits">
          {hits.results.length === 0 ? (
            <div className="hint">nothing matched. {hits.note}</div>
          ) : (
            <>
              <ol className="ranking-list">
                {hits.results.map((h) => (
                  <li key={h.step_id} className={h.error ? "err" : ""}>
                    <button
                      className="ghost sm"
                      onClick={() => void openHit(h.trace_id, h.step_id)}
                      title="Open this run with that step selected"
                    >
                      {h.kind} · {h.name || "unnamed"}
                    </button>
                    <span className="mid">
                      {h.trace_name}
                      {h.trace_started_at && (
                        <span className="meta"> · {h.trace_started_at.slice(0, 10)}</span>
                      )}
                    </span>
                    <span className="meta">
                      {h.duration_ms == null
                        ? "duration not recorded"
                        : `${h.duration_ms}ms`}
                      {h.error && " · FAILED"}
                      {h.truncated_by > 0 &&
                        ` · ${h.truncated_by.toLocaleString()} chars not stored`}
                    </span>
                  </li>
                ))}
              </ol>
              <div className="hint">{hits.note}</div>
            </>
          )}
        </div>
      )}
        </>
      )}

      <div className="row" style={{ marginBottom: 10 }}>
        <span className="meta">
          {/* `rows` is the STORE, and the viewer has none — the run it shows
              is the carried one. Counting rows there printed "0 runs in this
              file" directly above the row displaying the run, which is a
              count contradicting the thing beneath it. Introduced when the
              viewer stopped listing the carried run as a store row, and
              caught by reading the count rather than the change. */}
          {VIEWER
            ? `${carried?.available ? 1 : 0} run${carried?.available ? "" : "s"} in this file`
            : `${rows.length} recording${rows.length === 1 ? "" : "s"}`}
          {demos > 0 && ` · ${demos} bundled sample${demos === 1 ? "" : "s"}`}
        </span>
        <span className="spacer" />
        {/* Both buttons DELETE from the store. This page writes nothing, and
            the run on screen is inside the file — there is no copy of it here
            to remove. */}
        {!VIEWER && (
        <>
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
        </>
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

      {/* EVERY BLOCK BELOW NEEDS A STORE, A DISK OR WEIGHTS, and the
          standalone viewer has none of the three. They are not mounted there
          rather than mounted and refusing: a control that can only ever
          refuse teaches a reader that the feature is broken, which is the
          rule `Playground` states three times for the same situation.

          What the viewer keeps is everything that reads the file it was
          given — the run, its timeline, the step inspector and the token
          tables. That is the whole of what a bundle carries. */}
      {!VIEWER && (
        <>
      {/* Sited above the list, because what it produces IS a row in that
          list — an Inspect sample becomes an ordinary trace and everything
          below reads it without knowing where it came from. */}
      <InspectDrop
        onImported={(id) => {
          setSel(null);
          // The list, so the imported sample appears as a row; then the
          // trace itself, so the timeline is already showing it. `load` is
          // scoped inside the polling effect, so this calls the same two
          // clients directly rather than reaching into it.
          void getTraces().then(setList);
          setFromSession(false);
          void getTrace(id).then(setDoc);
        }}
      />

      <RubricPanel
        onPick={(id) => {
          setSel(null);
          setFromSession(false);
          void getTrace(id).then(setDoc);
        }}
      />

      {/* Directly beneath the deterministic scorer, because the pair is the
          argument. The rubric above asks exact predicates and never touches a
          model, so a match is a match. This one asks the loaded model — and
          reads its probability mass rather than sampling a label off it, which
          is only possible because the weights are on this machine. The step
          selected on the timeline is offered as the text to judge, so what was
          scored is something you can read. */}
      <JudgePanel step={sel} />

      {/* ACROSS runs, above the list of runs. The rubric block scores each run
          against rules you wrote; this counts what the runs have in common
          without being told what to look for, and both answer questions about
          the whole store rather than about whichever run is open below.

          The agent names come from the list this panel already holds, so the
          filter offers the names that exist rather than a box that can only be
          got wrong — and its counts are the same counts the rows show. */}
      <PatternsAcross
        agents={groups.map((g) => ({ name: g.name, runs: g.runs.length }))}
        onPick={(id) => {
          setSel(null);
          setFromSession(false);
          void getTrace(id).then(setDoc);
        }}
      />
        </>
      )}

      {/* THE OTHER SOURCE. The list below is what this machine recorded or
          imported; this is what arrived inside the `.mri` somebody sent. It
          is deliberately outside the list rather than merged into it: a
          carried run is not in the store, cannot be deleted from it, and
          disappears when the file is closed, so showing it as an ordinary row
          would promise three things that are not true of it. */}
      {carried?.available && (
        <button
          className={`trace-row carried ${fromSession ? "sel" : ""}`}
          onClick={() => void openCarried()}
          title="the agent run inside the session file that is open"
        >
          <span className="tname">
            {carried.name || "agent run"}
            <span className="pill tiny" title="carried by the open .mri, not stored on this machine">
              in this file
            </span>
          </span>
          <span className="meta">
            {carried.n_steps} step{carried.n_steps === 1 ? "" : "s"}
            {/* Only when they differ. "4 of 4" is noise; "4 of 9" is the
                whole point — the sender's run was longer than what fits. */}
            {carried.n_steps_total > carried.n_steps &&
              ` of ${carried.n_steps_total}`}
            {carried.step_ref && " · opens on the step it was sent for"}
          </span>
        </button>
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
                    setFromSession(false);
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
                    {i === 0 && t.source === "app" && (
                      // A generation from the playground above, whose group
                      // name is a model id. Unlabelled, it sits in a list of
                      // agent names looking like an agent someone wrote and
                      // happened to name after a model.
                      <span
                        className="pill tiny"
                        title="a generation you ran in this page, not an agent you instrumented"
                      >
                        this app
                      </span>
                    )}
                  </span>
                  <span className="tmeta">
                    {/* A generation is exactly one step, so "1 steps" is now
                        the common case rather than a rarity. */}
                    {t.n_steps} step{t.n_steps === 1 ? "" : "s"}
                    {/* NOT A TOTAL when a step carries no duration. The store
                        sums `started_ms + COALESCE(duration_ms, 0)`, so an
                        untimed step contributes nothing — measured on four
                        runs, a single untimed step rendered "0.0s", four
                        untimed steps rendered the last one's start, and a run
                        whose final step was untimed lost that step entirely.
                        A floor printed as a total is the same defect as an
                        unknown printed as a zero. */}
                    {t.n_timed === 0 ? (
                      <span title="no step in this run recorded a duration">
                        {" "}
                        · not timed
                      </span>
                    ) : (
                      <>
                        {" "}
                        ·{" "}
                        {t.n_timed < t.n_steps && (
                          <span title={`${t.n_steps - t.n_timed} of ${t.n_steps} steps recorded no duration, so this is a floor`}>
                            ≥
                          </span>
                        )}
                        {howLong(t.total_ms)}
                      </>
                    )}
                    {/* Pluralised, like the step count two lines up. "3
                        error" beside "3 steps" reads as a truncated word
                        rather than a count. */}
                    {t.n_errors > 0 && (
                      <em>
                        {" "}
                        · {t.n_errors} error{t.n_errors === 1 ? "" : "s"}
                      </em>
                    )}
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
          {/* Beside the timeline, not only on the row that opened it. A
              reader looking at four blocks is looking at a run that had nine
              steps, and the cap is invisible from here — the timeline draws
              what it was given and looks complete either way. */}
          {fromSession && (doc as SessionTraceDoc).truncated > 0 && (
            <div className="hint">
              This is {doc.steps.length} of{" "}
              {(doc as SessionTraceDoc).n_steps_total} steps.{" "}
              {(doc as SessionTraceDoc).truncated} did not fit in the session
              file, so the timeline below is a section of the run rather than
              the run.
            </div>
          )}
          <div className="timeline" style={{ height: nLanes * 36 + 8 }}>
            {lanes.map(({ step, lane }) => (
              <button
                key={step.id}
                // `no-dur` is not decoration. A step with no recorded
                // duration used to be drawn at the 0.8% minimum width, which
                // is exactly what a genuinely instant call looks like — so
                // the bar said "this took no time" about a fact nobody wrote
                // down. The detail panel beside it already refuses to do
                // that, in its own words: "0ms reads as an instant call
                // rather than as a fact nobody wrote down". Two views of one
                // step disagreeing about what is known is the defect, not the
                // width.
                className={`tl-block ${step.error ? "err" : ""} ${
                  sel?.id === step.id ? "sel" : ""
                } ${step.duration_ms == null ? "no-dur" : ""}`}
                style={{
                  left: `${(step.started_ms / maxMs) * 100}%`,
                  width: `${Math.max(((step.duration_ms ?? 0) / maxMs) * 100, 0.8)}%`,
                  top: lane * 36 + 4,
                  background: KIND_COLOR[step.kind],
                }}
                title={
                  `${step.kind} · ${step.name} · ` +
                  (step.duration_ms == null
                    ? "duration not recorded"
                    : `${step.duration_ms}ms`)
                }
                onClick={() => {
                  setSel(step);
                  // Belongs to the step that produced it, not to the panel.
                  setAdopted(null);
                  setAdoptErr("");
                }}
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
          {/* The whole run. Tokens are the honest unit for an audience running
              local models, and they are free — so they are always here, while
              the cost line appears only if the reader brought their own
              prices. */}
          <div className="tok-run">
            <TokenTable rollup={doc.tokens} title="this run" />
            <CostBanner cost={doc.cost} />
          </div>
          {/* The share half. Sited under the run summary rather than beside
              the timeline: what leaves the machine is a decision about the
              whole run, not about whichever step happens to be selected. */}
          {/* Not for a carried run. `ShareRun` builds a `.mri` by asking the
              STORE for the trace, and a run read out of an opened file is not
              in the store — the preview call 404'd with "trace not found",
              which reads as the run being broken rather than as the button
              not applying to it. The reader is already holding the bundle
              this would produce. */}
          {fromSession || VIEWER ? (
            <div className="hint">
              This run arrived inside the session file you have open, so there
              is nothing here to package — the bundle already exists and you
              are reading it.
            </div>
          ) : (
            <ShareRun
              traceId={doc.id}
              selected={sel}
              ready={Boolean(sel?.adoptable)}
            />
          )}

          {sel && (
            <div className={`inspector ${sel.error ? "err" : ""}`}>
              <div className="row" style={{ marginBottom: 8 }}>
                <span className="lbl-chip" style={{ borderColor: KIND_COLOR[sel.kind], color: KIND_COLOR[sel.kind] }}>
                  {sel.kind}
                </span>
                <span className="meta">
                  {sel.name} · step {sel.seq} ·{" "}
                  {/* Null is not zero. A step recorded without a duration
                      used to render as "0ms", which reads as an instant
                      call rather than as a fact nobody wrote down. */}
                  {sel.duration_ms == null
                    ? "duration not recorded"
                    : `${sel.duration_ms}ms`}
                  {/* Was `tokens_in != null && \`${in}→${out} tok\``, which
                      rendered "100→null tok" when a provider reported one and
                      not the other. Each field is now drawn only if it is
                      actually there. */}
                  <StepTokens step={sel} />
                  {sel.error && " · FAILED"}
                </span>
              </div>
              {/* The clip marker is a marker, not characters the agent
                  produced. It used to render inside the payload as "… [+18412]"
                  where it read as part of the tool's own output — a truncated
                  result that looks complete is how you debug the wrong thing
                  for an hour. */}
              {sel.input && (
                <pre className="io">
                  <span className="io-l">IN</span>
                  {sel.input}
                  {(sel.truncated_in ?? 0) > 0 && (
                    <span className="clipped">
                      {"\n"}— {sel.truncated_in!.toLocaleString()} characters not
                      stored (payloads are capped at 20,000 so one runaway tool
                      output cannot fill your disk)
                    </span>
                  )}
                </pre>
              )}
              {sel.output && (
                <pre className="io">
                  <span className="io-l">OUT</span>
                  {sel.output}
                  {(sel.truncated_out ?? 0) > 0 && (
                    <span className="clipped">
                      {"\n"}— {sel.truncated_out!.toLocaleString()} characters
                      not stored (payloads are capped at 20,000 so one runaway
                      tool output cannot fill your disk)
                    </span>
                  )}
                </pre>
              )}
              {/* This step and everything beneath it. A subagent's own row
                  carries no tokens — they belong to its llm_call children —
                  so the subtree total is the only figure that answers "what
                  did this branch cost". */}
              {doc?.tokens_by_step?.[sel.id] && (
                <TokenTable
                  rollup={doc.tokens_by_step[sel.id]}
                  title="this step and everything under it"
                />
              )}
              {/* The join. Every other panel on this page reads whatever the
                  server is holding as its current generation, so a step that
                  ran on this machine can simply BECOME that — no re-running,
                  no substitute model, the recorded token ids themselves. */}
              {sel.adoptable ? (
                <div className="row adopt-row">
                  <button
                    className="sm"
                    onClick={() => void adopt(sel)}
                    disabled={adopting}
                    title={
                      "Point the attention, logit-lens, ablation and patching " +
                      "panels at the generation this step made"
                    }
                  >
                    {adopting ? "opening…" : "open in the panels"}
                  </button>
                  {typeof sel.meta?.model === "string" && (
                    <span className="meta">
                      recorded from <code>{sel.meta.model}</code>
                    </span>
                  )}
                </div>
              ) : (
                sel.kind === "llm_call" &&
                // A sentence, not a disabled button. A control that can only
                // ever refuse teaches the reader that the feature is broken
                // rather than that the weights are somewhere else.
                //
                // TWO different reasons, and saying the wrong one sends
                // somebody to fix the wrong thing. A step from the store did
                // not run here. A step from an opened `.mri` may well have run
                // on the sender's machine with weights and all — the bundle
                // simply does not carry token ids, by design, because they are
                // the model's reading of somebody's private prompt.
                // VIEWER counts as `fromSession`: everything it shows came
                // out of the open file, so the reason a step cannot be
                // adopted there is the file's shape and not where it ran.
                (fromSession || VIEWER ? (
                  <div className="hint">
                    A session file carries this run's shape — its steps, timing
                    and token counts — and not the token ids underneath them,
                    so there is nothing here for the panels above to read. Open
                    the run where it was recorded, or ask for a bundle exported
                    from the step itself.
                  </div>
                ) : (
                  <div className="hint">
                    This call did not run on this machine, so there are no
                    weights here to look inside. Steps recorded through{" "}
                    <code>instrument_transformers()</code> carry the token ids
                    that make the panels above work on them; a hosted API
                    returns text and nothing underneath it.
                  </div>
                ))
              )}
              {adopted && (
                <div className="hint ok">
                  <strong>
                    The panels above are now reading this step's generation.
                  </strong>{" "}
                  {adopted.n_prompt_tokens} prompt tokens +{" "}
                  {adopted.n_tokens - adopted.n_prompt_tokens} generated, from{" "}
                  <code>{adopted.model}</code>. {adopted.means}
                </div>
              )}
              {adoptErr && <div className="hint err">{adoptErr}</div>}
            </div>
          )}
          {!sel && <div className="hint">click any block to inspect the step</div>}
        </>
      )}
    </div>
  );
}
