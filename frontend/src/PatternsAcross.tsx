import { useState } from "react";
import {
  PatternsAcrossRuns,
  RecurringFinding,
  errorText,
  patternsAcross,
} from "./api";

/**
 * The same structural finding, counted over many recorded runs.
 *
 * WHY IT IS HERE AND NOT IN THE TIMELINE
 *
 * `/api/patterns` answers one run: this tool call ran four times with the
 * same input, these three failures were consecutive, this five-step sequence
 * repeated. Every one of those is an anecdote. A pattern that shows up in 12
 * of 19 runs of the SAME agent is a different kind of claim from one seen
 * once — and it is the claim somebody actually wants before they go and
 * change their prompt.
 *
 * `/api/patterns/across` has answered that since it was written and nothing
 * could call it.
 *
 * THREE THINGS THIS RENDERS THAT ARE EASY TO DROP
 *
 * **The cap.** The server reads at most `limit` runs, NEWEST FIRST, and
 * reports how many it left behind. "12 of 19" quietly meaning "12 of the 19
 * newest" is a different number, so `truncated` is on screen whenever it is
 * not zero.
 *
 * **Runs, not occurrences.** A finding that happens twice in one run counts
 * as one run. Both figures are printed because they answer different
 * questions — "how reliably" and "how much".
 *
 * **Nothing is a verdict.** The server writes each finding's own sentence and
 * it ends with what it refuses to claim: whether a repeat is a loop worth
 * fixing or a page-by-page walk of an API is not something a count can tell
 * you. That sentence is rendered whole.
 *
 * IT IS AN ACTION, NOT A LOAD
 *
 * Answering means opening every recorded run and analysing its steps. That is
 * not something a panel should do because you scrolled past it.
 */

/** How many runs to read. A REQUEST parameter, so it cannot come from a
 *  response — there is nothing to have asked yet. 500 is the server's own
 *  ceiling (`min(limit, 500)`), so nothing here can be silently clamped. */
const LIMITS = [10, 50, 200, 500];
const DEFAULT_LIMIT = 50;

/** Trace ids printed before the rest are counted rather than listed. */
const IDS_SHOWN = 6;

function Finding({
  f,
  onPick,
}: {
  f: RecurringFinding;
  onPick?: (traceId: string) => void;
}) {
  const shown = f.trace_ids.slice(0, IDS_SHOWN);
  const extra = f.trace_ids.length - shown.length;
  return (
    <li className="pat-item">
      <div className="pat-item-head">
        <span className="lbl-chip pat-kind">{f.kind}</span>
        <span className="mid pat-label">{f.label}</span>
        <span className="spacer" />
        <span className="meta pat-count">
          {/* Two numbers, because they answer different questions. */}
          {f.n_runs} of {f.of_runs} run(s) · {f.total_count} occurrence(s)
        </span>
      </div>
      {/* The server's sentence, whole. It ends by naming what it will not
          conclude from the count, which is the half a summary always cuts. */}
      <div className="hint">{f.means}</div>
      <div className="pat-traces">
        <span className="meta">seen in</span>
        {shown.map((id) => (
          <button
            key={id}
            className="ghost sm mid"
            title="Open this run in the timeline above"
            disabled={!onPick}
            onClick={() => onPick?.(id)}
          >
            {id.slice(0, 8)}
          </button>
        ))}
        {/* The cap on the LIST, said rather than left as a short row. */}
        {extra > 0 && (
          <span className="meta">+ {extra} more run(s) not listed</span>
        )}
        {f.signature && (
          <span className="meta pat-sig" title="What the grouping was done on: the input hash for a repeat, the name for a storm, the sequence for a cycle">
            signature {f.signature}
          </span>
        )}
      </div>
    </li>
  );
}

export default function PatternsAcross({
  agents,
  onPick,
}: {
  /** Every trace name on the page and how many runs each has, taken from the
   *  list this panel already fetched. `name` narrows the count to one agent,
   *  and the options are the names that actually exist rather than a typed
   *  box that can only be got wrong. */
  agents: { name: string; runs: number }[];
  /** Open one run in the timeline above. */
  onPick?: (traceId: string) => void;
}) {
  const [name, setName] = useState("");
  const [limit, setLimit] = useState(DEFAULT_LIMIT);
  const [out, setOut] = useState<PatternsAcrossRuns | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  async function count() {
    setBusy(true);
    setErr("");
    try {
      setOut(await patternsAcross(name, limit));
    } catch (e) {
      setErr(errorText(e));
      setOut(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="pat-across">
      <div className="row pat-controls">
        <span className="meta">
          <b>the same finding, counted over runs</b> — a pattern in one run is
          an anecdote
        </span>
        <span className="spacer" />
        <select
          className="sm"
          value={name}
          disabled={busy}
          aria-label="which agent to count over"
          onChange={(e) => {
            setName(e.target.value);
            // The answer belongs to the set it was counted over. Leaving the
            // old one on screen under a new selection is how "12 of 19"
            // becomes a claim about the wrong nineteen runs.
            setOut(null);
            setErr("");
          }}
        >
          <option value="">every recorded run</option>
          {agents.map((a) => (
            <option key={a.name} value={a.name}>
              {a.name} ({a.runs})
            </option>
          ))}
        </select>
        <select
          className="sm"
          value={limit}
          disabled={busy}
          aria-label="how many runs to read"
          title="How many runs to open and analyse, newest first. The server reports how many it left behind."
          onChange={(e) => {
            setLimit(Number(e.target.value));
            setOut(null);
            setErr("");
          }}
        >
          {LIMITS.map((n) => (
            <option key={n} value={n}>
              read {n} runs
            </option>
          ))}
        </select>
        <button className="ghost sm" onClick={() => void count()} disabled={busy}>
          {busy ? "counting…" : out ? "count again" : "count across runs"}
        </button>
      </div>

      {err && <div className="hint err">{err}</div>}

      {out && (
        <div className="pat-result">
          <span className="meta pat-scope">
            {/* The set, echoed from the response rather than from the select —
                so what is on screen is what was actually counted. */}
            {out.name ? (
              <>
                runs named <code>{out.name}</code>
              </>
            ) : (
              <>every recorded run</>
            )}{" "}
            · {out.n_runs} run(s) read of {out.n_runs_available} available
          </span>

          {/* The cap, whenever it bit. Without this line "3 of 3 runs" from a
              machine holding 90 of them reads as a property of the agent. */}
          {out.truncated > 0 && (
            <div className="hint warn">
              {out.truncated} run(s) were NOT read. Every count below is out of
              the {out.n_runs} newest, not out of all {out.n_runs_available} —
              raise the number of runs to read, or narrow to one agent.
            </div>
          )}

          {out.findings.length === 0 ? (
            <div className="hint">
              Nothing repeated across those {out.n_runs} run(s). That is a
              result, not a gap: it means no tool call ran twice on the same
              input, no name failed consecutively, and no sequence of steps
              came back — in the runs that were read.
            </div>
          ) : (
            <ul className="pat-list">
              {out.findings.map((f) => (
                <Finding
                  key={`${f.kind}:${f.signature || f.label}`}
                  f={f}
                  onPick={onPick}
                />
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
