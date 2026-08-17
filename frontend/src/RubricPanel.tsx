import { useState } from "react";
import { errorText, RubricReport, RubricRow, RubricRule, scoreRubric } from "./api";

/**
 * Score every recorded run against exact predicates.
 *
 * Every competitor's "evaluation" asks a language model whether a run went
 * well. That answer needs a calibration gate nobody ships, and it costs a key
 * and a round trip per run. A predicate costs neither and is reproducible next
 * Tuesday with the network off.
 *
 * **Nothing here is a verdict.** A rule says a run MATCHED. The name beside it
 * is the reader's own phrase — if they call a rule "too many retries", that
 * judgement is theirs and is shown as theirs. This panel never adds one.
 *
 * The distribution rule is drawn differently on purpose: below its minimum n
 * it is NOT ANSWERED, and says so. Rendering it as "0 matches" would read
 * identically to having looked and found nothing.
 */

/** The starting set. Chosen to be obviously exact rather than clever — each
 *  one is a thing the reader can check by eye on a single trace. */
const STARTERS: RubricRule[] = [
  { name: "had an error", kind: "has_error" },
  {
    name: "more than 5 tool calls",
    kind: "kind_count",
    step_kind: "tool_call",
    op: "gt",
    value: 5,
  },
  {
    name: "slowest tenth",
    kind: "slowest_percent",
    value: 10,
  },
];

/** When it ran, in the READER'S timezone.
 *
 *  The store writes UTC and the reader is not in UTC. Rendering the ISO
 *  string raw puts a hundred rows an hour off from the clock the reader is
 *  comparing them against, which is worse than no time at all — it looks
 *  precise. `Intl` is the browser's own answer and needs no table here.
 *
 *  Empty for a trace with no `started_at`, and the row draws nothing rather
 *  than "Invalid Date".
 */
function when(iso: string): string {
  if (!iso) return "";
  const t = new Date(iso);
  if (Number.isNaN(t.getTime())) return "";
  return t.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** A duration a reader can compare at a glance, or "" for one nobody has.
 *
 *  `null` is not zero. A run whose length the store never recorded did not
 *  finish instantly, and printing "0ms" for it would invent the fastest run
 *  in the list. */
function howLong(ms: number | null): string {
  if (ms === null) return "";
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  return `${m}m ${Math.round((ms % 60_000) / 1000)}s`;
}

/** What KIND of thing this run is, in the store's own terms.
 *
 *  The trace store keeps `demo` and `source` apart on purpose: scripted
 *  sample data, a generation made in the playground, and a run of the
 *  reader's own agent code are three different things that were landing in
 *  one undifferentiated list here. Empty for the third — the ordinary case
 *  needs no badge, and badging everything is the same as badging nothing.
 */
function kindOf(row: RubricRow): string {
  if (row.demo) return "sample";
  if (row.source === "app") return "playground";
  return "";
}

const KIND_LABEL: Record<string, string> = {
  has_error: "has an error step",
  kind_count: "count of one step kind",
  step_count: "total steps",
  tool_input_matches: "regex over tool input",
  any_input_matches: "regex over any input",
  output_matches: "regex over any output",
  duration_over: "wall clock above",
  slowest_percent: "slowest N% (needs 8+ runs)",
};

export default function RubricPanel({
  onPick,
}: {
  /** Clicking a matching run opens it on the timeline that ships today. */
  onPick: (traceId: string) => void;
}) {
  const [rules, setRules] = useState<RubricRule[]>(STARTERS);
  const [report, setReport] = useState<RubricReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [only, setOnly] = useState("");

  async function run() {
    setBusy(true);
    setErr("");
    try {
      setReport(await scoreRubric(rules));
    } catch (e) {
      setReport(null);
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  function edit(i: number, patch: Partial<RubricRule>) {
    setRules((r) => r.map((rule, j) => (j === i ? { ...rule, ...patch } : rule)));
  }

  const shown = report
    ? only
      ? report.rows.filter((r) => r.matched.includes(only))
      : report.rows
    : [];

  return (
    <div className="rub-panel">
      <div className="row">
        <span className="meta">
          <b>score every run against exact rules</b> — no model is asked, so a
          match is a match rather than a judgement
        </span>
      </div>

      <ul className="rub-rules">
        {rules.map((rule, i) => (
          <li key={i}>
            <input
              className="rub-name"
              value={rule.name}
              aria-label="rule name"
              onChange={(e) => edit(i, { name: e.target.value })}
            />
            <select
              value={rule.kind}
              aria-label="rule kind"
              onChange={(e) => edit(i, { kind: e.target.value })}
            >
              {Object.entries(KIND_LABEL).map(([k, label]) => (
                <option key={k} value={k}>
                  {label}
                </option>
              ))}
            </select>
            {rule.kind.endsWith("_matches") && (
              <input
                className="rub-pattern"
                value={rule.pattern ?? ""}
                placeholder="regex"
                aria-label="pattern"
                onChange={(e) => edit(i, { pattern: e.target.value })}
              />
            )}
            {rule.kind === "kind_count" && (
              <select
                value={rule.step_kind ?? "tool_call"}
                aria-label="step kind"
                onChange={(e) => edit(i, { step_kind: e.target.value })}
              >
                {["llm_call", "tool_call", "subagent", "mcp_call", "user_turn", "error"].map(
                  (k) => (
                    <option key={k} value={k}>
                      {k}
                    </option>
                  ),
                )}
              </select>
            )}
            {rule.kind !== "has_error" && !rule.kind.endsWith("_matches") && (
              <input
                className="rub-value"
                type="number"
                value={rule.value ?? 0}
                aria-label="value"
                onChange={(e) => edit(i, { value: Number(e.target.value) })}
              />
            )}
            <button
              className="ghost sm"
              onClick={() => setRules((r) => r.filter((_, j) => j !== i))}
              title="remove this rule"
            >
              ×
            </button>
          </li>
        ))}
      </ul>

      <div className="row">
        <button
          className="ghost sm"
          onClick={() =>
            setRules((r) => [...r, { name: `rule ${r.length + 1}`, kind: "has_error" }])
          }
        >
          add a rule
        </button>
        <button className="cta" onClick={() => void run()} disabled={busy || !rules.length}>
          {busy ? "scoring…" : "Score every run"}
        </button>
      </div>

      {err && <div className="hint err">{err}</div>}

      {report && (
        <div className="rub-out">
          <p className="meta">{report.means}</p>

          {/* The cap, if one applied. `slowest_percent` is a claim ABOUT the
              set it was measured on, so which set that was has to be here. */}
          {report.truncated > 0 && (
            <p className="meta rub-cap">
              Scored the {report.n_traces} newest of {report.n_traces_available}{" "}
              recorded runs.
            </p>
          )}

          {/* NOT ANSWERED, drawn apart from the counts. A distribution rule
              below its minimum n has no answer, and rendering it as a zero
              would read identically to having looked. */}
          {Object.keys(report.skipped).length > 0 && (
            <div className="rub-skipped">
              {Object.entries(report.skipped).map(([name, why]) => (
                <div key={name} className="meta">
                  <b>{name}</b> — not evaluated. {why}
                </div>
              ))}
            </div>
          )}

          <div className="row rub-counts">
            <button
              className={`rub-chip ${only === "" ? "on" : ""}`}
              onClick={() => setOnly("")}
            >
              all {report.rows.length}
            </button>
            {report.rules
              .filter((r) => !(r.name in report.skipped))
              .map((r) => (
                <button
                  key={r.name}
                  className={`rub-chip ${only === r.name ? "on" : ""}`}
                  onClick={() => setOnly(only === r.name ? "" : r.name)}
                  title={r.means}
                >
                  {r.name} · {report.counts[r.name] ?? 0}
                </button>
              ))}
          </div>

          {/* Each row has to be one RUN, not one name. Runs share names — a
              hundred playground generations are all called after the model,
              and a failed one with no model loaded is called "generation" —
              so without the time, the length and what kind of run it was, a
              long list is a wall of identical rows with a rule stuck to it.
              Every field here is the store's own; none is computed twice. */}
          <ol className="rub-rows">
            {shown.map((row) => {
              const kind = kindOf(row);
              const at = when(row.started_at);
              const took = howLong(row.total_ms);
              return (
                <li key={row.trace_id}>
                  <button className="rub-row" onClick={() => onPick(row.trace_id)}>
                    <span className="rub-row-head">
                      <span className="mid rub-row-name">{row.name}</span>
                      {kind && <span className="rub-kind">{kind}</span>}
                    </span>
                    <span className="meta rub-row-facts">
                      {at && <span>{at}</span>}
                      {/* Absent rather than "0ms" when the store has no
                          duration — see `howLong`. */}
                      {took && <span>{took}</span>}
                      <span>
                        {row.n_steps} step{row.n_steps === 1 ? "" : "s"}
                      </span>
                      {row.n_errors > 0 && (
                        <span className="rub-row-err">
                          {row.n_errors} error{row.n_errors === 1 ? "" : "s"}
                        </span>
                      )}
                    </span>
                    <span className="meta rub-row-hits">
                      {row.matched.length
                        ? row.matched.join(" · ")
                        : "no rule matched"}
                    </span>
                  </button>
                </li>
              );
            })}
          </ol>
          {!shown.length && (
            <p className="meta">No run matched {only ? `“${only}”` : "any rule"}.</p>
          )}
        </div>
      )}
    </div>
  );
}
