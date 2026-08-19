import { useCallback, useEffect, useState } from "react";
import { useScanOnData } from "./useScanOnData";
import {
  ResumePlan,
  SavedSweep,
  SweepList,
  errorText,
  savedSweeps,
  sweepResumePlan,
} from "./api";

/**
 * SWEEPS — every sweep saved on this machine, and how far each one got.
 *
 * WHY THIS PANEL EXISTS
 *
 * "A number measured once is a sample, not a property" is the line this
 * project opens with, and `modelmri sweep` is the loop that makes it true:
 * the same measurement over a set of prompts, reported as median, IQR and n
 * rather than as a number. It has saved every run it made since the day it
 * was written — and nothing ever read one back. A saved sweep was write-only:
 * sitting in the database, findable with `sqlite3`, unreachable from the tool
 * that wrote it. The thesis of the project was the one thing you could not
 * see.
 *
 * TWO THINGS THIS IS CAREFUL ABOUT
 *
 * **The cap.** `/api/sweeps` takes a `limit` and answers with at most that
 * many rows — and it carries no total. So a list of exactly `limit` rows is
 * indistinguishable from a complete one, and its own `means` sentence counts
 * the rows it RETURNED. Both facts are on screen, because a list that
 * silently shortens reads as complete.
 *
 * **`blocked`.** A resume is not merely expensive when the prompt set has
 * been edited or a different model is loaded — it is WRONG, and it produces
 * one table of numbers that came from two different runs, which looks exactly
 * like a table of numbers that came from one. The server's sentence is
 * rendered as a refusal, never as a warning with a button beside it.
 */

/** How many rows to ask for. A REQUEST parameter, so it is the one list here
 *  that cannot come from a response — there is nothing to have asked yet. The
 *  middle value is the route's own default, so the panel's resting behaviour
 *  is the server's. */
const LIMITS = [10, 50, 200];
const DEFAULT_LIMIT = 50;

/** How many remaining prompt indices to print before saying how many are left
 *  unprinted. A 500-prompt sweep would otherwise put 500 numbers on screen. */
const INDICES_SHOWN = 24;

/**
 * When it started, in the READER'S timezone.
 *
 * The store writes UTC and the reader is not in UTC. An empty or unparseable
 * timestamp returns "" and the row prints a sentence saying it was not
 * recorded — never an epoch date, which is what `new Date("")` renders as and
 * which reads as a real measurement from 1970.
 */
function when(iso: string): string {
  if (!iso) return "";
  const t = new Date(iso);
  if (Number.isNaN(t.getTime())) return "";
  return t.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function ResumeBlock({ plan }: { plan: ResumePlan }) {
  const extra = Math.max(0, plan.remaining_indices.length - INDICES_SHOWN);
  return (
    <div className="sw-plan">
      {/* Verbatim. It already states what is measured, what is left, and — when
          there is one — the reason this cannot run, in one sentence written to
          travel with the numbers. */}
      <div className="hint">{plan.means}</div>
      <span className="meta">
        {plan.model || "the model was not recorded"} · {plan.metric} ·{" "}
        {plan.n_measured} of {plan.n_prompts} measured · {plan.n_remaining} left
      </span>
      {plan.blocked === null ? (
        // `null` is not an unknown here — the server documents it as "nothing
        // blocks it", which is a checked answer and not a missing one.
        //
        // The wording is exact about the third check and had to be corrected
        // once: it read "the model that ran them is the model loaded now",
        // which was observed rendering with NOTHING loaded. `sweep._resumable`
        // only fires that check when a model is resident and its id differs,
        // so a pass means "nothing loaded disagrees" and not "the right model
        // is here" — and the response carries no field that could tell the two
        // apart. Claiming the stronger one is exactly the kind of small
        // invention this panel exists to avoid.
        <div className="hint ok">
          Nothing blocks this resume. The three checks that would make it{" "}
          <em>wrong</em> rather than merely expensive all passed: every saved
          row indexes a prompt in this set, every measured row's prompt still
          hashes to the text it was measured on, and no model loaded now
          disagrees with the <code>{plan.model || "unnamed model"}</code> this
          sweep ran on. Finishing it costs only the {plan.n_remaining} prompt(s)
          below.
        </div>
      ) : (
        // A sentence, not a warning. There is deliberately no button beside
        // it: the three things checked here make a resume WRONG rather than
        // expensive, and each one produces a table whose numbers came from two
        // different runs while looking like one.
        <div className="hint err">
          <strong>This sweep must not be resumed.</strong> {plan.blocked}
        </div>
      )}
      {plan.remaining_indices.length > 0 && (
        <div className="sw-idx">
          <span className="meta">still to run — prompt index</span>
          <span className="mid">
            {plan.remaining_indices.slice(0, INDICES_SHOWN).join(" ")}
          </span>
          {/* The cap on the LIST, distinct from the count above it. */}
          {extra > 0 && (
            <span className="meta">
              + {extra} more index(es) not printed here
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function SweepRow({
  s,
  plan,
  planErr,
  planning,
  onPlan,
}: {
  s: SavedSweep;
  plan: ResumePlan | null;
  planErr: string;
  planning: boolean;
  onPlan: () => void;
}) {
  const at = when(s.started_at);
  // Guarded rather than computed blind: a sweep row with no prompts would
  // otherwise divide by zero and render a NaN-wide bar, which is a shape on
  // screen that means nothing.
  const share = s.n_prompts > 0 ? s.n_measured / s.n_prompts : null;
  return (
    <li className="sw-row">
      <div className="sw-row-head">
        <span className="mid sw-id">{s.sweep_id}</span>
        <span className={`pill tiny ${s.complete ? "on" : ""}`}>
          {s.complete ? "complete" : "unfinished"}
        </span>
        <span className="spacer" />
        <span className="meta">
          {/* Never "1 Jan 1970". An empty timestamp is a fact nobody wrote
              down, and it is said in words. */}
          {at || "start time not recorded"}
        </span>
      </div>
      <div className="sw-row-facts meta">
        <span className="sw-model">{s.model || "model not recorded"}</span>
        <span>metric {s.metric || "not recorded"}</span>
        <span>
          {s.n_measured} of {s.n_prompts} prompt(s) measured
        </span>
        {/* A prompt the measurement could not be taken on is a ROW, not a gap
            — so a refusal count sits beside a complete sweep rather than
            contradicting it. Shown only when there is one. */}
        {s.n_refused > 0 && (
          <span className="sw-refused">{s.n_refused} refused</span>
        )}
        <span>{s.n_remaining} left</span>
      </div>
      {share === null ? (
        <span className="meta">
          no prompts recorded for this sweep, so there is no progress to draw
        </span>
      ) : (
        <div
          className="sw-bar"
          role="img"
          aria-label={`${s.n_measured} of ${s.n_prompts} prompts measured`}
        >
          <i style={{ width: `${share * 100}%` }} />
        </div>
      )}
      {!s.complete && (
        <div className="row sw-actions">
          <button className="ghost sm" onClick={onPlan} disabled={planning}>
            {planning
              ? "pricing…"
              : plan
                ? "price the resume again"
                : "what would finishing this cost?"}
          </button>
          <span className="meta">
            reads the saved rows and checks them against what is loaded now —
            nothing is run
          </span>
        </div>
      )}
      {planErr && <div className="hint err">{planErr}</div>}
      {plan && <ResumeBlock plan={plan} />}
    </li>
  );
}

export default function SweepsPanel() {
  const [list, setList] = useState<SweepList | null>(null);
  const [limit, setLimit] = useState(DEFAULT_LIMIT);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  // Keyed by sweep id: several sweeps can be priced without the second answer
  // replacing the first, which is how you compare two interrupted runs.
  const [plans, setPlans] = useState<Record<string, ResumePlan>>({});
  const [planErrs, setPlanErrs] = useState<Record<string, string>>({});
  const [planning, setPlanning] = useState("");
  const scanRef = useScanOnData(list ? `${limit}:${list.sweeps.length}` : null);

  const load = useCallback(async (n: number) => {
    setLoading(true);
    setErr("");
    try {
      setList(await savedSweeps(n));
    } catch (e) {
      setErr(errorText(e));
      setList(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load(limit);
  }, [load, limit]);

  async function price(id: string) {
    setPlanning(id);
    setPlanErrs((prev) => ({ ...prev, [id]: "" }));
    try {
      const plan = await sweepResumePlan(id);
      setPlans((prev) => ({ ...prev, [id]: plan }));
    } catch (e) {
      // The server's own sentence. A sweep written by a different version of
      // ModelMRI, or an id that is not on this machine, both answer with a
      // paragraph saying exactly that.
      setPlanErrs((prev) => ({ ...prev, [id]: errorText(e) }));
      setPlans((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
    } finally {
      setPlanning("");
    }
  }

  const sweeps = list?.sweeps ?? [];
  // The cap, and the only thing a caller can honestly say about it: the
  // response has no total field, so a full page of rows means "there may be
  // more" rather than "there are more".
  const maybeMore = sweeps.length >= limit;
  const unfinished = sweeps.filter((s) => !s.complete).length;

  return (
    <div className="panel sweeps" ref={scanRef}>
      <div className="sect">
        <span className="dot d-sweep" />
        <h2 className="h-sweep">SWEEPS — THE SAME MEASUREMENT, MANY PROMPTS</h2>
        <span className="rule" />
      </div>

      <p className="meta sw-what">
        A number measured once is a sample, not a property.{" "}
        <code>modelmri sweep</code> runs one measurement over a set of prompts
        and reports each row as median, IQR and n instead of as a number. Every
        run it has made is saved beside your traces; this is the list, and what
        finishing an interrupted one would cost.
      </p>

      <div className="row sw-controls">
        <label className="meta" htmlFor="sw-limit">
          ask for at most
        </label>
        <select
          id="sw-limit"
          className="sm"
          value={limit}
          disabled={loading}
          onChange={(e) => setLimit(Number(e.target.value))}
          title="How many rows to request. The response carries no total, so this cap is the only thing that bounds the list."
        >
          {LIMITS.map((n) => (
            <option key={n} value={n}>
              {n} rows
            </option>
          ))}
        </select>
        <button
          className="ghost sm"
          onClick={() => void load(limit)}
          disabled={loading}
        >
          {loading ? "reading…" : "re-read"}
        </button>
        <span className="spacer" />
        {list && (
          <span className="meta">
            {sweeps.length} row(s) here · {unfinished} unfinished
          </span>
        )}
      </div>

      {err && <div className="hint err">{err}</div>}

      {list && (
        <>
          {/* The server's own sentence, whole. Note that it counts the rows it
              RETURNED, which is why the cap notice below it is not optional. */}
          <div className="hint">{list.means}</div>

          {maybeMore ? (
            <div className="hint warn">
              That is exactly the {limit} rows this asked for, and the response
              carries no total — so there may be sweeps on this machine that
              this list does not show, and the sentence above counts what came
              back rather than what exists. Ask for more to find out.
            </div>
          ) : (
            <span className="meta">
              Fewer rows came back than the {limit} asked for, so this is every
              sweep saved on this machine.
            </span>
          )}

          {sweeps.length === 0 ? (
            <div className="hint">
              No sweep has been saved on this machine yet. One is made by{" "}
              <code>modelmri sweep</code> — it needs a prompt file and a metric,
              runs headless, and saves whatever it got even if you stop it
              partway.
            </div>
          ) : (
            <ul className="sw-list">
              {sweeps.map((s) => (
                <SweepRow
                  key={s.sweep_id}
                  s={s}
                  plan={plans[s.sweep_id] ?? null}
                  planErr={planErrs[s.sweep_id] ?? ""}
                  planning={planning === s.sweep_id}
                  onPlan={() => void price(s.sweep_id)}
                />
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  );
}
