import { useEffect, useState } from "react";
import { measured } from "./measured";
import {
  ApiError,
  errorText,
  judgePlan,
  judgeScore,
  JudgePlan,
  JudgeScore,
  TraceStep,
} from "./api";

/**
 * Score a rubric by READING the model's probability mass, not by sampling it.
 *
 * It sits beside the rubric scorer on purpose, and the two are opposites worth
 * seeing together. That one asks exact predicates and never needs a model, so
 * a match is a match. This one asks a model, and everything below exists to
 * stop that answer from being read as a property of the text.
 *
 * WHAT THE PANEL HAS TO KEEP TRUE
 *
 *   - **The mass, beside the ratio, always.** `p_yes` is which of the two
 *     verdicts, GIVEN the model answered at all. A weak judge puts three
 *     percent of its mass on a verdict token and then splits it near 50/50
 *     whatever it is shown; the ratio alone hides that and the mass does not.
 *   - **A phrasing below the floor is NOT in the median.** It is drawn, and
 *     drawn differently, because a model that will not answer one wording of a
 *     rubric is a fact about the rubric. Its p(yes) is a ratio between two
 *     rounding errors, so its bar is hollow rather than filled.
 *   - **A refusal is a finding.** "The model did not answer the rubric in any
 *     of 4 phrasings" is the measurement's answer — it is a 409 like every
 *     other deliberate no in this project — and it renders as a result rather
 *     than as red text under a broken button.
 *   - **The price before the spend.** `/api/judge/plan` returns the exact
 *     prompts that would be run, and no model is touched to produce them. They
 *     are on screen, in full, before the button that spends the passes.
 *
 * Nothing here aggregates across rubrics or across runs. That aggregate is
 * exactly where one model's sample starts being treated as a property.
 */

/** A number the server may not have sent at all.
 *
 *  `undefined` here is UNKNOWN — the server writes `median` and its four
 *  neighbours only when there is at least one pass to compute them from — and
 *  it must never come out as `0.000`, which is a claim that the model said no
 *  with total confidence.
 */
function num(v: number | undefined, dp: number): string {
  return v === undefined ? "unknown" : v.toFixed(dp);
}

/** How much of the model's whole vocabulary landed on a verdict token. */
function pct(v: number): string {
  return `${(v * 100).toFixed(v < 0.01 ? 2 : 1)}%`;
}

export default function JudgePanel({ step }: { step: TraceStep | null }) {
  const [text, setText] = useState("");
  const [rubric, setRubric] = useState("");
  // Held as a string so the field can be BLANK. Blank is not zero paraphrases
  // — it is "ask it every way this build knows", which the server resolves and
  // the plan then reports back as a real count.
  const [k, setK] = useState("");
  const [plan, setPlan] = useState<JudgePlan | null>(null);
  const [planErr, setPlanErr] = useState("");
  const [score, setScore] = useState<JudgeScore | null>(null);
  // The plan the run on screen was actually priced with. Kept apart from
  // `plan`, which follows the text box — otherwise editing the rubric after a
  // run would relabel the prompts under a result they did not produce.
  const [ran, setRan] = useState<JudgePlan | null>(null);
  const [refusal, setRefusal] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  // Characters `traces._clip` did not store, for the step this text came from.
  // Judging a clipped output is judging a different text, so it is carried
  // with the text rather than left in the timeline the reader has scrolled
  // away from.
  const [fromStep, setFromStep] = useState<TraceStep | null>(null);

  const kNum = Number(k.trim()) || 0;

  // The plan follows the boxes, so the cost is on screen before the button
  // rather than after the wait. It runs no model — it formats prompts — which
  // is what makes it safe to fire on every keystroke behind a debounce.
  useEffect(() => {
    if (!text.trim() || !rubric.trim()) {
      setPlan(null);
      setPlanErr("");
      return;
    }
    let live = true;
    const timer = window.setTimeout(() => {
      void judgePlan({ text, rubric, n_paraphrases: kNum })
        .then((p) => {
          if (!live) return;
          setPlan(p);
          setPlanErr("");
        })
        .catch((e) => {
          if (!live) return;
          setPlan(null);
          setPlanErr(errorText(e));
        });
    }, 250);
    return () => {
      live = false;
      window.clearTimeout(timer);
    };
  }, [text, rubric, kNum]);

  async function run() {
    setBusy(true);
    setErr("");
    setRefusal("");
    setScore(null);
    const priced = plan;
    try {
      const out = await judgeScore({ text, rubric, n_paraphrases: kNum });
      setRan(priced);
      setScore(out);
    } catch (e) {
      // 409 is this project's deliberate no, and the judge's is the most
      // interesting one it has: the model did not put enough mass on a verdict
      // token to have answered. That is a reading, not a fault, so it is not
      // drawn as one. 422 IS the reader's to fix — a rubric past the cap, an
      // empty text — and stays an error.
      if (e instanceof ApiError && e.status === 409) setRefusal(errorText(e));
      else setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  const passes = score?.passes ?? [];
  const unanswered = passes.filter((p) => !p.answered);

  return (
    <div className="judge-panel">
      <div className="row">
        <span className="meta">
          <b>ask the loaded model a yes/no rubric</b> — one forward pass per
          phrasing, no generation, and the number is the mass it put on the
          verdict token rather than one sample of what it would have said
        </span>
      </div>

      <label className="meta" htmlFor="judge-text">
        text to judge
      </label>
      <textarea
        id="judge-text"
        className="judge-text"
        rows={4}
        spellCheck={false}
        placeholder="the passage this rubric is about"
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          setFromStep(null);
        }}
      />
      <div className="row judge-meter">
        <span className="meta">
          {text.length.toLocaleString()} character
          {text.length === 1 ? "" : "s"}
        </span>
        {/* The obvious thing to judge is a step of a run you recorded, and it
            is already on screen above. Filling the box rather than judging it
            invisibly, so what was measured is the text you can read. */}
        {step && step.output ? (
          <button
            className="ghost sm"
            onClick={() => {
              setText(step.output);
              setFromStep(step);
            }}
            title={`the output of the selected ${step.kind} step`}
          >
            use the selected step&rsquo;s output
          </button>
        ) : null}
      </div>

      {/* A CAP, and one that changes what the score is about: the store clipped
          this output, so judging it judges a prefix. */}
      {fromStep && (fromStep.truncated_out ?? 0) > 0 && (
        <p className="hint judge-cap">
          This came from a step whose output the store clipped:{" "}
          <b>{(fromStep.truncated_out ?? 0).toLocaleString()}</b> character
          {(fromStep.truncated_out ?? 0) === 1 ? " was" : "s were"} not kept.
          The rubric will be answered about what is in the box, which is a
          prefix of what the step actually produced.
        </p>
      )}

      <label className="meta" htmlFor="judge-rubric">
        rubric
      </label>
      <input
        id="judge-rubric"
        className="judge-rubric"
        spellCheck={false}
        placeholder="a question this model can answer yes or no to"
        value={rubric}
        onChange={(e) => setRubric(e.target.value)}
      />
      <div className="row judge-meter">
        <span className="meta">
          {rubric.length.toLocaleString()} character
          {rubric.length === 1 ? "" : "s"}
        </span>
        <label className="meta" htmlFor="judge-k">
          phrasings
        </label>
        <input
          id="judge-k"
          className="vla-num"
          type="number"
          min={1}
          placeholder="all"
          value={k}
          onChange={(e) => setK(e.target.value)}
        />
        <span className="meta">
          blank asks it every way this build knows, and the plan below says how
          many that is
        </span>
      </div>

      {/* Every refusal the plan can give is about the text or the rubric — no
          rubric, no text, past a cap — so it is advice rather than a failure,
          and the cap it names is the server's own. */}
      {planErr && <p className="hint judge-planerr">{planErr}</p>}

      {plan && (
        <div className="judge-plan">
          <p className="meta">
            <b>{plan.n_passes}</b> forward pass
            {plan.n_passes === 1 ? "" : "es"}, one per phrasing, and no
            generation.
            {kNum === 0
              ? " That is every phrasing this build asks the rubric."
              : ""}
          </p>
          <details>
            <summary className="meta">
              the exact prompts, before any of them is run
            </summary>
            <ol className="judge-prompts">
              {plan.prompts.map((prompt, i) => (
                <li key={i}>
                  <pre>{prompt}</pre>
                </li>
              ))}
            </ol>
          </details>
        </div>
      )}

      <div className="row judge-go">
        <button
          className="cta"
          disabled={busy || !plan}
          onClick={() => void run()}
        >
          {busy
            ? "Reading the mass…"
            : plan
              ? `Score · ${plan.n_passes} forward pass${plan.n_passes === 1 ? "" : "es"}`
              : "Score"}
        </button>
      </div>

      {/* A DELIBERATE NO. The model not committing enough mass to have answered
          is the measurement's result, and this project's whole argument is that
          such an answer is worth more than a confident-looking number made by
          normalising two rounding errors. So it is drawn as a reading. */}
      {refusal && (
        <div className="judge-refusal">
          <span className="judge-tag">refused, and that is the reading</span>
          <p>{refusal}</p>
        </div>
      )}

      {err && <div className="hint err">{err}</div>}

      {score && (
        <div className="judge-out">
          {/* VERBATIM. Every caveat this measurement carries — how much mass
              was committed, which phrasings are missing from the median, how
              far the paraphrases disagree, and who the judge was — is in this
              sentence, and a summary is where they get dropped. */}
          <p className="meta judge-means">{score.means}</p>

          {passes.length === 0 ? (
            <p className="meta">
              No paraphrase was scored, so there is no median — not a median of
              zero.
            </p>
          ) : (
            <div className="row judge-nums">
              <span className="judge-num">
                <b>{num(score.median, 3)}</b> p(yes), median
              </span>
              <span className="judge-num">
                {num(score.low, 3)} – {num(score.high, 3)}
              </span>
              <span className="judge-num">
                spread {num(score.spread, 3)}
              </span>
              <span className="judge-num">
                over{" "}
                {score.n_paraphrases === undefined
                  ? "unknown"
                  : score.n_paraphrases}{" "}
                of {passes.length} phrasing
                {passes.length === 1 ? "" : "s"}
              </span>
            </div>
          )}

          <ol className="judge-passes">
            {passes.map((p) => (
              <li key={p.paraphrase} className={p.answered ? "" : "unanswered"}>
                <span className="mid">#{p.paraphrase + 1}</span>
                <span className="judge-track">
                  {/* Hollow for a phrasing the model did not answer: the bar
                      would otherwise draw a ratio between two rounding errors
                      at the same weight as a real verdict. */}
                  <span
                    className="judge-bar"
                    style={{ width: `${Math.min(100, p.p_yes * 100)}%` }}
                  />
                </span>
                <span className="mid">{measured(p.p_yes, 3)}</span>
                <span className="meta">
                  {p.answered ? (
                    <>
                      {pct(p.mass)} of its mass on a verdict token · p(no){" "}
                      {measured(p.p_no, 3)}
                    </>
                  ) : (
                    <>
                      <b>NOT IN THE MEDIAN</b> — only {pct(p.mass)} of its mass
                      landed on a verdict token, so there was no answer here to
                      include
                    </>
                  )}
                </span>
                {/* Which wording produced this row, from the plan this run was
                    priced with rather than from whatever is in the box now. */}
                {ran && ran.prompts[p.paraphrase] !== undefined && (
                  <details className="judge-pass-prompt">
                    <summary className="meta">the phrasing</summary>
                    <pre>{ran.prompts[p.paraphrase]}</pre>
                  </details>
                )}
              </li>
            ))}
          </ol>

          {unanswered.length > 0 && (
            <p className="meta judge-note">
              {unanswered.length} of {passes.length} phrasing
              {passes.length === 1 ? "" : "s"} sit outside the median above.
              They were run and are drawn, because a wording this model will not
              answer is a fact about the rubric rather than a gap in the
              measurement.
            </p>
          )}

          {/* The ids the mass was read off, because which token counted as
              "yes" is part of what the number means. */}
          {score.tokens === null ? (
            <p className="meta judge-note">
              The verdict tokens were not reported with this score, so nothing
              here says which ids the mass was read off.
            </p>
          ) : (
            <p className="meta judge-note">
              Mass summed over {score.tokens.yes_ids.length} yes id
              {score.tokens.yes_ids.length === 1 ? "" : "s"} (
              {score.tokens.yes_forms.map((f) => JSON.stringify(f)).join(", ")})
              and {score.tokens.no_ids.length} no id
              {score.tokens.no_ids.length === 1 ? "" : "s"} (
              {score.tokens.no_forms.map((f) => JSON.stringify(f)).join(", ")}).
              One form per DISTINCT id: several casings can share one, and
              adding a repeated id once per casing would report a probability
              above 1.
            </p>
          )}

          <p className="meta judge-note">
            {/* `null` is "no seed was fixed" and is NOT seed 0. Nothing here
                samples, so there is no draw to fix — which is exactly why the
                field says nothing rather than showing a zero. */}
            seed:{" "}
            {score.seed === null
              ? "none was fixed — this reads a distribution rather than drawing from it"
              : score.seed}
          </p>
        </div>
      )}
    </div>
  );
}
