import { useEffect, useState } from "react";
import { Counterfactual, ControlArm, errorText, tokenCounterfactual } from "./api";
import { measured } from "./measured";
import Disclosure from "./Disclosure";

/** The smallest edit to your words that makes the model say something ELSE.
 *
 *  The third question on this panel. "Rank tokens" masks a word out and
 *  measures what breaks — NECESSITY. "Anchors" keeps a few and perturbs the
 *  rest — SUFFICIENCY. Both describe the answer the model already gives. This
 *  one is directional and produces a recipe: what to write INSTEAD.
 *
 *  A FLIPPED ANSWER IS NOT A FINDING ON ITS OWN. Edit enough of a prompt and
 *  any model says something else, so the edit is scored against two controls —
 *  random words at the same positions, and random words at as many positions
 *  anywhere. Both are rendered as counts with their Wilson bounds, and the
 *  verdict line reads "not a finding" unless both arms were actually drawn and
 *  both came back empty. An arm that took no draws is rendered as "not
 *  measured", never as 0%: zero out of zero is an absence, and showing it as a
 *  rate would turn the absence of evidence into the strongest evidence on
 *  screen.
 *
 *  The edit it finds is often not natural language. A gradient-guided token
 *  search finds ADVERSARIAL substitutions — measured on Qwen3-1.7B, steering
 *  "The Eiffel Tower is in the city of" to " Rome" produced
 *  "The皇家cente虹桥LTR is in the city of", which beat both controls 0/24 and
 *  reads as junk. That is a true property of the method and it is printed
 *  under the result rather than hidden by it.
 */

function Arm({ arm, label, asks }: { arm: ControlArm; label: string; asks: string }) {
  if (!arm.measured || arm.point === null || arm.interval === null) {
    return (
      <li className="cf-arm">
        <span className="meta">{label}</span>
        <span className="meta warn">
          not measured — no draw was taken, so this is an absence rather than a
          rate of zero
        </span>
      </li>
    );
  }
  const [low, high] = arm.interval;
  return (
    <li className="cf-arm">
      <span className="meta">{label}</span>
      <span className="cf-track">
        <i
          className="cf-band"
          style={{
            left: `${(low * 100).toFixed(2)}%`,
            width: `${Math.max(0.6, (high - low) * 100).toFixed(2)}%`,
          }}
        />
        <i className="cf-point" style={{ left: `${(arm.point * 100).toFixed(2)}%` }} />
      </span>
      <b>
        {arm.successes}/{arm.samples}
        <span className="meta">
          {" "}
          [{measured(low, 3)}, {measured(high, 3)}]
        </span>
      </b>
      <span className="meta cf-asks">{asks}</span>
    </li>
  );
}

export default function TokenCounterfactual({
  position,
  epoch,
  disabled,
}: {
  position: number;
  /** Bumped on every generation. The payload below is about ONE of them. */
  epoch: number;
  disabled?: boolean;
}) {
  const [target, setTarget] = useState("");
  const [maxEdits, setMaxEdits] = useState(3);
  const [proposals, setProposals] = useState(24);
  const [data, setData] = useState<Counterfactual | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  // A NEW GENERATION MUST NOT LEAVE THE OLD ONE'S ANSWER ON SCREEN. The edit
  // below is a set of indices into ONE token strip; against a different strip
  // those indices point at different words and the panel would be labelling
  // the new prompt with the old prompt's finding.
  useEffect(() => {
    setData(null);
    setErr("");
  }, [epoch, position]);

  const run = () => {
    if (!target.trim()) {
      setErr("Name the token the model should be steered toward.");
      return;
    }
    setBusy(true);
    setErr("");
    tokenCounterfactual({
      position,
      target: target.trim(),
      max_edits: maxEdits,
      n_proposals: proposals,
    })
      .then((d) => {
        setData(d);
        setBusy(false);
      })
      .catch((e) => {
        setErr(errorText(e));
        setData(null);
        setBusy(false);
      });
  };

  return (
    <div className="token-counterfactual">
      <Disclosure
        dot="d-attn"
        title="WHAT WOULD MAKE IT SAY SOMETHING ELSE"
        asks="What is the smallest edit to your words that makes the next token become one you name? Reachability — the recipe, where ranking and anchors describe the answer it already gives."
        hasResult={data !== null}
        disabled={disabled}
      >

      <div className="cf-controls">
        <label>
          Steer toward
          <input
            type="text"
            value={target}
            placeholder="e.g. Rome"
            onChange={(e) => setTarget(e.target.value)}
            disabled={disabled || busy}
          />
        </label>
        <label>
          Edits
          <input
            type="number"
            min={1}
            max={8}
            value={maxEdits}
            onChange={(e) => setMaxEdits(Number(e.target.value))}
            disabled={disabled || busy}
          />
        </label>
        <label>
          Shortlist
          <input
            type="number"
            min={1}
            max={128}
            value={proposals}
            onChange={(e) => setProposals(Number(e.target.value))}
            disabled={disabled || busy}
          />
        </label>
        <button onClick={run} disabled={disabled || busy}>
          {busy ? "Searching…" : "Find an edit"}
        </button>
      </div>

      {err && <div className="hint err">{err}</div>}

      {data && (
        <div className="cf-result">
          <p className="cf-verdict">
            {data.found ? (
              <>
                <b>{data.size}</b> substitution{data.size === 1 ? "" : "s"} make
                it predict <code>{data.target_token ?? data.target_token_id}</code>{" "}
                instead of <code>{data.base_token ?? data.base_token_id}</code>.
              </>
            ) : (
              <>No edit within this budget reached{" "}
                <code>{data.target_token ?? data.target_token_id}</code>.</>
            )}
          </p>
          <p className="meta">{data.stopped_because}.</p>

          {/* The verdict that matters, and it is NOT `found`. An edit that
              random edits of the same size also achieve has isolated nothing. */}
          <p className={data.beats_controls ? "cf-finding" : "cf-notfinding"}>
            {data.beats_controls
              ? "This beats both controls — a random edit of the same size did not reach the target."
              : data.found
                ? "NOT A FINDING — a random edit of the same size reaches the target too, or the controls were never drawn."
                : "Nothing to control: no edit was found."}
          </p>

          {data.edit.length > 0 && (
            <ol className="cf-steps">
              {data.edit.map((s) => (
                <li key={s.step}>
                  <span className="meta">index {s.index}</span>
                  <code className="cf-from">{s.from_token ?? s.from_token_id}</code>
                  <span className="cf-arrow">→</span>
                  <code className="cf-to">{s.to_token ?? s.to_token_id}</code>
                  <span className="meta">
                    p(target) = {measured(s.target_p_after, 6)}
                  </span>
                </li>
              ))}
            </ol>
          )}

          {data.edited_text && (
            <p className="cf-edited">
              <span className="meta">Edited prompt</span>
              <code>{data.edited_text}</code>
            </p>
          )}

          <ul className="cf-arms">
            <Arm
              arm={data.controls.same_positions}
              label="Random words, same positions"
              asks={data.controls.same_positions_asks}
            />
            <Arm
              arm={data.controls.any_positions}
              label="Random words, any positions"
              asks={data.controls.any_positions_asks}
            />
          </ul>
          {data.controls.not_measured_because && (
            <p className="meta warn">{data.controls.not_measured_because}.</p>
          )}

          <p className="meta cf-how">How this was searched, and what it cost</p>
            <ul className="meta cf-detail">
              <li>{data.screen.source}.</li>
              {data.screen.top_choice_won.measured && (
                <li>
                  The estimate's first choice won{" "}
                  <b>
                    {data.screen.top_choice_won.successes}/
                    {data.screen.top_choice_won.samples}
                  </b>{" "}
                  of its steps. {data.screen.top_choice_won_asks}
                </li>
              )}
              <li>
                The target started at rank {data.base_target_rank} with
                p = {measured(data.base_target_p, 6)}.
              </li>
              <li>
                {data.passes} forward passes ({data.screen.backward_passes}{" "}
                backward), {measured(data.seconds, 3)} s. Expected{" "}
                {data.passes_expected}: {data.trials_skipped_self} candidate
                {data.trials_skipped_self === 1 ? "" : "s"} skipped as no-ops,{" "}
                {data.trials_short_circuited} never reached after a hit,{" "}
                {data.trials_unavailable} never offered.
              </li>
              <li>{data.minimality.search}. A smaller edit may exist.</li>
              <li>{data.controls.no_self_substitution}</li>
              <li>{data.beats_controls_means}</li>
              <li>{data.edited_ids_are}</li>
              <li>
                A gradient-guided token search finds ADVERSARIAL substitutions,
                not natural sentences. An edit that reads as junk and still
                beats its controls is a true statement about this model's
                decision boundary and a poor paraphrase of your prompt. Both
                things are the case at once.
              </li>
            </ul>
        </div>
      )}
      </Disclosure>
    </div>
  );
}
