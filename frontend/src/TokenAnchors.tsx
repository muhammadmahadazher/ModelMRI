// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

import { useEffect, useState } from "react";
import { Anchors, errorText, Proportion, tokenAnchors } from "./api";
import { measured } from "./measured";
import Disclosure from "./Disclosure";

/** The smallest set of the prompt's own tokens that HOLDS the answer.
 *
 *  The other half of "Rank tokens", and the pair is the point: that one masks
 *  a token out and measures what breaks — NECESSITY. This one keeps a few and
 *  perturbs everything else, and asks whether the prediction survives —
 *  SUFFICIENCY. A token can be necessary without being sufficient, and a
 *  reader shown one ranking alone will read it as both.
 *
 *  THREE NUMBERS, AND ONE OF THEM ALONE MEANS NOTHING. A precision of 0.86
 *  sounds like a finding until you see the base rate — how often the answer
 *  survives with NOTHING held — sitting at 0.84. So all three are rendered on
 *  one scale, as intervals rather than points, with the target drawn as a
 *  line: the anchor's precision, the floor it has to beat, and the ceiling
 *  above which no anchor at this position can go.
 */

function Interval({ p, label, target }: { p: Proportion; label: string; target?: number }) {
  // `measured: false` means EVERY number below is null. The route is explicit
  // that it must never put a 0 or a 1 where a measured proportion goes, and
  // `implied` is a separate key so arithmetic can be told from evidence.
  if (!p.measured || p.point === null || p.low === null || p.high === null) {
    return (
      <li className="an-interval">
        <span className="meta">{label}</span>
        <span className="meta warn">
          not measured — {p.reason}
          {p.implied !== null && p.implied !== undefined && (
            <> The arithmetic implies {measured(p.implied, 3)}, which nobody sampled.</>
          )}
        </span>
      </li>
    );
  }
  return (
    <li className="an-interval">
      <span className="meta">{label}</span>
      <span className="an-track">
        {/* The whole interval, not the point. A Wilson bound at 40 draws is
            wide, and a bare point hides that the measurement barely
            distinguishes 0.2 from 0.4. */}
        <i
          className="an-band"
          style={{
            left: `${(p.low * 100).toFixed(2)}%`,
            width: `${Math.max(0.6, (p.high - p.low) * 100).toFixed(2)}%`,
          }}
        />
        <i className="an-point" style={{ left: `${(p.point * 100).toFixed(2)}%` }} />
        {target !== undefined && (
          <i className="an-target" style={{ left: `${(target * 100).toFixed(2)}%` }} />
        )}
      </span>
      <b>
        {measured(p.point, 3)}
        <span className="meta">
          {" "}
          [{measured(p.low, 3)}, {measured(p.high, 3)}]
        </span>
      </b>
      <span className="meta">
        {p.held}/{p.samples}
      </span>
    </li>
  );
}

export default function TokenAnchors({
  position,
  epoch,
  disabled,
}: {
  position: number;
  /** Bumped on every generation. The payload below is about ONE of them. */
  epoch: number;
  disabled?: boolean;
}) {
  const [samples, setSamples] = useState(64);
  const [maxSize, setMaxSize] = useState(3);
  const [data, setData] = useState<Anchors | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  // A NEW GENERATION MUST NOT LEAVE THE OLD ONE'S ANSWER ON SCREEN.
  //
  // `ArcCanvas` calls `pin(-1)` whenever the token strip changes, so every
  // generation resets the pin. The old guard was
  //
  //     data !== null && position >= 0 && data.position !== position
  //
  // whose `position >= 0` clause short-circuits at exactly that moment — so
  // generating again left the PREVIOUS run's payload rendered under the new
  // token strip, labelled with a position that now holds a different token.
  // `epoch` changes with the generation and clearing on it is the fix; the
  // position check stays for the ordinary case of moving the pin.
  useEffect(() => {
    setData(null);
    setErr("");
  }, [epoch]);

  async function run() {
    if (busy) return;
    setBusy(true);
    setErr("");
    try {
      setData(
        await tokenAnchors({
          position: position >= 0 ? position : undefined,
          n_samples: samples,
          max_size: maxSize,
          max_candidates: 8,
        }),
      );
    } catch (e) {
      setData(null);
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  const stale = data !== null && position >= 0 && data.position !== position;

  return (
    <div className="token-anchors">
      <Disclosure
        dot="d-attn"
        title="WHAT HOLDS THE ANSWER ON ITS OWN"
        asks="Which of your tokens hold the prediction on their own? Sufficiency — the opposite question from ranking, and the two are expected to disagree."
        hasResult={data !== null}
      >

      <div className="row">
        <label className="meta" htmlFor="an-samples">
          draws
        </label>
        <input
          id="an-samples"
          type="range"
          min={20}
          max={200}
          step={4}
          value={samples}
          onChange={(e) => setSamples(Number(e.target.value))}
          disabled={disabled || busy}
        />
        <span className="meta">{samples}</span>
        <label className="meta" htmlFor="an-size">
          largest anchor
        </label>
        <input
          id="an-size"
          type="range"
          min={1}
          max={6}
          value={maxSize}
          onChange={(e) => setMaxSize(Number(e.target.value))}
          disabled={disabled || busy}
        />
        <span className="meta">{maxSize} tokens</span>
        <button className="ghost sm" onClick={() => void run()} disabled={disabled || busy}>
          {busy ? "searching…" : data ? "search again" : "find an anchor"}
        </button>
      </div>
      {/* The draw count is the RESOLUTION, not a speed dial: the search gates
          on a Wilson lower bound, so too few draws makes a target
          unreachable however good the anchor is. The route refuses with the
          arithmetic when that happens; saying it here means the reader does
          not have to trigger the refusal to learn it. */}
      <p className="meta">
        Draws are resolution, not speed. The search gates on the lower end of
        the interval, so a target can be unreachable at a low count however
        good the anchor is — the refusal says the exact number that would
        reach it.
      </p>
      {/* MEASURED, and it is the first thing anybody will hit: with nothing
          pinned the route anchors at the last PROMPT token, which on a chat
          model is the template's own `<think>` or `assistant` — and the route
          refuses a control token, correctly. Left alone, the only obvious
          click on this panel always fails. Saying so costs a line; letting a
          reader discover it costs them the impression that the measurement is
          broken. */}
      {position < 0 && (
        <p className="meta warn">
          No token pinned, so this anchors at the last prompt token — which on
          a chat-template model is a control token, and the route refuses those
          (an anchor for one would be a set of your words that keeps the
          template working). Click a token in the strip above to choose a
          position where the model is answering.
        </p>
      )}

      {err && <div className="hint err">{err}</div>}

      {data && !stale && (
        <>
          <div className={`an-verdict ${data.found ? "found" : "none"}`}>
            {/* `found` is true for BOTH "target-reached" and
                "empty-anchor-sufficient", and the second is the opposite
                reading: the anchor is EMPTY, nothing in the candidate window
                holds the answer, and it is being held by the template, the
                sink, the position or the prior. Rendered as a find, that said
                "0 tokens hold it" in the success styling. */}
            {data.found && data.stopped_because === "empty-anchor-sufficient" ? (
              <>
                <b>Nothing in your words was needed.</b> The prediction{" "}
                <code>{data.target_token}</code> survived with the anchor
                EMPTY, in {data.base_rate.held ?? 0} of {data.base_rate.samples}{" "}
                draws — so it is being held by the chat template, the attention
                sink, the position or the model's prior, none of which this
                search can perturb.
              </>
            ) : data.found ? (
              <>
                <b>
                  {data.size} token{data.size === 1 ? "" : "s"} hold it.
                </b>{" "}
                Perturbing everything else left <code>{data.target_token}</code>{" "}
                as the prediction in {data.precision.held ?? 0} of{" "}
                {data.precision.samples} draws.
              </>
            ) : (
              <>
                <b>No anchor found</b> at up to {data.candidates.max_size}{" "}
                tokens — <code>{data.stopped_because}</code>.
                {data.stopped_because === "ceiling-below-target" && (
                  <>
                    {" "}
                    The best any anchor could do here measured{" "}
                    {data.ceiling.point === null
                      ? "nothing — that ceiling was never sampled"
                      : measured(data.ceiling.point, 3)}
                    , under the target of{" "}
                    {measured(data.target, 3)}. That is a fact about this
                    prompt, not a failure of the search.
                  </>
                )}
              </>
            )}
          </div>

          {data.anchor.length > 0 && (
            <div className="an-tokens">
              {data.anchor.map((t) => (
                <code key={t.index} title={`position ${t.index}`}>
                  {t.token}
                </code>
              ))}
            </div>
          )}

          <ul className="an-scale">
            <Interval
              p={data.precision}
              label="held only the anchor"
              target={data.target}
            />
            <Interval p={data.base_rate} label="held nothing — the floor" />
            {/* "held every candidate", NOT "perturbed nothing". The ceiling
                holds the whole candidate set and still perturbs everything
                outside it — `n_perturbed` says how many, and it was 14 on the
                first run this was read against. Labelling it "perturbed
                nothing" would have told a reader that a ceiling of 0.000 meant
                the model is non-deterministic, which is a different and much
                more alarming claim than the true one. */}
            <Interval
              p={data.ceiling}
              label={
                // `undefined` and `0` are different answers and a falsy test
                // conflates them: the route omits the key entirely on the
                // path where the ceiling was never measured, and sends 0 when
                // it WAS measured with nothing outside the candidate set.
                data.ceiling.n_perturbed === undefined
                  ? "the ceiling"
                  : data.ceiling.n_perturbed > 0
                    ? `held every candidate, ${data.ceiling.n_perturbed} still perturbed — the ceiling`
                    : "held every candidate, nothing left to perturb — the ceiling"
              }
            />
          </ul>
          <p className="meta">
            A precision means nothing without the floor beside it: the marker is
            the {measured(data.target, 2)} target
            {/* `confidence` and `method` are null on an unmeasured proportion,
                and `null * 100` is 0 — so this read "the band is the 0% ."
                with `method` rendering as nothing, a fabricated confidence
                level under a row that correctly said "not measured". */}
            {data.precision.measured && data.precision.confidence !== null ? (
              <>
                , and the band is the{" "}
                {measured(data.precision.confidence * 100, 0)}%{" "}
                {data.precision.method}
              </>
            ) : (
              <>, and there is no band: nothing was sampled here</>
            )}
            .
            {data.target_ceiling < data.target && (
              <>
                {" "}
                <span className="warn">
                  {data.precision.samples} draws can certify at most{" "}
                  {measured(data.target_ceiling, 4)}, which is under this
                  target — raise the draw count.
                </span>
              </>
            )}
          </p>

          <ul className="an-notes">
            <li className="meta">{data.minimality.note}</li>
            {data.candidates.truncated && (
              <li className="meta warn">{data.candidates.coverage}</li>
            )}
            {data.perturbation.quality.below_min_distinct_ids && (
              <li className="meta warn">{data.perturbation.quality.note}</li>
            )}
            <li className="meta">{data.held_fixed.why}</li>
            {/* The floor under the agreement number. A run that drifts from
                itself by as much as the anchor moved it has measured nothing. */}
            <li className="meta">
              The unperturbed run drifts from itself by{" "}
              {measured(data.noise_floor_kl, 5)} nats; the anchored run sits{" "}
              {measured(data.agreement_kl, 5)} from it.
            </li>
          </ul>

          <div className="meta">
            {data.passes} forward passes · {measured(data.elapsed_s, 2)} s · seed{" "}
            {data.seed}
          </div>
          <div className="hint">{data.means}</div>
        </>
      )}
      {stale && (
        <div className="hint">
          This anchor is for position {data?.position}. Search again to read
          position {position}.
        </div>
      )}
    </Disclosure>
    </div>
  );
}
