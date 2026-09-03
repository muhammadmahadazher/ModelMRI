// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

import { useEffect, useState } from "react";
import { errorText, TokenGradients as Grads, tokenGradients } from "./api";
import { measured } from "./measured";
import Disclosure from "./Disclosure";

/** What the answer was SENSITIVE to, with the gap that says whether the bars
 *  add up to anything.
 *
 *  NOT a causal measurement, and this is the one place in the tool where that
 *  distinction has to be read rather than assumed. A gradient says what the
 *  output was sensitive to in the limit of an infinitesimal nudge; "Rank
 *  tokens" beside it says what happens when a token is actually removed.
 *  They disagree, both are real, and a reader shown one of them alone will
 *  read it as the other.
 *
 *  THE VERDICT COMES BEFORE THE BARS, on purpose. Integrated gradients is an
 *  approximation of an integral, and the completeness gap is the error in
 *  that approximation. A bar chart with no gap beside it cannot be told from
 *  a converged one — so the gap is rendered first, at the top, and a bar
 *  under it is struck through rather than drawn as a small measurement.
 */
/** Bars drawn at once. The route ranks by absolute attribution and caps its
 *  own listing; this caps the picture, because a 248-row chart is scrolled
 *  past rather than read, and the rows past this one are the smallest. */
const MAX_BARS = 40;

/** Where the path starts. NOT a detail: MEASURED on Qwen3-1.7B over a
 *  232-token generation at 32 steps, the completeness gap came back
 *
 *      zero   99.9% of the move   diverged      0 of 232 bars readable
 *      pad   213.0% of the move   diverged      0 of 232 bars readable
 *      mean   21.8% of the move   approximate   2 of 232 bars readable
 *
 *  — so on a real prompt the baseline decides whether this measurement says
 *  anything at all, and a panel offering only a step count would leave a
 *  reader turning the dial that does not move. The route's own default is
 *  `zero` and this matches it rather than quietly picking the flattering
 *  one; the difference is on screen instead. */
const BASELINES = ["zero", "pad", "mean"];

const VERDICT_SAYS: Record<string, string> = {
  converged: "the bars add up to the move they claim to explain",
  approximate: "the bars are short by their share of the gap",
  diverged: "the bars do NOT add up to what happened, so they are not a decomposition of it",
  undefined: "the check could not be scored",
};

export default function TokenGradients({
  position,
  epoch,
  disabled,
}: {
  /** The pinned token, or `-1` for the route's own default (the last prompt
   *  token). Passed rather than chosen here so this and "Rank tokens" are
   *  always answering about the same position. */
  position: number;
  /** Bumped on every generation. The payload below is about ONE of them. */
  epoch: number;
  disabled?: boolean;
}) {
  const [steps, setSteps] = useState(32);
  const [baseline, setBaseline] = useState(BASELINES[0]);
  const [data, setData] = useState<Grads | null>(null);
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
    setShowAll(false);
  }, [epoch]);
  const [showAll, setShowAll] = useState(false);

  async function run() {
    if (busy) return;
    setBusy(true);
    setErr("");
    setShowAll(false);
    try {
      setData(
        await tokenGradients({
          position: position >= 0 ? position : undefined,
          steps,
          baseline,
          // REPORT, not refuse. The refusal is right for a caller that will
          // draw the bars regardless; this panel renders the verdict above
          // them and strikes through the ones under the gap, so a diverged
          // run is worth showing — it is the answer.
          on_gap: "report",
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
  const peak = data
    ? Math.max(
        ...data.tokens
          .map((t) => t.attribution)
          .filter((v): v is number => v !== null && Number.isFinite(v))
          .map(Math.abs),
        0,
      )
    : 0;
  // The route already ranks by absolute attribution and caps the listing; this
  // caps the DRAWING, because a 248-row chart is not read, it is scrolled past.
  const shown = data ? data.tokens.slice(0, MAX_BARS) : [];
  // `n_unreadable` is counted over ALL `n_tokens`, not over the listed rows,
  // so comparing it against `tokens.length` would call a run "nothing to
  // read" the moment the route started capping its own listing.
  const allUnreadable =
    data !== null && data.n_tokens > 0 && data.n_unreadable >= data.n_tokens;

  return (
    <div className="token-gradients">
      <Disclosure
        dot="d-attn"
        title="WHAT IT WAS SENSITIVE TO"
        asks="What was the output sensitive to, in the limit of an infinitesimal nudge? A gradient, not a removal — a different claim from ranking."
        hasResult={data !== null}
      >

      <div className="row">
        <label className="meta" htmlFor="ig-steps">
          steps
        </label>
        <input
          id="ig-steps"
          type="range"
          min={4}
          max={128}
          step={4}
          value={steps}
          onChange={(e) => setSteps(Number(e.target.value))}
          disabled={disabled || busy}
        />
        <span className="meta">
          {steps} · {steps} forward and {steps} backward passes
        </span>
        <label className="meta" htmlFor="ig-baseline">
          from
        </label>
        <select
          id="ig-baseline"
          className="combo"
          value={baseline}
          onChange={(e) => setBaseline(e.target.value)}
          disabled={disabled || busy}
        >
          {BASELINES.map((b) => (
            <option key={b} value={b}>
              {b}
            </option>
          ))}
        </select>
        <button
          className="ghost sm"
          onClick={() => void run()}
          disabled={disabled || busy}
        >
          {busy ? "integrating…" : data ? "attribute again" : "attribute"}
        </button>
      </div>

      {err && <div className="hint err">{err}</div>}

      {data && !stale && (
        <>
          {/* THE GATE. Before a single bar. */}
          <div className={`ig-verdict ${data.completeness.verdict}`}>
            <b>{data.completeness.verdict.toUpperCase()}</b> —{" "}
            {VERDICT_SAYS[data.completeness.verdict] ?? ""}
            <div className="meta">{data.completeness.sentence}</div>
          </div>

          <div className="row ig-chips">
            <span className="pill">
              predicted <code>{data.target_token}</code> at position{" "}
              {data.position}
            </span>
            <span className="meta">{data.baseline_note}</span>
            {data.n_unreadable > 0 && (
              <span className="meta warn">
                {data.n_unreadable} of {data.n_tokens} bars are under the gap
              </span>
            )}
            {data.n_nonfinite > 0 && (
              <span className="meta warn">
                {data.n_nonfinite} attribution(s) came back non-finite and are
                marked unreadable rather than drawn as zero
              </span>
            )}
          </div>

          {/* EVERY BAR UNDER THE GAP IS NOT A CHART. MEASURED on Qwen3-1.7B
              at 32 steps over a 248-token generation: the gap came back at
              99.90% of the move and all 248 bars were unreadable. Drawing 248
              struck-through rows says "here is the answer, and it is crossed
              out", which reads as a rendering fault rather than as the
              finding. So the chart is folded away behind the sentence that
              names the fix, and a reader who wants to see it anyway can. */}
          {allUnreadable && (
            <div className="ig-nothing">
              <p className="meta">
                Every one of the {data.n_listed} listed bars is under the gap,
                so there is nothing here to read: at {data.completeness.steps}{" "}
                steps from the <code>{data.baseline}</code> baseline, the error
                in the approximation is larger than any token's share of the
                move. More steps shrink an error that comes from the step
                count — a gap near the whole move usually does not. That one
                says the baseline sits somewhere the model's gradients say
                little about this prompt, which is the other dial above.
              </p>
              <button className="ghost sm" onClick={() => setShowAll((v) => !v)}>
                {showAll ? "hide the bars" : "show them anyway"}
              </button>
            </div>
          )}

          {/* Signed bars about a zero line. A token that pushed the answer
              AWAY is not a small positive one, and a shared centre is the
              only rendering where that reads at a glance. */}
          {(!allUnreadable || showAll) && (
          <ul className="ig-bars">
            {shown.map((t) => {
              // `null` is "the backward pass came back non-finite", not zero.
              // `t.attribution < 0` is false for null, so the old code drew a
              // zero-width POSITIVE bar — contradicting the chip above it,
              // which promises non-finite values are "marked unreadable
              // rather than drawn as zero".
              const value = t.attribution;
              const known = value !== null && Number.isFinite(value);
              const share = known && peak > 0 ? Math.abs(value) / peak : 0;
              return (
                <li
                  key={t.index}
                  className={t.unreadable ? "unreadable" : undefined}
                  title={
                    !known
                      ? "the backward pass came back non-finite here, so there is no attribution to draw"
                      : t.unreadable
                        ? "under the completeness gap — this bar cannot be told from the error in the approximation"
                        : undefined
                  }
                >
                  <span className="ig-index meta">{t.index}</span>
                  <code className="ig-token">{t.token}</code>
                  <span className="ig-track">
                    {known && (
                      <i
                        className={value < 0 ? "neg" : "pos"}
                        style={{ width: `${(share * 50).toFixed(2)}%` }}
                      />
                    )}
                  </span>
                  <b>{known ? measured(value, 4) : "not a number"}</b>
                </li>
              );
            })}
          </ul>
          )}
          {shown.length < data.tokens.length && (!allUnreadable || showAll) && (
            <p className="meta">
              The {shown.length} largest of {data.tokens.length} listed, by
              absolute attribution — the cap is on this chart, and every token
              is in the sums above.
            </p>
          )}

          {/* `peak_bytes` is null on EVERY CPU run — `torch.cpu` publishes no
              `max_memory_allocated`. `null / 1e6` is 0, so this printed
              "0.0 MB held" immediately before a `peak_note` reading "there is
              no allocator peak to read on cpu, so the memory this run needed
              was not measured": the same line stating a figure and then
              saying nobody took it. */}
          <div className="meta ig-cost">
            {data.forward_passes} forward · {data.backward_passes} backward
            passes · {measured(data.elapsed_s, 2)} s ·{" "}
            {data.peak_bytes === null
              ? data.peak_note
              : `${(data.peak_bytes / 1e6).toFixed(1)} MB held. ${data.peak_note}`}
          </div>
          <div className="hint">{data.means}</div>
        </>
      )}
      {stale && (
        <div className="hint">
          These bars are for position {data?.position}. Attribute again to read
          position {position}.
        </div>
      )}
    </Disclosure>
    </div>
  );
}
