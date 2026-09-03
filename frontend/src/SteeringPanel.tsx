// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

import { CSSProperties, useCallback, useEffect, useState } from "react";
import Disclosure from "./Disclosure";
import ReceiptLine from "./ReceiptLine";
import RunsOn, { useModelIdentity, useModelReady } from "./RunsOn";
import { measured, scaled } from "./measured";
import { useScanOnData } from "./useScanOnData";
import {
  applySteerDirection,
  clearSteer,
  DirectionFit,
  DirectionLayer,
  errorText,
  fitSteerDirection,
  promptOnce,
  removeSteerDirection,
  SavedDirection,
  SteerCatalogue,
  steerDirections,
  SteerStatus,
  steerStatus,
} from "./api";

/**
 * Steering without a sparse autoencoder — the store, the push, and the null.
 *
 * `modelmri/steer_vectors.py` has been fitting, scoring and persisting
 * contrastive directions since it was written, and the probe panel's "save the
 * best layer's direction as" field has been writing into that same store.
 * Nothing ever read one back. A user could create directions they could not
 * see, apply, or delete. This panel is the reader.
 *
 * THREE RULES IT IS BUILT AROUND, all of them the module's own:
 *
 *   - **A direction is only meaningful against the model it came from.** Every
 *     row is judged against whatever is loaded RIGHT NOW, and one that cannot
 *     be applied says so with the server's exact refusal sentence printed on
 *     the card. Never hidden, never dimmed into invisibility, and never
 *     silently reshaped into something that would steer, plausibly, at random.
 *
 *   - **A direction that did not beat its shuffled null is information, not a
 *     failure.** Difference-of-means always returns a vector, so the honest
 *     reading of "p = 0.44" is "this estimator produces that separation from
 *     these activations regardless of the labels". The badge for it is neutral
 *     with an explainer, not red — a panel that alarms on the ordinary outcome
 *     teaches people to ignore it, and the ordinary outcome here is common.
 *
 *   - **A coefficient is not portable, so it is reported relative to the
 *     stream.** "5.0" means nothing across models or even across layers;
 *     residual norms differ by an order of magnitude between early and late
 *     blocks of one network. What gets applied is constant alpha (standard
 *     CAA, so two runs stay comparable) and what gets SAID is alpha over the
 *     residual norm measured at that layer, with the absolute underneath and
 *     the measurement's own basis a hover away.
 *
 * The fit chart is the centrepiece. It is a signed effect against a null band
 * drawn symmetrically around zero, because `effect` is a standardised
 * separation that can point either way and `null_max` is the worst shuffle in
 * absolute value. A bar that ends inside the band did not clear it, and that
 * is the whole reading.
 */

/** A p-value that can sit several orders below 1 must not print as 0.000000.
 *  Copied from `TokenCounterfactual`, deliberately — one rule for one
 *  quantity, and the two panels print the same kind of number. */
function prob(x: number): string {
  if (x === 0) return "0";
  return Math.abs(x) < 1e-4 ? x.toExponential(2) : measured(x, 6);
}

function lines(text: string): string[] {
  return text
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
}

/** A starting pair that can actually run.
 *
 *  `steer_vectors.MIN_PAIRS` is 8 and half of every set is held out for
 *  scoring, so a tidier-looking default of six would hit its own refusal on
 *  the first click. Twelve, matched line for line: row i of one set is the
 *  pair of row i of the other, which is what the estimator fits from. */
const POSITIVE_DEFAULT = [
  "I would be delighted to help with that.",
  "Thank you so much for asking.",
  "That is a wonderful idea, truly.",
  "It would be my genuine pleasure.",
  "Please do let me know if I can do more.",
  "You have been extremely kind about this.",
  "I am very grateful for your patience.",
  "What a thoughtful thing to suggest.",
  "I really appreciate you raising it.",
  "That is exactly the sort of help I enjoy giving.",
  "Of course — happily, and at once.",
  "I am glad you brought this to me.",
].join("\n");

const NEGATIVE_DEFAULT = [
  "I would rather not bother with that.",
  "Whatever, if you insist on asking.",
  "That is a fairly stupid idea, frankly.",
  "It would be a considerable nuisance.",
  "Please stop asking me to do more.",
  "You have been extremely tiresome about this.",
  "I am very tired of your patience running out.",
  "What a pointless thing to suggest.",
  "I really wish you had not raised it.",
  "That is exactly the sort of help I hate giving.",
  "Fine — grudgingly, and eventually.",
  "I am annoyed you brought this to me.",
].join("\n");

/** The empty state's illustration: a stream of tokens, and a vector nudging
 *  one of them off the line.
 *
 *  Every element carries a CLASS rather than a fill or a stroke, so both
 *  palettes and every contrast tier follow from `styles.css` for free, and
 *  `vectorEffect` keeps the hairlines honest when the box is scaled.
 *  Decorative — the sentence under it carries the meaning. */
function SteerSketch() {
  const tokens = [8, 30, 52, 74, 96, 118];
  return (
    <div className="st-sketch" aria-hidden="true">
      <svg viewBox="0 0 160 70" preserveAspectRatio="xMidYMid meet">
        <line
          className="st-sk-rail"
          x1="4"
          y1="46"
          x2="156"
          y2="46"
          vectorEffect="non-scaling-stroke"
        />
        {tokens.map((x, i) => (
          <rect
            key={x}
            className={`st-sk-tok${i === 3 ? " moved" : ""}`}
            x={x}
            y={i === 3 ? 24 : 38}
            width="18"
            height="16"
            rx="4"
            vectorEffect="non-scaling-stroke"
          />
        ))}
        <path
          className="st-sk-arrow"
          d="M 83 62 L 83 30"
          vectorEffect="non-scaling-stroke"
        />
        <path
          className="st-sk-head"
          d="M 78 34 L 83 26 L 88 34 Z"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
    </div>
  );
}

/** The verdict badge for one saved direction.
 *
 *  Three states, and the middle one is the point. `beats_null` absent means
 *  nothing ever judged this vector — older files carry no verdict — which is
 *  not the same claim as "it was judged and it failed", and printing them the
 *  same way would be fabricating one. */
function NullBadge({ row }: { row: SavedDirection }) {
  if (row.beats_null === undefined) {
    return (
      <span className="st-badge st-unjudged" title="no null was recorded when this was saved">
        null not recorded
      </span>
    );
  }
  const p = typeof row.p_value === "number" ? ` · p=${prob(row.p_value)}` : "";
  if (row.beats_null) {
    return (
      <span className="st-badge st-beat">
        beat its shuffled null{p}
      </span>
    );
  }
  return (
    <span
      className="st-badge st-notbeat"
      title="difference-of-means always returns a direction; this one is what the estimator produces from these activations regardless of the labels"
    >
      did not beat its shuffled null{p}
    </span>
  );
}

/** The headline over the chart, and its denominator is the whole point.
 *
 *  "{survived} of {fit.layers.length} layers beat their own null" counted the
 *  layers that never HAD a null in the population that failed to clear one —
 *  so a 30-layer sweep with a degenerate layer 0 said "12 of 30", telling the
 *  reader 18 layers were tested and lost when 17 were and one was never
 *  entered. That is the same claim the row verdict below had to stop making,
 *  restated in the sentence a reader reads first. The layers with nothing in
 *  them are counted separately and named, because they are a third state and
 *  not a worse version of the second. */
function FitVerdict({ fit }: { fit: DirectionFit }) {
  const nulled = fit.layers.filter((l) => typeof l.p_value === "number").length;
  const none = fit.layers.length - nulled;
  const aside = none > 0 ? ` · ${none} had no direction to test` : "";
  if (fit.best_layer === null) {
    return (
      <div className="st-verdict none">
        <b>Nothing cleared.</b> No layer that had a null beat it, so there is
        no direction here — not a weak one, none. That is a result, and it is
        the one this estimator is built to be able to give.
        {none > 0 && (
          <>
            {" "}
            The other {none} of {fit.layers.length} had no null to beat: the
            two sets have identical mean activations there, so nothing was
            fitted and nothing was scored.
          </>
        )}
      </div>
    );
  }
  return (
    <div className="st-verdict">
      Strongest at <b>layer {fit.best_layer}</b> · {fit.survived} of {nulled}{" "}
      layers beat their own null{aside}
    </div>
  );
}

/** The fit table, drawn. Signed effect, null band around zero, chosen layer lit.
 *
 *  The band is drawn FIRST and behind: it is the thing the bar has to clear,
 *  so it is scenery the bar sits in rather than an annotation beside it. Same
 *  argument, and the same shape, as the probe panel's permutation band. */
function FitChart({ fit }: { fit: DirectionFit }) {
  // The half-width of the axis: whichever is larger, the strongest separation
  // or the worst shuffle. Floored at 0.5 so a fit where everything is tiny
  // does not get magnified into a dramatic-looking chart of nothing.
  const reach = Math.max(
    ...fit.layers.map((l) =>
      // A layer with no direction contributes only its `effect`, which is 0:
      // it has no null, so there is no band of its own for the axis to have
      // to contain. `typeof` rather than `?? 0` so nothing here can quietly
      // start treating an absent measurement as a small one.
      typeof l.null_max === "number"
        ? Math.max(Math.abs(l.effect), l.null_max)
        : Math.abs(l.effect),
    ),
    0.5,
  );
  const x = (v: number) => 50 + (50 * v) / reach;

  return (
    <>
      <div className="st-scale meta" aria-hidden="true">
        <span />
        <span className="st-axis">
          <span>−{measured(reach, 2)}</span>
          <span className="st-axis-mid">0</span>
          <span>+{measured(reach, 2)}</span>
        </span>
        <span />
      </div>
      <ol className="st-rows stagger">
        {[...fit.layers].reverse().map((row: DirectionLayer, i) => {
          const best = row.layer === fit.best_layer;
          // NOTHING WAS SCORED AT THIS LAYER — which is not a weak result and
          // must not be drawn as one. The backend publishes no `p_value` for a
          // layer whose two sets had identical mean activations: there was no
          // direction to project onto and no null to take a quantile of. That
          // ABSENCE is the field to read, not the effect — an effect of
          // exactly 0.0 is a legitimate measurement everywhere else, and the
          // `Math.max(…, 0.4)` below would paint it as a sliver of bar that
          // says "measured, and weak".
          const nothing = typeof row.p_value !== "number";
          // The band's own half-width. Present on every row that was scored,
          // which is every row that reaches the branch drawing it; the 0 is
          // the type's floor and not a reading, and nothing renders it.
          const nullMax = typeof row.null_max === "number" ? row.null_max : 0;
          const from = Math.min(50, x(row.effect));
          const to = Math.max(50, x(row.effect));
          return (
            <li
              key={row.layer}
              className={
                (best ? "st-best " : "") +
                (nothing ? "st-none" : row.beats_null ? "st-clears" : "st-inside")
              }
              style={{ "--i": i } as CSSProperties}
            >
              <span className="mid st-l">L{row.layer}</span>
              <span className="st-track">
                {nothing ? (
                  // The hatch this stylesheet already uses for a step whose
                  // duration was never recorded: there is no honest width for
                  // an unknown, so the difference is carried by the fill and
                  // not by the size. The note is the backend's own sentence.
                  <>
                    <span className="st-nodir" title={row.notes.join(" ")} />
                    <span className="st-zero" />
                  </>
                ) : (
                  <>
                    <span
                      className="st-band"
                      style={{
                        left: `${x(-nullMax)}%`,
                        width: `${x(nullMax) - x(-nullMax)}%`,
                      }}
                      title={`the worst label-shuffled refit reached ${measured(nullMax, 3)}`}
                    />
                    <span className="st-zero" />
                    <span
                      className="st-bar"
                      style={{ left: `${from}%`, width: `${Math.max(to - from, 0.4)}%` }}
                      title={`separation ${measured(row.effect, 3)} on ${row.n_score} held-out pairs`}
                    />
                  </>
                )}
              </span>
              <span className="mid st-eff">
                {nothing ? "—" : measured(row.effect, 2)}
              </span>
              <span className="meta st-verd">
                {/* "inside its null" is the wrong words for a row that never
                    had a null, and it is the sentence a reader would take as
                    a measurement. */}
                {typeof row.p_value === "number"
                  ? row.beats_null
                    ? `p=${prob(row.p_value)}`
                    : "inside its null"
                  : "no direction here"}
              </span>
            </li>
          );
        })}
      </ol>
    </>
  );
}

export default function SteeringPanel({
  epoch,
  prompt,
  onSteering,
}: {
  /** Bumps on every generation, and on load and unload. */
  epoch: number;
  /** The prompt of the last generation, for the A/B. */
  prompt: string;
  /** Raised only while this panel is CHANGING the hook underneath a reader —
   *  its A/B turns steering off, on, and off again in three calls. A direction
   *  the reader deliberately applied is NOT raised: generating under it is the
   *  whole point of applying it, and locking Generate out for that would make
   *  the feature unusable. */
  onSteering?: (active: boolean) => void;
}) {
  const ready = useModelReady(epoch);
  const model = useModelIdentity(epoch);

  const [cat, setCat] = useState<SteerCatalogue | null>(null);
  const [catErr, setCatErr] = useState("");
  const [status, setStatus] = useState<SteerStatus | null>(null);
  const [selected, setSelected] = useState("");
  const [strength, setStrength] = useState(4);
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");
  const [ab, setAb] = useState<{ base: string; steered: string; at: number } | null>(
    null,
  );

  const [positive, setPositive] = useState(POSITIVE_DEFAULT);
  const [negative, setNegative] = useState(NEGATIVE_DEFAULT);
  const [method, setMethod] = useState("caa");
  const [saveAs, setSaveAs] = useState("");
  const [fit, setFit] = useState<DirectionFit | null>(null);
  const [fitErr, setFitErr] = useState("");
  // KEYED ON THE CONTENT, NOT THE OBJECT. `useScanOnData` compares by
  // identity and `refresh` runs on every `epoch` bump — which counts
  // generations — so handing it `cat` scanned the whole panel after each
  // generation whether or not a single row had changed. Its own docstring:
  // "A panel that animates every time React re-runs is not telling you
  // anything — you learn to ignore it."
  const scanRef = useScanOnData(
    cat ? `${cat.model ?? ""}:${cat.directions.map((d) => d.name).join("|")}` : null,
  );

  const refresh = useCallback(async () => {
    try {
      setCat(await steerDirections());
      setCatErr("");
    } catch (e) {
      setCat(null);
      setCatErr(errorText(e));
    }
    try {
      setStatus(await steerStatus());
    } catch {
      // The status is a decoration on top of the catalogue, and the catalogue
      // already reported whatever went wrong. A second copy of one sentence
      // in two places on one panel reads as two problems.
      setStatus(null);
    }
  }, []);

  // A direction belongs to a model, and `epoch` counts GENERATIONS as well as
  // loads — so the identity is what this keys on. `RunsOn` spells out why:
  // `ensureLoaded` ends with `setEpoch(0)`, which is a no-op when epoch is
  // already 0, and "load A, list, load B" left A's rows on screen under B's
  // name.
  useEffect(() => {
    setAb(null);
    setErr("");
    void refresh();
  }, [epoch, model, refresh]);

  const rows = cat?.directions ?? [];
  const chosen = rows.find((r) => r.name === selected) ?? null;
  const active =
    status && status.active && status.kind === "direction" ? status : null;
  // THE OTHER ARM OF THE UNION, NOT DISCARDED. `_steer` and `_steer_dir` are
  // mutually exclusive, so an SAE-feature steer being live means Apply and the
  // A/B below will replace it — and dropping it here left the panel saying
  // nothing is installed while an intervention was running, then uninstalling
  // it without a word. The payload already carries everything needed to say
  // so; what this must NOT do is dress it up as a direction, so it gets a
  // plain line rather than the live band.
  const featureSteer =
    status && status.active && status.kind === "feature" ? status : null;

  /** The relative label for the slider, and where the number came from.
   *
   *  Two possible norms and they are the SAME statistic — the mean L2 norm of
   *  the last-token residual stream entering that layer — taken on two
   *  different samples. The applied one is measured on the prompt in front of
   *  you and wins whenever it exists; before that the fitted one is a preview
   *  and is labelled as one. Neither is invented, and neither is a zero.
   *
   *  ONLY THE NORM COMES FROM THE APPLIED STATE. `strength.relative` is the
   *  APPLIED alpha already divided by it, and this label sits over a slider
   *  whose value is independent — initialised to a default and never synced
   *  back from the status. Reusing the pre-divided quotient printed one
   *  coefficient's relative figure above another coefficient's "alpha +4":
   *  the two halves of one control disagreeing, wrong on first render for
   *  every already-applied direction and wrong again on every drag. The
   *  division happens here, from the number the reader is holding. */
  function relativeLabel(): { headline: string; basis: string } {
    if (active && active.name === selected && active.strength) {
      const s = active.strength;
      if (s.residual_norm !== null && s.residual_norm !== 0) {
        return {
          headline: `${scaled(strength / s.residual_norm)} × residual norm @ L${s.layer}`,
          basis: s.measured,
        };
      }
      return { headline: "relative strength not measured", basis: s.unmeasured };
    }
    if (chosen && typeof chosen.residual_norm === "number" && chosen.residual_norm > 0) {
      return {
        headline:
          `≈ ${scaled(strength / chosen.residual_norm)} × residual norm ` +
          `@ L${chosen.layer ?? "?"}`,
        basis:
          `a preview against the norm measured when this direction was ` +
          `fitted (${measured(chosen.residual_norm, 3)}). Applying it measures ` +
          `the same statistic on the prompt in front of you and replaces this.`,
      };
    }
    return {
      headline: "relative strength not known yet",
      basis:
        "this direction records no residual norm, so there is nothing to " +
        "read the coefficient against until it is applied and the stream is " +
        "measured here.",
    };
  }

  async function apply() {
    if (!selected) return;
    setBusy("apply");
    setErr("");
    try {
      setStatus(await applySteerDirection(selected, strength));
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }

  async function clear() {
    setBusy("clear");
    setErr("");
    try {
      setStatus(await clearSteer());
      setAb(null);
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }

  async function remove(name: string) {
    setBusy(`rm:${name}`);
    setErr("");
    try {
      await removeSteerDirection(name);
      if (selected === name) setSelected("");
      await refresh();
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }

  /** Steered against unsteered, on the prompt that produced this session.
   *
   *  Four rules, all of them `FeaturesPanel`'s and all load-bearing: clear
   *  FIRST so the baseline is really a baseline, both completions with
   *  `commit=false` so two throwaway runs do not rebase every other panel's
   *  analysis target, restore on the catch arm, and lower `onSteering` in
   *  `finally` on every path including the throw — the hook is global to the
   *  runtime and leaving that latched locks Generate out for the session.
   *
   *  It differs from `FeaturesPanel` in one deliberate way: it LEAVES the
   *  direction applied. There the A/B is the whole interaction and the model
   *  has to be left clean; here the reader applied a direction on purpose and
   *  taking it off underneath them would undo the thing they just did. The
   *  indicator above says it is still on, and Clear is one button away. */
  async function runAb() {
    if (!selected || !prompt.trim()) return;
    setBusy("ab");
    setErr("");
    onSteering?.(true);
    try {
      await clearSteer();
      const base = (await promptOnce(prompt, 24, 0, false)).generation;
      const applied = await applySteerDirection(selected, strength);
      const steered = (await promptOnce(prompt, 24, 0, false)).generation;
      setStatus(applied);
      setAb({ base, steered, at: sharedPrefix(base, steered) });
    } catch (e) {
      setErr(errorText(e));
      try {
        setStatus(await clearSteer());
      } catch {
        // Reported already, and a second sentence about the same failed round
        // trip would not tell the reader anything the first did not.
      }
    } finally {
      onSteering?.(false);
      setBusy("");
    }
  }

  async function runFit(estimateOnly: boolean, confirm = false) {
    const pos = lines(positive);
    const neg = lines(negative);
    setBusy(estimateOnly ? "price" : "fit");
    setFitErr("");
    if (!estimateOnly) setFit(null);
    try {
      const out = await fitSteerDirection({
        positive_texts: pos,
        negative_texts: neg,
        method,
        save_as: estimateOnly ? "" : saveAs.trim(),
        estimate_only: estimateOnly,
        confirm,
      });
      setFit(out);
      if (out.ran) await refresh();
    } catch (e) {
      setFitErr(errorText(e));
    } finally {
      setBusy("");
    }
  }

  const nPos = lines(positive).length;
  const nNeg = lines(negative).length;
  const label = relativeLabel();
  const overBudget = fit?.estimate.verdict === "refuse";

  return (
    <div className="panel steer" ref={scanRef}>
      <div className="sect">
        <span className="dot d-steer" />
        <h2 className="h-steer">STEERING — PUSH A DIRECTION AND SEE WHAT MOVES</h2>
        <span className="rule" />
      </div>
      <RunsOn epoch={epoch} needsModel={false} />
      <p className="meta">
        A contrastive direction needs no sparse autoencoder and no training
        run: take pairs of prompts that differ in the property you care about,
        read the residual stream at the last token of each, and the difference
        of the means is a direction. Add it back during generation and the
        model moves. The catch is that <b>this always returns a vector</b>, so
        every direction here is scored against its own label-shuffled null and
        the panel shows you that number rather than a verdict.
      </p>

      {active && (
        <div className="st-live" role="status" aria-live="polite">
          <span className="st-pulse" aria-hidden="true" />
          <span>
            <b className="mid">{active.name}</b> is installed at layer{" "}
            {active.layer ?? "?"} ·{" "}
            {active.strength && active.strength.relative !== null
              ? `${scaled(active.strength.relative)} × the residual norm there`
              : "strength not measured against the stream"}{" "}
            {/* NOT `?? 0`. `measured` takes null and answers "—" precisely so
                this call site does not have to invent a coefficient; on the
                one band that says an intervention is live, "alpha 0.00" reads
                as a push too small to matter. */}
            <span className="meta">(alpha {measured(active.scale, 2)})</span>
            . Every generation in this tab runs through it until you clear it.
          </span>
          <button
            className="ghost sm"
            onClick={() => void clear()}
            disabled={busy !== ""}
          >
            {busy === "clear" ? "clearing…" : "clear"}
          </button>
        </div>
      )}

      {featureSteer && (
        <p className="hint" role="status" aria-live="polite">
          An SAE feature steer is installed right now — feature{" "}
          {/* `?? "?"` and not `?? 0`: this is an identifier, not a
              measurement, and feature 0 is a real feature that must survive
              the fallback. */}
          <b className="mid">{featureSteer.feature_id ?? "?"}</b> at scale{" "}
          {measured(featureSteer.scale, 2)}, from the features panel. There is
          one steering slot, so applying a direction here replaces it.
        </p>
      )}

      {(active?.warnings ?? []).map((w) => (
        <div className="hint warn" key={w}>
          {w}
        </div>
      ))}

      {catErr && <div className="hint err refusal">{catErr}</div>}

      {cat && rows.length === 0 && (
        <div className="st-empty">
          <SteerSketch />
          <p className="hint">
            No directions saved yet. Fit one from contrast pairs below — or
            save one from the <b>Probe</b> panel, whose "save the best layer's
            direction as" field writes into this same store.
          </p>
        </div>
      )}

      {rows.length > 0 && (
        <>
          <p className="meta">
            {cat?.model
              ? `Judged against ${cat.model}, whose residual stream is ${cat.hidden_size}.`
              : "Nothing is loaded, so nothing here has been judged for fit — load a model and these rows will say which of them belong to it."}
          </p>
          <ol className="st-cards stagger">
            {rows.map((row, i) => {
              // `false` ONLY. `compatible: null` means nothing is loaded to
              // judge against, and dimming every card for that would render an
              // unknown as a no — with no sentence under it to say why, since
              // there is no mismatch to report. The line above already says
              // nothing has been judged.
              const blocked = row.compatible === false;
              return (
                <li
                  key={row.name}
                  className={
                    "st-card" +
                    (blocked ? " st-incompatible" : "") +
                    (selected === row.name ? " st-picked" : "")
                  }
                  style={{ "--i": i } as CSSProperties}
                >
                  <button
                    type="button"
                    className="st-pick"
                    onClick={() => setSelected(row.name)}
                    disabled={blocked}
                    aria-pressed={selected === row.name}
                    title={blocked ? row.mismatch : "select this direction"}
                  >
                    <span className="st-name mid">{row.name}</span>
                    <span className="st-chips">
                      {row.layer !== undefined && (
                        <span className="st-chip">L{row.layer}</span>
                      )}
                      {row.method && <span className="st-chip">{row.method}</span>}
                      {row.dims !== undefined && (
                        <span className="st-chip">{row.dims}d</span>
                      )}
                    </span>
                    <span className="meta st-origin">
                      {row.unreadable
                        ? "damaged file"
                        : `fitted on ${row.model || "an unrecorded model"}`}
                      {/* An empty timestamp is UNKNOWN. Rendering it through a
                          date constructor would print 1970 for every row the
                          store wrote before it stamped one. */}
                      {row.saved_at ? ` · ${row.saved_at.slice(0, 10)}` : " · date not recorded"}
                      {typeof row.residual_norm === "number"
                        ? ` · norm ${measured(row.residual_norm, 2)}`
                        : ""}
                    </span>
                    {!row.unreadable && <NullBadge row={row} />}
                  </button>
                  {row.mismatch && (
                    <p className="hint warn st-mismatch">{row.mismatch}</p>
                  )}
                  {row.warnings.map((w) => (
                    <p className="hint warn st-mismatch" key={w}>
                      {w}
                    </p>
                  ))}
                  {row.note && <p className="meta st-note">{row.note}</p>}
                  <button
                    className="ghost sm st-rm"
                    onClick={() => void remove(row.name)}
                    disabled={busy !== ""}
                    title={`delete ${row.name} from this machine`}
                  >
                    {busy === `rm:${row.name}` ? "deleting…" : "delete"}
                  </button>
                </li>
              );
            })}
          </ol>
        </>
      )}

      {chosen && (
        <div className="st-apply">
          <div className="row">
            <label className="meta" htmlFor="st-strength">
              strength
            </label>
            <input
              id="st-strength"
              type="range"
              min={-40}
              max={40}
              step={1}
              value={strength}
              onChange={(e) => setStrength(Number(e.target.value))}
              disabled={busy !== ""}
            />
            <span className="st-strength">
              <b>{label.headline}</b>
              <span className="meta">alpha {strength > 0 ? `+${strength}` : strength}</span>
            </span>
          </div>
          <p className="meta st-basis">{label.basis}</p>
          <div className="row">
            {/* `compatible === false` disables the card's own pick button, and
                leaving these two live for the same row meant the panel refused
                a direction in one place and offered it in another. `selected`
                survives a model swap on purpose — the reader's choice is not
                the model's to discard — so the row it names can become blocked
                underneath it. Three-state to the end: only `false` disables,
                `null` (nothing loaded) is already covered by `ready`. */}
            <button
              className="violet"
              onClick={() => void apply()}
              disabled={busy !== "" || ready === false || chosen.compatible === false}
            >
              {busy === "apply" ? "applying…" : "Apply this direction"}
            </button>
            <button
              className="ghost sm"
              onClick={() => void runAb()}
              disabled={
                busy !== "" ||
                !prompt.trim() ||
                ready === false ||
                chosen.compatible === false
              }
              title={
                prompt.trim()
                  ? "Same prompt, greedy decoding, once clean and once steered"
                  : "Generate in this tab first — the A/B needs the prompt"
              }
            >
              {busy === "ab" ? "Running A/B…" : "Run steering A/B"}
            </button>
            {active && (
              <button
                className="ghost sm"
                onClick={() => void clear()}
                disabled={busy !== ""}
              >
                Clear
              </button>
            )}
          </div>
        </div>
      )}

      {err && <div className="hint err refusal">{err}</div>}

      {ab && (
        <>
          <div className="compare st-ab">
            <div className="card">
              <span className="lbl">BASELINE</span>
              {ab.base}
            </div>
            <div className="card steered">
              <span className="lbl">
                {selected.toUpperCase()} @ {strength > 0 ? `+${strength}` : strength}
              </span>
              {ab.at > 0 && <span className="st-shared">{ab.steered.slice(0, ab.at)}</span>}
              {ab.steered.slice(ab.at)}
            </div>
          </div>
          <p className="meta">
            {ab.at > 0
              ? `The two completions share their first ${ab.at} characters — dimmed above — and come apart after that. Compared as TEXT, because the browser has no tokenizer: the split is where the strings differ, which is at or after the token where the intervention first changed the argmax.`
              : "The two completions differ from their first character."}
          </p>
        </>
      )}

      <Disclosure
        dot="d-steer"
        title="FIT A NEW DIRECTION"
        asks="Where in this model do two sets of your own sentences come apart, and does that beat shuffling their labels?"
        hasResult={fit !== null}
        disabled={ready === false}
      >
        <p className="meta">
          Matched pairs, one per line: row <i>i</i> of one set is the pair of
          row <i>i</i> of the other, because the direction is fitted from their
          differences. Half of them are held out for scoring — a direction
          scored on the pairs it was fitted from separates them by
          construction.
        </p>
        <div className="probe-inputs">
          <label>
            <span className="meta">
              positive — {nPos} line{nPos === 1 ? "" : "s"}
            </span>
            <textarea
              value={positive}
              onChange={(e) => setPositive(e.target.value)}
              spellCheck={false}
              rows={7}
            />
          </label>
          <label>
            <span className="meta">
              negative — {nNeg} line{nNeg === 1 ? "" : "s"}
            </span>
            <textarea
              value={negative}
              onChange={(e) => setNegative(e.target.value)}
              spellCheck={false}
              rows={7}
            />
          </label>
        </div>

        <div className="row st-fitrow">
          <label className="meta" htmlFor="st-method">
            estimator
          </label>
          <select
            id="st-method"
            value={method}
            onChange={(e) => setMethod(e.target.value)}
          >
            <option value="caa">caa — difference of means</option>
            <option value="repe">repe — first component of the differences</option>
          </select>
          <label className="meta" htmlFor="st-save">
            save as
          </label>
          <input
            id="st-save"
            className="probe-name"
            value={saveAs}
            onChange={(e) => setSaveAs(e.target.value)}
            placeholder="optional — a steerable name"
            spellCheck={false}
          />
        </div>
        <p className="meta">
          Every layer is fitted, because which one carries a property is the
          thing being measured rather than a setting. Saving keeps the best
          surviving layer's vector; if no layer beats its null, nothing is
          saved and the server says why.
        </p>

        <div className="row">
          <button
            className="ghost sm"
            onClick={() => void runFit(true)}
            disabled={busy !== "" || ready === false}
          >
            {busy === "price" ? "measuring one pass…" : "What would this cost?"}
          </button>
          <button
            className="cta"
            onClick={() => void runFit(false, overBudget)}
            disabled={busy !== "" || ready === false || !nPos || !nNeg}
          >
            {busy === "fit"
              ? "Fitting every layer…"
              : overBudget
                ? "Fit it anyway"
                : "Fit the direction"}
          </button>
        </div>
        {/* The override, offered from the projection rather than from parsing
            a refusal's words. `budget.TooCostly` is deliberately overridable —
            unlike free disk, free VRAM can be made by closing something — so
            the button changes rather than disappearing, and the sentence
            beside it is the estimate's own. */}
        {overBudget && (
          <p className="hint warn">
            The projection says this needs more than this accelerator has
            free, so it would probably run out partway through. Close
            something on the GPU, use fewer pairs, or run it anyway if you
            know the estimate is pessimistic — it is built from one probe
            pass and runs low by design.
          </p>
        )}

        {fitErr && <div className="hint err refusal">{fitErr}</div>}

        {fit && (
          <>
            <p className="meta st-price">
              {fit.passes} forward passes — one per prompt, both sets — over{" "}
              {fit.n_pairs} pairs.{" "}
              {fit.estimate.seconds !== null ? (
                <>
                  About <b>{measured(fit.estimate.seconds, 1)}s</b> here, from{" "}
                  {fit.estimate.basis}.
                </>
              ) : (
                <>
                  The time could not be projected: {fit.estimate.unmeasured || "no probe reading"}.
                </>
              )}
              {fit.estimate.verdict === "tight" && (
                <b> That is a large share of what is free on this accelerator.</b>
              )}
            </p>

            {fit.ran ? (
              <>
                <FitVerdict fit={fit} />
                <FitChart fit={fit} />
                <p className="meta">
                  The translucent band is where the same estimator landed with
                  the labels shuffled, drawn symmetrically because the null is
                  measured in absolute separation. A bar ending inside it is
                  what this pipeline produces from these activations regardless
                  of the labels. A hatched row had no band and no bar: its two
                  sets have identical mean activations at that layer, so
                  nothing was fitted and nothing was scored there. With eight
                  refits the smallest attainable p-value is 0.111, so <b>this
                  is a screen and not a significance test</b> — measured on
                  structureless data, 16% of what <code>caa</code> reports as
                  real is noise.
                </p>
                {fit.layers
                  /* The best layer's notes, AND every note from a layer with
                     nothing in it. A degenerate row is `beats_null: false` by
                     construction and so can never be the best layer, which
                     left the sentence the backend writes about it — why layer
                     0 is the token's own embedding, why its p-value is absent
                     rather than zero — reachable only as the `title` on a
                     hatched span: mouse-only, not keyboard-reachable, not in
                     the accessible name, invisible on touch. Every writer
                     needs a reader. Each note names its own layer, so several
                     of them read correctly in sequence. */
                  .filter(
                    (l) =>
                      l.layer === fit.best_layer ||
                      typeof l.p_value !== "number",
                  )
                  .flatMap((l) => l.notes)
                  .map((n) => (
                    <p className="hint warn" key={n}>
                      {n}
                    </p>
                  ))}
                {fit.saved && (
                  <p className="meta">
                    {fit.saved.replaced ? "Replaced" : "Saved"}{" "}
                    <b>{fit.saved.name}</b> — {fit.saved.dims} dimensions, in
                    the store above and in the same space this panel pushes
                    through.
                    {fit.saved.replaced &&
                      " A direction was already stored under that name; this one is now in its place."}
                  </p>
                )}
                <p className="meta">{fit.means}</p>
                <ReceiptLine receipt={fit.receipt} />
              </>
            ) : (
              <p className="meta">
                Nothing has been fitted yet — that is the price, measured on
                this machine. Fit the direction to spend it.
              </p>
            )}
          </>
        )}
      </Disclosure>
    </div>
  );
}

/** How many leading characters two completions share.
 *
 *  Deliberately characters and not tokens, and the caption says so. The
 *  browser has no tokenizer, and calling a character offset a token index
 *  would be a fabricated unit on a panel whose whole argument is that units
 *  travel. */
function sharedPrefix(a: string, b: string): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i += 1;
  return i;
}
