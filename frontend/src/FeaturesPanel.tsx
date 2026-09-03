// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

import LensPanel from "./LensPanel";
import { fmtKL, pair, scaled, percent } from "./measured";
import { useEffect, useRef, useState } from "react";
import RunsOn from "./RunsOn";
import { useScanOnData } from "./useScanOnData";
import {
  errorText,
  fastestRate,
  FeatureAblation,
  FeatureScore,
  FeaturesSummary,
  getFeatureDetail,
  getFeaturesSummary,
  getSAE,
  getSAEOptions,
  humanSeconds,
  loadSAEFrom,
  promptOnce,
  rankFeatures,
  SAEOption,
  SAEStatus,
  setSteer,
} from "./api";
import ReceiptLine from "./ReceiptLine";
import FeatureEvidencePanel from "./FeatureEvidence";
import NeuronEvidencePanel from "./NeuronEvidencePanel";
import SaeFidelityCard from "./SaeFidelity";
import { DEMO } from "./demo";
import { VIEWER } from "./viewer";

interface Props {
  epoch: number; // bumps after each generation
  prompt: string; // the prompt of the last generation, for steering A/B
  /** The loaded model's dtype, from /api/session. The feature ranking refuses
   *  anything but float32 and ModelMRI picks bfloat16 for every NVIDIA GPU, so
   *  on the machine this project is developed on the control would render,
   *  quote a cost, and answer 409 — the panel's own argument against a button
   *  that can only fail, applied to the dtype instead of to the build. */
  dtype: string | null;
  /** Raised while the steering hook is installed on the model. Generate must
   *  be locked out for that window: the hook is global to the runtime, so a
   *  generation started mid-A/B comes back steered with nothing on screen
   *  saying so. */
  onSteering?: (active: boolean) => void;
}

/** SAE feature browser: token -> top features -> heat view -> steering A/B. */
/** Where the default SAE reads. Overridden per registry entry. */
const DEFAULT_HOOK = "blocks.8.hook_resid_pre";

/** FVU spans five orders of magnitude between a working SAE and a wrong one
 *  (0.0010 against 13579.24 on the default release), so one fixed number of
 *  decimals is either noise or a row of zeroes. */
const fmtFVU = scaled;

/** Passes a feature ranking spends on top of TWO per tested feature.
 *
 *  Mirrors `feature_ablate.rank_features`' own accounting: the base pass, a
 *  plain replay, the write-back floor, one joint ablation of everything
 *  tested, one substitution of the SAE's reconstruction over the window the
 *  edits use, and one check that the edit landed exactly.
 *
 *  Two per row rather than one because every score is paired with a control —
 *  the same tokens edited by a random direction of the same norm. Checked
 *  against real runs rather than counted off the source: 43 features tested
 *  came back as 92 passes, 256 as 518.
 */
const PASS_OVERHEAD = 6;
const PASSES_PER_FEATURE = 2;

/** How many rows of the causal ranking are printed. Every tested feature is
 *  still annotated in the activation list above, so nothing measured is
 *  hidden — this only caps how far down a 256-row list the panel reads out. */
const SHOWN_CAUSAL = 10;

export default function FeaturesPanel({
  epoch,
  prompt,
  dtype,
  onSteering,
}: Props) {
  const scanRef = useScanOnData(epoch);
  const [sae, setSae] = useState<SAEStatus | null>(null);
  const [busy, setBusy] = useState("");
  const [summary, setSummary] = useState<FeaturesSummary | null>(null);
  const [tokenSel, setTokenSel] = useState(-1);
  const [featSel, setFeatSel] = useState(-1);
  const [heat, setHeat] = useState<number[] | null>(null);
  const [peak, setPeak] = useState(-1);
  const peakRef = useRef<HTMLSpanElement>(null);
  const [scale, setScale] = useState(-40);
  const [ab, setAb] = useState<{ base: string; steered: string } | null>(null);
  const [err, setErr] = useState("");
  // What removing a feature actually does to the answer. Null means the
  // question has not been asked — which is a different state from "asked and
  // nothing came back", and the two must never render the same way.
  const [fab, setFab] = useState<FeatureAblation | null>(null);
  const [fabBusy, setFabBusy] = useState(false);
  // Its own error slot, not the panel's `err`. What this endpoint refuses with
  // — "this SAE does not reconstruct the stream it is attached to", "no
  // feature fires at position N", "the write-back is not landing where the
  // capture came from" — are the feature, and they already say what would make
  // the measurement work. The shared slot sits under a steering hint and would
  // read as advice about steering.
  const [fabErr, setFabErr] = useState("");
  const [fabScope, setFabScope] = useState<"position" | "prompt">("position");
  // Seconds per forward pass, measured on THIS model by THIS measurement.
  // Null until one ranking has run — an estimate before that would be a number
  // we made up. The rule (fastest rate seen, not latest) lives in api.ts so
  // the two panels that need it cannot drift apart.
  const [secPerPass, setSecPerPass] = useState<number | null>(null);
  // Which SAEs exist for the model that is loaded. Empty is the common,
  // honest answer — an SAE is trained per model and public ones exist for
  // only a handful; `catalogue` below is the ones this build knows.
  const [opts, setOpts] = useState<{
    model: string | null;
    matching: SAEOption[];
    usable: SAEOption[];
    catalogue: SAEOption[];
  } | null>(null);
  const [custom, setCustom] = useState("");

  useEffect(() => {
    void getSAE().then(setSae);
  }, []);

  useEffect(() => {
    setSummary(null);
    setTokenSel(-1);
    setFeatSel(-1);
    setHeat(null);
    setPeak(-1);
    setAb(null);
    // A ranking is a claim about one position in ONE sequence, and the
    // per-pass rate belongs to whichever model produced it.
    setFab(null);
    setFabErr("");
    setSecPerPass(null);
    if (sae?.loaded) void refreshSummary();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [epoch, sae?.loaded]);

  // Moving the attribution position invalidates the ranking outright. Token 7's
  // scores under token 9's feature list would be the panel's own version of the
  // mistake it exists to catch: a plausible ordering attached to the wrong
  // question. `secPerPass` survives — it is a property of the model and the
  // machine, not of the position.
  useEffect(() => {
    setFab(null);
    setFabErr("");
  }, [tokenSel]);

  async function refreshSummary() {
    try {
      setSummary(await getFeaturesSummary(8));
      setErr("");
    } catch (e) {
      setErr(errorText(e));
    }
  }

  useEffect(() => {
    let live = true;
    void getSAEOptions()
      .then((o) => live && setOpts(o))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [epoch]);

  async function onLoadFrom(repo: string, hook: string) {
    setBusy("sae");
    setErr("");
    try {
      setSae(await loadSAEFrom(repo, hook));
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }


  async function onPickFeature(fid: number) {
    setFeatSel(fid);
    setAb(null);
    try {
      const d = await getFeatureDetail(fid);
      const max = d.max || 1;
      setHeat(d.activations.map((a) => a / max));
      // The API already tells us which token fires hardest. A default
      // generation is 256 tokens, so without this the one chip worth looking
      // at is somewhere in a strip several thousand pixels wide.
      setPeak(d.argmax);
      requestAnimationFrame(() =>
        peakRef.current?.scrollIntoView({ block: "nearest", inline: "center" }),
      );
    } catch (e) {
      setErr(errorText(e));
    }
  }

  /** Ask what removing each feature does, at the token the user picked.
   *
   *  The position is not a detail around the measurement, it IS the
   *  measurement's scope — so it is always `tokenSel`, always the token whose
   *  activation list is on screen, and never the server's default. There is no
   *  version of this where the position is implicit and the claim stays true.
   */
  async function onRankFeatures() {
    if (tokenSel < 0) return;
    setFabBusy(true);
    setFabErr("");
    try {
      const result = await rankFeatures(tokenSel, fabScope);
      setFab(result);
      setSecPerPass((prev) => fastestRate(prev, result.elapsed_s, result.passes));
    } catch (e) {
      setFabErr(errorText(e));
      // A refusal replaces the previous answer rather than sitting beside it.
      setFab(null);
    } finally {
      setFabBusy(false);
    }
  }

  async function onSteerTest() {
    if (featSel < 0) return;
    // The A/B re-runs the prompt that produced this analysis. After a reload
    // the panels are restored from the server, which keeps the activations
    // but not the prompt text — running the A/B on "" would compare two
    // completions of nothing and present them as a steering result.
    if (!prompt.trim()) {
      setErr(
        "Generate once in this tab first — the A/B re-runs your prompt, and " +
          "this analysis was restored from the server without it.",
      );
      return;
    }
    setBusy("steer");
    setErr("");
    onSteering?.(true);
    try {
      await setSteer(null);
      const base = (await promptOnce(prompt, 24, 0, false)).generation;
      await setSteer(featSel, scale);
      const steered = (await promptOnce(prompt, 24, 0, false)).generation;
      await setSteer(null); // always leave the model clean
      setAb({ base, steered });
    } catch (e) {
      setErr(errorText(e));
      await setSteer(null);
    } finally {
      // Must pair with the raise above on EVERY path, including the throw:
      // the hook is global to the runtime, so leaving this latched locks
      // Generate out for the rest of the session.
      onSteering?.(false);
      setBusy("");
    }
  }

  if (!sae) return null;

  if (!sae.loaded) {
    return (
      <div ref={scanRef} className="panel feat">
        <div className="sect">
          <span className="dot d-feat" />
          <h2 className="h-feat">FEATURES — THE CONCEPTS INSIDE</h2>
          <span className="rule" />
        </div>
          <RunsOn epoch={epoch} />
        {opts?.usable.length ? (
          <>
            {opts.usable.map((o) => (
              <div className="row" style={{ marginTop: 12 }} key={o.repo}>
                <button
                  className="violet"
                  onClick={() => void onLoadFrom(o.repo, o.default_hook)}
                  disabled={busy !== ""}
                >
                  {busy === "sae" ? "Loading SAE…" : `Load ${o.label}`}
                </button>
                <span className="meta">
                  matches {opts.model} · d_in {o.d_in} · first run downloads it
                </span>
              </div>
            ))}
          </>
        ) : opts?.matching.length ? (
          /* REGISTERED, AND THIS BUILD CANNOT OPEN IT. A different situation
             from "none exists", with a different next step, and the branch
             below said the false one: "No sparse autoencoder exists for
             gemma-2-9b" — two lines above "Known SAEs: gemma-scope-9b-pt-res",
             which is the release registered for exactly that model.

             `matching` minus `usable` is non-empty here by construction: every
             entry registered for this model is unsupported. Each carries the
             registry's own `note`, and those notes are not interchangeable —
             one says the layout has never been run against this model and is
             expected to work, the other says the layout cannot be read at all.
             Printing one sentence for both would put the reader back where
             this branch found them. */
          <div className="resting-empty">
            <b>
              {opts.matching.length === 1 ? "An SAE is" : `${opts.matching.length} SAEs are`}{" "}
              registered for {opts.model}, and this build cannot open{" "}
              {opts.matching.length === 1 ? "it" : "them"} yet.
            </b>
            <ul className="feat-unsupported">
              {opts.matching.map((m) => (
                <li key={m.repo}>
                  <code>{m.repo}</code> — {m.note || "no reason was recorded."}
                </li>
              ))}
            </ul>
            The box below takes a repo id directly, so one marked{" "}
            <em>expected to work but not measured here</em> is still worth a
            try — it will either load or refuse with the dimension it found.
            The logit lens further down asks a different question of the same
            residual stream and works on every model.
          </div>
        ) : (
          <div className="resting-empty">
            <b>No sparse autoencoder exists for {opts?.model ?? "this model"}.</b>{" "}
            An SAE is trained against one model at one layer — it is GPU-months
            of someone else's work, not a setting. Public ones exist for only
            a handful of models.
            {opts?.catalogue.length ? (
              <> Known SAEs: {opts.catalogue.map((c) => c.repo.split("/")[1]).join(", ")}.</>
            ) : null}{" "}
            The logit lens below asks a different question of the same
            residual stream, and works on every model.
          </div>
        )}

        <div className="row cand-manual">
          <input
            className="combo grow"
            placeholder="…or a SAELens repo: owner/name"
            value={custom}
            onChange={(e) => setCustom(e.target.value)}
          />
          <button
            className="ghost sm"
            disabled={busy !== "" || !custom.trim()}
            onClick={() =>
              void onLoadFrom(custom.trim(), DEFAULT_HOOK)
            }
          >
            Load
          </button>
        </div>
        <div className="hint">
          Any SAE is accepted, and refused if its d_in does not match the
          model — a mismatched one would produce confident features describing
          a different network.
        </div>
        {err && <div className="hint err">{err}</div>}

        {/* THIS BRANCH IS THE ONE IT EXISTS FOR. Everything above needs a
            published sparse autoencoder in the registry, and this branch is
            reached precisely because there is none — which is the state most
            models are in. Reading the MLP neurons directly needs no registry
            at all, so "no SAE for this model" stops being a dead end and
            becomes a blunter answer that says it is blunter. */}
        {!DEMO && !VIEWER && <NeuronEvidencePanel epoch={epoch} />}

        <LensPanel epoch={epoch} />
      </div>
    );
  }

  // Absent until the first encode, because it is measured rather than read.
  const cal = sae.calibration ?? null;

  // The server's float32 gate, read on this side so the control is absent
  // rather than broken. `null` is "we do not know yet" — /api/session has not
  // answered — and an unknown dtype is not a reason to hide a control, so the
  // button renders and the runtime gets the last word.
  const rankable = dtype === null || dtype === "float32";

  // ---- what a ranking would cost, and whether we are entitled to say -------
  // The rule is AttentionPanel's, imported rather than re-derived: passes x the
  // fastest per-pass rate measured on this model, and no estimate at all until
  // there has been a measurement to extrapolate from.
  const measured =
    fab && fab.position === tokenSel && fab.scope === fabScope ? fab : null;
  // At "position" scope the candidate count is the number of features firing at
  // ONE token, which the panel cannot read — the summary carries only the top
  // 8. The SAE's calibrated mean L0 is the best available stand-in and it is a
  // mean, not this token's count — per-token counts spread widely around it.
  // So this is an order-of-magnitude answer to "seconds or minutes", the
  // tooltip says exactly that, and the result reports what it really spent.
  //
  // At "prompt" scope there is no such stand-in — the candidate set is the
  // UNION over every token up to this one, which collapses by an amount only
  // the SAE knows (666 firings to 494 distinct features on that prompt) and is
  // then capped server-side. So no number is offered until a run has produced
  // one, and the sentence beside the control says why rather than leaving a
  // silent gap where a cost used to be.
  const estPasses = measured
    ? measured.passes
    : fabScope === "position" && cal
      ? PASSES_PER_FEATURE * Math.round(cal.l0) + PASS_OVERHEAD
      : null;
  const costBadge =
    estPasses === null
      ? ""
      : secPerPass !== null
        ? `≈ ${humanSeconds(estPasses * secPerPass)}`
        : `≈ ${estPasses} passes`;
  const passesNote =
    `${PASSES_PER_FEATURE} forward passes per feature tested — its own edit ` +
    `and one random direction of the same size at the same tokens, without ` +
    `which a score cannot be told apart from the size of the edit — plus ` +
    `${PASS_OVERHEAD}: the base pass, a plain replay, the write-back floor, ` +
    `one joint ablation of everything tested, one substitution of the SAE's ` +
    `own reconstruction over the tokens the edits land in, and one check that ` +
    `the edit landed exactly.`;
  const rankTitle =
    fabScope === "position"
      ? `${passesNote} Candidates are the features firing at token ${tokenSel}.` +
        (cal
          ? ` The estimate uses this SAE's mean of ${cal.l0.toFixed(1)} features ` +
            `per token over the ${cal.n_tokens} tokens it was calibrated on — ` +
            `this token's own count is what decides the real cost, and it can ` +
            `sit well either side of that mean.`
          : "") +
        (estPasses !== null && secPerPass !== null
          ? ` About ${humanSeconds(estPasses * secPerPass)} on this model.`
          : "")
      : `${passesNote} Candidates are every feature firing at ANY token up to ` +
        `${tokenSel}, each removed everywhere it fires — a different question ` +
        `from the list above, and a much larger one.`;

  return (
    <div ref={scanRef} className="panel feat">
      <div className="sect">
        <span className="dot d-feat" />
        <h2 className="h-feat">FEATURES — THE CONCEPTS INSIDE</h2>
        <span className="rule" />
      </div>
      <div className="row" style={{ margin: "10px 0" }}>
        <span className="pill violet">
          {sae.repo?.split("/")[1]} · L{sae.layer} · {sae.d_sae?.toLocaleString()} features
        </span>
        <span className="meta">
          click a token → its top features · click a feature → heat + steering
        </span>
      </div>

      {/* An SAE fed the wrong activation convention does not error. It returns
          features, in the right shape, with plausible magnitudes, for a vector
          it never saw — which is exactly what this panel used to plot. So the
          reconstruction it achieved against THIS model is stated before any
          feature is, and every number here is the server's: the panel does not
          own the threshold that decides whether a measurement can be trusted. */}
      {/* "measured", not "checked". `usable` is `fvu < 1.0`, which only rules
          out an SAE that carries less than a constant vector would — an SAE at
          FVU 0.83 explains 17% of the variance and passes. The word "checked"
          read as a verdict on all of them, so the banner now says what it did
          and shows the number, and the verdict is left to the reader and to
          the ranking below, where an SAE that bad shows up as zero scores
          clearing its own reconstruction error. Grading it in between would
          need a threshold nobody here has measured. */}
      {cal && (
        <div className={`hint ${cal.usable ? "" : "err"}`}>
          <b>Reconstruction {cal.usable ? "measured" : "failed"}.</b> Measured
          against your model over {cal.n_tokens} tokens rather than read from
          the SAE's config — which{" "}
          {cal.declared_b_dec === null
            ? "does not say"
            : `declares b_dec ${cal.declared_b_dec}`}
          . Best of four input conventions: activations{" "}
          <b>{cal.center ? "centered" : "not centered"}</b> along d_model,{" "}
          <b>b_dec {cal.subtract_b_dec ? "subtracted" : "left in"}</b>, leaving{" "}
          <b>{fmtFVU(cal.fvu)}</b> of the variance unexplained (
          {percent(cal.rel_err, 1)} of the stream's norm) with{" "}
          <b>{cal.l0.toFixed(1)}</b> of {sae.d_sae?.toLocaleString()} features
          firing per token. Both are aggregates over those {cal.n_tokens}{" "}
          tokens; a ranking below reports what the SAE misses at the token you
          asked about, which can be a good deal worse than this.
        </div>
      )}

      {cal && !cal.usable && (
        <div className="resting-empty">
          <b>Not plotting these features.</b> The best of the four conventions
          still leaves an FVU of {fmtFVU(cal.fvu)}, and anything at or above{" "}
          {cal.unusable_at} carries less of the activation than a constant
          would. What comes out is not a decomposition of anything, so a chart
          of it would be a picture of nothing — which is the failure this panel
          is least able to show you and most likely to be believed about.
          Tried, best first:{" "}
          {cal.ranked.map(([name, fvu]) => `${name} ${fmtFVU(fvu)}`).join(" · ")}
          . The logit lens below asks a different question of the same residual
          stream and needs no SAE.
        </div>
      )}

      {/* The output-space half of the banner above, sited beside it because
          they answer the same question in two spaces and disagreeing is the
          interesting case: an SAE can reconstruct the activations beautifully
          and still cost the model most of its predictive loss, because the
          directions carrying the residual stream's variance are not the
          directions the next token depends on.

          `!DEMO && !VIEWER`, like the ranking below. Every pass here is
          against a live model over a corpus the reader supplies, so there is
          nothing to bake — and `/api/sae/fidelity` sits under `/api/sae/`
          precisely so that a demo build cannot be answered 200 by demo.ts's
          `/api/features/` prefix with a single feature's detail payload. */}
      {!DEMO && !VIEWER && (
        <SaeFidelityCard sae={sae} epoch={epoch} disabled={busy !== ""} />
      )}

      {summary && (!cal || cal.usable) && (
        <div className="attn-scroll">
          <div className="attn-inner">
            <div className="tokens">
              {summary.tokens.map((t, i) => {
                const h = heat?.[i] ?? 0;
                return (
                  <span
                    key={i}
                    ref={i === peak ? peakRef : undefined}
                    className={`tok ${tokenSel === i ? "feat-sel" : ""} ${i === peak ? "peak" : ""}`}
                    tabIndex={0}
                    role="button"
                    aria-pressed={tokenSel === i}
                    aria-label={`token ${i + 1} of ${summary.tokens.length}: ${t.trim() || "space"}${i === peak ? ", peak activation" : ""}`}
                    style={
                      heat
                        ? { backgroundColor: `rgba(160,140,255,${(0.42 * h).toFixed(3)})` }
                        : undefined
                    }
                    onClick={() => setTokenSel(i)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        setTokenSel(i);
                      }
                    }}
                  >
                    {t.replace(/ /g, "·") || "·"}
                  </span>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {summary && tokenSel >= 0 && (
        <>
          {/* Beside the list it re-annotates, and gated the same way the
              attention panel's token ranking is.

              A recording, the static demo and the `.mri` viewer have no model,
              so this control could do nothing there but fail — and a button
              that only ever fails does not teach a visitor that the page has no
              model behind it, it teaches them the measurement does not work.
              Worse here than there: demo.ts answers `/api/features/` as a
              PREFIX, so an ungated button would be served the single-feature
              detail payload with a 200 and the panel would render a fabricated
              ranking as a measurement. `rankFeatures` refuses in these builds
              too, so the gate is not the only thing standing between a visitor
              and that.

              The dtype is that same argument on the machine this was written
              on. `runtime.rank_features` refuses anything but float32, and
              ModelMRI selects bfloat16 for every NVIDIA GPU — so on the default
              configuration here the control rendered, quoted a cost badge of
              67 passes, and could only ever answer 409. The runtime's refusal
              is shown INSTEAD of the button rather than behind it. */}
          {!DEMO && !VIEWER && !rankable && (
            <div className="hint">
              <b>No feature ranking in {dtype}.</b> Removing a feature moves the
              residual stream by a small vector, and in {dtype} the model's own
              arithmetic charges for that move: an edit whose true effect is
              vanishingly small can read as a large one and outrank a feature
              with many times its activation — while writing the stream back
              unchanged is still bit-exact and still scores 0.0, so the noise
              floor cannot catch it. It works in float32, which ModelMRI
              selects for CPU and never
              for a GPU: start the server with the GPU hidden (PowerShell{" "}
              <code>$env:CUDA_VISIBLE_DEVICES=''</code>) and load the model
              again. The list above is unaffected — it was never a causal claim.
            </div>
          )}
          {!DEMO && !VIEWER && rankable && (
            <div className="row feat-rank-row">
              <button
                className="ghost sm"
                onClick={() => void onRankFeatures()}
                disabled={fabBusy}
                title={rankTitle}
              >
                {fabBusy ? "ranking…" : "Rank features"}
                {costBadge && <span className="meta"> {costBadge}</span>}
              </button>
              <label className="meta" htmlFor="feat-scope">
                remove each feature
              </label>
              <select
                id="feat-scope"
                className="sm"
                value={fabScope}
                onChange={(e) =>
                  setFabScope(e.target.value as "position" | "prompt")
                }
                disabled={fabBusy}
                title="Which features are on trial, and where they are removed. These are two different questions and they give different answers."
              >
                <option value="position">at this token only</option>
                <option value="prompt">wherever it fires, up to here</option>
              </select>
            </div>
          )}
          {!DEMO && !VIEWER && rankable && fabScope === "prompt" && !measured && (
            <div className="hint">
              Ranking across the prompt costs several times what ranking at one
              token does, and the panel cannot price it in advance: the
              candidates are every feature firing at any token up to this one,
              and the list above carries only the top 8 per token. It is also
              the scope that finds features this panel cannot otherwise show —
              ones that fire only at earlier tokens and reach the answer
              through attention.
            </div>
          )}
          <div className={`feat-list${measured ? " ranked" : ""}`}>
            <div className="meta" style={{ marginBottom: 2 }}>
              top features on {summary.tokens[tokenSel].replace(/ /g, "·")} —
              token {tokenSel}, by activation
              {measured && (
                <>
                  {" "}
                  · KL = removing it{" "}
                  {measured.scope === "position"
                    ? "here"
                    : "everywhere it fires"}
                </>
              )}
            </div>
            {(() => {
            const rows = summary.top[tokenSel] ?? [];
            const maxAct = rows[0]?.[1] || 1;
            // Which of the plotted features the ranking has an answer for. A
            // feature the run did not reach must not borrow the appearance of
            // one that scored nothing — at prompt scope the cap is ordered by
            // peak activation over the whole window, so a feature that is loud
            // HERE can still be dropped.
            const scoreOf = measured
              ? new Map(measured.ranked.map((r) => [r.feature_id, r]))
              : null;
            // …and absence from that map only means "never tested" when the
            // response carries every scored row. It did not: `rankFeatures`
            // used to send no `top_k`, the server trimmed 256 scored rows to
            // 64, and every plotted feature below causal rank 64 was labelled
            // "not tested" — the exact inversion of what the word means.
            // Measured: #18994 scored 0.00031514 at causal rank 72 and the
            // chip said it had not been asked. `rankFeatures` now requests
            // every row, and this asserts the response really carries them
            // rather than trusting the number in the URL.
            const complete = measured
              ? measured.n_returned >= measured.n_scored
              : false;
            // SIGNATURE — reveal ordered by magnitude, not by DOM order. The
            // strongest activation starts at t=0 and the rest follow by rank,
            // so the eye lands on the maximum before the others exist. Every
            // row is the same violet, so rank is the only channel left to say
            // which one matters; spending time instead of colour is free.
            const rank = new Map(
              rows
                .map((r, i) => [i, r[1]] as const)
                .sort((a, b) => b[1] - a[1])
                .map(([i], r) => [i, r]),
            );
            return rows.map(([fid, act], i) => {
              const score = scoreOf?.get(fid);
              return (
                <div
                  key={fid}
                  className={`feat-row ${featSel === fid ? "sel" : ""}`}
                  style={{ ["--i" as string]: rank.get(i) ?? 0 }}
                  onClick={() => void onPickFeature(fid)}
                >
                  <span className="feat-id">#{fid}</span>
                  <div className="feat-bar" style={{ width: `${(160 * act) / maxAct}px` }} />
                  <span className="feat-act">{act.toFixed(1)}</span>
                  {/* The point of the whole control: the chart's own order,
                      annotated with what each bar actually did. Four states,
                      and the last two must never look alike — a feature the run
                      had no budget for is not a feature that scored nothing,
                      and a feature trimmed out of the response is neither.

                      Gated on !DEMO && !VIEWER as well as on `measured`, which
                      is redundant at runtime and is the point: the constants
                      fold at build time, so rollup drops this block and its
                      prose from the demo bundle instead of shipping ~2.4 kB of
                      text that can never render. */}
                  {!DEMO && !VIEWER && measured &&
                    (score ? (
                      <span
                        className={`feat-kl${score.below_resolution ? " faint" : ""}`}
                        title={
                          score.below_resolution
                            ? `KL ${fmtKL(score.kl)} nats — at or below this measurement's numerical resolution of ${measured.resolution_kl.toExponential(0)}, which is arithmetic rather than the model. The noise floor is a different and smaller number (${measured.noise_floor_kl}); greying out at the floor would grey out nothing.`
                            : `KL ${fmtKL(score.kl)} nats · p(${JSON.stringify(measured.target_token)}) ${pair(score.p_top_before, score.p_top_after)[0]} → ${pair(score.p_top_before, score.p_top_after)[1]}${score.flips_top ? " · changes the top token" : ""} · ONE random direction of the same size at the same tokens cost ${fmtKL(score.control_kl)}${score.clears_control ? "" : ", MORE than this feature's own edit — this score is not distinguished from the size of the edit"} · that control is a single draw, and it moves: over 8 draws per row the median row's control spans a factor of about 2.5, and roughly half the rows fall between their own smallest and largest draw, so a score near its control is left undecided by this test rather than settled by it · after the edit the SAE still reads ${percent(score.encoder_residual, 0)} of this feature`
                        }
                      >
                        {score.below_resolution
                          ? `< ${measured.resolution_kl.toExponential(0)}`
                          : fmtKL(score.kl)}
                        {!score.below_resolution && !score.clears_control && (
                          <span className="feat-kl-flag"> ≤ctrl</span>
                        )}
                      </span>
                    ) : complete ? (
                      <span
                        className="feat-kl untested"
                        title={`This run tested ${measured.n_tested} of ${measured.n_candidates} candidate features, chosen by peak activation, and every one it tested is in this response. #${fid} was not one of them — not asked, not found unimportant.`}
                      >
                        not tested
                      </span>
                    ) : (
                      <span
                        className="feat-kl untested"
                        title={`This response carries ${measured.n_returned} of the ${measured.n_scored} rows this run scored, so #${fid} was either never tested (${measured.n_tested} of ${measured.n_candidates} candidates were) or tested and trimmed out. The panel cannot tell which from what it was sent, and will not guess.`}
                      >
                        no score sent
                      </span>
                    ))}
                </div>
              );
            });
          })()}
          </div>
          {/* The server's own sentence, unwrapped. Most of what this endpoint
              refuses with are refusals rather than failures — an SAE that does
              not reconstruct the stream it is attached to, a position where
              nothing fires, a write-back that is not landing where the capture
              came from — and each already says what would make the measurement
              work. Anything put in front of them is the client guessing. */}
          {!DEMO && !VIEWER && fabErr && (
            <div className="hint err">{fabErr}</div>
          )}
          {/* Gated as well as unreachable. `measured` can only be non-null if
              the button above ran, and the button does not exist in these
              builds — but rollup cannot prove that from a runtime value, so
              without this the whole ranking component and its wording ship in
              the demo bundle as text that can never render. Repeating the
              constant is what makes the exclusion a build-time fact: measured
              here, the demo bundle contains neither "Rank features" nor
              "/api/features/ablate". */}
          {!DEMO && !VIEWER && measured && (
            <FeatureRanking
              a={measured}
              tokens={summary.tokens}
              plotted={(summary.top[tokenSel] ?? []).map(([fid]) => fid)}
              selected={featSel}
              onPick={(fid) => void onPickFeature(fid)}
            />
          )}
        </>
      )}

      {/* What it fires on in YOUR text, and what it promotes — sited here
          rather than in a panel of its own because the third readout is
          already on this page. The causal ranking above measures what removing
          the feature does; a claim that survives activation, weights AND
          ablation is worth something a claim resting on top activations alone
          is not. */}
      {featSel >= 0 && !DEMO && !VIEWER && (
        <FeatureEvidencePanel feature={featSel} epoch={epoch} />
      )}

      {/* AND THE ANSWER FOR EVERY MODEL WITHOUT ONE OF THOSE, which is most
          of them. Everything above this line needs a published sparse
          autoencoder in the registry; this reads the MLP neurons directly, so
          it works on any model at all. It is blunter and it says so at the
          top of its own answer — a neuron is polysemantic, which is the whole
          reason the autoencoders above exist — and it sits below them rather
          than above for exactly that reason. */}
      {!DEMO && !VIEWER && <NeuronEvidencePanel epoch={epoch} />}

      {featSel >= 0 && (
        <div className="row" style={{ marginTop: 14 }}>
          <span className="meta">steer #{featSel}</span>
          <input
            type="range"
            min={-60}
            max={60}
            step={5}
            value={scale}
            onChange={(e) => setScale(Number(e.target.value))}
          />
          <span className="meta" style={{ minWidth: 34 }}>
            {scale > 0 ? `+${scale}` : scale}
          </span>
          <button
            className="violet"
            onClick={onSteerTest}
            disabled={busy !== "" || !prompt.trim()}
            title={
              prompt.trim()
                ? "Same prompt, greedy decoding, once clean and once steered"
                : "Generate in this tab first — the A/B needs the prompt"
            }
          >
            {busy === "steer" ? "Running A/B…" : "Run steering A/B"}
          </button>
        </div>
      )}

      {ab && (
        <div className="compare" style={{ marginTop: 14 }}>
          <div className="card">
            <span className="lbl">BASELINE</span>
            {ab.base}
          </div>
          <div className="card steered">
            <span className="lbl">FEATURE #{featSel} @ {scale > 0 ? `+${scale}` : scale}</span>
            {ab.steered}
          </div>
        </div>
      )}

      {err && <div className="hint">{err}</div>}
      <div className="hint">
        steering adds the feature's decoder direction to the residual stream during
        generation — deterministic (temp 0), fully reversible
      </div>
    </div>
  );
}

/** The causal ranking: what removing each feature actually did.
 *
 *  It sits UNDER the activation list rather than replacing it, and the list
 *  above is annotated with these same scores, because the panel's claim is not
 *  "here is a better order" — it is "these are two different orderings of the
 *  same features and here is where they disagree". Swapping one list for the
 *  other silently would answer a question the reader did not know had two
 *  answers.
 *
 *  Everything below is either a field the server measured or is computed from
 *  two of them on this run. In particular the additivity caveat prints this
 *  run's own `sum_of_singles` against its own `joint_kl` rather than a
 *  remembered direction: features and heads can miss opposite ways, so a
 *  copied sentence would be exactly backwards.
 */
function FeatureRanking({
  a,
  tokens,
  plotted,
  selected,
  onPick,
}: {
  a: FeatureAblation;
  tokens: string[];
  /** The feature ids the bar chart above is plotting, in its own order. The
   *  comparison is against THAT list and not against a peak-activation list
   *  the panel never showed — at prompt scope the two overlap the causal top
   *  by different amounts, and quoting the one the reader never saw would
   *  overstate the finding. */
  plotted: number[];
  selected: number;
  onPick: (fid: number) => void;
}) {
  const n = plotted.length;
  const causalTop = a.ranked.slice(0, n).map((r) => r.feature_id);
  const rankOf = new Map(a.ranked.map((r, i) => [r.feature_id, i + 1]));
  const shared = causalTop.filter((f) => plotted.includes(f));
  const entered = causalTop.filter((f) => !plotted.includes(f));
  const demoted = plotted.filter((f) => !causalTop.includes(f));
  const ratio = a.joint_kl > 0 ? a.sum_of_singles / a.joint_kl : null;
  // How many measured scores are bigger than what the decomposition itself
  // gets wrong over the window these edits landed in. Counted over the rows
  // that came back, and the sentence says which set that is — the server's
  // `n_scored` is the number tested, and the two are equal only when the
  // response was not trimmed.
  const clearing = a.ranked.filter((r) => r.kl > a.residual_kl).length;
  const resolution = a.resolution_kl.toExponential(0);
  // Rows this response did not carry. `a.ranked.length` is the response, not
  // the run: at prompt scope with the old default top_k the panel printed
  // "54 more were tested and scored lower" directly above the server's own
  // "256 of 494 firing features were tested".
  const moreScored = a.n_scored - Math.min(SHOWN_CAUSAL, a.ranked.length);
  const trimmed = a.n_returned < a.n_scored;

  /** Where a feature was removed, in tokens rather than indices. Only shown at
   *  prompt scope, where it is the whole point: a feature that fires at token 1
   *  and nowhere near the answer still reaches it through attention, and the
   *  chart above cannot show that at all. */
  const fires = (r: FeatureScore) => {
    const head = r.positions
      .slice(0, 3)
      .map((p) => `${p} ${JSON.stringify(tokens[p] ?? "?")}`)
      .join(", ");
    const rest = r.positions.length - 3;
    return `fires at ${head}${rest > 0 ? ` and ${rest} more` : ""}`;
  };

  const name = (f: number) => {
    const r = rankOf.get(f);
    return r ? `#${f} (causal rank ${r})` : `#${f} (not tested)`;
  };

  return (
    <div className="ranking">
      <div className="ranking-head">
        {/* The intervention and the position are both in the sentence, and
            neither is decoration. "Removing a feature" is three different
            experiments and two of them are measurably indefensible on this
            SAE; and every score here is about ONE next-token distribution at
            ONE token, which is the claim the sentence has to carry. */}
        <strong>
          Removing one feature at a time — subtracting its activation × decoder
          direction from the residual stream at {a.hook}
          {a.scope === "position"
            ? `, at token ${a.position} only`
            : `, at every token up to ${a.position} where it fires`} — and
          measuring how far the answer at token {a.position},{" "}
          {JSON.stringify(a.target_token)}, moves.
        </strong>
        <span className="meta">
          {a.passes} forward passes · {a.elapsed_s}s · {a.intervention} · floor{" "}
          {a.noise_floor_kl} · resolution {resolution}
        </span>
      </div>

      {/* Two states in which the leaderboard below is not worth reading in
          order, said BEFORE it rather than in the small print under it. A
          caveat placed after a ranked list is read after the ranking has
          already been believed. */}
      {!a.removal_verified && (
        <div className="hint err">
          <b>The edit did not land exactly.</b> The stream the model received
          differs from <code>x − activation × W_dec</code> by{" "}
          {a.edit_deviation.toExponential(2)}, so the order below may be
          describing the model's arithmetic rather than the features. Read{" "}
          {"nothing"} into the ranking until that is zero.
        </div>
      )}
      {clearing === 0 && (
        <div className="hint err">
          <b>
            Not one of these {a.ranked.length} features moves the answer as much
            as the SAE's own reconstruction error does.
          </b>{" "}
          Substituting the reconstruction over the tokens these edits land in,
          with nothing removed, costs {fmtKL(a.residual_kl)} nats — more than
          every score below. The order may be real; it is smaller than what the
          decomposition gets wrong while measuring it.
        </div>
      )}

      {/* The finding the control exists for, stated before the list rather
          than left for the reader to derive by comparing two orders. */}
      <div className="hint">
        <b>
          {shared.length} of the {n} features the chart plots are also in the
          causal top {n}.
        </b>
        {entered.length > 0 && (
          <>
            {" "}
            Reached it from outside the chart: {entered.map(name).join(", ")}.
          </>
        )}
        {demoted.length > 0 && (
          <>
            {" "}
            Plotted, but lower once measured: {demoted.map(name).join(", ")}.
          </>
        )}{" "}
        Activation and causal effect are different quantities and this is where
        they disagree — the bar chart's smooth decay is not what the causal
        picture looks like.
      </div>

      <ol className="ranking-list">
        {a.ranked.slice(0, SHOWN_CAUSAL).map((r) => (
          <li
            key={r.feature_id}
            className={`${r.below_resolution ? "faint" : ""}${selected === r.feature_id ? " sel" : ""}`}
          >
            <button
              className="ghost sm"
              onClick={() => onPick(r.feature_id)}
              title="Show where this feature fires across the generation"
            >
              #{r.feature_id}
            </button>
            {/* Never "0.00000". A score below the resolution is named as such
                rather than printed as a small number, and the threshold is the
                RESOLUTION the server measured, not its noise floor — the floor
                is exactly 0.0 on this path and two measured scores came back
                negative, so greying out at the floor would grey out nothing. */}
            <span className="mid">
              {r.below_resolution
                ? `below the resolution (< ${resolution})`
                : `KL ${fmtKL(r.kl)}`}
            </span>
            <span className="meta">
              act {r.activation.toFixed(1)} · p(
              {JSON.stringify(a.target_token)}){" "}
              {pair(r.p_top_before, r.p_top_after)[0]} →{" "}
              {pair(r.p_top_before, r.p_top_after)[1]}
              {r.flips_top && " · changes the top token"}
              {!plotted.includes(r.feature_id) && " · not in the chart above"}
              {/* The control, on every row rather than in a footnote. A row
                  that does not clear it has a score explained by the size of
                  its edit, and rows like that can be plotted in the chart
                  above. */}
              {!r.below_resolution &&
                (!r.clears_control
                  ? ` · NOT above a same-size random edit (${fmtKL(r.control_kl)})`
                  : r.control_kl > 0
                    ? ` · ${(r.kl / r.control_kl).toFixed(1)}x a same-size random edit (${fmtKL(r.control_kl)})`
                    : ` · above a same-size random edit (${fmtKL(r.control_kl)})`)}
            </span>
            {a.scope === "prompt" && (
              <>
                <span className="spacer" />
                <span className="meta">{fires(r)}</span>
              </>
            )}
          </li>
        ))}
        {/* Counted off `n_scored`, which is the run, not `a.ranked.length`,
            which is the response. They differ whenever the server trimmed, and
            the old version printed "54 more" a few pixels above the server's
            own "256 of 494 were tested". */}
        {moreScored > 0 && (
          <li className="meta">
            {moreScored} more were tested and scored lower.
          </li>
        )}
      </ol>

      {/* In the server's words, because "not listed" and "not important" are
          the two things a truncated leaderboard is read as meaning. And the
          server's OTHER sentence, `rows_note`, exists precisely to keep
          "trimmed from this response" apart from "never measured" — it was
          being built and never rendered, which is how the two counts above
          could contradict each other on screen. */}
      {a.truncated && <div className="hint">{a.coverage}</div>}
      {/* `rows_note` also carries how many rows scored BELOW the measurement's
          own resolution — a fact about the numbers on screen that has nothing
          to do with whether the list was trimmed. Gating it on `trimmed` meant
          an untrimmed ranking never said that some of its rows are arithmetic
          rather than measurement, which is exactly the row a reader would
          otherwise act on. */}
      {(trimmed || (a.n_below_resolution ?? 0) > 0) && (
        <div className="hint">{a.rows_note}</div>
      )}

      {/* The caveat travels with the numbers, and the direction is READ OFF
          THIS RUN. Copying the head panel's sentence would state the opposite
          of what these scores do. */}
      <div className="hint">
        These are <em>not</em> each feature's share of the prediction, and they
        do not add up. Read which way off this run rather than remembering one:
        the {a.n_tested} tested features sum to{" "}
        <b>{fmtKL(a.sum_of_singles)}</b> nats, while one joint ablation removing
        all of them at once gives <b>{fmtKL(a.joint_kl)}</b>
        {ratio !== null && (
          <>
            {" "}
            — a ratio of <b>{ratio.toFixed(2)}x</b>
          </>
        )}
        . Below 1 the singles under-state the joint, above 1 they over-state
        it, and the attention panel's version of this caveat does not carry
        over in either direction.
      </div>
      <div className="hint">{a.means}</div>

      {/* The reconstruction caveat, and the reason the aggregate is not
          allowed to stand on its own: it is an aggregate over every token and
          is dominated by the attention sink, so beside a feature at THIS token
          it would let a reader believe the decomposition is nearly complete.

          The aggregate quoted here is `rel_err`, NOT `fvu`. Both are on this
          payload and only one of them is in the same units as
          `residual_share`: fvu is a squared-error fraction and residual_share
          is a norm fraction, so pairing 0.0010 with 0.204 under a "but"
          announces a 200x gap where the like-for-like one — 0.029 against
          0.204 — is 7x. The measurement was never three orders of magnitude;
          the sentence was. `fvu` still gets printed, beside the convention it
          belongs to, where nothing is being compared to it. */}
      <div className="hint">
        <b>
          This is an intervention on the SAE's model of the stream, not on the
          model's own units.
        </b>{" "}
        Calibrated as <code>{a.convention}</code> (FVU {fmtFVU(a.fvu)}), the SAE
        leaves <b>{percent(a.rel_err, 1)}</b> of the stream's norm
        unmodelled averaged over every token
        {a.residual_share !== null && (
          <>
            {" "}
            — but up to{" "}
            <b>{percent(a.residual_share, 1)}</b> at a token{" "}
            {a.scope === "position"
              ? "these edits land in"
              : `in the window they land in (tokens ${a.residual_window[0]}–${a.residual_window[1]})`}
            {a.residual_share_at_position !== null &&
              a.scope !== "position" &&
              `, and ${percent(a.residual_share_at_position, 1)} at the token being attributed`}
          </>
        )}
        . Substituting the reconstruction over that same window with no feature
        removed costs <b>{fmtKL(a.residual_kl)}</b> nats. <b>{clearing}</b> of
        the {a.ranked.length} scores here are larger than it.
      </div>
      <div className="hint">{a.residual_means}</div>

      {/* What a score is worth against an edit of the same size that means
          nothing. Not a footnote: a top feature can clear its own control by a
          modest factor rather than by everything, and some rows do not clear
          it at all.

          The count is one draw's verdict and the sentence now says so. With 8
          draws per row, the number of rows clearing one draw and the number
          clearing all 8 come apart, and about half fall inside their own draw
          spread — "move the answer more than a random direction does" was a
          claim about the distribution, taken from one sample of it.

          Replicated with a second, independent set of 8 draws, both counts
          move. A shipped run reproduces exactly; every count derived from a
          FRESH draw does not, which is the point rather than a caveat on it,
          so the wording above avoids quoting one. */}
      <div className={a.n_clearing_control > 0 ? "hint" : "hint err"}>
        <b>
          {a.n_clearing_control} of the {a.n_scored} scored features beat the
          one random direction of the same size each was given.
        </b>{" "}
        {a.control_means}
      </div>

      {/* Whether the EDIT landed — the one claim that is a property of the edit
          rather than of each feature. Red when it could not be confirmed: every
          score above would then be describing something other than the edit
          this panel names. What the SAE still READS afterwards is a separate
          and per-row matter, and it is in `removal_check`'s second half and in
          each row's tooltip rather than folded into this tick. */}
      <div className={a.removal_verified ? "hint" : "hint err"}>
        {a.removal_check}
      </div>
      {/* The SAE is as much a part of this measurement as the model, and the
          receipt names both -- the same prompt through a different SAE ranks
          different features. */}
      <ReceiptLine receipt={a.receipt} />
    </div>
  );
}
