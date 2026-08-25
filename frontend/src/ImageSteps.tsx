import { useState } from "react";
import { measured, percent } from "./measured";
import {
  errorText,
  imageFilmstrip,
  imageFilmstripCost,
  imageStepsRun,
  ImageFilmstripRun,
  ImageStepTrace,
  shareImageRun,
} from "./api";

/**
 * When the denoiser committed, and what the picture looked like on the way.
 *
 * WHY THESE TWO ARE ONE PANEL AND STILL TWO MEASUREMENTS
 *
 * The image panel listed `step_commit` and `latent_trace` among what it can
 * measure and had no control for either — the backend could price them and
 * not run them, which is most of what "the image tool doesn't work" means
 * from outside.
 *
 * They sit together because they answer the same QUESTION — where in the run
 * did the picture stop changing — and they must not be shown as one answer,
 * because they are not:
 *
 *   the trace measures how far the LATENT moved per step. It decodes nothing
 *   and reports `vae_decodes: 0` as a checkable claim, because a decode makes
 *   the answer a property of the VAE as much as of the denoiser.
 *
 *   the filmstrip decodes a SUBSET of steps to actual pictures. That is the
 *   one you can look at, and the one that cannot tell you when the latent
 *   settled — a small late change in the latent can be a visible change in
 *   texture, and a large early one can vanish through a non-linear decoder.
 *
 * So the strip carries its own caveat, and the frame count is never presented
 * as the step count. Eight frames from a fifty-step run is eight frames, and
 * the skipped steps are listed rather than implied.
 */
export default function ImageSteps({ steps }: { steps: number }) {
  const [trace, setTrace] = useState<ImageStepTrace | null>(null);
  const [strip, setStrip] = useState<ImageFilmstripRun | null>(null);
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");
  const [prompt, setPrompt] = useState("a photograph of an astronaut riding a horse");
  const [seedFixed, setSeedFixed] = useState(true);
  const [seed, setSeed] = useState(0);
  // Every Nth step. The API refuses to choose a subset for you — decoding all
  // of them is the cost the filmstrip exists not to pay, and a default subset
  // would be the tool deciding what you get to look at.
  const [every, setEvery] = useState(4);
  const [plan, setPlan] = useState<Awaited<
    ReturnType<typeof imageFilmstripCost>
  > | null>(null);

  const shot = () => (seedFixed ? seed : null);

  async function runTrace() {
    setBusy("trace");
    setErr("");
    try {
      setTrace(await imageStepsRun({ prompt, steps, seed: shot() }));
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }

  async function price() {
    setErr("");
    try {
      setPlan(await imageFilmstripCost({ steps, every }));
    } catch (e) {
      setErr(errorText(e));
    }
  }

  async function runStrip() {
    setBusy("strip");
    setErr("");
    try {
      setStrip(await imageFilmstrip({ prompt, steps, every, seed: shot() }));
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }

  /** Write the run this server just made into a `.mri`.
   *
   *  The blob is turned into a click rather than navigated to, because the
   *  server refuses with JSON and a 409 when there is nothing to share -- a
   *  plain link would save that sentence to disk under a `.mri` extension and
   *  the recipient would open a file with nothing in it.
   *
   *  The object URL is revoked after the click. A data URL of a megabyte
   *  filmstrip held on the document until reload is a leak with a plausible
   *  excuse.
   */
  async function share() {
    setBusy("share");
    setErr("");
    try {
      const blob = await shareImageRun({
        note: prompt ? `denoising run: ${prompt}` : "",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "image-run.mri";
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }

  /** The tallest per-step move, so the bars have a scale.
   *
   *  Nulls are SKIPPED rather than treated as 0: the first step has no
   *  previous latent to have moved from, so its change is unknown, not zero. */
  const peak = trace
    ? trace.steps.reduce(
        (m, s) => (s.rms_change !== null && s.rms_change > m ? s.rms_change : m),
        0,
      ) || 1
    : 1;

  return (
    <div className="istep">
      <div className="row istep-controls">
        <input
          className="istep-prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="the prompt to run"
          aria-label="Prompt to run through the denoiser"
        />
        <label className="meta istep-seed">
          <input
            type="checkbox"
            checked={seedFixed}
            onChange={(e) => setSeedFixed(e.target.checked)}
          />
          fixed seed
        </label>
        {seedFixed && (
          <input
            type="number"
            className="istep-num"
            value={seed}
            min={0}
            onChange={(e) => setSeed(Math.max(0, Number(e.target.value) || 0))}
            aria-label="Seed"
          />
        )}
      </div>

      {/* ── the trace: nothing decoded ─────────────────────────────────── */}
      <div className="row istep-controls">
        <button
          className="ghost"
          onClick={() => void runTrace()}
          disabled={busy !== ""}
        >
          {busy === "trace" ? "Running…" : "Measure the steps"}
        </button>
        <span className="meta">
          per-step latent movement · decodes nothing
        </span>
      </div>

      {trace && (
        <div className="istep-trace">
          <div className="istep-bars" role="img" aria-label="Per-step latent movement">
            {trace.steps.map((s) => (
              <span
                key={s.step}
                className={`istep-bar${s.rms_change === null ? " unknown" : ""}${
                  trace.commit_step !== null && s.step === trace.commit_step
                    ? " commit"
                    : ""
                }`}
                style={
                  s.rms_change === null
                    ? undefined
                    : { height: `${Math.max((s.rms_change / peak) * 100, 2)}%` }
                }
                title={
                  s.rms_change === null
                    ? `step ${s.step} · no previous latent to move from, so the change is unknown`
                    : `step ${s.step} · moved ${measured(s.rms_change, 4)} RMS` +
                      (s.cumulative !== null
                        ? ` · ${percent(s.cumulative, 1)} of the run's total by here`
                        : "")
                }
              />
            ))}
          </div>
          {/* The sentence is the server's, beside the numbers it describes. */}
          <p className="meta">{trace.means}</p>
          <p className="meta istep-claim">
            vae_decodes {trace.vae_decodes} — this measured the LATENT. Latent
            distance is not visible difference: a non-linear decoder can turn a
            small late change into visible texture and swallow a large early
            one. Decode some steps below to see the picture instead.
          </p>
        </div>
      )}

      {/* ── the filmstrip: a subset decoded ───────────────────────────── */}
      <div className="row istep-controls">
        <label className="meta">
          every
          <input
            type="number"
            className="istep-num"
            value={every}
            min={1}
            max={steps}
            onChange={(e) => setEvery(Math.max(1, Number(e.target.value) || 1))}
            aria-label="Decode every Nth step"
          />
          th step
        </label>
        <button className="ghost" onClick={() => void price()} disabled={busy !== ""}>
          What will it cost?
        </button>
        <button
          className="green"
          onClick={() => void runStrip()}
          disabled={busy !== ""}
        >
          {busy === "strip" ? "Decoding…" : "Watch it form"}
        </button>
      </div>

      {plan && (
        <p className="meta">
          {plan.frames} frame{plan.frames === 1 ? "" : "s"} from {plan.steps}{" "}
          steps · {plan.vae_decodes} VAE decode
          {plan.vae_decodes === 1 ? "" : "s"} · steps{" "}
          {plan.decoded_steps.join(", ")}
          {plan.skipped_steps.length > 0 &&
            ` · skipping ${plan.skipped_steps.length}`}
        </p>
      )}

      {strip && (
        <div className="istep-strip">
          <div className="istep-frames">
            {strip.frames.map((f) => (
              <figure key={f.step} className="istep-frame">
                {/* `null` rather than "" when there are no bytes, so a decode
                    that never happened cannot render as a broken image that
                    looks like one producing black. */}
                {f.png ? (
                  <img src={f.png} alt={`the latent at step ${f.step}, decoded`} />
                ) : (
                  <span className="meta">not decoded</span>
                )}
                <figcaption className="meta">
                  step {f.step}
                  {f.latent_rms !== null && ` · rms ${measured(f.latent_rms, 3)}`}
                  {f.downsampled && (
                    <>
                      {" "}
                      · shown at {f.width}px, decoded at {f.decoded_width}px
                    </>
                  )}
                </figcaption>
              </figure>
            ))}
          </div>
          <p className="meta">{strip.means}</p>
          {/* Never let a strip read as a full recording. */}
          <p className="meta istep-claim">
            {strip.frames_decoded} of {strip.steps_run} steps were decoded
            {strip.skipped_steps.length > 0 && (
              <> · skipped {strip.skipped_steps.join(", ")}</>
            )}
            {strip.steps_never_reached.length > 0 && (
              <>
                {" "}
                · never reached {strip.steps_never_reached.join(", ")}
              </>
            )}
            . A frame is what the decoder made of that step's latent, not
            evidence of when the latent settled — the measurement above is.
          </p>

          {/* A6. Every other result this tool produces could be sent to
              somebody; this one could only be screenshot — and a screenshot
              carries no provenance, no seed, no scheduler and no statement of
              what was shrunk. The file carries the strip AS MEASURED: the
              server writes it from the run it made, so nothing this button
              sends becomes a claim in it.

              Bytes and a Content-Disposition rather than an `<a download>`,
              like `exportSession`: the server answers a refusal as JSON, and
              a link would cheerfully save that sentence to disk as a `.mri`
              the recipient then cannot open. */}
          <div className="row">
            <button
              className="ghost sm"
              onClick={() => void share()}
              disabled={busy !== ""}
            >
              {busy === "share" ? "writing…" : "Share this run (.mri)"}
            </button>
            <span className="meta">
              {strip.frames_decoded} frame(s), the seed and the scheduler, in a
              file that opens with nothing installed
            </span>
          </div>
        </div>
      )}

      {/* `hint err`, not a bare `err`: there is NO `.err` rule in the
          stylesheet, so this refusal rendered as ordinary body text —
          the same colour and weight as the explanatory prose around
          it, with nothing marking it as the answer to what was just
          clicked. Every other panel uses `hint err`. */}
      {err && <p className="hint err">{err}</p>}
    </div>
  );
}
