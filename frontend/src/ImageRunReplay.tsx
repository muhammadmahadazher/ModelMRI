import { useEffect, useState } from "react";
import { errorText, getImageReplay, ImageRunSection } from "./api";
import { measured } from "./measured";

/** An image run somebody sent you, opened with nothing installed.
 *
 *  A6, and the last unbuilt item in Theme A. Every other result this tool
 *  produces could be sent to somebody — a generation, a patching trace, a head
 *  ranking, a robot episode — and the one that is a PICTURE could not. So an
 *  image finding was the only kind that had to be screenshot to be shared, and
 *  a screenshot carries no provenance, no seed, no scheduler and no statement
 *  of what was shrunk on the way out.
 *
 *  WHAT THIS PANEL REFUSES TO LET A READER CONCLUDE:
 *
 *    a strip is not a run       four frames of a fifty-step run look like a
 *                               four-step run unless every frame says which
 *                               step it is, and unless the steps that ran and
 *                               were not decoded are counted separately from
 *                               the ones that never arrived. One is a choice,
 *                               the other is a gap.
 *    a picture is not a size    a frame that was shrunk to fit the file says
 *                               so, and says from what. A map drawn over a
 *                               silently resized picture is wrong in the way
 *                               that looks like a finding.
 *    no seed is not seed 0      an unseeded run does not repeat. That is the
 *                               loudest thing this panel says about it,
 *                               because it is the thing that decides whether
 *                               anybody can check the claim.
 *    a score is not a score     a classifier's probability, a detector's
 *                               confidence and a segmenter's share of the map
 *                               all render as a number between 0 and 1 and do
 *                               not compare. The kind is printed beside them.
 *
 *  It renders NOTHING when there is no image run, which is most sessions.
 *  `available: false` is a state and not an error, and a panel that announced
 *  "no image run in this file" on every text session would be noise.
 */
export default function ImageRunReplay({
  /** Bumped whenever the page resets. */
  epoch,
  /** Whether a `.mri` is open, and WHICH one — the `created_at` of the file
   *  rather than a bare boolean. Opening a second recording without closing
   *  the first leaves `open` at `true` the whole way through, so a boolean
   *  would leave the previous file's strip on screen under the new file's
   *  name. That is the worst failure this panel could have: a picture
   *  attributed to a run it did not come from. */
  sessionKey,
}: {
  epoch: number;
  sessionKey: string;
}) {
  const [run, setRun] = useState<ImageRunSection | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    let live = true;
    setErr("");
    void getImageReplay()
      .then((got) => {
        if (!live) return;
        setRun(got.available ? (got as ImageRunSection) : null);
      })
      .catch((e) => {
        if (!live) return;
        // NOT swallowed into `null`. A section that failed to load is not a
        // session without one, and rendering nothing for both would hide a
        // file the recipient cannot open behind a page that looks fine.
        setRun(null);
        setErr(errorText(e));
      });
    return () => {
      live = false;
    };
  }, [epoch, sessionKey]);

  if (err) {
    return (
      <div className="panel">
        <div className="hint err">{err}</div>
      </div>
    );
  }
  if (!run) return null;

  const frames = run.frames ?? [];
  const attn = run.attention;
  const readout = run.readout;
  const shrunk = frames.filter((f) => f.downsampled).length;

  return (
    <div className="panel" data-mri-group-label="image">
      <div className="row">
        <span className="dot d-image" />
        <h2>IMAGE RUN — SHARED, NOT RE-RUN</h2>
      </div>

      {/* THE PROVENANCE FIRST. Every number below is about this checkpoint and
          no other, and a strip rendered under ModelMRI's chrome with no model
          named is the confusion this whole section exists to prevent. */}
      <p className="meta">
        <b>{run.provenance.kind}</b> · {run.provenance.repo}
        {run.provenance.architecture && <> · {run.provenance.architecture}</>}
        {" · "}
        {run.provenance.revision ? (
          <>
            revision <code>{run.provenance.revision}</code>
          </>
        ) : (
          /* NOT blank. "" is the CLAIM that the checkpoint published none,
             and it is a different statement from nobody having looked — the
             server refuses the second and accepts the first. */
          <>this checkpoint published no revision</>
        )}
      </p>

      {/* THE ANSWER SLOT, and for an image run the answer is whether anybody
          can repeat it. A strip that cannot be reproduced is a picture, not a
          measurement, and that fact outranks every count below. */}
      <p className={`answer${run.seed === null ? " unmeasured" : ""}`}>
        <span className="answer-n">
          {run.seed === null ? "not repeatable" : `seed ${run.seed}`}
        </span>
        <span className="answer-of">
          {run.seed === null
            ? "no seed was fixed, so running this prompt again gives a different trajectory — which is why the file carries no seed rather than a 0"
            : `run this prompt through ${run.provenance.repo} at this seed and you get this trajectory back`}
        </span>
      </p>

      {run.prompt && (
        <p className="meta">
          prompt — <b>{run.prompt}</b>
          {run.scheduler && <> · {run.scheduler}</>}
        </p>
      )}

      {frames.length > 0 && (
        <>
          <div className="row irr-strip">
            {frames.map((f) => (
              <figure key={f.step} className="irr-frame">
                <img
                  src={f.png}
                  alt={`step ${f.step}`}
                  width={f.size[0]}
                  height={f.size[1]}
                />
                <figcaption className="meta">
                  step {f.step}
                  {/* The frame's OWN resolution, always — not the element's.
                      An <img> scales to its box and the box is the CSS's
                      choice, so the only honest place to read the pixels the
                      model produced is the number the file carries. */}
                  <br />
                  {f.size[0]}×{f.size[1]}
                  {f.downsampled && f.decoded_size && (
                    <>
                      {" "}
                      <span className="warn">
                        shrunk from {f.decoded_size[0]}×{f.decoded_size[1]}
                      </span>
                    </>
                  )}
                  {f.latent_rms !== null && (
                    <>
                      <br />
                      latent rms {measured(f.latent_rms, 3)}
                    </>
                  )}
                </figcaption>
              </figure>
            ))}
          </div>

          {/* A CHOICE AND A GAP, NEVER ONE NUMBER. Both mean "this strip is
              not the whole run", and they mean it for opposite reasons: one
              is a sampling decision and the other is a pipeline whose
              callback never fired. */}
          <p className="meta">
            {frames.length} of {run.steps_run || run.steps_requested} step(s)
            decoded
            {run.skipped_steps.length > 0 && (
              <>
                {" "}
                · {run.skipped_steps.length} ran and were not decoded (a
                choice)
              </>
            )}
            {run.steps_never_reached.length > 0 && (
              <>
                {" "}
                ·{" "}
                <span className="warn">
                  {run.steps_never_reached.length} were selected and never
                  arrived (a gap)
                </span>
              </>
            )}
            {run.steps_run > run.steps_requested && (
              <>
                {" "}
                · the pipeline ran {run.steps_run} for a request of{" "}
                {run.steps_requested}, which is a fact about the scheduler
                rather than a rounding
              </>
            )}
            {shrunk > 0 && (
              <>
                {" "}
                · {shrunk} frame(s) were shrunk to fit the file and each says
                from what
              </>
            )}
          </p>
        </>
      )}

      {attn && (
        <>
          <h3 className="irr-h">CROSS-ATTENTION, PER DENOISING STEP</h3>
          <p className="meta">
            Early steps decide layout and late steps decide texture, so a
            single averaged map hides the thing worth seeing. Each row is one
            step; each cell is one prompt token's share of the attention mass.
          </p>
          <div className="irr-maps">
            {attn.steps.map((s) => {
              const peak = Math.max(...s.per_token, 1e-9);
              return (
                <div key={s.step} className="irr-maprow">
                  <span className="meta irr-step">
                    step {s.step}
                    {s.timestep !== null && <> · t {measured(s.timestep, 0)}</>}
                  </span>
                  <div className="irr-cells">
                    {s.per_token.map((v, i) => (
                      <span
                        key={i}
                        className={`irr-cell${i >= attn.padding_from ? " pad" : ""}`}
                        style={{ opacity: Math.max(0.06, v / peak) }}
                        title={`${attn.tokens[i] ?? `column ${i}`} — ${measured(v, 4)}${
                          i >= attn.padding_from ? " (padding, not your prompt)" : ""
                        }`}
                      />
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
          {/* The padding boundary is a finding, not a footnote: CLIP pads to
              77 and the padded tail carries real attention mass. Plotting it
              as words would tell a reader the model is fascinated by
              `<pad>`. */}
          <p className="meta">
            columns from {attn.padding_from} are padding, not your prompt —
            they carry real attention mass and are shown dimmed rather than
            labelled as words
            {attn.columns_unlabelled > 0 && (
              <>
                {" "}
                ·{" "}
                <span className="warn">
                  {attn.columns_unlabelled} column(s) were MEASURED and have no
                  label to put on them — a cap on what can be shown, not on
                  what was measured
                </span>
              </>
            )}
            {attn.conditioning_width > 0 && (
              <> · the denoiser's conditioning was {attn.conditioning_width} wide</>
            )}
          </p>
          {attn.means && <div className="hint">{attn.means}</div>}
        </>
      )}

      {readout && (
        <>
          <h3 className="irr-h">READOUT</h3>
          {/* THE KIND, BEFORE THE NUMBERS. Three different quantities render
              as a number between 0 and 1 here and none of them compares to
              the others. */}
          <p className="meta">
            these are <b>{readout.kind}</b> scores —{" "}
            {readout.kind === "detection"
              ? "per-query detector confidences, which are not probabilities over anything and do not sum to one"
              : readout.kind === "classification"
                ? "probabilities over the class list"
                : "shares of the map's cells, which is not a confidence at all"}
          </p>
          <ul className="irr-rows">
            {readout.rows.map((r, i) => (
              <li key={i}>
                <b>{r.label}</b> <span className="meta">{measured(r.score, 4)}</span>
                {r.box_xyxy && (
                  <span className="meta">
                    {" "}
                    · box {r.box_xyxy.map((v) => measured(v, 0)).join(", ")}
                  </span>
                )}
                {r.query !== null && <span className="meta"> · query {r.query}</span>}
              </li>
            ))}
          </ul>
          {readout.means && <div className="hint">{readout.means}</div>}
        </>
      )}

      {run.means && <div className="hint">{run.means}</div>}
    </div>
  );
}
