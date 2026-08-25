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
    // CLEARED FIRST. `sessionKey` exists so opening a second `.mri` cannot
    // leave the first one's strip on screen under the second one's name --
    // and without this line it did exactly that for the length of the fetch,
    // which is the same wrong attribution for a shorter time. A panel showing
    // nothing while it loads is honest; one showing the previous file's
    // picture is not.
    setRun(null);
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

  // EVERY SHAPE HERE IS CHECKED, because on the two paths that reach this
  // panel only one of them has been validated. `/api/image/replay` on the
  // server returns what `session._image` accepted; the VIEWER build serves
  // the raw section straight out of the file with only `provenance.repo`
  // checked, and that copy is the one that runs in a recipient's browser on a
  // file a stranger forwarded. A missing `steps` or `rows` array there is not
  // a blank panel, it is a thrown TypeError with no error boundary above it
  // — a white page where the whole recording used to be.
  const list = <T,>(v: T[] | undefined | null): T[] => (Array.isArray(v) ? v : []);
  /** A frame that states its own resolution, which the viewer path does not
   *  guarantee. Without it `f.size[0]` is a TypeError on a hostile file. */
  const sized = (f: { size?: [number, number] }) =>
    Array.isArray(f.size) && f.size.length === 2;

  // A `.mri` NEVER FETCHES. `session._image` refuses a frame whose `png` is
  // not a `data:image/` URL, in as many words: "the picture travels inside it
  // or the frame does not travel at all". The viewer shim does not, so a
  // hostile file could carry `png: "https://…/1x1.gif"` and the recipient's
  // browser would announce them opening it. Filtered here, where both paths
  // meet, and COUNTED so the drop is reported rather than only applied.
  const carried = list(run.frames);
  const frames = carried.filter(
    (f) => typeof f?.png === "string" && f.png.startsWith("data:image/"),
  );
  const linked = carried.length - frames.length;
  const attn = run.attention;
  const readout = run.readout;
  const shrunk = frames.filter((f) => f.downsampled).length;

  // WHICH KIND OF RUN THIS IS. A classifier reading one photograph has no
  // seed, no trajectory and no denoising steps, and this panel printed "not
  // repeatable — running this prompt again gives a different trajectory" over
  // it, captioned the input photo "step 0", and announced "1 of 0 step(s)
  // decoded". Three false sentences about a run that is none of those things.
  const denoising = !readout || attn !== undefined || frames.length > 1;

  // How long the run was: what the pipeline RAN, else what was asked for,
  // else `null`. Never 0 — a 0-step run decoded no frames, so a file claiming
  // both is stating a contradiction rather than a length, and the counts line
  // says so instead of printing "3 of 0".
  const positive = (n: number | null | undefined) =>
    typeof n === "number" && n > 0 ? n : null;
  const length = positive(run.steps_run) ?? positive(run.steps_requested);

  // WHERE THE PROMPT STOPS, or `null` because nobody measured it. Kept apart
  // from 0 everywhere it is used: 0 is a boundary at column zero, which says
  // every column is padding.
  const boundary =
    attn && typeof attn.padding_from === "number" && attn.padding_from >= 0
      ? attn.padding_from
      : null;
  // The widest row actually measured, not `tokens.length` — a file can label
  // fewer columns than it measured, and `columns_unlabelled` is the count it
  // carries for exactly that.
  const columns = attn
    ? Math.max(
        0,
        list(attn.tokens).length,
        ...list(attn.steps).map((st) => list(st.per_token).length),
      )
    : 0;
  const isPad = (i: number) => boundary !== null && i >= boundary;

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
        <b>{run.provenance.kind}</b> · {run.provenance.repo} ·{" "}
        {run.provenance.architecture ? (
          run.provenance.architecture
        ) : (
          /* Same rule as `revision` below, and the server now accepts "" here
             for the same reason: a `config.json` with no `architectures` is a
             checkpoint that published none, which is a fact worth printing
             rather than a gap worth hiding. */
          <>no architecture published</>
        )}
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

      {/* THE ANSWER SLOT, and WHICH question it answers depends on the run.
          For a sampled run the answer is whether anybody can repeat it: a
          strip that cannot be reproduced is a picture, not a measurement, and
          that outranks every count below.

          For a READOUT there is no sampling, so "not repeatable" is not the
          answer — it is a false statement about a forward pass that gives the
          same numbers every time. The answer there is what the model said. */}
      {denoising ? (
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
      ) : (
        readout &&
        list(readout.rows).length > 0 && (
          <p className="answer">
            <span className="answer-n">{list(readout.rows)[0].label}</span>
            <span className="answer-of">
              at {measured(list(readout.rows)[0].score, 4)} — a{" "}
              {readout.kind} score, which is why the kind is printed with it.
              One forward pass over one picture: no sampling, so this repeats
              exactly and carries no seed.
            </span>
          </p>
        )
      )}

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
                  alt={denoising ? `step ${f.step}` : "the picture this was read from"}
                  {...(sized(f) ? { width: f.size[0], height: f.size[1] } : {})}
                />
                <figcaption className="meta">
                  {denoising ? (
                    `step ${f.step}`
                  ) : (
                    /* A readout carries ONE frame: the picture the model was
                       shown. Captioning it "step 0" made a classifier's input
                       photograph read as the first frame of a denoising run
                       that never happened. */
                    <>the picture this was read from</>
                  )}
                  {/* The frame's OWN resolution, always — not the element's.
                      An <img> scales to its box and the box is the CSS's
                      choice, so the only honest place to read the pixels the
                      model produced is the number the file carries. */}
                  <br />
                  {/* The frame's stated size, or the fact that it did not
                      state one. The server refuses a frame with no resolution
                      -- "a picture that has been shrunk without saying so puts
                      every cell in the wrong place" -- but the viewer shim
                      hands the file's own frames straight through. */}
                  {sized(f) ? (
                    <>
                      {f.size[0]}×{f.size[1]}
                    </>
                  ) : (
                    <span className="warn">states no resolution</span>
                  )}
                  {f.downsampled && Array.isArray(f.decoded_size) && (
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
            {/* `steps_run` and `steps_requested` are `number | null`, and the
                null is load-bearing: this printed "1 of 0 step(s) decoded"
                for any file that did not carry them, which is a sentence
                about a run that cannot have happened. */}
            {!denoising ? (
              <>one picture, read in a single forward pass</>
            ) : length === null ? (
              <>
                {frames.length} decoded frame(s) ·{" "}
                <span className="warn">
                  this file does not say how many steps the run had, so these
                  frames are not a fraction of anything
                </span>
              </>
            ) : frames.length > length ? (
              <>
                {frames.length} frames ·{" "}
                <span className="warn">
                  this file says the run was {length} step(s) long, which is
                  fewer steps than there are frames — one of the two numbers in
                  it is wrong, and nothing here can say which
                </span>
              </>
            ) : (
              <>
                {frames.length} of {length} step(s) decoded
              </>
            )}
            {list(run.skipped_steps).length > 0 && (
              <>
                {" "}
                · {list(run.skipped_steps).length} ran and were not decoded (a
                choice)
              </>
            )}
            {list(run.steps_never_reached).length > 0 && (
              <>
                {" "}
                ·{" "}
                <span className="warn">
                  {list(run.steps_never_reached).length} were selected and
                  never arrived (a gap)
                </span>
              </>
            )}
            {run.steps_run !== null &&
              run.steps_requested !== null &&
              run.steps_run > run.steps_requested && (
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

      {linked > 0 && (
        <p className="meta warn">
          {linked} frame(s) in this file point at a picture somewhere else
          instead of carrying one, and were not rendered. A `.mri` never
          fetches: opening one must not tell whoever wrote it that you did.
        </p>
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
            {list(attn.steps).map((s) => {
              // `1e-9` is the floor, so an all-zero row divides by that
              // rather than by 0 — and `Math.max` of an EMPTY list is
              // -Infinity, which the floor also covers.
              const peak = Math.max(...list(s.per_token), 1e-9);
              return (
                <div key={s.step} className="irr-maprow">
                  <span className="meta irr-step">
                    step {s.step}
                    {s.timestep !== null && <> · t {measured(s.timestep, 0)}</>}
                  </span>
                  <div className="irr-cells">
                    {list(s.per_token).map((v, i) => (
                      <span
                        key={i}
                        className={`irr-cell${isPad(i) ? " pad" : ""}`}
                        style={{ opacity: Math.max(0.06, v / peak) }}
                        title={`${list(attn.tokens)[i] ?? `column ${i}`} — ${measured(v, 4)}${
                          isPad(i) ? " (padding, not your prompt)" : ""
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
            {/* `padding_from` is an INDEX, so 0 is not "none of it" — it is
                the claim that the padding starts at column zero and NONE of
                these columns is your prompt. This read a missing boundary as
                0 and then said so in prose over every measured column, which
                is the exact conclusion the field exists to prevent. */}
            {boundary === null ? (
              <span className="warn">
                where your prompt stops and the padding starts was not recorded
                in this file, so nothing here is dimmed — some of these columns
                are almost certainly `&lt;pad&gt;` and this cannot say which
              </span>
            ) : boundary >= columns ? (
              <>every measured column is your prompt — none of it is padding</>
            ) : (
              <>
                columns from {boundary} are padding, not your prompt — they
                carry real attention mass and are shown dimmed rather than
                labelled as words
              </>
            )}
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
            {list(readout.rows).map((r, i) => (
              <li key={i}>
                <b>{r.label}</b> <span className="meta">{measured(r.score, 4)}</span>
                {Array.isArray(r.box_xyxy) && (
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
