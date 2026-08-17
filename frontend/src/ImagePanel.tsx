import { CSSProperties, useEffect, useState } from "react";
import RestingSketch from "./RestingSketch";
import { useScanOnData } from "./useScanOnData";
import {
  captureImageAttention,
  errorText,
  getImage,
  getImageAvailable,
  imageAttentionCost,
  imageKnockout,
  imageStepsCost,
  ImageAttentionCost,
  ImageAttentionRun,
  ImageAvailable,
  ImageKnockout,
  ImageStatus,
  ImageTraceCost,
  loadImage,
  unloadImage,
} from "./api";

/** Text → image: which words the picture is looking at, and when.
 *
 *  A diffusion model attends to the prompt at every denoising step. Early
 *  steps decide layout, late steps decide texture, and a single averaged map
 *  hides that completely — so the step axis is kept rather than collapsed, and
 *  the grid below is steps down, words across.
 *
 *  ## Everything is gated on `capabilities`, never on the family's name
 *
 *  `imaging.detect` reads the checkpoint and answers what may be measured on
 *  it. A UNet pipeline offers cross-attention and knockout; a ViT offers patch
 *  attention and neither of those; an architecture the server cannot name
 *  offers an EMPTY list. So every control here asks `status.capabilities`
 *  rather than matching on a repo id — a panel drawn for the wrong family is a
 *  picture of something that does not exist, and it looks exactly like a
 *  picture of something that does.
 *
 *  ## Attention is only half of it
 *
 *  A word can be attended to and change nothing. That is why the knockout sits
 *  under the map rather than beside it: it removes one word, regenerates at
 *  the SAME seed, and measures what actually moved. The seed is doing the
 *  work — at a different seed per arm the numbers would be sampling noise with
 *  a word's name on them.
 */

/** The three things a cross-attention width can be, kept as three.
 *
 *  A positive width is the only one of them that permits a word-to-pixel map.
 *  **0 is UNCONDITIONAL** — the denoiser never sees a prompt, so there is
 *  nothing to draw and drawing anything would be inventing it. **`null` is
 *  that the denoiser's config never stated a width**, which is a gap in what
 *  is known rather than a property of the model.
 *
 *  Collapsing the last two is how a panel comes to tell somebody their model
 *  ignores their prompt because one config field was missing, so they are
 *  three branches here and three sentences below.
 */
function crossAttentionNote(dim: number | null): string {
  if (dim === null) {
    return (
      "The denoiser's config does not state a cross-attention width, so " +
      "nothing here knows how wide it is. That is a gap in what was read, " +
      "not a claim that there is none — the map below is still offered, and " +
      "the run itself will say if there was nothing to capture."
    );
  }
  if (dim === 0) {
    return (
      "This model is UNCONDITIONAL — no cross-attention to a prompt — so " +
      "there are no word-to-pixel maps here to draw. Nothing is offered " +
      "rather than a map of something that does not exist."
    );
  }
  return `It attends to prompt tokens through a ${dim}-wide cross-attention.`;
}

/** Enough digits to tell two readings apart, chosen once per RUN.
 *
 *  Deciding it cell by cell gives one column in whole numbers and its
 *  neighbour in thousandths, which reads as two different kinds of quantity —
 *  the same trap `VLAPanel.vec` was written for. These are attention masses
 *  summed over pixels, so the scale depends entirely on the latent resolution:
 *  a 64x64 map puts them in the hundreds, a 16x16 one in single figures.
 */
function masses(peak: number): (v: number) => string {
  const dp = peak >= 100 ? 0 : peak >= 1 ? 1 : 3;
  return (v: number) => v.toFixed(dp);
}

/** An RMS distance small enough that fixed decimals would print it as zero.
 *  A word that moved the image by 3e-5 moved it; "0.0000" says it did not. */
function distance(d: number): string {
  return d !== 0 && Math.abs(d) < 0.0001 ? d.toExponential(2) : d.toFixed(4);
}

/** The words a knockout will actually have arms for.
 *
 *  `image_attention.knockout` splits the prompt on whitespace and removes one
 *  of THESE per arm. The map's columns are the tokenizer's tokens, which is a
 *  different vocabulary — one word can be several tokens — and the panel says
 *  so rather than letting the two lists look like one.
 */
function promptWords(prompt: string): string[] {
  return prompt.split(/\s+/).filter((w) => w.length > 0);
}

export default function ImagePanel() {
  const [status, setStatus] = useState<ImageStatus | null>(null);
  const [available, setAvailable] = useState<ImageAvailable | null>(null);
  // Typed rather than picked: the list is what is cached, and somebody with
  // nothing cached still has a Hub id in their head.
  const [repo, setRepo] = useState("");
  const [prompt, setPrompt] = useState("a photograph of an astronaut riding a horse");
  const [steps, setSteps] = useState(20);
  // Optional for a capture, REQUIRED for a knockout. `null` is not 0: unfixed
  // means another run gives another trajectory, and the knockout's whole claim
  // dies without the same seed on every arm.
  const [seedFixed, setSeedFixed] = useState(true);
  const [seed, setSeed] = useState(0);
  const [run, setRun] = useState<ImageAttentionRun | null>(null);
  const [picked, setPicked] = useState<string[]>([]);
  const [knock, setKnock] = useState<ImageKnockout | null>(null);
  // Priced before anything is spent, and three separate questions: one render,
  // every arm of a knockout, and what keeping a latent per step would hold.
  const [renderCost, setRenderCost] = useState<ImageAttentionCost | null>(null);
  const [armsCost, setArmsCost] = useState<ImageAttentionCost | null>(null);
  const [traceCost, setTraceCost] = useState<ImageTraceCost | null>(null);
  const [costErr, setCostErr] = useState("");
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");
  // Set when a load was refused. Some of those refusals are overridable and
  // some are not, and the wire carries only the sentence — so this offers to
  // ask again with `confirm` rather than pretending to know which it was.
  const [refused, setRefused] = useState(false);
  const scanRef = useScanOnData(
    run ? `${run.model}:${run.seed}:${run.steps_measured}:${run.tokens.length}` : "",
  );

  // Status and the cached list only. Neither reads a weight: `/api/image`
  // reports what this process is holding, and `/api/image/available` reads
  // `model_index.json` and `config.json` off the disk. Nothing on this panel
  // opens a pipeline before you click.
  useEffect(() => {
    let live = true;
    void getImage()
      .then((s) => live && setStatus(s))
      .catch(() => undefined);
    void getImageAvailable()
      .then((a) => live && setAvailable(a))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, []);

  const caps = new Set(status?.capabilities ?? []);
  const words = promptWords(prompt);
  // A capability says the measurement EXISTS for this architecture. A measured
  // width of 0 says this particular checkpoint has no prompt to attend to, so
  // both have to hold before a map is offered — and the second is read off the
  // checkpoint rather than guessed from the first.
  const canCapture = caps.has("cross_attention") && status?.cross_attention_dim !== 0;
  const canKnock = caps.has("token_knockout") && status?.cross_attention_dim !== 0;
  const canTrace = caps.has("latent_trace");

  // THE PREFLIGHT, and it runs before any button is pressed rather than after
  // one is. Each line is a different question about the same `steps`, and each
  // is asked only when the capability behind it is present — a cost quoted for
  // a measurement this architecture cannot make is a number about nothing.
  const loaded = status?.loaded ?? false;
  const nWords = words.length;
  useEffect(() => {
    if (!loaded) return;
    let live = true;
    setCostErr("");
    const fail = (e: unknown) => live && setCostErr(errorText(e));
    if (canCapture) {
      void imageAttentionCost(steps, 0)
        .then((c) => live && setRenderCost(c))
        .catch(fail);
    }
    if (canKnock && nWords > 0) {
      void imageAttentionCost(steps, nWords)
        .then((c) => live && setArmsCost(c))
        .catch(fail);
    }
    if (canTrace) {
      void imageStepsCost(steps)
        .then((c) => live && setTraceCost(c))
        .catch(fail);
    }
    return () => {
      live = false;
    };
  }, [loaded, steps, nWords, canCapture, canKnock, canTrace]);

  async function onLoad(which: string, confirm = false) {
    setBusy("load");
    setErr("");
    setRefused(false);
    try {
      const s = await loadImage(which, confirm);
      setStatus(s);
      setRepo(which);
      // A new pipeline makes every reading on screen a claim about a model
      // that is no longer here.
      setRun(null);
      setKnock(null);
      setPicked([]);
    } catch (e) {
      // Verbatim. The server's refusals name the checkpoint, the two byte
      // counts and what to do about them, and a rewrite here would lose the
      // half that tells you which.
      setErr(errorText(e));
      setRefused(true);
    } finally {
      setBusy("");
    }
  }

  async function onUnload() {
    setBusy("unload");
    setErr("");
    try {
      setStatus(await unloadImage());
      setRun(null);
      setKnock(null);
      setPicked([]);
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }

  async function onCapture() {
    setBusy("capture");
    setErr("");
    setKnock(null);
    try {
      setRun(await captureImageAttention(prompt, steps, seedFixed ? seed : null));
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }

  async function onKnockout() {
    setBusy("knock");
    setErr("");
    try {
      setKnock(await imageKnockout(prompt, picked, seed, steps));
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }

  // ─────────────────────────────────────────────────────────────── resting

  if (!status?.loaded) {
    return (
      <div className="panel">
        <div className="sect">
          <span className="dot d-image" />
          <h2 className="h-image">IMAGE MODEL — WORDS TO PIXELS</h2>
          <span className="rule" />
        </div>
        <div className="resting">
          <RestingSketch kind="image" />
          <p>
            Which words a diffusion model is looking at, step by denoising step
            — and what actually changes when one of them is removed. Nothing is
            loaded yet.
          </p>

          {available && available.models.length > 0 && (
            <div className="image-models">
              {available.models.map((m) => (
                <div className="image-model" key={m.path}>
                  <span className="mid image-model-id">{m.path}</span>
                  {/* The family in the server's own words. The identifier is
                      kept beside it because that is what `capabilities` is
                      keyed on, and an unknown family carries its reason
                      instead of a bare row. */}
                  <span className="meta image-model-family">
                    {m.label}
                    {m.known ? ` · ${m.family}` : ""}
                  </span>
                  {m.known ? (
                    <button
                      className="green"
                      onClick={() => void onLoad(m.path)}
                      disabled={busy !== ""}
                    >
                      {busy === "load" ? "Loading…" : "Load"}
                    </button>
                  ) : (
                    <span className="meta">{m.reason}</span>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Rendered as the server wrote it: how many are cached, how many
              of them this can open, and that nothing was downloaded to find
              out. A count re-typed here could drift from the list above it. */}
          {available && <span className="meta">{available.means}</span>}

          <div className="row">
            <label className="meta" htmlFor="image-repo">
              or a checkpoint by name
            </label>
            <input
              id="image-repo"
              className="share-note"
              placeholder="stabilityai/sd-turbo, or a directory on this machine"
              value={repo}
              onChange={(e) => setRepo(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && repo.trim() && void onLoad(repo)}
              spellCheck={false}
            />
            <button
              className="green"
              onClick={() => void onLoad(repo)}
              disabled={busy !== "" || repo.trim() === ""}
            >
              {busy === "load" ? "Loading pipeline…" : "Load it"}
            </button>
          </div>
          <span className="meta">
            There is no default worth guessing: the checkpoint decides which
            controls apply, so nothing is loaded until you name one. What is
            identified from JSON, scanned for anything that executes on load,
            and priced against this machine — three refusals that cost
            nothing — happens before a byte moves.
          </span>

          {/* The status's own sentence about why nothing is held. It is the
              only line that can distinguish "not loaded yet" from "the last
              one was unloaded and its memory handed back". */}
          {status && <span className="meta">{status.means}</span>}
        </div>

        {err && <div className="hint err">{err}</div>}
        {err && refused && (
          <div className="row">
            <button
              className="ghost sm"
              onClick={() => void onLoad(repo.trim() || status?.repo || "", true)}
              disabled={busy !== "" || (repo.trim() === "" && !status?.repo)}
            >
              ask again with confirm
            </button>
            <span className="meta">
              Some of those refusals can be overridden — holding a pipeline
              beside a resident text model, mainly. One that cannot answers
              again with the same sentence.
            </span>
          </div>
        )}
      </div>
    );
  }

  // ──────────────────────────────────────────────────────────────── loaded

  // The family's own words, when the cached list carried them for this repo.
  // Not derived here: `ImageStatus` sends the identifier and `ImageModelInfo`
  // sends the prose, so this reads the prose across rather than inventing a
  // mapping that would be a second place for family names to live.
  const label = available?.models.find((m) => m.path === status.repo)?.label ?? "";
  const dim = status.cross_attention_dim;

  // Columns are the REAL prompt tokens. CLIP pads to 77 and the padding
  // carries genuine attention mass — a finding, and an unreadable chart — so
  // the padded tail is reported below rather than plotted as sixty blank
  // words.
  const padded =
    run !== null && run.padding_from > 0 && run.padding_from < run.tokens.length;
  const columns = run ? (padded ? run.tokens.slice(0, run.padding_from) : run.tokens) : [];
  const peak = run
    ? Math.max(
        0,
        ...run.steps.flatMap((s) => s.per_token.slice(0, columns.length)),
      )
    : 0;
  const fmt = masses(peak);

  return (
    <div ref={scanRef} className="panel image">
      <div className="sect">
        <span className="dot d-image" />
        <h2 className="h-image">IMAGE MODEL — WHICH WORDS THE PICTURE LOOKED AT</h2>
        <span className="rule" />
      </div>

      <div className="row" style={{ marginBottom: 10 }}>
        <span className="pill on">{status.repo}</span>
        {/* The identifier is what the capability list is keyed on, so it is
            shown; the prose sits beside it when the disk scan supplied it. */}
        <span className="pill">{label ? `${label} · ${status.family}` : status.family}</span>
        <span className="pill">
          {status.device || "device not stated"}
          {status.dtype ? ` · ${status.dtype}` : ""}
        </span>
        {/* Read from the checkpoint's own headers. Zero is not rendered as a
            size: it means no weight file could be measured, which is a
            different fact from a pipeline that weighs nothing. */}
        <span className="pill">
          {status.bytes_resident > 0
            ? `${(status.bytes_resident / 1e9).toFixed(2)} GB resident`
            : "resident weights could not be sized"}
        </span>
        <span className="pill">
          {dim === null
            ? "cross-attention width unknown"
            : dim === 0
              ? "unconditional — no cross-attention"
              : `cross-attention ${dim} wide`}
        </span>
        <span className="spacer" />
        <button className="ghost sm" onClick={() => void onUnload()} disabled={busy !== ""}>
          {busy === "unload" ? "unloading…" : "unload"}
        </button>
      </div>

      <p className="meta">{status.means}</p>
      <p className="meta">{crossAttentionNote(dim)}</p>

      {status.capabilities.length === 0 && (
        <div className="hint">
          This is an architecture the server could not name, so it offers no
          measurements at all rather than every measurement. Nothing below is
          shown because nothing below could be honest about this checkpoint.
        </div>
      )}

      {/* ─── the run, and what it costs before it is made ─────────────── */}
      {(canCapture || canKnock) && (
        <>
          <label className="meta" htmlFor="image-prompt">
            prompt
          </label>
          <input
            id="image-prompt"
            className="share-note image-prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            spellCheck={false}
          />

          <div className="row image-controls">
            <label className="meta" htmlFor="image-steps">
              steps
            </label>
            <input
              id="image-steps"
              type="number"
              min={1}
              max={200}
              value={steps}
              onChange={(e) => setSteps(Math.max(1, Math.min(200, Number(e.target.value) || 1)))}
            />
            <label className="meta">
              <input
                type="checkbox"
                checked={seedFixed}
                onChange={(e) => setSeedFixed(e.target.checked)}
              />{" "}
              fix the seed
            </label>
            <input
              type="number"
              min={0}
              aria-label="seed"
              value={seed}
              disabled={!seedFixed}
              onChange={(e) => setSeed(Math.max(0, Number(e.target.value) || 0))}
            />
            {canCapture && (
              <button
                className="cta"
                onClick={() => void onCapture()}
                disabled={busy !== "" || prompt.trim() === ""}
              >
                {busy === "capture" ? "Denoising…" : "Capture attention"}
              </button>
            )}
          </div>

          {/* BEFORE the run, not after it. Every line is the server's own
              sentence: what a render costs, what every arm of a knockout
              costs, and what keeping a latent per step would hold. */}
          <div className="image-cost">
            {renderCost && (
              <p className="meta">
                <b>one capture</b> · {renderCost.means}
              </p>
            )}
            {armsCost && canKnock && (
              <p className="meta">
                <b>a knockout of this prompt</b> · {armsCost.means}
              </p>
            )}
            {traceCost && (
              <p className="meta">
                <b>keeping a latent per step</b> · {traceCost.means}
              </p>
            )}
            {costErr && <p className="meta">{costErr}</p>}
          </div>
        </>
      )}

      {err && <div className="hint err">{err}</div>}

      {/* ─── the map: words across, steps down ────────────────────────── */}
      {run && columns.length > 0 && (
        <>
          <div className="image-grid-wrap">
            <table
              className="image-grid"
              aria-label="cross-attention mass per prompt token, per denoising step"
            >
              <thead>
                <tr>
                  <th />
                  {columns.map((t, i) => (
                    <th key={i} className="mid" title={t}>
                      {t.trim() || "␣"}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="stagger">
                {run.steps.map((s, ri) => (
                  <tr key={s.step} style={{ "--i": ri } as CSSProperties}>
                    <th className="mid" title={`scheduler timestep ${s.timestep}`}>
                      {s.step}
                    </th>
                    {columns.map((t, ci) => {
                      const v = s.per_token[ci];
                      // A column the run did not report is not a zero. It
                      // happens when a step captured fewer blocks than the
                      // map has columns, and an empty cell says so.
                      if (v === undefined) {
                        return (
                          <td key={ci} className="image-cell missing" title="not reported">
                            ·
                          </td>
                        );
                      }
                      return (
                        <td
                          key={ci}
                          className="image-cell"
                          style={{
                            background: `color-mix(in oklab, var(--color-image) ${
                              peak > 0 ? (v / peak) * 100 : 0
                            }%, transparent)`,
                          }}
                          title={`step ${s.step} (timestep ${s.timestep}), ${t} — ${v}, from ${s.blocks} cross-attention block(s)`}
                        >
                          {fmt(v)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="meta image-read">
            Rows are denoising steps in the order they ran, columns are the
            tokenizer's tokens. The number in each cell is the attention mass
            that token received, summed over pixels and averaged over heads and
            over the {run.resolutions.length} resolution
            {run.resolutions.length === 1 ? "" : "s"} that reported (
            {run.resolutions.join(", ")}). The SHADE is that number as a share
            of the largest cell in this run ({fmt(peak)}) — the shade is
            derived here, the number is the measurement.
          </div>

          {padded && (
            <div className="meta">
              Columns from index {run.padding_from} of {run.tokens.length} are
              padding rather than your prompt. They carry real attention mass
              and are deliberately not plotted as words.
            </div>
          )}

          {run.steps_measured < run.steps_requested && (
            <div className="hint">
              {run.steps_requested - run.steps_measured} of the{" "}
              {run.steps_requested} requested steps were not captured, so the
              rows above are the {run.steps_measured} that were.
            </div>
          )}

          {/* The server's own paragraph, including the sentence that matters
              most on this panel: attention is not a cause. */}
          <p className="meta image-means">{run.means}</p>
        </>
      )}

      {/* ─── the interventional half ──────────────────────────────────── */}
      {run && canKnock && (
        <div className="image-knock">
          {/* A sub-heading, not a second `.sect`: SectionNav treats every
              `.sect` with an h2 as a place to jump to, and this is half of one
              panel rather than a section of the page. */}
          <div className="image-subhead">
            <h3 className="h-image">KNOCKOUT — WHAT A WORD ACTUALLY DID</h3>
            <span className="rule" />
          </div>
          <p className="meta">
            Pick the words you want the answer for. The arms are the prompt's
            whitespace-separated words, which is a different vocabulary from
            the map's columns above — one word can be several tokens — and the
            run removes <b>every</b> word in turn rather than only the ones
            picked, so your picks are marked in the result rather than
            narrowing the work.
          </p>

          <div className="image-words">
            {words.map((w, i) => (
              <button
                key={`${w}:${i}`}
                className={`tok${picked.includes(w) ? " pin" : ""}`}
                aria-pressed={picked.includes(w)}
                onClick={() =>
                  setPicked((prev) =>
                    prev.includes(w) ? prev.filter((x) => x !== w) : [...prev, w],
                  )
                }
              >
                {w}
              </button>
            ))}
          </div>

          <div className="row">
            <button
              className="cta"
              onClick={() => void onKnockout()}
              disabled={busy !== "" || picked.length === 0 || !seedFixed}
            >
              {busy === "knock" ? "Regenerating, one word at a time…" : "Knock words out"}
            </button>
            {!seedFixed ? (
              <span className="meta">
                A knockout needs a fixed seed. Every arm has to run at the
                identical one or the difference between two images is the
                sampler rather than the word.
              </span>
            ) : picked.length === 0 ? (
              <span className="meta">
                Pick at least one word — which words matter is the question,
                not something for this to choose.
              </span>
            ) : (
              <span className="meta">
                {words.length} arms plus the unmodified prompt, all at seed{" "}
                {seed}.
              </span>
            )}
          </div>

          {knock && knock.arms.length > 0 && (
            <>
              <ol className="image-arms stagger">
                {knock.arms.map((a, i) => {
                  const top = knock.arms[0].distance;
                  return (
                    <li
                      key={`${a.word}:${a.index}`}
                      className={knock.tokens.includes(a.word) ? "asked" : undefined}
                      style={{ "--i": i } as CSSProperties}
                      title={a.prompt_without}
                    >
                      <span className="mid image-arm-word">{a.word}</span>
                      <span className="image-arm-track">
                        <span
                          className="image-arm-bar"
                          style={{
                            width: `${top > 0 ? Math.min(100, (a.distance / top) * 100) : 0}%`,
                          }}
                        />
                      </span>
                      <span className="mid image-arm-val">{distance(a.distance)}</span>
                    </li>
                  );
                })}
              </ol>
              <p className="meta">
                The bar is each row against the furthest-moving one; the number
                is the measured RMS distance itself. Rows you picked are
                outlined — every word was measured either way.
              </p>
              <p className="meta image-means">{knock.means}</p>
            </>
          )}
        </div>
      )}
    </div>
  );
}
