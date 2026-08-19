import { CSSProperties, useEffect, useRef, useState } from "react";
import { measured, signed } from "./measured";
import DevicePicker from "./DevicePicker";
import LoadBar, { gb } from "./LoadBar";
import RestingSketch from "./RestingSketch";
import ImageModelPicker from "./ImageModelPicker";
import AdapterPanel from "./AdapterPanel";
import ImageCV from "./ImageCV";
import ImageSteps from "./ImageSteps";
import { useScanOnData } from "./useScanOnData";
import {
  captureImageAttention,
  errorText,
  getImage,
  getImageLocal,
  imageAttentionCost,
  imageAttribution,
  imageAttributionCost,
  imageKnockout,
  imageSize,
  imageStepsCost,
  ImageAttentionCost,
  ImageAttentionRun,
  ImageAttribution,
  ImageAttributionCost,
  ImageAttributionWindow,
  ImageKnockout,
  ImageLocal,
  ImageSize,
  ImageStatus,
  LoadProgress,
  getImageProgress,
  cancelImageLoad,
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
  // The DECIMALS follow the peak, because the scale of the whole map does;
  // the small end still goes through `measured`, because a single cell can be
  // far below the peak and a cell that holds 4e-4 of the mass holds some.
  return (v: number) => measured(v, dp);
}

/** An RMS distance small enough that fixed decimals would print it as zero.
 *  A word that moved the image by 3e-5 moved it; "0.0000" says it did not. */
const distance = (d: number) => measured(d, 4);

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

/** Formatting a model's size, and a download count, moved to
 *  `ImageModelPicker` along with the only lists that ever rendered one.
 *
 *  Deliberately not re-exported back into this file. The rule those functions
 *  carry — **`size_bytes: null` is UNKNOWN and must never come out as
 *  "0.0 GB"** — survives by there being exactly one place that formats a size,
 *  and a second copy of it here would be a second place to forget it. This
 *  panel reports sizes only through the server's own `means` sentences, which
 *  are the only ones that can tell "already here, nothing would move" from
 *  "publishes no size metadata, so this is UNKNOWN rather than small".
 */

/** The four occluders `vision_attr.FILLS` accepts.
 *
 *  Typed here because no route publishes the list, and the module is the
 *  source of truth — a name outside it is refused server-side and says so.
 *  None of them is neutral, which is the point of shipping several: a finding
 *  that only survives one fill is a finding about the fill.
 */
const FILLS = ["grey", "black", "white", "image_mean"];

/** A SIGNED score on a DIVERGING scale with a neutral midpoint at zero.
 *
 *  This is the one rule the map cannot bend. A positive drop means covering
 *  that window COST the class its evidence; a negative one means covering it
 *  HELPED — a region that was arguing against the class, which is a real
 *  finding about the picture and not an error in the sweep. An absolute value
 *  paints those two identically, and a single-hue ramp paints "argued
 *  against" as a paler shade of "argued for": both erase the distinction the
 *  sign exists to carry.
 *
 *  Two hues the stylesheet actually defines, and normalised by the largest
 *  MAGNITUDE either way so zero lands on transparent in both directions.
 *  `PatchPanel.cell` records why the token names matter: `color-mix` with an
 *  undefined custom property yields transparent rather than an error, so a
 *  misspelt token silently deletes half a diverging scale.
 */
function attrShade(v: number, mag: number): string {
  if (mag <= 0 || v === 0) return "transparent";
  // Capped below full strength: this paints OVER the picture, and an opaque
  // cell would hide the region it is a claim about.
  const a = Math.min(1, Math.abs(v) / mag) * 76;
  return v > 0
    ? `color-mix(in oklab, var(--color-image) ${a}%, transparent)`
    : `color-mix(in oklab, var(--color-probe) ${a}%, transparent)`;
}

/** A signed logit movement, with the sign always shown.
 *
 *  The sign is the reading, so it is never dropped and never implied by a
 *  colour alone. Scores arrive at six decimals and a map can span a
 *  thousandth of a logit, so a value too small for fixed decimals goes to
 *  exponential rather than printing as "0.0000" — a window that moved the
 *  logit by 3e-5 moved it.
 */
const drop = (v: number) => signed(v, 4);

/** Where one window sat, in the pixels of the tensor the model saw. */
function box(w: { top: number; left: number; height: number; width: number }): string {
  return `pixels ${w.top}-${w.top + w.height} by ${w.left}-${w.left + w.width}`;
}

/** The span a peak has to be read against, at the precision it was measured.
 *
 *  Six decimals rather than four, because that is what the scores are
 *  reported at and the whole question this number answers is whether the map
 *  is bigger than its own rounding. A span printed as "0.000000" would hide
 *  exactly the case it exists to expose, so anything below four decimals goes
 *  to exponential instead.
 */
function span(v: number): string {
  return v > 0 && v < 0.0001 ? v.toExponential(2) : v.toFixed(6);
}

/** Which of the two image sections this instance is.
 *
 *  `diffusion` is text-to-image: a prompt goes in, pixels come out, and the
 *  questions are about which words the picture attended to and when it
 *  settled. `vision` is the other direction entirely -- pixels go in and a
 *  label, a box or a mask comes out -- and the questions are what it said and
 *  what supported it. Sharing one panel made the category bar offer "Text ->
 *  Image" for a segmentation model.
 */
export type ImageKind = "diffusion" | "vision";

/** The capabilities each section is the home for.
 *
 *  Read against the checkpoint's own capability list, never against its repo
 *  id or family name — the same rule the individual controls already follow.
 */
const OWNED: Record<ImageKind, readonly string[]> = {
  diffusion: ["cross_attention", "token_knockout", "step_commit", "latent_trace"],
  vision: ["patch_attention", "attribution", "layer_readout"],
};

/** The families each section is the home for.
 *
 *  Section membership is a question about WHAT THE MODEL IS; the individual
 *  controls ask what can be measured on it, and those are different
 *  questions. They were the same test until the capability list started being
 *  checked against the real pipeline — at which point a checkpoint supporting
 *  none of the four (a class-conditioned DiT) stopped belonging to any
 *  section, and the panel showed its resting sketch with 3.3 GB of that model
 *  resident. "I loaded it and the panel went blank" is worse than a refusal.
 */
const OWNED_FAMILIES: Record<ImageKind, readonly string[]> = {
  diffusion: ["unet_diffusion", "dit_diffusion"],
  vision: ["vit", "detector", "segmenter", "vlm"],
};

export default function ImagePanel({ kind = "diffusion" }: { kind?: ImageKind } = {}) {
  const [status, setStatus] = useState<ImageStatus | null>(null);
  // What is on this disk, sized. `null` is "the scan has not answered yet",
  // which is a different thing from an empty list — the second is a real
  // finding and gets the server's own sentence under it.
  const [local, setLocal] = useState<ImageLocal | null>(null);
  // The sheet, and whether it is up. Every list of models lives inside it —
  // three sources, a search and the counts — so the resting panel keeps one
  // control instead of a wall of rows that is either empty or too long.
  const [pickerOpen, setPickerOpen] = useState(false);
  // What one named repo weighs, asked before anything moves.
  const [sized, setSized] = useState<ImageSize | null>(null);
  // Typed rather than picked: the lists are what is cached and what the Hub
  // returned, and somebody with neither still has an id in their head.
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

  // ── occlusion attribution, for the checkpoints that have a class to lose
  //
  // The picture is held as a DATA URL and there is no path box anywhere near
  // it. A path in a request body names a file on the SERVER's disk — which is
  // somebody else's machine as often as it is yours — and a browser cannot
  // produce one for a file a person picked in any case.
  const [picture, setPicture] = useState("");
  const [pictureName, setPictureName] = useState("");
  // What the FILE is, which is not what the model sees: the checkpoint's own
  // processor resizes (and may crop) before a window is ever placed.
  const [pictureDims, setPictureDims] = useState<{ w: number; h: number } | null>(null);
  const [aPatch, setAPatch] = useState(16);
  const [aStride, setAStride] = useState(16);
  const [aFill, setAFill] = useState("grey");
  const [aBatch, setABatch] = useState(32);
  // Empty means "whatever the model itself predicted", which is the ordinary
  // question. A number asks a different one — auditing a class you named —
  // and the result says which of the two it answered.
  const [aTarget, setATarget] = useState("");
  const [attrCost, setAttrCost] = useState<ImageAttributionCost | null>(null);
  const [attrCostErr, setAttrCostErr] = useState("");
  const [attr, setAttr] = useState<ImageAttribution | null>(null);
  const pickRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState("");
  // Which device the NEXT load goes to. "" is Automatic, which sends nothing
  // and lets the server detect — see DevicePicker for why that is not the
  // same as preselecting the card it would have chosen.
  const [device, setDevice] = useState("");
  const [prog, setProg] = useState<LoadProgress | null>(null);
  // What a STOPPED load said. Deliberately not `err`: nothing went
  // wrong, somebody pressed Stop, and the red box that `err` renders
  // would contradict both the 200 the route answers and the
  // `cancelled` stage the tracker publishes.
  const [stopped, setStopped] = useState("");
  const [err, setErr] = useState("");
  // Set when a load was refused. Some of those refusals are overridable and
  // some are not, and the wire carries only the sentence — so this offers to
  // ask again with `confirm` rather than pretending to know which it was.
  const [refused, setRefused] = useState(false);
  // WHICH checkpoint the in-flight load is for. Without it every row in the
  // list reads "Loading…" at once, and the retry above has nothing to retry:
  // a load from the list never touches the name box, so the box would send an
  // empty string back to a route that refuses one.
  const [tried, setTried] = useState("");
  const scanRef = useScanOnData(
    run ? `${run.model}:${run.seed}:${run.steps_measured}:${run.tokens.length}` : "",
  );

  // Status and the local list only. Neither opens a pipeline: `/api/image`
  // reports what this process is holding, and `/api/image/local` reads
  // `model_index.json` and `config.json` off the disk and sizes the weight
  // files beside them without reading one. Nothing on this panel loads
  // anything before you click, and nothing here touches the network.
  useEffect(() => {
    let live = true;
    void getImage()
      .then((s) => live && setStatus(s))
      .catch(() => undefined);
    void getImageLocal()
      .then((l) => live && setLocal(l))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, []);

  // Read once, here, because three effects below branch on it and each of them
  // is the resting panel deciding whether it is the thing on screen at all.
  const loaded = status?.loaded ?? false;

  const caps = new Set(status?.capabilities ?? []);
  const words = promptWords(prompt);
  // A capability says the measurement EXISTS for this architecture. A measured
  // width of 0 says this particular checkpoint has no prompt to attend to, so
  // both have to hold before a map is offered — and the second is read off the
  // checkpoint rather than guessed from the first.
  // Does the LOADED checkpoint belong to this section? One image model is
  // resident at a time, so the other section shows its resting copy rather
  // than a set of controls that would act on somebody else's model.
  // The FAMILY decides the section; the capabilities decide the controls.
  // Falling back to the capability test keeps this working for a family this
  // build does not have in the table above, which is the case a new
  // architecture arrives as.
  const mine =
    (status?.loaded &&
      (OWNED_FAMILIES[kind].includes(status.family) ||
        OWNED[kind].some((c) => caps.has(c)))) ||
    false;

  // What this architecture offers that THIS checkpoint cannot do, with the
  // server's reason for each. Absent controls with no explanation read as a
  // missing feature; this reads as a fact about the model in front of you.
  const withheld = Object.entries(status?.unavailable ?? {}).filter(([c]) =>
    OWNED[kind].includes(c),
  );
  const canCapture =
    mine && caps.has("cross_attention") && status?.cross_attention_dim !== 0;
  const canKnock =
    mine && caps.has("token_knockout") && status?.cross_attention_dim !== 0;
  const canTrace = mine && caps.has("latent_trace");
  // A ViT, a detector or a segmentation head has something to lose when a
  // region is covered. A diffusion pipeline has NO class logit to move, so it
  // never carries this capability and the whole block below is absent for it —
  // gated on what the server said, never on what the repo id looked like.
  const canAttribute = mine && caps.has("attribution");
  // A classifier, detector or segmenter can be ASKED what it thinks, which
  // is a different question from what supports the answer. Every family
  // that carries `attribution` or `patch_attention` has a prediction to
  // report; a diffusion pipeline has neither and gets no control.
  const canPredict =
    mine && (caps.has("attribution") || caps.has("patch_attention"));
  const canReadout = mine && caps.has("layer_readout");

  // WHICH geometry the preflight prices, and the two are not interchangeable.
  //
  // The sweep runs on the tensor the checkpoint's own processor produced, not
  // on the file that was picked: a 4000x3000 photograph reaches the model as
  // 224x224, so pricing the file would quote a map a hundred times the size of
  // the one the run actually makes. `image_size` is that input size when the
  // checkpoint states it. When it does not, the file's own dimensions are the
  // only geometry anything here knows — priced, and labelled as the weaker
  // claim it is rather than passed off as the run's.
  const shotSize = status && status.image_size && status.image_size > 0 ? status.image_size : 0;
  const pricedH = shotSize || pictureDims?.h || 0;
  const pricedW = shotSize || pictureDims?.w || 0;

  // THE PREFLIGHT, and it runs before any button is pressed rather than after
  // one is. Each line is a different question about the same `steps`, and each
  // is asked only when the capability behind it is present — a cost quoted for
  // a measurement this architecture cannot make is a number about nothing.
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

  // The occlusion preflight, on the same principle and one step earlier: this
  // route needs no model at all, so the number arrives as soon as a geometry
  // is known — before a picture has even been picked, when the checkpoint
  // states its own input size. It is the number that decides whether to run,
  // because the same image at stride 1 rather than stride 16 is not a slower
  // run, it is a different afternoon.
  useEffect(() => {
    if (!loaded || !canAttribute || pricedH < 1 || pricedW < 1) return;
    let live = true;
    setAttrCostErr("");
    // Dropped before the new one is asked for, not after it arrives. The Run
    // button is gated on this number, so a price left on screen from the
    // previous stride is a button offering a run that the ceiling it was
    // checked against no longer describes.
    setAttrCost(null);
    void imageAttributionCost(pricedH, pricedW, aPatch, aStride, aBatch)
      .then((c) => live && setAttrCost(c))
      .catch((e) => {
        if (!live) return;
        // A refused geometry has no cost, and a stale one from the last
        // setting would price a schedule nobody is about to run.
        setAttrCost(null);
        setAttrCostErr(errorText(e));
      });
    return () => {
      live = false;
    };
  }, [loaded, canAttribute, pricedH, pricedW, aPatch, aStride, aBatch]);

  // A cold image load is minutes long — 6.3 GB of pickle weights off a synced
  // drive was the case that prompted this — and with no heartbeat a slow load
  // and a hung one look identical. They were reported as the same bug.
  useEffect(() => {
    if (busy !== "load") {
      setProg(null);
      return;
    }
    let live = true;
    const tick = () =>
      void getImageProgress()
        .then((q) => live && setProg(q))
        .catch(() => {});
    tick();
    const id = window.setInterval(tick, 700);
    return () => {
      live = false;
      window.clearInterval(id);
    };
  }, [busy]);

  async function onStopLoad() {
    try {
      await cancelImageLoad();
    } catch {
      /* the load reports its own outcome either way */
    }
  }

  async function onLoad(which: string, confirm = false) {
    setBusy("load");
    setErr("");
    setStopped("");
    setRefused(false);
    setTried(which);
    try {
      const s = await loadImage(which, confirm, device);
      // Stopped on request. NOT an error: leave every reading on screen
      // exactly as it was, because nothing about the loaded model changed.
      if ("cancelled" in s) {
        setStopped(s.message);
        return;
      }
      setStatus(s);
      // A new pipeline makes every reading on screen a claim about a model
      // that is no longer here.
      setRun(null);
      setKnock(null);
      setPicked([]);
      // The picture stays — it is the reader's file, not a reading — but the
      // map over it was a claim about a checkpoint that is no longer here.
      setAttr(null);
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

  /** How big one named repo is, before anything moves.
   *
   *  The name box could always start a load; it could never say what the load
   *  was about to cost. This is the question a reader asks first, and asking
   *  it downloads nothing. */
  async function onSize() {
    setBusy("size");
    setErr("");
    setSized(null);
    try {
      setSized(await imageSize(repo));
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }

  /** A pick fills the box; it does not start a load.
   *
   *  Opening a diffusion pipeline is minutes and gigabytes, so the click that
   *  starts one is the Load button beside the trigger and never a row in a
   *  sheet that closes under you — the same grammar the text picker uses. The
   *  size goes with the old name: leaving the last answer up while the box
   *  says something else is how a reader prices one checkpoint and loads
   *  another. */
  function onPickRepo(next: string) {
    setRepo(next);
    setSized(null);
    setErr("");
    setRefused(false);
    setPickerOpen(false);
  }

  async function onUnload() {
    setBusy("unload");
    setErr("");
    try {
      setStatus(await unloadImage());
      setRun(null);
      setKnock(null);
      setPicked([]);
      // The picture stays — it is the reader's file, not a reading — but the
      // map over it was a claim about a checkpoint that is no longer here.
      setAttr(null);
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

  /** The picked file, as a data URL, plus what it actually is.
   *
   *  Read in the browser and sent in the body. Nothing here uploads a path:
   *  see the note on `picture` above for why there is no box to type one in.
   */
  function onPick(file: File | undefined) {
    // A new picture makes any map on screen a map of the previous one.
    setAttr(null);
    setErr("");
    if (!file) {
      setPicture("");
      setPictureName("");
      setPictureDims(null);
      return;
    }
    const reader = new FileReader();
    reader.onerror = () =>
      setErr(
        `${file.name} could not be read from disk. Nothing was sent — this ` +
          `failed in the browser, before the measurement was asked for.`,
      );
    reader.onload = () => {
      const url = typeof reader.result === "string" ? reader.result : "";
      if (!url.startsWith("data:image/")) {
        setErr(
          `${file.name} did not read back as an image. The sweep needs a ` +
            `picture the checkpoint's own processor can open.`,
        );
        return;
      }
      setPicture(url);
      setPictureName(file.name);
      // Decoded here only to SAY what was picked, and — when the checkpoint
      // never states its own input size — to have a geometry to price at all.
      const probe = new Image();
      probe.onload = () =>
        setPictureDims({ w: probe.naturalWidth, h: probe.naturalHeight });
      probe.onerror = () => setPictureDims(null);
      probe.src = url;
    };
    reader.readAsDataURL(file);
  }

  async function onAttribute() {
    setBusy("attribute");
    setErr("");
    try {
      const named = aTarget.trim();
      setAttr(
        await imageAttribution(
          picture,
          aPatch,
          aStride,
          aFill,
          aBatch,
          named === "" ? null : Number(named),
        ),
      );
    } catch (e) {
      // Cleared rather than kept: a refusal beside the previous run's map
      // reads as though the map on screen is the answer to what was just
      // asked, and it is the answer to something else.
      setAttr(null);
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }

  // ─────────────────────────────────────────────────────────────── resting

  // `!mine` counts as resting for THIS section. One image model is resident at
  // a time, and without this the other section rendered its full loaded state
  // about a model that is not its kind: with a ViT held, the text-to-image
  // panel reported "cross-attention width unknown", explained what the
  // DENOISER's config failed to state, and offered a map "below" that its own
  // capability gate had already removed. Several paragraphs about a model
  // nobody loaded here.
  if (!status?.loaded || !mine) {
    return (
      <div className="panel">
        <div className="sect">
          <span className={`dot ${kind === "vision" ? "d-vision" : "d-image"}`} />
          <h2 className={kind === "vision" ? "h-vision" : "h-image"}>{kind === "vision" ? "VISION MODEL — PIXELS TO AN ANSWER" : "IMAGE MODEL — WORDS TO PIXELS"}</h2>
          <span className="rule" />
        </div>
        <div className="resting">
          <RestingSketch kind="image" />
          {/* Each section says what IT is for. The other one's sentence is
              not a smaller version of this one — pixels-to-a-label and
              words-to-pixels are opposite directions, and one paragraph
              covering both is how they ended up in one panel. */}
          <p>
            {kind === "vision" ? (
              <>
                What a classifier, detector or segmenter says about a picture,
                and which parts of it the answer actually rested on. Nothing is
                loaded yet.
              </>
            ) : (
              <>
                Which words a diffusion model is looking at, step by denoising
                step — and what actually changes when one of them is removed.
                Nothing is loaded yet.
              </>
            )}
          </p>
          {/* Says which model is in the way, rather than looking empty. One
              image model is resident at a time, so "loaded, but not the kind
              this section measures" is a real state and a common one. */}
          {status?.loaded && !mine && (
            <p className="meta">
              {status.repo} is loaded, and it is a {status.family} model —
              measured in the{" "}
              {kind === "vision" ? "text-to-image" : "vision"} section rather
              than this one. Loading something here replaces it.
            </p>
          )}

          {/* --- the picker ------------------------------------------
              One trigger and one Load button, the same pair the text
              workbench has, and the reason this replaced a flat list. That
              list rendered every cached checkpoint with its own Load button:
              a wall on a machine that has downloaded a lot, an empty box on
              one that has downloaded nothing. Both are answered inside the
              sheet now, where three sources can each explain their own empty
              result. The trigger keeps saying what is chosen, so the
              collapsed control still answers that on its own. */}
          <div className="row">
            <button
              className="model-btn glass"
              onClick={() => setPickerOpen(true)}
              disabled={busy !== ""}
              aria-haspopup="dialog"
              aria-expanded={pickerOpen}
              aria-controls="image-model-sheet"
            >
              <span className="model-btn-label">model</span>
              <span className="model-btn-id">{repo.trim() || "none chosen"}</span>
              <span className="model-btn-caret">⌄</span>
            </button>
            <button
              className="green"
              onClick={() => void onLoad(repo)}
              disabled={busy !== "" || repo.trim() === ""}
            >
              {busy === "load" && tried === repo ? "Loading pipeline…" : "Load"}
            </button>
            {/* Renders nothing on a machine with one device — see
                DevicePicker. Disabled during a load because it describes the
                NEXT one, and a control that appears to change a job already
                running is a lie about what the click did. */}
            <DevicePicker
              value={device}
              onChange={setDevice}
              disabled={busy !== ""}
            />
          </div>

          {/* Only while something is loading. The report that produced this
              was "its not loading the model its been a long time", against a
              checkpoint that was loading correctly and slowly — a wait with
              no heartbeat cannot be told apart from a hang. The same bar the
              text side uses, so the two cannot drift. */}
          {busy === "load" && (
            <LoadBar p={prog} id={tried || repo} onStop={() => void onStopLoad()} />
          )}

          {/* The same selection, typed. The box and the trigger are ONE
              state on purpose: two that can disagree is how somebody prices
              one checkpoint, reads the answer, and loads another. A path is
              also the only way to reach an entry the walk could not size,
              which the sheet says so on the row rather than offering a
              button that may not work. */}
          <div className="row">
            <label className="meta" htmlFor="image-repo">
              or a checkpoint by name
            </label>
            <input
              id="image-repo"
              className="share-note"
              placeholder="stabilityai/sd-turbo, or a directory on this machine"
              value={repo}
              onChange={(e) => {
                setRepo(e.target.value);
                // A size belongs to the name it was asked about. Leaving the
                // last answer up while the box says something else is how a
                // reader prices one checkpoint and loads another.
                setSized(null);
              }}
              onKeyDown={(e) => e.key === "Enter" && repo.trim() && void onLoad(repo)}
              spellCheck={false}
            />
            {/* Asked before the click, not discovered during it. The name box
                could always start a load and could never say what it was
                about to cost — which is the one thing a reader wants from it
                and the one thing a typed id does not carry. */}
            <button
              className="ghost sm"
              onClick={() => void onSize()}
              disabled={busy !== "" || repo.trim() === ""}
            >
              {busy === "size" ? "asking…" : "How big is it?"}
            </button>
          </div>

          {/* The server's own sentence, which is the only one that can tell
              "already here, nothing would move" from "publishes no size
              metadata, so this is UNKNOWN rather than small". Re-deriving
              either from `size_bytes` here would be this panel making the
              claim instead of reporting it. */}
          {sized && <p className="meta image-sized">{sized.means}</p>}

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

        {/* Outside `.resting`, because a modal is not part of the layout it
            covers. Every list of models lives in here: the Hub cache, the
            directories this process was started from, and the Hub itself. */}
        <ImageModelPicker
          kind={kind}
          open={pickerOpen}
          current={repo.trim()}
          onClose={() => setPickerOpen(false)}
          onPick={onPickRepo}
          /* A typed directory has no row to click, so its button both
             chooses and commits. It still routes through the same two
             steps every other load here takes: the pick fills the trigger
             and the name box, then the load starts — which is what leaves
             `tried` set, so the "ask again with confirm" retry underneath
             has something to retry. */
          onLoadPath={(chosen) => {
            onPickRepo(chosen);
            void onLoad(chosen);
          }}
          busy={busy !== ""}
        />

        {err && <div className="hint err">{err}</div>}
        {/* Plain, above the error slot, because it is the outcome of a
            deliberate act and reads as a report rather than a fault. */}
        {stopped && <div className="hint">{stopped}</div>}
        {err && refused && (
          <div className="row">
            <button
              className="ghost sm"
              onClick={() => void onLoad(tried, true)}
              disabled={busy !== "" || tried.trim() === ""}
            >
              ask again for {tried} with confirm
            </button>
            <span className="meta">
              Some of those refusals can be overridden — holding a pipeline
              beside a resident text model, mainly. One that cannot answers
              again with the same sentence.
            </span>
          </div>
        )}
        {/* Needs no model: it reads an adapter file. So it belongs in the
            RESTING state too, which is where somebody comparing two LoRAs
            before committing to either actually is. Diffusion side only —
            a LoRA targets a denoiser or a text encoder, and the vision
            section has neither. */}
        {kind !== "vision" && <AdapterPanel />}
      </div>
    );
  }

  // ──────────────────────────────────────────────────────────────── loaded

  // The family's own words, when the local list carried them for this repo.
  // Not derived here: `ImageStatus` sends the identifier and `ImageLocalModel`
  // sends the prose, so this reads the prose across rather than inventing a
  // mapping that would be a second place for family names to live.
  const label = local?.models.find((m) => m.path === status.repo)?.label ?? "";
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

  // The two ends of the map, and then the largest MAGNITUDE either way —
  // which is what a diverging scale has to be normalised by. Normalising the
  // halves separately would make the deepest negative exactly as saturated as
  // the strongest positive, so a map with one faint region arguing against the
  // class would paint it as that region's equal.
  const attrFlat = attr ? attr.map.flat() : [];
  const attrHi = attrFlat.reduce((m, v) => (v > m ? v : m), attrFlat[0] ?? 0);
  const attrLo = attrFlat.reduce((m, v) => (v < m ? v : m), attrFlat[0] ?? 0);
  const attrMag = Math.max(Math.abs(attrHi), Math.abs(attrLo));
  // Two different unrankable maps, and they are NOT the same finding.
  //
  // A spread of exactly 0 is the model saying it did not use the image at
  // all: identical logits under every occlusion, which is why `strongest`
  // comes back null rather than as the first window of a tie. A spread that
  // is merely tiny is a map made of ROUNDING — real differences, all of them
  // inside the precision the scores are reported at, so ranking its windows
  // is ranking the last digit. Collapsing the two would report a model that
  // ignored the picture as one whose map was just a bit noisy.
  //
  // Flatness is read off `strongest`, NEVER off `spread === 0`. `spread`
  // arrives rounded to the six decimals the scores are reported at, so a map
  // that really did move — by 4e-7, say — comes over the wire as a spread of
  // 0.0 with a strongest window intact. Testing the number would call that
  // model "did not use the image", which is the confident version of the
  // wrong answer. The server sets `strongest` to null on the UNROUNDED
  // spread, so it is the only field that can tell the two apart.
  const attrIsFlat = attr !== null && attr.strongest === null;
  const attrIsNoise = attr !== null && !attrIsFlat && attr.spread <= 1e-6;
  // Either way no peak is drawn and no cell is shaded: a diverging scale
  // normalised to a span that small paints rounding as structure.
  const attrUnrankable = attrIsFlat || attrIsNoise;
  // A run whose batch was silently reduced is a different run time from the
  // one that was asked for, and somebody is going to time it.
  const batchCut = attr !== null && attr.batch_used < attr.batch_requested;
  // Which edge, if either, had its last window pulled back to cover the strip
  // a plain tiling would have left out.
  const attrClamped = !attr
    ? ""
    : attr.grid.edge_row_clamped && attr.grid.edge_col_clamped
      ? "row and the last column"
      : attr.grid.edge_row_clamped
        ? "row"
        : attr.grid.edge_col_clamped
          ? "column"
          : "";
  // Every window by its cell, so a cell can say where in the picture it was
  // and what the probability did as well as the logit.
  const attrWindows = new Map<string, ImageAttributionWindow>();
  attr?.windows.forEach((w) => attrWindows.set(`${w.row}:${w.col}`, w));

  return (
    <div ref={scanRef} className="panel image">
      <div className="sect">
        <span className={`dot ${kind === "vision" ? "d-vision" : "d-image"}`} />
        <h2 className={kind === "vision" ? "h-vision" : "h-image"}>{kind === "vision" ? "VISION MODEL — WHAT IT SAW, AND WHAT IT SAID" : "IMAGE MODEL — WHICH WORDS THE PICTURE LOOKED AT"}</h2>
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
          {/* THE SAME FORMATTER THE SERVER USES. This read "0.69 GB
              resident" while the sentence two lines down said "693 MB of
              weights" — one quantity, two units, on one screen. `gb` is
              LoadBar's, and it is the same GB/MB/bytes rule as the server's
              `fmt.bytes_si`. */}
          {status.bytes_resident > 0
            ? `${gb(status.bytes_resident)} resident`
            : "resident weights could not be sized"}
        </span>
        {/* Cross-attention is a DENOISER's relationship to a prompt. A
            classifier has no prompt, so this pill and the sentence below it
            are absent in the vision section rather than reporting an unknown
            width for a component that is not there. */}
        {kind !== "vision" && (
          <span className="pill">
            {dim === null
              ? "cross-attention width unknown"
              : dim === 0
                ? "unconditional — no cross-attention"
                : `cross-attention ${dim} wide`}
          </span>
        )}
        <span className="spacer" />
        <button className="ghost sm" onClick={() => void onUnload()} disabled={busy !== ""}>
          {busy === "unload" ? "unloading…" : "unload"}
        </button>
      </div>

      <p className="meta">{status.means}</p>
      {kind !== "vision" && <p className="meta">{crossAttentionNote(dim)}</p>}

      {/* TWO DIFFERENT NOTHINGS, and they used to share one sentence.
          "The server could not name this architecture" was printed for
          `facebook/DiT-XL-2-256`, which the server names exactly — a
          class-conditioned DiT — and simply cannot measure anything on. A
          reader told the tool did not recognise their model goes looking for
          a newer build; a reader told WHICH measurements exist and why this
          checkpoint cannot take them knows to try a different one. */}
      {status.capabilities.length === 0 && withheld.length === 0 && (
        <div className="hint">
          This is an architecture the server could not name, so it offers no
          measurements at all rather than every measurement. Nothing below is
          shown because nothing below could be honest about this checkpoint.
        </div>
      )}

      {withheld.length > 0 && (
        <div className="hint">
          <b>
            {status.capabilities.length === 0
              ? "None of this section's measurements can be taken on this checkpoint."
              : "Some of this section's measurements are not available here."}
          </b>{" "}
          The architecture is recognised — it is {status.family.replace("_", " ")}
          {status.conditioning === "class" && status.n_classes
            ? `, class-conditioned on ${status.n_classes.toLocaleString()} labels`
            : ""}
          . What is missing is a way to reach the numbers, and that is a fact
          about this pipeline rather than about the tool:
          <ul className="withheld">
            {withheld.map(([name, why]) => (
              <li key={name}>
                <span className="mid">{name.replace(/_/g, " ")}</span> — {why}
              </li>
            ))}
          </ul>
          {status.conditioning === "class" && (
            <>
              A class-conditioned model is steered by a number from its own
              label list, not by words, so the word-to-pixel questions this
              section asks do not apply to it. A text-to-image checkpoint —
              anything with a text encoder — answers all of them.
            </>
          )}
        </div>
      )}

      {/* A loaded model with capabilities, none of which this panel's controls
          are for. Saying so is the difference between a panel that decided not
          to draw and a panel that looks broken: a half-empty card with no
          sentence in it is the second one. */}
      {status.capabilities.length > 0 &&
        !canCapture &&
        !canKnock &&
        !canAttribute &&
        !canTrace &&
        !canPredict && (
        <div className="hint">
          What this checkpoint offers is{" "}
          <b>{status.capabilities.join(", ")}</b>, and none of the map, the
          knockout or the occlusion sweep below is among them — so those
          controls are absent rather than present and unable to answer.{" "}
          {dim === 0
            ? "Nothing here attends to a prompt, so there is nothing for a word-to-pixel map to be about."
            : "This panel reads words against pixels; measurements over image patches belong to a different one."}
        </div>
      )}

      {/* Offers occlusion and nothing else this panel draws — the ordinary
          case for a ViT, and worth a sentence rather than a silently
          half-empty card above a working control. */}
      {canAttribute && !canCapture && !canKnock && (
        <div className="hint">
          What this checkpoint offers is{" "}
          <b>{status.capabilities.join(", ")}</b>. There is no prompt for it to
          attend to, so the word-to-pixel map and the knockout are absent — but
          it has a class to lose, which is what the occlusion sweep below
          measures.
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

      {/* ─── when it committed, and what it looked like getting there ───
          Gated on `latent_trace` like every other control here. The cost of
          this measurement was already being shown above; until now there was
          nothing to spend it on, so the panel priced a run it could not
          perform. */}
      {canTrace && (
        <div className="isect">
          <h3 className="mid isect-head">
            steps — where the picture stopped changing
          </h3>
          <ImageSteps steps={steps} />
        </div>
      )}

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
                {/* Keyed on the row's ORDER as well as its step index. A
                    scheduler that reports the same index twice is a real
                    thing, and React would silently drop the second row for a
                    duplicate key — a step measured and not shown. */}
                {run.steps.map((s, ri) => (
                  <tr key={`${ri}:${s.step}`} style={{ "--i": ri } as CSSProperties}>
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

      {/* ─── the interventional half ────────────────────────────────────
          Gated on the CAPABILITY alone, not on a map having been drawn first.
          A knockout needs no capture — it removes a word and regenerates — so
          requiring one would be a dependency this panel invented rather than
          one the measurement has. */}
      {canKnock && (
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
            the tokenizer's tokens the map is drawn in — one word can be
            several tokens — and the run removes <b>every</b> word in turn
            rather than only the ones picked, so your picks are marked in the
            result rather than narrowing the work.
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

      {/* Same control in the loaded state, because "what does this LoRA
          change" is asked just as often with a pipeline already resident —
          and with one resident the reader can also fill in each module's size
          RELATIVE to the weight it modifies, which is the number people
          actually want. */}
      {kind !== "vision" && <AdapterPanel />}

      {/* ─── ONE INSTRUMENT OVER ONE PICTURE ─────────────────────────────
          Pick a picture, then ask things of it: what it says, where it
          looked, and what actually supported the answer. Each ask carries
          its own answer directly beneath it.

          They used to be three unrelated blocks with the subject buried in
          the third, which is how "What does it say?" came to render
          disabled beside the words "pick a picture first" while the picker
          sat in a different section entirely. The subject is ONE thing, so
          it is stated once at the top; the asks are several, so they are
          sections that look alike.

          Which of them appear is unchanged and still comes from the loaded
          checkpoint's own capability list, read from the server. Nothing
          here infers what a model can do from its name. */}
      {(canPredict || canAttribute) && (
        <div className="vis">
          <div className="vis-subject">
            <h3 className="mid isect-head">the picture</h3>
            <input
              ref={pickRef}
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => {
                onPick(e.target.files?.[0]);
                e.target.value = ""; // so re-picking the same file fires again
              }}
            />
            <div className="row image-pick">
              <button
                className="ghost sm"
                onClick={() => pickRef.current?.click()}
                disabled={busy !== ""}
              >
                {picture ? "pick another picture" : "pick a picture"}
              </button>
              {picture ? (
                <span className="meta">
                  <b>{pictureName}</b>
                  {pictureDims
                    ? ` — ${pictureDims.w}x${pictureDims.h} as it sits on your disk`
                    : " — its dimensions did not decode in the browser"}
                </span>
              ) : (
                <span className="meta">
                  A file from this machine, read here and sent in the body. There
                  is no path box on purpose: a path in a request names a file on
                  the server's disk rather than on yours, and a browser cannot
                  produce one for a file you picked in any case.
                </span>
              )}
            </div>
            {picture && (
              <img className="image-shot" src={picture} alt="the picture being measured" />
            )}
          </div>

          <div className="vis-asks">
            {/* Asks one and two live in `ImageCV`: what it says, and where
                it looked. Each is its own section in there, with its own
                answer under it. */}
            {canPredict && <ImageCV picture={picture} canReadout={canReadout} />}

            {/* Ask three. The interventional one, and the only kind of
                saliency this project was ever going to draw: a gradient
                or an attention weight says a region CORRELATES with the
                answer, this covers the region up, runs the model again
                and measures what the class actually lost — the same
                argument `patch.py` makes on the text side.

                Gated on `attribution` alone. A diffusion pipeline has no
                class logit to move and never carries that capability, so
                this ask is absent there rather than present and unable to
                answer. */}
            {canAttribute && (
              <section className="vis-ask image-attr">
                <h4 className="vis-ask-h">what supported the answer?</h4>
                <p className="meta vis-ask-1l">
                  Cover one window of the picture with a flat fill, run the model
                  again, record what the class logit did — then do it for every
                  window. Evidence measured by removing it, rather than inferred
                  from a gradient.
                </p>

                <div className="row image-controls">
                  <label className="meta" htmlFor="image-patch">
                    patch
                  </label>
                  <input
                    id="image-patch"
                    type="number"
                    min={1}
                    max={512}
                    value={aPatch}
                    onChange={(e) => {
                      setAttr(null);
                      setAPatch(Math.max(1, Math.min(512, Number(e.target.value) || 1)));
                    }}
                  />
                  <label className="meta" htmlFor="image-stride">
                    stride
                  </label>
                  <input
                    id="image-stride"
                    type="number"
                    min={1}
                    max={512}
                    value={aStride}
                    onChange={(e) => {
                      setAttr(null);
                      setAStride(Math.max(1, Math.min(512, Number(e.target.value) || 1)));
                    }}
                  />
                  <label className="meta" htmlFor="image-fill">
                    fill
                  </label>
                  <select
                    id="image-fill"
                    className="combo"
                    value={aFill}
                    onChange={(e) => {
                      setAttr(null);
                      setAFill(e.target.value);
                    }}
                  >
                    {FILLS.map((f) => (
                      <option key={f} value={f}>
                        {f}
                      </option>
                    ))}
                  </select>
                  <label className="meta" htmlFor="image-batch">
                    batch
                  </label>
                  <input
                    id="image-batch"
                    type="number"
                    min={1}
                    max={64}
                    value={aBatch}
                    onChange={(e) => {
                      setAttr(null);
                      setABatch(Math.max(1, Math.min(64, Number(e.target.value) || 1)));
                    }}
                  />
                  <label className="meta" htmlFor="image-target">
                    class
                  </label>
                  <input
                    id="image-target"
                    className="image-attr-class"
                    type="number"
                    min={0}
                    placeholder="the model's own"
                    value={aTarget}
                    onChange={(e) => {
                      setAttr(null);
                      setATarget(e.target.value);
                    }}
                  />
                </div>

                {/* The preflight, BEFORE any run, and now down to the lines a
                    reader acts on. This route needs no model, so the number
                    arrives as soon as a geometry is known, and it is the number
                    that decides whether to run at all. The paragraphs that used
                    to sit here — what a fill IS, and what the price is a price
                    OF — are under the disclosure below the button. Still
                    reachable; no longer three paragraphs standing between the
                    controls and the button that spends them. */}
                <div className="image-cost">
                  {attrCost ? (
                    <>
                      <p className="meta">
                        <b>this sweep</b> · {attrCost.means}
                      </p>
                      <p className="meta">
                        Priced at {pricedH}x{pricedW} —{" "}
                        {shotSize > 0
                          ? "the input size this checkpoint states, not the dimensions of the file you picked."
                          : "the dimensions of the file you picked, because this checkpoint states no input size of its own."}{" "}
                        The note under the button says what that changes.
                      </p>
                      {/* `null` is UNKNOWN and is never rendered as 0. A wait
                          invented from a typed constant would be this tool
                          making a number up. */}
                      {attrCost.seconds === null && (
                        <p className="meta">
                          No per-pass time has been measured on this machine, so
                          there is no forecast — <b>that is "nobody measured",
                          not "instant"</b> — and an invented one would be a
                          number this tool made up.
                        </p>
                      )}
                    </>
                  ) : attrCostErr ? null : pricedH < 1 ? (
                    <p className="meta">
                      This checkpoint states no input size, so nothing here knows a
                      geometry to price yet. Pick a picture and the sweep is costed
                      against its dimensions before you can run it.
                    </p>
                  ) : (
                    <p className="meta">pricing the sweep…</p>
                  )}
                  {attrCostErr && <p className="meta">{attrCostErr}</p>}
                </div>

                {/* Past the ceiling: the button is not offered, and the passes and
                    the ceiling are, because those two are what a reader needs to
                    pick a stride. `estimate` prices a run it would refuse on
                    purpose — being told only "no" leaves you guessing at the
                    parameter that caused it. */}
                <div className="row">
                  {attrCost && !attrCost.within_ceiling ? (
                    <span className="meta">
                      <b>No run is offered at this setting.</b> It is{" "}
                      <b>{attrCost.passes}</b> forward passes against a ceiling of{" "}
                      <b>{attrCost.ceiling}</b>, and the sweep refuses it — a job
                      rather than a click, and finding that out by waiting is the
                      failure the ceiling exists to prevent. The preflight above
                      still priced it, and names a stride that fits.
                    </span>
                  ) : (
                    <>
                      <button
                        className="cta"
                        onClick={() => void onAttribute()}
                        disabled={busy !== "" || picture === "" || !attrCost}
                      >
                        {busy === "attribute"
                          ? "Covering it up, a window at a time…"
                          : "Cover it up and re-run"}
                      </button>
                      {picture === "" ? (
                        <span className="meta">
                          Pick a picture first. This measures what a model looked at
                          in ONE image, so there is no default worth substituting.
                        </span>
                      ) : !attrCost ? (
                        <span className="meta">
                          The preflight has no number for this setting, so there is
                          nothing to run it against — its sentence above says why.
                        </span>
                      ) : null}
                    </>
                  )}
                </div>

                {/* Everything in here is load-bearing and none of it is
                    deleted: a fill is a BASELINE rather than a deletion, and the
                    price is a price of the tensor the model sees rather than of
                    the file on your disk. Both are the kind of thing a reader
                    needs once and then needs to be able to find again — which is
                    what a disclosure is for, and what four paragraphs between the
                    controls and the button is not. */}
                <details className="vis-more">
                  <summary>how the cover-up works, and what the price is a price of</summary>
                  <p className="meta">
                    <b>There is no neutral fill.</b> Nothing here can delete a
                    region, only replace it with something else, so a flat square
                    is a specific baseline and a different one gives a different
                    map — which is why all four of {FILLS.join(", ")} are offered
                    rather than one. A stride below the patch overlaps the windows
                    for a smoother map at a quadratic price. Leave <b>class</b>
                    empty to attribute whatever the model itself predicted; naming
                    one audits a label you supplied instead, which is a different
                    question with the same picture.
                  </p>
                  {pricedH > 0 && (
                    <p className="meta">
                      The sweep is priced at {pricedH}x{pricedW},{" "}
                      {shotSize > 0
                        ? "the input size this checkpoint states — which is what its own processor resizes your picture to before a single window is placed, so the file's own dimensions never reach the sweep."
                        : "the dimensions of the file you picked, because this checkpoint states no input size of its own. Its processor may resize to something else entirely, and the run would then be over a geometry this preflight does not know — a weaker number than the one above it looks like."}
                    </p>
                  )}
                </details>

                {/* ── the map ─────────────────────────────────────────────────
                    Drawn over the picture in a DIVERGING scale with a neutral
                    midpoint at zero, because the scores are signed and the sign is
                    the finding. */}
                {attr && (
                  <>
                    <div className="image-attr-panes">
                      <div
                        className="image-attr-shot"
                        style={{ aspectRatio: `${attr.grid.width} / ${attr.grid.height}` }}
                      >
                        <img src={picture} alt={`the picture this sweep covered up: ${pictureName}`} />
                        {/* Colour only, and hidden from the accessibility tree: the
                            table below is the same map with the numbers in it, and
                            196 unlabelled cells announced twice is worse than once. */}
                        <div
                          className="image-attr-over"
                          aria-hidden="true"
                          style={{
                            gridTemplateColumns: `repeat(${attr.grid.map_cols}, 1fr)`,
                            gridTemplateRows: `repeat(${attr.grid.map_rows}, 1fr)`,
                          }}
                        >
                          {attr.map.map((row, r) =>
                            row.map((v, c) => (
                              <span
                                key={`${r}:${c}`}
                                className={
                                  [
                                    "image-attr-cell",
                                    !attrUnrankable &&
                                    attr.strongest &&
                                    attr.strongest.row === r &&
                                    attr.strongest.col === c
                                      ? "peak"
                                      : "",
                                    !attrUnrankable &&
                                    attr.most_negative &&
                                    attr.most_negative.row === r &&
                                    attr.most_negative.col === c
                                      ? "against"
                                      : "",
                                  ]
                                    .filter(Boolean)
                                    .join(" ") || undefined
                                }
                                style={{
                                  background: attrUnrankable ? "transparent" : attrShade(v, attrMag),
                                }}
                              />
                            )),
                          )}
                        </div>
                      </div>

                      {/* The scale, spelled out. A reader who takes the two hues for
                          "strong" and "weak" has read the map backwards in half of
                          it, so both ends are named in words as well as numbers. */}
                      <div className="image-attr-key">
                        <span className="meta">
                          covering it <b>RAISED</b> the class — that region was
                          arguing against the answer
                        </span>
                        <div className="image-attr-ramp" aria-hidden="true" />
                        <div className="image-attr-ends">
                          <span className="mid">{drop(attrLo)}</span>
                          <span className="mid">0</span>
                          <span className="mid">{drop(attrHi)}</span>
                        </div>
                        <span className="meta">
                          covering it <b>COST</b> the class its evidence
                        </span>
                        <span className="meta">
                          Zero sits at the middle of the scale and paints as nothing,
                          so a window that changed the answer in neither direction
                          reads as neither.
                        </span>
                      </div>
                    </div>

                    <p className="meta image-read">
                      The overlay is over the tensor the model saw — this
                      checkpoint's own processor produced it from your file, doing
                      the resize the model was trained with. If that processor crops
                      as well as resizing, your picture underneath is stretched into
                      the same frame and the alignment is off by exactly that crop.
                      {attr.grid.overlap > 0 &&
                        ` Neighbouring windows share ${attr.grid.overlap} pixels at this stride, so the cells above are drawn as a plain tiling of the frame while the windows they stand for overlapped.`}
                      {attrClamped !== "" &&
                        ` The last ${attrClamped} had to be pulled back to the edge so that no strip of the image sat under no window at all, which makes that final overlap larger than the stride.`}
                    </p>

                    {/* The same map with the numbers in it. The colour is derived
                        here; the number is the measurement. */}
                    <div className="image-grid-wrap">
                      <table
                        className="image-grid image-attr-grid"
                        aria-label="signed logit movement per occluded window, by map row and column"
                      >
                        <thead>
                          <tr>
                            <th />
                            {attr.map[0].map((_, c) => (
                              <th key={c} className="mid">
                                c{c}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="stagger">
                          {attr.map.map((row, r) => (
                            <tr key={r} style={{ "--i": r } as CSSProperties}>
                              <th className="mid">r{r}</th>
                              {row.map((v, c) => {
                                const w = attrWindows.get(`${r}:${c}`);
                                return (
                                  <td
                                    key={c}
                                    className="image-cell"
                                    style={{
                                      background: attrUnrankable
                                        ? "transparent"
                                        : attrShade(v, attrMag),
                                    }}
                                    title={
                                      `row ${r}, column ${c}` +
                                      (w ? ` — ${box(w)}` : "") +
                                      ` — ${drop(v)} logits` +
                                      (w && w.prob_drop !== null
                                        ? `, ${drop(w.prob_drop)} in softmax probability`
                                        : ", no probability: this head has a single output")
                                    }
                                  >
                                    {drop(v)}
                                  </td>
                                );
                              })}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    <div className="row image-attr-facts">
                      <span className="pill">{attr.model_name}</span>
                      <span className="pill on">
                        {attr.target_label || `class ${attr.target}`}
                        {attr.target_chosen_by_model
                          ? " · the model's own top prediction"
                          : " · the class you named"}
                      </span>
                      <span className="pill">
                        {attr.fill} fill
                        {attr.fill_value.length > 0
                          ? ` at ${attr.fill_value.map((v) => measured(v, 3)).join(", ")}`
                          : ""}
                      </span>
                      <span className="pill">
                        {attr.seconds}s measured, batches of {attr.batch_used}
                      </span>
                    </div>

                    {/* ── what the map is allowed to claim ──────────────────
                        Three verdicts, and the first two are not the same one. A
                        spread of exactly zero is the model saying it did not use
                        the picture; a spread inside the reported precision is a map
                        of real differences that are all smaller than the last digit
                        they are printed to. Neither may name a peak, and the reason
                        they may not is different in each. */}
                    {attrIsFlat ? (
                      <div className="hint">
                        <b>There is no strongest window: this map is exactly flat.</b>{" "}
                        The model returned the same logit under every one of these
                        occlusions, which is it telling you it did not use the image
                        — nothing moved in either direction, so nothing argued for
                        the class and nothing argued against it. Naming a peak here
                        would be reading rank order out of a tie, and the cells above
                        are unshaded rather than uniformly pale.
                      </div>
                    ) : attrIsNoise ? (
                      <div className="hint">
                        <b>This map is made of rounding.</b> Its entire span is{" "}
                        {span(attr.spread)} logits, at or below the precision these
                        scores are reported at — so ranking its windows is ranking
                        the last digit, and no peak is named or drawn. The cells above
                        are left unshaded for the same reason: a diverging scale
                        normalised to a span that small paints noise as structure. The
                        paragraph below is the server's own account of it.
                      </div>
                    ) : (
                      <>
                        {attr.strongest !== null && (
                          <p className="meta">
                            <b>Strongest</b> — row {attr.strongest.row}, column{" "}
                            {attr.strongest.col} ({box(attr.strongest)}): covering it
                            moved the logit by <b>{drop(attr.strongest.logit_drop)}</b>
                            {attr.strongest.prob_drop !== null
                              ? `, and the softmax probability by ${drop(attr.strongest.prob_drop)}`
                              : ""}
                            . That is a peak <i>relative to the other windows of this
                            picture</i>, and the whole map spans{" "}
                            <b>{span(attr.spread)}</b> logits — the span is the scale
                            the peak means anything on.
                          </p>
                        )}

                        {attr.most_negative ? (
                          <p className="meta">
                            <b>Arguing against the class</b> — row{" "}
                            {attr.most_negative.row}, column {attr.most_negative.col}{" "}
                            ({box(attr.most_negative)}) at{" "}
                            <b>{drop(attr.most_negative.logit_drop)}</b>: covering it
                            RAISED the logit, so that region was evidence against the
                            answer rather than for it. An absolute value would have
                            printed it as the same size of evidence <i>for</i>.
                          </p>
                        ) : (
                          <p className="meta">
                            <b>Nothing argued against the class.</b> No window raised
                            the logit when it was covered — which is a different
                            finding from a window that moved it by 0.0, and the reason
                            there is a sentence here rather than an empty slot where a
                            most-negative window would go.
                          </p>
                        )}
                      </>
                    )}

                    {/* Rule of the module, not of this panel: a silent cap is a
                        defect, so both numbers are shown whenever they differ. */}
                    {batchCut && (
                      <div className="hint">
                        <b>The batch was reduced.</b> {attr.batch_requested} occluded
                        copies per call were asked for and {attr.batch_used} were
                        used. Both are shown because the smaller one is more forward
                        calls and a different wall-clock time — the {attr.seconds}s
                        above is the reduced run's, not the run you asked for.
                      </div>
                    )}

                    {attr.value_range_inferred && (
                      <div className="hint">
                        <b>The input value range was inferred from this one image's
                        extremes</b>, not read from the checkpoint's processor — it
                        published too little to compute one. That changes what the
                        fill actually was: the range used was{" "}
                        {measured(attr.value_range[0], 4)} to{" "}
                        {measured(attr.value_range[1], 4)} as guessed from this picture,
                        and one picture's extremes are a lower bound on the model's
                        input range rather than the range. A photograph that never
                        reaches the bottom of it puts "{attr.fill}" somewhere that is
                        not the midpoint the word implies.
                      </div>
                    )}

                    {attr.class_names_dropped && (
                      <div className="hint">
                        <b>The class names did not match this head's {attr.classes}{" "}
                        outputs</b>, so they were dropped entirely rather than applied
                        by position. The target is numbered above rather than named,
                        which is the right way round: a picture captioned with the
                        wrong class name is worse than one captioned with a number.
                      </div>
                    )}

                    {/* The server's own paragraph, verbatim — including the four
                        sentences this panel is not allowed to soften: that a fill is
                        a baseline and not a removal, that occlusion is out of
                        distribution, that softmax confidence is not the probability
                        of being right, and that one image is a sample rather than a
                        property of the model. */}
                    <p className="meta image-means">{attr.means}</p>
                  </>
                )}
              </section>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
