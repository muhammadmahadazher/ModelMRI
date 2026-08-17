import { CSSProperties, useEffect, useState } from "react";
import RestingSketch from "./RestingSketch";
import { useScanOnData } from "./useScanOnData";
import {
  captureImageAttention,
  errorText,
  getImage,
  getImageLocal,
  getImageTasks,
  imageAttentionCost,
  imageKnockout,
  imageSize,
  imageStepsCost,
  ImageAttentionCost,
  ImageAttentionRun,
  ImageKnockout,
  ImageLocal,
  ImageSearch,
  ImageSize,
  ImageStatus,
  ImageTasks,
  ImageTraceCost,
  loadImage,
  searchImageModels,
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

/** A size, or the admission that nobody here knows one.
 *
 *  **`null` is UNKNOWN and must never come out as "0.0 GB".** The Hub
 *  publishes no per-dtype parameter counts for most GGUF and pickle repos,
 *  and `image_catalog` deliberately passes that through as `null` rather than
 *  as a number — so the one thing this function may not do is turn the
 *  absence of a measurement into the smallest possible one. A row reading
 *  "0.0 GB" invites exactly the click a size column exists to prevent, and it
 *  invites it hardest on the repos whose real weight is largest.
 *
 *  0 is folded in with `null` for the same reason: it can only arrive from a
 *  server that has not been taught the rule, and rendering it as a size would
 *  be this panel making the claim on its behalf.
 */
function sizeText(bytes: number | null): string {
  if (bytes === null || bytes <= 0) return "size unknown";
  if (bytes >= 1e12) return `${(bytes / 1e12).toFixed(2)} TB`;
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(bytes >= 1e10 ? 0 : 1)} GB`;
  return `${Math.round(bytes / 1e6)} MB`;
}

/** Download counts, shortened. Six significant digits of popularity is noise
 *  in a column whose job is to say "lots of people use this one". */
function downloads(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}k`;
  return String(n);
}

export default function ImagePanel() {
  const [status, setStatus] = useState<ImageStatus | null>(null);
  // What is on this disk, sized. `null` is "the scan has not answered yet",
  // which is a different thing from an empty list — the second is a real
  // finding and gets the server's own sentence under it.
  const [local, setLocal] = useState<ImageLocal | null>(null);
  // Which half of the browser is open. Two tabs rather than one long list
  // because they answer different questions: what can I open right now, and
  // what could I go and get.
  const [tab, setTab] = useState<"here" | "find">("here");
  // The Hub half, all of it fetched only once that tab is opened. A task list
  // and a search are two network calls to pay for a tab most sessions never
  // visit — and the resting panel's whole claim is that it costs nothing.
  const [tasks, setTasks] = useState<ImageTasks | null>(null);
  const [task, setTask] = useState("");
  const [q, setQ] = useState("");
  const [found, setFound] = useState<ImageSearch | null>(null);
  const [searching, setSearching] = useState(false);
  // The find tab's own error line. Separate from `err` because the two tabs
  // fail for unrelated reasons — an unreachable Hub is not a refused load —
  // and one shared string leaves the message from the tab you just left
  // sitting under the tab you just opened.
  const [findErr, setFindErr] = useState("");
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
  const [busy, setBusy] = useState("");
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

  // The task list, the moment the find tab is opened and never before.
  useEffect(() => {
    if (loaded || tab !== "find" || tasks !== null) return;
    let live = true;
    void getImageTasks()
      .then((t) => {
        if (!live) return;
        setTasks(t);
        // The server names its own default rather than this picking one. Every
        // tag at once is not a valid Hub filter — the API ANDs repeated
        // `filter` values — so somebody has to choose, and it is not the
        // panel's choice to make silently.
        setTask((cur) => cur || t.default);
      })
      .catch((e) => live && setFindErr(errorText(e)));
    return () => {
      live = false;
    };
  }, [loaded, tab, tasks]);

  // The search itself, debounced, re-run when the task or the query changes.
  // Gated on a task being known: an empty one would be the server's default
  // under a dropdown showing something else.
  useEffect(() => {
    if (loaded || tab !== "find" || task === "") return;
    let live = true;
    setSearching(true);
    setFindErr("");
    const timer = window.setTimeout(() => {
      void searchImageModels(q, task)
        .then((s) => {
          if (!live) return;
          setFound(s);
          setSearching(false);
        })
        .catch((e) => {
          if (!live) return;
          // Verbatim, both of them. 422 names the tasks this can open and 503
          // says the models already downloaded still load — a rewrite here
          // would lose whichever half the reader needed.
          setFindErr(errorText(e));
          // `null` is the honest terminal state. Leaving the previous results
          // up would attribute them to the search that just failed.
          setFound(null);
          setSearching(false);
        });
    }, 280);
    return () => {
      live = false;
      window.clearTimeout(timer);
    };
  }, [loaded, tab, task, q]);

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
    setTried(which);
    try {
      const s = await loadImage(which, confirm);
      setStatus(s);
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

  /** Switching tabs clears the messages the other tab left behind.
   *
   *  One error line per tab is not enough on its own: `err` belongs to a
   *  refused LOAD, which either tab can cause, so it survives — but a size
   *  lookup for a name you have moved on from does not, and neither does a
   *  Hub failure under a list of what is on your own disk. That last one was
   *  the bug ModelPicker records: "searching HuggingFace is a live call" sat
   *  under the Ollama tab as if it were Ollama's explanation. */
  function openTab(next: "here" | "find") {
    setFindErr("");
    setSized(null);
    setTab(next);
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

          {/* ─── the browser: what is here, and what could be ───────────
              Two tabs rather than one list, because they answer different
              questions and fail for unrelated reasons. "On this machine"
              reads files and cannot fail for want of a network; "Find one"
              is a live Hub call and has nothing to say about your disk. A
              single list would have to pick one of those two explanations
              for an empty result. */}
          <div className="seg image-tabs" role="tablist" aria-label="find an image model">
            <button
              role="tab"
              aria-selected={tab === "here"}
              className={tab === "here" ? "on" : ""}
              onClick={() => openTab("here")}
            >
              On this machine
            </button>
            <button
              role="tab"
              aria-selected={tab === "find"}
              className={tab === "find" ? "on" : ""}
              onClick={() => openTab("find")}
            >
              Find one
            </button>
          </div>

          {tab === "here" ? (
            <div className="image-browse">
              {local === null && (
                <span className="meta">
                  reading this machine's cache and sizing what is in it…
                </span>
              )}

              {local && local.models.length > 0 && (
                <div className="image-models">
                  {local.models.map((m) => (
                    <div className="image-model" key={m.path}>
                      <span className="mid image-model-id" title={m.path}>
                        {m.path}
                      </span>
                      {/* The family in the server's own words. The identifier
                          is kept beside it because that is what `capabilities`
                          is keyed on, and an unknown family carries its reason
                          instead of a bare row. */}
                      <span className="meta image-model-family">
                        {m.label}
                        {m.known ? ` · ${m.family}` : ""}
                      </span>
                      {/* Never "0.0 GB" for something nobody sized. */}
                      <span
                        className={`meta image-size${
                          m.size_bytes === null ? " unknown" : ""
                        }`}
                      >
                        {sizeText(m.size_bytes)}
                      </span>
                      {/* An interrupted download is the state a browse list
                          cannot show and this one must: configs arrived, the
                          weights did not. It looks exactly like a model that
                          is ready, right up until the load fails minutes
                          later, so the row says so INSTEAD of offering a
                          button that cannot work. */}
                      {!m.complete ? (
                        <span className="meta image-partial">
                          configs but no weights — an interrupted download
                          rather than a model that is ready. Fetch it again
                          before it can be opened.
                        </span>
                      ) : m.known ? (
                        <button
                          className="green"
                          onClick={() => void onLoad(m.path)}
                          disabled={busy !== ""}
                        >
                          {busy === "load" && tried === m.path ? "Loading…" : "Load"}
                        </button>
                      ) : (
                        <span className="meta">{m.reason}</span>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* Rendered as the server wrote it: how many are here, how many
                  have weights actually present, and what the whole lot
                  weighs. A count re-typed here could drift from the list
                  above it. */}
              {local && <span className="meta">{local.means}</span>}
            </div>
          ) : (
            <div className="image-browse">
              <div className="image-find">
                <label className="meta" htmlFor="image-task">
                  what it should do
                </label>
                <select
                  id="image-task"
                  className="combo"
                  value={task}
                  onChange={(e) => setTask(e.target.value)}
                  disabled={!tasks || tasks.tasks.length === 0}
                >
                  {tasks?.tasks.map((t) => (
                    <option key={t.task} value={t.task}>
                      {t.label}
                    </option>
                  ))}
                </select>
                {/* Both controls die together, because a search is only ever
                    run against a task. Where no task list could be fetched —
                    the static demo, which has no Hub to ask — a live-looking
                    box that silently does nothing is worse than a dead one:
                    the sentence under it is the answer, and a reader who is
                    still typing has not read it. */}
                <input
                  id="image-q"
                  className="combo grow"
                  placeholder="search the Hub — a name, an author, a word"
                  aria-label="search the Hub for an image model"
                  value={q}
                  onChange={(e) => setQ(e.target.value)}
                  disabled={!tasks || tasks.tasks.length === 0}
                  spellCheck={false}
                />
              </div>

              {/* What the chosen task IS, in the catalogue's own words, and
                  what it is CONSISTENT with. Never what the model is: a tag
                  covers a UNet and a DiT, which keep their cross-attention in
                  different places, so the architecture stays an open question
                  until the checkpoint's own config is read at load. */}
              {tasks?.tasks
                .filter((t) => t.task === task)
                .map((t) => (
                  <div key={t.task} className="image-task-note">
                    <p className="meta">{t.means}</p>
                    <p className="meta">
                      Checkpoints listed under this task are usually{" "}
                      <b>{t.families.join(" or ")}</b> — but a task says what a
                      model does, not what it is built from, and which of those
                      any one of them turns out to be is settled by reading its
                      own config when it loads. Nothing on this row claims to
                      know yet.
                    </p>
                  </div>
                ))}

              {searching && <span className="meta">asking the Hub…</span>}

              {!searching && found && found.models.length > 0 && (
                <div className="image-models">
                  {found.models.map((m) => (
                    <div className="image-model" key={m.id}>
                      <span className="mid image-model-id" title={m.id}>
                        {m.id}
                      </span>
                      <span className="meta image-model-family">
                        {m.task_label}
                        {m.downloads > 0 ? ` · ${downloads(m.downloads)} downloads` : ""}
                        {/* Answered by looking at this disk, not guessed from
                            the listing — so it is worth saying. */}
                        {m.cached ? " · already on this machine" : ""}
                      </span>
                      <span
                        className={`meta image-size${
                          m.size_bytes === null ? " unknown" : ""
                        }`}
                      >
                        {sizeText(m.size_bytes)}
                      </span>
                      {/* A gated repo will not hand over its weights until its
                          licence is accepted and a token is on this machine.
                          Saying so is the difference between a row that
                          explains itself and a Load button that fails on the
                          first byte.

                          `&& !m.cached` is load-bearing, and this machine has
                          the case that proves it: `facebook/sam3` comes back
                          from the Hub gated AND already downloaded. Gating is
                          about the TRANSFER, and for a repo whose weights are
                          already sitting in the cache there is no transfer
                          left to authorise — so sending that reader off to
                          accept a licence would be withholding a button that
                          works, over a credential they do not need. */}
                      {m.gated && !m.cached ? (
                        <span className="meta image-gated">
                          needs credentials — accept its licence on the Hub and
                          sign in, then search again
                        </span>
                      ) : (
                        <button
                          className="green"
                          onClick={() => void onLoad(m.id)}
                          disabled={busy !== ""}
                        >
                          {busy === "load" && tried === m.id
                            ? "Loading…"
                            : m.cached
                              ? "Load"
                              : "Get it"}
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* How many came back, how many are already here, and how many
                  publish no size at all — the server's sentence, because that
                  last count is the one a reader most needs and the one this
                  panel would be guessing at. */}
              {!searching && found && <span className="meta">{found.means}</span>}

              {tasks && <span className="meta">{tasks.means}</span>}

              {found && found.models.some((m) => m.gated && !m.cached) && (
                <span className="meta">
                  A gated model still opens once you have accepted its licence
                  and signed in — the model picker's HuggingFace tab is where
                  the token goes — or you can name it in the box below and let
                  the load answer for itself.
                </span>
              )}

              {/* Verbatim: 422 names every task this can open, 503 says the
                  models already downloaded still load. Both are the server's
                  own sentences and neither survives a rewrite. */}
              {findErr && <div className="hint err">{findErr}</div>}
            </div>
          )}

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
            <button
              className="green"
              onClick={() => void onLoad(repo)}
              disabled={busy !== "" || repo.trim() === ""}
            >
              {busy === "load" && tried === repo ? "Loading pipeline…" : "Load it"}
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

        {err && <div className="hint err">{err}</div>}
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

      {/* A loaded model with capabilities, none of which this panel's controls
          are for. Saying so is the difference between a panel that decided not
          to draw and a panel that looks broken: a half-empty card with no
          sentence in it is the second one. */}
      {status.capabilities.length > 0 && !canCapture && !canKnock && (
        <div className="hint">
          What this checkpoint offers is{" "}
          <b>{status.capabilities.join(", ")}</b>, and neither the map nor the
          knockout below is among them — so both controls are absent rather
          than present and unable to answer.{" "}
          {dim === 0
            ? "Nothing here attends to a prompt, so there is nothing for a word-to-pixel map to be about."
            : "This panel reads words against pixels; measurements over image patches belong to a different one."}
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
    </div>
  );
}
