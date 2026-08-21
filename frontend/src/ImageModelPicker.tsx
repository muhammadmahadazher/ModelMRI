import { KeyboardEvent, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  errorText,
  getImageDiscovered,
  getImageLocal,
  getImageTasks,
  ImageDiscovered,
  ImageDiscoveredModel,
  ImageLocal,
  ImageLocalModel,
  ImageSearch,
  ImageTasks,
  searchImageModels,
} from "./api";
import { bytesSI } from "./measured";
import { ModelSkeleton } from "./ModelPicker";

/** Choosing an image model, as a sheet rather than as a wall.
 *
 *  The panel used to render every cached checkpoint as a flat list with a Load
 *  button on each row — a list that is either empty, on the machines that have
 *  downloaded nothing, or long enough to push the rest of the resting panel
 *  off the screen. This is the same control `ModelPicker` is for text models:
 *  a compact trigger that says what you are on, and one sheet behind it
 *  holding every way of finding something else.
 *
 *  ## Three sources, and they are three because they fail for three reasons
 *
 *  **On this machine** reads the Hub cache. **In this folder** walks the
 *  directory ModelMRI was started from. **Find one** asks the Hub. A single
 *  merged list would have to pick ONE explanation for an empty result, and the
 *  three are not interchangeable: a disk read cannot fail for want of a
 *  network, a Hub search has nothing to say about your working directory, and
 *  an empty cache says nothing at all about the checkpoint somebody cloned
 *  into their project folder an hour ago.
 *
 *  ## Nothing in here is typed into the source
 *
 *  Every row, count, size and sentence comes off a response. The only thing
 *  this file decides is which of the server's own sentences to show.
 */

/** A size, or the admission that nobody here knows one.
 *
 *  **`null` is UNKNOWN and must never come out as "0.0 GB".** The Hub
 *  publishes no per-dtype parameter counts for most GGUF and pickle repos, and
 *  `image_catalog` deliberately passes that through as `null` rather than as a
 *  number — so the one thing this function may not do is turn the absence of a
 *  measurement into the smallest possible one. A row reading "0.0 GB" invites
 *  exactly the click a size column exists to prevent, and it invites it
 *  hardest on the repos whose real weight is largest.
 *
 *  0 is folded in with `null` for the same reason: it can only arrive from a
 *  server that has not been taught the rule, and rendering it as a size would
 *  be this picker making the claim on its behalf.
 */
export function sizeText(bytes: number | null): string {
  if (bytes === null || bytes <= 0) return "size unknown";
  if (bytes >= 1e12) return `${(bytes / 1e12).toFixed(2)} TB`;
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(bytes >= 1e10 ? 0 : 1)} GB`;
  // Through the shared rule below a gigabyte. The last arm was an unbounded
  // `Math.round(bytes / 1e6)} MB`, so a real 400 kB checkpoint rendered as
  // "0 MB" on a row whose Load button was enabled — a size of nothing beside
  // an offer to load it. The TB and GB arms above keep their own precision
  // deliberately: this column is scanned vertically and "1.2 TB" against
  // "4.0 GB" is the comparison it exists to make.
  return bytesSI(bytes);
}

/** Download counts, shortened. Six significant digits of popularity is noise
 *  in a column whose job is to say "lots of people use this one". */
function downloads(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}k`;
  return String(n);
}

/** Why a row on disk cannot be picked, in the words the panel already used.
 *
 *  **`complete` is THREE states and `!complete` is the wrong test for all of
 *  them.** `true` — the weights are here. `false` — they are not: configs
 *  arrived and the weights did not, which is an interrupted download, and it
 *  looks exactly like a model that is ready right up until the load fails
 *  minutes later. `null` — the entry could not be sized AT ALL, so neither
 *  claim is available; reporting that as `false` would send somebody off to
 *  re-download a model they may already have.
 *
 *  An empty string means the row is pickable.
 */
function blockedBecause(m: ImageLocalModel | ImageDiscoveredModel): string {
  // A pipeline can hold weights and still not be buildable: `complete` asks
  // whether ANY weights are on the disk, and a checkpoint whose unet is
  // missing while its VAE is present answers yes. MEASURED on the cached
  // `segmind/tiny-sd` and `stabilityai/sd-turbo`, both of which listed as
  // ready and failed at the click with a diffusers error naming a filename
  // the reader never chose. The server already worked out which component is
  // short; this says so instead.
  //
  // `!fit.loadable` is the whole test. Gating on `missing.length` missed the
  // other way a checkpoint is unopenable — no single variant covering every
  // component, which leaves `missing` empty and puts the explanation in
  // `reason`. Such a row came back pickable AND badge-less, so it looked
  // like an ordinary healthy model right up until the click.
  const fit = "fit" in m ? m.fit : null;
  if (fit && !fit.loadable) {
    const why = fit.missing.join("; ") || fit.reason;
    return why ? `${why} Fetch it again before it can be opened.` : "";
  }
  if (m.complete === false) {
    return (
      "configs but no weights — an interrupted download rather than a model " +
      "that is ready. Fetch it again before it can be opened."
    );
  }
  if (m.complete === null) {
    return (
      "this entry could not be sized — the files are there but unreadable, " +
      "so whether the weights arrived is unknown. Loading it may still work: " +
      "name it in the box under the picker."
    );
  }
  if (!m.known) return m.reason;
  return "";
}

/** Will this one run on the card in this machine, as a word and a number.
 *
 *  The size column beside it is what the checkpoint weighs ON DISK, and a
 *  reader comparing that against the GPU they remember buying gets the wrong
 *  answer twice over: an F32 checkpoint loaded bf16 takes half its file size,
 *  and a pipeline shipping two copies of one component takes less than its
 *  folder. So this is a SECOND number rather than a recolouring of the first —
 *  what it costs once loaded, against what is free right now.
 *
 *  `unknown` is rendered, not hidden. A card that does not report its free
 *  memory is a fact about the machine, and a badge that quietly disappears
 *  reads as "fine".
 */
function FitBadge({ m }: { m: ImageLocalModel | ImageDiscoveredModel }) {
  const fit = "fit" in m ? m.fit : null;
  if (!fit) return null;
  // A row that cannot be loaded already carries `blockedBecause`; a fit badge
  // beside it would be answering a question the row has stopped asking.
  if (!fit.loadable) return null;
  const cost =
    fit.card_bytes === null ? "size unknown" : `${sizeText(fit.card_bytes)} on card`;
  return (
    <span className={`meta image-fit fit-${fit.verdict}`} title={fit.means}>
      <span className="fit-dot" aria-hidden="true" />
      {fit.verdict === "unknown" ? "fit unknown" : cost}
    </span>
  );
}

/** Arrow keys move within the list, the way a listbox is supposed to.
 *
 *  Tab is already spoken for — it cycles the sheet, and the trap below keeps
 *  it inside — so without this the only way to reach the fortieth row is forty
 *  Tabs through every row above it. Home and End are here for the same reason:
 *  the list is the tallest thing in the sheet and it scrolls.
 */
function moveWithin(e: KeyboardEvent<HTMLDivElement>) {
  if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(e.key)) return;
  const items = Array.from(
    e.currentTarget.querySelectorAll<HTMLElement>('[role="option"]:not([disabled])'),
  );
  if (items.length === 0) return;
  e.preventDefault();
  const at = items.indexOf(document.activeElement as HTMLElement);
  const to =
    e.key === "Home"
      ? 0
      : e.key === "End"
        ? items.length - 1
        : at < 0
          ? 0
          : e.key === "ArrowDown"
            ? Math.min(items.length - 1, at + 1)
            : Math.max(0, at - 1);
  items[to]?.focus();
}

/** Where a checkpoint can come from, and they are four because they fail
 *  for four unrelated reasons — see the note at the top of this file.
 *
 *  `path` is the one that has no list behind it: a directory that is
 *  neither in the Hub cache nor under a root this process walks cannot be
 *  enumerated, only named. It is the only source whose control is a box.
 */
type Source = "cache" | "folder" | "hub" | "path";

interface Props {
  /** The section asking. Its families are the ones listed; the rest are
   *  counted, not hidden. */
  kind?: "diffusion" | "vision";
  open: boolean;
  onClose: () => void;
  /** A pick fills the trigger. It does NOT load: opening a diffusion pipeline
   *  is minutes and gigabytes, and a sheet that closes under you is the wrong
   *  place to start one from — the same grammar `ModelPicker` uses, where the
   *  sheet chooses and the Load button beside the trigger commits. */
  onPick: (repo: string) => void;
  /** Whatever the trigger is showing — the loaded pipeline, or the last pick.
   *  Marks its row, so the open sheet answers "what am I on?" as well. */
  current: string;
  /** Pick AND commit, for the one source that has no row to click.
   *
   *  Every other source lists things, so a pick can fill the trigger and
   *  leave the Load button beside it to commit — the grammar the rest of
   *  this sheet uses on purpose. A typed path has no list: the box and the
   *  button next to it ARE the whole control, which is the shape
   *  `CustomPanel` already uses for the same job on the text side.
   *
   *  Optional, and it falls back to `onPick`. A sheet that silently did
   *  nothing when a parent forgot to wire this would be worse than one
   *  that fills the box and hands the commit back to the panel.
   */
  onLoadPath?: (path: string) => void;
  /** Whether the panel behind is already loading something. The Load
   *  button in here spends the same single load slot the panel's does. */
  busy?: boolean;
}

/** One row on disk: cached, or found in a folder.
 *
 *  Both shapes carry the same fields, and the only difference is what `path`
 *  means — a repo id from the cache, a directory from the walk — which is said
 *  on the row's own title rather than left for the reader to infer.
 */
function DiskRow({
  m,
  i,
  where,
  current,
  onPick,
}: {
  m: ImageLocalModel | ImageDiscoveredModel;
  i: number;
  where: string;
  current: string;
  onPick: (repo: string) => void;
}) {
  const blocked = blockedBecause(m);
  const identity = (
    <>
      <span className="mid">{m.path}</span>
      {/* The family in the SERVER's words. The identifier rides beside it
          because that is what `capabilities` is keyed on, and a family this
          could not name carries its reason instead of a bare row. */}
      <span className="meta">
        {m.label}
        {m.known ? ` · ${m.family}` : ""}
      </span>
      <span className="spacer" />
      <FitBadge m={m} />
      {/* Never "0.0 GB" for something nobody sized. */}
      <span className={`meta image-size${m.size_bytes === null ? " unknown" : ""}`}>
        {sizeText(m.size_bytes)}
      </span>
    </>
  );

  // A control that can only fail teaches a reader that the tool is broken, so
  // the sentence takes the pickable row's place rather than sitting beside it.
  if (blocked) {
    return (
      <div
        className="model-row static locked said"
        style={{ ["--i" as string]: i }}
        title={`${where} ${m.path}`}
      >
        {identity}
        <span className="model-said">{blocked}</span>
      </div>
    );
  }
  return (
    <button
      className={`model-row ${m.path === current ? "sel" : ""}`}
      style={{ ["--i" as string]: i }}
      role="option"
      aria-selected={m.path === current}
      onClick={() => onPick(m.path)}
      title={`${where} ${m.path}`}
    >
      {identity}
    </button>
  );
}

/** One Hub result. Says what it DOES, never what it is — see the task note. */
function HubRow({
  m,
  i,
  current,
  onPick,
}: {
  m: ImageSearch["models"][number];
  i: number;
  current: string;
  onPick: (repo: string) => void;
}) {
  // `&& !m.cached` is load-bearing, and this machine has the case that proves
  // it: a repo can come back from the Hub gated AND already downloaded. Gating
  // is about the TRANSFER, and for weights already sitting in the cache there
  // is no transfer left to authorise — so sending that reader off to accept a
  // licence would be withholding a row that works, over a credential they do
  // not need.
  const locked = m.gated && !m.cached;
  const identity = (
    <>
      <span className="mid">{m.id}</span>
      <span className="meta">
        {m.task_label}
        {m.downloads > 0 ? ` · ${downloads(m.downloads)} downloads` : ""}
        {/* THREE STATES, THREE RENDERINGS. This printed the first and
            collapsed the other two into silence:

              true   the weights are here, nothing to download
              false  we looked, and they are not
              null   the local cache could not be walked at all

            A `null` shown as blank tells the reader they do not have a model
            when nobody managed to check — and the row beside it quotes the
            full download. `partial` is the third real state the server
            separates on purpose: a cache entry holding configs and no weights
            looks present to a directory listing and has its ENTIRE transfer
            still ahead of it. */}
        {m.cached === true ? " · already on this machine" : ""}
        {m.cached === null ? " · whether it is here could not be checked" : ""}
        {m.partial ? " · an interrupted download is here, not the weights" : ""}
      </span>
      <span className="spacer" />
      <span className={`meta image-size${m.size_bytes === null ? " unknown" : ""}`}>
        {sizeText(m.size_bytes)}
      </span>
    </>
  );
  if (locked) {
    return (
      <div className="model-row static locked said" style={{ ["--i" as string]: i }}>
        {identity}
        <span className="model-said">
          needs credentials — accept its licence on the Hub and sign in, then
          search again
        </span>
      </div>
    );
  }
  return (
    <button
      className={`model-row ${m.id === current ? "sel" : ""}`}
      style={{ ["--i" as string]: i }}
      role="option"
      aria-selected={m.id === current}
      onClick={() => onPick(m.id)}
      title={m.id}
    >
      {identity}
    </button>
  );
}

/** Which section is asking, and therefore what it can actually measure.
 *
 *  `imaging.py` is the source of these groupings — the same split the panel
 *  uses to decide which controls to draw — so a family added there shows up
 *  in the right picker without anybody editing this list twice.
 */
const FAMILIES: Record<string, readonly string[]> = {
  diffusion: ["unet_diffusion", "dit_diffusion"],
  vision: ["vit", "clip", "detection", "segmentation", "vlm"],
};

export default function ImageModelPicker({
  open,
  onClose,
  onPick,
  onLoadPath,
  busy = false,
  current,
  kind = "diffusion",
}: Props) {
  /** The rows this section can measure, and how many it is not showing.
   *
   *  Counted rather than dropped: a list that silently shrinks reads as "you
   *  only have three image models", when the truth is the other seven are in
   *  the sibling section. */
  const split = <T extends { family: string }>(rows: T[] | undefined) => {
    const all = rows ?? [];
    const ours = all.filter((r) => mine(r.family));
    return { ours, hidden: all.length - ours.length };
  };

  /** Does this row belong to the section that opened the picker?
   *
   *  An UNKNOWN family is kept rather than hidden. `imaging.detect` reports
   *  unknown-with-a-reason for a checkpoint it cannot place, and hiding those
   *  would mean the one model a reader cannot identify is also the one the
   *  picker refuses to show them. */
  const mine = (family: string) =>
    !family || family === "unknown" || (FAMILIES[kind] ?? []).includes(family);
  const [tab, setTab] = useState<Source>("cache");
  // `null` is "not answered yet", which is a different thing from an empty
  // list. The second is a real finding and gets the server's own sentence
  // under it; the first gets a skeleton.
  const [local, setLocal] = useState<ImageLocal | null>(null);
  const [localErr, setLocalErr] = useState("");
  const [disco, setDisco] = useState<ImageDiscovered | null>(null);
  const [discoErr, setDiscoErr] = useState("");
  // The Hub half, fetched only once that tab is opened. A task list and a
  // search are two network calls to pay for a tab most sessions never visit.
  const [tasks, setTasks] = useState<ImageTasks | null>(null);
  const [task, setTask] = useState("");
  const [q, setQ] = useState("");
  const [found, setFound] = useState<ImageSearch | null>(null);
  const [searching, setSearching] = useState(false);
  // One error line per source. They fail for unrelated reasons — an
  // unreachable Hub is not an unreadable directory — and one shared string
  // leaves the message from the tab you just left sitting under the tab you
  // just opened. `ModelPicker` records that exact bug from the hosted demo.
  const [findErr, setFindErr] = useState("");
  // The typed directory. Its own state rather than the trigger's, because
  // the two are different claims: the trigger says what is CHOSEN, and
  // this box says what somebody is in the middle of typing. Sharing one
  // string would have every keystroke rewrite the trigger behind the
  // sheet, and a half-typed path is not a choice.
  const [path, setPath] = useState("");

  const sheetRef = useRef<HTMLDivElement>(null);
  // `onClose` is an inline arrow in the parent, so its identity changes on
  // every parent render. Depending on it directly would re-run the modal
  // effect — re-capturing the opener, re-locking scroll and yanking focus.
  const closeRef = useRef(onClose);
  closeRef.current = onClose;

  // Both disk reads, the moment the sheet opens. Neither touches the network
  // and neither loads anything: one reads the Hub cache, the other walks the
  // same roots the text picker walks.
  useEffect(() => {
    if (!open) return;
    let live = true;
    setLocalErr("");
    setDiscoErr("");
    void getImageLocal()
      .then((l) => live && setLocal(l))
      // The skeleton is keyed on `local === null`, so a failure that only sets
      // an error would leave it shimmering forever. The message is the
      // terminal state.
      .catch((e) => live && setLocalErr(errorText(e)));
    void getImageDiscovered()
      .then((d) => live && setDisco(d))
      .catch((e) => live && setDiscoErr(errorText(e)));
    return () => {
      live = false;
    };
  }, [open]);

  // The task list, the moment the Hub tab is opened and never before.
  useEffect(() => {
    if (!open || tab !== "hub" || tasks !== null) return;
    let live = true;
    void getImageTasks()
      .then((t) => {
        if (!live) return;
        setTasks(t);
        // The server names its own default rather than this picking one. Every
        // tag at once is not a valid Hub filter — the API ANDs repeated
        // `filter` values — so somebody has to choose, and it is not the
        // picker's choice to make silently.
        // Deliberately NOT `cur || t.default`. Preselecting a task made
        // the first search silently narrow to one tag, which is how a
        // search for a segmenter came back empty with no explanation.
        // "" is a real choice here and it means all of them.
      })
      .catch((e) => live && setFindErr(errorText(e)));
    return () => {
      live = false;
    };
  }, [open, tab, tasks]);

  // The search itself, debounced. Gated on a task being known: an empty one
  // would be the server's default under a dropdown showing something else.
  useEffect(() => {
    // No `task === ""` guard any more: empty means every image task, the
    // way an empty box in the text picker means "show me what there is".
    // It used to short-circuit here, so the search box did nothing until
    // a dropdown had been visited first.
    if (!open || tab !== "hub") return;
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
          // Verbatim, both of them: 422 names the tasks this can open and 503
          // says the models already downloaded still load. A rewrite here
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
  }, [open, tab, task, q]);

  // Modal behaviour, per the ARIA dialog pattern: Esc closes, Tab cannot
  // escape into the page behind the scrim, the page behind cannot scroll, and
  // focus goes back where it came from on close. Without the trap, one Tab
  // past the last row lands on the panel underneath.
  useEffect(() => {
    if (!open) return;
    // Captured BEFORE focus is moved. React's autoFocus fires during commit,
    // i.e. before this effect, so an autoFocus'd control would have made this
    // read that control instead of the trigger that opened the sheet — and
    // closing would then restore focus to a node that no longer exists, which
    // means body. Hence the explicit focus() below rather than autoFocus.
    const opener = document.activeElement as HTMLElement | null;
    const focusables = () =>
      sheetRef.current?.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])',
      );
    focusables()?.[0]?.focus();
    const { overflow } = document.body.style;
    document.body.style.overflow = "hidden";

    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") {
        closeRef.current();
        return;
      }
      if (e.key !== "Tab") return;
      const list = focusables();
      if (!list?.length) return;
      const first = list[0];
      const last = list[list.length - 1];
      const here = document.activeElement;
      if (e.shiftKey && (here === first || !sheetRef.current?.contains(here))) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && here === last) {
        e.preventDefault();
        first.focus();
      }
    };

    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = overflow;
      opener?.focus?.();
    };
  }, [open]);

  // EVERY hook above this line, without exception. A hook placed below does
  // not run while the sheet is closed, so opening it renders MORE hooks than
  // the previous render did and React tears the tree down (#310) rather than
  // showing it — a picker that never appears, which reads like a dead button
  // rather than like a crash. `ModelPicker` records catching this twice.
  if (!open) return null;

  /** Switching sources clears the message the previous one left behind. */
  function openTab(next: Source) {
    setFindErr("");
    setTab(next);
  }

  /** Hand the typed directory to the panel, which starts the load.
   *
   *  Trimmed once, here, so the box, the disabled test and the request
   *  all agree about what was typed — a path with a trailing space is the
   *  same directory and must not be a different button state.
   */
  function commitPath() {
    const want = path.trim();
    if (want === "") return;
    if (onLoadPath) onLoadPath(want);
    else onPick(want);
  }

  // Portalled to <body>. MEASURED: the scrim is `position: fixed; inset: 0`,
  // and it was rendering 935x546 at (36,136) inside a 1006x626 viewport --
  // the panel's own box. `.panel` carries `transform: matrix(1,0,0,1,0,0)`
  // and `filter: blur(0px)` left over from its entrance animation, and EITHER
  // of those makes a descendant's `fixed` resolve against that ancestor
  // instead of the viewport. So the dim-and-blur only ever covered the panel
  // it was opened from, which is exactly what "only blur in a small part of
  // the background" looks like. An identity transform still creates the
  // containing block, so there is nothing to "turn off" -- the sheet has to
  // leave the panel.
  return createPortal(
    <div className="sheet-scrim" onClick={onClose}>
      <div
        id="image-model-sheet"
        className="sheet glass"
        ref={sheetRef}
        role="dialog"
        aria-modal="true"
        aria-label="Choose an image model"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="sheet-head">
          {/* Three, because they fail for three unrelated reasons — see the
              note at the top of this file. Counts come off the responses, so a
              tab can never advertise a row the list below it does not have. */}
          <div className="seg" role="tablist" aria-label="where to look">
            <button
              role="tab"
              id="img-src-cache"
              aria-selected={tab === "cache"}
              aria-controls="img-src-panel"
              className={tab === "cache" ? "on" : ""}
              onClick={() => openTab("cache")}
            >
              On this machine{local ? ` · ${split(local.models).ours.length}` : ""}
            </button>
            <button
              role="tab"
              id="img-src-folder"
              aria-selected={tab === "folder"}
              aria-controls="img-src-panel"
              className={tab === "folder" ? "on" : ""}
              onClick={() => openTab("folder")}
            >
              In this folder{disco ? ` · ${split(disco.models).ours.length}` : ""}
            </button>
            <button
              role="tab"
              id="img-src-hub"
              aria-selected={tab === "hub"}
              aria-controls="img-src-panel"
              className={tab === "hub" ? "on" : ""}
              onClick={() => openTab("hub")}
            >
              Find one
            </button>
            {/* No count on this one, and there could not be: the whole
                point of it is the directories the other two cannot
                enumerate. A tab that advertised a number here would be
                advertising a list it does not have. */}
            <button
              role="tab"
              id="img-src-path"
              aria-selected={tab === "path"}
              aria-controls="img-src-panel"
              className={tab === "path" ? "on" : ""}
              onClick={() => openTab("path")}
            >
              A path
            </button>
          </div>
          <span className="spacer" />
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>

        <div
          className="sheet-tabpanel"
          id="img-src-panel"
          role="tabpanel"
          aria-labelledby={`img-src-${tab}`}
        >
          {tab === "cache" && (
            <div
              className="model-list stagger"
              role="listbox"
              aria-label="image models in this machine's cache"
              onKeyDown={moveWithin}
            >
              {local === null && localErr === "" && (
                <ModelSkeleton label="reading this machine's cache and sizing what is in it" />
              )}
              {/* The maintainer's own case: a host with nothing installed. An
                  empty box teaches nothing, so this names the two other places
                  a model can come from. */}
              {local?.models.length === 0 && (
                <div className="meta pad">
                  Nothing is cached on this machine yet — an ordinary state
                  rather than a fault. A checkpoint sitting in a folder instead
                  of in the cache is under <b>In this folder</b>; something to
                  download is under <b>Find one</b>.
                </div>
              )}
              {split(local?.models).ours.map((m, i) => (
                <DiskRow
                  key={m.path}
                  m={m}
                  i={i}
                  where="cached as"
                  current={current}
                  onPick={onPick}
                />
              ))}
              {/* The server's own sentence: how many are here, how many hold
                  weights, and what the whole lot weighs. A count re-typed here
                  could drift from the list above it. */}
              {split(local?.models).hidden > 0 && (
                <div className="meta pad">
                  {split(local?.models).hidden} more {split(local?.models).hidden === 1 ? "is" : "are"} for the{" "}
                  {kind === "vision" ? "text-to-image" : "vision"} section. They are
                  counted rather than dropped — a list that quietly shrinks reads
                  as "this machine has fewer models than it does", and loading one
                  here would land on a panel that cannot measure it.
                </div>
              )}
              {local && <div className="meta pad">{local.means}</div>}
              {localErr && <div className="hint err">{localErr}</div>}
            </div>
          )}

          {tab === "folder" && (
            <div
              className="model-list stagger"
              role="listbox"
              aria-label="image models found in the working directory"
              onKeyDown={moveWithin}
            >
              {disco === null && discoErr === "" && (
                <ModelSkeleton label="walking the directory ModelMRI was started from" />
              )}
              {disco?.models.length === 0 && (
                <div className="meta pad">
                  No image model was found in any of the directories listed
                  below. Start ModelMRI from the folder your checkpoints live
                  in, or point <code>MODELMRI_MODELS_DIR</code> straight at it —
                  this walk is the only one that sees a checkpoint cloned
                  somewhere ordinary rather than downloaded into the Hub cache.
                </div>
              )}
              {split(disco?.models).ours.map((m, i) => (
                <DiskRow
                  key={m.path}
                  m={m}
                  i={i}
                  where="found in"
                  current={current}
                  onPick={onPick}
                />
              ))}
              {/* WHERE it looked, always — full list or empty. "Nothing found"
                  on its own tells somebody their model is missing when the
                  truth may be that the directory holding it was never walked. */}
              {disco && (
                <div className="meta pad">
                  {disco.roots.length > 0 ? (
                    <>
                      looked in:
                      {disco.roots.map((r) => (
                        <code key={r} className="looked-root">
                          {r}
                        </code>
                      ))}
                    </>
                  ) : (
                    "No directory was walked at all — the server returned no roots, so nothing on this machine has been searched."
                  )}
                </div>
              )}
              {/* A truncation nobody is told about reads as "this is all there
                  is", which is the one thing it is not. */}
              {disco?.truncated && (
                <div className="meta pad">
                  The walk stopped after {disco.scan_limit}{" "}
                  {disco.scan_limit === 1 ? "directory" : "directories"}, so
                  this is what was reached rather than everything there is.
                  Set{" "}
                  <code>MODELMRI_MODELS_DIR</code> to point straight at your
                  models folder and it will not have to guess.
                </div>
              )}
              {split(disco?.models).hidden > 0 && (
                <div className="meta pad">
                  {split(disco?.models).hidden} more {split(disco?.models).hidden === 1 ? "is" : "are"} for the{" "}
                  {kind === "vision" ? "text-to-image" : "vision"} section. They are
                  counted rather than dropped — a list that quietly shrinks reads
                  as "this machine has fewer models than it does", and loading one
                  here would land on a panel that cannot measure it.
                </div>
              )}
              {disco && <div className="meta pad">{disco.means}</div>}
              {discoErr && <div className="hint err">{discoErr}</div>}
            </div>
          )}

          {tab === "path" && (
            <div className="image-path">
              <p className="meta">
                A checkpoint neither disk tab can reach: a clone somewhere
                else on this machine, a directory exported by hand, a
                snapshot outside the Hub cache and outside the roots this
                process walks. <code>/api/image/load</code> opens a directory
                exactly the way it opens a Hub id — the family, the
                capabilities and the input size are read from the
                checkpoint's own config either way, so nothing downstream
                treats it as a lesser kind of model.
              </p>
              {/* An input and a Load button, the shape `CustomPanel` uses
                  for the same job. Every other source here lists things,
                  so a pick fills the trigger and the Load button beside it
                  commits; a typed path has nothing to list, so the box and
                  this button are the whole control. */}
              <div className="row cand-manual">
                <input
                  className="combo grow"
                  placeholder="a directory on this machine: C:\models\vit-base, /home/me/models/sd-turbo"
                  value={path}
                  onChange={(e) => setPath(e.target.value)}
                  onKeyDown={(e) =>
                    e.key === "Enter" && path.trim() !== "" && commitPath()
                  }
                  spellCheck={false}
                  aria-label="a directory on this machine to open an image model from"
                />
                <button
                  className="ghost sm"
                  onClick={commitPath}
                  disabled={busy || path.trim() === ""}
                >
                  Load
                </button>
              </div>
              <p className="meta">
                A path names a directory on the disk the SERVER is running
                on. Started from your own machine those are the same disk;
                reached from anywhere else the route refuses the path and
                says so in its own words rather than opening whatever that
                name happens to mean over there.
              </p>
              <p className="meta">
                Nothing prices this one first. <b>How big is it?</b> asks the
                Hub, and a directory has no Hub entry to ask about — the two
                disk tabs size what they find, and this box exists for the
                directories they did not reach.
              </p>
              <p className="meta">
                A name that is not a directory on that disk is tried as a Hub
                id instead, so the refusal will name the Hub rather than your
                filesystem. That is the server reporting what it actually
                attempted, which is the honest answer even though it is not
                the one a mistyped path invites.
              </p>
            </div>
          )}

          {tab === "hub" && (
            <>
              <div className="image-find">
                <label className="sr-only" htmlFor="img-pick-q">
                  search the Hub for an image model
                </label>
                <input
                  id="img-pick-q"
                  className="combo search grow"
                  placeholder="Search HuggingFace for an image model…  (empty = the most downloaded of each kind)"
                  value={q}
                  onChange={(e) => setQ(e.target.value)}
                  disabled={!tasks || tasks.tasks.length === 0}
                  spellCheck={false}
                  autoFocus
                />
                {/* A REFINEMENT, after the box, not a gate in front of it.
                    "Any kind" is the default and is a real search across every
                    task rather than a silent one. */}
                <select
                  id="img-pick-task"
                  className="combo"
                  aria-label="narrow to one kind of image model"
                  value={task}
                  onChange={(e) => setTask(e.target.value)}
                  disabled={!tasks || tasks.tasks.length === 0}
                >
                  <option value="">Any kind</option>
                  {tasks?.tasks.map((t) => (
                    <option key={t.task} value={t.task}>
                      {t.label}
                    </option>
                  ))}
                </select>
                {/* Both controls die together where no task list could be
                    fetched — a static recording has no Hub to ask, and a
                    live-looking box that silently does nothing is worse than
                    a visibly dead one. */}
              </div>

              <div
                className="model-list stagger"
                role="listbox"
                aria-label="image models on the Hub"
                onKeyDown={moveWithin}
              >
                {/* Results arrive asynchronously, so a screen reader is told
                    the count rather than left guessing whether anything
                    happened at all. */}
                <div className="sr-only" role="status" aria-live="polite">
                  {searching
                    ? "Searching the Hub"
                    : found
                      ? `${found.models.length} model${
                          found.models.length === 1 ? "" : "s"
                        } found`
                      : ""}
                </div>
                {searching && <ModelSkeleton label="asking the Hub" />}

                {/* What the chosen task IS, in the catalogue's own words, and
                    what it is CONSISTENT with. Never what a listed model is: a
                    tag covers a UNet and a DiT, which keep their
                    cross-attention in different places, so the architecture
                    stays an open question until the checkpoint's own config is
                    read at load. */}
                {!searching &&
                  tasks?.tasks
                    .filter((t) => t.task === task)
                    .map((t) => (
                      <div key={t.task} className="image-task-note">
                        <p className="meta">{t.means}</p>
                        <p className="meta">
                          Checkpoints listed under this task are usually{" "}
                          <b>{t.families.join(" or ")}</b> — but a task says
                          what a model does, not what it is built from, and
                          which of those any one of them turns out to be is
                          settled by reading its own config when it loads.
                          Nothing on these rows claims to know yet.
                        </p>
                      </div>
                    ))}

                {!searching &&
                  found?.models.map((m, i) => (
                    <HubRow key={m.id} m={m} i={i} current={current} onPick={onPick} />
                  ))}

                {!searching && found?.models.length === 0 && (
                  <div className="meta pad">
                    Nothing on the Hub matched that under this task. Whatever is
                    already on this machine is under <b>On this machine</b>, and
                    it opens with no network at all.
                  </div>
                )}

                {/* How many came back, how many are already here, and how many
                    publish no size at all — the server's sentence, because that
                    last count is the one a reader most needs and the one this
                    picker would be guessing at. */}
                {!searching && found && <div className="meta pad">{found.means}</div>}
                {tasks && <div className="meta pad">{tasks.means}</div>}
                {!searching && found?.models.some((m) => m.gated && !m.cached) && (
                  <div className="meta pad">
                    A gated model still opens once you have accepted its licence
                    and signed in — the text model picker's HuggingFace tab is
                    where the token goes.
                  </div>
                )}
                {/* Verbatim: 422 names every task this can open, 503 says the
                    models already downloaded still load. Both are the server's
                    own sentences and neither survives a rewrite. */}
                {findErr && <div className="hint err">{findErr}</div>}
              </div>
            </>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}
