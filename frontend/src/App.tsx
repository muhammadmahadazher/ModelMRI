import { useCallback, useEffect, useState } from "react";
import {
  Accelerator,
  closeSession,
  errorText,
  getAccelerator,
  getSession,
  getSessionState,
  ModelStatus,
  SessionState,
  unloadModel,
} from "./api";
import AgentsPanel from "./AgentsPanel";
import CategoryBar from "./CategoryBar";
import AsciiField from "./AsciiField";
import CustomPanel from "./CustomPanel";
import { DEMO } from "./demo";
import Playground from "./Playground";
import GraphPanel from "./GraphPanel";
import ImagePanel from "./ImagePanel";
import ModelDiffPanel from "./ModelDiffPanel";
import SectionNav from "./SectionNav";
import SessionBar from "./SessionBar";
import StoragePanel from "./StoragePanel";
import PalettePicker from "./PalettePicker";
import ThemeToggle from "./ThemeToggle";
import { VIEWER } from "./viewer";
import VLAPanel from "./VLAPanel";

export default function App() {
  const [model, setModel] = useState<ModelStatus | null>(null);
  const [version, setVersion] = useState<string | null>(null);
  const [accel, setAccel] = useState<Accelerator | null>(null);
  // Sessions live on the server, so a reload must find one that is still open
  // rather than quietly showing an empty page beside a loaded recording.
  const [session, setSession] = useState<SessionState>({ open: false });
  // Top-bar actions. `busyTop` disables both while either runs, because they
  // act on the same thing and a reset landing mid-unload would report a state
  // neither of them produced.
  const [busyTop, setBusyTop] = useState<"" | "unload" | "reset">("");
  // What the unload actually gave back, kept on screen until something else
  // happens. Not a toast: the number is the point of pressing the button.
  const [freed, setFreed] = useState("");
  // Bumping this remounts the playground, which is what "reset" means — every
  // panel keys its state off its own mount, so there is no list of things to
  // clear that could fall out of date as panels are added.
  const [resetKey, setResetKey] = useState(0);
  // Every finished generation is a recorded run now, and the panel that shows
  // recorded runs is a sibling of the playground rather than a child of it.
  // A counter is the whole message — the panel refetches, it does not need to
  // be told what happened.
  const [runs, setRuns] = useState(0);

  useEffect(() => {
    let live = true;
    void getAccelerator()
      .then((a) => live && setAccel(a))
      .catch(() => undefined);
    void getSessionState()
      .then((s) => live && setSession(s))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, []);

  const refresh = useCallback(async () => {
    try {
      const s = await getSession();
      setModel(s.model);
      setVersion(s.version);
    } catch {
      setModel(null);
    }
    // The server can close a shared session on its own — loading a model or
    // committing a generation both do — so re-read it rather than trusting
    // the last value the client happened to set.
    try {
      setSession(await getSessionState());
    } catch {
      /* leave the banner as it is rather than blanking it on a blip */
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function onUnload() {
    setBusyTop("unload");
    setFreed("");
    try {
      const out = await unloadModel();
      await refresh();
      setResetKey((k) => k + 1);
      // Report what came back, not what should have. An allocator that keeps
      // its arena is a real outcome and rounding it away would be a small lie
      // in the one place someone is watching for a number.
      const mb = out.freed_bytes / 1e6;
      setFreed(
        out.freed_bytes > 0
          ? `freed ${mb >= 1000 ? (mb / 1000).toFixed(2) + " GB" : Math.round(mb) + " MB"}`
          : "nothing to free",
      );
    } catch (err) {
      setFreed(errorText(err));
    } finally {
      setBusyTop("");
    }
  }

  async function onReset() {
    setBusyTop("reset");
    setFreed("");
    try {
      // A shared recording is view state too, so reset closes it — otherwise
      // "reset" leaves the most conspicuous thing on the page untouched.
      if (session.open) {
        await closeSession().catch(() => undefined);
      }
      await refresh();
      setResetKey((k) => k + 1);
    } finally {
      setBusyTop("");
    }
  }

  const pill = session.open
    ? `replay · ${session.meta?.model ?? "shared session"}`
    : model?.loaded
      ? `${model.hf_id} · ${model.device}`
      : "no model loaded";

  // A model built from a GGUF is the QUANTISED weights, dequantised — so every
  // number in every panel below describes that, not the original of the same
  // name. The plan block in the reader says so, but it is one panel deep and
  // disappears when you navigate away, which left the caveat attached to
  // nothing while the whole page reported on a model it never qualified.
  // `status.gguf` was already being sent and simply never read.
  const q = model?.loaded ? model.gguf : null;

  return (
    <main>
      {/* Reads the page rather than being told what is on it, so it is right
          in the viewer build, the demo build and mid-run — three states with
          three different sets of panels. Placed inside <main> only because
          that is where React needs it; it positions itself against the
          viewport. */}
      <SectionNav />
      <div className="topbar">
        <span className="logomark">
          <span className="ast">✳</span> ModelMRI
        </span>
        <span className="spacer" />
        <ThemeToggle />
        {/* Beside the mode toggle, not inside it, and now two controls rather
            than one: hues, light/dark and contrast level are three orthogonal
            axes. See PalettePicker for why folding them into one list made
            "Amber" quietly mean "Amber, dark, standard" — and made high
            contrast something you could only have INSTEAD of a palette. */}
        <PalettePicker />
        {accel && (
          <span
            className={`pill accel ${accel.kind !== "cpu" && accel.kind !== "recorded" ? "gpu" : ""}`}
            title={accel.reason}
          >
            <i className="accel-dot" />
            {/* "recorded" is not a device. The hosted demo has no accelerator
                to report — it is a static recording being read in a browser —
                and the baked bundle used to answer this with the GPU of the
                laptop that produced it, so a phone was told it was running
                CUDA on an RTX 4060. A device the page cannot see is not a
                device it may name. */}
            {accel.kind === "recorded"
              ? "recorded"
              : accel.kind === "cpu"
                ? "CPU"
                : `${accel.name}${accel.vram_gb ? ` · ${accel.vram_gb} GB` : ""}`}
          </span>
        )}
        <span className={`pill ${model?.loaded ? "on" : ""}`}>{pill}</span>

        {/* Unload and Reset live up here beside the model pill because they
            are about the whole session rather than any one panel, and because
            the thing they act on — what is resident — is named two inches to
            the left. Both are hidden when there is nothing to act on rather
            than shown disabled: a control that is never usable is furniture. */}
        {!session.open && model?.loaded && (
          <button
            className="pill act"
            onClick={() => void onUnload()}
            disabled={busyTop !== ""}
            title="Drop the model and give the memory back"
          >
            {busyTop === "unload" ? "unloading…" : "unload"}
          </button>
        )}
        {/* Always present, unlike unload. Reset re-reads the server and
            remounts the panels, which is meaningful whatever is loaded — and a
            control that disappears when you are looking for it is worse than
            one that occasionally does nothing. */}
        {!DEMO && (
          <button
            className="pill act"
            onClick={() => void onReset()}
            disabled={busyTop !== ""}
            title="Clear every panel and start from nothing"
          >
            {busyTop === "reset" ? "resetting…" : "reset"}
          </button>
        )}
        {freed && <span className="pill freed">{freed}</span>}
      </div>

      {q && (
        <div className="quantised-banner">
          <span className="pill warn">quantised</span>
          <span className="meta">
            Every measurement below describes the dequantised{" "}
            <code>{q.plan.dtype}</code> weights of this GGUF —{" "}
            {(q.plan.file_bytes / 1e9).toFixed(2)} GB on disk became{" "}
            {(q.measured_resident_bytes / 1e9).toFixed(2)} GB resident — not the
            original model of the same name. Point <code>quantdiff</code> at
            both to see how far apart they are.
          </span>
        </div>
      )}

      <div className="hero">
        <AsciiField modelId={model?.hf_id ?? null} />
        <h1 className="headline">
          See inside <span className="c">the model.</span>
        </h1>
        <p className="subline">
          {VIEWER ? (
            <>
              Open a shared analysis — a <code>.mri</code> someone sent you.
              No install, no model, no GPU. The file is read in this page and
              never leaves your browser.
            </>
          ) : (
            <>
              Attention, concepts, and steering for any local model — plus a
              flight recorder for your agents. One pip install, everything on
              your machine.
            </>
          )}
        </p>
        {DEMO && (
          <p className="demo-banner">
            Live demo — real recorded output from a local run. Install it to
            point these instruments at your own models:{" "}
            <code>pip install modelmri</code>
          </p>
        )}
        <div className="specrow">
          {VIEWER ? (
            <>
              <span>no install</span>
              <span>no upload</span>
              <span>attention</span>
              <span>read-only</span>
              <span>mit ©2026</span>
            </>
          ) : (
            <>
              <span>local-first</span>
              <span>attention</span>
              <span>sae features</span>
              <span>steering</span>
              <span>agent traces</span>
              <span>your own models</span>
              <span>mit ©2026</span>
            </>
          )}
        </div>
      </div>

      {/* Under the .mri bar, above everything it filters. It reads the page
          rather than carrying a list, so it is correct by construction — see
          the file for why that matters more here than it looks. */}
      <CategoryBar />

      <SessionBar session={session} onChange={setSession} />

      {/* Renders only when the open session carries one, which is why it sits
          unconditionally here: the panel decides, and it returns null when
          there is no graph or no provenance to show it under. */}
      <GraphPanel key={`graph-${session.open}-${resetKey}`} />
      <Playground
        key={resetKey}
        model={model}
        onModelChange={refresh}
        replay={session.open}
        sessionPatch={session.patch}
        sessionPatchGraph={session.patch_graph}
        sessionGround={session.ground}
        onGenerated={() => setRuns((n) => n + 1)}
      />
      {/* The viewer has no machine behind it. Panels that can only ever say
          "install ModelMRI" are worse than absent — the one thing this page
          does, it should do without three dead ends around it. */}
      {!VIEWER && (
        <>
          {/* Loading a GGUF makes it the live model, so the header and
              every panel have to re-ask. `refresh` updates the status;
              bumping resetKey remounts the playground, which is the same
              path the adopt button takes. */}
          <CustomPanel
            onModelChange={() => {
              void refresh();
              setResetKey((k) => k + 1);
            }}
          />
          {/* Its own surface, and one that needs nothing loaded: it runs
              its own two models and its own prompts, so it sits beside the
              custom-model panel rather than inside the playground. */}
          <ModelDiffPanel epoch={resetKey} />
          {/* Text → Image. Its own handle, its own lifecycle, and nothing
              above it loaded: a diffusion pipeline is several models and the
              server refuses to hold one beside a resident text model without
              being asked twice, so this panel is inert until you name a
              checkpoint. Sited beside the robot panel because both are a
              second modality rather than another view of the language model.
              Inside `!VIEWER` for the same reason VLAPanel is — a shared
              `.mri` has no machine to run a pipeline on. */}
          <ImagePanel />
          <VLAPanel />
          {/* Adopting a step makes the server's current generation that
              step's. Remounting the playground is what gets the panels to
              re-ask what the server can answer — the same path a page reload
              already takes, without the reload.

              `runs` is the other direction: a generation in the playground
              files a step, and the panel has no way to see the stream end
              from over here. */}
          <AgentsPanel
            runs={runs}
            onAdopted={() => setResetKey((k) => k + 1)}
          />
        </>
      )}
      <footer>
        {/* Read from /api/session. It was the literal "MRI-0.3" and had been
            wrong since 0.4.0 — a version string nobody remembers to bump is a
            version string that lies. */}
        <span>{version ? `MRI-${version}` : "MRI"}</span>
        <span>MIT ©2026</span>
        <StoragePanel />
        <span className="spacer" />
        {/* The panels can only get better if the gap between what you wanted
            and what you saw reaches me, and nobody files an issue for "I
            expected this to show something else". A named, low-ceremony
            destination asks for exactly that. */}
        <a
          href="https://github.com/muhammadmahadazher/ModelMRI/discussions/new?category=ideas"
          target="_blank"
          rel="noopener noreferrer"
        >
          what would you want this to show?
        </a>
        <a href="https://github.com/muhammadmahadazher/ModelMRI">
          github.com/muhammadmahadazher/ModelMRI
        </a>
      </footer>
    </main>
  );
}
