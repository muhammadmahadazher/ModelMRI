import { useCallback, useEffect, useState } from "react";
import {
  Accelerator,
  getAccelerator,
  getSession,
  getSessionState,
  ModelStatus,
  SessionState,
} from "./api";
import AgentsPanel from "./AgentsPanel";
import AsciiField from "./AsciiField";
import CustomPanel from "./CustomPanel";
import { DEMO } from "./demo";
import Playground from "./Playground";
import SessionBar from "./SessionBar";
import StoragePanel from "./StoragePanel";
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

  const pill = session.open
    ? `replay · ${session.meta?.model ?? "shared session"}`
    : model?.loaded
      ? `${model.hf_id} · ${model.device}`
      : "no model loaded";

  return (
    <main>
      <div className="topbar">
        <span className="logomark">
          <span className="ast">✳</span> ModelMRI
        </span>
        <span className="spacer" />
        <ThemeToggle />
        {accel && (
          <span
            className={`pill accel ${accel.kind !== "cpu" ? "gpu" : ""}`}
            title={accel.reason}
          >
            <i className="accel-dot" />
            {accel.kind === "cpu"
              ? "CPU"
              : `${accel.name}${accel.vram_gb ? ` · ${accel.vram_gb} GB` : ""}`}
          </span>
        )}
        <span className={`pill ${model?.loaded ? "on" : ""}`}>{pill}</span>
      </div>

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

      <SessionBar session={session} onChange={setSession} />
      <Playground model={model} onModelChange={refresh} replay={session.open} />
      {/* The viewer has no machine behind it. Panels that can only ever say
          "install ModelMRI" are worse than absent — the one thing this page
          does, it should do without three dead ends around it. */}
      {!VIEWER && (
        <>
          <CustomPanel />
          <VLAPanel />
          <AgentsPanel />
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
