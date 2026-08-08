import { useCallback, useEffect, useState } from "react";
import { Accelerator, getAccelerator, getSession, ModelStatus } from "./api";
import AgentsPanel from "./AgentsPanel";
import AsciiField from "./AsciiField";
import CustomPanel from "./CustomPanel";
import { DEMO } from "./demo";
import Playground from "./Playground";
import ThemeToggle from "./ThemeToggle";
import VLAPanel from "./VLAPanel";

export default function App() {
  const [model, setModel] = useState<ModelStatus | null>(null);
  const [accel, setAccel] = useState<Accelerator | null>(null);

  useEffect(() => {
    let live = true;
    void getAccelerator()
      .then((a) => live && setAccel(a))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, []);

  const refresh = useCallback(async () => {
    try {
      const s = await getSession();
      setModel(s.model);
    } catch {
      setModel(null);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const pill = model?.loaded
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
          Attention, concepts, and steering for any local model — plus a
          flight recorder for your agents. One pip install, everything on your
          machine.
        </p>
        {DEMO && (
          <p className="demo-banner">
            Live demo — real recorded output from a local run. Install it to
            point these instruments at your own models:{" "}
            <code>pip install modelmri</code>
          </p>
        )}
        <div className="specrow">
          <span>local-first</span>
          <span>attention</span>
          <span>sae features</span>
          <span>steering</span>
          <span>agent traces</span>
          <span>your own models</span>
          <span>mit ©2026</span>
        </div>
      </div>

      <Playground model={model} onModelChange={refresh} />
      <CustomPanel />
      <VLAPanel />
      <AgentsPanel />
      <footer>
        <span>MRI-0.3</span>
        <span>MIT ©2026</span>
        <span className="spacer" />
        <a href="https://github.com/muhammadmahadazher/ModelMRI">
          github.com/muhammadmahadazher/ModelMRI
        </a>
      </footer>
    </main>
  );
}
