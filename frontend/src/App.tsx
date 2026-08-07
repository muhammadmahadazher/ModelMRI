import { useCallback, useEffect, useState } from "react";
import { getSession, ModelStatus } from "./api";
import AgentsPanel from "./AgentsPanel";
import AsciiField from "./AsciiField";
import Playground from "./Playground";
import VLAPanel from "./VLAPanel";

export default function App() {
  const [model, setModel] = useState<ModelStatus | null>(null);

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
        <span className={`pill ${model?.loaded ? "on" : ""}`}>{pill}</span>
      </div>

      <div className="hero">
        <AsciiField />
        <h1 className="headline">
          See inside <span className="c">the model.</span>
        </h1>
        <p className="subline">
          Attention, concepts, and steering for any local model — plus a
          flight recorder for your agents. One pip install, everything on your
          machine.
        </p>
        <div className="specrow">
          <span>local-first</span>
          <span>attention</span>
          <span>sae features</span>
          <span>steering</span>
          <span>agent traces</span>
          <span>mit ©2026</span>
        </div>
      </div>

      <Playground model={model} onModelChange={refresh} />
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
