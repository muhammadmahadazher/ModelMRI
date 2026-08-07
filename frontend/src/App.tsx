import { useCallback, useEffect, useState } from "react";
import { getSession, ModelStatus } from "./api";
import AsciiField from "./AsciiField";
import Playground from "./Playground";

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
    ? `${model.hf_id} · ${model.device} · ${model.dtype}`
    : "no model loaded";

  return (
    <main>
      <div className="hero">
        <AsciiField />
        <div className="in">
          <h1 className="wordmark">
            Model<span className="pop">MRI</span>
          </h1>
          <div className="specrow">
            <span className="bar" />
            <span>see inside the model</span>
            <span>attention / features / steering</span>
            <span>local-first</span>
            <span className="spacer" />
            <span className={`pill ${model?.loaded ? "on" : ""}`}>{pill}</span>
          </div>
        </div>
      </div>
      <Playground model={model} onModelChange={refresh} />
      <footer>
        <span>MRI-0.2</span>
        <span>MIT ©2026</span>
        <span className="spacer" />
        <a href="https://github.com/muhammadmahadazher/ModelMRI">
          github.com/muhammadmahadazher/ModelMRI
        </a>
      </footer>
    </main>
  );
}
