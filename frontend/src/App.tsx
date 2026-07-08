import { useCallback, useEffect, useState } from "react";
import { getSession, ModelStatus } from "./api";
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
      <header>
        <div className="mark" aria-hidden="true" />
        <div className="brand">
          <h1>
            <span className="g">Model</span>MRI
          </h1>
          <p className="tagline">see inside the model — attention · features · steering</p>
        </div>
        <div className="spacer" />
        <span className={`pill ${model?.loaded ? "on" : ""}`}>{pill}</span>
      </header>
      <Playground model={model} onModelChange={refresh} />
      <footer>
        v0.2 · local-first · MIT ·{" "}
        <a href="https://github.com/muhammadmahadazher/ModelMRI">
          github.com/muhammadmahadazher/ModelMRI
        </a>
      </footer>
    </main>
  );
}
