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
        <h1>
          <span>Model</span>MRI
        </h1>
        <span className={`pill ${model?.loaded ? "on" : ""}`}>{pill}</span>
      </header>
      <Playground model={model} onModelChange={refresh} />
      <footer>
        v0.2 — attention + SAE features + steering. Agent traces land next.{" "}
        <a href="https://github.com/muhammadmahadazher/ModelMRI">
          github.com/muhammadmahadazher/ModelMRI
        </a>
      </footer>
    </main>
  );
}
