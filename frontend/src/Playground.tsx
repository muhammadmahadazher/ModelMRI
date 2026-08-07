import { useEffect, useRef, useState } from "react";
import {
  getLocalModels,
  getOllama,
  loadModel,
  ModelStatus,
  streamGenerate,
} from "./api";
import AttentionPanel from "./AttentionPanel";
import FeaturesPanel from "./FeaturesPanel";

interface Props {
  model: ModelStatus | null;
  onModelChange: () => Promise<void>;
}

const CURATED = ["Qwen/Qwen2.5-0.5B-Instruct", "gpt2"];

export default function Playground({ model, onModelChange }: Props) {
  const [source, setSource] = useState<"hf" | "ollama">("hf");
  const [pick, setPick] = useState(CURATED[0]);
  const [localModels, setLocalModels] = useState<string[]>([]);
  const [ollama, setOllama] = useState<{ up: boolean; models: string[] }>({
    up: false,
    models: [],
  });
  const [prompt, setPrompt] = useState(
    "The Eiffel Tower is located in the city of",
  );
  const [output, setOutput] = useState("");
  const [busy, setBusy] = useState<"" | "loading" | "generating">("");
  const [meta, setMeta] = useState("");
  const [epoch, setEpoch] = useState(0);
  const [lastPrompt, setLastPrompt] = useState("");
  const pieces = useRef(0);
  const t0 = useRef(0);

  useEffect(() => {
    void getLocalModels().then((l) => setLocalModels(l.map((m) => m.id)));
    void getOllama().then(setOllama);
  }, []);

  const isLoadedPick = model?.loaded && model.hf_id === pick;

  async function ensureLoaded(): Promise<boolean> {
    if (isLoadedPick) return true;
    setBusy("loading");
    setMeta(`loading ${pick}… (first time downloads the weights)`);
    try {
      const t = performance.now();
      await loadModel(pick, source);
      setMeta(`loaded in ${((performance.now() - t) / 1000).toFixed(1)}s`);
      await onModelChange();
      setEpoch(0);
      return true;
    } catch (err) {
      setMeta(String(err instanceof Error ? err.message : err));
      return false;
    } finally {
      setBusy((b) => (b === "loading" ? "" : b));
    }
  }

  async function onGenerate() {
    if (busy || !prompt.trim()) return;
    if (!(await ensureLoaded())) return;
    setBusy("generating");
    setOutput("");
    pieces.current = 0;
    t0.current = performance.now();
    const p = prompt;
    streamGenerate(p, {
      onToken: (text) => {
        pieces.current += 1;
        setOutput((o) => o + text);
      },
      onDone: () => {
        const dt = (performance.now() - t0.current) / 1000;
        setMeta(`${pieces.current} pieces · ${dt.toFixed(1)}s`);
        setBusy("");
        setLastPrompt(p);
        setEpoch((e) => e + 1);
      },
      onError: (message) => {
        setOutput(`Error: ${message}`);
        setBusy("");
      },
    });
  }

  const hfSuggestions = [...new Set([...CURATED, ...localModels])];
  const introspectable = model?.device !== "ollama";

  return (
    <>
      <div className="row">
        <div className="seg" role="tablist" aria-label="model source">
          <button
            className={source === "hf" ? "on" : ""}
            onClick={() => {
              setSource("hf");
              setPick(CURATED[0]);
            }}
            disabled={busy !== ""}
          >
            HuggingFace
          </button>
          <button
            className={source === "ollama" ? "on" : ""}
            onClick={() => {
              setSource("ollama");
              if (ollama.models.length) setPick(ollama.models[0]);
            }}
            disabled={busy !== ""}
          >
            Ollama {ollama.up ? `· ${ollama.models.length}` : "· off"}
          </button>
        </div>

        {source === "hf" ? (
          <>
            <input
              className="combo"
              list="hf-models"
              value={pick}
              onChange={(e) => setPick(e.target.value)}
              placeholder="any HuggingFace model id…"
              disabled={busy !== ""}
            />
            <datalist id="hf-models">
              {hfSuggestions.map((id) => (
                <option key={id} value={id} />
              ))}
            </datalist>
          </>
        ) : ollama.up && ollama.models.length ? (
          <select
            className="combo"
            value={pick}
            onChange={(e) => setPick(e.target.value)}
            disabled={busy !== ""}
          >
            {ollama.models.map((m) => (
              <option key={m}>{m}</option>
            ))}
          </select>
        ) : (
          <span className="meta">
            {ollama.up
              ? "no models installed — run: ollama pull llama3.2"
              : "Ollama not detected — install from ollama.com, then reload"}
          </span>
        )}

        <button className="ghost" onClick={() => void ensureLoaded()} disabled={busy !== "" || !!isLoadedPick}>
          {isLoadedPick ? "Loaded ✓" : busy === "loading" ? "Loading…" : "Load"}
        </button>
      </div>

      {source === "ollama" && (
        <div className="hint" style={{ marginTop: -6 }}>
          ollama mode runs any open model as text — attention &amp; features need
          a HuggingFace model
        </div>
      )}

      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && e.ctrlKey) void onGenerate();
        }}
        placeholder="Type a prompt… (Ctrl+Enter to generate)"
      />
      <div className="row">
        <button className="cta" onClick={() => void onGenerate()} disabled={busy !== "" || !prompt.trim()}>
          {busy === "loading"
            ? "Loading model…"
            : busy === "generating"
              ? "Generating…"
              : "Generate"}
        </button>
        <span className="meta">{meta}</span>
      </div>

      <div className="panel output">
        {output === "" && busy !== "generating" ? (
          <span style={{ color: "var(--color-mute)" }}>
            Output appears here — Generate loads the model automatically if
            needed.
          </span>
        ) : (
          <>
            {output}
            {busy === "generating" && <span className="cursor">▋</span>}
          </>
        )}
      </div>

      {epoch > 0 && introspectable && <AttentionPanel epoch={epoch} />}
      {epoch > 0 && introspectable && (
        <FeaturesPanel epoch={epoch} prompt={lastPrompt} />
      )}
    </>
  );
}
