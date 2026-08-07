import { useRef, useState } from "react";
import { loadModel, ModelStatus, streamGenerate } from "./api";
import AttentionPanel from "./AttentionPanel";
import FeaturesPanel from "./FeaturesPanel";
import ModelPicker from "./ModelPicker";

interface Props {
  model: ModelStatus | null;
  onModelChange: () => Promise<void>;
}

const CURATED = ["Qwen/Qwen2.5-0.5B-Instruct", "gpt2"];

export default function Playground({ model, onModelChange }: Props) {
  const [source, setSource] = useState<"hf" | "ollama">("hf");
  const [pick, setPick] = useState(CURATED[0]);
  const [pickerOpen, setPickerOpen] = useState(false);
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

  const introspectable = model?.device !== "ollama";

  return (
    <>
      <div className="row">
        <button className="model-btn glass" onClick={() => setPickerOpen(true)} disabled={busy !== ""}>
          <span className="model-btn-label">model</span>
          <span className="model-btn-id">{pick}</span>
          <span className="model-btn-caret">⌄</span>
        </button>
        {source === "ollama" && <span className="chip">via Ollama</span>}
        <button className="ghost" onClick={() => void ensureLoaded()} disabled={busy !== "" || !!isLoadedPick}>
          {isLoadedPick ? "Loaded ✓" : busy === "loading" ? "Loading…" : "Load"}
        </button>
      </div>

      <ModelPicker
        open={pickerOpen}
        current={pick}
        onClose={() => setPickerOpen(false)}
        onPick={(id, src) => {
          setPick(id);
          setSource(src);
          setPickerOpen(false);
        }}
      />

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
