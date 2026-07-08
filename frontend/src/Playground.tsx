import { useRef, useState } from "react";
import { loadModel, ModelStatus, streamGenerate } from "./api";
import AttentionPanel from "./AttentionPanel";
import FeaturesPanel from "./FeaturesPanel";

interface Props {
  model: ModelStatus | null;
  onModelChange: () => Promise<void>;
}

const MODELS = [
  { id: "Qwen/Qwen2.5-0.5B-Instruct", label: "Qwen2.5 · chat + attention" },
  { id: "gpt2", label: "GPT-2 · features + steering" },
];

export default function Playground({ model, onModelChange }: Props) {
  const [pick, setPick] = useState(MODELS[0].id);
  const [prompt, setPrompt] = useState(
    "Explain in two sentences why the sky is blue.",
  );
  const [output, setOutput] = useState("");
  const [busy, setBusy] = useState<"" | "loading" | "generating">("");
  const [meta, setMeta] = useState("");
  const [epoch, setEpoch] = useState(0);
  const [lastPrompt, setLastPrompt] = useState("");
  const t0 = useRef(0);
  const pieces = useRef(0);

  const loaded = model?.loaded ?? false;

  async function onLoad() {
    setBusy("loading");
    setMeta("loading model… first run downloads the weights");
    try {
      const t = performance.now();
      await loadModel(pick);
      setMeta(`loaded in ${((performance.now() - t) / 1000).toFixed(1)}s`);
      await onModelChange();
      setEpoch(0);
      setOutput("");
    } catch (err) {
      setMeta(`load failed: ${String(err)}`);
    } finally {
      setBusy("");
    }
  }

  function onGenerate() {
    if (!loaded || busy) return;
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

  return (
    <>
      <div className="row">
        <div className="seg" role="tablist" aria-label="model picker">
          {MODELS.map((m) => (
            <button
              key={m.id}
              className={pick === m.id ? "on" : ""}
              onClick={() => setPick(m.id)}
              disabled={busy !== ""}
            >
              {m.label}
            </button>
          ))}
        </div>
        <button className="ghost" onClick={onLoad} disabled={busy !== ""}>
          {model?.hf_id === pick ? "Loaded ✓" : busy === "loading" ? "Loading…" : "Load model"}
        </button>
      </div>

      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && e.ctrlKey) onGenerate();
        }}
        placeholder="Type a prompt… (Ctrl+Enter to generate)"
      />
      <div className="row">
        <button className="cta" onClick={onGenerate} disabled={!loaded || busy !== ""}>
          Generate
        </button>
        <span className="meta">{meta}</span>
      </div>

      <div className="panel output">
        {output === "" && busy !== "generating" ? (
          <span style={{ color: "var(--muted)" }}>
            Output appears here — streamed token by token.
          </span>
        ) : (
          <>
            {output}
            {busy === "generating" && <span className="cursor">▋</span>}
          </>
        )}
      </div>

      {epoch > 0 && <AttentionPanel epoch={epoch} />}
      {epoch > 0 && <FeaturesPanel epoch={epoch} prompt={lastPrompt} />}
    </>
  );
}
