import { useRef, useState } from "react";
import { loadModel, ModelStatus, streamGenerate } from "./api";
import AttentionPanel from "./AttentionPanel";

interface Props {
  model: ModelStatus | null;
  onModelChange: () => Promise<void>;
}

export default function Playground({ model, onModelChange }: Props) {
  const [prompt, setPrompt] = useState(
    "Explain in two sentences why the sky is blue.",
  );
  const [output, setOutput] = useState("");
  const [busy, setBusy] = useState<"" | "loading" | "generating">("");
  const [meta, setMeta] = useState("");
  const [attnEpoch, setAttnEpoch] = useState(0);
  const t0 = useRef(0);
  const pieces = useRef(0);

  const loaded = model?.loaded ?? false;

  async function onLoad() {
    setBusy("loading");
    setMeta("loading model… first run downloads ~1 GB");
    try {
      const t = performance.now();
      await loadModel();
      setMeta(`loaded in ${((performance.now() - t) / 1000).toFixed(1)}s`);
      await onModelChange();
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
    streamGenerate(prompt, {
      onToken: (text) => {
        pieces.current += 1;
        setOutput((o) => o + text);
      },
      onDone: () => {
        const dt = (performance.now() - t0.current) / 1000;
        setMeta(`${pieces.current} pieces · ${dt.toFixed(1)}s`);
        setBusy("");
        setAttnEpoch((e) => e + 1);
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
        <button className="ghost" onClick={onLoad} disabled={loaded || busy !== ""}>
          {loaded ? "Model loaded ✓" : "Load Qwen2.5-0.5B-Instruct"}
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
        <button onClick={onGenerate} disabled={!loaded || busy !== ""}>
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

      {attnEpoch > 0 && <AttentionPanel epoch={attnEpoch} />}
    </>
  );
}
