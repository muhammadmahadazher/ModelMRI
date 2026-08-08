import { useEffect, useRef, useState } from "react";
import {
  getAttentionMeta,
  getLoadProgress,
  loadModel,
  LoadProgress,
  ModelStatus,
  streamGenerate,
} from "./api";
import AttentionPanel from "./AttentionPanel";
import FeaturesPanel from "./FeaturesPanel";
import ModelPicker from "./ModelPicker";

interface Props {
  model: ModelStatus | null;
  onModelChange: () => Promise<void>;
}

const CURATED = ["Qwen/Qwen2.5-0.5B-Instruct", "gpt2"];

// Sent with every generation and echoed in the readout, so what you read is
// what the run used. These were previously implicit server defaults, which
// meant the UI could not honestly name them.
const DECODE = { max_new_tokens: 256, temperature: 0.7 };

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
  const [prog, setProg] = useState<LoadProgress | null>(null);
  // The steering hook lives on the runtime, not on the panel, so any
  // generation fired while it is installed is silently steered.
  const [steering, setSteering] = useState(false);
  const pieces = useRef(0);
  const t0 = useRef(0);

  const isLoadedPick = model?.loaded && model.hf_id === pick;

  // The server keeps its model across page loads; the picker did not. You
  // came back to a tab that said one model in the badge and another in the
  // picker, and Generate quietly swapped to the second. Adopt what is
  // actually loaded, once, on first sight.
  const adopted = useRef(false);
  useEffect(() => {
    if (adopted.current || !model?.loaded || !model.hf_id) return;
    adopted.current = true;
    setPick(model.hf_id);
    setSource(model.device === "ollama" ? "ollama" : "hf");
  }, [model?.loaded, model?.hf_id, model?.device]);

  // Same bug as the picker above, one layer up: `epoch` is a client-side
  // counter, so a reload dropped it to 0 and unmounted the attention and
  // feature panels — while the server still held attention for the last
  // generation and would have served it. You generated 141 tokens, hit
  // refresh, and your analysis was simply gone with nothing saying why.
  // Ask what the server can actually answer, and mount accordingly.
  const restored = useRef(false);
  useEffect(() => {
    if (restored.current || epoch > 0) return;
    let live = true;
    void getAttentionMeta()
      .then((m) => {
        if (!live || !m.available) return;
        restored.current = true;
        setEpoch(1);
      })
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [epoch, model?.loaded]);

  // A cold load is minutes long. Poll the server so the wait is legible
  // instead of a frozen button.
  useEffect(() => {
    if (busy !== "loading") {
      setProg(null);
      return;
    }
    let live = true;
    const tick = () =>
      void getLoadProgress()
        .then((p) => live && setProg(p))
        .catch(() => {});
    tick();
    const id = window.setInterval(tick, 700);
    return () => {
      live = false;
      window.clearInterval(id);
    };
  }, [busy]);

  async function ensureLoaded(): Promise<boolean> {
    if (isLoadedPick) return true;
    setBusy("loading");
    setMeta("");
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
    if (busy || steering || !prompt.trim()) return;
    if (!(await ensureLoaded())) return;
    setBusy("generating");
    setOutput("");
    pieces.current = 0;
    t0.current = performance.now();
    const p = prompt;
    streamGenerate(
      p,
      {
      onToken: (text) => {
        pieces.current += 1;
        setOutput((o) => o + text);
      },
      onDone: () => {
        const dt = (performance.now() - t0.current) / 1000;
        // What an instrument reports: count, elapsed, rate, and the decode
        // settings that make the run reproducible. "12 pieces" was neither
        // a token count nor a rate.
        const rate = dt > 0 ? pieces.current / dt : 0;
        setMeta(
          `${pieces.current} tok · ${dt.toFixed(2)}s · ${rate.toFixed(1)} tok/s` +
            ` · ${DECODE.temperature > 0 ? `T ${DECODE.temperature}` : "greedy"}`,
        );
        setBusy("");
        setLastPrompt(p);
        setEpoch((e) => e + 1);
      },
        onError: (message) => {
          setOutput(`Error: ${message}`);
          setBusy("");
        },
      },
      DECODE,
    );
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

      {busy === "loading" && <LoadBar p={prog} id={pick} />}

      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && e.ctrlKey) void onGenerate();
        }}
        placeholder="Type a prompt… (Ctrl+Enter to generate)"
      />
      <div className="row">
        <button
          className="cta"
          onClick={() => void onGenerate()}
          disabled={busy !== "" || steering || !prompt.trim()}
          title={steering ? "steering is active — finish the A/B first" : undefined}
        >
          {steering
            ? "Steering…"
            : busy === "loading"
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
            <Generation text={output} />
            {busy === "generating" && <span className="cursor">▋</span>}
          </>
        )}
      </div>

      {epoch > 0 && introspectable && <AttentionPanel epoch={epoch} />}
      {epoch > 0 && introspectable && (
        <FeaturesPanel epoch={epoch} prompt={lastPrompt} onSteering={setSteering} />
      )}
    </>
  );
}

/** Reasoning models (Qwen3, DeepSeek-R1, and every model that copies them)
 *  wrap their scratchpad in <think>. Showing the raw tag looks like a bug;
 *  hiding it throws away the most interesting part of the generation. So it
 *  gets its own labelled block, open while it streams and collapsed once the
 *  answer arrives. */
function Generation({ text }: { text: string }) {
  const open = text.includes("<think>") && !text.includes("</think>");
  const parts: { think: boolean; text: string }[] = [];
  let rest = text;
  while (true) {
    const start = rest.indexOf("<think>");
    if (start === -1) break;
    if (start > 0) parts.push({ think: false, text: rest.slice(0, start) });
    const after = rest.slice(start + 7);
    const end = after.indexOf("</think>");
    if (end === -1) {
      parts.push({ think: true, text: after });
      rest = "";
      break;
    }
    parts.push({ think: true, text: after.slice(0, end) });
    rest = after.slice(end + 8);
  }
  if (rest) parts.push({ think: false, text: rest });
  if (!parts.some((p) => p.think)) return <>{text}</>;

  return (
    <>
      {parts.map((p, i) =>
        p.think ? (
          <details key={i} className="think" open={open}>
            <summary>reasoning · {p.text.trim().split(/\s+/).length} words</summary>
            {p.text.trim()}
          </details>
        ) : (
          <span key={i}>{p.text}</span>
        ),
      )}
    </>
  );
}

const STAGES: Record<string, string> = {
  resolving: "Resolving on the Hub",
  weights: "Fetching weights",
  device: "Moving to the accelerator",
  ready: "Ready",
  error: "Failed",
};

/** Progress for an in-flight load: named stage, bytes when we know them,
 *  and an indeterminate sweep when we don't. */
function LoadBar({ p, id }: { p: LoadProgress | null; id: string }) {
  const total = p?.bytes_total ?? 0;
  const done = p?.bytes_done ?? 0;
  const pct = total > 0 ? Math.min(100, (done / total) * 100) : null;
  return (
    <div className="loadbar glass-inset" role="status" aria-live="polite">
      <div className="loadbar-row">
        <span className="loadbar-stage">{STAGES[p?.stage ?? ""] ?? "Loading"}</span>
        <span className="mid loadbar-id">{id}</span>
        <span className="spacer" />
        <span className="meta">
          {pct !== null && `${gb(done)} / ${gb(total)} · `}
          {(p?.elapsed_s ?? 0).toFixed(0)}s
        </span>
      </div>
      <div className={`loadbar-track ${pct === null ? "indeterminate" : ""}`}>
        <div
          className="loadbar-fill"
          style={pct === null ? undefined : { width: `${pct}%` }}
        />
      </div>
      {p?.detail && <div className="meta loadbar-detail">{p.detail}</div>}
    </div>
  );
}

const gb = (n: number) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : `${Math.round(n / 1e6)} MB`;
