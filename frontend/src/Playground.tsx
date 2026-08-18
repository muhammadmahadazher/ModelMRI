import { useEffect, useRef, useState } from "react";
import {
  cancelLoad,
  errorText,
  getAttentionMeta,
  getLoadProgress,
  loadModel,
  LoadProgress,
  ModelStatus,
  streamGenerate,
} from "./api";
import AttentionPanel from "./AttentionPanel";
import TelemetryBar from "./TelemetryBar";
import FeaturesPanel from "./FeaturesPanel";
import GroundPanel from "./GroundPanel";
import ProbePanel from "./ProbePanel";
import PatchscopePanel from "./PatchscopePanel";
import PatchGraphPanel from "./PatchGraphPanel";
import PatchPanel from "./PatchPanel";
import ModelPicker from "./ModelPicker";
import { DEMO } from "./demo";
import { VIEWER } from "./viewer";

interface Props {
  model: ModelStatus | null;
  onModelChange: () => Promise<void>;
  /** A shared `.mri` is open: the attention below is a recording. */
  replay?: boolean;
  /** The recorded patching trace that `.mri` carries, if it carries one. */
  sessionPatch?: { available: boolean; clean: string; corrupt: string };
  /** The recorded patching GRAPH, which is a separate section and can be
   *  present without the grid or absent beside it. */
  sessionPatchGraph?: { available: boolean; n_nodes: number; n_edges: number };
  sessionGround?: { available: boolean; question: string };
  /**
   * A generation finished — succeeded or failed, the server records both.
   * The agents panel is a sibling of this component, so it cannot see the
   * stream end any other way.
   */
  onGenerated?: () => void;
}

const CURATED = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-1.7B"];

// Sent with every generation and echoed in the readout, so what you read is
// what the run used. These were previously implicit server defaults, which
// meant the UI could not honestly name them.
const DECODE = { max_new_tokens: 256, temperature: 0.7 };

export default function Playground({
  model,
  onModelChange,
  replay,
  sessionPatch,
  sessionPatchGraph,
  sessionGround,
  onGenerated,
}: Props) {
  const [source, setSource] = useState<"hf" | "ollama">("hf");
  const [pick, setPick] = useState(CURATED[0]);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [prompt, setPrompt] = useState(
    "The Eiffel Tower is located in the city of",
  );
  const [output, setOutput] = useState("");
  // Did the last generation fail? The output box holds an error sentence
  // then, and nothing below it describes a model's behaviour.
  const [failed, setFailed] = useState(false);
  const [busy, setBusy] = useState<"" | "loading" | "generating">("");
  const [meta, setMeta] = useState("");
  const [epoch, setEpoch] = useState(0);
  const [lastPrompt, setLastPrompt] = useState("");
  const [prog, setProg] = useState<LoadProgress | null>(null);
  // The size guard's refusal, WITH the model it was about. Holding only the
  // message meant "Download it anyway" applied the override to whatever was
  // selected at the moment of the click — so picking a different model left
  // a refusal on screen about a model no longer chosen, and then confirmed
  // past the ceiling for one the guard had never looked at.
  const [oversize, setOversize] = useState<{
    id: string;
    source: "hf" | "ollama";
    message: string;
  } | null>(null);
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

  // Opening or closing a shared session swaps what /api/attention answers
  // without any generation happening, so the one-shot restore above never
  // fires again. Re-ask on every transition: opening must mount the panel,
  // and closing must unmount it rather than leave the recording on screen
  // labelled as live.
  const wasReplay = useRef(replay);
  useEffect(() => {
    if (wasReplay.current === replay) return;
    wasReplay.current = replay;
    let live = true;
    void getAttentionMeta()
      .then((m) => {
        if (!live) return;
        restored.current = m.available;
        setEpoch(m.available ? (e) => e + 1 : 0);
      })
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [replay]);

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

  async function ensureLoaded(
    confirm = false,
    target?: { id: string; source: "hf" | "ollama" },
  ): Promise<boolean> {
    const id = target?.id ?? pick;
    const src = target?.source ?? source;
    if (!target && isLoadedPick) return true;
    setBusy("loading");
    setMeta("");
    setOversize(null);
    try {
      const t = performance.now();
      const result = await loadModel(id, src, confirm);
      // A stopped load is not a failure. Say what happened and stay put.
      if ("cancelled" in result) {
        setMeta(result.message);
        return false;
      }
      setMeta(`loaded in ${((performance.now() - t) / 1000).toFixed(1)}s`);
      await onModelChange();
      setEpoch(0);
      return true;
    } catch (err) {
      const message = errorText(err);
      // The size guard refuses with 422 and a sentence naming both numbers.
      // Offering the override here is the difference between a guard and a
      // wall — but it must be a deliberate second click, never a default.
      if (message.includes("Load it anyway")) {
        setOversize({ id, source: src, message });
        setMeta("");
      } else {
        setMeta(message);
      }
      return false;
    } finally {
      setBusy((b) => (b === "loading" ? "" : b));
    }
  }

  async function onStopLoading() {
    setMeta("stopping…");
    try {
      await cancelLoad();
    } catch {
      /* the load will report its own outcome either way */
    }
  }

  async function onGenerate() {
    if (busy || steering || !prompt.trim()) return;
    if (!(await ensureLoaded())) return;
    setBusy("generating");
    setOutput("");
    setFailed(false);
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
        // The server filed this run as a trace before it sent `done`, so the
        // agents panel has something to find the moment it is told to look.
        onGenerated?.();
        // The server drops any open session on a committed generation, so
        // the banner has to go with it — otherwise the page keeps claiming
        // you are reading a recording while showing your own output.
        if (replay) void onModelChange();
      },
        onError: (message) => {
          // Marked, not inferred from the text: the caveat under the output
          // must not explain a generation that never happened.
          setFailed(true);
          setOutput(`Error: ${message}`);
          setBusy("");
          // A failed run is recorded too — it is the one you most want a
          // record of — so the panel is told about this end as well.
          onGenerated?.();
        },
      },
      DECODE,
    );
  }

  const introspectable = model?.device !== "ollama";

  // The viewer has no model to pick, nothing to load, and nowhere to
  // generate. Showing those controls would be offering three buttons that
  // can only answer "install ModelMRI".
  if (VIEWER) {
    return epoch > 0 ? <AttentionPanel epoch={epoch} replay /> : null;
  }

  return (
    <>
      {/* This was the only region on the page with no card and no header: a
          model button, a prompt and a Generate floating on the background
          above five panels that all had both. It is the part of the tool
          somebody uses FIRST, and it looked the least finished. */}
      <div className="panel workbench">
        <div className="sect">
          <span className="dot d-run" />
          <h2 className="h-run">RUN — A PROMPT THROUGH A MODEL</h2>
          <span className="rule" />
        </div>
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

      {source === "ollama" && (
        <div className="hint" style={{ marginTop: -6 }}>
          ollama mode runs any open model as text — attention &amp; features need
          a HuggingFace model
        </div>
      )}

      {busy === "loading" && (
        <LoadBar p={prog} id={pick} onStop={() => void onStopLoading()} />
      )}

      {/* The guard refused. It names both numbers; the override is one more
          deliberate click, and it is never the default. */}
      {oversize && (
        <div className="oversize" role="alert">
          <span className="oversize-mark" aria-hidden="true">
            !
          </span>
          <div>
            <p>{oversize.message}</p>
            <div className="row">
              <button
                className="ghost sm"
                onClick={() => {
                  // Re-select as well as load. The override names its own
                  // model, so without this the picker would keep showing a
                  // different one while that model downloaded.
                  setPick(oversize.id);
                  setSource(oversize.source);
                  void ensureLoaded(true, { id: oversize.id, source: oversize.source });
                }}
              >
                Download {oversize.id} anyway
              </button>
              <button className="ghost sm" onClick={() => setOversize(null)}>
                Pick something else
              </button>
            </div>
          </div>
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

      {/* SAID BEFORE THE GENERATION, NOT UNDER IT.
          This warning existed and rendered beneath the output, which is after
          the reader has already typed a question, waited, and read something
          confidently wrong. By then they have concluded the tool is broken --
          asking a base model "whats 2+2" and getting a confident non-answer
          is not a bug report anybody writes as "I used a base model".
          A caveat that arrives after the surprise is a footnote; the same
          sentence before it is a warning. */}
      {model?.loaded && model.instruct === false && !replay && (
        <div className="base-warn" role="note">
          <b>{model.hf_id}</b> is a base model. It continues text rather than
          answering questions, so a question will usually come back as
          plausible nonsense — that is the model, not a fault. Pick an{" "}
          <b>-Instruct</b> or <b>-it</b> model to get answers, or write a
          prompt it can continue: <code>The capital of France is</code>
        </div>
      )}

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

      <div className="output">
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
        {/* Why the answer can be wrong, said before anyone has to wonder.
            Two different causes get mistaken for a broken tool:

            a base model is a text CONTINUER — it finishes your sentence, it
            was never trained to answer a question — and any temperature above
            zero SAMPLES, so the same prompt gives a different answer each
            time.

            ModelMRI shows what the model did; it does not improve it. That is
            the product. But a reader who knows neither of these concludes the
            instrument is faulty, and this panel is where they find out. */}
        {/* Only under an answer the model actually produced. It used to render
            under a FAILED generation too — "connection closed before
            completion" followed by a note explaining that the panels below
            describe it, when there are no panels and nothing was described.

            And `sampled` is false on a replay: the demo and an opened .mri
            play back one fixed recording, so claiming a different answer each
            run is exactly wrong there. That inversion shipped: on the hosted
            demo the true half (the replayed model was a base model) was
            suppressed because the payload carried no `instruct`, while the
            false half was asserted. */}
        {(() => {
          const sampled = DECODE.temperature > 0 && !DEMO && !replay;
          const base = model?.instruct === false;
          const show =
            output !== "" &&
            busy !== "generating" &&
            !failed &&
            model?.loaded &&
            (base || sampled);
          if (!show) return null;
          return (
            <p className="gen-caveat">
              {base && (
                <>
                  <b>{model.hf_id}</b> is a base model — it continues text rather
                  than answering questions, and is often factually wrong.{" "}
                </>
              )}
              {sampled && (
                <>
                  Temperature {DECODE.temperature} samples, so the same prompt
                  gives a different answer each run.{" "}
                </>
              )}
              The panels below show what it actually did, not a corrected
              version.
            </p>
          );
        })()}
      </div>
      </div>

      <ModelPicker
        open={pickerOpen}
        current={pick}
        onClose={() => setPickerOpen(false)}
        onPick={(id, src) => {
          setPick(id);
          setSource(src);
          setPickerOpen(false);
          // A refusal is about one model. Keeping it on screen after a
          // different pick is an assertion that is no longer true.
          setOversize(null);
        }}
      />

      {/* Above the panels, because it describes the run they are all reading.
          Renders nothing until something has been generated — a bar of zeros
          is a claim about a run that never happened. Not shown for a replay:
          a `.mri` carries no timings, and the numbers would belong to
          somebody else's machine. */}
      {epoch > 0 && !replay && <TelemetryBar epoch={epoch} />}
      {epoch > 0 && introspectable && (
        <AttentionPanel epoch={epoch} replay={replay} />
      )}
      {/* Features need the model's activations, which a `.mri` does not carry
          — it is an observation, not a checkpoint. Mounting the panel anyway
          would offer a control that can only ever answer "no model loaded". */}
      {/* Patching needs no generation — it runs its own two prompts — but it
          does need a live HuggingFace model to re-run, which is exactly what a
          recording is not. */}
      {/* On a live model, whenever there is something to trace. On a
          recording, only when the file actually carries a trace — a panel
          whose one button can only apologise is worse than no panel. */}
      {introspectable && !replay && <PatchPanel epoch={epoch} />}
      {/* Directly under the grid it walks backwards from, and gated the same
          way: it seeds from the grid's own flagged sites, so it needs the same
          live model the grid does.

          NOT OFFERED in the demo or viewer builds, for the reason
          `AttentionPanel` gates token attribution off: the control would be a
          button whose only outcome is a refusal, and a visitor reads that as
          "this measurement is broken" rather than "this page has no model
          behind it". `api.ts`'s `patchGraph` refuses in those builds too —
          that is the second lock, and `tests/demo_check.py` checks this one. */}
      {introspectable && !replay && !DEMO && !VIEWER && (
        <PatchGraphPanel epoch={epoch} />
      )}
      {/* Two surfaces of their own, deliberately.

          A probe fits to YOUR labelled examples rather than to the current
          generation, so it needs no prompt and appears before one — but it
          does need a residual stream and a live model, which is the same gate
          the patching grid uses.

          A patchscope reports a SENTENCE the model produced. Beside the logit
          lens it would read as a second measurement of the same thing, and it
          is a different kind of evidence entirely: the lens reads a state
          through the unembedding, this hands the state back to the model. */}
      {introspectable && !replay && <ProbePanel epoch={epoch} />}
      {introspectable && !replay && <PatchscopePanel epoch={epoch} />}
      {/* Grounding runs its own document and question, so like the two above
          it needs no generation — but it masks passages out of an attention
          mask, which needs a live HuggingFace model. */}
      {introspectable && !replay && <GroundPanel epoch={epoch} />}
      {/* On a recording, only when the file actually carries one. Grounding
          cannot be re-taken from a `.mri` — masking a passage out needs the
          model, and the document is not in the file — so a panel offering the
          form here would be a form whose only button refuses. */}
      {replay && sessionGround?.available && (
        <GroundPanel epoch={epoch} recorded={sessionGround} />
      )}
      {replay && sessionPatch?.available && (
        <PatchPanel epoch={epoch} recorded={sessionPatch} />
      )}
      {/* Same rule again. The graph cost ~1,500 forward passes to build and
          the recipient has no weights to rebuild it with, so the panel appears
          only when the file actually carries one. */}
      {replay && sessionPatchGraph?.available && (
        <PatchGraphPanel epoch={epoch} recorded={sessionPatch ?? undefined} />
      )}
      {epoch > 0 && introspectable && !replay && (
        <FeaturesPanel
          epoch={epoch}
          prompt={lastPrompt}
          /* The feature ranking is float32-only and ModelMRI picks bfloat16 for
             every NVIDIA GPU, so the panel needs the dtype to know whether to
             offer the control at all. It is already on /api/session, which this
             component is handed. */
          dtype={model?.dtype ?? null}
          onSteering={setSteering}
        />
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
function LoadBar({
  p,
  id,
  onStop,
}: {
  p: LoadProgress | null;
  id: string;
  onStop: () => void;
}) {
  const total = p?.bytes_total ?? 0;
  // Clamped, and not only in the bar. The width was already capped at 100%
  // while the text beside it was not, so a mis-count showed as a full bar
  // labelled "5.0 GB / 2.5 GB" — the number that gave the bug away.
  const done = Math.min(p?.bytes_done ?? 0, total || Infinity);
  const pct = total > 0 ? Math.min(100, (done / total) * 100) : null;
  const stopping = (p?.detail ?? "").startsWith("stopping");
  // The model the server is loading, which is not necessarily the one the
  // picker is showing: pick a second model while the first is still loading
  // and `id` is already the new one, so the running load's bytes and elapsed
  // time appeared under a model that had not started.
  const loading = p?.hf_id ?? id;
  return (
    <div className="loadbar glass-inset" role="status" aria-live="polite">
      <div className="loadbar-row">
        <span className="loadbar-stage">{STAGES[p?.stage ?? ""] ?? "Loading"}</span>
        <span className="mid loadbar-id">{loading}</span>
        <span className="spacer" />
        <span className="meta">
          {pct !== null && `${gb(done)} / ${gb(total)} · ${gb(total - done)} left · `}
          {(p?.elapsed_s ?? 0).toFixed(0)}s
          {/* Only when the server is willing to estimate. It withholds the
              number until there is enough history to divide by, because a
              countdown that opens with "4 hours" and settles at "40 seconds"
              is one the reader learns to ignore. */}
          {p?.eta_s != null && ` · ~${remaining(p.eta_s)} left`}
        </span>
        {/* The whole reason this component was revisited. A minutes-long
            download with no way out is a trap, and this one could run for
            days before failing. */}
        <button className="ghost sm stop" onClick={onStop} disabled={stopping}>
          {stopping ? "stopping…" : "Stop"}
        </button>
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

/** A duration somebody can act on. Seconds under a minute, then minutes, then
 *  hours and minutes — "312 minutes" is a number you have to do arithmetic on
 *  before it means anything. */
export function remaining(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`;
}
