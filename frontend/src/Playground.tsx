// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

import { useEffect, useRef, useState } from "react";
import LoadBar from "./LoadBar";
import { invalidateSession } from "./RunsOn";
import DevicePicker from "./DevicePicker";
import {
  cancelLoad,
  errorText,
  getAttentionMeta,
  getDiscovered,
  getLoadProgress,
  getSessionState,
  loadModel,
  LoadProgress,
  ModelStatus,
  SessionState,
  streamGenerate,
} from "./api";
import AttentionPanel from "./AttentionPanel";
import TelemetryBar from "./TelemetryBar";
import FeaturesPanel from "./FeaturesPanel";
import GroundPanel from "./GroundPanel";
import LensPanel from "./LensPanel";
import ProbePanel from "./ProbePanel";
import SteeringPanel from "./SteeringPanel";
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
  sessionPatchGraph?: {
    available: boolean;
    n_nodes: number;
    n_edges: number;
    /** The graph's OWN pair, so the panel prefills with the prompts it was
     *  measured on rather than the patch section's -- a `.mri` can carry a
     *  graph and no patch trace. */
    clean?: string;
    corrupt?: string;
  };
  sessionGround?: { available: boolean; question: string };
  /**
   * A generation finished — succeeded or failed, the server records both.
   * The agents panel is a sibling of this component, so it cannot see the
   * stream end any other way.
   */
  onGenerated?: () => void;
}

// What the model button says before anything has been measured. The button
// has to name something on the first paint, and a machine with an empty cache
// has nothing to name — so this is a starting point, not a claim about this
// machine. `suggested` below replaces it with a model this machine actually
// holds as soon as the scan answers.
//
// This was a two-element `CURATED` array whose second element nothing read
// and whose name promised a list rendered somewhere. The list that exists is
// the picker sheet, and it is built from the disk scan, not from here.
const FALLBACK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct";

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
  const [pick, setPick] = useState(FALLBACK_MODEL);
  const [pickerOpen, setPickerOpen] = useState(false);
  // "" is Automatic: nothing is sent and the server chooses, which is
  // what every load did before this control existed. Only a deliberate
  // choice names a device.
  const [device, setDevice] = useState("");
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

  // Has anything but the fallback decided what this button says? Set by every
  // path that means a model was CHOSEN — adopting what the server already has,
  // the picker sheet, the size guard's override. The scan below is the one
  // writer that defers to it, because a suggestion must never overwrite a
  // choice somebody made while it was still walking the disk.
  const chosen = useRef(false);

  // The server keeps its model across page loads; the picker did not. You
  // came back to a tab that said one model in the badge and another in the
  // picker, and Generate quietly swapped to the second. Adopt what is
  // actually loaded, once, on first sight.
  const adopted = useRef(false);
  useEffect(() => {
    if (adopted.current || !model?.loaded || !model.hf_id) return;
    adopted.current = true;
    chosen.current = true;
    setPick(model.hf_id);
    setSource(model.device === "ollama" ? "ollama" : "hf");
  }, [model?.loaded, model?.hf_id, model?.device]);

  // Nothing loaded, so nothing has named a model yet and the button is showing
  // a baked guess. Ask the disk instead: `/api/models/discovered` is the same
  // scan the picker's "On this machine" tab renders, so the suggestion and the
  // list you check it against come from one answer rather than two.
  //
  // Why the scan and not `/api/models/local`, which is cheaper: the cache is
  // not all language models. It also holds SAEs, embedders, diffusion and
  // segmentation weights, and metadata-only directories with no weights at
  // all. The scan reads each config and marks `loadable`, which is exactly
  // the difference between suggesting a name and suggesting a name that will
  // fail minutes later on a tokenizer traceback. Smallest first, because a
  // suggestion is a starting point — the 70B on the same disk is one click
  // away in the sheet, and the sheet sorts largest-first for that.
  //
  // Waits for the session answer rather than firing on mount: a loaded model
  // outranks any suggestion, and racing the adoption above would flash a name
  // this session is not using. The fallback stays the synchronous initial
  // value, so the button never paints empty and never paints twice on a
  // machine that has nothing cached.
  const suggested = useRef(false);
  useEffect(() => {
    if (suggested.current || VIEWER || !model) return;
    suggested.current = true;
    if (model.loaded) return;
    let live = true;
    void getDiscovered()
      .then((d) => {
        if (!live || chosen.current) return;
        const usable = d.models
          .filter((m) => m.loadable && m.kind === "hf-cache")
          // Ties on size are common — two quantisations of one repo — so the
          // id breaks them and the button does not depend on scan order.
          .sort((a, b) => a.size_gb - b.size_gb || a.id.localeCompare(b.id));
        if (usable.length) setPick(usable[0].id);
      })
      // An empty cache, a refusing scan, no server: the fallback above is
      // already on screen and is still a name you can load.
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [model]);

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

  // WHETHER THE OPEN RECORDING CARRIES A LOGIT LENS.
  //
  // Read here rather than handed down beside `sessionPatch`, `sessionGround`
  // and `sessionPatchGraph`. Those three are published by BOTH halves of the
  // wire — `runtime.session_info` for the app and `viewer.ts`'s `state()` for
  // the zero-install build — so `App.tsx` has one answer to pass down. `lens`
  // is not yet in `session_info`, and a prop that is `undefined` in one of the
  // two builds is a gate that silently never opens in that build, with
  // nothing on screen saying why. Asking `/api/session/state` here reads
  // whichever answer THIS build actually serves, and the app build starts
  // showing the panel the moment `session_info` grows the same key — without
  // a second edit in a third file to remember.
  const [sessionLens, setSessionLens] = useState<SessionState["lens"]>(
    undefined,
  );
  useEffect(() => {
    if (!replay) {
      // Not `available: false`. The file is closed, so there is no file to
      // have an opinion about — and the mounts below ask `?.available`, which
      // reads both states the same way on purpose.
      setSessionLens(undefined);
      return;
    }
    let live = true;
    void getSessionState()
      .then((s) => live && setSessionLens(s.lens))
      // A refusing route or an older server: the panel stays absent, which is
      // the same answer as "this file carries no lens" and is the safe one.
      // Mounting on an unknown would put a heading over a refusal.
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
      const result = await loadModel(id, src, confirm, device);
      // A stopped load is not a failure. Say what happened and stay put.
      if ("cancelled" in result) {
        setMeta(result.message);
        return false;
      }
      setMeta(`loaded in ${((performance.now() - t) / 1000).toFixed(1)}s`);
      await onModelChange();
      // THE SIX PANELS BELOW read the session through a shared cache keyed on
      // `epoch`, and `setEpoch(0)` is a no-op when it is already 0 — which it
      // is on a fresh page, because epoch counts GENERATIONS. Without this
      // they keep the answer fetched before the model existed and go on
      // telling the reader to load one.
      invalidateSession();
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
    // Same rule in the zero-install viewer, which only ever shows recordings.
    // But attention is not the only thing a recording carries: `session.build`
    // writes `patch`, `patch_graph` and `ground` sections too, and returning
    // the attention panel alone meant a file sent BECAUSE of its patching
    // trace, its patching graph or its grounding result opened with that
    // finding sitting unread inside it.
    //
    // Each of the three is gated on the same `available` flag the full build
    // gates its own replay mounts on below, read off the same session state
    // App.tsx already hands down — so a section appears exactly when the
    // opened file holds one, and never as a form with nothing behind it.
    //
    // ProbePanel and PatchscopePanel stay out. Neither is a section a `.mri`
    // can carry, so both would need the live model this page does not have.
    return (
      <>
        <AttentionPanel epoch={epoch} replay />
        {sessionGround?.available && (
          <GroundPanel epoch={epoch} recorded={sessionGround} />
        )}
        {sessionPatch?.available && (
          <PatchPanel epoch={epoch} recorded={sessionPatch} />
        )}
        {sessionPatchGraph?.available && (
          <PatchGraphPanel
            epoch={epoch}
            /* ITS OWN PROMPTS, not the patch section's. A `.mri` can carry a
               graph and no patch trace, and this handed the panel the other
               section's pair — so those files prefilled with the hardcoded
               demo prompts and showed a measured graph above a pair it had
               nothing to do with. */
            recorded={{
              clean: sessionPatchGraph.clean ?? "",
              corrupt: sessionPatchGraph.corrupt ?? "",
            }}
          />
        )}
        {/* The fourth section, and the one that had been in the format from
            the beginning. `LensPanel` is mounted from inside `FeaturesPanel`
            in the app, and that panel is `!replay` because it also holds
            live-model controls — so the lens was unreachable on both
            surfaces at once, and `viewer.ts` answered `/api/lens` with
            "install ModelMRI" over bytes that already held the trajectory. */}
        {sessionLens?.available && (
          <LensPanel epoch={epoch} recorded={sessionLens} />
        )}
      </>
    );
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
        {/* Renders nothing unless this machine has more than one device: a
            select with one option implies a decision nobody has. Empty value
            means Automatic, so a machine that never touches it behaves
            exactly as it did before the control existed. */}
        {source !== "ollama" && (
          <DevicePicker value={device} onChange={setDevice} disabled={busy !== ""} />
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
                  chosen.current = true;
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
          chosen.current = true;
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
      {/* `|| replay` is not redundant. A recording with no attention slices
          leaves `epoch` at 0 — the meta call that sets it says `available:
          false` — so the section vanished from a page that is otherwise all
          about a file somebody sent, with nothing anywhere saying the file
          carries no attention. It carries its own explanation now, and the
          live side is unchanged: with no model and no generation the RUN
          section above is the answer, and a second box saying so is noise. */}
      {(epoch > 0 || replay) && introspectable && (
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
          behind it". `api.ts`'s `patchGraph` refuses in the demo too — that is
          the second lock, and `tests/demo_check.py` checks this one. The
          viewer is the other half of that split: it has no model either, but a
          `.mri` CAN carry a graph, so the VIEWER branch above mounts the
          recorded panel and `patchGraph` lets that call reach the file. */}
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
      {/* Beside the probe because they share a store: the probe's "save the
          best layer's direction as" field writes into the same directory this
          panel lists, and a reader who has just saved one should not have to
          go looking for where it went.

          NOT OFFERED in the demo or viewer builds, for the reason
          `PatchGraphPanel` above is not: every control here needs either a
          live residual stream or a directory on the reader's own disk, and a
          panel whose buttons can only apologise reads as a broken measurement
          rather than as a page with nothing behind it. `api.ts` refuses each
          of its calls as the second lock, and `tests/demo_check.py` records
          both. */}
      {introspectable && !replay && !DEMO && !VIEWER && (
        <SteeringPanel
          epoch={epoch}
          prompt={lastPrompt}
          /* Raised only around this panel's own A/B, which turns steering off
             and on again underneath whatever else is on screen. A direction
             the reader applied deliberately does NOT raise it: generating
             under one is the point of applying it. */
          onSteering={setSteering}
        />
      )}
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
        <PatchGraphPanel
            epoch={epoch}
            /* ITS OWN PROMPTS, not the patch section's. A `.mri` can carry a
               graph and no patch trace, and this handed the panel the other
               section's pair — so those files prefilled with the hardcoded
               demo prompts and showed a measured graph above a pair it had
               nothing to do with. */
            recorded={{
              clean: sessionPatchGraph.clean ?? "",
              corrupt: sessionPatchGraph.corrupt ?? "",
            }}
          />
      )}
      {/* Same rule once more, for the section that has been in the format
          since the format existed. On a live model the lens lives INSIDE the
          features panel below — it is the answer for a model with no sparse
          autoencoder, which is most of them, and it belongs beside the one it
          stands in for. That panel is `!replay` because its other controls
          need the model, so on a recording the lens has to be mounted here or
          it is mounted nowhere: `runtime.logit_lens` would serve the recorded
          trajectory and no surface could ask. */}
      {replay && sessionLens?.available && (
        <LensPanel epoch={epoch} recorded={sessionLens} />
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

