import { useEffect, useState } from "react";
import {
  CustomCandidates,
  CustomLayer,
  CustomRun,
  CustomStatus,
  errorText,
  getCustom,
  getCustomCandidates,
  scanFolder,
  loadCustom,
  runCustom,
  unloadCustom,
  readGguf,
  GgufReport,
} from "./api";
import { useScanOnData } from "./useScanOnData";

/** Why the measurement controls are off for a TorchScript archive.
 *
 *  Not a guess: `custom.py` raises this from the hook-registration loop and
 *  its comment says the failure is universal — "torch installs a generated `fail()` over both hook APIs on RecursiveScriptModule, which is what `torch.jit.load` returns". */
const TORCHSCRIPT_WHY =
  "torch installs a generated `fail()` over both hook APIs on RecursiveScriptModule, which is what `torch.jit.load` returns — so every TorchScript archive on disk is un-hookable. Load the .py adapter that built this model to measure it.";
import CustomAblate from "./CustomAblate";
import GgufReader from "./GgufReader";
import RestingSketch from "./RestingSketch";

/** Health of one layer, in the order a person would notice it. */
function verdict(l: CustomLayer): { label: string; tone: string } | null {
  if (l.n_nonfinite > 0) return { label: `${l.n_nonfinite} nan/inf`, tone: "bad" };
  if (l.pct_saturated !== null && l.pct_saturated >= 50)
    return { label: `${l.pct_saturated.toFixed(0)}% saturated`, tone: "warn" };
  if (l.is_activation && l.pct_zero !== null && l.pct_zero >= 90)
    return { label: `${l.pct_zero.toFixed(0)}% dead`, tone: "bad" };
  if (l.is_activation && l.pct_zero !== null && l.pct_zero >= 60)
    return { label: `${l.pct_zero.toFixed(0)}% dead`, tone: "warn" };
  return null;
}

/** Where this layer's mean sits inside its own range, 0..1. */
function meanAt(l: CustomLayer): number | null {
  if (l.mean === null || l.min === null || l.max === null) return null;
  const span = l.max - l.min;
  if (!isFinite(span) || span <= 0) return 0.5;
  return Math.min(1, Math.max(0, (l.mean - l.min) / span));
}

function shapeText(s: number[]): string {
  return s.length ? s.join("×") : "—";
}

export default function CustomPanel({
  onModelChange,
}: {
  /** Called when this panel has changed which model the server holds — today
   *  only the GGUF loader does. Optional so the panel stays usable standalone. */
  onModelChange?: () => void;
} = {}) {
  const [status, setStatus] = useState<CustomStatus | null>(null);
  // The SHARED type, not a local copy of its shape. This inline literal is
  // why three fields the server had started returning were invisible here:
  // the panel described the payload for itself, so nothing told it when the
  // payload grew.
  const [cands, setCands] = useState<CustomCandidates | null>(null);
  const [run, setRun] = useState<CustomRun | null>(null);
  const [shape, setShape] = useState("");
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");
  // A folder to also look in. Kept here rather than in the server's config so
  // it lasts exactly one run — widening where a local tool will import from
  // is not a setting that should quietly persist.
  const [folder, setFolder] = useState("");
  const [manual, setManual] = useState("");
  // A GGUF the reader opened. Kept beside the candidate list rather than
  // replacing the panel, so "back to the list" is one click and the scan is
  // not lost.
  const [gguf, setGguf] = useState<GgufReport | null>(null);
  // A counter, not layers.length: two runs of the same model have the same
  // layer count, and a scan that never fires again says nothing.
  const [runId, setRunId] = useState(0);
  const scanRef = useScanOnData(runId || null);

  // Status only. Finding candidates walks the filesystem, so it waits to be
  // asked — same rule as every other panel.
  useEffect(() => {
    let live = true;
    void getCustom()
      .then((s) => live && setStatus(s))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, []);

  async function onFind() {
    setBusy("find");
    setErr("");
    try {
      setCands(await getCustomCandidates());
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }

  async function onScanFolder() {
    if (!folder.trim()) return;
    setBusy("scan");
    setErr("");
    try {
      const found = await scanFolder(folder);
      setCands(found);
      if (found.adapters.length === 0 && found.torchscript.length === 0) {
        // Saying "nothing found" without saying WHERE it looked sends people
        // to check a path that was never the problem.
        setErr(`Nothing under ${found.added}. Looked in: ${found.roots.join(", ")}`);
      }
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }

  /** Read a GGUF's header instead of trying to load it.
   *
   *  Kept as its own path rather than a branch inside `onLoad`, because these
   *  are different verbs with different outcomes: loading builds a model this
   *  tool can run, and a quantised GGUF is not one. Offering the same button
   *  for both would promise something that can only refuse.
   */
  async function onRead(path: string) {
    setBusy(path);
    setErr("");
    try {
      setGguf(await readGguf(path));
    } catch (e) {
      setErr(errorText(e));
      setGguf(null);
    } finally {
      setBusy("");
    }
  }

  async function onLoad(path: string) {
    setBusy(path);
    setErr("");
    setRun(null);
    setGguf(null);
    try {
      const s = await loadCustom(path);
      setStatus(s);
      setShape((s.input_shape ?? []).join(", "));
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }

  async function onRun() {
    setBusy("run");
    setErr("");
    try {
      const parsed = shape
        .split(/[,\s×x]+/)
        .map((p) => p.trim())
        .filter(Boolean)
        .map(Number);
      const bad = parsed.some((n) => !Number.isFinite(n) || n <= 0);
      if (shape.trim() && bad) {
        setErr(`"${shape}" is not a shape — give positive integers, like 8, 20`);
        return;
      }
      const r = await runCustom(parsed.length ? parsed : null);
      setRun(r);
      setRunId((n) => n + 1);
      setShape(r.input_shape.join(", "));
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }

  async function onUnload() {
    setBusy("unload");
    try {
      setStatus(await unloadCustom());
      setRun(null);
      setShape("");
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }

  const header = (
    <div className="sect">
      <span className="dot d-custom" />
      <h2 className="h-custom">CUSTOM MODEL — YOUR OWN NETWORK</h2>
      <span className="rule" />
    </div>
  );

  // ---------------------------------------------------------- resting state
  if (!status?.loaded) {
    return (
      <div className="panel">
        {header}
        {!cands ? (
          <div className="resting">
            <RestingSketch kind="custom" />
            <p>
              Trained something yourself? Point ModelMRI at it and get a
              layer-by-layer map of one real forward pass — shapes, activation
              statistics, dead units, and anything that has gone non-finite.
            </p>
            <button className="green" onClick={() => void onFind()} disabled={busy !== ""}>
              {busy === "find" ? "Looking…" : "Find models here"}
            </button>
            {/* EVERY root, not the first of them. The two "nothing found"
                messages in this same panel already join the whole list — so
                the sentence before the click promised one directory and the
                sentence after it reported three, and a model sitting in the
                second one looked like a model the scan had missed. */}
            <span
              className="meta"
              title={status?.roots?.length ? status.roots.join(", ") : undefined}
            >
              scans{" "}
              {!status?.roots?.length
                ? "the directory you launched in"
                : status.roots.length === 1
                  ? status.roots[0]
                  : `${status.roots.length} directories, starting with ${status.roots[0]}`}{" "}
              for adapters and TorchScript · reads text, imports nothing
            </span>

            {/* The scan used to be limited to the directory the server was
                launched in, which is the wrong question to ask somebody whose
                model lives on another drive: their answer is "it is over
                there" and the tool's was "restart me somewhere else". The
                folder joins the allowed roots for this run — it does not
                bypass them, so the boundary moves once, deliberately, when a
                person asks it to. */}
            <div className="row cust-elsewhere">
              <input
                value={folder}
                onChange={(e) => setFolder(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && void onScanFolder()}
                placeholder="…or a folder anywhere on this machine"
                spellCheck={false}
              />
              <button
                className="ghost sm"
                onClick={() => void onScanFolder()}
                disabled={busy !== "" || !folder.trim()}
              >
                {busy === "scan" ? "scanning…" : "Look here too"}
              </button>
            </div>
          </div>
        ) : (
          <div className="cand-wrap">
            {/* The reader sits above the list rather than replacing the panel,
                so "back to the list" costs one click and the scan survives. */}
            {gguf && (
              <GgufReader
                report={gguf}
                onClose={() => setGguf(null)}
                onLoaded={onModelChange}
              />
            )}
            {cands.adapters.length === 0 && cands.torchscript.length === 0 && (
              <p className="resting-empty">
                Nothing found under {cands.roots.join(", ")}. An adapter is a
                Python file with <code>def load(): return your_model</code> —
                copy <code>examples/adapter_template.py</code> next to your
                training code, or set <code>MODELMRI_MODELS_DIR</code>.
              </p>
            )}
            {/* THE CAP, SAID. Both walks stop at 40 and a reader choosing
                from this list believes it is every model they have. Rendered
                once above both lists rather than twice, because the walks
                share a limit and two identical warnings read as two
                problems. */}
            {cands.truncated && (
              <p className="hint">
                Showing {cands.adapters.length} of {cands.n_adapters_found}{" "}
                adapter(s) and {cands.torchscript.length} of{" "}
                {cands.n_torchscript_found} checkpoint(s) found here — the walk
                stops after that many so a large folder cannot hang the panel.
                Anything not listed can still be loaded by typing its path.
              </p>
            )}
            {cands.adapters.length > 0 && (
              <>
                <div className="meta cand-head">adapters</div>
                <div className="cand-list">
                  {cands.adapters.map((c) => (
                    <button
                      key={c.path}
                      className="cand"
                      disabled={busy !== ""}
                      onClick={() => void onLoad(c.path)}
                      title={c.path}
                    >
                      <span className="cand-name">{c.name}</span>
                      {c.has_example && <span className="pill tiny">example_input</span>}
                      <span className="spacer" />
                      <span className="cand-dir">{c.dir}</span>
                    </button>
                  ))}
                </div>
              </>
            )}
            {cands.torchscript.length > 0 && (
              <>
                {/* "checkpoints", not "torchscript". The heading used to
                    name one of the things this list can hold, so a state_dict
                    was filed under TorchScript -- and those fail differently,
                    so the heading was telling the reader the wrong story
                    before they clicked. Each row now says what it is. */}
                <div className="meta cand-head">checkpoints</div>
                <div className="cand-list">
                  {cands.torchscript.map((c) => (
                    <button
                      key={c.path}
                      className="cand"
                      disabled={busy !== ""}
                      // A GGUF is read first and loaded only on request:
                      // dequantising costs about 3x the file in memory, so the
                      // reader prices it before offering the button. Older
                      // copy here said transformers could not
                      // run a quantised one, and offering the same button for
                      // both would promise something that can only refuse.
                      onClick={() =>
                        void (c.kind === "gguf" ? onRead(c.path) : onLoad(c.path))
                      }
                      title={c.path}
                    >
                      <span className="cand-name">{c.name}</span>
                      <span className="pill tiny">{c.mb} MB</span>
                      {c.kind && (
                        <span
                          className={`pill tiny ${
                            c.kind === "torchscript" || c.kind === "gguf" ? "ok" : ""
                          }`}
                          title={
                            c.kind === "gguf"
                              ? "Open it for architecture, every metadata key, and a full tensor table with real bits-per-weight — then load it for the lens, attention and patching. Dequantising costs about 3x the file in memory, and the panel says how much before you commit."
                              : c.kind === "torchscript"
                                ? "A TorchScript archive — it loads, but PyTorch strips the hooks this panel reads activations through"
                                : c.kind === "checkpoint"
                                  ? "Weights only. It needs an adapter that builds your model class and loads them in."
                                  : c.kind === "legacy"
                                    ? "Saved by a torch older than 1.6, or not a torch file at all"
                                    : "Could not be read as an archive"
                          }
                        >
                          {c.kind === "checkpoint"
                            ? "weights only"
                            : c.kind === "gguf"
                              ? "open it"
                              : c.kind}
                        </span>
                      )}
                      <span className="spacer" />
                      <span className="cand-dir">{c.dir}</span>
                    </button>
                  ))}
                </div>
              </>
            )}
            <div className="row cand-manual">
              <input
                className="combo grow"
                placeholder="…or a path: models/my_net_adapter.py"
                value={manual}
                onChange={(e) => setManual(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && manual.trim() && void onLoad(manual.trim())}
              />
              <button
                className="ghost sm"
                onClick={() => manual.trim() && void onLoad(manual.trim())}
                disabled={busy !== "" || !manual.trim()}
              >
                Load
              </button>
              <button
                className="ghost sm"
                // Clears the error as well as the list. "Back" is the control
                // for "I am done with this" and an error that outlives the
                // thing it was about is a stale claim about the current state
                // -- the refusal for a file you are no longer looking at,
                // sitting under the resting copy as if it applied to it.
                onClick={() => {
                  setCands(null);
                  setErr("");
                }}
                disabled={busy !== ""}
              >
                Back
              </button>
            </div>
          </div>
        )}
        {err && <div className="hint err refusal">{err}</div>}
      </div>
    );
  }

  // ------------------------------------------------------------- loaded
  const layers = run?.layers ?? [];
  const slowest = layers.reduce((m, l) => Math.max(m, l.ms), 0.0001);

  return (
    <div ref={scanRef} className="panel custom">
      {header}

      <div className="row" style={{ marginBottom: 10 }}>
        <span className="pill on">{status.name}</span>
        <span className="pill">{(status.n_params ?? 0).toLocaleString()} params</span>
        <span className="pill">{status.n_modules ?? 0} modules</span>
        <span className="pill">{status.source}</span>
        {status.n_trainable !== status.n_params && (
          <span className="pill">
            {(status.n_trainable ?? 0).toLocaleString()} trainable
          </span>
        )}
        <span className="spacer" />
        <button className="ghost sm" onClick={() => void onUnload()} disabled={busy !== ""}>
          Unload
        </button>
      </div>

      <div className="row custom-input">
        <label className="meta" htmlFor="custom-shape">
          input shape
        </label>
        <input
          id="custom-shape"
          className="combo"
          value={shape}
          onChange={(e) => setShape(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && busy === "" && void onRun()}
          placeholder="8, 20"
        />
        {/* DISABLED FOR TORCHSCRIPT, because `custom.inspect` — which this
            calls — registers a forward hook on every leaf module, and
            custom.py's own comment says the failure is universal rather than
            a corner case: "every TorchScript archive on disk is un-hookable,
            all of them". The click answered 422 every time. */}
        <button
          className="green"
          onClick={() => void onRun()}
          disabled={busy !== "" || status.source === "torchscript"}
          title={status.source === "torchscript" ? TORCHSCRIPT_WHY : undefined}
        >
          {busy === "run" ? "Running…" : "Run forward pass"}
        </button>
        {status.input_reason && (
          <span className={`meta ${status.input_origin === "inferred" ? "guessy" : ""}`}>
            {status.input_origin === "inferred" ? "inferred — " : ""}
            {status.input_reason}
          </span>
        )}
      </div>

      {err && <div className="hint err refusal">{err}</div>}

      {run && (
        <>
          <div className="layer-table" role="table" aria-label="layer map">
            <div className="layer-row head" role="row">
              <span>layer</span>
              <span>type</span>
              <span>output</span>
              <span className="num">params</span>
              <span>activation</span>
              <span className="num">ms</span>
            </div>
            {layers.map((l) => {
              const v = verdict(l);
              const at = meanAt(l);
              return (
                <div className={`layer-row ${v ? v.tone : ""}`} key={`${l.order}-${l.name}`} role="row">
                  <span className="l-name" title={l.name}>
                    {l.name}
                    {/* A module that ran more than once in this pass. Without
                        it the table repeats a name and looks like it has
                        duplicated a row -- correct data reading as a bug,
                        which is the failure this panel keeps having. A
                        two-branch model applies one encoder to two inputs;
                        every leaf under it fires twice, and both readings are
                        real and different. */}
                    {(l.calls_total ?? 1) > 1 && (
                      <span
                        className="l-call"
                        title={`This module ran ${l.calls_total} times in one forward pass — your forward() calls it more than once. This row is reading ${l.call}.`}
                      >
                        {l.call}/{l.calls_total}
                      </span>
                    )}
                  </span>
                  <span className="l-kind">{l.kind}</span>
                  <span className="l-shape">{shapeText(l.out_shape)}</span>
                  <span className="num l-params">
                    {l.n_params ? l.n_params.toLocaleString() : "—"}
                  </span>
                  <span className="l-act">
                    {l.mean === null ? (
                      <span className="meta">{l.note || "—"}</span>
                    ) : (
                      <>
                        <span className="l-bar" title={`min ${l.min}  mean ${l.mean}  max ${l.max}`}>
                          {at !== null && <i className="l-mean" style={{ left: `${at * 100}%` }} />}
                        </span>
                        <span className="l-nums">
                          {/* `± 0.00` is not a smaller claim than a real
                              spread — it is a LARGER one. It says this
                              activation was measured with zero variance,
                              which is a statement about precision nobody
                              made. `mean === null` is already handled three
                              lines up; the deviation beside it was still
                              being invented. */}
                          {l.mean.toFixed(2)}
                          {l.std === null ? "" : ` ± ${l.std.toFixed(2)}`}
                        </span>
                      </>
                    )}
                    {v && <span className={`pill tiny ${v.tone}`}>{v.label}</span>}
                  </span>
                  <span className="num l-ms">
                    <i className="l-time" style={{ width: `${(l.ms / slowest) * 100}%` }} />
                    {l.ms.toFixed(2)}
                  </span>
                </div>
              );
            })}
          </div>

          <div className="row custom-out">
            <span className="pill">
              {shapeText(run.input_shape)} → {shapeText(run.output_shape)}
            </span>
            <span className="pill">{run.total_ms.toFixed(1)} ms</span>
            {run.output.argmax !== undefined && (
              <span className="pill on">
                argmax {run.output.argmax}
                {run.labels?.[run.output.argmax]
                  ? ` · ${run.labels[run.output.argmax]}`
                  : ""}
              </span>
            )}
            {run.output.nonfinite && <span className="pill bad">output is nan/inf</span>}
            {/* SOME of the outputs, which is the case that used to be
                invisible: `nonfinite` is only true when every slot is
                unusable, so a half-nan output showed a confident argmax and
                nothing else. The argmax beside this is ranked over the finite
                values, and this says how many it stepped over. */}
            {!run.output.nonfinite && (run.output.n_nonfinite ?? 0) > 0 && (
              <span className="pill warn">
                {run.output.n_nonfinite} of {run.output.n_out} outputs are
                nan/inf
              </span>
            )}
            {run.truncated && (
              <span className="pill warn">only the first 512 layers are shown</span>
            )}
          </div>
        </>
      )}

      {/* The causal half, sited under the map rather than beside it. The map
          says what each layer emitted; this says what the answer would be
          without it, and reading the second without having seen the first is
          how a reader ends up believing a dead layer was load-bearing. Shown
          only once a model is loaded — there is nothing to sweep before. */}
      {status.loaded && <CustomAblate epoch={runId} source={status.source} />}

      <div className="hint">
        one real forward pass, hooked at every leaf module · dead = exactly
        zero, saturated = within 1% of the activation's own bound · statistics
        exclude nan and inf so one bad value can't hide where it started
      </div>
    </div>
  );
}
