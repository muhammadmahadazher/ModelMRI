import { useEffect, useState } from "react";
import {
  CustomCandidate,
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
} from "./api";
import { useScanOnData } from "./useScanOnData";
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

export default function CustomPanel() {
  const [status, setStatus] = useState<CustomStatus | null>(null);
  const [cands, setCands] = useState<{
    adapters: CustomCandidate[];
    torchscript: CustomCandidate[];
    roots: string[];
  } | null>(null);
  const [run, setRun] = useState<CustomRun | null>(null);
  const [shape, setShape] = useState("");
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");
  // A folder to also look in. Kept here rather than in the server's config so
  // it lasts exactly one run — widening where a local tool will import from
  // is not a setting that should quietly persist.
  const [folder, setFolder] = useState("");
  const [manual, setManual] = useState("");
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

  async function onLoad(path: string) {
    setBusy(path);
    setErr("");
    setRun(null);
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
            <span className="meta">
              scans {status?.roots?.[0] ?? "the directory you launched in"} for
              adapters and TorchScript · reads text, imports nothing
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
            {cands.adapters.length === 0 && cands.torchscript.length === 0 && (
              <p className="resting-empty">
                Nothing found under {cands.roots.join(", ")}. An adapter is a
                Python file with <code>def load(): return your_model</code> —
                copy <code>examples/adapter_template.py</code> next to your
                training code, or set <code>MODELMRI_MODELS_DIR</code>.
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
                      onClick={() => void onLoad(c.path)}
                      title={c.path}
                    >
                      <span className="cand-name">{c.name}</span>
                      <span className="pill tiny">{c.mb} MB</span>
                      {c.kind && (
                        <span
                          className={`pill tiny ${c.kind === "torchscript" ? "ok" : ""}`}
                          title={
                            c.kind === "torchscript"
                              ? "A TorchScript archive — it loads, but PyTorch strips the hooks this panel reads activations through"
                              : c.kind === "checkpoint"
                                ? "Weights only. It needs an adapter that builds your model class and loads them in."
                                : c.kind === "legacy"
                                  ? "Saved by a torch older than 1.6, or not a torch file at all"
                                  : "Could not be read as an archive"
                          }
                        >
                          {c.kind === "checkpoint" ? "weights only" : c.kind}
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
        <span className="pill">{status.n_params.toLocaleString()} params</span>
        <span className="pill">{status.n_modules} modules</span>
        <span className="pill">{status.source}</span>
        {status.n_trainable !== status.n_params && (
          <span className="pill">
            {status.n_trainable.toLocaleString()} trainable
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
        <button className="green" onClick={() => void onRun()} disabled={busy !== ""}>
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
                          {l.mean.toFixed(2)} ± {(l.std ?? 0).toFixed(2)}
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
            {run.truncated && (
              <span className="pill warn">only the first 512 layers are shown</span>
            )}
          </div>
        </>
      )}

      <div className="hint">
        one real forward pass, hooked at every leaf module · dead = exactly
        zero, saturated = within 1% of the activation's own bound · statistics
        exclude nan and inf so one bad value can't hide where it started
      </div>
    </div>
  );
}
