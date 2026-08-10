import LensPanel from "./LensPanel";
import { useEffect, useRef, useState } from "react";
import { useScanOnData } from "./useScanOnData";
import {
  errorText,
  FeaturesSummary,
  getFeatureDetail,
  getFeaturesSummary,
  getSAE,
  getSAEOptions,
  loadSAEFrom,
  promptOnce,
  SAEOption,
  SAEStatus,
  setSteer,
} from "./api";

interface Props {
  epoch: number; // bumps after each generation
  prompt: string; // the prompt of the last generation, for steering A/B
  /** Raised while the steering hook is installed on the model. Generate must
   *  be locked out for that window: the hook is global to the runtime, so a
   *  generation started mid-A/B comes back steered with nothing on screen
   *  saying so. */
  onSteering?: (active: boolean) => void;
}

/** SAE feature browser: token -> top features -> heat view -> steering A/B. */
/** Where the default SAE reads. Overridden per registry entry. */
const DEFAULT_HOOK = "blocks.8.hook_resid_pre";

export default function FeaturesPanel({ epoch, prompt, onSteering }: Props) {
  const scanRef = useScanOnData(epoch);
  const [sae, setSae] = useState<SAEStatus | null>(null);
  const [busy, setBusy] = useState("");
  const [summary, setSummary] = useState<FeaturesSummary | null>(null);
  const [tokenSel, setTokenSel] = useState(-1);
  const [featSel, setFeatSel] = useState(-1);
  const [heat, setHeat] = useState<number[] | null>(null);
  const [peak, setPeak] = useState(-1);
  const peakRef = useRef<HTMLSpanElement>(null);
  const [scale, setScale] = useState(-40);
  const [ab, setAb] = useState<{ base: string; steered: string } | null>(null);
  const [err, setErr] = useState("");
  // Which SAEs exist for the model that is loaded. Empty is the common,
  // honest answer — an SAE is trained per model and public ones exist for
  // only a handful; `catalogue` below is the ones this build knows.
  const [opts, setOpts] = useState<{
    model: string | null;
    matching: SAEOption[];
    usable: SAEOption[];
    catalogue: SAEOption[];
  } | null>(null);
  const [custom, setCustom] = useState("");

  useEffect(() => {
    void getSAE().then(setSae);
  }, []);

  useEffect(() => {
    setSummary(null);
    setTokenSel(-1);
    setFeatSel(-1);
    setHeat(null);
    setPeak(-1);
    setAb(null);
    if (sae?.loaded) void refreshSummary();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [epoch, sae?.loaded]);

  async function refreshSummary() {
    try {
      setSummary(await getFeaturesSummary(8));
      setErr("");
    } catch (e) {
      setErr(errorText(e));
    }
  }

  useEffect(() => {
    let live = true;
    void getSAEOptions()
      .then((o) => live && setOpts(o))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [epoch]);

  async function onLoadFrom(repo: string, hook: string) {
    setBusy("sae");
    setErr("");
    try {
      setSae(await loadSAEFrom(repo, hook));
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }


  async function onPickFeature(fid: number) {
    setFeatSel(fid);
    setAb(null);
    try {
      const d = await getFeatureDetail(fid);
      const max = d.max || 1;
      setHeat(d.activations.map((a) => a / max));
      // The API already tells us which token fires hardest. A default
      // generation is 256 tokens, so without this the one chip worth looking
      // at is somewhere in a strip several thousand pixels wide.
      setPeak(d.argmax);
      requestAnimationFrame(() =>
        peakRef.current?.scrollIntoView({ block: "nearest", inline: "center" }),
      );
    } catch (e) {
      setErr(errorText(e));
    }
  }

  async function onSteerTest() {
    if (featSel < 0) return;
    // The A/B re-runs the prompt that produced this analysis. After a reload
    // the panels are restored from the server, which keeps the activations
    // but not the prompt text — running the A/B on "" would compare two
    // completions of nothing and present them as a steering result.
    if (!prompt.trim()) {
      setErr(
        "Generate once in this tab first — the A/B re-runs your prompt, and " +
          "this analysis was restored from the server without it.",
      );
      return;
    }
    setBusy("steer");
    setErr("");
    onSteering?.(true);
    try {
      await setSteer(null);
      const base = (await promptOnce(prompt, 24, 0, false)).generation;
      await setSteer(featSel, scale);
      const steered = (await promptOnce(prompt, 24, 0, false)).generation;
      await setSteer(null); // always leave the model clean
      setAb({ base, steered });
    } catch (e) {
      setErr(errorText(e));
      await setSteer(null);
    } finally {
      // Must pair with the raise above on EVERY path, including the throw:
      // the hook is global to the runtime, so leaving this latched locks
      // Generate out for the rest of the session.
      onSteering?.(false);
      setBusy("");
    }
  }

  if (!sae) return null;

  if (!sae.loaded) {
    return (
      <div ref={scanRef} className="panel feat">
        <div className="sect">
          <span className="dot d-feat" />
          <h2 className="h-feat">FEATURES — THE CONCEPTS INSIDE</h2>
          <span className="rule" />
        </div>
        {opts?.usable.length ? (
          <>
            {opts.usable.map((o) => (
              <div className="row" style={{ marginTop: 12 }} key={o.repo}>
                <button
                  className="violet"
                  onClick={() => void onLoadFrom(o.repo, o.default_hook)}
                  disabled={busy !== ""}
                >
                  {busy === "sae" ? "Loading SAE…" : `Load ${o.label}`}
                </button>
                <span className="meta">
                  matches {opts.model} · d_in {o.d_in} · first run downloads it
                </span>
              </div>
            ))}
          </>
        ) : (
          <div className="resting-empty">
            <b>No sparse autoencoder exists for {opts?.model ?? "this model"}.</b>{" "}
            An SAE is trained against one model at one layer — it is GPU-months
            of someone else's work, not a setting. Public ones exist for only
            a handful of models.
            {opts?.catalogue.length ? (
              <> Known SAEs: {opts.catalogue.map((c) => c.repo.split("/")[1]).join(", ")}.</>
            ) : null}{" "}
            The logit lens below asks a different question of the same
            residual stream, and works on every model.
          </div>
        )}

        <div className="row cand-manual">
          <input
            className="combo grow"
            placeholder="…or a SAELens repo: owner/name"
            value={custom}
            onChange={(e) => setCustom(e.target.value)}
          />
          <button
            className="ghost sm"
            disabled={busy !== "" || !custom.trim()}
            onClick={() =>
              void onLoadFrom(custom.trim(), DEFAULT_HOOK)
            }
          >
            Load
          </button>
        </div>
        <div className="hint">
          Any SAE is accepted, and refused if its d_in does not match the
          model — a mismatched one would produce confident features describing
          a different network.
        </div>
        {err && <div className="hint err">{err}</div>}

        <LensPanel epoch={epoch} />
      </div>
    );
  }

  return (
    <div ref={scanRef} className="panel feat">
      <div className="sect">
        <span className="dot d-feat" />
        <h2 className="h-feat">FEATURES — THE CONCEPTS INSIDE</h2>
        <span className="rule" />
      </div>
      <div className="row" style={{ margin: "10px 0" }}>
        <span className="pill violet">
          {sae.repo?.split("/")[1]} · L{sae.layer} · {sae.d_sae?.toLocaleString()} features
        </span>
        <span className="meta">
          click a token → its top features · click a feature → heat + steering
        </span>
      </div>

      {summary && (
        <div className="attn-scroll">
          <div className="attn-inner">
            <div className="tokens">
              {summary.tokens.map((t, i) => {
                const h = heat?.[i] ?? 0;
                return (
                  <span
                    key={i}
                    ref={i === peak ? peakRef : undefined}
                    className={`tok ${tokenSel === i ? "feat-sel" : ""} ${i === peak ? "peak" : ""}`}
                    tabIndex={0}
                    role="button"
                    aria-pressed={tokenSel === i}
                    aria-label={`token ${i + 1} of ${summary.tokens.length}: ${t.trim() || "space"}${i === peak ? ", peak activation" : ""}`}
                    style={
                      heat
                        ? { backgroundColor: `rgba(160,140,255,${(0.42 * h).toFixed(3)})` }
                        : undefined
                    }
                    onClick={() => setTokenSel(i)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        setTokenSel(i);
                      }
                    }}
                  >
                    {t.replace(/ /g, "·") || "·"}
                  </span>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {summary && tokenSel >= 0 && (
        <div className="feat-list">
          <div className="meta" style={{ marginBottom: 2 }}>
            top features on {summary.tokens[tokenSel].replace(/ /g, "·")}
          </div>
          {(() => {
            const rows = summary.top[tokenSel] ?? [];
            const maxAct = rows[0]?.[1] || 1;
            // SIGNATURE — reveal ordered by magnitude, not by DOM order. The
            // strongest activation starts at t=0 and the rest follow by rank,
            // so the eye lands on the maximum before the others exist. Every
            // row is the same violet, so rank is the only channel left to say
            // which one matters; spending time instead of colour is free.
            const rank = new Map(
              rows
                .map((r, i) => [i, r[1]] as const)
                .sort((a, b) => b[1] - a[1])
                .map(([i], r) => [i, r]),
            );
            return rows.map(([fid, act], i) => (
              <div
                key={fid}
                className={`feat-row ${featSel === fid ? "sel" : ""}`}
                style={{ ["--i" as string]: rank.get(i) ?? 0 }}
                onClick={() => void onPickFeature(fid)}
              >
                <span className="feat-id">#{fid}</span>
                <div className="feat-bar" style={{ width: `${(160 * act) / maxAct}px` }} />
                <span className="feat-act">{act.toFixed(1)}</span>
              </div>
            ));
          })()}
        </div>
      )}

      {featSel >= 0 && (
        <div className="row" style={{ marginTop: 14 }}>
          <span className="meta">steer #{featSel}</span>
          <input
            type="range"
            min={-60}
            max={60}
            step={5}
            value={scale}
            onChange={(e) => setScale(Number(e.target.value))}
          />
          <span className="meta" style={{ minWidth: 34 }}>
            {scale > 0 ? `+${scale}` : scale}
          </span>
          <button
            className="violet"
            onClick={onSteerTest}
            disabled={busy !== "" || !prompt.trim()}
            title={
              prompt.trim()
                ? "Same prompt, greedy decoding, once clean and once steered"
                : "Generate in this tab first — the A/B needs the prompt"
            }
          >
            {busy === "steer" ? "Running A/B…" : "Run steering A/B"}
          </button>
        </div>
      )}

      {ab && (
        <div className="compare" style={{ marginTop: 14 }}>
          <div className="card">
            <span className="lbl">BASELINE</span>
            {ab.base}
          </div>
          <div className="card steered">
            <span className="lbl">FEATURE #{featSel} @ {scale > 0 ? `+${scale}` : scale}</span>
            {ab.steered}
          </div>
        </div>
      )}

      {err && <div className="hint">{err}</div>}
      <div className="hint">
        steering adds the feature's decoder direction to the residual stream during
        generation — deterministic (temp 0), fully reversible
      </div>
    </div>
  );
}
