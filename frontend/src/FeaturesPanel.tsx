import { useEffect, useRef, useState } from "react";
import { useScanOnData } from "./useScanOnData";
import {
  FeaturesSummary,
  getFeatureDetail,
  getFeaturesSummary,
  getSAE,
  loadSAE,
  promptOnce,
  SAEStatus,
  setSteer,
} from "./api";

interface Props {
  epoch: number; // bumps after each generation
  prompt: string; // the prompt of the last generation, for steering A/B
}

/** SAE feature browser: token -> top features -> heat view -> steering A/B. */
export default function FeaturesPanel({ epoch, prompt }: Props) {
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
      setErr(String(e));
    }
  }

  async function onLoadSAE() {
    setBusy("sae");
    setErr("");
    try {
      setSae(await loadSAE());
    } catch (e) {
      setErr(String(e));
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
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  async function onSteerTest() {
    if (featSel < 0) return;
    setBusy("steer");
    setErr("");
    try {
      await setSteer(null);
      const base = (await promptOnce(prompt, 24, 0, false)).generation;
      await setSteer(featSel, scale);
      const steered = (await promptOnce(prompt, 24, 0, false)).generation;
      await setSteer(null); // always leave the model clean
      setAb({ base, steered });
    } catch (e) {
      setErr(String(e));
      await setSteer(null);
    } finally {
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
        <div className="row" style={{ marginTop: 12 }}>
          <button className="violet" onClick={onLoadSAE} disabled={busy !== ""}>
            {busy === "sae"
              ? "Loading SAE… (first run downloads ~150 MB)"
              : "Load SAE (GPT-2 · layer 8 · 24,576 features)"}
          </button>
          <span className="meta">works with the gpt2 model</span>
        </div>
        {err && <div className="hint">{err}</div>}
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
          <button className="violet" onClick={onSteerTest} disabled={busy !== ""}>
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
