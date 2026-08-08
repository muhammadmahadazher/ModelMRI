import { useEffect, useRef, useState } from "react";
import { useScanOnData } from "./useScanOnData";
import {
  analyseVLA,
  getVLA,
  getVLAAttention,
  getVLAEpisodes,
  getVLAFrame,
  loadVLA,
  VLADataset,
  VLAFrame,
  VLAStatus,
} from "./api";
import FrameCanvas from "./FrameCanvas";

/** Robot-policy introspection: scrub a real episode, then see what the
 *  policy's vision tower attends to on that exact frame. */
export default function VLAPanel() {
  const [vla, setVla] = useState<VLAStatus | null>(null);
  const [ds, setDs] = useState<VLADataset | null>(null);
  const [episode, setEpisode] = useState(0);
  const [t, setT] = useState(0);
  const [frame, setFrame] = useState<VLAFrame | null>(null);
  const [heat, setHeat] = useState<number[][] | null>(null);
  const [heatKey, setHeatKey] = useState<string>("");
  const scanRef = useScanOnData(heatKey);
  const [layer, setLayer] = useState(0);
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");
  const [note, setNote] = useState("");
  const debounce = useRef<number | undefined>(undefined);

  // Status only. Opening the dataset imports pyarrow and pyav and decodes
  // video — 396 MB and ~4.4s measured — so it waits until asked. Nothing on
  // this page reads a model or a dataset before you click.
  useEffect(() => {
    let live = true;
    void getVLA()
      .then((s) => live && setVla(s))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, []);

  // scrubbing: debounce the frame fetch (server-side decode is ~50ms)
  useEffect(() => {
    if (!ds) return;
    let live = true;
    window.clearTimeout(debounce.current);
    debounce.current = window.setTimeout(() => {
      void getVLAFrame(episode, t)
        .then((f) => live && setFrame(f))
        .catch((e) => live && setErr(String(e)));
    }, 90);
    return () => {
      live = false;
    };
  }, [ds, episode, t]);

  useEffect(() => {
    if (!vla?.loaded || !heatKey) return;
    let live = true;
    void getVLAAttention(layer, -1)
      .then((h) => live && setHeat(h.heat))
      .catch((e) => live && setErr(String(e)));
    return () => {
      live = false;
    };
  }, [layer, heatKey, vla?.loaded]);

  const currentKey = `${episode}:${t}`;
  const stale = heatKey !== "" && heatKey !== currentKey;

  async function onOpen() {
    setBusy("open");
    setErr("");
    try {
      setDs(await getVLAEpisodes());
      setLayer(0);
    } catch (e) {
      setErr(String(e instanceof Error ? e.message : e));
    } finally {
      setBusy("");
    }
  }

  async function onLoad() {
    setBusy("load");
    setErr("");
    try {
      setVla(await loadVLA());
    } catch (e) {
      setErr(String(e instanceof Error ? e.message : e));
    } finally {
      setBusy("");
    }
  }

  async function onAnalyse() {
    setBusy("run");
    setErr("");
    try {
      const r = await analyseVLA(episode, t);
      setNote(`${r.layers} layers · ${r.heads} heads · ${r.latency_ms} ms`);
      setHeatKey(currentKey);
      const h = await getVLAAttention(layer, -1);
      setHeat(h.heat);
    } catch (e) {
      setErr(String(e instanceof Error ? e.message : e));
    } finally {
      setBusy("");
    }
  }

  if (!ds) {
    return (
      <div className="panel">
        <div className="sect">
          <span className="dot d-vla" />
          <h2 className="h-vla">ROBOT POLICY — VLA</h2>
          <span className="rule" />
        </div>
        <div className="resting">
          <p>
            Watch what a real robot policy looks at, frame by frame, on recorded
            episodes. Nothing is loaded yet.
          </p>
          <button className="green" onClick={() => void onOpen()} disabled={busy !== ""}>
            {busy === "open" ? "Opening dataset…" : "Open dataset"}
          </button>
          <span className="meta">
            reads {vla?.dataset_repo ?? "a cached LeRobot dataset"} from your
            HuggingFace cache · nothing is downloaded
          </span>
        </div>
        {err && (
          <div className="hint">
            {/robot dataset|not cached|No such|FileNotFound/i.test(err)
              ? "no robot dataset cached · install the extra and pull one: "
              : `${err} · `}
            <b>pip install modelmri[vla-lite]</b>
          </div>
        )}
      </div>
    );
  }

  const ep = ds.episodes.find((e) => e.index === episode) ?? ds.episodes[0];

  return (
    <div ref={scanRef} className="panel vla">
      <div className="sect">
        <span className="dot d-vla" />
        <h2 className="h-vla">ROBOT POLICY — WHAT IT LOOKS AT</h2>
        <span className="rule" />
      </div>

      <div className="row" style={{ marginBottom: 10 }}>
        <span className="pill vla">
          {ds.repo_id} · {ds.n_episodes} episodes · {ds.fps} fps
        </span>
        {vla?.loaded ? (
          <span className="pill vla">
            {vla.repo} · {vla.n_layers}L × {vla.n_heads}H · {vla.grid.join("×")} patches
          </span>
        ) : (
          <button className="green" onClick={() => void onLoad()} disabled={busy !== ""}>
            {busy === "load" ? "Loading policy…" : "Load SmolVLA vision tower"}
          </button>
        )}
        <span className="meta">{note}</span>
      </div>

      <div className="vla-grid">
        <div>
          <FrameCanvas src={frame?.image ?? ""} heat={stale ? null : heat} scale={4} />
          <div className="meta" style={{ marginTop: 8 }}>
            episode {episode} · frame {t}/{ep.length - 1} · t={frame?.timestamp ?? 0}s
            {stale && <span className="pill" style={{ marginLeft: 8 }}>heatmap is from another frame</span>}
          </div>
        </div>

        <div className="vla-side">
          <div className="meta">{ep.task}</div>
          <label className="meta">episode</label>
          <select
            className="combo"
            value={episode}
            onChange={(e) => {
              setEpisode(Number(e.target.value));
              setT(0);
            }}
          >
            {ds.episodes.slice(0, 60).map((e) => (
              <option key={e.index} value={e.index}>
                {e.index} · {e.length} frames
              </option>
            ))}
          </select>

          <label className="meta">frame {t}</label>
          <input
            type="range"
            min={0}
            max={Math.max(ep.length - 1, 0)}
            value={t}
            onChange={(e) => setT(Number(e.target.value))}
          />

          {frame && (
            <div className="meta">
              state [{frame.state.map((v) => v.toFixed(0)).join(", ")}] · action [
              {frame.action.map((v) => v.toFixed(0)).join(", ")}]
            </div>
          )}

          {vla?.loaded && (
            <>
              <button className="green" onClick={() => void onAnalyse()} disabled={busy !== ""}>
                {busy === "run" ? "Running policy…" : "Run policy on this frame"}
              </button>
              {heatKey && (
                <>
                  <label className="meta">vision layer {layer}</label>
                  <input
                    type="range"
                    min={0}
                    max={vla.n_layers - 1}
                    value={layer}
                    onChange={(e) => setLayer(Number(e.target.value))}
                  />
                </>
              )}
            </>
          )}
        </div>
      </div>

      {err && <div className="hint">{err}</div>}
      <div className="hint">
        heat = attention each image patch receives inside SmolVLA's own vision
        tower · deeper layers concentrate on task-relevant regions
      </div>
    </div>
  );
}
