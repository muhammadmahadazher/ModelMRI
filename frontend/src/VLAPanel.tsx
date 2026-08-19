import { useEffect, useRef, useState } from "react";
import { useScanOnData } from "./useScanOnData";
import RestingSketch from "./RestingSketch";
import {
  analyseVLA,
  errorText,
  getVLA,
  getVLAAttention,
  getVLADatasets,
  getVLAEpisodes,
  getVLAFrame,
  loadVLA,
  setVLADataset,
  VLADataset,
  VLADatasetInfo,
  VLAFrame,
  VLAStatus,
} from "./api";
import FrameCanvas from "./FrameCanvas";
import PolicyStrip from "./PolicyStrip";

/** Shown as the placeholder, and used when the box is left blank. */
const DEFAULT_POLICY = "lerobot/smolvla_base";

/** Robot-policy introspection: scrub a real episode, then see what the
 *  policy's vision tower attends to on that exact frame. */
/** Enough digits to distinguish two readings, whatever the units are.
 *
 *  `toFixed(0)` read fine on pusht, whose state is pixel coordinates in the
 *  hundreds, and destroyed every normalised dataset: joint angles in [-1, 1]
 *  all printed as "0" or "-0".
 *
 *  The precision is chosen once per VECTOR rather than per element. Deciding
 *  it element by element gave `[222, 97.0]` — two axes of the same
 *  measurement in two different formats, which reads as though they are
 *  different kinds of number. */
function vec(xs: number[]): string {
  const peak = Math.max(...xs.map(Math.abs), 0);
  const dp = peak >= 100 ? 0 : peak >= 1 ? 1 : 3;
  return xs.map((v) => v.toFixed(dp)).join(", ");
}

import VLACausal from "./VLACausal";
import VLAAudit from "./VLAAudit";
import VLAActions from "./VLAActions";

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
  // Every cached LeRobot dataset, not the one that happened to be configured.
  const [datasets, setDatasets] = useState<VLADatasetInfo[]>([]);
  const [chosen, setChosen] = useState("");
  // The policy checkpoint. Blank means the server's default; anything else
  // is loaded by discovering its vision tower rather than assuming SmolVLA's.
  const [policy, setPolicy] = useState("");
  // Which view. A dataset recorded with a wrist camera and an overhead one
  // has both in `cameras`; the reader used to show whichever came first and
  // never mention the others.
  const [camera, setCamera] = useState("");
  const debounce = useRef<number | undefined>(undefined);

  // Status only. Opening the dataset imports pyarrow and pyav and decodes
  // video — 396 MB and ~4.4s measured — so it waits until asked. Nothing on
  // this page reads a model or a dataset before you click.
  useEffect(() => {
    let live = true;
    void getVLA()
      .then((s) => live && setVla(s))
      .catch(() => undefined);
    // Names only — directory entries and one refs file each. It never opens a
    // parquet or decodes a frame, so this stays cheap enough to run on mount.
    void getVLADatasets()
      .then((d) => {
        if (!live) return;
        setDatasets(d.datasets);
        setChosen(d.current);
      })
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
      void getVLAFrame(episode, t, camera)
        .then((f) => live && setFrame(f))
        .catch((e) => live && setErr(errorText(e)));
    }, 90);
    return () => {
      live = false;
    };
  }, [ds, episode, t, camera]);

  useEffect(() => {
    if (!vla?.loaded || !heatKey) return;
    let live = true;
    void getVLAAttention(layer, -1)
      .then((h) => live && setHeat(h.heat))
      .catch((e) => live && setErr(errorText(e)));
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
      // Switch first when the pick differs, so "Open" always opens what the
      // dropdown says rather than whatever the server last had.
      if (chosen && chosen !== vla?.dataset_repo) await setVLADataset(chosen);
      const opened = await getVLAEpisodes(camera || undefined);
      setDs(opened);
      // The server answers with the camera it actually used, so the selector
      // shows a real view rather than an empty string.
      setCamera(opened.video_key ?? opened.cameras?.[0] ?? "");
      setLayer(0);
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }

  async function onLoad() {
    setBusy("load");
    setErr("");
    try {
      setVla(await loadVLA(policy));
    } catch (e) {
      setErr(errorText(e));
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
      setErr(errorText(e));
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
            <RestingSketch kind="vla" />
          <p>
            Watch what a real robot policy looks at, frame by frame, on recorded
            episodes. Nothing is loaded yet.
          </p>
          <div className="row">
            {datasets.length > 1 ? (
              <select
                className="combo"
                aria-label="dataset"
                value={chosen}
                onChange={(e) => setChosen(e.target.value)}
              >
                {datasets.map((d) => (
                  <option key={d.repo_id} value={d.repo_id} disabled={!d.usable}>
                    {d.repo_id} · {d.size_gb} GB{d.usable ? "" : " — incomplete"}
                  </option>
                ))}
              </select>
            ) : null}
            <button className="green" onClick={() => void onOpen()} disabled={busy !== ""}>
              {busy === "open" ? "Opening dataset…" : "Open dataset"}
            </button>
          </div>
          <span className="meta">
            {datasets.length > 1
              ? `${datasets.length} LeRobot datasets cached · nothing is downloaded`
              : datasets.length === 1
                ? `${datasets[0].repo_id} is the only one cached — any LeRobot v3.0 dataset works, and this list grows as you pull them`
                : "no LeRobot dataset cached — pull any LeRobot v3.0 dataset and it appears here"}
          </span>

          {/* The POLICY, not just the dataset.
              The dataset side was always dynamic; the policy was pinned to
              SmolVLA by three hardcoded values — the tensor prefix, the repo
              its vision config came from, and the module class. All three are
              read from the checkpoint now, so this box is the difference
              between "a SmolVLA viewer" and "a VLA viewer". */}
          <div className="row policy-row">
            <label className="meta" htmlFor="vla-policy">
              policy
            </label>
            <input
              id="vla-policy"
              className="share-note"
              placeholder={DEFAULT_POLICY}
              value={policy}
              onChange={(e) => setPolicy(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && void onOpen()}
              spellCheck={false}
            />
          </div>
          <span className="meta">
            any checkpoint whose weights carry a vision tower — the tensor
            prefix and the vision config are read from the file, not assumed.
            Blank uses <code>{DEFAULT_POLICY}</code>.
          </span>
        </div>
        {/* THE EXTRA ONLY FOR THE ERROR THE EXTRA FIXES. This appended
            `pip install modelmri[vla-lite]` to every failure, so a policy that
            is not installed, a file that will not parse and a permission
            error were all answered with "install the dataset readers" — advice
            that cannot help, and in the policy case advice the server
            explicitly warns against: its own hint says to run `modelmri policy
            install` BECAUSE installing lerobot beside ModelMRI breaks both.
            Every other refusal here already ends with its own next step, which
            is why there is nothing to add to it. */}
        {err && (
          <div className="hint">
            {/robot dataset|not cached|No such|FileNotFound/i.test(err) ? (
              <>
                no robot dataset cached · install the readers and pull one:{" "}
                <b>pip install modelmri[vla-lite]</b>
              </>
            ) : (
              err
            )}
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
            {busy === "load"
              ? "Loading policy…"
              : `Load ${(policy.trim() || vla?.policy_repo || DEFAULT_POLICY).split("/").pop()} vision tower`}
          </button>
        )}
        <span className="meta">{note}</span>
      </div>

      <PolicyStrip vla={vla} />

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
            {/* Every episode. This was `.slice(0, 60)`: lerobot/pusht has
                206, so 146 of them were unreachable and the control said
                nothing about it — the same shape of bug as reading only the
                first parquet shard. A <select> handles 206 options fine. */}
            {ds.episodes.map((e) => (
              <option key={e.index} value={e.index}>
                {e.index} · {e.length} frames
              </option>
            ))}
          </select>

          {(ds.cameras?.length ?? 0) > 1 && (
            <>
              <label className="meta" htmlFor="vla-camera">
                camera ({ds.cameras!.length} views)
              </label>
              <select
                id="vla-camera"
                className="combo"
                value={camera}
                onChange={(e) => setCamera(e.target.value)}
              >
                {ds.cameras!.map((c) => (
                  <option key={c} value={c}>
                    {c.replace(/^observation\.(images?\.)?/, "")}
                  </option>
                ))}
              </select>
            </>
          )}

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
              state [{vec(frame.state)}] · action [{vec(frame.action)}]
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

      {/* Directly under the attention map, on purpose: the causal map is
          meant to be read against it, and the number that matters most is how
          far apart the two rank the same blocks. */}
      <VLACausal
        episode={episode}
        timestep={t}
        layer={layer}
        ready={Boolean(vla?.loaded)}
      />

      {/* What the policy would DO, which needs the other half of a VLA — the
          action expert, in its own process. Sited under the causal map because
          it is the same question one step further out: the map says what the
          representation depended on, these say what came out of it. */}
      <VLAActions episode={episode} timestep={t} />

      {/* Last, and it needs neither half. It reads the files on disk and
          proves — or disproves — that the episodes above are what they claim
          to be. Nothing is downloaded and no GPU is touched, which is why it
          is the one control here that costs nothing to press. */}
      <VLAAudit />

      {err && <div className="hint">{err}</div>}
      <div className="hint">
        heat = attention each image patch receives inside{" "}
        {vla?.repo ?? "the policy"}'s own vision tower · deeper layers
        concentrate on task-relevant regions
      </div>
    </div>
  );
}
