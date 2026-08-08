/** Demo mode: serve pre-baked real responses instead of calling the API.
 *
 *  Built with VITE_DEMO=1 for GitHub Pages, so the whole tool is explorable
 *  with no install, no model download and no GPU. Every payload in
 *  public/demo/ was captured from a real local run (see scripts/bake_demo.py)
 *  — nothing here is mocked-up numbers.
 */

export const DEMO = import.meta.env.VITE_DEMO === "1";

const cache = new Map<string, unknown>();

/** Mirrors the server's steering state so /api/model/prompt can answer with
 *  the baked steered output — without this the demo's A/B silently returns
 *  the baseline twice. */
let steerActive = false;

/** Same idea for the custom panel: without it, /api/custom would answer
 *  "loaded" on first paint and the resting state would never be seen. */
let customLoaded = false;

async function bundle<T>(name: string): Promise<T> {
  if (!cache.has(name)) {
    const res = await fetch(`${import.meta.env.BASE_URL}demo/${name}.json`);
    if (!res.ok) throw new Error(`demo bundle ${name} missing`);
    cache.set(name, await res.json());
  }
  return cache.get(name) as T;
}

/** Route a would-be API call to the baked bundle. Returns undefined when the
 *  demo has nothing for this endpoint, so callers can fall back to a message. */
export async function demoFetch(path: string, body?: unknown): Promise<unknown> {
  const url = new URL(path, "http://x");
  const p = url.pathname;
  const q = url.searchParams;

  if (p === "/api/session") {
    const llm = await bundle<any>("llm");
    return {
      app: "modelmri",
      version: "demo",
      model: {
        loaded: true,
        hf_id: "gpt2",
        device: "cpu (recorded)",
        dtype: "float32",
        n_params: 124439808,
      },
      _demo: llm,
    };
  }
  if (p === "/api/models/local") return [{ id: "gpt2", size_gb: 1.1 }];
  if (p === "/api/ollama") return { up: false, models: [] };
  if (p === "/api/model/load") {
    return { loaded: true, hf_id: "gpt2", device: "cpu (recorded)", dtype: "float32", n_params: 124439808 };
  }
  if (p === "/api/model/prompt") {
    const f = await bundle<any>("features");
    return { generation: steerActive ? f.steered : f.baseline };
  }
  if (p === "/api/attention/meta") return (await bundle<any>("llm")).meta;
  if (p === "/api/attention") {
    const llm = await bundle<any>("llm");
    const layer = q.get("layer") ?? "0";
    return llm.attention[layer] ?? llm.attention[String(llm.layers[0])];
  }
  if (p === "/api/sae") return (await bundle<any>("features")).sae;
  if (p === "/api/sae/load") return (await bundle<any>("features")).sae;
  if (p === "/api/features/summary") return (await bundle<any>("features")).summary;
  if (p.startsWith("/api/features/")) return (await bundle<any>("features")).detail;
  if (p === "/api/steer") {
    const id = (body as any)?.feature_id;
    steerActive = id != null;
    return steerActive
      ? { active: true, feature_id: id, scale: (body as any).scale }
      : { active: false };
  }
  if (p === "/api/traces") return (await bundle<any>("traces")).list;
  if (p.startsWith("/api/traces/")) return (await bundle<any>("traces")).trace;

  // Custom models. The demo can't read your filesystem, so it serves the
  // adapter template's real inspection and keeps the panel's own flow —
  // resting, find, load, run — exactly as it behaves locally.
  if (p === "/api/custom") {
    const c = await bundle<any>("custom");
    return customLoaded ? c.status : { ...c.status, loaded: false, path: null };
  }
  if (p === "/api/custom/candidates") return (await bundle<any>("custom")).candidates;
  if (p === "/api/custom/load") {
    customLoaded = true;
    return (await bundle<any>("custom")).status;
  }
  if (p === "/api/custom/run") return (await bundle<any>("custom")).run;
  if (p === "/api/custom/unload") {
    customLoaded = false;
    const c = await bundle<any>("custom");
    return { ...c.status, loaded: false, path: null };
  }

  if (p === "/api/vla" || p === "/api/vla/load") {
    const v = await bundle<any>("vla");
    // The resting panel names what a click will read; bundles baked before
    // /api/vla carried these fields fall back to the dataset it recorded.
    return {
      ...v.status,
      dataset_repo: v.status.dataset_repo ?? v.dataset?.repo_id ?? "lerobot/pusht",
      policy_repo: v.status.policy_repo ?? v.status.repo ?? "lerobot/smolvla_base",
    };
  }
  if (p === "/api/vla/episodes") {
    const v = await bundle<any>("vla");
    const frames = Object.keys(v.frames).map(Number).sort((a, b) => a - b);
    return {
      ...v.dataset,
      n_episodes: 1,
      episodes: [
        {
          index: v.episode,
          length: frames[frames.length - 1] + 1,
          task: v.frames[String(v.frame)].task,
          from_ts: 0,
          to_ts: 0,
        },
      ],
    };
  }
  if (p === "/api/vla/frame") {
    const v = await bundle<any>("vla");
    const want = Number(q.get("t") ?? v.frame);
    const keys = Object.keys(v.frames).map(Number);
    const nearest = keys.reduce((a, b) => (Math.abs(b - want) < Math.abs(a - want) ? b : a));
    return v.frames[String(nearest)];
  }
  if (p === "/api/vla/analyse") {
    const v = await bundle<any>("vla");
    return { layers: v.status.n_layers, heads: v.status.n_heads, grid: v.status.grid, latency_ms: 1786 };
  }
  if (p === "/api/vla/attention/meta") {
    const v = await bundle<any>("vla");
    return { available: true, reason: "", n_layers: v.status.n_layers, n_heads: v.status.n_heads, grid: v.status.grid };
  }
  if (p === "/api/vla/attention") {
    const v = await bundle<any>("vla");
    const want = Number(q.get("layer") ?? 0);
    const have: number[] = v.layers;
    const nearest = have.reduce((a, b) => (Math.abs(b - want) < Math.abs(a - want) ? b : a));
    return v.attention[String(nearest)];
  }
  return undefined;
}

/** The steered output for the baked A/B (used by the features panel). */
export async function demoSteered(): Promise<{ baseline: string; steered: string; feature: number; scale: number }> {
  const f = await bundle<any>("features");
  return { baseline: f.baseline, steered: f.steered, feature: f.feature, scale: f.scale };
}
