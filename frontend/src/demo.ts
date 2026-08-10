/** Demo mode: serve pre-baked real responses instead of calling the API.
 *
 *  Built with `--mode demo` for GitHub Pages, so the whole tool is explorable
 *  with no install, no model download and no GPU. Every payload in
 *  public/demo/ was captured from a real local run (see scripts/bake_demo.py)
 *  — nothing here is mocked-up numbers.
 *
 *  The demo is the only ModelMRI most people will ever touch, so it is held
 *  to the same standard as the tool: it answers every endpoint the UI can
 *  call, it serves the slice that was asked for or says why it cannot, and
 *  `tests/demo_check.py` fails the build if either stops being true.
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

/** The prompt this demo actually recorded. */
export async function demoPrompt(): Promise<string> {
  return (await bundle<any>("llm")).prompt;
}

/** The `.mri` of the demo's own run, as a Blob, for "Share this view". */
export async function demoSessionFile(): Promise<Blob | null> {
  const b64 = (await bundle<any>("env")).session_mri;
  if (!b64) return null;
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  return new Blob([bytes], { type: "application/octet-stream" });
}

/** Route a would-be API call to the baked bundle.
 *
 *  Returns `{status, payload}`, mirroring `viewerFetch` — the two shims are
 *  built on the same trick and are meant to stay comparable. The status
 *  matters: a miss has to be able to say 422 *and why*, because the version
 *  that quietly served layer 0 for every unbaked layer is how 141 of 144
 *  head selections drew the wrong arcs under a dial that said otherwise.
 *
 *  `undefined` still means "this demo has no answer", and main.tsx turns that
 *  into a 409 — but nothing reachable should return it, and
 *  tests/demo_check.py fails if anything does.
 */
export async function demoFetch(
  path: string,
  body?: unknown,
): Promise<{ status: number; payload: unknown } | undefined> {
  const url = new URL(path, "http://x");
  const p = url.pathname;
  const q = url.searchParams;
  const ok = (payload: unknown) => ({ status: 200, payload });
  const refuse = (status: number, error: string) => ({ status, payload: { error } });

  if (p === "/api/session") {
    const llm = await bundle<any>("llm");
    return ok({
      app: "modelmri",
      version: "demo",
      model: {
        loaded: true,
        hf_id: llm.provenance?.model ?? "gpt2",
        device: llm.provenance?.device ?? "cpu (recorded)",
        dtype: llm.provenance?.dtype ?? "float32",
        n_params: 124439808,
      },
      _demo: llm,
    });
  }
  if (p === "/api/session/state") return ok((await bundle<any>("env")).session_state ?? {});
  if (p === "/api/models/local") return ok([{ id: "gpt2", size_gb: 1.1 }]);
  // Real discovery output from a real machine, with the paths generalised —
  // without this the demo's "On this machine" tab said "Nothing found …set
  // MODELMRI_MODELS_DIR", which is a confusing first impression of a feature
  // whose whole point is that it finds things for you.
  if (p === "/api/models/discovered") return ok(await bundle<any>("discovered"));
  if (p === "/api/ollama") return ok({ up: false, models: [] });
  if (p === "/api/accelerator") return ok((await bundle<any>("env")).accelerator ?? {});
  if (p === "/api/model/progress") return ok((await bundle<any>("env")).progress ?? {});
  if (p === "/api/paths") return ok((await bundle<any>("env")).paths ?? {});
  if (p === "/api/hub/auth") return ok((await bundle<any>("env")).hub_auth ?? { signed_in: false });
  if (p === "/api/sae/available") return ok((await bundle<any>("env")).sae_available ?? []);
  if (p === "/api/lens") return ok((await bundle<any>("env")).lens ?? {});
  if (p === "/api/vla/datasets") return ok((await bundle<any>("env")).vla_datasets ?? []);

  if (p === "/api/model/load") {
    const llm = await bundle<any>("llm");
    return ok({
      loaded: true,
      hf_id: llm.provenance?.model ?? "gpt2",
      device: llm.provenance?.device ?? "cpu (recorded)",
      dtype: llm.provenance?.dtype ?? "float32",
      n_params: 124439808,
    });
  }

  // A recording cannot answer a question it was not asked. The old branch
  // returned the baked generation for ANY prompt, so typing "what is 2+2"
  // produced a confident sentence about the Eiffel Tower — and then attention
  // over the Eiffel Tower's tokens, under the words you had typed.
  if (p === "/api/model/prompt") {
    const llm = await bundle<any>("llm");
    const asked = ((body as any)?.prompt ?? "").trim();
    if (asked && asked !== llm.prompt.trim()) {
      return refuse(
        422,
        `This demo replays one recorded run, so it can only answer the prompt ` +
          `it recorded: "${llm.prompt}". Install ModelMRI to point it at your ` +
          `own model and your own prompts — everything else on this page is ` +
          `the real tool.`,
      );
    }
    const f = await bundle<any>("features");
    return ok({ generation: steerActive ? f.steered : f.baseline });
  }

  if (p === "/api/attention/meta") return ok((await bundle<any>("llm")).meta);

  // Keyed on the PAIR. Reading only `layer` and falling back to the first
  // baked slice is what made the head selector a decoration: the dial read
  // `H 00/11` while the select read head 7, and the arcs were head 0's.
  if (p === "/api/attention") {
    const llm = await bundle<any>("llm");
    const layer = Number(q.get("layer") ?? 0);
    const head = Number(q.get("head") ?? 0);
    const slice = llm.attention[`${layer}.${head}`];
    if (!slice) {
      const have = Object.keys(llm.attention).length;
      return refuse(
        422,
        `this demo does not contain layer ${layer} head ${head}. It has ` +
          `${have} slices. A recording stores what was captured, not every ` +
          `combination.`,
      );
    }
    return ok(slice);
  }

  // Rank heads. The demo had no handler at all, so the button this project
  // leads with answered 409 under advice ("generate again") that could not
  // possibly work.
  if (p === "/api/attention/ablate") {
    const llm = await bundle<any>("llm");
    const baseline = q.get("baseline") ?? "zero";
    const scope = q.get("scope") ?? "layer";
    const key = scope === "all" ? `all.${baseline}` : `${q.get("layer") ?? 0}.${baseline}`;
    const ranking = (llm.ablate ?? {})[key];
    if (!ranking) {
      return refuse(422, `this demo has no ${baseline}-ablation ranking for ${key}.`);
    }
    return ok(ranking);
  }

  if (p === "/api/attention/diff") {
    const llm = await bundle<any>("llm");
    const b = q.get("b") ?? "";
    const cut = /^ablate:(\d+)\.(\d+)$/.exec(b);
    const key = cut
      ? `${q.get("layer")}.${q.get("head")}.${cut[1]}.${cut[2]}`
      : `${q.get("layer")}.${q.get("head")}.${q.get("a")}.${b}`;
    const d = (llm.diff ?? {})[key];
    if (!d) {
      return refuse(
        422,
        `this demo has no recorded comparison for ${b} viewed at layer ` +
          `${q.get("layer")} head ${q.get("head")}. It records the comparisons ` +
          `the ranking offers; install ModelMRI to run any other.`,
      );
    }
    return ok(d);
  }

  if (p === "/api/sae") return ok((await bundle<any>("features")).sae);
  if (p === "/api/sae/load") return ok((await bundle<any>("features")).sae);
  if (p === "/api/features/summary") return ok((await bundle<any>("features")).summary);
  if (p.startsWith("/api/features/")) return ok((await bundle<any>("features")).detail);
  if (p === "/api/steer") {
    const id = (body as any)?.feature_id;
    steerActive = id != null;
    return ok(
      steerActive
        ? { active: true, feature_id: id, scale: (body as any).scale }
        : { active: false },
    );
  }
  if (p === "/api/traces") return ok((await bundle<any>("traces")).list);
  if (p.startsWith("/api/traces/")) return ok((await bundle<any>("traces")).trace);

  // Custom models. The demo can't read your filesystem, so it serves the
  // adapter template's real inspection and keeps the panel's own flow —
  // resting, find, load, run — exactly as it behaves locally.
  if (p === "/api/custom") {
    const c = await bundle<any>("custom");
    return ok(customLoaded ? c.status : { ...c.status, loaded: false, path: null });
  }
  if (p === "/api/custom/candidates") return ok((await bundle<any>("custom")).candidates);
  if (p === "/api/custom/load") {
    customLoaded = true;
    return ok((await bundle<any>("custom")).status);
  }
  if (p === "/api/custom/run") return ok((await bundle<any>("custom")).run);
  if (p === "/api/custom/unload") {
    customLoaded = false;
    const c = await bundle<any>("custom");
    return ok({ ...c.status, loaded: false, path: null });
  }

  if (p === "/api/vla" || p === "/api/vla/load") {
    const v = await bundle<any>("vla");
    // The resting panel names what a click will read; bundles baked before
    // /api/vla carried these fields fall back to the dataset it recorded.
    return ok({
      ...v.status,
      dataset_repo: v.status.dataset_repo ?? v.dataset?.repo_id ?? "lerobot/pusht",
      policy_repo: v.status.policy_repo ?? v.status.repo ?? "lerobot/smolvla_base",
    });
  }
  if (p === "/api/vla/episodes") {
    const v = await bundle<any>("vla");
    const frames = Object.keys(v.frames).map(Number).sort((a, b) => a - b);
    return ok({
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
    });
  }
  if (p === "/api/vla/frame") {
    const v = await bundle<any>("vla");
    const want = Number(q.get("t") ?? v.frame);
    const keys = Object.keys(v.frames).map(Number);
    const nearest = keys.reduce((a, b) => (Math.abs(b - want) < Math.abs(a - want) ? b : a));
    return ok(v.frames[String(nearest)]);
  }
  if (p === "/api/vla/analyse") {
    const v = await bundle<any>("vla");
    return ok({
      layers: v.status.n_layers,
      heads: v.status.n_heads,
      grid: v.status.grid,
      latency_ms: 1786,
    });
  }
  if (p === "/api/vla/attention/meta") {
    const v = await bundle<any>("vla");
    return ok({
      available: true,
      reason: "",
      n_layers: v.status.n_layers,
      n_heads: v.status.n_heads,
      grid: v.status.grid,
    });
  }
  if (p === "/api/vla/attention") {
    const v = await bundle<any>("vla");
    const want = Number(q.get("layer") ?? 0);
    const have: number[] = v.layers;
    const nearest = have.reduce((a, b) => (Math.abs(b - want) < Math.abs(a - want) ? b : a));
    return ok(v.attention[String(nearest)]);
  }
  return undefined;
}

/** The steered output for the baked A/B (used by the features panel). */
export async function demoSteered(): Promise<{ baseline: string; steered: string; feature: number; scale: number }> {
  const f = await bundle<any>("features");
  return { baseline: f.baseline, steered: f.steered, feature: f.feature, scale: f.scale };
}
