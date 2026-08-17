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

import { dequantise } from "./viewer";

export const DEMO = import.meta.env.VITE_DEMO === "1";

const cache = new Map<string, unknown>();

/** Which recorded model is on screen.
 *
 *  The picker offers every model it discovers, so with one recording you
 *  could select Qwen3-0.6B and the demo would keep replaying gpt2 underneath
 *  — the page attributing one model's sentence to another, which is how a
 *  visitor concludes Qwen3 thinks the Eiffel Tower is the tallest building in
 *  the world. Either a scenario exists for what you picked, or the load is
 *  refused by name.
 */
let active: string | null = null;

interface Scenario {
  id: string;
  slug: string;
  n_layers: number;
  n_heads: number;
  n_tokens: number;
  generation: string;
  n_params: number | null;
  device: string | null;
  dtype: string | null;
  /** Base model or instruction-tuned. Drives the caveat under the output. */
  instruct?: boolean;
}

async function index(): Promise<{ default: string; scenarios: Scenario[] }> {
  return await bundle("scenarios");
}

async function current(): Promise<Scenario> {
  const idx = await index();
  const want = active ?? idx.default;
  return idx.scenarios.find((s) => s.id === want) ?? idx.scenarios[0];
}

/** The recorded run for whichever model is selected. */
async function llm(): Promise<any> {
  return await bundle(`llm-${(await current()).slug}`);
}

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
  return (await llm()).prompt;
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
    const s = await current();
    return ok({
      app: "modelmri",
      version: "demo",
      model: {
        loaded: true,
        hf_id: s.id,
        device: s.device ?? "recorded",
        dtype: s.dtype ?? "bfloat16",
        n_params: s.n_params,
        instruct: s.instruct ?? null,
      },
      _demo: { ...(await llm()), scenarios: (await index()).scenarios.map((x) => x.id) },
    });
  }
  if (p === "/api/session/state") return ok((await bundle<any>("env")).session_state ?? {});
  if (p === "/api/models/local") return ok([{ id: "gpt2", size_gb: 1.1 }]);
  // Real discovery output from a real machine, with the paths generalised —
  // without this the demo's "On this machine" tab said "Nothing found …set
  // MODELMRI_MODELS_DIR", which is a confusing first impression of a feature
  // whose whole point is that it finds things for you.
  if (p === "/api/models/discovered") return ok(await bundle<any>("discovered"));
  // Live third-party lookups. A static page genuinely cannot do these — but
  // "not available in the demo" in red reads as a broken tool, so each says
  // what it is and what would make it work. The HuggingFace and Ollama tabs
  // behave identically here, because they behave identically in the product.
  if (p === "/api/hub/models") {
    return refuse(
      501,
      `Searching HuggingFace is a live call to huggingface.co, and this page ` +
        `is a static recording with no backend. Installed, this box searches ` +
        `the real Hub and filters to what fits your GPU. The "On this machine" ` +
        `tab beside it works here, because that data was recorded.`,
    );
  }
  if (p === "/api/ollama/resolve" || p === "/api/ollama/size") {
    return refuse(
      501,
      `This is a live lookup against the Ollama registry, which a static page ` +
        `cannot make. Installed, ModelMRI reads your running Ollama and lists ` +
        `what you have pulled.`,
    );
  }
  if (p === "/api/ollama/pull") {
    return refuse(
      501,
      `Pulling a model downloads gigabytes to an Ollama daemon, and there is ` +
        `no daemon behind this page. Installed, this streams the pull with a ` +
        `size guard in front of it.`,
    );
  }
  if (p === "/api/hub/signin" || p === "/api/hub/signout") {
    return refuse(
      501,
      `Signing in writes a token to your own config directory, which a web ` +
        `page has no business doing and this one cannot. Installed, ` +
        `ModelMRI reads the token you already gave \`huggingface-cli\`, or ` +
        `you can paste one — it never leaves your machine.`,
    );
  }

  // Ollama is genuinely not running behind a static page. Say which, rather
  // than showing an "off" pill with no explanation.
  if (p === "/api/ollama") {
    return ok({
      up: false,
      models: [],
      reason:
        "No Ollama daemon behind this page — it is a static recording. " +
        "Installed, this tab lists the models you have pulled.",
    });
  }
  if (p === "/api/accelerator") return ok((await bundle<any>("env")).accelerator ?? {});
  if (p === "/api/model/progress") return ok((await bundle<any>("env")).progress ?? {});
  // A pull has its own progress slot, separate from a model load. Nothing can
  // be downloading on a static page, so the honest answer is the idle
  // snapshot rather than a refusal -- the picker polls this on open and a 501
  // would paint an error over a panel that is working correctly.
  if (p === "/api/pull/progress") {
    return ok({
      active: false,
      hf_id: null,
      stage: "",
      detail: "",
      bytes_done: 0,
      bytes_total: 0,
      elapsed_s: 0,
      eta_s: null,
      error: null,
    });
  }
  // Unloading frees a model this page never held. Answering with the recorded
  // status keeps the button honest: it reports nothing was freed, which is
  // true, instead of claiming to have released memory that was never taken.
  if (p === "/api/model/unload") {
    return ok({
      unloaded: false,
      was: null,
      freed_bytes: 0,
      accelerator_bytes_in_use: 0,
      status: (await bundle<any>("env")).model ?? {},
    });
  }
  // Scanning a folder means reading the reader's disk, which a page served
  // from GitHub Pages cannot do and should not pretend to.
  if (p === "/api/custom/scan") {
    return refuse(
      501,
      `Scanning a folder reads your disk, and this page is a static recording ` +
        `served from the web with no access to it. Installed, this box takes ` +
        `any folder on your machine and lists the models in it -- including ` +
        `the ones that will not load, and why.`,
    );
  }
  if (p === "/api/paths") return ok((await bundle<any>("env")).paths ?? {});
  if (p === "/api/hub/auth") return ok((await bundle<any>("env")).hub_auth ?? { signed_in: false });
  if (p === "/api/sae/available") return ok((await bundle<any>("env")).sae_available ?? []);
  if (p === "/api/lens") return ok((await bundle<any>("env")).lens ?? {});
  if (p === "/api/vla/datasets") return ok((await bundle<any>("env")).vla_datasets ?? []);
  // Sizes are real registry facts; `fits` is not, because a static page has
  // no idea what GPU is reading it. Null means unknown, which is a different
  // answer from "too big" and is rendered as such.
  if (p === "/api/ollama/suggested") {
    const rows = (await bundle<any>("env")).ollama_suggested ?? [];
    return ok(rows.map((r: any) => ({ ...r, fits: null })));
  }

  // Switch scenarios if one was recorded for this model, and refuse by name
  // if not. Answering "loaded" for a model the demo cannot replay is what put
  // Qwen3-0.6B in the picker above gpt2's output.
  if (p === "/api/model/load") {
    const idx = await index();
    const want = (body as any)?.hf_id;
    const match = idx.scenarios.find((s) => s.id === want);
    if (!match) {
      return refuse(
        422,
        `this demo has recordings for ${idx.scenarios.map((s) => s.id).join(" and ")}. ` +
          `${want} is not one of them, and replaying another model's run under ` +
          `its name would be a lie about which model said what. ` +
          `Install ModelMRI to load it for real.`,
      );
    }
    active = match.id;
    // The real runtime clears steering on every load, because a hook left
    // installed silently steers the next generation of a different model.
    steerActive = false;
    return ok({
      loaded: true,
      hf_id: match.id,
      device: match.device ?? "recorded",
      dtype: match.dtype ?? "bfloat16",
      n_params: match.n_params,
      // Without this the base-model caveat never fires in the demo, and the
      // temperature half of it fires alone — asserting sampling on a replay
      // that is deterministic by construction.
      instruct: match.instruct ?? null,
    });
  }

  // A recording cannot answer a question it was not asked. The old branch
  // returned the baked generation for ANY prompt, so typing "what is 2+2"
  // produced a confident sentence about the Eiffel Tower — and then attention
  // over the Eiffel Tower's tokens, under the words you had typed.
  if (p === "/api/model/prompt") {
    const run = await llm();
    const asked = ((body as any)?.prompt ?? "").trim();
    if (asked && asked !== run.prompt.trim()) {
      return refuse(
        422,
        `This demo replays recorded runs, so it can only answer the prompt it ` +
          `recorded: "${run.prompt}". Install ModelMRI to point it at your ` +
          `own model and your own prompts — everything else on this page is ` +
          `the real tool.`,
      );
    }
    // Steering was only ever recorded against gpt2's SAE, so it is the only
    // scenario whose A/B has a steered side to show.
    if (steerActive) {
      const f = await bundle<any>("features");
      return ok({ generation: f.steered });
    }
    return ok({ generation: run.generation });
  }

  if (p === "/api/attention/meta") return ok((await llm()).meta);

  // Keyed on the PAIR. Reading only `layer` and falling back to the first
  // baked slice is what made the head selector a decoration: the dial read
  // `H 00/11` while the select read head 7, and the arcs were head 0's.
  if (p === "/api/attention") {
    const run = await llm();
    const layer = Number(q.get("layer") ?? 0);
    const head = Number(q.get("head") ?? 0);
    const slice = run.attention[`${layer}.${head}`];
    if (!slice) {
      const have = Object.keys(run.attention).length;
      return refuse(
        422,
        `this demo does not contain layer ${layer} head ${head}. It has ` +
          `${have} slices. A recording stores what was captured, not every ` +
          `combination.`,
      );
    }
    const tokens: string[] = run.tokens;
    return ok({
      layer,
      head,
      tokens,
      // Without this the demo's panel would rest on nothing and show the
      // blank canvas this replaced — the demo is held to the tool's standard.
      n_prompt: run.n_prompt ?? 0,
      // Decoded with the viewer's function, against the same uint8 encoding
      // the .mri format uses — one decoder, so the two surfaces cannot
      // disagree about what a byte meant.
      matrix: dequantise(slice.q, slice.scale, tokens.length),
    });
  }

  // Rank heads. The demo had no handler at all, so the button this project
  // leads with answered 409 under advice ("generate again") that could not
  // possibly work.
  if (p === "/api/attention/ablate") {
    const run = await llm();
    const baseline = q.get("baseline") ?? "zero";
    const scope = q.get("scope") ?? "layer";
    const key = scope === "all" ? `all.${baseline}` : `${q.get("layer") ?? 0}.${baseline}`;
    const ranking = (run.ablate ?? {})[key];
    if (!ranking) {
      return refuse(422, `this demo has no ${baseline}-ablation ranking for ${key}.`);
    }
    return ok(ranking);
  }

  if (p === "/api/attention/diff") {
    const run = await llm();
    const b = q.get("b") ?? "";
    const cut = /^ablate:(\d+)\.(\d+)$/.exec(b);
    const key = cut
      ? `${q.get("layer")}.${q.get("head")}.${cut[1]}.${cut[2]}`
      : `${q.get("layer")}.${q.get("head")}.${q.get("a")}.${b}`;
    const d = (run.diff ?? {})[key];
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

  // Features, SAE and steering belong to ONE scenario.
  //
  // features.json is gpt2's: a 768-dim GPT-2 SAE, gpt2's 23-token sentence,
  // and a steered completion of it. Served unconditionally, selecting
  // Qwen3-0.6B left the page reporting a GPT-2 SAE loaded against a 1024-dim
  // model, a feature strip showing gpt2's words under Qwen3's output, and an
  // A/B pairing Qwen3's baseline against gpt2's steered text. That is the
  // failure this module's own docstring says was eliminated — one model's
  // sentence attributed to another.
  //
  // The real server cannot do this: `load()` clears `sae`, `_feats` and
  // `_steer` on every model change, so the panel falls to "no SAE exists for
  // this model". These branches now behave the same way.
  const saeScenario = async () => (await index()).default;
  const hasSAE = async () => (await current()).id === (await saeScenario());

  if (p === "/api/sae") {
    if (!(await hasSAE())) {
      return ok({ loaded: false, repo: null, hook: null, layer: null, d_in: null });
    }
    return ok((await bundle<any>("features")).sae);
  }
  if (p === "/api/sae/load") {
    if (!(await hasSAE())) {
      const s = await current();
      return refuse(
        422,
        `No public sparse autoencoder exists for ${s.id}. This demo recorded ` +
          `one for ${await saeScenario()}; installed, ModelMRI checks the SAE's ` +
          `d_in against the model's hidden size and refuses a mismatch rather ` +
          `than showing you another model's features.`,
      );
    }
    return ok((await bundle<any>("features")).sae);
  }
  if (p === "/api/features/summary" || p.startsWith("/api/features/")) {
    if (!(await hasSAE())) {
      return refuse(409, "Load an SAE first.");
    }
    const f = await bundle<any>("features");
    return ok(p === "/api/features/summary" ? f.summary : f.detail);
  }
  if (p === "/api/steer") {
    const id = (body as any)?.feature_id;
    if (id != null && !(await hasSAE())) {
      return refuse(409, "Load an SAE first.");
    }
    steerActive = id != null;
    return ok(
      steerActive
        ? { active: true, feature_id: id, scale: (body as any).scale }
        : { active: false },
    );
  }
  // Patching re-runs the model hundreds of times with an activation replaced.
  // There is no model here to re-run, and unlike the other panels there is no
  // honest recording of it either: the answer depends on the two prompts the
  // reader types, so a baked grid would be a fabricated measurement wearing
  // the reader's own words. It refuses, and says why.
  if (p === "/api/patch") {
    return refuse(
      409,
      "Patching moves an activation from one live run into another, so it " +
        "needs a model on your machine to re-run — this page has recordings, " +
        "not weights. `pip install modelmri` and the panel works against any " +
        "model you load.",
    );
  }
  // The edge-level follow-up to a grid that is not here. Reached only by
  // clicking a flagged cell, and there are no flagged cells to click, so this
  // says the same thing its parent says rather than inventing a sender list.
  if (p === "/api/patch/path") {
    return refuse(
      409,
      "Asking what wrote into one site patches every earlier head and MLP " +
        "into it one at a time, against the same live model the grid above " +
        "needs. This page has recordings, not weights.",
    );
  }

  // Six recordings that ARE here. Each is a GET with no argument the reader
  // chooses, which is what makes it bakeable: the answer is a property of this
  // recording rather than of something typed on the day.
  //
  // Written as six literal `p === "..."` lines on purpose. `demo_check.py`
  // finds handlers by exactly that pattern, and its docstring records why it
  // refuses to be cleverer — an earlier version treated every handler as a
  // prefix and reported `/api/sae/available` as covered by `/api/sae`, so the
  // check under-reported the very gaps it exists to find. A lookup table would
  // be invisible to it, which is the same failure wearing nicer clothes.
  const recorded = async (key: string) => {
    const extra = (await llm()).extra ?? {};
    // A key the bundle lacks refuses BY NAME rather than returning undefined:
    // a panel handed `undefined` renders blank, and blank reads as a
    // measurement that found nothing.
    if (!(key in extra)) {
      return refuse(
        409,
        "This recording does not carry that measurement — it was baked " +
          "before the endpoint existed, or the bake could not reach it. " +
          "Either way the honest answer is that nothing is recorded here, " +
          "rather than an empty panel that reads as a measurement of nothing.",
      );
    }
    return ok(extra[key]);
  };
  // The robot half of the same line the LLM panels already draw. What is
  // baked is CORRELATIONAL — the frames the policy saw and where its attention
  // went — and those replay honestly. Occlusion and the sweep are CAUSAL: they
  // black out a patch of the frame and run the policy again to see what the
  // action does. That needs the policy loaded, which is what a static page
  // does not have, and a baked occlusion map would be the one thing this
  // project will not ship: a fabricated causal claim rendered beside real
  // recordings, indistinguishable from them.
  if (p === "/api/vla/occlude" || p === "/api/vla/sweep") {
    return refuse(
      409,
      "Occluding the frame re-runs the policy once per patch to see what the " +
        "action does without it — a causal measurement, against a policy this " +
        "page does not carry. The frames and attention above are recordings " +
        "and replay honestly; this one cannot. `pip install modelmri` to run " +
        "it on a policy of your own.",
    );
  }
  if (p === "/api/vla/occlude/cost" || p === "/api/vla/sweep/cost") {
    return refuse(
      409,
      "This projects the cost of a run this page cannot make, so the number " +
        "would describe a wait nobody here is going to have.",
    );
  }
  if (p === "/api/vla/share") {
    return refuse(
      409,
      "Exporting the robot analysis as a `.mri` writes a file from the live " +
        "runtime's own state. Use the share control on a local install, where " +
        "there is a run behind the file.",
    );
  }
  if (p === "/api/attention/types") return recorded("types");
  if (p === "/api/attention/direct") return recorded("direct");
  if (p === "/api/attention/ablate/estimate") return recorded("ablate_estimate");
  if (p === "/api/telemetry") return recorded("telemetry");
  if (p === "/api/lens/tuned") return recorded("lens_tuned");

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
  if (p === "/api/policy") {
    // The ACTION half, answered from here rather than fetched. A recording has
    // no process that could hold a policy, and "not installed" is the true
    // state of this page — so this is the honest answer rather than a stub.
    //
    // It lives in the shim rather than in `api.ts` so the coverage check in
    // `demo_check.py` sees it. That check exists because an endpoint the demo
    // cannot answer is a panel that 404s on Pages, and it caught exactly that
    // here: the first version of this branch was a `DEMO || VIEWER` ternary in
    // `api.ts`, invisible to the check and reported as "unhandled".
    return ok({
      running: false,
      contract: null,
      policy_repo: "",
      revision: "",
      device: "",
      dtype: "",
      normalisation: {},
      port: 0,
      reachable: false,
      reason:
        "This page is a recording, so there is no process here that could " +
        "hold a policy. Install ModelMRI (`pip install modelmri`) and run " +
        "`modelmri policy install` to ask a real one what it would do.",
      means:
        "No policy sidecar is running, so nothing here can say what the " +
        "robot would DO — only where it looked.",
      family: "",
      cameras: [],
      camera_shapes: {},
      state_width: null,
      action_width: null,
      chunk_size: null,
      samples: false,
      lerobot_version: "",
      torch_version: "",
      accelerated: null,
      installed: false,
      venv: "",
      contract_here: 1,
      install_hint:
        "Run `modelmri policy install` — it builds a separate virtual " +
        "environment for the policy and its pinned lerobot, because " +
        "installing lerobot beside ModelMRI breaks both.",
      venv_disk_bytes: 6_000_000_000,
      assumed_policy_bytes: 3_500_000_000,
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
  // The robot panel's scrubber and layer dial are the same class of control
  // as the attention panel's, and were serving the same silent lie: the
  // nearest BAKED frame or layer, under a control naming the one you asked
  // for. The attention side was fixed to refuse; these now match.
  if (p === "/api/vla/frame") {
    const v = await bundle<any>("vla");
    const want = Number(q.get("t") ?? v.frame);
    const have = Object.keys(v.frames).map(Number).sort((a, b) => a - b);
    const frame = v.frames[String(want)];
    if (!frame) {
      return refuse(
        422,
        `this demo recorded frames ${have.join(", ")} of episode ${v.episode}, ` +
          `not frame ${want}. Installed, ModelMRI decodes any frame of any ` +
          `LeRobot episode you have pulled.`,
      );
    }
    return ok(frame);
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
    const block = v.attention[String(want)];
    if (!block) {
      return refuse(
        422,
        `this demo recorded the vision tower at layers ${have.join(", ")}, ` +
          `not layer ${want}. Installed, every layer is available.`,
      );
    }
    return ok(block);
  }

  // ---------------------------------------------------------- image models
  //
  // NOT-LOADED, not refused. `/api/image` describes what this process is
  // holding, and a static page holds no pipeline — which is precisely what
  // "nothing is loaded" means, so the honest answer is the real resting
  // status with the reason filled in. A 501 here would paint an error over a
  // panel that is working correctly, which is the mistake `/api/pull/progress`
  // above already records.
  //
  // `capabilities: []` matters more than it looks. It is the list every
  // control on that panel is gated on, so an empty one is what keeps a demo
  // with no pipeline behind it from offering a capture button.
  const noPipeline = (reason: string) => ({
    loaded: false,
    repo: "",
    family: "",
    architecture: "",
    device: "",
    dtype: "",
    capabilities: [],
    // `null` is "nothing here knows", which is a different claim from 0 —
    // 0 would say this page is holding an unconditional model.
    cross_attention_dim: null,
    image_size: null,
    components: {},
    bytes_resident: 0,
    load_seconds: null,
    reason,
    means:
      `No image model is held in this process, so nothing here can say what ` +
      `one attends to or when it commits. ${reason}`,
  });
  if (p === "/api/image") {
    return ok(
      noPipeline(
        "This page is a static recording with no process behind it, so there " +
          "is nothing here that could hold a diffusion pipeline.",
      ),
    );
  }
  if (p === "/api/image/unload") {
    return ok(noPipeline("There was nothing to unload — this page holds no pipeline."));
  }
  if (p === "/api/image/available") {
    // An empty list with the reason attached, rather than somebody else's
    // cache. `/api/models/discovered` shipped one person's 17 repositories to
    // every visitor once; the rule that came out of it is that this page may
    // only offer what it can actually replay, and it can replay no pipeline.
    return ok({
      models: [],
      known: 0,
      means:
        "This page is a static recording and cannot read a disk, so it lists " +
        "no image models — not because none are cached, but because there is " +
        "no machine here to ask. Installed, this names every diffusion " +
        "pipeline already on your disk and downloads nothing to do it.",
    });
  }
  if (p === "/api/image/load") {
    return refuse(
      501,
      `Loading a diffusion pipeline reads several gigabytes of weights into a ` +
        `process, and this page is a static recording with no process to read ` +
        `them into. Installed, ModelMRI identifies the checkpoint from JSON, ` +
        `scans it for anything that executes on load, and prices it against ` +
        `your card — three refusals that cost nothing — before a byte moves.`,
    );
  }
  if (p === "/api/image/attention" || p === "/api/image/knockout") {
    return refuse(
      409,
      "Both of these run the real pipeline: the map captures cross-attention " +
        "where it is computed during a live denoising run, and the knockout " +
        "regenerates the image once per word at the same seed. There is no " +
        "pipeline behind this page, and a baked cross-attention map would be " +
        "a picture of a run nobody made. `pip install modelmri` to point it " +
        "at a pipeline of your own.",
    );
  }
  if (p === "/api/image/attention/cost" || p === "/api/image/steps/cost") {
    return refuse(
      409,
      "This prices a run this page cannot make, so the number would describe " +
        "a wait nobody here is going to have — and the memory half of it is " +
        "read off the loaded pipeline's own latent shape, which there is none " +
        "of here.",
    );
  }
  return undefined;
}

/** The steered output for the baked A/B (used by the features panel). */
export async function demoSteered(): Promise<{ baseline: string; steered: string; feature: number; scale: number }> {
  const f = await bundle<any>("features");
  return { baseline: f.baseline, steered: f.steered, feature: f.feature, scale: f.scale };
}
