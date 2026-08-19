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

/** Path segments that sit where a trace id sits and are not one.
 *
 *  `/api/traces/search` and `/api/traces/{id}` are the same shape, so a route
 *  table that matches on shape alone reads "search" as the id of a trace and
 *  serves a recording under it. Named here rather than ordered around,
 *  because ordering is a rule nobody can see. */
const TRACE_COLLECTIONS = new Set(["search", "import", "dataset", "adopt"]);

/** Which recorded model is on screen.
 *
 *  The picker offers every model it discovers, so with one recording you
 *  could select any model in that list and the demo would keep replaying
 *  Qwen/Qwen3-1.7B underneath — the page attributing one model's sentence to
 *  another, which is how a visitor comes away believing the model they picked
 *  wrote a sentence it never saw. Either a scenario exists for what you
 *  picked, or the load is refused by name.
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
 *  that quietly served layer 0 for every unbaked layer drew one head's arcs
 *  under a dial naming another, for nearly every selection the controls
 *  offered.
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
  if (p === "/api/session/trace") {
    // The run carried INSIDE an opened `.mri`. This page is a recording of a
    // live session rather than a bundle somebody opened in it, so there is no
    // carried run — which is the ordinary answer for most sessions, not a
    // limitation of this page. The agent runs this demo does have are in the
    // trace store and reachable from `/api/traces`, where the panel lists
    // them. Saying `available: false` here and letting those show is the
    // truthful split; claiming a carried run would put the demo's own traces
    // under a label saying they arrived in a file.
    return ok({ available: false });
  }
  // Reading the HuggingFace cache means reading a disk, which this page does
  // not have. This used to answer with a model list written into this file --
  // an inventory of a machine nobody here can see, at a size nothing
  // measured. The "On this machine" tab below answers from `discovered.json`
  // because that payload was RECORDED and is labelled as the demo's own
  // recordings; serving it here instead would relabel recordings as cache
  // entries and claim they are sitting on the reader's disk.
  if (p === "/api/models/local") {
    return refuse(
      501,
      `Listing the models already in your HuggingFace cache reads your disk, ` +
        `and this page is a static recording served from the web with no ` +
        `access to one. Nothing here will name a model instead — a list ` +
        `written into the page would be an inventory of somebody else's ` +
        `machine. Installed, this reads your own cache and reports what each ` +
        `model actually occupies on disk.`,
    );
  }
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
  // An EMPTY list, and that is the true answer rather than an evasion. This
  // route reports the accelerators a machine has for a model to be placed on,
  // and a static recording has no machine and places no model. `DevicePicker`
  // renders nothing below two options — one device is not a choice — so the
  // control correctly disappears instead of offering hardware nobody has.
  //
  // Deliberately NOT the bundled accelerator: that field describes the card
  // the recording was MADE on, and offering it here as somewhere to send a
  // load would be this page claiming somebody else's GPU as the visitor's.
  if (p === "/api/devices") {
    return ok({
      devices: [],
      reason:
        "This page is a static recording with no machine behind it, so there " +
        "is no device to place a model on.",
    });
  }
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
  // Qwen3-0.6B in the picker above Qwen/Qwen3-1.7B's output.
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
    // Steering was only ever recorded against one scenario's SAE -- the one
    // `saeScenario()` below reads out of the index -- so it is the only
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
  // features.json carries ONE scenario's SAE, that model's recorded sentence
  // and a steered completion of it. Served unconditionally, selecting any
  // other model left the page reporting an SAE loaded against a model of a
  // different width, a feature strip showing one model's words under
  // another's output, and an A/B pairing one model's baseline against
  // another's steered text. That is the failure this module's own docstring
  // says was eliminated — one model's sentence attributed to another.
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

  // `/api/traces/…` is EIGHT routes on the real server, not one, and this
  // used to answer the whole prefix with the recorded trace document. So the
  // bundle preview, the pattern finder, the dataset builder, search, import
  // and adopt were each handed a document of the WRONG SHAPE and a 200 to go
  // with it. The preview then read `n_steps` off a trace, got `undefined`,
  // and `.toLocaleString()` blanked the entire page a few seconds after load.
  //
  // The crash was the honest outcome; the quiet ones were worse. A panel
  // handed the wrong document mostly does not crash — it renders whatever it
  // can find and shows the reader numbers that belong to something else.
  //
  // The recording carries exactly two of these. The rest refuse BY NAME.
  if (p === "/api/traces") return ok((await bundle<any>("traces")).list);

  // A single segment that is not one of the sibling collection routes is a
  // trace id. `search`, `import`, `dataset` and `adopt` sit at the same depth
  // as an id and are not one.
  const traceId = /^\/api\/traces\/([^/]+)$/.exec(p)?.[1];
  if (traceId && !TRACE_COLLECTIONS.has(traceId)) {
    return ok((await bundle<any>("traces")).trace);
  }

  if (p.endsWith("/bundle/preview")) {
    return refuse(
      409,
      "The preview counts the steps and the redactions in a file this page " +
        "would write from a live runtime's own state, and there is no such " +
        "runtime behind a recording. Numbers invented for it would describe " +
        "a file nobody is going to get.",
    );
  }
  if (p.endsWith("/patterns")) {
    return refuse(
      409,
      "Finding patterns means querying every run recorded on a machine. This " +
        "page carries one recording, so any answer would be a pattern of one.",
    );
  }
  if (p === "/api/traces/search") {
    return refuse(
      409,
      "Search runs against the full-text index of every step recorded on your " +
        "machine. This page has a single recording and no index behind it.",
    );
  }
  if (p === "/api/traces/dataset") {
    return refuse(
      409,
      "Building an evaluation set draws cases from the runs recorded on your " +
        "machine. One recording is not a set.",
    );
  }
  if (p.startsWith("/api/traces/import")) {
    return refuse(
      409,
      "Importing reads a file from your disk into a local database. This page " +
        "has neither. `pip install modelmri` to bring your own traces in.",
    );
  }
  if (p.startsWith("/api/traces/")) {
    return refuse(
      409,
      "This acts on the trace database a local install keeps. The recording " +
        "on this page is read-only and has nothing behind it to change.",
    );
  }

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
      // The walk that did not happen did not stop early either. Both fields
      // are on the live shape and were absent here, which is how a demo
      // drifts from the tool it stands in for.
      truncated: false,
      scan_limit: 0,
      means:
        "This page is a static recording and cannot read a disk, so it lists " +
        "no image models — not because none are cached, but because there is " +
        "no machine here to ask. Installed, this names every diffusion " +
        "pipeline already on your disk and downloads nothing to do it.",
    });
  }
  // ---- finding one, which needs a disk and a network this page has neither of
  //
  // Split deliberately three ways rather than answered with one refusal. The
  // "on this machine" tab fails for a different reason from the "find one"
  // tab, and a visitor who cannot tell them apart learns that model discovery
  // is broken rather than that a static page has nothing to discover with.
  if (p === "/api/image/local") {
    // The same rule `/api/image/available` above is written for, and the same
    // scar behind it: `/api/models/discovered` once shipped one person's 17
    // repositories to every visitor. This page may offer only what it can
    // replay, and it can replay no pipeline.
    return ok({
      models: [],
      bytes_on_disk: 0,
      unsized: 0,
      // No walk ran, so it did not stop early either. Zero rather than the
      // server's 200: copying that constant here would be a second home for
      // a number that lives in `imaging.SCAN_CACHE_LIMIT`.
      truncated: false,
      scan_limit: 0,
      means:
        "This page is a static recording and cannot read a disk, so it lists " +
        "no image models and no bytes — not because none are here, but " +
        "because there is no machine here to ask. Installed, this names " +
        "every diffusion pipeline on your disk with what each one weighs, " +
        "and marks the ones holding configs and no weights as the " +
        "interrupted downloads they are rather than as models ready to load.",
    });
  }
  if (p === "/api/image/discovered") {
    // The folder walk, answered the same way the cache read above is: an empty
    // list with the reason attached. A recording has no working directory to
    // walk, and `roots: []` is the truthful answer to "where did you look" —
    // nowhere. Naming a plausible directory would be this page inventing a
    // path on a machine it cannot see.
    return ok({
      models: [],
      roots: [],
      truncated: false,
      // Zero because no walk ran, not because the limit is zero. Copying the
      // server's 120 here would be a second home for a number that lives in
      // `imaging.SCAN_DIRS_LIMIT`, and it would drift the day that changes.
      // Never rendered either way: the sentence carrying it is behind
      // `truncated`.
      scan_limit: 0,
      means:
        "This page is a static recording and has no working directory to " +
        "walk, so it lists no folders and no models found in them. " +
        "Installed, this searches the directory ModelMRI was started from " +
        "(plus anything in MODELMRI_MODELS_DIR) and names every image model " +
        "sitting there outside the Hub cache — and it reports which " +
        "directories it walked, so an empty answer says where it looked.",
    });
  }
  if (p === "/api/image/tasks") {
    // Empty rather than a copy. The table lives in `image_catalog.TASKS`, and
    // re-typing it here would be a second source of truth for what this tool
    // can open — one that drifts the day a tag is added on the server and
    // offers a visitor a task no checkpoint here could ever load.
    return ok({
      tasks: [],
      default: "",
      means:
        "Choosing what to search for is the first half of a Hub call this " +
        "page cannot make, so no tasks are offered here rather than a copy " +
        "of the list that would drift from the one the tool actually reads. " +
        "Installed, this names every kind of image model ModelMRI can open — " +
        "and what each one offers is settled by the checkpoint's own config " +
        "when it loads, never by the task it was listed under.",
    });
  }
  if (p === "/api/image/search") {
    // An empty list with the reason attached, not a red error: the tab is
    // working correctly and has nothing to show, which is the distinction
    // `/api/image` above is written for.
    return ok({
      models: [],
      task: q.get("task") ?? "",
      means:
        "Searching for a model to download is a live call to the " +
        "HuggingFace Hub, and this page is a static recording with nothing " +
        "behind it to make one — so no results are listed. A baked list " +
        "would be a snapshot of a download count that has moved since, " +
        "offered under a Load button with no process to load into. " +
        "Installed, this searches the Hub by task, says what each result " +
        "weighs before you click, and marks the ones already on your disk.",
    });
  }
  // ---- what this page is doing, and how to make it stop -------------------
  //
  // Nothing, and nothing. Both are honest answers rather than refusals, for
  // the reason `/api/pull/progress` above already records: the panel polls
  // progress the moment a load starts, and a 501 would draw an error across a
  // panel that is behaving correctly.
  if (p === "/api/image/progress") {
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
  if (p === "/api/image/cancel") {
    return ok({
      stopping: false,
      means:
        "There is no load in flight to stop — this page is a static " +
        "recording and holds no pipeline.",
    });
  }
  // ---- the computer-vision asks -------------------------------------------
  //
  // All four need the classifier itself. `capabilities: []` on the resting
  // status already gates every control that would call them, so these are
  // belt and braces — but an endpoint whose only protection is a UI gate is
  // one refactor away from being reachable, and `/api/features/ablate` above
  // is the note about what a prefix handler answering the wrong shape costs.
  if (
    p === "/api/image/cv/predict" ||
    p === "/api/image/cv/readout" ||
    p === "/api/image/cv/attribute"
  ) {
    return refuse(
      409,
      "Each of these runs the classifier on YOUR picture: the prediction is " +
        "one forward pass, the readout reads the patch grid out of that same " +
        "pass, and the occlusion sweep re-runs it once per window. There is " +
        "no model behind this page to run, and a baked answer would be a " +
        "claim about somebody else's photograph.",
    );
  }
  if (p === "/api/image/cv/cost") {
    return refuse(
      409,
      "This prices the sweep from the loaded checkpoint's OWN input geometry " +
        "— the size its processor resizes your picture to, not the size of " +
        "the file you picked — and there is no checkpoint here to ask.",
    );
  }
  // ---- the step trace and the filmstrip ------------------------------------
  if (p === "/api/image/steps" || p === "/api/image/filmstrip") {
    return refuse(
      409,
      "Both watch a real denoising run: the trace records what the model " +
        "committed to at each step, and the filmstrip decodes the latent " +
        "between them. There is no pipeline behind this page to step, and a " +
        "baked strip would be frames from a run nobody made.",
    );
  }
  if (p === "/api/image/filmstrip/cost") {
    return refuse(
      409,
      "The decode cost is read off the loaded pipeline's own latent shape " +
        "and its VAE, which there is none of here — so the number would " +
        "describe a wait nobody on this page is going to have.",
    );
  }
  // ---- the adapter reader --------------------------------------------------
  if (p === "/api/image/adapter") {
    return refuse(
      501,
      "Reading a LoRA means opening a file on your disk and measuring what " +
        "it moves, and a page served from the web cannot see a filesystem. " +
        "Installed, this takes any adapter you have and says which modules " +
        "it touches and by how much — no base model needed.",
    );
  }
  if (p === "/api/image/size") {
    return refuse(
      501,
      `Pricing a download means asking the Hub what \`${
        q.get("repo") || "that model"
      }\` publishes, which is a live call this static page has no process to ` +
        `make. Nothing here will guess a size instead: a number invented for ` +
        `a picker is the one thing a size column exists to prevent.`,
    );
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
  // ---- the occlusion preflight, answered FOR REAL --------------------------
  //
  // The one image route this page can answer honestly, and the reason is that
  // it needs nothing: `vision_attr.estimate` is arithmetic over a geometry,
  // not a measurement of a model. Refusing it would have been the easy answer
  // and the wrong one — the whole argument of that route is that the number
  // arrives BEFORE anything is spent, and a visitor who only ever sees it
  // refused never learns that a stride of 1 is a different afternoon from a
  // stride of 16.
  //
  // Duplicated arithmetic is a real cost and it is taken deliberately here.
  // `modelmri/vision_attr.py` is the source of truth; this mirrors `_axis`,
  // `_count_windows`, `_clamp_batch` and `estimate` exactly, ceiling
  // included, so the sentence a visitor reads is the sentence the tool
  // writes. Anything that drifts there has to be brought across.
  if (p === "/api/image/attribution/cost") {
    const num = (name: string, fallback: number) => {
      const raw = q.get(name);
      if (raw === null || raw.trim() === "") return fallback;
      return Number(raw);
    };
    const height = num("height", 224);
    const width = num("width", 224);
    const patch = num("patch", 16);
    // 0 is the query-string way of saying "not stated", and the module then
    // uses the patch size — non-overlapping windows.
    const asked = num("stride", 0);
    const stride = asked || patch;
    const batchAsked = num("batch", 32);

    const whole = (v: number, name: string) =>
      Number.isInteger(v) ? "" : `${name} must be a whole number of pixels, not ${v}.`;
    // Python's format rounds a half to EVEN and `toFixed` rounds it away from
    // zero, so `{12.5:.0f}` is "12" here and "13" there. The geometry that
    // lands exactly on a half is a patch of 14 at stride 16 — the ViT-large
    // patch size against the ordinary one — which is far too ordinary to let
    // the two sentences differ by a digit.
    const asPython = (v: number) => {
      const floor = Math.floor(v);
      const rest = v - floor;
      if (rest > 0.5) return floor + 1;
      if (rest < 0.5) return floor;
      return floor % 2 === 0 ? floor : floor + 1;
    };
    // The same refusals, in the same order, with the same sentences. A
    // geometry the tool would reject must be rejected here too, or the demo
    // teaches a schedule that cannot run.
    const bad =
      whole(height, "height") ||
      whole(width, "width") ||
      whole(patch, "patch") ||
      whole(stride, "stride") ||
      (patch < 1 ? "patch must be at least 1 pixel." : "") ||
      (stride < 1
        ? "stride must be at least 1 pixel. A stride of 0 would place every " +
          "window at the same place forever."
        : "") ||
      (height < 1 || width < 1
        ? `an image of ${height}x${width} pixels has nothing to occlude.`
        : "") ||
      (patch > Math.min(height, width)
        ? `a patch of ${patch} does not fit inside a ${height}x${width} ` +
          `image. The occluder has to be smaller than what it is occluding.`
        : "") ||
      (stride > patch
        ? `a stride of ${stride} with a patch of ${patch} leaves ` +
          `${asPython((1 - patch / stride) * 100)}% of the pixels under no ` +
          `window at all, so the map would have holes in it and still look ` +
          `like a map of the whole image. Set the stride to ${patch} or less.`
        : "") ||
      (batchAsked < 1 ? "batch must be at least 1 occluded copy per call." : "");
    if (bad) return refuse(422, bad);

    // `_axis`: starts every `stride` pixels, plus a final one pulled back to
    // the edge when the last window would leave a strip uncovered. Without
    // that clamp the map is silent about part of the image while still being
    // presented as a map OF the image.
    const axis = (length: number) => {
      const starts: number[] = [];
      for (let s = 0; s + patch <= length; s += stride) starts.push(s);
      if (starts.length === 0) starts.push(0);
      if (starts[starts.length - 1] + patch < length) starts.push(length - patch);
      return starts.length;
    };
    const count = (h: number, w: number, st: number) => {
      const one = (length: number) => {
        const starts: number[] = [];
        for (let s = 0; s + patch <= length; s += st) starts.push(s);
        if (starts.length === 0) starts.push(0);
        if (starts[starts.length - 1] + patch < length) starts.push(length - patch);
        return starts.length;
      };
      return [one(h), one(w)] as const;
    };

    const rows = axis(height);
    const cols = axis(width);
    const nWindows = rows * cols;
    const passes = nWindows + 1;
    // MAX_BATCH, and both numbers travel because a silent cap is a defect.
    const batch = Math.min(batchAsked, 64);
    const calls = 1 + Math.ceil(nWindows / batch);
    const inputBytes = batch * 3 * height * width * 4;
    // MAX_PASSES. `estimate` NEVER refuses on this — a caller about to be
    // refused needs the number that got them refused.
    const ceiling = 4096;
    const within = passes <= ceiling;

    let fits = "";
    if (!within) {
      for (let s = 1; s <= patch; s++) {
        const [r, c] = count(height, width, s);
        if (r * c + 1 <= ceiling) {
          fits = `A stride of ${s} would be ${r * c} windows (${r}x${c}) and fits.`;
          break;
        }
      }
      if (!fits) {
        const [r, c] = count(height, width, patch);
        fits =
          `Even a plain tiling at stride ${patch} is ${r * c} windows, so ` +
          `this image needs a larger patch or a raised ceiling — and a ` +
          `larger patch means a coarser map, which is the trade being made.`;
      }
    }

    return ok({
      map_rows: rows,
      map_cols: cols,
      n_windows: nWindows,
      passes,
      forward_calls: calls,
      batch,
      patch,
      stride,
      input_bytes_per_call: inputBytes,
      // `null` is "nobody measured", not "instant". Nothing here has timed a
      // forward pass, and a forecast off a typed constant would be invented.
      seconds: null,
      within_ceiling: within,
      ceiling,
      means:
        `A ${patch}x${patch} occluder at stride ${stride} over a ` +
        `${height}x${width} image is ${nWindows} windows — a ${rows}x${cols} ` +
        `map — and ${passes} forward passes, sent ${batch} at a time in ` +
        `${calls} calls. The occluded copies alone are ` +
        `${(inputBytes / 1e6).toLocaleString(undefined, {
          minimumFractionDigits: 1,
          maximumFractionDigits: 1,
        })} MB per call; the activations behind them are a multiple of that ` +
        `which nothing here can know without running the model.` +
        ` No per-pass time was measured, so there is no forecast here — an ` +
        `invented one would be a number this tool made up.` +
        (within
          ? ""
          : ` THIS IS PAST THE CEILING OF ${ceiling} PASSES and \`sweep\` ` +
            `will refuse it. ${fits}`),
    });
  }
  if (p === "/api/image/attribution") {
    return refuse(
      409,
      "The preflight above is real arithmetic and answers here; this is the " +
        "sweep itself, and it is the half that needs a model. Every window " +
        "of your picture is covered up and the checkpoint re-run — 197 " +
        "forward passes for a 224x224 image at a 16-pixel patch — and the " +
        "score is the signed movement in the class logit that results. " +
        "There is no model behind this page to move, and a baked map would " +
        "be somebody else's photograph attributed to a run you did not " +
        "make. `pip install modelmri` and point it at a ViT, a detector or " +
        "a segmentation head of your own.",
    );
  }
  return undefined;
}

/** The steered output for the baked A/B (used by the features panel). */
export async function demoSteered(): Promise<{ baseline: string; steered: string; feature: number; scale: number }> {
  const f = await bundle<any>("features");
  return { baseline: f.baseline, steered: f.steered, feature: f.feature, scale: f.scale };
}
