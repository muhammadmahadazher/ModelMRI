import { DEMO } from "./demo";

export interface ModelStatus {
  loaded: boolean;
  hf_id: string | null;
  device: string | null;
  dtype: string | null;
  n_params: number | null;
}

export interface SessionInfo {
  app: string;
  version: string;
  model: ModelStatus;
}

export interface AttentionMeta {
  available: boolean;
  n_layers?: number;
  n_heads?: number;
  n_tokens?: number;
  /** True when these numbers came from an opened `.mri`, not a live model. */
  replay?: boolean;
}

export interface AttentionData {
  layer: number;
  head: number;
  tokens: string[];
  matrix: number[][];
  replay?: boolean;
}

/** An opened `.mri` — someone else's analysis, without their model. */
export interface SessionState {
  open: boolean;
  meta?: {
    model: string | null;
    device: string | null;
    dtype: string | null;
    n_params: number | null;
    note?: string;
    scope?: string;
    precision?: string;
    created_at?: string | null;
    modelmri?: string | null;
  };
  prompt?: string;
  generation?: string;
  n_tokens?: number;
  n_slices?: number;
  /** "layer:head" keys this session actually captured. */
  slices?: string[];
}

/** A failure the server took the trouble to explain.
 *
 *  This API answers failures as `{"error": "..."}`, and those sentences are
 *  the good part — "SAE d_in=768 does not match model hidden_size=896
 *  (Qwen/Qwen2.5-0.5B-Instruct). This SAE was trained on a different model."
 *  tells you exactly what to do. The old helper threw the whole envelope, so
 *  what actually reached the screen was:
 *
 *      Error: 422: {"error":"SAE d_in=768 does not match model …"}
 *
 *  Status code and JSON braces around a sentence written for a human. Every
 *  panel showed errors that way, because every panel goes through here.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly body: string;

  constructor(status: number, body: string) {
    super(explain(body) || `the server returned ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

function explain(body: string): string {
  try {
    const parsed = JSON.parse(body);
    if (typeof parsed?.error === "string") return parsed.error;
    if (typeof parsed?.detail === "string") return parsed.detail;
    // FastAPI request-validation failures: [{loc, msg, type}, …]
    if (Array.isArray(parsed?.detail)) {
      const msgs = parsed.detail
        .map((d: { msg?: string }) => d?.msg)
        .filter(Boolean);
      if (msgs.length) return msgs.join("; ");
    }
  } catch {
    // not JSON — fall through to the raw text
  }
  return body.trim().slice(0, 300);
}

/** The sentence to put in front of a person, whatever was thrown. */
export function errorText(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) throw new ApiError(r.status, await r.text());
  return r.json() as Promise<T>;
}

export const getSession = () =>
  fetch("/api/session").then((r) => json<SessionInfo>(r));

/** A load that the user stopped. Not an error — they asked. */
export interface LoadCancelled {
  cancelled: true;
  message: string;
}

export const loadModel = (
  hf_id?: string,
  source: "hf" | "ollama" = "hf",
  confirm = false,
) =>
  fetch("/api/model/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(hf_id ? { hf_id, source, confirm } : { source, confirm }),
  }).then((r) => json<ModelStatus | LoadCancelled>(r));

/** Stop an in-flight download.
 *
 *  This exists because a click in the picker began fetching 1.5 TB and the
 *  only way to stop it was to kill the server process.
 */
export const cancelLoad = () =>
  fetch("/api/model/cancel", { method: "POST" }).then((r) =>
    json<{ stopping: boolean }>(r),
  );

export interface LoadProgress {
  active: boolean;
  hf_id: string | null;
  stage: string; // resolving | weights | device | ready | error
  detail: string;
  bytes_done: number;
  bytes_total: number;
  elapsed_s: number;
  error: string | null;
}

export const getLoadProgress = () =>
  fetch("/api/model/progress").then((r) => json<LoadProgress>(r));

export const getAttentionMeta = () =>
  fetch("/api/attention/meta").then((r) => json<AttentionMeta>(r));

export const getAttention = (layer: number, head: number) =>
  fetch(`/api/attention?layer=${layer}&head=${head}`).then((r) =>
    json<AttentionData>(r),
  );

/** How far removing one head moves the next-token answer.
 *
 *  Deliberately not called "importance". These are marginal sensitivities to
 *  removing one head alone; they are not additive and not shares of the
 *  prediction — measured on gpt2 layer 0, the twelve per-head scores sum to
 *  1.995 while ablating the whole layer gives 0.208.
 */
export interface HeadScore {
  layer: number;
  head: number;
  kl: number;
  p_top_before: number;
  p_top_after: number;
  flips_top: boolean;
}

export interface Ablation {
  baseline: string;
  position: number;
  target_token: string;
  /** Same forward pass twice with nothing ablated. Anything at or below
   *  this is arithmetic, not the model. */
  noise_floor_kl: number;
  passes: number;
  elapsed_s: number;
  ranked: HeadScore[];
  means: string;
}

/** One run's attention minus another's, over the same token sequence. */
export interface AttentionDiff {
  layer: number;
  head: number;
  a: string;
  b: string;
  tokens: string[];
  matrix: number[][];
  max_abs: number;
  moved: number;
  cells: number;
  /** Set when the difference is exactly zero and there is a reason worth
   *  stating — e.g. ablating a head cannot change its own layer. */
  note: string;
}

export const getAttentionDiff = (
  layer: number,
  head: number,
  a: string,
  b: string,
) =>
  fetch(
    `/api/attention/diff?layer=${layer}&head=${head}&a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`,
  ).then((r) => json<AttentionDiff>(r));

export const rankHeads = (
  layer: number,
  baseline: "zero" | "mean" = "zero",
  scope: "layer" | "all" = "layer",
) =>
  fetch(
    `/api/attention/ablate?layer=${layer}&baseline=${baseline}&scope=${scope}`,
  ).then((r) => json<Ablation>(r));

export const getSessionState = () =>
  fetch("/api/session/state").then((r) => json<SessionState>(r));

export const openSession = (data: ArrayBuffer) =>
  fetch("/api/session/open", {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: data,
  }).then((r) => json<SessionState>(r));

export const closeSession = () =>
  fetch("/api/session/close", { method: "POST" }).then((r) =>
    json<SessionState>(r),
  );

/** Download the current analysis as a `.mri`.
 *
 *  Not a plain `<a download>`: the server answers a failure as JSON with a
 *  409, and a link would cheerfully save that JSON as a `.mri` file. Fetching
 *  it means an error stays an error and reaches the panel as a sentence.
 */
export async function exportSession(
  layer: number,
  head: number,
  note: string,
): Promise<{ blob: Blob; filename: string }> {
  const r = await fetch(
    `/api/session/export?layer=${layer}&head=${head}&note=${encodeURIComponent(note)}`,
  );
  if (!r.ok) throw new ApiError(r.status, await r.text());
  const disposition = r.headers.get("Content-Disposition") || "";
  const match = /filename="([^"]+)"/.exec(disposition);
  return { blob: await r.blob(), filename: match?.[1] || "session.mri" };
}

export interface SAEStatus {
  loaded: boolean;
  repo: string | null;
  hook: string | null;
  layer: number | null;
  d_in: number | null;
  d_sae: number | null;
}

export interface FeaturesSummary {
  tokens: string[];
  top: [number, number][][]; // per token: [feature_id, activation][]
}

export interface FeatureDetail {
  feature_id: number;
  activations: number[];
  max: number;
  argmax: number;
}

export const getSAE = () => fetch("/api/sae").then((r) => json<SAEStatus>(r));

export const loadSAE = () =>
  fetch("/api/sae/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  }).then((r) => json<SAEStatus>(r));

export const getFeaturesSummary = (topK = 8) =>
  fetch(`/api/features/summary?top_k=${topK}`).then((r) =>
    json<FeaturesSummary>(r),
  );

export const getFeatureDetail = (id: number) =>
  fetch(`/api/features/${id}`).then((r) => json<FeatureDetail>(r));

export const setSteer = (feature_id: number | null, scale = 0) =>
  fetch("/api/steer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ feature_id, scale }),
  }).then((r) => json<{ active: boolean }>(r));

export const promptOnce = (
  prompt: string,
  max_new_tokens = 24,
  temperature = 0,
  /** false = run the model without rebasing the server's analysis target.
   *  The steering A/B must not commit: its two short completions would
   *  silently replace the long generation the panels are describing. */
  commit = true,
) =>
  fetch("/api/model/prompt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, max_new_tokens, temperature, commit }),
  }).then((r) => json<{ generation: string }>(r));

export interface VLAStatus {
  loaded: boolean;
  mode: string;
  reason: string;
  repo: string | null;
  n_layers: number;
  n_heads: number;
  grid: number[];
  warmup_ms: number | null;
  /** Configured server-side; named before anything is opened. */
  dataset_repo: string;
  policy_repo: string;
}

export interface VLAEpisode {
  index: number;
  length: number;
  task: string;
  from_ts: number;
  to_ts: number;
}

export interface VLADataset {
  repo_id: string;
  fps: number;
  image_shape: number[];
  n_episodes: number;
  episodes: VLAEpisode[];
}

export interface VLAFrame {
  episode: number;
  t: number;
  timestamp: number;
  state: number[];
  action: number[];
  task: string;
  image: string;
  width: number;
  height: number;
}

export interface VLAHeat {
  layer: number;
  head: number;
  grid: number[];
  heat: number[][];
  min: number;
  max: number;
}

export const getVLA = () => fetch("/api/vla").then((r) => json<VLAStatus>(r));

export const loadVLA = () =>
  fetch("/api/vla/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  }).then((r) => json<VLAStatus>(r));

export const getVLAEpisodes = () =>
  fetch("/api/vla/episodes").then((r) => json<VLADataset>(r));

export const getVLAFrame = (episode: number, t: number) =>
  fetch(`/api/vla/frame?episode=${episode}&t=${t}`).then((r) => json<VLAFrame>(r));

export const analyseVLA = (episode: number, t: number) =>
  fetch("/api/vla/analyse", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ episode, t }),
  }).then((r) => json<{ layers: number; heads: number; latency_ms: number }>(r));

export const getVLAAttention = (layer: number, head = -1) =>
  fetch(`/api/vla/attention?layer=${layer}&head=${head}`).then((r) => json<VLAHeat>(r));

export interface HubAuth {
  signed_in: boolean;
  user: string | null;
  source: string | null;
}

export interface HubModel {
  id: string;
  downloads: number;
  likes: number;
  gated: boolean;
  usable: boolean;
  updated: string;
  params: string | null;
  /** Download size from the repo's own metadata. null when it publishes
   *  none — GGUF and pickle repos mostly. Never render null as 0. */
  size_gb: number | null;
  suggested?: boolean;
}

export interface OllamaModel {
  name: string;
  size_gb: number;
  family: string;
  params: string;
  quant: string;
}

export interface OllamaState {
  up: boolean;
  models: string[];
  installed: OllamaModel[];
  suggested: { name: string; size: string; note: string }[];
  host: string;
}

export const getHubAuth = () => fetch("/api/hub/auth").then((r) => json<HubAuth>(r));

export const hubSignIn = (token: string) =>
  fetch("/api/hub/signin", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
  }).then((r) => json<HubAuth>(r));

export const hubSignOut = () =>
  fetch("/api/hub/signout", { method: "POST" }).then((r) => json<HubAuth>(r));

export const getHubModels = (q = "") =>
  fetch(`/api/hub/models?q=${encodeURIComponent(q)}`).then((r) => json<HubModel[]>(r));

export const pullOllama = (name: string, confirm = false) =>
  fetch("/api/ollama/pull", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, confirm }),
  }).then((r) => json<{ pulled: string }>(r));

/** What a pull would cost, and whether this machine can take it.
 *
 *  Advisory only — the server enforces the same rule on the pull itself, so
 *  skipping this check gains nothing.
 */
export interface OllamaSize {
  name: string;
  bytes: number;
  free_bytes: number;
  ok: boolean;
  overridable: boolean;
  warning: string;
}

export const getOllamaSize = (name: string) =>
  fetch(`/api/ollama/size?name=${encodeURIComponent(name)}`).then((r) =>
    json<OllamaSize>(r),
  );

/** Look up any Ollama model by name.
 *
 *  Ollama publishes no search API, so the tab offers a name box rather than
 *  a result list — which reaches strictly more models: namespaced ones, any
 *  tag, and anything published since whatever list we might have shipped.
 */
export interface OllamaResolved {
  found: boolean;
  name: string;
  bytes: number;
  free_bytes?: number;
  ok: boolean;
  overridable: boolean;
  warning: string;
  error: string;
}

export const resolveOllama = (name: string) =>
  fetch(`/api/ollama/resolve?name=${encodeURIComponent(name)}`).then((r) =>
    json<OllamaResolved>(r),
  );

export interface Accelerator {
  kind: string; // cuda | rocm | xpu | mps | cpu
  torch_device: string;
  name: string;
  vram_gb: number | null;
  dtype: string;
  reason: string;
}

export const getAccelerator = () =>
  fetch("/api/accelerator").then((r) => json<Accelerator>(r));

export interface LocalModel {
  id: string;
  size_gb: number;
}

export const getLocalModels = () =>
  fetch("/api/models/local").then((r) => json<LocalModel[]>(r));

export interface DiscoveredModel {
  id: string;
  name: string;
  path: string;
  kind: "hf-cache" | "folder" | "gguf";
  size_gb: number;
  loadable: boolean;
  note: string;
}

export interface Discovery {
  models: DiscoveredModel[];
  roots: string[];
  truncated: boolean;
}

export const getDiscovered = () =>
  fetch("/api/models/discovered").then((r) => json<Discovery>(r));

export const getOllama = () => fetch("/api/ollama").then((r) => json<OllamaState>(r));

export interface TraceSummary {
  /** Scripted sample data, not a run you recorded. */
  demo?: boolean;
  id: string;
  name: string;
  started_at: string;
  n_steps: number;
  total_ms: number;
  n_errors: number;
}

export interface TraceStep {
  id: string;
  parent_id: string | null;
  kind: "llm_call" | "tool_call" | "subagent" | "mcp_call" | "user_turn" | "error";
  name: string;
  started_ms: number;
  duration_ms: number;
  input: string;
  output: string;
  tokens_in: number | null;
  tokens_out: number | null;
  error: boolean;
  seq: number;
}

export interface TraceDoc {
  id: string;
  name: string;
  started_at: string;
  steps: TraceStep[];
}

export const getTraces = () => fetch("/api/traces").then((r) => json<TraceSummary[]>(r));
export const getTrace = (id: string) =>
  fetch(`/api/traces/${id}`).then((r) => json<TraceDoc>(r));

export type StreamHandlers = {
  onToken: (text: string) => void;
  onDone: () => void;
  onError: (message: string) => void;
};

export interface Decode {
  max_new_tokens: number;
  temperature: number;
}

/** The settings are sent explicitly rather than left to the server's defaults.
 *  The readout reports what a run used, and a readout that reports a guess is
 *  worse than no readout. */
export function streamGenerate(
  prompt: string,
  h: StreamHandlers,
  decode: Decode = { max_new_tokens: 256, temperature: 0.7 },
): () => void {
  if (DEMO) {
    // No WebSocket on a static host: replay the baked generation word by
    // word so the streaming UI behaves exactly as it does locally.
    let cancelled = false;
    void (async () => {
      const { generation } = await fetch("/api/model/prompt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      }).then((r) => r.json());
      const pieces = String(generation).match(/\s*\S+/g) ?? [];
      for (const piece of pieces) {
        if (cancelled) return;
        h.onToken(piece);
        await new Promise((r) => setTimeout(r, 45));
      }
      if (!cancelled) h.onDone();
    })();
    return () => {
      cancelled = true;
    };
  }

  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/generate`);
  let finished = false;
  let sawToken = false;

  // Watchdog: if the socket produces nothing at all, fail loudly instead of
  // leaving the UI spinning forever (first token can be slow on cold CPU).
  const watchdog = window.setTimeout(() => {
    if (!finished && !sawToken) {
      finished = true;
      ws.close();
      h.onError("no tokens after 90s — the server may be stuck; retry or restart it");
    }
  }, 90_000);

  const finish = (fn: () => void) => {
    if (finished) return;
    finished = true;
    window.clearTimeout(watchdog);
    fn();
  };

  ws.onopen = () => ws.send(JSON.stringify({ prompt, ...decode }));
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data as string);
    if (msg.type === "token") {
      sawToken = true;
      h.onToken(msg.text);
    } else if (msg.type === "done") {
      finish(h.onDone);
      ws.close();
    } else if (msg.type === "error") {
      finish(() => h.onError(msg.message));
    }
  };
  ws.onerror = () => finish(() => h.onError("websocket error"));
  ws.onclose = () => finish(() => h.onError("connection closed before completion"));
  return () => {
    finished = true;
    window.clearTimeout(watchdog);
    ws.close();
  };
}

// ---------------------------------------------------------- custom models

export interface CustomLayer {
  order: number;
  name: string;
  kind: string;
  out_shape: number[];
  n_params: number;
  trainable: boolean;
  ms: number;
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
  pct_zero: number | null;
  pct_saturated: number | null;
  n_nonfinite: number;
  is_activation: boolean;
  note: string;
}

export interface CustomStatus {
  loaded: boolean;
  path: string | null;
  source: string | null;
  name: string | null;
  device: string;
  n_params: number;
  n_trainable: number;
  n_modules: number;
  input_shape: number[] | null;
  input_origin: string;
  input_reason: string;
  labels: string[] | null;
  reason: string;
  roots: string[];
}

export interface CustomCandidate {
  path: string;
  name: string;
  dir: string;
  has_example?: boolean;
  hint?: boolean;
  mb?: number;
}

export interface CustomRun {
  layers: CustomLayer[];
  input_shape: number[];
  labels: string[] | null;
  total_ms: number;
  n_layers: number;
  truncated: boolean;
  output_shape: number[];
  output: {
    top_index?: number[];
    top_value?: number[];
    argmax?: number;
    n_out?: number;
    nonfinite?: boolean;
  };
}

export const getCustom = () =>
  fetch("/api/custom").then((r) => json<CustomStatus>(r));

export const getCustomCandidates = () =>
  fetch("/api/custom/candidates").then((r) =>
    json<{
      adapters: CustomCandidate[];
      torchscript: CustomCandidate[];
      roots: string[];
    }>(r),
  );

export const loadCustom = (path: string) =>
  fetch("/api/custom/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  }).then((r) => json<CustomStatus>(r));

export const runCustom = (shape: number[] | null) =>
  fetch("/api/custom/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ shape }),
  }).then((r) => json<CustomRun>(r));

export const unloadCustom = () =>
  fetch("/api/custom/unload", { method: "POST" }).then((r) =>
    json<CustomStatus>(r),
  );

// ------------------------------------------------- SAE registry + the lens

export interface SAEOption {
  repo: string;
  models: string[];
  d_in: number;
  layers: number[];
  point: string;
  label: string;
  supported: boolean;
  note: string;
  default_hook: string;
}

export const getSAEOptions = () =>
  fetch("/api/sae/available").then((r) =>
    json<{
      model: string | null;
      matching: SAEOption[];
      usable: SAEOption[];
      catalogue: SAEOption[];
    }>(r),
  );

export const loadSAEFrom = (repo: string, hook: string) =>
  fetch("/api/sae/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo, hook }),
  }).then((r) => json<SAEStatus>(r));

export interface LensRow {
  layer: number;
  tokens: string[];
  probs: number[];
  entropy: number;
}

export const getLens = (topK = 5) =>
  fetch(`/api/lens?top_k=${topK}`).then((r) =>
    json<{
      layers: LensRow[];
      n_layers: number;
      final: string;
      settled_at: number | null;
    }>(r),
  );

// ------------------------------------------------------------ VLA datasets

export interface VLADatasetInfo {
  repo_id: string;
  ref: string | null;
  size_gb: number;
  usable: boolean;
  note: string;
}

export const getVLADatasets = () =>
  fetch("/api/vla/datasets").then((r) =>
    json<{ datasets: VLADatasetInfo[]; current: string }>(r),
  );

export const setVLADataset = (repo_id: string) =>
  fetch("/api/vla/dataset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo_id }),
  }).then((r) => json<VLADataset>(r));

export const deleteTrace = (id: string) =>
  fetch(`/api/traces/${id}`, { method: "DELETE" }).then((r) =>
    json<{ deleted: string }>(r),
  );

/** Clears recordings. Keeps the shipped sample by default — the docs point
 *  at it, and "clear my runs" should not throw away the thing that explains
 *  what a run looks like. */
export const clearTraces = (keepDemo = true) =>
  fetch(`/api/traces?keep_demo=${keepDemo}`, { method: "DELETE" }).then((r) =>
    json<{ deleted: number }>(r),
  );

// ----------------------------------------------------------------- storage

export interface PathInfo {
  override: string | null;
  data: string;
  config: string;
  cache: string;
  hf_home: string;
  hf_hub_cache: string;
  /** The actual files, not the directories that would contain them on a
   *  clean install. An upgrade from <=0.5.1 keeps reading `~/.modelmri`, and
   *  naming only the directory sent people looking in the wrong place. */
  trace_db: string;
  hub_token: string;
  undelivered_traces: string;
  models_dirs: string[];
  cwd: string;
  legacy: string | null;
  platform: string;
}

export const getPaths = () => fetch("/api/paths").then((r) => json<PathInfo>(r));
