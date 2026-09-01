import { DEMO, demoSessionFile } from "./demo";
import { VIEWER } from "./viewer";

export interface ModelStatus {
  loaded: boolean;
  hf_id: string | null;
  device: string | null;
  dtype: string | null;
  n_params: number | null;
  /** Was this model instruction-tuned? False means a base model, which
   *  continues text rather than answering — the usual reason an answer
   *  looks wrong when nothing is broken. */
  instruct?: boolean;
  /** Present only when this model was built from a GGUF. Every number
   *  measured on it then describes the QUANTISED weights, not the original —
   *  which is a caveat about the numbers, not about the model, so it rides
   *  with the status rather than being recomputed per panel. */
  gguf?: GgufLoaded | null;
  /** Transformer blocks in the loaded model. A property of the MODEL, not of
   *  a run — `/api/attention/meta` carries the same number but only after a
   *  generation, and the probe and patchscope panels both need to offer a
   *  layer before there is anything to generate from. */
  n_layers?: number | null;
}

/** One held thing, trimmed to what a header needs. The full status of each
 *  is on its own route; this is the answer to "is anything loaded". */
export interface HeldModel {
  loaded: boolean;
  repo: string;
  device: string;
  family?: string;
}

export interface SessionInfo {
  app: string;
  version: string;
  /** The TEXT model. Unchanged, and still the only one most panels care
   *  about. */
  model: ModelStatus;
  /** The image pipeline and the robot policy. The process can hold all three
   *  at once, and the header used to know about one — so a resident 3.3 GB
   *  pipeline sat under a badge reading "no model loaded". */
  image?: HeldModel;
  vla?: HeldModel;
}

export interface AttentionMeta {
  available: boolean;
  n_layers?: number;
  n_heads?: number;
  n_tokens?: number;
  /** WHY there is none, when there is none. The server distinguishes five
   *  cases — no model, nothing generated yet, the model changed since that
   *  generation, Ollama, and an architecture that publishes no attention at
   *  all — and they ask the reader for opposite things: "pick a model" versus
   *  "press the button you are looking at" versus "nothing here can show you
   *  this". Undeclared here, all five were dropped and the panel removed
   *  itself from the page instead. */
  reason?: string;
  /** True when these numbers came from an opened `.mri`, not a live model. */
  replay?: boolean;
}

export interface AttentionData {
  layer: number;
  head: number;
  tokens: string[];
  /** How many leading tokens are the prompt; the rest is the model's own
   *  output. The panel rests on the last prompt token. */
  n_prompt?: number;
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
  /** A recorded activation-patching trace, when the file carries one. A `.mri`
   *  used to hold attention and the logit lens only, so the one finding in
   *  this tool that is causal rather than correlational was the one you could
   *  not send anybody. */
  patch?: { available: boolean; clean: string; corrupt: string };
  /** A recorded PATCHING GRAPH, a separate section from `patch` above. It
   *  costs ~1,500 forward passes to build and cannot be rebuilt from a `.mri`
   *  at all, so the panel needs to know the file has one rather than offering
   *  a button whose only outcome is a refusal. */
  patch_graph?: {
    available: boolean;
    n_nodes: number;
    n_edges: number;
    /** The graph's OWN pair. `_patch_graph` has preserved these all along and
     *  nothing read them — the panel was handed the PATCH section's pair
     *  instead, so a file carrying a graph and no patch trace prefilled with
     *  the hardcoded demo prompts. */
    clean?: string;
    corrupt?: string;
  };
  /** Whether the recording carries a grounding result, and what it asked.
   *  The document itself is NOT in the file — a `.mri` carries passage
   *  previews, deliberately, because a grounded document is usually the
   *  private half of the pair. */
  ground?: { available: boolean; question: string };
  /** Whether the recording carries a LOGIT LENS, and what the model ended up
   *  saying.
   *
   *  Same reason as the three above: `runtime.logit_lens` will serve a
   *  recorded trajectory, but the only surface that draws one lives inside
   *  `FeaturesPanel`, which is `!replay` by construction — so the panel has to
   *  be told the file holds a lens before it can offer it.
   *
   *  `n_rows` is the number of rows carried, which is usually one MORE than
   *  the model's layer count because the trajectory starts at the embedding.
   *  `final` is "" when the file never named the answer, which is how the
   *  panel already reads an unnamed one. */
  lens?: { available: boolean; n_rows: number; final: string };
  /** Whether the recording carries a HEAD RANKING, and what it ranked heads
   *  against.
   *
   *  Same reason as the four above. `runtime.ablate_heads` has served a
   *  recorded ranking since the section landed, and `AttentionPanel` gated
   *  the only button that asks on `!replay` — a comment true of MEASURING a
   *  ranking and false of SHOWING one already in the file — so the tool's
   *  headline measurement was the one nobody could read out of a `.mri`.
   *
   *  `target_token` is `null`, never "", when the file did not name the token
   *  it watched: the panel prints it beside the button, and an empty string
   *  there would read as a target rather than as a silence.
   *
   *  `n_heads` is the number of RANKED ROWS the file carries, which is not
   *  the model's head count — a one-layer ranking scores one layer's heads. */
  ranking?: {
    available: boolean;
    target_token: string | null;
    n_heads: number;
  };
  /** Whether the recording carries BEHAVIOURAL HEAD LABELS.
   *
   *  The panel's only caller of `/api/attention/types` sits inside its
   *  `{ranked && …}` block, so the labels were locked behind a button a
   *  recording could never press. Unlocking that button is not enough on its
   *  own: without this flag the panel would offer the labels for a file that
   *  carries none, and the one outcome of pressing it would be a refusal. */
  head_types?: { available: boolean; n_labels: number };
  /** Whether the recording carries an AGENT RUN, and how much of one.
   *
   *  The three fields above all exist so a panel can show the recorded
   *  finding instead of a button that can only refuse. This one was missing
   *  while its siblings were here, so a bundle built around a failing step
   *  opened to an agents panel reading "0 recordings" with the run sitting
   *  inside the file.
   *
   *  `n_steps` is what the file holds; `n_steps_total` is what the sender's
   *  run held, and they differ when the section was capped on the way in.
   *  `step_ref` is the step the bundle was built AROUND — the reason it was
   *  sent — so the panel can open on it rather than on step one. */
  trace?: {
    available: boolean;
    /** The carried run's id, so a panel can tell a swapped file from a
     *  re-read one. */
    id: string;
    name: string;
    n_steps: number;
    n_steps_total: number;
    truncated: number;
    step_ref: string | null;
  };
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

/** Refuse, in the builds that have no model behind them, naming what this
 *  particular measurement would need.
 *
 *  The demo and viewer are static pages on GitHub Pages. `demo.ts` answers the
 *  endpoints the bundle has REAL recorded data for; everything else would
 *  reach the network and 404, and a 404 inside a panel reads as "this
 *  measurement is broken" rather than "this page has no model behind it".
 *  `tests/demo_check.py` holds the line: every endpoint the frontend can reach
 *  is either answered by the shim or carries a written exemption saying it is
 *  gated off here.
 *
 *  One implementation, a sentence per call site. The sentence is the whole
 *  point — "install it" without saying what the measurement actually costs
 *  teaches nobody anything — so every caller passes its own.
 */
function noModelHere(needs: string): Promise<never> {
  return Promise.reject(
    new ApiError(
      409,
      JSON.stringify({
        error:
          needs +
          " There is no model behind this page — install ModelMRI " +
          "(`pip install modelmri`) to run it on your own.",
      }),
    ),
  );
}

function explain(body: string): string {
  try {
    const parsed = JSON.parse(body);
    if (typeof parsed?.error === "string") return parsed.error;
    if (typeof parsed?.detail === "string") return parsed.detail;
    // FastAPI request-validation failures: [{loc, msg, type}, …]
    //
    // NAMED. `msg` alone is "Field required", and a call missing two of them
    // rendered "Field required; Field required" — a sentence that tells the
    // reader a field is missing without telling them which. `loc` carries the
    // parameter, so it goes in front: "height: Field required".
    if (Array.isArray(parsed?.detail)) {
      const msgs = parsed.detail
        .map((d: { msg?: string; loc?: unknown[] }) => {
          if (!d?.msg) return "";
          // `loc` is ["query", "height"] or ["body", "steps"]; the last entry
          // is the field, and the first is where it belongs.
          const field =
            Array.isArray(d.loc) && d.loc.length
              ? String(d.loc[d.loc.length - 1])
              : "";
          return field && field !== "body" ? `${field}: ${d.msg}` : d.msg;
        })
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

/** One device a model could be sent to.
 *
 *  `free_bytes` is `null` where the backend cannot report it -- Apple's
 *  unified memory, Intel XPU, and system RAM on every platform. Null is
 *  UNKNOWN; rendering it as 0 would say the machine is out of memory when
 *  nobody asked it.
 */
export interface DeviceOption {
  /** What to send as `device` on a load: "cuda:0", "cpu". */
  id: string;
  kind: string;
  name: string;
  vram_gb: number | null;
  dtype: string;
  reason: string;
  free_bytes: number | null;
  total_bytes: number | null;
  /** Where a load with no device named goes -- i.e. what has always happened. */
  is_default: boolean;
}

export interface DeviceList {
  devices: DeviceOption[];
  default: string;
  means: string;
}

/** Every device on this machine, not just the one in use.
 *
 *  `/api/session` reports the ONE device a model is on, which is a different
 *  question and the only one the app could answer before this.
 */
export const getDevices = () =>
  fetch("/api/devices").then((r) => json<DeviceList>(r));

export const loadModel = (
  hf_id?: string,
  source: "hf" | "ollama" = "hf",
  confirm = false,
  /** "" keeps the existing behaviour exactly: the server chooses, as it always
   *  has. Only a deliberate choice sends anything else. */
  device = "",
) =>
  fetch("/api/model/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(
      hf_id ? { hf_id, source, confirm, device } : { source, confirm, device },
    ),
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
  /** Seconds left at the average rate so far. `null` means the server does
   *  not have enough signal to say — which is a real answer, not zero. */
  eta_s: number | null;
  error: string | null;
}

export const getLoadProgress = () =>
  fetch("/api/model/progress").then((r) => json<LoadProgress>(r));

/** A download in flight, which is NOT the same slot as a model load — the
 *  picker can be pulling one model while the page behind it loads another. */
export const getPullProgress = () =>
  fetch("/api/pull/progress").then((r) => json<LoadProgress>(r));

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
 *  prediction — the per-head scores of a layer do not sum to what ablating
 *  the whole layer costs.
 */
export type Baseline = "zero" | "mean" | "resample";

export interface HeadScore {
  layer: number;
  head: number;
  /** For `resample` this is the MEDIAN over draws, not a single run. */
  kl: number;
  p_top_before: number;
  /** Null under `resample`: there is no single "after" across eight draws. */
  p_top_after: number | null;
  flips_top: boolean;
  /** Resample only — the spread across draws, which can run several times
   *  wider than the median it surrounds, so a single draw could have reported
   *  a very different number as the head's score. */
  kl_min?: number;
  kl_max?: number;
  draws?: number;
}

/** What produced one number: the setup, stamped onto the measurement rather
 *  than printed on the page. Every field that can genuinely fail to resolve is
 *  nullable AND paired with a note saying why, because "no revision" and
 *  "several revisions were cached so naming one would be a guess" send the
 *  reader to different places. Optional on every response: a `.mri` written
 *  before receipts existed carries none, and an older file must read as
 *  "no provenance recorded" rather than failing to open. */
export interface Receipt {
  op: string;
  request?: Record<string, unknown>;
  tool_version?: string | null;
  model?: string | null;
  revision?: string | null;
  revision_note?: string | null;
  dtype?: string | null;
  device?: string | null;
  attn_implementation?: string | null;
  /** null is "not seeded", which is NOT the same as seed 0. */
  seed?: number | null;
  tokenizer_sha256?: string | null;
  tokenizer_note?: string | null;
  prompt_sha256?: string | null;
  n_prompt_tokens?: number | null;
  measured_at?: string | null;
}

export interface Ablation {
  baseline: string;
  position: number;
  target_token: string;
  /** Same forward pass twice with nothing ablated. Anything at or below
   *  this is arithmetic, not the model. */
  noise_floor_kl: number;
  passes: number;
  /** How long the sweep took. OPTIONAL because a recorded one may not carry
   *  it: `session._ranking` copies `elapsed_s` only when it is an int, and
   *  `ablate.py` writes `round(seconds, 2)` — a float — so a ranking read
   *  back out of a `.mri` arrives with no duration at all. Typed as required,
   *  the header printed "18 forward passes · s" and the rate arithmetic below
   *  produced a NaN seconds-per-pass that priced the whole-model button. */
  elapsed_s?: number;
  ranked: HeadScore[];
  means: string;
  /** Set by `runtime.ablate_heads`'s replay branch and by `viewer.ts`'s
   *  `/api/attention/ablate`: these passes and seconds were spent on the
   *  SENDER'S machine. The panel reads it to keep a recording out of its
   *  seconds-per-pass estimate, which exists to price a sweep on THIS one. */
  recorded?: boolean;
  /** Resample only. Part of the measurement, not provenance: the same head
   *  scores differently against a different corpus. */
  corpus?: string;
  draws?: number;
  receipt?: Receipt | null;
}

/** One pair of baselines, and how far apart they rank the same heads. */
export interface BaselinePair {
  baselines: [string, string];
  /** Null when one side is constant — "uncorrelated" and "that is not a
   *  ranking" are different statements. */
  spearman: number | null;
  heads_compared: number;
  top_k: number;
  top_k_shared: number;
  top_k_disagree: number;
}

export interface BaselineComparison {
  pairs: BaselinePair[];
  means: string;
  rankings: Record<string, HeadScore[]>;
}

/** What a sweep would cost on THIS machine, before starting it. */
export interface CostEstimate {
  estimate: {
    passes: number;
    seconds: number | null;
    peak_bytes: number | null;
    free_bytes: number | null;
    fraction_of_free: number | null;
    verdict: "ok" | "tight" | "refuse" | "unknown";
    basis: string;
    unmeasured: string;
    notes: string[];
  };
  baseline: string;
  layers: number;
  n_heads: number;
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
  baseline: Baseline = "zero",
  scope: "layer" | "all" = "layer",
) =>
  fetch(
    `/api/attention/ablate?layer=${layer}&baseline=${baseline}&scope=${scope}`,
  ).then((r) => json<Ablation>(r));

/** Price the sweep before running it. `resample` is eight times the work. */
export const estimateAblation = (
  layer: number,
  baseline: Baseline = "zero",
  scope: "layer" | "all" = "layer",
) =>
  fetch(
    `/api/attention/ablate/estimate?layer=${layer}&baseline=${baseline}&scope=${scope}`,
  ).then((r) => json<CostEstimate>(r));

/** Run every baseline on one layer and report how much they disagree.
 *
 *  Deliberately not called on load: it runs all three, and resample dominates
 *  the total. */
export const compareBaselines = (layer: number) =>
  DEMO || VIEWER
    ? noModelHere(
        "Comparing the three ablation baselines runs all three, and the " +
          "resample arm draws its replacements from a corpus of at least 8 " +
          "sentences you supply — there is no corpus here, and a bundled one " +
          "would be a baseline measured against somebody else's text.",
      )
    : fetch(`/api/attention/baselines?layer=${layer}`).then((r) =>
        json<BaselineComparison>(r),
      );

/** How far masking one token moves the answer at one position.
 *
 *  Same units as `HeadScore.kl` — both come from `kl_nats` in ablate.py, so a
 *  head score and a token score on one screen mean the same thing. What they
 *  are NOT is comparable in behaviour: `modelmri/attribute.py` measured the
 *  singles under-stating one joint mask by 0.35x on gemma-3-270m-it over the
 *  typed span. The panel prints both live numbers rather than a factor.
 */
export interface TokenScore {
  index: number;
  token: string;
  kl: number;
  p_top_before: number;
  p_top_after: number;
  flips_top: boolean;
  /** Which list this row belongs in. The server decides: it is the only side
   *  that knows where the chat template ends, where the user's own words
   *  begin, and where the prompt stops.
   *
   *  Four values, and each of the three that are not `typed` exists because
   *  collapsing it into another one made the panel state something nobody
   *  measured. `generated` is the model's OWN output — folded into `template`
   *  it put the model's own words under a heading reading "chat template
   *  scaffold", on a model whose span_note says in the same breath that it has
   *  no chat template. `unknown` is "the server could not locate your words"
   *  — folded into `typed` it put the chat template under "what you typed". */
  group: "typed" | "template" | "generated" | "unknown";
}

/** One run of the token ranking, exactly as the server reports it.
 *
 *  Every field the panel renders is here. Nothing about a result is computed
 *  or worded on this side — the refusals, the coverage sentence, the mask
 *  check and the "what this means" paragraph are all measured or written
 *  where the measurement happened, because a caveat re-typed in the client is
 *  a caveat that can drift away from the number it belongs to.
 */
export interface TokenAttribution {
  baseline: string;
  position: number;
  target_token: string;
  /** The same forward pass twice, nothing masked. Measured at exactly 0.0 on
   *  Qwen3-0.6B and gemma-3-270m-it; anything at or below it is arithmetic
   *  rather than the model. */
  noise_floor_kl: number;
  passes: number;
  elapsed_s: number;
  /** What those passes were spent on, and the warning that goes with showing
   *  a duration at all: the count transfers between machines, the seconds do
   *  not. Rendered rather than summarised — the breakdown is the reason the
   *  cost is what it is. */
  passes_note: string;
  ranked: TokenScore[];
  /** Measured, reported, and deliberately outside the order — the score at
   *  index 0 follows the POSITION rather than the token sitting in it. `note`
   *  is the server's evidence for that, and it travels with the row. */
  index0: TokenScore & { note: string };
  n_tested: number;
  n_candidates: number;
  truncated: boolean;
  /** Half-open index window that was actually tested. When `truncated`, the
   *  candidates below `tested_span[0]` were asked nothing — and on the strip
   *  they would otherwise be indistinguishable from tokens that were tested
   *  and scored nothing. */
  tested_span: [number, number];
  /** Where the prompt ends. Rows at or past it are the model's own output,
   *  which is a third thing from "your words" and "the chat template". */
  n_prompt: number;
  /** "N of M were tested; one not listed was not tested, not found
   *  unimportant." Shown verbatim when the run was capped. */
  coverage: string;
  sum_of_singles: number;
  joint_kl: number;
  /** Did `attention_mask[0, j] = 0` actually empty column j at every layer?
   *  Checked once, at one index, on the mechanism rather than on each row. */
  mask_verified: boolean;
  max_residual_weight: number | null;
  mask_check: string;
  means: string;
  /** Half-open token span of the text the user actually typed, or `null` when
   *  the server could not locate it.
   *
   *  `null` does NOT mean "all of it is yours". It means the panel has no
   *  basis for splitting the list, and must therefore show one list with the
   *  server's reason attached rather than a "what you typed" heading it cannot
   *  justify. If a build ever drops this field the panel lands in exactly that
   *  branch, which is the conservative failure and not a silent one.
   */
  typed_span: [number, number] | null;
  /** The server's sentence about that span — why it is what it is, or why it
   *  could not be found. Rendered as-is in BOTH cases: it is the only place
   *  that can say "this model has no chat template wrapped around it, so the
   *  scaffold list is empty because there is no scaffold" rather than leaving
   *  an empty heading to be read as a missing measurement. */
  span_note: string;
}

/** Rank the tokens before `position`, or before the server's own default when
 *  no position is given.
 *
 *  The position is the whole claim. "Which words mattered" is meaningless
 *  without "mattered to WHICH answer", so the parameter is optional only in
 *  the sense that the server names its default in the response; it is never
 *  absent from what gets rendered.
 */
export const attributeTokens = (position?: number) =>
  fetch(
    position === undefined
      ? "/api/attention/attribute"
      : `/api/attention/attribute?position=${position}`,
  ).then((r) => json<TokenAttribution>(r));

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
  /** Bundle an agent run alongside the mechanistic snapshot, and optionally
   *  name the step that failed. The recipient clicks it and lands in the
   *  attention view of the generation that produced the bad argument. */
  bundle?: { trace_id: string; step_ref?: string },
): Promise<{ blob: Blob; filename: string }> {
  // This one call wants bytes and a Content-Disposition, so it never went
  // through the patched fetch — which meant "Share this view" 404'd on the
  // demo against its own origin. The demo ships a real `.mri` of its own run,
  // so the demo -> viewer hop is something a visitor can do rather than read
  // about.
  if (DEMO) {
    const blob = await demoSessionFile();
    if (!blob) throw new ApiError(409, "this demo bundle carries no .mri");
    return { blob, filename: "modelmri-demo.mri" };
  }
  const params = new URLSearchParams({
    layer: String(layer),
    head: String(head),
    note,
  });
  if (bundle?.trace_id) {
    params.set("trace_id", bundle.trace_id);
    if (bundle.step_ref) params.set("step_ref", bundle.step_ref);
  }
  const r = await fetch(`/api/session/export?${params}`);
  if (!r.ok) throw new ApiError(r.status, await r.text());
  const disposition = r.headers.get("Content-Disposition") || "";
  const match = /filename="([^"]+)"/.exec(disposition);
  return { blob: await r.blob(), filename: match?.[1] || "session.mri" };
}

/** What the SAE worked out about the activations it is being fed.
 *
 *  Absent until the first `encode`, because it is measured against the model
 *  the SAE is attached to rather than read from a config file. Every field is
 *  the server's — including `usable` and the threshold behind it, so the panel
 *  never decides for itself whether a reconstruction is good enough.
 */
export interface SAECalibration {
  convention: string; // "raw" | "b_dec" | "centered" | "centered+b_dec"
  center: boolean;
  subtract_b_dec: boolean;
  fvu: number; // fraction of variance unexplained
  l0: number; // features firing per token
  rel_err: number;
  n_tokens: number;
  declared_b_dec: boolean | null; // null = the config did not say
  ranked: [string, number][]; // every convention tried, best first
  usable: boolean;
  unusable_at: number;
}

/** Which published SAE this is, and how each coordinate got chosen.
 *
 *  `chosen_by` is the half that matters: every coordinate is either "caller"
 *  or a sentence naming the rule and the alternatives it beat, so the panel
 *  can tell a deliberate choice from a default. `available` is the index for
 *  THIS layer only, and `null` in it never means "none exist" — it means the
 *  Hub listing could not be read.
 *
 *  Open-ended by design, but a SAELens release answers with these keys and
 *  they are the ones worth rendering:
 *    `architecture`  whether the gate was DECLARED by cfg.json or READ off a
 *                    schema too old to have the key — different facts, and a
 *                    reader has to be able to tell them apart.
 *    `apply_b_dec_to_input`  what the config said, beside
 *                    `SAECalibration.declared_b_dec`. "Absent" is its own
 *                    answer and is not "false".
 *    `model`         the model the release names. Nothing checks it against
 *                    the loaded one — only that the widths agree — so it is
 *                    a declaration to read, not a guarantee.
 *    `weights`       present only when the weight file carried tensors this
 *                    does not apply. Its absence means nothing was dropped.
 *    `rescale_acts_by_decoder_norm`  present only when the decoder-norm fold
 *                    was applied, in which case the loaded `W_dec` rows are
 *                    unit-norm and are NOT the published ones. */
export interface SAERelease {
  repo: string;
  layout: string;
  /** `null` when the release is several files, as SAELens releases are. */
  file: string | null;
  layer: number;
  point: string;
  width: string | null;
  /** What the directory name CLAIMS the average L0 is. The MEASURED one is
   *  `SAECalibration.l0`, computed on a different corpus — read side by
   *  side, never interchanged. */
  advertised_l0: number | null;
  chosen_by: Record<string, string>;
  available: Record<string, number[]> | null;
}

export interface SAEStatus {
  loaded: boolean;
  repo: string | null;
  hook: string | null;
  layer: number | null;
  d_in: number | null;
  d_sae: number | null;
  calibration?: SAECalibration | null;
  /** "relu" | "jumprelu" | "topk" | "gated" — the rule the release NAMED,
   *  not one inferred from which tensors it shipped. `null` when nothing is
   *  loaded: an unloaded panel does not have a plain-ReLU SAE, it has no
   *  SAE. */
  activation: string | null;
  /** `[min, max]` of the JumpReLU thresholds. `null` for every architecture
   *  with no thresholds at all, rather than thresholds of zero. */
  threshold_span: [number, number] | null;
  /** How many features a top-k gate lets fire per token. `null` for every
   *  other architecture, which have no such number — not 0, which would read
   *  as a gate that fires nothing. */
  k: number | null;
  release: SAERelease | null;
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

// ------------------------------------------- readouts shared between panels
//
// Two rules that belong to the measurement rather than to a panel, kept in one
// place so a second panel cannot quietly adopt a different one. AttentionPanel
// still carries its own copies of both — it is owned elsewhere right now — and
// these are byte-for-byte the same rule, deliberately, so switching that file
// over is a deletion rather than a decision. If they ever disagree, THIS is
// the one that was written next to the endpoint contract.

/** "11s" / "2m 17s" — a wait the user can decide about. */
export const humanSeconds = (s: number) =>
  s < 90
    ? `${Math.max(1, Math.round(s))}s`
    : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;

/** Seconds per forward pass, kept as the FASTEST rate seen rather than the
 *  latest.
 *
 *  The first ranking after a load pays for CUDA warm-up and runs several times
 *  slower — measured on an RTX 4060, Qwen3-0.6B's first layer took 3.05 s and
 *  the next two 0.80 and 0.78. Since a cost estimate only appears once there
 *  is a measurement, a latest-rate estimator quotes its worst possible number
 *  (46.8% over on that run). Warm-up only ever inflates, so the minimum is the
 *  honest estimator; once warm the extrapolation held to within 2.5% across
 *  repeats on both models.
 */
export const fastestRate = (
  prev: number | null,
  elapsed_s: number,
  passes: number,
): number | null => {
  if (passes <= 0 || !(elapsed_s > 0)) return prev;
  const rate = elapsed_s / passes;
  return prev === null ? rate : Math.min(prev, rate);
};

// ------------------------------------------------------- feature ablation

/** One feature's causal effect, exactly as `modelmri/feature_ablate.py`
 *  reports it. Nothing here is recomputed on this side.
 */
export interface FeatureScore {
  feature_id: number;
  /** Peak activation over the positions that were edited — re-encoded in
   *  float32 by the ablation, NOT read from the float16 cache the bar chart
   *  plots. fp16 rounding moves what the cache holds, so the two numbers can
   *  differ and neither is wrong. */
  activation: number;
  /** Every token index the feature was removed at. One entry at
   *  `scope="position"`. At `scope="prompt"` this is the field that lets the
   *  panel show a feature which fires nowhere near the token being attributed
   *  and still reaches the answer through attention. */
  positions: number[];
  kl: number;
  /** What a RANDOM direction of the same norm, subtracted at the same tokens,
   *  cost. It is not zero and it is not small. A row below its own control has
   *  a score that is the size of its edit rather than the identity of its
   *  feature. */
  control_kl: number;
  /** `kl > control_kl`. A row that fails it can still sit near the top of the
   *  bar chart above — activation rank and clearing the control are two
   *  different orderings. */
  clears_control: boolean;
  /** Share of this feature's original activation the SAE's ENCODER still
   *  reports after the feature's own contribution was subtracted, at the worst
   *  of the edited positions. Not a failure of the edit — the stream moves by
   *  exactly one rank-1 term, which `removal_verified` checks — but a property
   *  of this SAE: W_enc[:,f] and W_dec[f] are not dual, so the encoder reads
   *  other features' contributions through f's direction. */
  encoder_residual: number;
  p_top_before: number;
  p_top_after: number;
  flips_top: boolean;
  /** The server's verdict against `resolution_kl`, and never recomputed here
   *  against `noise_floor_kl`. The floor is exactly 0.0 on this path and two
   *  measured scores came back NEGATIVE (-1e-08, -3e-08) — float32 summation
   *  over the whole vocabulary — so a client greying out "at or below the
   *  floor" would grey out nothing. Measured: 2 of 43 scores are at or below
   *  the floor, 8 of 43 are below the resolution. */
  below_resolution: boolean;
}

/** One run of the feature ranking, exactly as the server reports it.
 *
 *  Every caveat rendered by the panel is either a field here or is computed
 *  from two fields here. Nothing is remembered: the additivity direction in
 *  particular is READ OFF THIS RUN, because features miss in the opposite
 *  direction from heads — the head panel's singles over-count the joint while
 *  these under-count it — and a remembered direction would be exactly
 *  backwards.
 */
export interface FeatureAblation {
  /** Which edit was made, named rather than assumed. "Removing a feature" is
   *  three different experiments in the literature and two of them are
   *  measurably indefensible on this SAE. */
  intervention: string;
  scope: "position" | "prompt";
  position: number;
  target_token: string;
  hook: string;
  layer: number;
  /** The edit hook installed with the stream written back UNCHANGED. Measured
   *  at exactly 0.0 over four repeats. */
  noise_floor_kl: number;
  /** The same forward pass again with no hook at all. */
  replay_kl: number;
  /** The number a score has to clear to be a measurement rather than
   *  arithmetic. A different number from the floor, and the one to grey out
   *  on. */
  resolution_kl: number;
  passes: number;
  elapsed_s: number;
  ranked: FeatureScore[];
  n_tested: number;
  /** How many rows were SCORED, which is not how many came back. `top_k` trims
   *  the response; a row it drops was tested and measured. Confusing the two
   *  is how the panel came to label a scored feature "not tested" — the exact
   *  inversion of what `truncated` means. */
  n_scored: number;
  /** How many rows are in `ranked`. When this equals `n_scored`, a feature
   *  absent from `ranked` really was never tested; when it is smaller, absence
   *  proves nothing. */
  n_returned: number;
  /** How many scored rows fell below `resolution_kl`, and how many came back
   *  NEGATIVE — which a KL cannot be, and which is the evidence that the line
   *  to grey out on is the resolution and not the floor. Both counted in THIS
   *  run rather than remembered. */
  n_below_resolution: number;
  n_negative_kl: number;
  /** The server's own sentence about the trim, which says the thing the panel
   *  needs and must not paraphrase: "A row left out here WAS tested and
   *  scored, which is not what `truncated` means." */
  rows_note: string;
  n_candidates: number;
  truncated: boolean;
  /** "N of M firing features were tested … one not listed was NOT TESTED, not
   *  found unimportant." Shown verbatim when the run was capped. */
  coverage: string;
  sum_of_singles: number;
  joint_kl: number;
  /** The SAE's calibrated input convention, and the aggregate FVU that goes
   *  with it. `fvu` is an aggregate over every token and is dominated by
   *  token 0; it is NOT this position's accuracy, which is what
   *  `residual_share` and `residual_kl` are for. */
  convention: string;
  fvu: number;
  /** The aggregate reconstruction error as a NORM fraction — the same units as
   *  `residual_share`, and the only one of the two aggregates that can be put
   *  beside it. `fvu` is a squared-error fraction: pairing 0.000984 with
   *  0.2036 states a 200x gap where the like-for-like one (0.0294 against
   *  0.2036) is 7x. */
  rel_err: number;
  /** The WORST share of a token's norm the SAE fails to model, over the window
   *  these edits actually landed in. At position scope that is the attributed
   *  token; at prompt scope it is the worst token in the window. Null when the
   *  stream has no norm there. */
  residual_share: number | null;
  /** The same quantity at the attributed token alone, kept beside the
   *  scope-matched one rather than replaced by it. */
  residual_share_at_position: number | null;
  /** [first, last] token index the reconstruction baseline was substituted
   *  over — the same window the edits use. */
  residual_window: [number, number];
  /** What that gap costs in the same units as every score above: substituting
   *  the reconstruction over `residual_window` with NO feature removed.
   *  Measured 0.07753 at position scope and 0.221217 at prompt scope, 2.85x
   *  more, and only 1 of 256 prompt-scope scores clears the second. */
  residual_kl: number;
  residual_means: string;
  /** Did the EDIT land — is the stream the model received exactly
   *  `x - activation x W_dec[f]`? Measured deviation 0.0 in float32. This is a
   *  property of the edit and the dtype, and it is all this flag now claims:
   *  whether the SAE still READS the feature afterwards is per row, in
   *  `encoder_residual`, and fails on 38 of 43 rows. */
  removal_verified: boolean;
  edit_deviation: number;
  removal_check: string;
  /** How many scored rows the SAE's encoder still reads above 1% of their
   *  original activation, and the worst of them. */
  n_encoder_residual: number;
  encoder_residual_max: number;
  /** How many scored rows score above their own same-norm random control. */
  n_clearing_control: number;
  control_means: string;
  means: string;
  receipt?: Receipt | null;
}

/** Rank the SAE's features by how far removing one moves the answer.
 *
 *  `position` is the whole claim and is never optional here: the panel always
 *  knows which token it is asking about, because the user clicked it. `scope`
 *  chooses between the features firing AT that token and every feature firing
 *  at or before it — two different questions with two different answers, so
 *  the response echoes back which one it answered.
 *
 *  The DEMO/VIEWER refusal is a second lock on a door the panel already keeps
 *  shut. It matters because of how demo.ts is written: `/api/features/ablate`
 *  falls inside its `p.startsWith("/api/features/")` handler, which would
 *  answer 200 with the single-feature DETAIL payload. A fabricated ranking
 *  rendered as a measurement is the one failure this project cannot ship, and
 *  it would pass demo_check's static coverage check, which only asks whether
 *  *some* handler matches the path.
 */
export const rankFeatures = (
  position: number,
  scope: "position" | "prompt" = "position",
) => {
  // ASK FOR EVERY SCORED ROW, and the reason is not size. The panel annotates
  // the bar chart with each plotted feature's score, so a feature that was
  // measured but trimmed out of the response is indistinguishable there from
  // one that was never tested — and the panel said "not tested" about it,
  // which is the exact inversion of what `truncated` means. Measured: at
  // prompt scope the server scored 256 rows and the default top_k=64 returned
  // 64, so #18994, scored 0.00031514 at causal rank 72, was labelled untested.
  //
  // The cap is MAX_CANDIDATES = 256 server-side, so this asks for more rows
  // than can exist and the response is never trimmed. The panel still checks
  // `n_returned === n_scored` before using the words "not tested", because a
  // number in a URL is not a guarantee.
  if (DEMO || VIEWER) {
    return Promise.reject(
      new ApiError(
        409,
        JSON.stringify({
          error:
            "Ranking features by causal effect runs a forward pass per firing " +
            "feature against a live model, and there is no model behind this " +
            "page. Install ModelMRI (`pip install modelmri`) to run it on your " +
            "own model — the control is not offered here rather than offered " +
            "and broken.",
        }),
      ),
    );
  }
  return fetch(
    `/api/features/ablate?position=${position}&scope=${scope}&top_k=1024`,
  ).then((r) => json<FeatureAblation>(r));
};

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
  /** "unavailable" until a tower is loaded, "perception" after. The Python
   *  comment listed "data" and "full" too and neither was ever assigned. */
  mode: string;
  reason: string;
  repo: string | null;
  /** `null` is UNKNOWN, never a zero or a "cpu" nobody chose. These are read
   *  off a checkpoint, so with nothing loaded there is no answer — and the
   *  sibling fields `repo` and `warmup_ms` already said so with `null` while
   *  these published a confident 0. A resting `/api/vla` reported
   *  `n_layers: 0, n_heads: 0` beside `repo: null`, which reads as a tower
   *  that exists and has no layers. */
  device: string | null;
  n_layers: number | null;
  n_heads: number | null;
  grid: number[];
  /** Input square and patch edge, in pixels. */
  image_size: number | null;
  patch_size: number | null;
  /** Tokens this tower prepends before the patches — a class token, plus
   *  registers in DINOv2-style towers. 0 for SigLIP, which SmolVLA uses.
   *  `null` until the first analysis, because it is COUNTED by running the
   *  tower rather than read from a config. */
  n_prefix_tokens: number | null;
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
  cameras?: string[];
  video_key?: string;
  /** Whether a picture can actually be produced from this dataset on this
   *  machine. The episode table comes out of parquet and arrives fine
   *  without a video decoder, so a panel that gates on the table alone draws
   *  a frame scrubber that can only ever refuse. */
  frames_readable: boolean;
  /** Why not, when `frames_readable` is false. `""` when it is true, so
   *  there is one field to test rather than two that can disagree. */
  frames_reason: string;
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

/** The ACTION half of a robot policy, which lives in another process.
 *
 *  `VLAStatus` above describes the vision tower this server holds — where a
 *  policy LOOKED. This describes the sidecar that would say what it would DO.
 *  They are separate because lerobot's pins cannot share an environment with
 *  ModelMRI's, and the panel shows them separately for the same reason: on
 *  most machines one is loaded and the other is not, and a single "VLA" light
 *  would have to pick one of those to lie about. */
export interface PolicyStatus {
  running: boolean;
  /** The wire version the sidecar answered with. `null` when none answered. */
  contract: number | null;
  policy_repo: string;
  revision: string;
  device: string;
  dtype: string;
  /** Empty means the policy did not publish its action statistics — which
   *  means an overlay against a dataset's recorded actions must be refused,
   *  never drawn on an assumed identity scale. */
  normalisation: Record<string, Record<string, number[]>>;
  port: number;
  /** Did a process ANSWER on that port? Distinct from `running`, which asks
   *  whether a policy is loaded, and from `port`, which is only what a file
   *  on the server's disk claimed. A sidecar that is up but drifted, wedged
   *  or slow is `reachable: true, running: false` — and the `reason` then
   *  carries what it actually said rather than a guess about a crash. */
  reachable: boolean;
  reason: string;
  means: string;
  /** Which policy family the checkpoint declared: smolvla, pi0, act, … */
  family: string;
  /** The camera keys this policy consumes. A request missing one is refused
   *  rather than blank-filled — a VLA given a subset of its views answers a
   *  different question in the same shape. */
  cameras: string[];
  /** `null` means the policy consumes no state, which is NOT a width of 0. */
  state_width: number | null;
  action_width: number | null;
  chunk_size: number | null;
  /** Whether the action head samples. `false` means #50's instruction-swap
   *  test has no reference to measure against — its denominator is the
   *  policy's own sampling spread — so that test refuses rather than
   *  reporting a spread of zero. */
  samples: boolean;
  /** The versions in the OTHER environment. The whole point of the separation
   *  is that these differ from this server's. */
  lerobot_version: string;
  torch_version: string;
  /** `null` when nothing has reported a torch build yet — distinct from
   *  `false`, which means it really is a CPU build and the policy will run
   *  forty times slower than the model in this process. */
  accelerated: boolean | null;
  installed: boolean;
  venv: string;
  contract_here: number;
  install_hint: string;
  venv_disk_bytes: number;
  assumed_policy_bytes: number;
}

/** The demo and the .mri viewer answer this from `demo.ts`'s shim, which
 *  patches fetch — so this stays a plain fetch and the coverage check in
 *  `demo_check.py` can see the handler. An answer written here instead was
 *  invisible to that check and reported as an unhandled endpoint. */
export const getPolicy = () =>
  fetch("/api/policy").then((r) => json<PolicyStatus>(r));

/** Load a policy's vision tower. Blank repo = the server's default.
 *
 *  Any checkpoint carrying a vision tower works: the tensor prefix and the
 *  vision config are read from the file rather than assumed, so this is not
 *  limited to the one policy it was built against. */
export const loadVLA = (repo?: string) =>
  fetch("/api/vla/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(repo && repo.trim() ? { repo: repo.trim() } : {}),
  }).then((r) => json<VLAStatus>(r));

export const getVLAEpisodes = (camera?: string) =>
  fetch(
    camera ? `/api/vla/episodes?camera=${encodeURIComponent(camera)}` : "/api/vla/episodes",
  ).then((r) => json<VLADataset>(r));

export const getVLAFrame = (episode: number, t: number, camera?: string) =>
  fetch(
    `/api/vla/frame?episode=${episode}&t=${t}` +
      (camera ? `&camera=${encodeURIComponent(camera)}` : ""),
  ).then((r) => json<VLAFrame>(r));

export const analyseVLA = (episode: number, t: number) =>
  fetch("/api/vla/analyse", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ episode, t }),
  }).then((r) => json<{ layers: number; heads: number; latency_ms: number }>(r));

export const getVLAAttention = (layer: number, head = -1) =>
  fetch(`/api/vla/attention?layer=${layer}&head=${head}`).then((r) => json<VLAHeat>(r));

/** One series over the episode's timesteps, with what it is a series OF. */
export interface TimelineTrack {
  column: string;
  /** Per-dimension series, each as long as `timesteps`. `null` at a timestep
   *  is "not measured there" — a non-finite value in the recording, or a
   *  point a stride dropped — and NEVER zero, which would be a reading. */
  series: (number | null)[][];
  /** The dataset's own names for each dimension, or `null`. Not "dim 0":
   *  a generated label looks exactly like a published one on screen. */
  names: string[] | null;
  /** The dataset's own unit. Most LeRobot datasets publish none, and two
   *  tracks with no units cannot be compared to each other. */
  unit: string | null;
  /** Bounds over what was actually read, per dimension — the axis to draw
   *  against. `null` for a dimension with no finite value at all. */
  low: (number | null)[];
  high: (number | null)[];
  /** Non-finite values found and left out, per dimension. Corruption in a
   *  recording, reported rather than interpolated away. */
  n_nonfinite: number[];
}

/** Several of an episode's series on ONE time axis.
 *
 *  The robot panel's frame and scrubber answer "what did the camera see at t"
 *  and nothing else. The questions people bring to a recorded episode are
 *  about COINCIDENCE — the gripper closed here, what was the state doing, did
 *  the reward move — and every one of those needs two series on one axis.
 *  Read off two panels with two x-ranges, a coincidence gets asserted that is
 *  not there, so the sameness of `timesteps` is the product. */
export interface EpisodeTimeline {
  episode: number;
  repo_id: string;
  /** The timesteps every track is indexed by. One axis, shared. */
  timesteps: number[];
  /** Seconds from the start of THIS episode, when the dataset records them.
   *  Not absolute: a LeRobot timestamp is seconds into the concatenated file,
   *  so a reader shown those sees episode 40 start at 700 seconds. */
  seconds: number[] | null;
  length: number;
  stride: number;
  /** True when `stride > 1` — the timesteps between are absent from every
   *  series rather than smoothed over. */
  strided: boolean;
  tracks: TimelineTrack[];
  /** Columns this dataset does not publish, with the reason. Absent, never an
   *  empty track: a reward line at zero says the reward WAS zero. */
  absent: { column: string; why: string }[];
  means: string;
}

export const getEpisodeTimeline = (episode: number, maxPoints = 600) =>
  fetch(`/api/vla/timeline?episode=${episode}&max_points=${maxPoints}`).then((r) =>
    json<EpisodeTimeline>(r),
  );

// ------------------------------------------------------------ image models
//
// Eight routes over one handle, and deliberately the same shape as the VLA
// block above: a status that always answers, a load and an unload, two cost
// routes that answer BEFORE anything is spent, and two measurements that
// refuse by name when the architecture has no such thing to measure.
//
// The field a panel must read before drawing any control is `capabilities`.
// It comes from `imaging.detect`, keyed on what the checkpoint IS rather than
// on what it is called, and a family the server cannot name arrives with an
// EMPTY list. So a control is offered because the server said the measurement
// exists, never because a repo id looked like Stable Diffusion.

/** One image model already on this disk, as `imaging.ImageModel` reports it.
 *
 *  `known: false` is a first-class answer rather than a gap: it arrives with
 *  `reason`, and with an empty `capabilities`, because a family this cannot
 *  identify must offer nothing rather than everything. The list is what is on
 *  the disk, not what the server happens to understand.
 */
export interface ImageModelInfo {
  /** The repo id, `owner/name`. */
  path: string;
  /** The identifier: `unet_diffusion`, `dit_diffusion`, `vit`, … */
  family: string;
  /** The same thing in words — "a UNet diffusion model". The server writes
   *  it, so a panel never has to turn an identifier into prose itself. */
  label: string;
  architecture: string;
  pipeline: string;
  /** `{unet: "UNet2DConditionModel", vae: "AutoencoderKL", …}`. Empty for a
   *  plain transformers model, which is a fact rather than a gap. */
  components: Record<string, string>;
  /** Three states and never two. A positive width means the denoiser attends
   *  to prompt tokens; **0 means UNCONDITIONAL** — there are no word-to-pixel
   *  maps to draw at all; `null` means the denoiser's config did not say, so
   *  nothing here knows. Rendering `null` as 0 would turn "not stated" into
   *  "this model ignores your prompt". */
  cross_attention_dim: number | null;
  image_size: number | null;
  capabilities: string[];
  /** {capability: why it cannot be measured on THIS checkpoint}. A control
   *  that is simply absent reads as a missing feature; the reason says
   *  whether another checkpoint would answer. */
  unavailable?: Record<string, string>;
  /** "text" | "class" | "none" — what steers it. A class-conditioned model
   *  takes a number from a fixed list and has no prompt at all. */
  conditioning?: string;
  n_classes?: number | null;
  known: boolean;
  reason: string;
  means: string;
}

export interface ImageAvailable {
  models: ImageModelInfo[];
  known: number;
  /** The cache walk stops at `scan_limit`, so a flat count is a claim that
   *  this is everything. `means` carries the sentence; these carry the fact. */
  truncated: boolean;
  scan_limit: number;
  means: string;
}

/** What `ImageHandle` is holding, or why it is holding nothing.
 *
 *  Never raises server-side: a resting panel asks this on every load and
 *  `loaded: false` on its own is not an answer, so `reason` and `means` are
 *  populated in that case rather than left blank.
 */
export interface ImageStatus {
  loaded: boolean;
  repo: string;
  family: string;
  architecture: string;
  device: string;
  dtype: string;
  /** What may be measured on this pipeline: `cross_attention`,
   *  `token_knockout`, `step_commit`, `latent_trace`, `patch_attention`, …
   *  A capability that is absent is a control that is not shown. */
  capabilities: string[];
  /** {capability: why it cannot be measured on THIS checkpoint} — checked
   *  against the loaded pipeline, not guessed from the family. A control that
   *  is simply absent reads as a missing feature; the reason says whether
   *  another checkpoint would answer. */
  unavailable?: Record<string, string>;
  /** "text" | "class" | "none". A class-conditioned model takes a number from
   *  a fixed list and has no prompt, so a prompt box asks it a question it
   *  cannot be asked. */
  conditioning?: string;
  n_classes?: number | null;
  /** The same tri-state as `ImageModelInfo.cross_attention_dim`. */
  cross_attention_dim: number | null;
  image_size: number | null;
  components: Record<string, string>;
  /** Read from the checkpoint's own safetensors headers rather than estimated
   *  from a parameter count. 0 when nothing is held. */
  bytes_resident: number;
  /** `null` when nothing was loaded, which is not a load that took no time. */
  load_seconds: number | null;
  /** The most words one request may MARK, from the module that enforces it.
   *  NOT a bound on the work: the knockout runs an arm for every word in the
   *  prompt and this list only says which rows the caller asked about. The
   *  panel discloses it before the click rather than letting the route refuse
   *  afterwards. */
  max_knockout_words: number;
  /** The largest data URL `image_input.decode` will accept, in bytes, and the
   *  largest picture it will decode, in pixels.
   *
   *  Published for the same reason as `max_knockout_words` above: both were
   *  enforced at decode time and stated nowhere, so somebody choosing a 40 MB
   *  photo paid the read and the base64 encode before learning the bound
   *  existed. The pixel bound is separate from the byte bound on purpose — a
   *  decompression bomb is a few kilobytes of PNG that expands to gigabytes,
   *  which a bound on the compressed size cannot catch. */
  max_image_bytes: number;
  max_image_pixels: number;
  reason: string;
  means: string;
}

/** One denoising step's cross-attention, already reduced by the server.
 *
 *  `per_token` is attention mass per prompt token, summed over pixels and
 *  averaged over heads AND over the cross-attention blocks that contributed.
 *  The mean over heads is a choice that hides head-level disagreement, and
 *  `blocks` is how many maps went into this row — a step where fewer blocks
 *  reported is a partial capture, visible in the data rather than silent.
 */
export interface ImageStepMap {
  step: number;
  /** The scheduler's own timestep. Carried because "step 12" means nothing
   *  across two schedulers with different step counts. */
  timestep: number;
  per_token: number[];
  blocks: number;
}

export interface ImageAttentionRun {
  tokens: string[];
  steps: ImageStepMap[];
  /** `null` means no seed was fixed, which is NOT seed 0 — another run then
   *  gives another trajectory and nothing downstream is comparable. */
  seed: number | null;
  model: string;
  revision: string;
  /** Where the padding starts. CLIP pads to 77 and the padded tail attracts
   *  real attention mass, which is a genuine finding and a terrible chart, so
   *  it travels as an index rather than as sixty blank columns. */
  padding_from: number;
  steps_requested: number;
  steps_measured: number;
  /** The attention resolutions the maps were averaged over. */
  resolutions: number[];
  /** How wide the denoiser's conditioning actually was, read off the maps —
   *  not the tokenizer's length and not the 77-token label cap. PixArt-Alpha
   *  is 120 and Sigma is 300. */
  conditioning_width: number;
  /** Columns that were MEASURED and have no word to label them, because the
   *  label list is capped. A limit on what can be shown, not on what was
   *  measured — `means` says so in words. */
  columns_unlabelled: number;
  means: string;
}

/** One arm of a knockout: the prompt with one word removed, and how far the
 *  image moved. `distance` is RMS over pixels — arithmetic anybody can check,
 *  rather than one model's opinion of how different two pictures look. */
export interface ImageKnockoutArm {
  word: string;
  index: number;
  prompt_without: string;
  distance: number;
}

export interface ImageKnockout {
  /** Already sorted by `distance`, furthest first. */
  arms: ImageKnockoutArm[];
  seed: number;
  steps: number;
  /** Echoed back: the words the caller asked about. The RUN measures every
   *  word of the prompt in turn regardless — see `imageKnockout`. */
  tokens: string[];
  means: string;
}

/** Renders and passes, before any are spent. `arms` is `words + 1`: every
 *  word plus the unmodified prompt each arm is compared against. */
export interface ImageAttentionCost {
  arms: number;
  steps_each: number;
  passes: number;
  means: string;
}

/** What keeping a latent per step would hold.
 *
 *  Every byte figure is `null` — never 0 — when the latent shape could not be
 *  read off the pipeline. A run whose memory could not be priced is not a run
 *  that costs nothing, and `fits: null` is "unknown", not "no".
 */
export interface ImageTraceCost {
  steps: number;
  denoiser_passes: number;
  vae_decodes: number;
  latents_kept: number;
  latent_bytes: number | null;
  total_bytes: number | null;
  fits: boolean | null;
  threshold: number;
  means: string;
}

/** Plain fetches, all eight of them, for the reason written over `getPolicy`:
 *  the demo and the `.mri` viewer answer through `demo.ts`'s patched fetch, so
 *  `tests/demo_check.py` can see the handler. An answer written here instead
 *  is invisible to that check and is reported as an unhandled endpoint — and
 *  the demo answers `/api/image` with a NOT-LOADED status rather than a
 *  refusal, because a static page holding no pipeline is exactly what
 *  "nothing is loaded" describes. */
export const getImage = () => fetch("/api/image").then((r) => json<ImageStatus>(r));

/** Every image model already on this disk. Downloads nothing. */
export const getImageAvailable = () =>
  fetch("/api/image/available").then((r) => json<ImageAvailable>(r));

/** Hold one pipeline. No default repo: the checkpoint decides which controls
 *  apply, so guessing one would silently decide what the reader is looking at.
 *
 *  `confirm` overrides the refusals that are overridable — a pipeline beside
 *  a resident text model, mainly, which the server refuses first because both
 *  are wanted resident at once and neither can be offloaded to rescue the
 *  other. A refusal that is NOT overridable answers again with the same
 *  sentence, which is the right outcome rather than a silent OOM.
 */
export const loadImage = (repo: string, confirm = false, device = "") =>
  fetch("/api/image/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo: repo.trim(), device, dtype: "", confirm }),
    // `LoadCancelled` rather than an error, because a load somebody stopped
    // is not a failure: the route answers 200 with a sentence and the panel
    // shows it plainly instead of painting a red box over a deliberate act.
  }).then((r) => json<ImageStatus | LoadCancelled>(r));

/** What the in-flight image load is doing.
 *
 *  The same `LoadProgress` shape as `/api/model/progress`, deliberately —
 *  the stages differ (a diffusion pipeline is scanned for live pickle
 *  opcodes; a language model is not) but everything around them is the same
 *  question, and one shape means one `LoadBar` renders both. */
export const getImageProgress = () =>
  fetch("/api/image/progress").then((r) => json<LoadProgress>(r));

/** Stop an in-flight image load.
 *
 *  `means` is the server being honest about the limit of its own button: the
 *  download runs in a child process and dies immediately, but if the pipeline
 *  is already opening, that call cannot be interrupted and the stop lands
 *  when it returns. */
export const cancelImageLoad = () =>
  fetch("/api/image/cancel", { method: "POST" }).then((r) =>
    json<{ stopping: boolean; means: string }>(r),
  );

/** Drop it and hand the memory back, not merely forget it. */
export const unloadImage = () =>
  fetch("/api/image/unload", { method: "POST" }).then((r) => json<ImageStatus>(r));

/** Price the renders before running any.
 *
 *  `words=0` prices ONE render — the capture. `words=n` prices a knockout of
 *  an n-word prompt, which is n arms plus the unmodified one they are each
 *  compared against. Same arithmetic, two questions.
 */
export const imageAttentionCost = (steps: number, words: number) =>
  fetch(`/api/image/attention/cost?steps=${steps}&words=${words}`).then((r) =>
    json<ImageAttentionCost>(r),
  );

/** What keeping a latent per step would hold, priced off the loaded
 *  pipeline's own latent shape when there is one. */
export const imageStepsCost = (steps: number) =>
  fetch(`/api/image/steps/cost?steps=${steps}`).then((r) => json<ImageTraceCost>(r));

/** Which prompt tokens the image attends to, per denoising step.
 *
 *  `seed` is optional and `null` is NOT 0: `null` means the sampler was not
 *  fixed, so another run gives another trajectory. Refuses 409 when nothing is
 *  loaded, and 409 again — with a different sentence — when the loaded family
 *  has no cross-attention to capture.
 */
export const captureImageAttention = (
  prompt: string,
  steps: number,
  seed: number | null,
) =>
  fetch("/api/image/attention", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, steps, seed }),
  }).then((r) => json<ImageAttentionRun>(r));

/** Remove one prompt word at a time and measure what actually moved.
 *
 *  The seed is required rather than optional here, and it is doing the work:
 *  every arm runs at the identical seed, so the difference between two images
 *  is the word rather than the sampler.
 *
 *  `words` is what the caller asks ABOUT and the route refuses an empty list —
 *  which words matter is the question, not an implementation detail. It does
 *  not narrow the run: `image_attention.knockout` splits the prompt itself and
 *  measures every word in turn, and echoes `words` back as `tokens`. The panel
 *  says so rather than implying the picks limited the work.
 */
export const imageKnockout = (
  prompt: string,
  words: string[],
  seed: number,
  steps: number,
) =>
  fetch("/api/image/knockout", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, words, seed, steps }),
  }).then((r) => json<ImageKnockout>(r));

// ------------------------------------------ occlusion attribution (images)
//
// The interventional saliency map, and the reason it is here rather than a
// gradient: covering a region and re-running MEASURES what the model loses,
// where a gradient or an attention weight only correlates with it. Gated on
// the `attribution` capability — ViT, detection and segmentation heads have
// it; a diffusion pipeline does NOT, because there is no class logit to move.
//
// Two routes, in the order they must be called. `cost` needs no model and is
// asked first, because the number it returns is the one that decides whether
// to run at all: the same image at stride 1 rather than stride 16 is not a
// slower run, it is a different afternoon.

/** Windows and passes, before a single one is taken.
 *
 *  `seconds` is `null` unless a per-pass time was measured, and `null` here
 *  is **"nobody measured"** rather than "instant" — this route deliberately
 *  publishes no forecast of its own, because a wait invented from a constant
 *  somebody typed would be a number this tool made up.
 *
 *  `within_ceiling: false` means `POST /api/image/attribution` will REFUSE
 *  this geometry. `estimate` still prices it, on purpose: a caller about to
 *  be refused needs the number that got them refused, or they are left
 *  guessing at the stride.
 */
export interface ImageAttributionCost {
  map_rows: number;
  map_cols: number;
  n_windows: number;
  /** Every window plus the one unoccluded reference run. */
  passes: number;
  forward_calls: number;
  /** The batch actually used, already clamped to what the module allows. */
  batch: number;
  /** What was ASKED for, which is not always what will be used.
   *
   *  Both numbers travel because a silent cap is a defect. `vision_attr.
   *  estimate` has always returned this and the preflight type dropped it on
   *  the way to the browser, so a caller asking for a batch of 200 was priced
   *  at 64 with nothing saying the request had been reduced. The sweep
   *  response reports both (`ImagePanel` draws "The batch was reduced");
   *  only the estimate lost it. The panel's own slider stops at 64, so this
   *  fires for direct API callers rather than from the UI — the type was
   *  wrong either way. */
  batch_requested: number;
  patch: number;
  stride: number;
  /** The occluded copies of the input alone. The activations behind them are
   *  a multiple of it that nothing can know without running the model, so
   *  they are absent here rather than estimated. */
  input_bytes_per_call: number;
  seconds: number | null;
  within_ceiling: boolean;
  ceiling: number;
  means: string;
}

/** Where the occluders went, and at what resolution the map therefore is. */
export interface ImageAttributionGrid {
  /** The dimensions of the tensor the model SAW, after its own processor
   *  resized the picture — not the dimensions of the file that was picked. */
  height: number;
  width: number;
  patch: number;
  stride: number;
  map_rows: number;
  map_cols: number;
  n_windows: number;
  passes: number;
  /** Pixels shared by neighbouring windows, from the stride alone. Positive
   *  means the map's cells are not disjoint regions of the image. */
  overlap: number;
  /** The last row (or column) had to be pulled back to the edge, so it
   *  overlaps its neighbour by more than `patch - stride`. A fact about this
   *  map rather than a defect — without the clamp a strip of the image would
   *  be under no window at all while the map still looked complete. */
  edge_row_clamped: boolean;
  edge_col_clamped: boolean;
}

/** One occluded region, and what covering it did to the target class.
 *
 *  **`logit_drop` is SIGNED.** Positive means covering this window COST the
 *  class evidence. Negative means covering it HELPED — a region that was
 *  arguing against the class, which is a finding rather than an error. An
 *  absolute value prints the same number for both, so nothing here may take
 *  one.
 */
export interface ImageAttributionWindow {
  row: number;
  col: number;
  top: number;
  left: number;
  height: number;
  width: number;
  logit_drop: number;
  /** The same movement in softmax probability, which is a different quantity
   *  and not a better one — it moves when any other class moves. `null` for a
   *  single-output head, where a softmax is 1.0 by construction. */
  prob_drop: number | null;
}

/** One occlusion map, and everything it is not allowed to claim. */
export interface ImageAttribution {
  grid: ImageAttributionGrid;
  /** `map_rows` x `map_cols` of signed logit drops, every cell filled. */
  map: number[][];
  windows: ImageAttributionWindow[];
  /** `grey`, `black`, `white` or `image_mean`. There is no neutral fill: a
   *  flat square is a specific baseline, not removal. */
  fill: string;
  /** What that word was in numbers, per channel where it varies. */
  fill_value: number[];
  value_range: [number, number];
  /** **True means the range was GUESSED from this one image's extremes**
   *  rather than read from the checkpoint's own processor. One picture's
   *  extremes are a lower bound on the model's input range, not the range —
   *  so "grey" landed somewhere that is not necessarily the midpoint, and
   *  the fill that was actually applied is a weaker claim than it looks. */
  value_range_inferred: boolean;
  target: number;
  target_label: string;
  /** Whether the class was the model's own top prediction or one that was
   *  named. Attributing the model's answer and auditing a label you supplied
   *  are different questions with the same picture. */
  target_chosen_by_model: boolean;
  classes: number;
  base_logit: number;
  /** `null` for a single-output head. */
  base_prob: number | null;
  /** **The scale the peak has to be read against**: largest drop minus
   *  smallest, over the whole map. A spread at or below the reported
   *  precision is a map made of rounding, and ranking its windows is ranking
   *  rounding — `means` says so, and a panel must not draw a confident peak
   *  over that sentence. */
  spread: number;
  /** `null` — not the first window — when the map is exactly flat. A model
   *  returning the same logits for every occlusion has said it did not use
   *  the image, and naming a peak there is reading rank order out of a tie. */
  strongest: ImageAttributionWindow | null;
  /** The window that most INCREASED the class when covered. **`null` means
   *  NOTHING argued against the class**, which is a different answer from a
   *  drop of 0.0 and must not render as an empty slot. */
  most_negative: ImageAttributionWindow | null;
  passes: number;
  forward_calls: number;
  /** Both travel because a silent cap is a defect: `batch_used` below
   *  `batch_requested` is a different run time from the one that was asked
   *  for, and somebody will time it. */
  batch_requested: number;
  batch_used: number;
  seconds: number;
  model_name: string;
  /** Class names were supplied and did not match the head's width, so they
   *  were dropped ENTIRELY rather than applied by position. */
  class_names_dropped: boolean;
  means: string;
}

/** Price the sweep before running it. Needs no model — it is arithmetic over
 *  the geometry — so it answers on a resting server too.
 *
 *  `stride` of 0 is the query-string way of saying "not stated", which the
 *  module then reads as the patch size: non-overlapping windows, the cheapest
 *  schedule that covers every pixel exactly once.
 */
export const imageAttributionCost = (
  height: number,
  width: number,
  patch: number,
  stride: number,
  batch: number,
) =>
  fetch(
    `/api/image/attribution/cost?height=${height}&width=${width}` +
      `&patch=${patch}&stride=${stride}&batch=${batch}`,
  ).then((r) => json<ImageAttributionCost>(r));

/** Cover each window of one image, re-run, and report what moved.
 *
 *  The picture travels as a data URL rather than as a path, deliberately: a
 *  path in a request body names a file on the SERVER's disk, which is
 *  somebody else's machine as often as it is yours — and a browser cannot
 *  produce one for a file the user picked anyway.
 *
 *  `target` of `null` asks the ordinary question: attribute whatever the
 *  model itself predicted. Naming a class asks a different one — auditing a
 *  label you supplied — and the result says which of the two it answered.
 */
export const imageAttribution = (
  image: string,
  patch: number,
  stride: number,
  fill: string,
  batch: number,
  target: number | null,
) =>
  fetch("/api/image/attribution", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image, patch, stride, fill, batch, target }),
  }).then((r) => json<ImageAttribution>(r));

// ------------------------------------------------- finding an image model
//
// The four routes over `image_catalog`, and they exist because the image side
// had a cache scan and nothing else: the only way to open a diffusion model
// was to already know its name. The text side has had `hub.search` and
// `discover.scan` for a long time; this is the same pair for pictures.
//
// ## A row here never claims a family
//
// A Hub pipeline tag is a TASK, not an architecture: `text-to-image` covers a
// UNet and a DiT, and those two keep their cross-attention in different
// places. So a row says what the model DOES, names the families that tag is
// CONSISTENT with, and leaves the architecture to `imaging.detect`, which
// reads the checkpoint's own config at load. A confident wrong family word in
// a list is exactly what `ImageStatus.capabilities` exists to prevent.
//
// ## `size_bytes` is `null` for unknown, and it is never 0
//
// `hub.weight_bytes` does arithmetic on the per-dtype parameter counts the
// Hub publishes, and most GGUF and pickle repos publish none. The server
// passes that through as `null` rather than 0 because a picker rendering
// "0.0 GB" for an unknown invites the exact click a size column exists to
// prevent. Every reader of these fields must branch on `null` before it
// formats.

/** One task the Hub publishes that this tool can open. */
export interface ImageTask {
  /** The Hub's own pipeline tag: `text-to-image`, `image-segmentation`, … */
  task: string;
  label: string;
  /** The families `imaging.detect` MIGHT name once a checkpoint of this task
   *  is read. Not a property of any one model — see the note above. */
  families: string[];
  means: string;
}

export interface ImageTasks {
  tasks: ImageTask[];
  /** Which task a search runs when none is chosen. Every tag at once is not a
   *  valid Hub filter, so one is named rather than silently picked. */
  default: string;
  means: string;
}

/** One row of a Hub search: what it does, what it weighs, whether it is here. */
export interface ImageHubModel {
  id: string;
  task: string;
  task_label: string;
  /** What the TASK is consistent with. Never this checkpoint's family. */
  families_possible: string[];
  /** `null` is UNKNOWN, never zero — the listing is sorted by downloads, so
   *  an absent count rendered as 0 sorts as the least popular thing on the
   *  page. `image_catalog._count` has always sent `null`; this declaration
   *  was the half that had not caught up. */
  downloads: number | null;
  likes: number | null;
  /** Its licence has to be accepted, and a token has to be on this machine,
   *  before the weights will move. */
  gated: boolean;
  /** `YYYY-MM-DD`, or `null` when the listing carried no date. */
  updated: string | null;
  /** `null` is UNKNOWN and must never render as a size. */
  size_bytes: number | null;
  /** Answered by looking at this machine's cache, not guessed from the
   *  listing. `null` when the cache could not be walked at all — "nobody
   *  could look" is a different answer from "we looked and it is not here",
   *  and only one of them justifies quoting the reader a download. */
  cached: boolean | null;
  /** A cache entry holding configs and NO weights: an interrupted download.
   *  It looks present to a directory listing and has its entire transfer
   *  still ahead of it, which is why the server separates it. */
  partial?: boolean | null;
}

export interface ImageSearch {
  models: ImageHubModel[];
  /** The task actually searched, which is the default when none was sent. */
  task: string;
  means: string;
}

/** One image model on this disk, with what it actually weighs.
 *
 *  `ImageModelInfo` answers what a cached model IS. This answers what it
 *  COSTS, read off the files rather than the Hub — including the state a
 *  browse list cannot show, which is `complete: false`.
 */
/** One denoising step's latent movement.
 *
 *  `rms_change` and `cumulative` are `null` on the FIRST step: there is no
 *  previous latent for it to have moved from, so the change is unknown rather
 *  than zero. A bar chart that treats them as 0 draws a claim nobody made.
 */
export interface ImageStepRow {
  step: number;
  timestep: number | null;
  rms_change: number | null;
  cumulative: number | null;
  rms_to_final: number | null;
  latent_rms: number | null;
}

/** Where the denoiser committed. NOTHING here was decoded.
 *
 *  `vae_decodes` is 0 and is a checkable claim, not a promise: a decode would
 *  make the answer a property of the VAE as much as of the denoiser, so the
 *  same denoiser behind two decoders would appear to commit at two different
 *  steps. `ImageFilmstripRun` is the one that draws pictures.
 */
export interface ImageStepTrace {
  model: string;
  prompt: string;
  seed: number | null;
  scheduler: string;
  steps_requested: number;
  steps_measured: number;
  threshold: number;
  /** `null` when no step met the threshold — not 0, which would name step 0. */
  commit_step: number | null;
  total_change: number | null;
  vae_decodes: number;
  latent_shape: number[] | null;
  bytes_held: number | null;
  steps: ImageStepRow[];
  means: string;
}

/** One decoded step. */
export interface ImageFrame {
  /** The step this latent was handed over at, NOT its position in the strip.
   *  A gap in these numbers is the whole point. */
  step: number;
  timestep: number | null;
  /** A data URL, or `null` when there are no bytes. Never "" — an empty data
   *  URL is a broken image and looks like a decode that produced black rather
   *  than one that never happened. */
  png: string | null;
  png_bytes: number;
  width: number | null;
  height: number | null;
  decoded_width: number | null;
  decoded_height: number | null;
  /** True when the emitted frame is smaller than what the decoder produced. A
   *  picture silently shrunk is a picture of a resolution the model never
   *  worked at. */
  downsampled: boolean;
  latent_rms: number | null;
}

export interface ImageFilmstripRun {
  model: string;
  prompt: string;
  seed: number | null;
  scheduler: string;
  frames_decoded: number;
  steps_requested: number;
  steps_run: number;
  decoded_steps: number[];
  /** Listed, not implied. Eight frames from a fifty-step run is eight frames,
   *  and a reader must not be able to mistake the strip for the run. */
  skipped_steps: number[];
  steps_never_reached: number[];
  vae_decodes: number;
  frame_pixels: number;
  png_bytes_total: number;
  peak_device_bytes: number | null;
  frames: ImageFrame[];
  means: string;
}

/** What a filmstrip would cost, before any of it runs.
 *
 *  This declared 7 of the 16 keys the route sends. Nine went undeclared, and
 *  most of them are `null` today because the route does not compute them yet
 *  — declared at that real nullability rather than dropped, because `null` is
 *  UNKNOWN here and a panel that renders one as 0 quotes a cost nobody
 *  measured. `json<T>` is a bare cast, so nothing caught the gap;
 *  `tests/test_api_contract.py` does now. */
export interface ImageFilmstripPlan {
  steps: number;
  /** One per step, or two when classifier-free guidance is on — which
   *  nothing here can know without the pipeline, so it is never assumed. */
  denoiser_passes: number;
  frames: number;
  decoded_steps: number[];
  skipped_steps: number[];
  vae_decodes: number;
  /** One more than `vae_decodes` when the pipeline will not accept
   *  `output_type="latent"` and decodes its own final frame anyway. */
  vae_decodes_if_pipeline_also_decodes: number;
  latents_kept: number;
  /** `null` is UNKNOWN — the route cannot size a latent without a pipeline
   *  resident. None of these may ever render as 0. */
  latent_bytes: number | null;
  total_bytes: number | null;
  fits: boolean | null;
  frame_pixels: number;
  png_bytes: number | null;
  peak_device_bytes: number | null;
  selection: {
    mode: string;
    every: number | null;
    at: number[] | null;
    include_final: boolean;
  };
  means: string;
}

export const imageStepsRun = (body: {
  prompt: string;
  steps: number;
  seed: number | null;
  threshold?: number;
}) =>
  fetch("/api/image/steps", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => json<ImageStepTrace>(r));

export const imageFilmstrip = (body: {
  prompt: string;
  steps: number;
  every: number;
  seed: number | null;
}) =>
  fetch("/api/image/filmstrip", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => json<ImageFilmstripRun>(r));

export const imageFilmstripCost = (q: { steps: number; every: number }) =>
  fetch(`/api/image/filmstrip/cost?steps=${q.steps}&every=${q.every}`).then((r) =>
    json<ImageFilmstripPlan>(r),
  );

/** Occlusion over a prediction this model actually made.
 *
 *  Richer than `/api/image/attribution`, which can only attribute a class
 *  logit: this also takes a detector's box (`query`) and a segmenter's mask
 *  area (`region`), and it says WHICH of the model's answers the map is of.
 */
export interface ImageCvAttribution {
  attribution: ImageAttribution | null;
  task: string;
  task_label: string;
  /** "model" when the tool took the top answer, "caller" when you named one.
   *  The difference between explaining the answer given and auditing one you
   *  supplied — and they are different questions. */
  region_chosen_by: string;
  what: string;
  query: number | null;
  region: number[] | null;
  target_label: string;
  map_height: number;
  map_width: number;
  dtype: string;
  names_dropped_by_the_sweep: boolean;
  means: string;
}

export interface ImageCvCost {
  predict: { forward_passes: number };
  readout: Record<string, number | null>;
  attribution: Record<string, unknown>;
}

/** What the three CV measurements cost, before any is spent. */
export const imageCvCost = (
  height: number,
  width: number,
  patch = 16,
  stride: number | null = null,
  batch = 32,
) =>
  fetch(
    `/api/image/cv/cost?height=${height}&width=${width}&patch=${patch}` +
      `${stride === null ? "" : `&stride=${stride}`}&batch=${batch}`,
  ).then((r) => json<ImageCvCost>(r));

export const imageCvAttribute = (body: {
  image: string;
  target?: number | null;
  /** A detector's box slot, from `ImageCvBox.query`. */
  query?: number | null;
  /** A per-pixel segmenter's region of the map, from `ImageCvSegment.bbox` —
   *  (top, left, height, width) in map cells. `CVAttributeRequest` has always
   *  taken it and this signature omitted it, so the segmenter half of the
   *  route was unreachable from the typed client.
   *
   *  Only one of `query` and `region` is ever set: the two heads are
   *  attributed through different reductions and the route refuses the wrong
   *  one by name. */
  region?: number[] | null;
  patch?: number;
  stride?: number | null;
  fill?: string;
  batch?: number;
}) =>
  fetch("/api/image/cv/attribute", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => json<ImageCvAttribution>(r));

/** One class the model scored. */
export interface ImageCvClass {
  index: number;
  /** The checkpoint's own `id2label` entry, or the index as text when it
   *  publishes none. Never a name borrowed from another checkpoint that
   *  happens to have the same number of classes. */
  label: string;
  logit: number;
  probability: number;
}

export interface ImageCvBox {
  /** The head's query slot — the ONLY handle `/api/image/cv/attribute` takes
   *  for a detector box. */
  query: number;
  index: number;
  label: string;
  score: number;
  /** `null` when the scoring convention could not be established. NOT a score
   *  of zero — see `ImageCvPrediction.scoring_reason`. */
  logit: number | null;
  /** `box: number[]` used to be declared here and is in no response this
   *  server can produce: `Box.to_dict` emits these two. Harmless only because
   *  nothing read it — a trap for the next writer rather than a live bug. */
  box_xyxy: number[];
  box_cxcywh: number[];
}

/** One label present in a mask, and how much of the picture it claims. */
export interface ImageCvSegment {
  index: number;
  label: string;
  cells: number;
  /** Of the MAP's cells, not the image's pixels — the map is coarser. */
  fraction: number;
  /** How decisively this label won its cells. The QUANTITY differs by head, so
   *  never read it without `ImageCvPrediction.margin_kind`: a per-pixel head's
   *  margin is the gap to the runner-up class, a mask head's is how far past
   *  the threshold its mask sat. `null` when it could not be computed. */
  mean_margin: number | null;
  /** (top, left, height, width) in map cells — the handle `attribute` takes
   *  as `region`. */
  bbox: number[];
  /** The query slot for a mask-query head, `null` for a per-pixel one. The two
   *  are attributed through different reductions. */
  query: number | null;
}

export interface ImageCvPrediction {
  task: string;
  task_label: string;
  model_name: string;
  dtype: string;
  height: number;
  width: number;
  classes: number;
  /** False when the checkpoint published no `id2label`. The labels are then
   *  indices, and `labels_note` says so in the server's own words. */
  labels_read: boolean;
  labels_published: number | null;
  labels_note: string;
  classes_top: ImageCvClass[];
  boxes?: ImageCvBox[];
  /** What a SEGMENTER says. `classes_top` is only ever filled by the
   *  classification path, so a segmenter left it empty — and this interface
   *  declared 12 of the 25 keys `Prediction.to_dict` sends, omitting every
   *  segmentation field. The panel rendered a header, an empty list, and
   *  "Click a class to see what supports it" with nothing clickable: no
   *  error, no refusal, an honest-looking answer of nothing.
   *
   *  `tsc --noEmit` cannot catch that — an interface narrower than the JSON is
   *  legal TypeScript. */
  segments: ImageCvSegment[];
  /** How many segments the model produced, before `MAX_SEGMENTS` truncated
   *  the list above. The cap disclosure. */
  segments_total: number;
  /** The per-cell winning label, as a grid. */
  label_map: number[][];
  map_height: number;
  map_width: number;
  /** How many image pixels one map cell covers. */
  map_stride: number;
  /** `null` for a head with no threshold. */
  mask_threshold: number | null;
  /** How the scores above were arrived at, and why — read from the checkpoint
   *  where it says, derived where it does not. */
  scoring: string;
  scoring_reason: string;
  /** Which quantity `mean_margin` is. Never read a margin without it. */
  margin_kind: string;
  queries_total: number;
  top_k_requested: number;
  forward_passes: number;
  seconds: number;
  means: string;
}

export interface ImageCvLayer {
  layer: number;
  rows: number;
  cols: number;
  values: number[][];
}

export interface ImageCvReadout {
  /** "attention" when there was something to read. Anything else means this
   *  architecture has none — a convolutional backbone has no per-layer
   *  attention — and `reason` says which. */
  kind: string;
  reason: string;
  model_name: string;
  dtype: string;
  layers: ImageCvLayer[];
  n_layers: number | null;
  heads: number | null;
  grid_rows: number | null;
  grid_cols: number | null;
  means: string;
}

export const imageCvPredict = (body: { image: string; top_k: number }) =>
  fetch("/api/image/cv/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => json<ImageCvPrediction>(r));

/** Just the picture. `top_k` used to travel here and the route discarded it —
 *  `layer_readout` returns per-layer maps, not a class ranking, so there is
 *  nothing for it to cut. Sending it now is a 422 naming the field. */
export const imageCvReadout = (body: { image: string }) =>
  fetch("/api/image/cv/readout", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => json<ImageCvReadout>(r));

/** One module a LoRA targets, and how far it moves it. */
export interface AdapterModule {
  name: string;
  component: string;
  role: string;
  rank: number | null;
  /** `alpha / rank`. `null` when the adapter published neither, in which case
   *  `delta_norm` is UNSCALED and not comparable with the scaled rows. */
  scale: number | null;
  scaled: boolean;
  /** Frobenius norm of the delta. A MAGNITUDE, never an effect: a large move
   *  in a layer the sampler barely exercises can matter less than a small one
   *  in a layer it leans on. */
  delta_norm: number;
  /** `||dW|| / ||W||`, or `null` when the base model was not resident. Never
   *  approximated — the denominator is the point of the ratio. */
  relative: number | null;
}

export interface AdapterGroup {
  component: string;
  role: string;
  modules: number;
  delta_norm: number;
}

export interface AdapterReport {
  path: string;
  /** A LIST: an adapter may mix ranks per module, and one number would be
   *  picking which. */
  ranks: number[];
  modules_total: number;
  modules_listed: number;
  components: string[];
  roles: string[];
  groups: AdapterGroup[];
  top: AdapterModule[];
  all_scaled: boolean;
  base_model: string | null;
  notes: string[];
  means: string;
}

/** Read a LoRA on THIS machine. A path, not an upload: the file is already on
 *  the server's disk and pushing 400 MB through the page to read a header
 *  would be absurd. The route refuses requests that did not come from here. */
export const readAdapter = (path: string, top = 40) =>
  fetch("/api/image/adapter", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, top }),
  }).then((r) => json<AdapterReport>(r));

/** One weighted part of a pipeline, priced from the files it actually has. */
export interface ImageFitComponent {
  name: string;
  /** The variant chosen for pricing — `""` for the unsuffixed files. */
  variant: string;
  /** Every variant on disk, so a reader can see what was NOT chosen. */
  variants: string[];
  files: number;
  disk_bytes: number;
  /** `null` is UNKNOWN, never 0 — 0 would read as "this part is free". */
  card_bytes: number | null;
  /** Priced from a tensor table (`true`) or from file sizes (`false`). */
  exact: boolean;
  note: string;
}

/** Whether one image model will actually run on the card in this machine.
 *
 *  `size_bytes` on the row beside this is what the checkpoint weighs ON DISK,
 *  which is a different number from what it costs once loaded and says nothing
 *  at all about whether the load can succeed. An F32 checkpoint loaded bf16
 *  allocates half its file size; `stabilityai/sd-turbo` keeps two copies of
 *  its VAE in one folder; and a component holding a config with no weights is
 *  a model listed as ready that fails at the click. All three are answered
 *  here and none of them are visible in the disk figure.
 */
export interface ImageFit {
  path: string;
  components: ImageFitComponent[];
  /** Components on disk deliberately not counted, with the reason. Rendered
   *  rather than dropped: a total that silently omits a folder somebody can
   *  see on their own disk reads as an arithmetic error. */
  excluded: string[];
  /** Declared components whose folder is here and CONTRADICTS ITSELF — a
   *  config with no weights, or weights with no config. Non-empty means the
   *  load FAILS, however comfortably the sizes fit. */
  missing: string[];
  /** Declared components whose folder is not here at all. Reported, never
   *  blocking: a component can be handed to `from_pretrained` directly, and
   *  the server cannot see that from the files on disk. */
  absent: string[];
  /** Components whose directory could not be listed at all — a permission,
   *  an ACL, or a sync client's virtual filesystem.
   *
   *  Neither counted nor assumed absent. With one of these `card_bytes` is
   *  `null`, because a total that silently omits a component is not a total.
   *  This used to read as "empty", which flipped `loadable` to false, halved
   *  the published size, and told the reader to re-download a model that was
   *  sitting there complete. */
  unreadable: string[];
  disk_bytes: number;
  /** Resident weight bytes at `dtype`. `null` when any component could not be
   *  priced — a total missing one part is not a total. */
  card_bytes: number | null;
  dtype: string;
  device: string;
  device_name: string;
  /** `null` is UNKNOWN. Apple's unified memory reports no free figure, and 0
   *  free would say the machine is out of memory when nobody asked it. */
  free_bytes: number | null;
  total_bytes: number | null;
  headroom_bytes: number | null;
  /** `"fits"` | `"tight"` | `"over"` | `"unknown"`, and `"unknown"` is a real
   *  answer rather than a cheerful default. */
  verdict: string;
  /** The variant `from_pretrained` must be given, `""` when the plain files
   *  are complete, and `null` when no variant covers every component — which
   *  means the checkpoint cannot be loaded as it stands. */
  variant: string | null;
  loadable: boolean;
  exact: boolean;
  /** The room left for latents and attention maps that the verdict used.
   *  REPORTED so a reader can check the arithmetic rather than taking a
   *  threshold on faith. */
  activation_headroom: number;
  reason: string;
  means: string;
}

export interface ImageLocalModel {
  /** The repo id, and the string `loadImage` takes. */
  path: string;
  family: string;
  label: string;
  known: boolean;
  architecture: string;
  capabilities: string[];
  reason: string;
  /** `null` is UNKNOWN. A cache entry that could not be sized is still worth
   *  listing, and it is not a model that weighs nothing. */
  size_bytes: number | null;
  /** Three states, and `!complete` is the wrong test for all of them.
   *
   *  `true` — the weights are on this disk.
   *  `false` — they are not: an interrupted download, and offering a Load
   *  button on one is offering a click that cannot work.
   *  `null` — the entry could not be sized at all, so neither claim is
   *  available. Reporting that as `false` sends somebody to re-download a
   *  model they may already have. */
  complete: boolean | null;
  /** Will it run HERE. `null` when the row could not be priced at all — the
   *  model still lists, because dropping it would hide a checkpoint the
   *  reader can see on their own disk. */
  fit: ImageFit | null;
}

export interface ImageLocal {
  models: ImageLocalModel[];
  /** Summed over the COMPLETE rows only: an interrupted download's bytes are
   *  not a model on this disk. */
  bytes_on_disk: number;
  /** How many entries could not be sized at all — a third state, not zero
   *  bytes. Carried in `means` as a sentence; here as the fact. */
  unsized: number;
  /** The cache walk stopped at `scan_limit`. `/api/image/available` has
   *  reported this on the same walk for months and this route did not — and
   *  this is the one the panel renders. A list that silently stops at 200
   *  reads as "everything on this disk", which is the one thing it is not. */
  truncated: boolean;
  scan_limit: number;
  means: string;
}

/** What downloading one model would cost, before any of it moves. */
export interface ImageSize {
  id: string;
  size_bytes: number | null;
  gated: boolean;
  /** `null` when the local cache could not be walked at all. "We looked and
   *  it is not here" and "nobody could look" are different answers, and only
   *  one of them justifies telling somebody to spend the download. */
  cached: boolean | null;
  partial: boolean | null;
  /** Whether the walk ran. `means` already says so in words; this is here so
   *  nothing branches on `cached === false` and gets it wrong. */
  cache_readable: boolean;
  /** The entry IS here and could not be measured — a permission error, an
   *  ACL, a sync client's virtual filesystem.
   *
   *  A third state beside `cached` and `partial`, and both of those are
   *  `null` when this is true. It used to be filed as `partial: true`, so the
   *  panel stated as fact that the repo "has a cache entry on this machine but
   *  NO WEIGHTS in it — an interrupted download", from a permission error, and
   *  sent the reader to re-download something that may be complete. */
  cache_unsized: boolean;
  means: string;
}

/** Which kinds of image model can be searched for. Reads no disk and makes no
 *  Hub call — it is `image_catalog.TASKS`, so a task added there appears here
 *  without a second edit. */
export const getImageTasks = () =>
  fetch("/api/image/tasks").then((r) => json<ImageTasks>(r));

/** Image models on the Hub, annotated with size and whether they are here.
 *
 *  Downloads nothing. Refuses 422 for a task outside `image_catalog.TASKS` —
 *  which would return checkpoints nothing here can load — and 503 when the
 *  Hub cannot be reached, with the sentence that says the models already on
 *  this machine still open. Both arrive as `ApiError`, so both go through
 *  `errorText` and reach the reader in the server's own words.
 */
export const searchImageModels = (q: string, task: string, limit = 24) =>
  fetch(
    `/api/image/search?q=${encodeURIComponent(q)}&task=${encodeURIComponent(
      task,
    )}&limit=${limit}`,
  ).then((r) => json<ImageSearch>(r));

/** Every image model on this disk, with what it weighs and whether it
 *  finished downloading. Reads files; asks the Hub nothing. */
export const getImageLocal = () =>
  fetch("/api/image/local").then((r) => json<ImageLocal>(r));

/** One image model found in an ORDINARY FOLDER rather than in the Hub cache.
 *
 *  The same fields `ImageLocalModel` carries and one different meaning: `path`
 *  is a DIRECTORY on this machine, not a repo id. `loadImage` takes either, so
 *  a row from here loads the same way a cached one does — but there is no repo
 *  to re-fetch it from, which is why the two lists are reported separately
 *  rather than merged into one.
 */
export interface ImageDiscoveredModel {
  /** The directory the weights were read from, and the string `loadImage`
   *  takes for it. */
  path: string;
  family: string;
  label: string;
  known: boolean;
  architecture: string;
  capabilities: string[];
  reason: string;
  /** `null` is UNKNOWN — see `ImageLocalModel.size_bytes`. Never render it as
   *  a size. */
  size_bytes: number | null;
  /** The same three states as `ImageLocalModel.complete`, and `!complete` is
   *  the wrong test for all of them. */
  complete: boolean | null;
}

export interface ImageDiscovered {
  models: ImageDiscoveredModel[];
  /** Every directory that was actually walked. RETURNED rather than assumed,
   *  and it is the half of the answer that makes an empty list usable: "found
   *  nothing" without "and here is where I looked" tells somebody their model
   *  is missing when the truth may be that the directory holding it was never
   *  searched. */
  roots: string[];
  /** The walk hit its budget, so the list is what was reached rather than
   *  everything there is. A truncation nobody is told about reads as "this is
   *  all there is". */
  truncated: boolean;
  /** The budget `truncated` refers to. The walk's OWN limit, which is not the
   *  cache walk's — they are separate numbers and quoting one for the other
   *  states a wrong figure with full confidence. */
  scan_limit: number;
  means: string;
}

/** Image models in ordinary folders — the running directory included.
 *
 *  `getImageLocal` reads the Hub cache. This walks the same roots the text
 *  picker walks, so a checkpoint cloned into the working directory turns up
 *  here rather than nowhere. Reads files; asks the Hub nothing.
 */
export const getImageDiscovered = () =>
  fetch("/api/image/discovered").then((r) => json<ImageDiscovered>(r));

/** How big one named repo is, before a byte of it moves.
 *
 *  The question a reader asks first, and the one the name box could not
 *  answer: whether it FITS is separate, and is answered against this
 *  machine's free memory when you load it.
 */
export const imageSize = (repo: string) =>
  fetch(`/api/image/size?repo=${encodeURIComponent(repo)}`).then((r) =>
    json<ImageSize>(r),
  );

export interface HubAuth {
  signed_in: boolean;
  user: string | null;
  source: string | null;
}

export interface HubModel {
  id: string;
  /** `null` is UNKNOWN, never zero. With the Hub unreachable the curated
   *  rows still carry their names and nothing else — a count of 0 there
   *  would be a measurement of popularity nobody took, and it read
   *  identically to a real repo nobody has downloaded. */
  downloads: number | null;
  likes: number | null;
  gated: boolean;
  usable: boolean;
  /** `YYYY-MM-DD`, or `null` when the listing carried no date or was never
   *  reached. `""` used to mean both. */
  updated: string | null;
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
  /** Why the daemon is unreachable, when the answer is not "start it".
   *
   *  Optional because the live route never sends one — a machine with no
   *  Ollama running has the ordinary next step, which the picker already
   *  states. The static demo DOES send one, and it went undeclared and
   *  unrendered: a visitor was told to install Ollama and reopen the panel,
   *  on a page where neither step can change anything. */
  reason?: string;
}

/** A curated Ollama model, sized live and marked against this GPU. */
export interface OllamaSuggestion {
  name: string;
  size_gb: number;
  /** null when the size or the GPU is unknown — "unknown" is not "fits". */
  fits: boolean | null;
}

/** Somewhere to start on the Ollama tab, the way the HuggingFace tab opens on
 *  curated picks. Ollama has no search API, so this is a starting point, not
 *  a substitute for one — the name box reaches every model, this reaches the
 *  eight most people want first. */
export const getOllamaSuggested = () =>
  fetch("/api/ollama/suggested").then((r) => json<OllamaSuggestion[]>(r));

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
  /** `null` when the volume could not be read. NOT 0 — 0 on the wire says the
   *  disk is full, which is the one reading that would stop a download the
   *  tool did not mean to stop. */
  free_bytes: number | null;
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
  /** Absent before the disk is consulted; `null` when it was and could not be
   *  read. Two different unknowns, and neither is 0. */
  free_bytes?: number | null;
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

// ------------------------------------- the corpora this machine can read

/** One .txt or .jsonl the server walked to, named by an id the SERVER minted.
 *
 *  `root` and `relative` rather than one absolute path, and that split is
 *  deliberate on the server's side: an absolute path is a fact about the
 *  machine and this list is rendered in a browser, while two files sharing a
 *  name still have to be told apart. The pair does both.
 */
export interface AvailableCorpus {
  id: string;
  name: string;
  /** Which corpus root it was found under. */
  root: string;
  /** Where it sits below that root. */
  relative: string;
  bytes: number;
}

/** What `GET /api/corpus/available` found, and every bound it stopped on.
 *
 *  NOTHING HERE IS NULLABLE, and that is a property of the route rather than
 *  an omission: `corpus_index.Listing.to_dict` computes every one of these
 *  fields on every call, including the two `truncated_*` flags, which are
 *  `False` when the walk finished rather than absent. So there is no field on
 *  this payload where "unknown" and 0 could be confused — the one place that
 *  distinction has to be defended is the RENDER, where a listing that failed
 *  to arrive must not read as a machine with no corpora on it.
 */
export interface AvailableCorpora {
  corpora: AvailableCorpus[];
  /** The directories the walk started from — the answer to "why is my file
   *  not in this list", so it is shown rather than kept for the sentence. */
  roots: string[];
  n_found: number;
  n_dirs_read: number;
  /** True when the listing stopped on its file cap. `n_found` is then the cap
   *  and NOT how many corpora this disk holds; a file missing from the list is
   *  not a file missing from the machine, and the panel has to say so or a
   *  reader concludes their corpus is gone. */
  truncated_files: boolean;
  /** True when the walk stopped after reading `n_dirs_read` directories, so
   *  part of the tree was never looked at at all. */
  truncated_dirs: boolean;
  max_depth: number;
  /** The suffixes that are listed — `.txt` and `.jsonl`, the two
   *  `load_prompts` reads. A `.csv` on disk is absent for a reason. */
  suffixes: string[];
  means: string;
}

/** List the corpora on this machine, so one can be picked instead of typed.
 *
 *  Refused in the static builds like every other route that reads this
 *  machine's disk: there is no filesystem behind a page on GitHub Pages, and
 *  a 404 inside the picker would read as "the corpus list is broken" rather
 *  than "there is nothing here to list". The typed field beside the picker is
 *  untouched by this — the sweep routes take either an id from this listing or
 *  a path, and each says for itself what it cannot do.
 */
export const getAvailableCorpora = () =>
  DEMO || VIEWER
    ? noModelHere(
        "Listing the corpora on this machine means reading its directories — " +
          "up to four levels below each corpus root — and a browser cannot " +
          "see a filesystem.",
      )
    : fetch("/api/corpus/available").then((r) => json<AvailableCorpora>(r));

export const getOllama = () => fetch("/api/ollama").then((r) => json<OllamaState>(r));

export interface TraceSummary {
  /** Scripted sample data, not a run you recorded. */
  demo?: boolean;
  /**
   * Where the run came from. `"app"` is a generation made in this page;
   * `""` (or absent) is a run of your own code, posted by modelmri-record.
   * Both belong in the panel, and they are not the same thing.
   */
  source?: string;
  id: string;
  name: string;
  started_at: string;
  n_steps: number;
  /** A FLOOR when `n_timed < n_steps`, not a total: the store sums
   *  `started_ms + COALESCE(duration_ms, 0)`, so a step whose duration was
   *  never recorded contributes nothing to it. */
  total_ms: number;
  /** How many steps carry a duration. `duration_ms` is nullable on purpose —
   *  "not recorded" and "took no measurable time" are different facts — and
   *  without this the panel cannot tell a real 0.0s from a run nobody timed. */
  n_timed: number;
  n_errors: number;
}

export interface TraceStep {
  id: string;
  parent_id: string | null;
  /** Every kind `modelmri/step_kinds.py` accepts, and it has to be all of
   *  them: nothing in this file fails to compile when the union is short — no
   *  exhaustive switch reads it — so a stale one is an under-specified type
   *  that keeps working while quietly making four real kinds unrepresentable.
   *  `tests/test_step_kinds.py` parses this line against `VALID_KINDS`. */
  kind:
    | "llm_call"
    | "tool_call"
    | "subagent"
    | "mcp_call"
    | "user_turn"
    | "error"
    | "retrieval"
    | "embedding"
    | "rerank"
    | "guardrail";
  name: string;
  started_ms: number;
  /** Null when the step was recorded without one. Not 0 — "not recorded" and
   *  "took no measurable time" are different facts, and the column used to
   *  flatten them together. */
  duration_ms: number | null;
  input: string;
  output: string;
  /** Characters `traces._clip` did not store. 0 when nothing was cut. Shown
   *  as a marker, because a truncated tool output that reads as a complete
   *  one is how you debug the wrong thing for an hour. */
  truncated_in?: number;
  truncated_out?: number;
  tokens_in: number | null;
  tokens_out: number | null;
  /** Providers report these only sometimes — Anthropic returns cache counts
   *  only when a cache was in play, reasoning tokens only from models that
   *  reason. **null is "the provider said nothing", 0 is "it said zero"**, and
   *  the panel must never print one as the other. */
  tokens_cache_read: number | null;
  tokens_cache_write: number | null;
  tokens_reasoning: number | null;
  error: boolean;
  seq: number;
  /** True when this step was produced by a model on THIS machine and carries
   *  the token ids needed to reopen it in the mechanistic panels. False for a
   *  hosted-API call — the weights are not here, and that is a sentence to
   *  print rather than a button to disable. */
  adoptable?: boolean;
  /** Machine facts the recorder captured: model, ids, dtype, device. Never
   *  prompt or completion text — redaction covers input/output only. */
  meta?: Record<string, unknown>;
}

/** One token field summed over a set of steps.
 *
 *  `total` is **null when nothing in the set reported it** — not 0. `reported`
 *  and `silent` are both carried so "3 of 11 calls said nothing" is answerable
 *  without re-walking the steps. */
export interface TokenCount {
  field: string;
  total: number | null;
  reported: number;
  silent: number;
}

export interface TokenRollup {
  counts: Record<string, TokenCount>;
  n_steps: number;
  n_llm_steps: number;
}

/** What a run cost, and how much of it is actually known.
 *
 *  There is no bundled price map: a map goes stale between releases and a user
 *  on a six-month-old install would see six-month-old prices with no way to
 *  know. Cost appears only when `MODELMRI_PRICES` points at the user's own
 *  file, matched by EXACT model id — never a prefix or a regex, because a
 *  regex matching the wrong model produces a plausible figure with no signal
 *  it is wrong. */
export interface TraceCost {
  total: number | null;
  currency: string;
  n_calls: number;
  n_priced: number;
  unpriced_models: string[];
  partial: boolean;
  means: string;
  /** Set when the price file itself could not be read. The token counts are
   *  complete and useful without it, so this is a field rather than a failed
   *  request. */
  error?: string;
}

/** One exact predicate over recorded runs. No model, so nothing is a verdict.
 *
 *  A rule says a run MATCHED. It does not say the run was good, bad or
 *  wasteful — the name is the reader's own words, and appears as theirs. */
export interface RubricRule {
  name: string;
  kind: string;
  pattern?: string;
  step_kind?: string;
  op?: string;
  value?: number;
  means?: string;
}

export interface RubricRow {
  trace_id: string;
  name: string;
  matched: string[];
  hits: { rule: string; matched: boolean; detail: string; step_ids: string[] }[];
  /** ISO 8601, formatted in the reader's own timezone rather than the
   *  server's — the server has no idea what that is. */
  started_at: string;
  /** `null` when the store has no duration for this run, which is not the
   *  same as a run that took no time. */
  total_ms: number | null;
  n_steps: number;
  n_errors: number;
  /** "app" for a generation made in the playground, "" for a trace written
   *  before the store carried the key. A playground generation and a run of
   *  your own agent code both belong in this list and are not the same
   *  thing, so the row says which. */
  source: string;
  /** Scripted sample data. It must never be indistinguishable from a run you
   *  actually recorded. */
  demo: boolean;
}

export interface RubricReport {
  rows: RubricRow[];
  n_traces: number;
  n_traces_available: number;
  truncated: number;
  rules: RubricRule[];
  /** Rules that could not be ANSWERED, keyed by name, with why. A
   *  distribution rule below its minimum n lands here rather than reporting
   *  "no matches" — which reads identically to having looked. */
  skipped: Record<string, string>;
  counts: Record<string, number>;
  means: string;
}

export const scoreRubric = (rules: RubricRule[]) =>
  DEMO || VIEWER
    ? noModelHere(
        "Scoring a rubric runs its predicates over a live generation.",
      )
    : fetch("/api/rubric/score", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rules }),
  }).then((r) => json<RubricReport>(r));

/** One UK AISI Inspect `.eval` sample, read onto this timeline.
 *
 *  Reader only, and version-gated: an unrecognised schema version is refused
 *  with the version named rather than guessed at. `mapping.dropped` counts
 *  every event kind that has no step kind here — Inspect's schema is not
 *  frozen, and "we showed you what we understood" is only honest when the
 *  rest is on screen too. */
export interface InspectImport {
  trace_id: string;
  trace: TraceDoc;
  mapping: {
    mapped: Record<string, number>;
    dropped: Record<string, number>;
    means: string;
  };
  scores: Record<string, string | number | boolean>;
  failed: boolean;
  error: string;
  header: {
    version: number;
    task: string;
    model: string;
    created: string;
    status: string;
    n_samples: number;
  };
  /** The samples LISTED, capped server-side. Use `samples_total` for how many
   *  the log carries — this array is what the picker can offer, not a count
   *  of the reader's file. */
  samples: { name: string; id: string; epoch: number }[];
  /** Every sample in the archive, counted before the cap.
   *
   *  `samples.length` was being printed as this, so a 6,000-sample log
   *  rendered "5000 samples" as a fact about the reader's own archive, while
   *  the picker held an arbitrary subset in zip order and a later sample was
   *  simply unselectable with nothing saying why. */
  samples_total: number;
  /** Whether the list above is short of `samples_total`. */
  samples_truncated: boolean;
  means: string;
}

export const importInspect = (file: File, sampleId = "") =>
  fetch(
    `/api/traces/import/inspect${sampleId ? `?sample_id=${encodeURIComponent(sampleId)}` : ""}`,
    { method: "POST", body: file },
  ).then((r) => json<InspectImport>(r));

/** What a shared bundle would contain — fetched BEFORE anything is written.
 *
 *  This is the one path in the project where data leaves the machine, so the
 *  panel shows what is in the file before offering to save it. A share button
 *  that ships a file without showing its contents asks the user to trust a
 *  process they cannot see. */
export interface BundlePreview {
  n_steps: number;
  n_steps_dropped: number;
  n_payloads_clipped: number;
  chars_clipped: number;
  redactions: { label: string; count: number }[];
  n_redactions: number;
  fields_scanned: number;
  means: string;
}

export const bundlePreview = (traceId: string) =>
  fetch(`/api/traces/${traceId}/bundle/preview`).then((r) =>
    json<BundlePreview>(r),
  );

export interface TraceDoc {
  id: string;
  name: string;
  started_at: string;
  steps: TraceStep[];
  /** Rolled up over the whole run, and per step over its own subtree. */
  tokens?: TokenRollup;
  tokens_by_step?: Record<string, TokenRollup>;
  cost?: TraceCost;
}

export const getTraces = () => fetch("/api/traces").then((r) => json<TraceSummary[]>(r));
export const getTrace = (id: string) =>
  fetch(`/api/traces/${id}`).then((r) => json<TraceDoc>(r));

/** The agent run carried INSIDE the open `.mri`, rather than one in the store.
 *
 *  Two sources answer "what runs can I look at", and they are not the same
 *  set: `getTraces` lists what this machine recorded or imported, and this is
 *  what arrived in the file somebody sent. A carried run is read, never
 *  written to the store — importing it would file a stranger's run in this
 *  machine's history as though it had been captured here.
 *
 *  `available: false` is the ordinary answer. Most sessions carry no run.
 */
export interface SessionTraceAbsent {
  available: false;
}

export interface SessionTraceCarried extends TraceDoc {
  available: true;
  /** What the sender's run held, against `steps.length` here. */
  n_steps_total: number;
  /** How many steps the file dropped to fit its cap. */
  truncated: number;
  /** The step the bundle was built around, if it names one. */
  step_ref?: string | null;
}

/** TWO SHAPES, and the docstring above already said so — "`available: false`
 *  is the ordinary answer" — while the type declared `id`, `name`,
 *  `started_at`, `steps`, `n_steps_total` and `truncated` as always present.
 *  MEASURED with no `.mri` open, the route sends `{available: false}` and
 *  nothing else. `json<T>` is a bare cast, so the declaration and the comment
 *  contradicted each other with nothing to notice. */
export type SessionTraceDoc = SessionTraceAbsent | SessionTraceCarried;

export const getSessionTrace = () =>
  fetch("/api/session/trace").then((r) => json<SessionTraceDoc>(r));

/** Nothing has been generated yet, so there is nothing to report.
 *
 *  The route sends exactly these two keys in this state — measured — while
 *  `TelemetryReport` used to declare seventeen more as required on one flat
 *  interface. `json<T>` is a bare cast, so nothing complained and every
 *  reader of a measurement field was reading `undefined` typed as a number.
 *
 *  A union rather than seventeen optional fields, because the two states are
 *  genuinely different documents and `available` says which. Absent, not
 *  null: `null` would say a measurement was attempted and failed, which is
 *  what the `| null` fields inside the measured shape mean. */
export interface TelemetryUnavailable {
  available: false;
  reason?: string;
}

/** What the last generation actually cost, including what watching it cost.
 *
 *  Every field may be null and none is ever faked — CPU has no allocator to
 *  ask, and a 0 in a memory column is a claim that nothing was used. */
export interface TelemetryMeasured {
  available: true;
  reason?: string;
  prompt_tokens: number;
  generated_tokens: number;
  prompt_ms: number | null;
  decode_ms: number | null;
  tokens_per_s: number | null;
  peak_bytes: number | null;
  reserved_bytes: number | null;
  memory_note: string;
  context_used: number;
  context_limit: number | null;
  context_fraction: number | null;
  /** `n_layers x n_heads x S^2 x dtype` — the attention scores ModelMRI asks
   *  for and a plain runner never allocates. */
  introspection_bytes: number | null;
  introspection_note: string;
  device: string;
  dtype: string;
  notes: string[];
  means: string;
}

export type TelemetryReport = TelemetryUnavailable | TelemetryMeasured;

export const getTelemetry = () =>
  fetch("/api/telemetry").then((r) => json<TelemetryReport>(r));

/** One step that matched a search, with the run it belongs to. */
export interface SearchHit {
  step_id: string;
  trace_id: string;
  trace_name: string;
  kind: TraceStep["kind"];
  name: string;
  started_ms: number;
  duration_ms: number | null;
  input: string;
  output: string;
  truncated_by: number;
  error: boolean;
  seq: number;
  /** The run's wall clock. `started_ms` is milliseconds since that run's own
   *  start, so it cannot order hits across runs — results are sorted by this. */
  trace_started_at: string;
}

export interface SearchResult {
  /** Which engine answered. FTS5 matches whole words; the substring scan
   *  matches inside them and gets slower as the store grows. Named because a
   *  feature that quietly becomes a different feature is worse than one that
   *  says it degraded. */
  engine: "fts5" | "substring-scan";
  results: SearchHit[];
  note: string;
  query: Record<string, unknown>;
}

/** Search every recorded step on this machine.
 *
 *  Free text plus filters — `kind:tool_call`, `error:true`, `duration>2000`,
 *  `name:pytest`. A filter binds only with no space after the colon, so a
 *  pasted log line like "error: connection refused" stays plain text. */
export const searchTraces = (q: string) =>
  fetch(`/api/traces/search?q=${encodeURIComponent(q)}`).then((r) =>
    json<SearchResult>(r),
  );

/** What `adopt` gives back once the panels are pointed at a recorded step. */
export interface Adopted {
  adopted: true;
  model: string;
  step_id: string;
  kind: string;
  n_tokens: number;
  n_prompt_tokens: number;
  prompt: string;
  generation: string;
  means: string;
}

/** Point every mechanistic panel at the generation this agent step made.
 *
 *  Nothing is re-run: these are the recorded token ids, verified against the
 *  loaded tokenizer. 409 when the weights are not on this machine, when the
 *  wrong model is loaded, or when re-tokenising disagrees with the recording. */
export const adoptStep = (traceId: string, stepId: string) =>
  fetch(`/api/traces/${traceId}/steps/${stepId}/adopt`, { method: "POST" }).then(
    (r) => json<Adopted>(r),
  );

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
      const res = await fetch("/api/model/prompt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      // The demo refuses a prompt it did not record, and the sentence saying
      // so is the entire point of the refusal. Destructuring straight off
      // `r.json()` took `generation` from a `{error}` body, got undefined,
      // and streamed the literal word "undefined" into the output panel as
      // the model's answer — then called onDone, so every panel below
      // refreshed as though a real generation had happened. Exactly the
      // confusion the refusal was written to prevent.
      if (!res.ok) {
        const message = errorText(new ApiError(res.status, await res.text()));
        if (!cancelled) h.onError(message);
        return;
      }
      const { generation } = await res.json();
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
  /** Which invocation this row is, when a module runs more than once in one
   *  forward pass — a shared encoder applied to two inputs fires every leaf
   *  twice. `1 of 1` for the ordinary case. */
  call?: number;
  calls_total?: number;
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
  /** `null` until something is loaded — nothing has been counted, and 0 is
   *  a count. */
  n_params: number | null;
  n_trainable: number | null;
  n_modules: number | null;
  input_shape: number[] | null;
  input_origin: string;
  input_reason: string;
  labels: string[] | null;
  reason: string;
  roots: string[];
}

/** One tensor from a GGUF's own table. */
export interface GgufTensor {
  name: string;
  ggml_type: number;
  type_name: string;
  dims: number[];
  elements: number;
  /** Null when the ggml type is one the reader does not know. Not 0 — an
   *  unknown type has an unknown size, and a guess corrupts every roll-up. */
  bytes: number | null;
  bpw: number | null;
  offset: number;
}

export interface GgufSummary {
  architecture: string | null;
  name: string | null;
  quantisation_label: string | null;
  /** Exact regardless of quantisation: element counts come from the tensor
   *  shapes and do not depend on the ggml type. */
  parameters: number;
  measured_parameters: number;
  /** Null when any tensor could not be sized — a byte total over the parts
   *  that happened to be recognised is not the file's byte total. */
  tensor_bytes: number | null;
  effective_bpw: number | null;
  by_type: Record<
    string,
    { tensors: number; elements: number; bytes: number; bpw: number | null }
  >;
  by_type_covers_whole_file: boolean;
  dominant_type: string | null;
  why_unmeasured: string | null;
  context_length: number | null;
  block_count: number | null;
  embedding_length: number | null;
  head_count: number | null;
  head_count_kv: number | null;
  tokenizer: string | null;
  higher_precision_tensors: { name: string; type: string; bpw: number }[];
  /** How many there ARE. The array above is the twelve highest and the panel
   *  shows six of those, so without this the list reads as the whole set when
   *  it can be a small fraction of it. */
  n_higher_precision_tensors: number;
  unmeasured_tensors: number;
  means: string;
}

export interface GgufReport {
  path: string;
  version: number;
  tensor_count: number;
  metadata: Record<string, unknown>;
  tensors: GgufTensor[];
  unknown_types: number[];
  summary: GgufSummary;
}

/** Read a GGUF's header. Nothing is loaded and no GPU is touched — a few
 *  hundred milliseconds and well under a megabyte for a multi-gigabyte file. */
export const readGguf = (path: string) =>
  fetch(`/api/gguf?path=${encodeURIComponent(path)}`).then((r) =>
    json<GgufReport>(r),
  );

/** What loading a GGUF would cost, computed from its header alone.
 *
 *  Two figures because they fail at different moments. `resident_bytes` is
 *  parameters x dtype bytes and has to sit on the device afterwards;
 *  `peak_host_bytes` is parameters x 4 and has to fit in host RAM while the
 *  dequantiser runs. Neither is the file size, and the file size is what
 *  people budget against — measured, a 0.397 GB Q4_K_M file becomes 1.192 GB
 *  of bfloat16 tensors. */
export interface GgufPlan {
  path: string;
  architecture: string | null;
  parameters: number;
  file_bytes: number;
  dtype: string;
  resident_bytes: number;
  peak_host_bytes: number;
  expansion: number | null;
  device: string;
  /** null, never 0 — "we could not ask" and "there is none left" are
   *  different answers and only one is a reason to refuse. */
  device_free_bytes: number | null;
  host_free_bytes: number | null;
  host_total_bytes: number | null;
  verdict: "fits" | "tight" | "will not fit" | "unknown";
  why: string;
  notes: string[];
  means: string;
  /** This file is already the loaded model. The verdict is then about loading
   *  a SECOND copy beside the first, which is not the question being asked. */
  already_loaded?: boolean;
}

/** The plan again, plus what the module actually weighed. The prediction is
 *  arithmetic on a header; this is the check against reality. */
export interface GgufLoaded {
  plan: GgufPlan;
  measured_resident_bytes: number;
  prediction_error: number | null;
  load_seconds: number;
}

/** Ask what a GGUF would cost. Reads the header only — no GPU is touched. */
export const planGguf = (path: string, dtype?: string) =>
  fetch(
    `/api/gguf/plan?path=${encodeURIComponent(path)}` +
      (dtype ? `&dtype=${encodeURIComponent(dtype)}` : ""),
  ).then((r) => json<GgufPlan>(r));

/** Load a GGUF as a full torch module, so every panel works on it.
 *
 *  `confirm` overrides a tight fit and only a tight fit — "will not fit" is
 *  that the RAM needed exceeds the RAM that exists. */
export const loadGguf = (path: string, dtype?: string, confirm = false) =>
  fetch("/api/gguf/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, dtype, confirm }),
  }).then((r) => json<ModelStatus>(r));

/** One position's disagreement between two models on the same token ids. */
export interface PositionDiff {
  index: number;
  token: string;
  /** Nats. The same quantity `ablate` reports for a head. */
  kl: number;
  top_a: string;
  top_b: string;
  p_a: number;
  p_b: number;
  flipped: boolean;
  /** top-1 minus top-2 on each side. */
  margin_a: number;
  margin_b: number;
  /** A flip where the reference model's own margin was under 0.05 — a tie
   *  being broken rather than an answer being changed. */
  contested: boolean;
}

export interface QuantBehaviour {
  model_a: string;
  model_b: string;
  prompt: string;
  tokens: string[];
  positions: PositionDiff[];
  flips: PositionDiff[];
  /** null, not an empty list, when either model returned no attention. */
  attention: { layer: number; mean_abs_diff: number }[] | null;
  attention_means: string | null;
  notes: string[];
  summary: {
    positions: number;
    flips: number;
    contested_flips: number;
    decisive_flips: number;
    mean_kl: number;
    median_kl: number;
    max_kl: number;
    max_kl_at: { index: number; token: string };
    worst_layer: { layer: number; mean_abs_diff: number } | null;
    means: string;
  };
}

/** What quantisation cost this model's behaviour, on one prompt.
 *
 *  Expensive and destructive: it loads two models one after the other and
 *  unloads whatever is currently held to make room. */
export const compareQuantisation = (
  quantised: string,
  original: string,
  prompt: string,
  attention = true,
) =>
  DEMO || VIEWER
    ? noModelHere(
        "Comparing what a quantisation cost loads BOTH builds of the model and runs them side by side — two multi-gigabyte downloads.",
      )
    : fetch("/api/quantdiff/behaviour", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ quantised, original, prompt, attention }),
  }).then((r) => json<QuantBehaviour>(r));

/** An attribution graph read from another tool's file.
 *
 *  `available: false` is a state, not an error — most sessions have no graph.
 *  `provenance.measured_by` is guaranteed present when `available` is true:
 *  both the server and the viewer refuse to report a graph without it. */
export interface GraphView {
  available: boolean;
  error?: string;
  n_nodes?: number;
  edges?: { source: number; target: number; weight: number }[];
  provenance?: {
    file?: string;
    producer?: string;
    model?: string | null;
    scan?: string | null;
    measured_by?: string;
  };
  prompt?: string;
  summary?: {
    nodes?: number;
    possible_edges?: number;
    nonzero_edges?: number;
    density?: number | null;
    max_abs_weight?: number | null;
    returned_edges?: number;
    truncated?: boolean;
    means?: string;
  };
  notes?: string[];
}

/** The attribution graph carried by the open session, if any. */
export const getGraph = () =>
  // DEMO ONLY. `viewer.ts` has answered this path since the graph section
  // shipped -- including a provenance guard written for exactly this case,
  // "this copy runs in the RECIPIENT'S browser on a file a stranger
  // forwarded" -- and this refusal fired before the patched fetch could ever
  // reach it. So the guard had never run once, and a recipient sent a file
  // carrying an attribution graph was told to install ModelMRI to see the
  // thing already in their hands. The demo has no file behind it, so its
  // refusal stays.
  DEMO
    ? noModelHere(
        "Opening a circuit-tracer attribution graph reads a `.pt` from your disk, which a static page cannot see.",
      )
    : fetch("/api/graph").then((r) => json<GraphView>(r));

export interface CustomCandidate {
  path: string;
  name: string;
  dir: string;
  has_example?: boolean;
  hint?: boolean;
  mb?: number;
  /** What the file actually is, read from its archive index rather than
   *  guessed from `.pt` vs `.pth` — which are the same container and say
   *  nothing about the contents. */
  kind?: "gguf" | "torchscript" | "checkpoint" | "legacy" | "unreadable";
}

/** #15 — one thing that was ablated in a network you trained yourself. */
export interface AblationSite {
  name: string;
  kind: string;
  /** The task's unit: nats for a classifier, output-spread for a regressor. */
  effect: number;
  /** The strongest control draw at this site. **null**, never 0, when this
   *  site was not among the strongest and so was never tested — "random edits
   *  here do nothing" is a claim, and nothing measured it. */
  control_max: number | null;
  control_draws: number;
  /** null when untested. */
  beats_control: boolean | null;
}

export interface Ablation {
  kind: "layers" | "inputs";
  task: string;
  unit: string;
  sites: AblationSite[];
  n_sites: number;
  n_controlled: number;
  n_samples: number;
  passes: number;
  seconds: number;
  /** Sites past the cap. Missing from the list, NOT measured as zero. */
  truncated: number;
  control_ceiling: number | null;
  /** n_tested / (draws + 1) — each site is compared against the strongest of
   *  its draws, so under a null where every site is equivalent the real edit
   *  wins one time in nine. */
  expected_false_positives: number;
  means: string;
}

export const ablateCustom = (kind: "layers" | "inputs", grid = 0) =>
  DEMO || VIEWER
    ? noModelHere(
        "Ablating your own network needs that network loaded, and this page carries recordings rather than weights.",
      )
    : fetch("/api/custom/ablate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ kind, grid }),
  }).then((r) => json<Ablation>(r));

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
    /**
     * How many of the outputs are nan or inf. 0 on a healthy model, and set
     * even when SOME are usable — `nonfinite` is only true when none are.
     * The argmax is ranked over the finite values, so it names a real
     * prediction; this is how many slots it had to step over to do it.
     */
    n_nonfinite?: number;
  };
}

export const getCustom = () =>
  fetch("/api/custom").then((r) => json<CustomStatus>(r));

export interface CustomCandidates {
  adapters: CustomCandidate[];
  torchscript: CustomCandidate[];
  roots: string[];
  /** How many the walk SAW, against how many it returned. Both walks stop at
   *  40, and a panel whose whole job is "here is what is on your disk" showed
   *  40 of 45 with nothing to say five were missing. */
  n_adapters_found: number;
  n_torchscript_found: number;
  truncated: boolean;
}

export const getCustomCandidates = () =>
  fetch("/api/custom/candidates").then((r) => json<CustomCandidates>(r));

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
  /** OPTIONAL, like `kl_to_final` below, because `session._lens` copies it
   *  only when the file carries a finite one — so a recorded row can arrive
   *  with no entropy at all. Typed `number` here, it was read as one: the
   *  panel divided by it to size a bar and called `.toFixed` on it, which is
   *  `NaN%` widths followed by a TypeError, and `frontend/src` has no error
   *  boundary above this. The recipient's viewer went white. */
  entropy?: number;
  /** KL(truth ‖ lens) in nats: how much information is lost by reading THIS
   *  layer instead of the model's own final answer.
   *
   *  `lens.py` computes it in "the same direction and same floor as
   *  `ablate.kl_nats`, so a lens error and a head score on one screen are the
   *  same quantity" — a deliberate choice that only pays off if the number
   *  reaches the screen, and it was not in this type at all. */
  kl_to_final?: number;
}

/** #44 — one occluded block of the camera frame. */
export interface OcclusionBlock {
  row: number;
  col: number;
  /** Shift in the tower's pooled embedding, in units of its own spread. */
  shift: number;
  /** **null**, never 0, when this block was never controlled. */
  control_max: number | null;
  control_draws: number;
  clears_control: boolean | null;
  attention: number | null;
}

export interface OcclusionMap {
  baseline: string;
  grid: number[];
  stride: number;
  blocks: OcclusionBlock[];
  n_blocks: number;
  n_controlled: number;
  passes: number;
  seconds: number;
  scale: number;
  scale_frames: number;
  /** Spearman between the causal map and the attention map for THIS frame.
   *  null when no attention map was supplied — a state, not a zero. */
  attention_agreement: number | null;
  /** WHICH attention map that was. The agreement is layer-dependent, so the
   *  Spearman is not reportable without it. null when nothing was compared. */
  compared_layer: number | null;
  compared_head: number | null;
  means: string;
}

export const occludeFrame = (body: {
  episode: number;
  t: number;
  baseline?: string;
  stride?: number;
  layer?: number;
  head?: number;
}) =>
  fetch("/api/vla/occlude", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => json<OcclusionMap>(r));

export const occlusionCost = (stride = 0) =>
  fetch(`/api/vla/occlude/cost?stride=${stride}`).then((r) =>
    json<{ blocks: number; passes: number; stride: number; grid: number[] }>(r),
  );

/** #48 — one frame of a cross-episode sweep. */
export interface SweepRow {
  episode: number;
  timestep: number;
  value: number;
}

export interface VLASweep {
  metric: string;
  unit: string;
  /** What was swept, carried on the result rather than assumed from whatever
   *  is loaded now. A sweep outlives the session that ran it — it is saved to
   *  sqlite and read back — so a table that did not name its own dataset,
   *  policy and camera could be read against the wrong three. All three were
   *  sent from the start and declared nowhere. */
  dataset: string;
  /** `null` when no policy was resident. It used to collapse to `""`, while
   *  the sibling `/api/vla` field for the same fact correctly said `null`. */
  policy: string | null;
  camera: string;
  rows: SweepRow[];
  n_frames: number;
  n_episodes: number;
  frames_total: number;
  episode_stride: number;
  frame_stride: number;
  seconds: number;
  /** A SAMPLE of the frames that could not be measured, capped server-side.
   *  Use `n_failed` for how many there were — this list is what to look at,
   *  not the measurement. */
  failed: { episode: number; timestep: number; why: string }[];
  /** How many frames failed in total.
   *
   *  Separate from `failed.length` because that list is truncated, and the
   *  server's own sentence used to count the truncated list: with PyAV absent
   *  over six episodes of a hundred frames, all 600 failed and the report read
   *  "20 frame(s) could not be measured". The true figure was not derivable
   *  from the payload at all. */
  n_failed: number;
  means: string;
  strip: {
    rows: { episode: number; timesteps: number[]; values: number[] }[];
    /** The strip repeats the metric and its unit, because it is rendered
       away from the header that names them and a heat strip with no unit is
       a picture of nothing in particular. */
    metric: string;
    unit: string;
    /** `null` when no row was measured — the RANGE of a metric nobody
     *  observed. 0.0 there read as a flat result rather than as no result. */
    low: number | null;
    high: number | null;
    frame_stride: number;
    /** Episodes have different lengths, so the strip is ragged rather than
     *  padded with zeros that would read as measured lows. */
    ragged: boolean;
  };
}

export const runVlaSweep = (body: {
  metric: string;
  episode_stride?: number;
  frame_stride?: number;
}) =>
  fetch("/api/vla/sweep", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => json<VLASweep>(r));

export const vlaSweepCost = (metric: string, frameStride: number) =>
  fetch(
    `/api/vla/sweep/cost?metric=${metric}&frame_stride=${frameStride}`,
  ).then((r) =>
    json<{
      frames: number;
      frames_total: number;
      passes: number;
      coverage: number;
      seconds: number | null;
      seconds_from: string;
    }>(r),
  );

/** #46 — a robot finding as a `.mri` somebody else can open. */
export const shareVlaFinding = (body: {
  episode: number;
  t: number;
  layer?: number;
  head?: number;
  occlusion?: unknown;
  note?: string;
}) =>
  fetch("/api/vla/share", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then(async (r) => {
    if (!r.ok) throw new ApiError(r.status, await r.text());
    return r.blob();
  });

/** Write a stored sweep into a container Foxglove already opens.
 *
 *  Bytes and a Content-Disposition, like `exportSession`, so it does not go
 *  through the demo shim's patched fetch and is not a plain `<a download>`
 *  either: the server answers a refusal as JSON with a 409 or a 422 — no sweep
 *  stored under those three keys, the `mcap` package absent, rerun's analytics on —
 *  and a link would cheerfully save that sentence to disk as an `.mcap` file
 *  the reader then opens in Foxglove and sees nothing in.
 *
 *  `dataset`, `metric` and `policy` are the three columns a sweep is STORED
 *  against, so they are sent from the `VLASweep` that ran rather than inferred
 *  from whatever is loaded now — the same reason that payload carries them.
 *  `policy: ""` is not a missing value; it is how the server records that no
 *  policy was resident. Omitting `camera` asks the server to resolve it, which
 *  it does only when one camera is stored and refuses when two are.
 */
export async function exportVlaSweep(body: {
  dataset: string;
  metric: string;
  policy?: string;
  camera?: string;
  /** "mcap" or "rrd". Both write real files as of 0.13.0; each refuses
   *  with a sentence when its writer is absent, and `rrd` refuses a
   *  second time when rerun's usage analytics are on, because ModelMRI
   *  promises no telemetry and cannot make that promise for rerun. */
  container?: string;
  /** The download's filename stem. The server rebuilds it from an allowlist. */
  name?: string;
  /** What the numbers are numbers of, when the caller knows — an entropy in
   *  nats over a 16-patch grid is bounded by ln(16), and a reader who knows
   *  the ceiling reads the number differently. Left out, the file carries the
   *  server's sentence saying no resolution was published, which is written
   *  INTO the file rather than omitted from it. */
  resolution?: string;
}): Promise<{ blob: Blob; filename: string }> {
  if (DEMO || VIEWER) {
    return refusedHere(
      "Exporting a sweep to MCAP reads a run this machine measured and " +
        "stored: the rows, the unit they are in, and the two strides that say " +
        "which frames were never opened. A static page has no trace database " +
        "and never ran a sweep, so there is nothing here to write out — and a " +
        "baked file would be somebody else's dataset arriving in your " +
        "Foxglove under our provenance. Install ModelMRI (`pip install " +
        "modelmri`), run a sweep on your own data, and this writes the file.",
    );
  }
  const r = await fetch("/api/vla/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new ApiError(r.status, await r.text());
  const disposition = r.headers.get("Content-Disposition") || "";
  const match = /filename="([^"]+)"/.exec(disposition);
  // The server's own filename when it sent one. The fallback names the
  // container rather than the sweep, because a stem invented here would claim
  // a dataset and a metric this response did not carry back.
  return { blob: await r.blob(), filename: match?.[1] || "sweep.mcap" };
}

/** #20 — one prompt of a finetune comparison. Never reported on its own. */
export interface DiffPromptResult {
  prompt: string;
  n_tokens: number;
  mean_kl: number;
  max_kl: number;
  flips: number;
  /** Where this prompt's residual cosine falls furthest in one step. No
   *  threshold is involved. null when the curve never decreases. */
  first_divergent_layer: number | null;
  drop: number;
  cosine: number[];
}

export interface DiffSpread {
  name: string;
  median: number;
  low: number;
  high: number;
  n: number;
  n_nonzero: number;
}

export interface DiffLayer {
  layer: number;
  median: number;
  low: number;
  high: number;
  n: number;
  /** On how many prompts this layer was where the cosine fell furthest. */
  n_first: number;
}

export interface DiffHead {
  layer: number;
  head: number;
  median_a: number;
  median_b: number;
  shift: number;
  n: number;
  top_a: number;
  top_b: number;
}

export interface DiffToken {
  prompt_index: number;
  index: number;
  token: string;
  kl_a: number;
  kl_b: number;
  shift: number;
  /** Crossed its own side's noise floor. The two models have different
   *  floors, so this is the only comparison that survives. */
  newly_used: boolean;
  newly_ignored: boolean;
}

export interface ModelDiffReport {
  model_a: string;
  model_b: string;
  n_prompts: number;
  n_layers: number;
  prompts: DiffPromptResult[];
  layers: DiffLayer[];
  kl: DiffSpread | null;
  flips: DiffSpread | null;
  consensus_layer: number | null;
  consensus_share: number;
  heads: DiffHead[];
  head_passes: number;
  tokens: DiffToken[];
  seconds: number;
  notes: string[];
  means: string;
  receipt?: Receipt | null;
}

export const diffModels = (body: {
  a: string;
  b: string;
  prompts?: string[];
  file?: string;
  include_heads?: boolean;
  include_tokens?: boolean;
}) =>
  DEMO || VIEWER
    ? noModelHere(
        "Diffing a finetune against its base loads two checkpoints and runs a prompt set through both.",
      )
    : fetch("/api/diff/models", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => json<ModelDiffReport>(r));

/** #14 — one passage of your document, and what it did to the answer. */
export interface GroundScore {
  index: number;
  preview: string;
  n_tokens: number;
  /** Nats. How far the answer moved when this passage was masked out. */
  dependence: number;
  /** Share of the answer position's attention that landed here, meaned over
   *  every layer and head. **null**, never 0, when this model's attention
   *  implementation never builds the score matrix — a passage nothing looked
   *  at and a number that was never returned are different facts. */
  attention: number | null;
  depended_on: boolean;
  /** Attention on it, no causal dependence on it — the signature of an answer
   *  coming from the weights rather than from your document.
   *
   *  **null** means the reading could not be taken: without the attention
   *  half there is no "looked at", and with a noise floor of exactly 0.0
   *  every passage that moved the answer counts as depended-on, so the flag
   *  can never fire. `false` would read as "this passage is fine", which on
   *  either of those runs nothing measured. */
  looked_not_used: boolean | null;
}

export interface Grounding {
  question: string;
  answer: string;
  answer_p: number;
  position: number;
  chunks: GroundScore[];
  n_chunks: number;
  n_prompt_tokens: number;
  noise_floor: number;
  /** Every passage masked at once. Printed beside the parts precisely because
   *  it is not their sum — which is why nothing here is a percentage. */
  joint: number;
  attention_share: number | null;
  attention_available: boolean;
  attention_note: string;
  /** The repeat pass reproduced the answer bit for bit, so the floor is 0.0
   *  and "cleared the floor" degrades to "moved the answer at all". */
  floor_degenerate: boolean;
  ungrounded: boolean;
  passes: number;
  seconds: number;
  means: string;
  receipt?: Receipt | null;
}

export const groundAnswer = (body: {
  document?: string;
  file?: string;
  question: string;
  max_chunks?: number;
}) =>
  // DEMO ONLY, and the split is the point. The Pages demo has no file behind
  // it and `demo.ts` has no handler for this path, so a refusal naming what
  // the measurement would cost is the honest answer there.
  //
  // The viewer DOES have a file. `session.build` writes a `ground` section
  // into every `.mri` that carries one, and this branch rejected before the
  // patched fetch in main.tsx could reach `viewerFetch` — so a file sent
  // because of its grounding result opened showing none of it. Re-running is
  // still impossible here, and deliberately so: the document is the private
  // half of the pair and is not in the file. The recording is what gets read.
  DEMO
    ? noModelHere(
        "Asking whether the answer came from your document or the weights masks passages out and re-runs the model — and the document is yours, not the bundle’s.",
      )
    : fetch("/api/ground", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => json<Grounding>(r));

/** #10 — a probe at every layer, with the two references behind the curve. */
export interface LayerProbe {
  layer: number;
  accuracy: number;
  null_low: number;
  null_high: number;
  /** The field the feature exists for: this layer did not beat what the same
   *  fit achieves on shuffled labels. */
  inside_null: boolean;
  beats_majority: boolean;
  /** The shuffled fit reached the top of the scale, so NO accuracy could have
   *  cleared it — untestable with this many examples, not uninformative. */
  null_saturated: boolean;
}

export interface ProbeReport {
  layers: LayerProbe[];
  majority: number;
  /** `{"0": nA, "1": nB}` — the class sizes the majority line is computed
   *  FROM. Shown beside it, because "majority 62%" is a different reading
   *  when it comes from 8 examples against 5 than from 800 against 500, and
   *  the percentage alone cannot say which. */
  counts?: Record<string, number>;
  n_train: number;
  n_test: number;
  n_permutations: number;
  /** null when nothing cleared anywhere, which is a result. */
  best_layer: number | null;
  n_readable_layers: number;
  n_underpowered_layers: number;
  /** n_layers x 5%: sweeping every layer against a 95th-percentile band is a
   *  multiple comparison, so this many clear by chance. */
  expected_false_positives: number;
  means: string;
  saved?: { name: string; path: string; dims: number };
  receipt?: Receipt | null;
}

export const runProbe = (body: {
  examples: { text: string; label: number }[];
  n_permutations?: number;
  save_as?: string;
}) =>
  DEMO || VIEWER
    ? noModelHere(
        "Fitting a probe trains on YOUR labelled examples and scores against shuffled nulls, which needs a live residual stream.",
      )
    : fetch("/api/probe", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => json<ProbeReport>(r));

/** #11 — one component's contribution into a later block's residual input. */
export interface PathSender {
  layer: number;
  head: number | null;
  name: string;
  recovery: number;
  delta_norm: number;
  control_max?: number;
  control_min?: number;
  control_draws?: number;
  shifted_position?: number;
  clears_control?: boolean;
  clears_position?: boolean;
}

export interface PathTrace {
  receiver: { layer: number; position: number };
  senders: PathSender[];
  n_senders: number;
  n_controlled: number;
  gap: number;
  /** Two senders closer than this are TIED, not ranked. */
  recovery_resolution: number;
  seeding: string;
  scope: string;
  means: string;
  passes: number;
  seconds: number;
  receipt?: Receipt | null;
}

export const pathTrace = (body: {
  clean: string;
  corrupt: string;
  layer: number;
  position: number;
}) =>
  fetch("/api/patch/path", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => json<PathTrace>(r));

/** #52 — a PATCHING graph, walked backwards from the node grid's own sites.
 *
 *  NOT `Graph` above, which is somebody else's transcoder attribution graph
 *  read out of a `.pt`. Two different objects from two different measurements,
 *  and the type names keep them apart for the same reason the `.mri` sections
 *  do.
 */
export interface PatchGraphNode {
  id: string;
  layer: number;
  /** null for an MLP or a residual site. */
  head: number | null;
  position: number;
  /** "seed" for a site the node grid flagged, "sender" otherwise. */
  role: string;
  depth: number;
}

export interface PatchGraphEdge {
  source: string;
  target: string;
  /** The SAME recovery fraction the node grid and `path_trace` report. */
  recovery: number;
  /** The strongest of `control_draws` same-norm random patches at this site.
   *  Every drawn edge has one — an edge with a score and no verdict is pruned
   *  server-side rather than drawn as though it had passed. */
  control_max: number | null;
  /** EVERY draw, not just the strongest. The spread is the finding: a verdict
   *  quoted as "beat 0.28" reads differently once you can see that seven of
   *  the eight were nowhere near it. Absent on a `.mri` written before they
   *  travelled — the verdict still rests on `control_max`, which is there. */
  controls?: number[];
  control_draws: number;
  /** `false` is a real verdict and the panel draws it differently. */
  clears_control: boolean | null;
  clears_position: boolean | null;
  tested?: boolean;
}

export interface PatchGraphView {
  nodes: PatchGraphNode[];
  edges: PatchGraphEdge[];
  clean: string;
  corrupt: string;
  answer: string;
  depth: number;
  max_receivers: number;
  n_receivers_expanded: number;
  n_nodes: number;
  n_edges: number;
  /** Senders scored against edges kept. The difference is the prune, and it
   *  is reported rather than implied. */
  n_scored: number;
  n_pruned: number;
  n_weak: number;
  n_untested: number;
  passes: number;
  seconds: number;
  prune_threshold: number;
  prune_from: string;
  /** Receivers that still had senders when the depth ran out. NOT an empty
   *  edge list: "nothing wrote this" and "we did not ask" differ. */
  frontier: string[];
  seeding: string;
  means: string;
  /** Set when the graph came out of an open `.mri` rather than the model. */
  recorded?: boolean;
  receipt?: Receipt | null;
}

export const patchGraph = (body: {
  clean: string;
  corrupt: string;
  depth?: number;
  max_receivers?: number;
}) => {
  // The second lock on a door the demo's panel already keeps shut. A graph is
  // thousands of forward passes with activations replaced — MEASURED at 4,165
  // on Qwen3-1.7B (bfloat16, cuda) at depth 2 on "The Eiffel Tower is in the
  // city of" against "The Colosseum is in the city of", reproduced twice to
  // the digit — and there is no model behind the Pages demo to run one.
  // `demo.ts` has no handler for this path, so without the refusal the call
  // would reach the real fetch and 404 on a static host: a visitor would learn
  // that the measurement is broken rather than that the page has no model.
  //
  // THE COUNT IS PAIR-DEPENDENT AND THE SENTENCE SAYS SO. It used to read
  // "1,735 forward passes" flat, and that number came from a DIFFERENT prompt
  // pair than the one it was attributed to. This repo calls two prompts "the
  // reference pair": "The Eiffel Tower is LOCATED in the city of" (saes, lens,
  // feature_ablate) and "The Eiffel Tower is in the city of" (runtime). They
  // resolve differently — MEASURED on Qwen3-1.7B/bfloat16, "is located in"
  // gives a recovery resolution of 0.006231 at a gap of 30.25 and "is in"
  // 0.007571 at a gap of 24.25 — and the resolution sets the prune threshold,
  // which sets how much of the graph survives, which sets the pass count. The
  // walk also seeds from the sites the node grid flags, and that count is
  // per-pair too.
  //
  // So a reader sizing a run against 1,735 under-budgets the "is in" pair by
  // 2.4x. Quoting one pair's cost as THE cost is the same error as quoting one
  // control draw's verdict as the feature's — see `feature_ablate` — one layer
  // down. The graph itself is deterministic: 4,165 passes, 2,227 senders
  // scored, 2,131 pruned, 52 nodes and 96 edges, reproduced twice to the digit.
  //
  // NOT the viewer. A graph is the one section here a recipient could never
  // rebuild — which is why `session.build` writes it into the `.mri` at all —
  // and this branch fired before the patched fetch could reach `viewerFetch`,
  // so the passes the sender spent arrived as an "install ModelMRI" refusal.
  // Nothing is re-run there; the recorded graph is read out of the file, and
  // the panel says so.
  if (DEMO) {
    return Promise.reject(
      new ApiError(
        409,
        JSON.stringify({
          error:
            "Building a patching graph replaces activations and re-runs the " +
            "model thousands of times — 4,165 forward passes on Qwen3-1.7B at " +
            "depth 2 on the pair this was measured with, and a different " +
            "number on yours — and there is no model behind this page. " +
            "Install ModelMRI (`pip install modelmri`) to build one on your " +
            "own.",
        }),
      ),
    );
  }
  return fetch("/api/patch/graph", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => json<PatchGraphView>(r));
};

/** #12 — a hidden state described in words, with two controls. */
export interface Patchscope {
  source: {
    prompt: string;
    layer: number;
    position: number;
    tokens: string[];
    norm: number;
  };
  /** Returned with EVERY response, because two decodes taken under
   *  different targets are not comparable and a hidden default would make
   *  that invisible. */
  target: {
    prompt: string;
    layer: number;
    position: number;
    tokens: string[];
  };
  decode: string;
  controls: { identity: string; random: string[]; draws: number };
  same_as_identity: boolean;
  same_as_random: boolean;
  /** How much of the decode's vocabulary each control already used. Reported,
   *  never thresholded. */
  overlap_identity: number;
  overlap_random: number;
  informative: boolean;
  cross_layer: boolean;
  means: string;
  receipt?: Receipt | null;
}

export const runPatchscope = (body: {
  prompt: string;
  layer: number;
  position?: number;
  target?: string;
  target_layer?: number | null;
  max_new_tokens?: number;
}) =>
  DEMO || VIEWER
    ? noModelHere(
        "A patchscope hands a hidden state back to the model and asks it to describe that state in words, so it needs the model.",
      )
    : fetch("/api/patchscope", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => json<Patchscope>(r));

/** #13 — what a feature fires on in YOUR corpus, and what it promotes. */
export interface FeatureSpan {
  text: string;
  token: string;
  activation: number;
  position: number;
  sequence: number;
  /** Character index of the firing token inside `text`. Not the same as
   *  `text.indexOf(token)` — a span can contain the same word twice and only
   *  one of those positions fired. */
  offset: number;
}

export interface FeatureEvidence {
  corpus: {
    corpus_label: string;
    n_tokens: number;
    n_sequences: number;
    n_features: number;
    n_never_fired: number;
    never_fired_share: number;
    truncated: boolean;
    means: string;
  };
  top_by_firing_rate: {
    feature: number;
    n_fired: number;
    max_activation: number;
  }[];
  evidence?: {
    feature: number;
    spans: FeatureSpan[];
    n_fired: number;
    n_tokens: number;
    firing_rate: number;
    max_activation: number;
    histogram: number[];
    bin_edges: number[];
    /** false when the feature fires on more than a fifth of tokens — not a
     *  concept, whatever its top spans look like. */
    selective: boolean;
    means: string;
    /** Always null. Naming the concept is the reader's job. */
    label: null;
  };
  logit_weights?: {
    feature: number;
    promotes: { token: string; logit: number }[];
    suppresses: { token: string; logit: number }[];
    exact: boolean;
    means: string;
  };
  receipt?: Receipt | null;
}

export const featureEvidence = (body: {
  texts?: string[];
  file?: string;
  label?: string;
  feature?: number;
  top_k?: number;
}) =>
  fetch("/api/features/evidence", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => json<FeatureEvidence>(r));

/** A behavioural label for one head, with the evidence it cleared. */
export interface HeadTypeLabel {
  layer: number;
  head: number;
  /** null is "no type detected", which is the finding for most heads — not a
   *  label that went missing. */
  label: string | null;
  /** Standard deviations above this head's own null. Null when unlabelled;
   *  0 would read as "exactly at the null", which is a measurement. */
  margin?: number | null;
  /** The winning pattern's score as a multiple of chance under the causal
   *  mask. Significance and effect size are both required, and both shown. */
  times_chance?: number | null;
  /** The most attention this head puts on any single target. A label is only
   *  attached when the pattern's offset IS this peak. */
  peak?: number;
  /** "repeat" (gated on non-repeating sequences) or "chance" (gated on chance
   *  under the causal mask). Not interchangeable. */
  null_kind?: string;
  scores?: Record<string, number>;
}

export interface HeadTypes {
  labels: HeadTypeLabel[];
  /** How many heads earned each label. OPTIONAL because a recorded set may
   *  not carry it: `session._head_types` copies `counts` only when the file
   *  has a dict there, so a `.mri` can legally arrive with labels and no
   *  tally. Typed as required, the panel called `Object.entries` on it — and
   *  there is no error boundary above this panel, so a file written without
   *  the tally took the whole page white in the recipient's browser. */
  counts?: Record<string, number>;
  n_layers: number;
  n_heads: number;
  seq_len: number;
  n_sequences: number;
  margin_sigma: number;
  means: string;
  recorded?: boolean;
  receipt?: Receipt | null;
}

export const getHeadTypes = () =>
  fetch("/api/attention/types").then((r) => json<HeadTypes>(r));

/** Where one head's columns sit in every projection it touches.
 *
 *  `n_kv_heads` is DERIVED from `v_proj`'s own width, never read from a config
 *  field — several architectures do not set it, and on the ones that do it
 *  would have to agree with this division anyway. Measured on Qwen3-1.7B:
 *  16 query heads over 8 value heads, `head_dim` 128. */
export interface HeadGeometry {
  n_heads: number;
  n_kv_heads: number;
  head_dim: number;
  d_model: number;
  /** How many query heads share one key/value head. 1 is ordinary
   *  multi-head attention, not a special case. */
  group_size: number;
}

export interface OvToken {
  token: string;
  /** Relative to the vocabulary mean and at UNIT SCALE. These rank tokens;
   *  they do not predict logit amounts, because the final norm's real scale
   *  depends on a stream that does not exist here. */
  score: number;
}

/** What a head writes into the stream when it attends to one token.
 *
 *  The only attention readout here that needs no prompt: it is a product of
 *  weights, so it is about the HEAD rather than about the current run, and it
 *  is the same every time. */
export interface HeadOv {
  layer: number;
  head: number;
  kv_head: number;
  source_token_id: number;
  source_token: string;
  /** How many tokens the text you sent encodes to. Above 1, the readout is of
   *  the FIRST of them — a head reads one token at a time. */
  source_token_count: number;
  geometry: HeadGeometry;
  promotes: OvToken[];
  suppresses: OvToken[];
  exact: boolean;
  means: string;
  receipt?: Receipt | null;
}

/** The eigenvalue readout of a head's OV circuit, over a NAMED sample.
 *
 *  The full circuit is vocabulary-by-vocabulary — 92 TB on Qwen3-1.7B — so
 *  this is measured over a sample and carries its size, its seed, and how
 *  much of the spectrum sits off the real line. There is deliberately no
 *  label: a fraction near chance is not a copying head, and a fraction that
 *  is not near chance is still a claim about the tokens nobody drew. */
export interface HeadOvSpectrum {
  layer: number;
  head: number;
  kv_head: number;
  geometry: HeadGeometry;
  n_sampled: number;
  n_vocab: number;
  seed: number;
  /** True when the vocabulary is smaller than the sample asked for. The cap
   *  is REPORTED, not silently applied. */
  sample_capped: boolean;
  positive_fraction: number;
  positive: number;
  trace: number;
  /** Share of the spectrum's mass off the real line. A real non-symmetric
   *  matrix has complex eigenvalues, so a high value means
   *  `positive_fraction` describes rotation as much as sign. */
  imaginary_mass: number;
  means: string;
  receipt?: Receipt | null;
}

export const getHeadOv = (layer: number, head: number, token: string, topK = 10) =>
  fetch(
    `/api/attention/ov?layer=${layer}&head=${head}` +
      `&token=${encodeURIComponent(token)}&top_k=${topK}`,
  ).then((r) => json<HeadOv>(r));

/** One place a head wrote hard, and what it was reading when it did.
 *
 *  A PAIR, unlike an SAE feature's span. A feature fires at one position and
 *  one offset locates it; a head writes at one position while attending to
 *  another, and reporting only the first is half a sentence. */
export interface HeadSpan {
  position: number;
  token: string;
  /** Character offset of `token` inside `text`. Not derivable by searching:
   *  a window can contain the same token twice and only one of them wrote. */
  offset: number;
  /** `null` when attention was not read on this sweep — never 0, which would
   *  say the head looked at position zero. */
  source_position: number | null;
  source_token: string | null;
  source_share: number | null;
  write_norm: number;
  sequence: number;
  text: string;
}

export interface HeadCorpus {
  layer: number;
  head: number;
  kv_head: number;
  corpus_label: string;
  corpus_sha256: string;
  n_sequences: number;
  n_tokens: number;
  truncated: boolean;
  /** False means every `source_*` on every span is null, and the sentence
   *  says so rather than leaving a column of nulls to read as "looked
   *  nowhere". */
  attention_read: boolean;
  spans: HeadSpan[];
  write_norm_mean: number;
  write_norm_median: number;
  write_norm_max: number;
  /** Positions READ. `spans.length` is how many were kept, and a zero write
   *  is never kept. */
  n_positions: number;
  /** Positions that carried a non-zero write. The gap between this and
   *  `n_positions` is what makes a head SPARSE rather than absent — two
   *  states that had one sentence between them until a head writing once in
   *  ten positions printed "not seen in this corpus". */
  n_wrote: number;
  passes: number;
  device: string;
  means: string;
}

export interface HeadEvidence {
  corpus: HeadCorpus;
  /** What it pushes the vocabulary toward for the token it most often read.
   *  `null` when attention was not read, so there is no such token. */
  pushes_at: HeadOv | null;
  /** NOT run — hundreds of forward passes against the two this took — and
   *  named rather than omitted, because a two-legged answer presented as the
   *  whole thing is what the SAE version argues against. */
  causal: { available: boolean; how: string; why: string };
  means: string;
  receipt?: Receipt | null;
}

export const headEvidence = (body: {
  texts?: string[];
  file?: string;
  label?: string;
  layer?: number;
  head?: number;
  read_attention?: boolean;
  top_k?: number;
}) =>
  fetch("/api/attention/head/evidence", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => json<HeadEvidence>(r));

export const headEvidenceCost = (nSequences: number, readAttention: boolean) =>
  fetch(
    `/api/attention/head/evidence/cost?n_sequences=${nSequences}` +
      `&read_attention=${readAttention}`,
  ).then((r) =>
    json<{
      n_sequences: number;
      passes: number;
      reads_attention: boolean;
      means: string;
    }>(r),
  );

export const getHeadOvSpectrum = (layer: number, head: number, seed = 0) =>
  fetch(`/api/attention/ov/spectrum?layer=${layer}&head=${head}&seed=${seed}`).then(
    (r) => json<HeadOvSpectrum>(r),
  );

/** One component's direct push on the predicted token, in logits. */
export interface DirectContribution {
  name: string;
  kind: "embed" | "head" | "mlp";
  layer: number | null;
  head: number | null;
  /** Signed, and already shift-corrected against this component's own
   *  vocabulary mean — a component that lifts the whole vocabulary equally
   *  has changed nothing, because softmax ignores a constant. */
  logits: number;
  /** Under the reconstruction residual: this component's direct effect cannot
   *  be told from the error the approximation already makes. NOT a claim that
   *  the component does not matter. */
  unreadable: boolean;
}

export interface DirectAttribution {
  token: string;
  token_id: number;
  position: number;
  real_logit: number;
  bias: number;
  /** The gap between every component summed and the logit the model really
   *  produced — what freezing the normalisation scale cost on this run. The
   *  panel must show it: without it the chart claims a decomposition it does
   *  not have. */
  residual: number;
  residual_share: number;
  norm_kind: string;
  components: DirectContribution[];
  /** Every component in the decomposition, counted by the server BEFORE its
   *  `top_k` cut. This — not `components.length` — is the denominator for
   *  `n_unreadable`, which is counted over the same whole. `components` is
   *  only the post-cut slice: the demo carries 40 rows out of 477 components,
   *  so dividing by it printed the impossible "434 of 40". Anything phrased
   *  as a fraction of the decomposition divides by this. */
  n_components: number;
  /** Components whose direct effect is under the reconstruction residual,
   *  counted over the whole decomposition — read against `n_components`. */
  n_unreadable: number;
  means: string;
  receipt?: Receipt | null;
}

export const getDirectAttribution = (topK = 40) =>
  fetch(`/api/attention/direct?top_k=${topK}`).then((r) =>
    json<DirectAttribution>(r),
  );

/** What a translator was fitted to, and what it bought on held-out text. */
export interface TunedLensInfo {
  corpus_label?: string;
  corpus_sha256?: string;
  n_tokens?: number;
  n_held_out?: number;
  n_layers_improved?: number;
  n_layers?: number;
  /** Tokens of text per translator parameter. Under 1 means the fit is
   *  under-determined, and `caution` then says so in a sentence. */
  tokens_per_parameter?: number;
  caution?: string;
  means?: string;
  seconds?: number;
  cached?: boolean;
  layers?: { layer: number; plain_kl: number; tuned_kl: number; gain: number }[];
}

export const trainTunedLens = (body: { texts?: string[]; file?: string; steps?: number }) =>
  DEMO || VIEWER
    ? noModelHere(
        "Training a tuned lens is a training run over a corpus — minutes of compute against a live model.",
      )
    : fetch("/api/lens/tune", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => json<TunedLensInfo>(r));

export const tunedLensStatus = () =>
  fetch("/api/lens/tuned").then((r) =>
    json<{ trained: boolean; info?: TunedLensInfo }>(r),
  );

export const getLens = (topK = 5, kind: "plain" | "tuned" | "both" = "plain") =>
  fetch(`/api/lens?top_k=${topK}&kind=${kind}`).then((r) =>
    json<{
      layers: LensRow[];
      n_layers: number;
      final: string;
      settled_at: number | null;
      receipt?: Receipt | null;
      /** BESIDE the plain rows, never instead of them: `layers` is the plain
       *  reading on every kind. Align by `layer`, not by index — the plain
       *  lens has one more row (the model's own final state), which has no
       *  translator because it is the answer rather than a guess at it. */
      tuned?: LensRow[] | null;
      tuned_info?: TunedLensInfo | null;
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
  /** Present only on the static demo, where every path above is a
   *  placeholder. It was baked from the start and typed nowhere, so seven
   *  unexplained placeholder rows rendered with nothing to say why. */
  demo_note?: string;
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
  /** Where MODELMRI_MODELS_DIR points, or `null` when it was never set. */
  models_home: string | null;
  /** Every other cache root this process will read from. */
  inherited_caches: string[];
  cwd: string;
  legacy: string | null;
  platform: string;
}

export const getPaths = () => fetch("/api/paths").then((r) => json<PathInfo>(r));

/** Activation patching: two prompts, one grid. See modelmri/patch.py.
 *
 *  The score is signed and is NOT a KL, unlike every other ranking here —
 *  patching has a direction, and 5 of 132 sites on the reference pair moved
 *  the answer further from the clean run rather than toward it.
 */
export interface PatchSide {
  prompt: string;
  tokens: string[];
  answer: { id: number; text: string; p: number };
}

export interface PatchSite {
  component: string;
  layer: number;
  position: number;
  recovery: number;
  control_max: number;
  control_min: number;
  control_draws: number;
  shifted_position: number;
  clears_control: boolean;
  clears_position: boolean;
}

export interface PatchTrace {
  /** True when this came out of a `.mri` rather than off a live model. */
  recorded?: boolean;
  clean: PatchSide;
  corrupt: PatchSide;
  gap: number;
  n_layers: number;
  n_positions: number;
  components: string[];
  /** Components this architecture does not expose, each with the refusal that
   *  named it — `"mlp: …"`. The trace catches a PatchError per component and
   *  carries on so the rest is still measured, and `patch.py`'s own comment
   *  says why this must be shown: without it "two grids would have arrived
   *  looking like the whole answer". */
  skipped?: string[];
  /** One grid per component. `resid` says where; `attn` and `mlp` say through
   *  what, and on the reference pair they disagree — MLP peaks at +0.365 on a
   *  subject token in layer 0, attention at +0.232 on the last token in
   *  layer 9. */
  grids: Record<string, number[][]>;
  sites: PatchSite[];
  controlled: number;
  dtype: string;
  passes: number;
  seconds: number;
  notes: string[];
  receipt?: Receipt | null;
}

/** The patching grid — live off the model, or read out of an opened `.mri`.
 *
 *  No DEMO/VIEWER branch, deliberately: both shims already answer this path.
 *  `demo.ts` refuses `/api/patch` in words that name what patching costs, and
 *  the viewer serves the `patch` section the opened file carries. A third copy
 *  of that sentence on this side would buy the demo nothing and would, in the
 *  viewer, be the lock that hid the recording — which is precisely what the
 *  branches on `groundAnswer` and `patchGraph` above were doing.
 */
export const patchTrace = (clean: string, corrupt: string) =>
  fetch("/api/patch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ clean, corrupt }),
  }).then((r) => json<PatchTrace>(r));

/** Drop the model and hand the memory back.
 *
 *  `freed_bytes` is the difference in allocator-reported bytes across the
 *  call, not a promise — an allocator that keeps its arena is a real outcome
 *  and the panel says so rather than claiming a round number.
 */
export interface UnloadResult {
  unloaded: boolean;
  was: string | null;
  freed_bytes: number;
  accelerator_bytes_in_use: number | null;
  status: ModelStatus;
}

export const unloadModel = () =>
  fetch("/api/model/unload", { method: "POST" }).then((r) =>
    json<UnloadResult>(r),
  );

/** Also look in this folder for custom models.
 *
 *  The folder joins the allowed roots for this run only — it does not bypass
 *  them. A local tool that will import any path handed to it is a nastier
 *  primitive than it looks.
 */
/** The same walk `/api/custom/candidates` runs, plus the folder that was
 *  added. One type for both, because the panel renders whichever answered
 *  last and a shape that differs between them would drop the truncation
 *  notice depending on which button was pressed. */
export const scanFolder = (path: string) =>
  fetch("/api/custom/scan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  }).then((r) => json<CustomCandidates & { added: string }>(r));

/* ══════════════════════════════════════════════════════════════════════════
   THE JUDGE, AND WHAT THE ROBOT POLICY WOULD DO
   ──────────────────────────────────────────────────────────────────────────
   Six routes that were tested, documented and unreachable. Everything below
   keeps the habits the rest of this file already keeps: a `null` is an
   UNKNOWN and never a zero, a cap travels beside the thing it capped, and the
   server's own `means` sentence is carried through verbatim because that is
   where the caveats live.
   ══════════════════════════════════════════════════════════════════════════ */

/** A deliberate no, raised on this side because the page cannot make the call.
 *
 *  `noModelHere` says "there is no model behind this page", which is the right
 *  sentence for a forward pass and the wrong one for a dataset audit — that
 *  reads FILES already on disk, and what a static bundle lacks there is a
 *  filesystem, not a checkpoint. Same 409 and same JSON shape, so `explain`,
 *  `errorText` and every refusal renderer treat it exactly as they treat the
 *  server's own.
 */
function refusedHere(sentence: string): Promise<never> {
  return Promise.reject(new ApiError(409, JSON.stringify({ error: sentence })));
}

// ------------------------------------------------------------------- judge
//
// LLM-as-judge, except the number is READ rather than sampled: one forward
// pass per paraphrase, softmax over the verdict token ids at the final
// position, no generation. Holding the weights is what makes that possible,
// and it is the whole argument for the feature.

/** One paraphrase, one forward pass. */
export interface JudgePass {
  paraphrase: number;
  /** p(yes) AMONG the verdict tokens — which of the two, GIVEN it answered.
   *  Read it beside `mass`, never alone: the ratio between two rounding
   *  errors is still a number and still looks like a considered answer. */
  p_yes: number;
  p_no: number;
  /** How much of the model's whole probability mass landed on a verdict token
   *  at all. THIS is the number that says whether it answered. */
  mass: number;
  /** Above the server's floor. A `false` here is CARRIED rather than dropped —
   *  that this model does not answer this phrasing is a fact about the
   *  rubric — and it is NOT in the median. */
  answered: boolean;
}

/** Which surface forms of the verdict this tokenizer can express in one token.
 *
 *  Reported because it is a fact about the measurement: `" yes"` and `"yes"`
 *  are different ids, several casings can share one id, and the mass is summed
 *  over the DISTINCT ids found here.
 */
export interface JudgeTokens {
  yes_ids: number[];
  no_ids: number[];
  yes_forms: string[];
  no_forms: string[];
}

export interface JudgeScore {
  rubric: string;
  passes: JudgePass[];
  /** `null` when the verdict tokens were never resolved — not an empty set. */
  tokens: JudgeTokens | null;
  /** The judge's name, attached to every score. A small local model is a weak
   *  evaluator and a well-calibrated report of a weak judge's opinion is still
   *  a weak judge's opinion; the name is what lets a reader weigh it. Empty
   *  when the checkpoint did not name itself. */
  judge_model: string;
  dtype: string;
  device: string;
  /** `null` is "no seed was fixed", which is NOT seed 0. */
  seed: number | null;
  means: string;
  /** Absent when nothing was scored — never 0. The server writes these five
   *  only when there is at least one pass, so an absent `median` means there
   *  is no median rather than a median of zero. */
  low?: number;
  median?: number;
  high?: number;
  spread?: number;
  /** How many paraphrases are IN that median — the answered ones. Read it
   *  against `passes.length`, which is how many were run. */
  n_paraphrases?: number;
}

/** The prompts that would be run, before any of them is. No model is touched. */
export interface JudgePlan {
  prompts: string[];
  n_passes: number;
}

const JUDGE_NEEDS =
  "Reading a rubric off a model's probability mass is one forward pass per " +
  "paraphrase against weights this page does not hold.";

export const judgePlan = (body: {
  text: string;
  rubric: string;
  n_paraphrases: number;
}) =>
  DEMO || VIEWER
    ? noModelHere(
        JUDGE_NEEDS +
          " Pricing that run means listing the prompts it would make, and " +
          "there is nothing here to make them against.",
      )
    : fetch("/api/judge/plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then((r) => json<JudgePlan>(r));

export const judgeScore = (body: {
  text: string;
  rubric: string;
  n_paraphrases: number;
}) =>
  DEMO || VIEWER
    ? noModelHere(JUDGE_NEEDS)
    : fetch("/api/judge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then((r) => json<JudgeScore>(r));

// ------------------------------------------------------ robot dataset audit

/** One proof, its verdict, and the numbers behind it. */
export interface AuditCheck {
  name: string;
  /** `ok`, `broken` or `unchecked`. Three, deliberately: a check either proved
   *  the thing, proved the opposite, or could not be run on this machine, and
   *  collapsing those into a score is what a grade does. Typed as the server's
   *  own string so a verdict added later renders as itself, not as nothing. */
  verdict: string;
  detail: string;
  /** What it measured and what it compared against. The keys are each check's
   *  own, so this is rendered generically — and an `n_<name>` beside a
   *  `<name>` list is that list's TRUE length, which is how a capped list
   *  says it was capped. */
  measured: Record<string, unknown>;
}

export interface AuditReport {
  repo_id: string;
  /** `null` is UNKNOWN and is NOT 0. "This dataset has no frames" and "the
   *  frame table could not be read" are different answers, and the second is
   *  the whole reason somebody opened an audit. */
  n_episodes: number | null;
  n_frames: number | null;
  checks: AuditCheck[];
  seconds: number;
  means: string;
}

export const vlaAudit = () =>
  DEMO || VIEWER
    ? refusedHere(
        "Auditing a robot dataset reads the parquet, the video files and the " +
          "recorded statistics already on disk — nothing is downloaded, no " +
          "policy is loaded and no GPU is touched. This page is a static " +
          "bundle with no filesystem behind it, so there is nothing here to " +
          "prove intact. `pip install modelmri` and audit a dataset of your " +
          "own.",
      )
    : fetch("/api/vla/audit").then((r) => json<AuditReport>(r));

// ------------------------------------------------------ what it would DO
//
// All three need the action expert, which lives in a second process with its
// own venv because lerobot's pins cannot share an environment with this one.
// Each refuses BEFORE spending any forward passes whenever the answer would
// not have depended on them.

/** Forward passes before any are spent. Frames and passes, never seconds. */
export interface VLAActionCost {
  episode: number;
  frames_in_episode: number;
  frames_measured: number;
  /** What the stride will miss. A divergence between sampled frames is not in
   *  the chart, and this is how many chances it had to hide. */
  frames_skipped: number;
  stride: number;
  passes: number;
  means: string;
}

/** One frame's policy action beside the human's, and the gap between. */
export interface VLADivergence {
  t: number;
  predicted: number[];
  recorded: number[];
  /** Signed, per dimension. "The policy consistently reaches further" and "the
   *  policy is noisy" are different findings; an absolute value erases the
   *  first. */
  delta: number[];
  distance: number;
}

export interface VLACompare {
  rows: VLADivergence[];
  /** EMPTY when the dataset named no dimensions, or named a count that
   *  disagreed with the policy's width — the server drops the whole list
   *  rather than mislabelling one joint. */
  joint_names: string[];
  dimensions: number;
  frames_measured: number;
  frames_in_episode: number;
  stride: number;
  frames_skipped: number;
  worst_frame: number;
  worst_distance: number;
  /** Per-dimension mean of the SIGNED delta. Bias, not error. */
  bias: number[];
  policy_repo: string;
  revision: string;
  /** `null` means no seed was fixed, so re-running gives a different curve. */
  seed: number | null;
  means: string;
}

export interface VLASwapArm {
  instruction: string;
  is_own: boolean;
  action: number[];
  distance_from_own: number;
  /** Against the policy's OWN sampling spread, never a threshold from a paper
   *  about a different policy. */
  ratio_to_sampling: number;
}

export interface VLASwap {
  arms: VLASwapArm[];
  instruction_spread: number;
  sampling_spread: number;
  ratio: number;
  listens: boolean;
  seeds: number;
  instructions_tried: number;
  /** A CAP. Distinct instructions in this dataset that were NOT tried, so
   *  above zero the spread across instructions is a lower bound. */
  instructions_dropped: number;
  means: string;
}

export interface VLAKnockoutRow {
  stream: string;
  label: string;
  action: number[];
  distance: number;
  /** `null` when this policy's sampling spread could not be measured. The bar
   *  is still a real measurement; there is simply no denominator, and
   *  inventing one would be worse than leaving it out. NOT 0. */
  ratio_to_sampling: number | null;
  /** `null` for the same reason — "nothing here says whether this bar is
   *  larger than the policy's own noise", which is not `false`. */
  above_noise: boolean | null;
}

export interface VLAKnockout {
  rows: VLAKnockoutRow[];
  baseline: number[];
  streams: number;
  /** `null` when it could not be measured. */
  sampling_spread: number | null;
  means: string;
}

const POLICY_NEEDS =
  "This asks what the robot policy would DO, which runs the action expert in " +
  "its own process against a dataset on disk.";

export const vlaActionCost = (episode: number, stride: number) =>
  DEMO || VIEWER
    ? refusedHere(
        POLICY_NEEDS +
          " Pricing a run this page cannot make would describe a wait nobody " +
          "here is going to have.",
      )
    : fetch(`/api/vla/actions/cost?episode=${episode}&stride=${stride}`).then(
        (r) => json<VLAActionCost>(r),
      );

export const vlaCompareActions = (body: {
  episode: number;
  stride: number;
  seed: number | null;
}) =>
  DEMO || VIEWER
    ? refusedHere(
        POLICY_NEEDS +
          " One forward pass per sampled frame, and a baked curve would be a " +
          "fabricated comparison sitting beside real recordings. `pip install " +
          "modelmri` to run it on a policy of your own.",
      )
    : fetch("/api/vla/actions/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then((r) => json<VLACompare>(r));

export const vlaSwapInstruction = (body: {
  episode: number;
  t: number;
  seed: number | null;
}) =>
  DEMO || VIEWER
    ? refusedHere(
        POLICY_NEEDS +
          " It re-runs one frame under every distinct task string the dataset " +
          "contains, and again under several seeds, none of which this page " +
          "can do.",
      )
    : fetch("/api/vla/actions/swap", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then((r) => json<VLASwap>(r));

export const vlaKnockoutInputs = (body: {
  episode: number;
  t: number;
  seed: number | null;
}) =>
  DEMO || VIEWER
    ? refusedHere(
        POLICY_NEEDS +
          " It replaces each input in turn with that episode's mean and runs " +
          "the policy again, which needs the policy.",
      )
    : fetch("/api/vla/actions/knockout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then((r) => json<VLAKnockout>(r));

/* ══════════════════════════════════════════════════════════════════════════
   THE CONTROL, THE SAVED SWEEPS, AND A FINDING COUNTED OVER MANY RUNS
   ──────────────────────────────────────────────────────────────────────────
   Three tested routes with nothing on the page able to call them. Appended
   as a block rather than filed beside their neighbours above for a mechanical
   reason: several people are editing this 4,000-line module at once, and an
   insertion in the middle of it is how two of them lose each other's work.

   Same habits as the rest of the file: a `null` is an UNKNOWN and never a
   zero, a cap travels beside the thing it capped, and the server's own
   `means` sentence is carried through verbatim because that is where the
   caveats live.
   ══════════════════════════════════════════════════════════════════════════ */

/** The same head ranking, run again on an untrained twin of this architecture.
 *
 *  The question underneath every ranking in this tool: would this measurement
 *  have produced a confident, ordered list anyway? The twin is built from
 *  `config.json` alone — no weights fetched, works offline — seeded, and put
 *  through the IDENTICAL `rank_heads` over the same tokens. Both sides go
 *  through one function deliberately: a second implementation of the
 *  measurement could differ from the one being checked, and then agreement
 *  would mean nothing in either direction.
 */
export interface ControlRanking {
  /** The seed the twin's weights came from. Echoed by the server rather than
   *  chosen here, so the number on screen is the one that was used. */
  seed: number;
  baseline: string;
  /** The loaded model's ranking, re-run through the same public method — so
   *  it obeys every gate the panel's own ranking does. ONE layer. */
  model: Ablation;
  /** The untrained twin's ranking over the same tokens at the same position. */
  untrained: Ablation;
  /** Rank correlation between the two orderings.
   *
   *  NULL when the twin produced no ranking to correlate against — its scores
   *  were all equal. That is not zero: "the two are uncorrelated" and "one
   *  side is not a ranking at all" are different statements about the data,
   *  and rendering the second as 0.00 invents a measurement. */
  spearman: number | null;
  /** How many of the top heads were compared, and how many are in both. Both
   *  can legitimately be 0 — a ranking too short for a top-k has nothing to
   *  say about shared heads, and the verdict says so rather than implying it. */
  top_k: number;
  top_k_shared: number;
  /** The server's own reading of the two numbers above, thresholds stated.
   *  Shown verbatim: it is written to be disagreed with, beside the
   *  correlation it came from. */
  verdict: string;
}

/** Run the control. `seed` is omitted unless asked for, so the server's own
 *  default is what answers and the response says what it was. */
export const controlRanking = (
  layer: number,
  /** The baseline that produced the ranking on screen — taken from that
   *  ranking's own response, not from the panel's select, so the control is
   *  about what the reader is looking at. */
  baseline: string,
  seed?: number,
) =>
  DEMO || VIEWER
    ? noModelHere(
        "The control builds a SECOND model — this architecture with random " +
          "weights — and runs the identical head ranking over the same " +
          "tokens, so it costs two full sweeps and a second model resident " +
          "for the duration.",
      )
    : fetch(
        `/api/attention/control?layer=${layer}&baseline=${encodeURIComponent(
          baseline,
        )}` + (seed == null ? "" : `&seed=${seed}`),
      ).then((r) => json<ControlRanking>(r));

/** One sweep saved on this machine, and how far it got. */
export interface SavedSweep {
  sweep_id: string;
  /** ISO 8601 as the store wrote it, or "" for a row that carries none. An
   *  empty string is UNKNOWN and must not be rendered as an epoch date. */
  started_at: string;
  model: string;
  metric: string;
  n_prompts: number;
  n_measured: number;
  /** A prompt the measurement could not be taken on is a ROW, not a gap —
   *  see modelmri/sweep.py rule 2 — so this can be non-zero on a sweep that
   *  is nonetheless complete. */
  n_refused: number;
  n_remaining: number;
  complete: boolean;
}

/** Every sweep saved on this machine, newest first.
 *
 *  THE RESPONSE CARRIES NO TOTAL. `/api/sweeps` takes a `limit` and returns
 *  at most that many rows with no field saying how many exist — so a list of
 *  exactly `limit` rows is indistinguishable from a complete one, and the
 *  only honest thing a caller can do is state the cap it asked for. Its
 *  `means` sentence counts the rows it RETURNED, not the rows there are.
 */
export interface SweepList {
  sweeps: SavedSweep[];
  means: string;
}

export const savedSweeps = (limit: number) =>
  DEMO || VIEWER
    ? noModelHere(
        "Saved sweeps are read out of the trace database on your own " +
          "machine. This page is a static recording with no database behind " +
          "it, so any list here would be a list of somebody else's runs.",
      )
    : fetch(`/api/sweeps?limit=${limit}`).then((r) => json<SweepList>(r));

/** What finishing a stopped sweep would cost, and whether it may run at all. */
export interface ResumePlan {
  sweep_id: string;
  model: string;
  metric: string;
  n_prompts: number;
  n_measured: number;
  n_remaining: number;
  /** Which prompt indices still need running. Can be as long as the sweep. */
  remaining_indices: number[];
  /** NULL when nothing blocks it. A string is the reason this resume would be
   *  WRONG rather than merely expensive — a prompt set that has been edited, a
   *  row indexed past the end, a different model loaded now — and it is never
   *  a warning to override. */
  blocked: string | null;
  means: string;
}

export const sweepResumePlan = (sweepId: string) =>
  DEMO || VIEWER
    ? noModelHere(
        "A resume plan is priced against the saved rows of a sweep on your " +
          "machine, and checked against the model that is loaded now.",
      )
    : fetch(`/api/sweeps/${encodeURIComponent(sweepId)}/resume`).then((r) =>
        json<ResumePlan>(r),
      );

/** What a finished sweep looks like when a resume completes it. */
export interface ResumedSweep {
  sweep_id: string;
  model: string;
  metric: string;
  rows: unknown[];
  stats: unknown[];
  n_prompts: number;
  n_measured: number;
  /** Prompts still unmeasured AFTER the resume — a refusal stays a refusal.
   *  This is why a sweep containing one could be listed as unfinished
   *  forever: `remaining()` counts every unmeasured row as still-to-run. */
  n_unmeasured: number;
  means: string;
}

/** Finish a stopped sweep, keeping every prompt already measured.
 *
 *  `sweep.resume` was written, tested and PRICED — `sweepResumePlan` above
 *  answers what finishing would cost — and had no way to run: no route, no
 *  CLI flag, no button. The panel rendered "Nothing blocks this resume…
 *  Finishing it costs only the N prompt(s) below" beside no control at all.
 *
 *  The server re-checks `_resumable` itself, so the price and the run cannot
 *  disagree about whether it may proceed.
 */
export const sweepResume = (sweepId: string) =>
  DEMO || VIEWER
    ? noModelHere(
        "Finishing a sweep runs the model on the prompts it has not measured " +
          "yet, which needs one loaded on your own machine.",
      )
    : fetch(`/api/sweeps/${encodeURIComponent(sweepId)}/resume`, {
        method: "POST",
      }).then((r) => json<ResumedSweep>(r));

/** One structural finding, and how many of the recorded runs contain it. */
export interface RecurringFinding {
  kind: string;
  label: string;
  /** Runs this finding appears in, out of the runs that were READ. Twice in
   *  one run still counts as one run; `total_count` is the occurrences. */
  n_runs: number;
  of_runs: number;
  total_count: number;
  /** The finding's own identity — the input hash for a repeat, the name for a
   *  storm, the sequence for a cycle. What the grouping was done on. */
  signature: string;
  trace_ids: string[];
  means: string;
}

/** The same structural finding, counted over many recorded runs. */
export interface PatternsAcrossRuns {
  findings: RecurringFinding[];
  /** How many runs were actually read. */
  n_runs: number;
  /** How many there were to read. */
  n_runs_available: number;
  /** Available minus read. Non-zero means "12 of 19" is 12 of the 19 NEWEST,
   *  which is a different claim — the server reports it rather than leaving
   *  the caller to subtract. */
  truncated: number;
  /** The trace name this was narrowed to, echoed back. "" is every run. */
  name: string;
}

/** Count findings across runs. `name` narrows to one agent by trace name — a
 *  pattern in 12 of 19 runs of the SAME agent is a different claim from one
 *  seen across 19 unrelated runs. */
export const patternsAcross = (name: string, limit: number) =>
  DEMO || VIEWER
    ? Promise.reject(
        // Not `noModelHere`: what is missing on a static page is the database
        // of recorded runs, not a checkpoint, and a refusal that names the
        // wrong absent thing sends the reader to the wrong place. Same 409 and
        // same JSON shape, so `explain` and `errorText` treat it identically.
        new ApiError(
          409,
          JSON.stringify({
            error:
              "Counting a finding across runs queries every run recorded on " +
              "a machine. This page carries a single recording, so any " +
              "answer would be a pattern of one — install ModelMRI " +
              "(`pip install modelmri`) to run it over your own.",
          }),
        ),
      )
    : fetch(
        `/api/patterns/across?name=${encodeURIComponent(name)}&limit=${limit}`,
      ).then((r) => json<PatternsAcrossRuns>(r));

/** How unusual one frame is, against a NAMED reference set.
 *
 *  A distance and a percentile, never a verdict. "OOD" as a boolean would be
 *  a threshold somebody chose, and this project does not ship those. */
export interface OodFrame {
  t: number;
  /** Mahalanobis distance from the reference mean, over the directions that
   *  reference set actually varies in — so it is in units of that set's own
   *  spread and nothing else. */
  distance: number;
  /** Share of reference rows AT OR BELOW this distance. Exact, not read off
   *  the histogram. */
  percentile: number;
  /** What one row of the sample is worth, in percentage points. A percentile
   *  taken in a sample of N cannot be resolved finer than one row. */
  percentile_resolution: number;
  /** Further out than EVERY reference row. The percentile then reads 100.0,
   *  which otherwise looks like "measured at exactly the top". */
  beyond_reference_max: boolean;
  /** Movement along directions the reference set never varied in, in the
   *  column's own raw units — a different quantity on a different scale, so
   *  it is beside the distance rather than folded into it. `null` when the
   *  reference varied everywhere. */
  off_manifold: number | null;
  /** Beat the largest distance any held-out in-distribution row reached.
   *  `null` — never `false` — when no null could be drawn. */
  clears_null: boolean | null;
}

export interface OodDistances {
  min: number;
  max: number;
  mean: number;
  count: number;
  percentile_levels: number[];
  percentiles: { q: number; value: number }[];
  /** Nearest rank over every reference distance, held exactly. These are NOT
   *  read off the histogram below, which exists for drawing. */
  percentile_method: string;
  histogram: { bin_edges: number[]; counts: number[]; bin_width: number };
}

/** What the distances were measured against. Without this the number means
 *  nothing, which is why it travels in every payload rather than in a tooltip. */
export interface OodReference {
  repo_id: string;
  space: string;
  /** Width taken from the first readable row, beside what `info.json` claims.
   *  Both, because when they disagree one is wrong and nothing here can say
   *  which. */
  dimensions: number;
  dimensions_declared: number | null;
  rows_total: number;
  rows_with_column: number;
  rows_eligible: number;
  rows_read: number;
  rows_malformed: number;
  /** 1 when every eligible row was read. Above 1 this is a SAMPLE. */
  row_stride: number;
  sampled: boolean;
  /** Held OUT of the reference, its mean, its covariance and its null — a
   *  frame compared against a distribution it helped define is partly
   *  measuring itself. */
  excluded_episode: number | null;
  excluded_rows: number;
  /** The dataset's own row map, or summed episode lengths — the second is an
   *  assumption and says so. */
  row_span_from: string;
  rows_per_dimension: number;
  directions_kept: number;
  directions_dropped: number;
  variance_floor: number;
  condition_ratio: number;
  metric: string;
  distances: OodDistances;
  percentile_resolution: number;
  /** The largest distance reached by rows drawn from this same dataset and
   *  held out of the reference. `null` when none could be drawn. */
  null_max: number | null;
  null_draws: number;
  /** Where the maximum of `null_draws` draws is EXPECTED to sit, from the
   *  draw count alone. The null's RESOLUTION — not a reading of this null. */
  null_covers_percentile: number | null;
  /** Where `null_max` ACTUALLY landed among the reference distances. These
   *  two are different quantities and are not expected to agree. */
  null_max_percentile: number | null;
  null_max_beyond_reference_max: boolean | null;
  null_position_caveat: string;
  null_reason: string;
  null_description: string;
}

export interface EpisodeOod {
  repo_id: string;
  episode: number;
  space: string;
  frame_stride: number;
  reference: OodReference;
  /** Every scored frame in TIME order — what a per-frame chart plots. */
  frames: OodFrame[];
  /** The most distant, CAPPED. `n_ranked_total` is how many were scored, so
   *  the cap is reported rather than only applied. */
  ranked: OodFrame[];
  n_ranked_total: number;
  n_frames: number;
  frames_total: number;
  /** A sample of what could not be scored, each with a sentence. Capped;
   *  `n_unscored` carries the true count. */
  unscored: { t: number; why: string }[];
  n_unscored: number;
  seconds: number;
  means: string;
}

/** What that read will cost, before it reads it.
 *
 *  `forward_passes` is 0 and says why: nothing here runs a model. The
 *  reference set is the dataset's own recorded vectors and the distance is
 *  arithmetic over them, so the cost is parquet rows. */
export interface EpisodeOodCost {
  repo_id: string;
  episode: number;
  space: string;
  frame_stride: number;
  forward_passes: number;
  forward_passes_why: string;
  cost_unit: string;
  passes_over_the_data: number;
  passes_why: string;
  rows_total: number;
  rows_with_column: number;
  rows_eligible: number;
  rows_to_read: number;
  rows_in_episode: number;
  rows_in_episode_shards: number;
  row_stride: number;
  sampled: boolean;
  /** A ceiling, not a count: the actual figure depends on what is readable. */
  reference_rows: number;
  reference_rows_is_a_ceiling: boolean;
  null_draws: number;
  null_draws_is_a_ceiling: boolean;
  null_reason: string;
  frames: number;
  frames_total: number;
  files_total: number;
  files_with_column: number;
  dimensions_declared: number | null;
  row_span_from: string;
  readability_why: string;
  peak_bytes: number;
  peak_basis: string;
  /** `null` when nothing has timed this machine's parquet reads — this is
   *  disk-bound, and a duration measured on somebody else's disk is a number
   *  people plan around. `seconds_from` says which of those it is. */
  seconds: number | null;
  seconds_from: string;
}

export const getEpisodeOod = (episode: number, space: string, frameStride = 1) =>
  fetch(
    `/api/vla/ood?episode=${episode}&space=${encodeURIComponent(space)}` +
      `&frame_stride=${frameStride}`,
  ).then((r) => json<EpisodeOod>(r));

export const getEpisodeOodCost = (episode: number, space: string, frameStride = 1) =>
  fetch(
    `/api/vla/ood/cost?episode=${episode}&space=${encodeURIComponent(space)}` +
      `&frame_stride=${frameStride}`,
  ).then((r) => json<EpisodeOodCost>(r));

// ------------------------------------------------ anchors: what HOLDS it
//
// The other half of `/api/attention/attribute`, and the pair is the point:
// that route masks a token out and measures what breaks (necessity), this one
// keeps a few and perturbs the rest (sufficiency). A token can be necessary
// and not sufficient. Both are real, they disagree, and a reader shown one of
// them alone will read it as both.

/** A proportion with the interval it was measured to, never a bare point.
 *
 *  `measured: false` means the denominator was zero — `reason` says which —
 *  and `point` is then meaningless rather than 0. */
export interface Proportion {
  /** False when nobody sampled it. EVERY field below except `samples` and
   *  `reason` is then `null` — `anchors._unmeasured` is explicit that the one
   *  thing it must never do is put a 0 or a 1 where a measured proportion
   *  goes. Read `point` without checking this and a proportion nobody took
   *  renders as 0.000. */
  measured: boolean;
  held: number | null;
  samples: number;
  point: number | null;
  low: number | null;
  high: number | null;
  confidence: number | null;
  method: string | null;
  reason: string;
  /** What the arithmetic implies when there was nothing to sample — a
   *  SEPARATE key from `point` precisely so a reader and a chart can tell
   *  arithmetic from evidence. */
  implied?: number | null;
  /** Ceiling only, and absent (not 0) on the paths that never measured one.
   *  `undefined` means "not measured"; 0 means "measured, nothing outside the
   *  candidate set was perturbed". */
  n_perturbed?: number;
}

export interface AnchorToken {
  index: number;
  token: string;
}

export interface Anchors {
  position: number;
  target_token: string;
  target_token_id: number;
  /** The tokens kept fixed. EMPTY with `found: false` is a real answer rather
   *  than a failure to report one, and `stopped_because` says which. */
  anchor: AnchorToken[];
  anchor_indices: number[];
  found: boolean;
  size: number;
  /** Each step of the greedy search, in the order it took them.
   *
   *  These are the route's own field names. An earlier version of this
   *  interface declared `{index, token, precision, size}` — four keys the
   *  route has never sent — so `steps[i].precision` was `undefined` at every
   *  call site that trusted it. */
  steps: {
    step: number;
    added_index: number;
    added_token: string;
    /** The candidates screened at this step, CAPPED at five with the count
     *  and the cap beside it. */
    screened: ({ index: number; token: string } & Proportion)[];
    n_screened: number;
    screened_truncated: boolean;
    screen_samples: number;
    verified: Proportion;
  }[];
  /** How often the prediction survived with only the anchor held. */
  precision: Proportion;
  /** How often it survived with NOTHING held — the floor any anchor has to
   *  beat, measured rather than assumed to be zero. */
  base_rate: Proportion;
  /** The best any anchor could do here. A target above this cannot be met at
   *  any size, which is a fact about the prompt, not about the search. */
  ceiling: Proportion;
  target: number;
  /** The highest target these draws could ever certify, from the Wilson bound
   *  alone. A target above it is refused rather than searched for forever. */
  target_ceiling: number;
  target_ceiling_exact: number;
  stopped_because: string;
  minimality: {
    search: string;
    smaller_may_exist: boolean;
    drop_one_checked: boolean;
    irreducible_under_single_removal: boolean | null;
    removed_by_elimination: number[];
    /** One backward-elimination attempt. The proportion is SPREAD into the
     *  entry rather than nested, which is why this extends `Proportion`. */
    drops: ({
      dropped_index: number;
      dropped_token: string;
      anchor_at_the_time: number[];
      still_sufficient: boolean;
      removed: boolean;
      /** Non-empty when this drop reused an existing measurement rather than
       *  paying for a new one. */
      reused: string;
    } & Proportion)[];
    note: string;
  };
  /** What the replacements were drawn from, and whether that pool was varied
   *  enough for the answer to mean anything. */
  perturbation: {
    replaces: string;
    corpus: string;
    vocabulary: {
      source: string;
      size: number;
      declared: number | null;
      measured: number;
    };
    pool_size: number;
    pool_distinct_ids: number;
    distinct_ids: number;
    distinct_templates: number;
    min_distinct_ids: number;
    draws_in_pool: number;
    samples: number;
    screen_samples: number;
    control_ids_dropped: number;
    /** A SENTENCE, not a flag. Declared `boolean` here once, which made
     *  `if (p.paired)` unconditionally true and would have printed a
     *  paragraph anywhere it was rendered as one. */
    paired: string;
    weighting: string;
    seed: number;
    sentence: string;
    /** How many distinct templates the corpus holds — a COUNT, not the
     *  sentences themselves. `.map` over it would have thrown. */
    sentences: number;
    quality: {
      distinct_ids: number;
      distinct_templates: number;
      below_min_distinct_ids: boolean;
      templates_repeat: boolean;
      note: string;
    };
  };
  candidates: {
    n_candidates: number;
    n_tested: number;
    truncated: boolean;
    tested_span: number[];
    max_candidates: number;
    max_size: number;
    coverage: string;
  };
  /** Positions that could never be part of an anchor, and why. */
  held_fixed: { count: number; indices: number[]; listed: number; why: string };
  accounting: {
    free_evaluations: number;
    covering_final_step: boolean;
    passes_with_another_mask: number;
    passes_with_another_position_ids: number;
    why: string;
  };
  /** `null` when there is no second token to take a margin against — a
   *  vocabulary of one. Never 0, which would say the top two tokens tied. */
  base_margin: number | null;
  base_p_top: number;
  /** How far the unperturbed run drifts from itself, and how far the anchor
   *  run sits from it. The first is the floor under the second. */
  noise_floor_kl: number;
  agreement_kl: number;
  passes: number;
  seed: number;
  elapsed_s: number;
  means: string;
  receipt?: Receipt | null;
}

export const tokenAnchors = (body: {
  position?: number;
  max_candidates?: number;
  max_size?: number;
  n_samples?: number;
  target?: number;
  seed?: number;
}) =>
  DEMO || VIEWER
    ? noModelHere(
        "Finding the smallest set of your prompt's tokens that holds the " +
          "answer means perturbing the rest of them and re-running the model " +
          "once per draw — 83 forward passes on the narrowest search this " +
          "offers, and thousands on a wide one.",
      )
    : fetch("/api/attention/anchors", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then((r) => json<Anchors>(r));

// ------------------------- the smallest edit that changes the answer

/** One control arm: a count out of a sample, never a bare rate.
 *
 *  `measured: false` means the arm took NO draws — every one was abandoned, or
 *  none was requested — and `successes` is then 0 out of 0, which is an
 *  absence. Rendering it as 0% would read as "we checked and it never
 *  happened", which is the strongest possible support for a finding nobody
 *  measured. */
export interface ControlArm {
  measured: boolean;
  successes: number;
  samples: number;
  /** `null` whenever `measured` is false. Never 0. */
  point: number | null;
  /** `[low, high]`, or `null` when nothing was drawn. */
  interval: [number, number] | null;
  confidence: number;
  method?: string;
}

/** One committed substitution. */
export interface CounterfactualStep {
  step: number;
  index: number;
  from_token_id: number;
  /** Always a string: the decoded token, or the bare id when no decoder was
   *  given. Never null — `anchors.py` uses the same contract, and a second
   *  nullable case is a second branch in every renderer. */
  from_token: string;
  to_token_id: number;
  to_token: string;
  target_p_after: number;
}

export interface Counterfactual {
  position: number;
  base_token_id: number;
  base_token: string;
  target_token_id: number;
  target_token: string;
  /** What the target was worth before any edit: its probability and its rank.
   *  Rank 1 is impossible here — a target the model already predicts is
   *  refused rather than answered with an empty edit. */
  base_target_p: number;
  base_target_rank: number;
  found: boolean;
  /** Which bound stopped the search, as a sentence. */
  stopped_because: string;
  edit: CounterfactualStep[];
  size: number;
  /** The prompt with the edit applied, every index preserved. Usable as the
   *  corrupt half of a patching pair. */
  edited_ids: number[];
  edited_ids_are: string;
  edited_text?: string;
  /** The closest the search came when it did NOT succeed. `null` when nothing
   *  was committed — which is the correct answer when no substitution raised
   *  the target at all, not a zero-size edit dressed up as an attempt. */
  best_effort: { size: number; target_p: number; reached_target: boolean } | null;
  controls: {
    same_positions: ControlArm;
    same_positions_asks: string;
    any_positions: ControlArm;
    any_positions_asks: string;
    measured: boolean;
    not_measured_because: string | null;
    draws_requested_per_arm: number;
    draws_abandoned: number;
    no_self_substitution: string;
  };
  /** True only when BOTH arms were measured and BOTH came back empty. */
  beats_controls: boolean;
  beats_controls_means: string;
  candidates: string;
  screen: {
    source: string;
    proposals_per_step: number | null;
    backward_passes: number;
    /** How often the first-order estimate's top choice actually won its step.
     *  A screen that never agrees is a screen to stop trusting. */
    top_choice_won: ControlArm;
    top_choice_won_asks: string;
  };
  minimality: {
    search: string;
    smaller_may_exist: boolean;
    positions_considered: number;
    donors_per_step: number;
  };
  trials_skipped_self: number;
  trials_short_circuited: number;
  trials_unavailable: number;
  passes: number;
  passes_expected: number;
  seconds: number;
  seed: number;
  target_named?: string | null;
  receipt?: Receipt | null;
}

export const tokenCounterfactual = (body: {
  position?: number;
  target?: string;
  target_token_id?: number;
  max_edits?: number;
  n_proposals?: number;
  n_controls?: number;
  candidates?: string;
  seed?: number;
}) =>
  DEMO || VIEWER
    ? noModelHere(
        "Finding the smallest edit that makes a model say something else " +
          "means re-running it once per candidate substitution, then once per " +
          "control draw — 78 forward passes at the default budget on " +
          "Qwen3-1.7B and 267 at four edits with a wider shortlist, both " +
          "measured — and there is no model behind this page.",
      )
    : fetch("/api/attention/counterfactual", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then((r) => json<Counterfactual>(r));

// ------------------------------- integrated gradients, with the gap named

/** One token's share of the move, with whether it can be read at all. */
export interface GradientToken {
  index: number;
  token: string;
  token_id: number;
  /** `null` when the backward pass came back non-finite. NEVER 0 — the module
   *  is explicit that a bar it could not compute must not be drawn as one it
   *  computed to be zero, and `unreadable` is how a renderer tells them
   *  apart. */
  attribution: number | null;
  /** `null` when there is nothing to take a share OF, and null for every
   *  token when the total itself is not a number. */
  share: number | null;
  /** True when this bar is under the completeness gap, i.e. its share cannot
   *  be told from the error in the approximation. */
  unreadable: boolean;
}

/** Do the attributions add up to what actually happened?
 *
 *  A bar chart with no gap beside it cannot be told from a converged one, so
 *  all three numbers travel always. */
export interface Completeness {
  steps: number;
  rule: string;
  /** `null` when non-finite. That is not a corner case here — it is exactly
   *  the state the `undefined` verdict names, so any arithmetic on these
   *  three has to check first. */
  sum_of_attributions: number | null;
  measured_delta: number | null;
  gap: number | null;
  /** `null` when the move is under `endpoint_floor` — there is no share of a
   *  quantity that could not be resolved, and 0 would read as "no gap". */
  gap_share: number | null;
  /** How far two repeats of the same two forward passes moved on their own.
   *  The resolution of `measured_delta`, and the floor under which a gap
   *  cannot be told from running the same forward twice. */
  endpoint_floor: number;
  verdict: "converged" | "approximate" | "diverged" | "undefined";
  sentence: string;
}

export interface TokenGradients {
  position: number;
  target_token: string;
  target_token_id: number;
  target_kind: string;
  baseline: string;
  /** What the baseline actually was, in words: "pad" is not the same sentence
   *  on two tokenizers and the reader cannot see the id from here. */
  baseline_note: string;
  tokens: GradientToken[];
  n_tokens: number;
  n_listed: number;
  n_unreadable: number;
  n_nonfinite: number;
  sum_of_absolute_attributions: number | null;
  completeness: Completeness;
  forward_passes: number;
  backward_passes: number;
  /** `null` on EVERY CPU run: `torch.cpu` publishes no
   *  `max_memory_allocated`, so there is no allocator peak to read. Rendering
   *  `null / 1e6` prints "0.0 MB held" beside a `peak_note` that says the
   *  memory was not measured — the payload contradicting itself in one line. */
  peak_bytes: number | null;
  peak_note: string;
  elapsed_s: number;
  means: string;
  receipt?: Receipt | null;
}

export const tokenGradients = (body: {
  position?: number;
  baseline?: string;
  target_kind?: string;
  steps?: number;
  on_gap?: string;
}) =>
  DEMO || VIEWER
    ? noModelHere(
        "Integrated gradients runs a forward AND a backward pass at every " +
          "step of the path from the baseline to your prompt, against a live " +
          "model holding its own graph.",
      )
    : fetch("/api/attention/gradients", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then((r) => json<TokenGradients>(r));

// -------------------------------------------- screening the patching grid

/** One site the screen ranked, and — for the shortlisted few — what the exact
 *  grid said about it. */
export interface ScreenSite {
  name: string;
  component: string;
  layer: number;
  position: number;
  /** The SCREEN's number. Deliberately not called `recovery`: it is a
   *  first-order approximation, and the field names keep it structurally
   *  distinguishable from `/api/patch`'s exact one. */
  attribution: number;
  grad_norm: number;
  delta_norm: number;
  /** `null` on a site the exact grid was never run on. Never 0, which would
   *  say it was measured and recovered nothing. */
  exact_recovery: number | null;
  exact_error: number | null;
}

export interface ScreenPrompt {
  prompt: string;
  tokens: string[];
  answer: { id: number; text: string; p: number };
}

export interface PatchScreen {
  clean: ScreenPrompt;
  corrupt: ScreenPrompt;
  method: string;
  /** In the payload rather than in a docstring: these numbers are not
   *  `/api/patch`'s and must never be read as though they were. */
  approximate: boolean;
  n_layers: number;
  n_positions: number;
  n_sites_scored: number;
  n_sites_nonfinite: number;
  nonfinite_sites: string[];
  /** Per-component grids for drawing. Screen values throughout. */
  screen_grids: { resid: number[][]; attn: number[][]; mlp: number[][] };
  shortlist: ScreenSite[];
  shortlist_size: number;
  /** What the caller asked for. */
  shortlist_requested: number;
  /** How many sites were SCORED — the pool the shortlist was taken from, not
   *  the number requested. Reading it as the request produced "12 of 486
   *  requested" on a dial set to 12. Never null. */
  shortlist_capped_from: number;
  /** Sites the screen called near zero, verified anyway — a screen checked
   *  only where it already agrees has not been checked. */
  near_zero_probes: ScreenSite[];
  near_zero_requested: number;
  near_zero_capped_from: number | null;
  strongest_negative: ScreenSite | null;
  strongest_negative_on_shortlist: boolean;
  strongest_negative_reason: string;
  /** How far the screen agreed with the exact grid where both ran. A screen
   *  whose agreement was never measured is a guess with a leaderboard. */
  agreement: {
    verified: number;
    spearman: number | null;
    spearman_reason: string;
    spearman_resolution: number | null;
    sign_flips: number;
    worst_rank_move: number | null;
    largest_disagreement: number | null;
    largest_disagreement_at: string | null;
    largest_disagreement_screen: number | null;
    largest_disagreement_exact: number | null;
    largest_disagreement_scope: string;
    largest_disagreement_measured_on: string;
    largest_disagreement_verified_only: number | null;
    largest_disagreement_verified_only_at: string | null;
    exact_recovery_resolution: number | null;
    near_zero_probed: number;
    near_zero_sign_flips: number;
    near_zero_largest_screen: number | null;
    near_zero_largest_exact: number | null;
    near_zero_largest_exact_at: string | null;
    nonfinite_exact: number;
    nonfinite_exact_at: string[];
    means: string;
  };
  /** What it cost and what the exact grid WOULD have, so the saving is a
   *  measurement rather than a claim. */
  cost: {
    screen_forward_passes: number;
    screen_backward_passes: number;
    verification_passes: number;
    exact_grid_passes: number;
    exact_trace_passes: number;
    exact_passes_basis: string;
    shortlist_remaining_passes: number;
    passes_saved_against_exact_grid: number;
    passes_saved_against_exact_trace: number;
    seconds: number | null;
    seconds_basis: string;
    seconds_gradient_pass: number | null;
    seconds_per_exact_pass: number | null;
    seconds_exact_grid_projected: number | null;
    seconds_saved_projected: number | null;
    memory: {
      peak_bytes: number | null;
      free_bytes: number | null;
      total_bytes: number | null;
      source: string;
      reason: string;
    };
    activation_bytes_held: {
      clean_cache: number;
      corrupt_values: number;
      grads: number;
      taps: number;
      total: number;
      patch_trace_equivalent: number | null;
      ratio_vs_patch_trace: number | null;
      means: string;
    };
    means: string;
  };
  gap: number;
  dtype: string;
  seeding: string;
  skipped: string[];
  notes: string[];
  means: string;
  receipt?: Receipt | null;
}

export const patchScreen = (body: {
  clean: string;
  corrupt: string;
  shortlist?: number;
  verify?: number;
}) =>
  DEMO || VIEWER
    ? noModelHere(
        "Screening the patching grid runs one gradient pass over a live " +
          "model and then re-runs it once per shortlisted site, to check the " +
          "screen against the exact answer rather than assert it.",
      )
    : fetch("/api/patch/screen", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then((r) => json<PatchScreen>(r));

// --------------------------------------- one MLP neuron, on your own text

export interface NeuronSpan {
  position: number;
  token: string;
  /** Character offset inside `text`. Not derivable by searching: a window can
   *  hold the same token twice and only one of them fired. */
  offset: number;
  activation: number;
  sequence: number;
  text: string;
}

export interface NeuronEvidence {
  neuron: number;
  layer_width: number;
  /** ALWAYS `null`, deliberately. Not the corpus's name — this is what the
   *  neuron REPRESENTS, and `neurons.py` is explicit that it stays null
   *  because "a generated label would be the one thing on the page that
   *  nothing measured". It is in the payload rather than absent so that a
   *  reader can see the field exists and is empty on purpose.
   *
   *  There is deliberately no `layer` here either: the route does not publish
   *  one. A panel that labels this answer with its own layer dial relabels it
   *  the moment the dial moves, so the requested layer has to be snapshotted
   *  beside the payload instead. */
  label: string | null;
  n_sequences: number;
  n_tokens: number;
  n_fired: number;
  n_negative: number;
  n_finite: number;
  n_nonfinite: number;
  /** `null` when nothing finite was read — never 0, which would say the
   *  neuron was measured and sat still. */
  max_activation: number | null;
  min_activation: number | null;
  mean_positive: number | null;
  firing_rate: number | null;
  /** This neuron's rate against its own layer's, which is the only scale a
   *  firing rate means anything on. */
  layer_median_firing_rate: number | null;
  histogram: number[];
  bin_edges: number[];
  spans: NeuronSpan[];
  n_spans_available: number;
  n_spans_shown: number;
  /** The caveat that is the whole reason sparse autoencoders exist, in the
   *  payload rather than in a tooltip. */
  polysemantic: string;
  means: string;
  receipt?: Receipt | null;
}

export const neuronEvidence = (body: {
  texts?: string[];
  file?: string;
  label?: string;
  neuron?: number;
  layer?: number;
  top_k?: number;
}) =>
  DEMO || VIEWER
    ? noModelHere(
        "Reading what one MLP neuron fires on means running YOUR text " +
          "through a live model and tapping that layer, once per sequence.",
      )
    : fetch("/api/neurons/evidence", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then((r) => json<NeuronEvidence>(r));

// -------------------------------------- an image run as a `.mri` (A6)

/** One decoded frame of a denoising run, or the picture a readout read.
 *
 *  `size` is the resolution this frame IS, and `decoded_size` — present only
 *  when `downsampled` — is the resolution it came from. Both are required by
 *  `session._image` for the same reason a robot frame's is: a cross-attention
 *  map is drawn over the picture, and a picture silently resized puts every
 *  cell in the wrong place. Wrong in the way that looks like a finding.
 */
export interface ImageRunFrame {
  step: number;
  /** `null` when the scheduler published none — never 0, which is a real
   *  timestep at the end of a schedule. */
  timestep: number | null;
  png: string;
  size: [number, number];
  downsampled: boolean;
  decoded_size?: [number, number];
  /** `null` when it was not measured. */
  latent_rms: number | null;
}

export interface ImageRunAttention {
  tokens: string[];
  steps: {
    step: number;
    timestep: number | null;
    per_token: number[];
    blocks: number;
  }[];
  /** Where the padding starts. CLIP pads to 77 and the padded tail carries
   *  real attention mass, which is a genuine finding and a terrible chart.
   *
   *  `null` when the file never measured the boundary, and NOT 0 — 0 is the
   *  claim that the padding starts at column zero, so every measured column
   *  is `<pad>` and none of them is the prompt. That is the exact conclusion
   *  this field exists to prevent. */
  padding_from: number | null;
  conditioning_width: number;
  /** Columns that were MEASURED and have no label to put on them. A cap on
   *  what can be shown, not on what was measured. */
  columns_unlabelled: number;
  steps_requested: number;
  steps_measured: number;
  resolutions: number[];
  means: string;
}

/** What a classifier, detector or segmenter said.
 *
 *  `kind` is not decoration: a classifier's probability, a detector's
 *  per-query confidence and a segmenter's share of the map are three
 *  different quantities that all render as a number between 0 and 1. The
 *  server refuses a readout that does not say which, so nothing here can be
 *  read side by side as though it compared.
 */
export interface ImageRunReadout {
  kind: string;
  rows: {
    label: string;
    score: number;
    index: number | null;
    query: number | null;
    /** `null`, never a zero rectangle — a box at the origin with no width is
     *  drawable, and it would be drawn. */
    box_xyxy: [number, number, number, number] | null;
  }[];
  means: string;
}

/** An image run carried inside a `.mri`.
 *
 *  A6, and the last unbuilt item in Theme A: every other result this tool
 *  produces could be sent to somebody, and the one that is a PICTURE could
 *  not — so an image finding was the only kind that had to be screenshot to
 *  be shared, and a screenshot carries no provenance, no seed and no
 *  statement of what was shrunk.
 */
export interface ImageRunSection {
  provenance: {
    repo: string;
    family: string;
    /** `""` is a CLAIM — "this checkpoint's config published no
     *  `architectures`" — and the server accepts it, exactly as it accepts an
     *  empty `revision`. */
    architecture: string;
    /** `""` is a CLAIM — "this checkpoint published none" — and the server
     *  accepts it. What it refuses is the field being absent, which cannot be
     *  told apart from nobody having looked. */
    revision: string;
    kind: string;
  };
  prompt: string;
  /** `null` means NO SEED WAS FIXED, which is not seed 0: rerun it and the
   *  trajectory differs, and nothing downstream compares. */
  seed: number | null;
  scheduler: string;
  frames?: ImageRunFrame[];
  png_bytes_total?: number;
  /** `null` when the file does not say — never 0. "3 of 0 step(s) decoded"
   *  is a sentence about a run that cannot have happened. */
  steps_requested: number | null;
  steps_run: number | null;
  decoded_steps: number[];
  /** Ran and was not decoded — a CHOICE. */
  skipped_steps: number[];
  /** Selected and never arrived — a GAP. Kept apart from the line above
   *  because a strip that folded them together reads as eight of fifty
   *  either way. */
  steps_never_reached: number[];
  attention?: ImageRunAttention;
  readout?: ImageRunReadout;
  means: string;
}

/** The image run inside an opened `.mri`, or nothing.
 *
 *  `available: false` is a STATE and not an error: most sessions carry no
 *  image run, and a panel that treated it as one would render "this
 *  measurement is broken" for the ordinary case.
 */
export const getImageReplay = () =>
  fetch("/api/image/replay").then((r) =>
    json<{ available: boolean } & Partial<ImageRunSection>>(r),
  );

/** A spread as it arrives inside a `.mri`.
 *
 *  NOT `DiffSpread`, which is the LIVE shape and requires every field. This
 *  one comes out of a file a stranger wrote: `session._model_diff` refuses a
 *  median with no `n` behind it -- the whole content of this section is that
 *  its numbers are distributions over a prompt set rather than single
 *  measurements -- but `n_nonzero` is genuinely optional there, so it is
 *  optional here. Declaring a second `DiffSpread` merged with the first and
 *  produced a type that lied about both.
 */
export interface RecordedSpread {
  n: number;
  name: string;
  median: number;
  low: number;
  high: number;
  /** How many of the `n` prompts moved at all. A median of 0 with most
   *  prompts nonzero is a different finding from one where nothing moved. */
  n_nonzero?: number;
}

/** A comparison of two models, carried inside a `.mri`.
 *
 *  `runtime` wrote this into every export that had a comparison behind it and
 *  nothing read it back on any surface -- the one section in the format with
 *  no reader anywhere.
 */
export interface ModelDiffSection {
  /** REQUIRED, both of them. A diff can ride in a file about a third model,
   *  so one that does not name its own two sides is read as being about the
   *  file's own -- the single confusion this section can cause. */
  model_a: string;
  model_b: string;
  prompts: {
    prompt?: string;
    n_tokens?: number;
    mean_kl?: number;
    max_kl?: number;
    flips?: number;
    /** `null` when the cosine never falls -- a result, not a gap. */
    first_divergent_layer?: number | null;
    drop?: number | null;
  }[];
  layers: {
    layer?: number;
    median?: number;
    low?: number;
    high?: number;
    n?: number;
    n_first?: number;
  }[];
  heads: {
    layer?: number;
    head?: number;
    median_a?: number;
    median_b?: number;
    shift?: number;
    n?: number;
    top_a?: string | null;
    top_b?: string | null;
  }[];
  tokens: {
    prompt_index?: number;
    index?: number;
    token?: string;
    kl_a?: number;
    kl_b?: number;
    shift?: number;
    newly_used?: boolean;
    newly_ignored?: boolean;
  }[];
  kl?: RecordedSpread;
  flips?: RecordedSpread;
  n_prompts?: number | null;
  n_layers?: number | null;
  /** `null` when nothing diverged. A result. */
  consensus_layer?: number | null;
  consensus_share?: number;
  head_passes?: number | null;
  seconds?: number;
  means?: string;
}

/** The model comparison inside an opened `.mri`, or nothing. */
export const getDiffReplay = () =>
  fetch("/api/diff/replay").then((r) =>
    json<{ available: boolean } & Partial<ModelDiffSection>>(r),
  );

/** One occlusion block: a patch of the frame, covered, and what the action
 *  did about it.
 *
 *  `control_max` and `clears_control` are NULLABLE and the null is the whole
 *  measurement. An uncontrolled block has no control maximum, and 0.0 there
 *  would read as "a random occlusion moved the action not at all" — a claim
 *  nobody made.
 */
export interface VlaBlock {
  row?: number;
  col?: number;
  /** How far the action moved when this patch was covered. Never null: every
   *  block was measured, so a missing shift is a broken row rather than an
   *  honest unknown, and the server refuses one. */
  shift: number;
  /** The largest shift a RANDOM occlusion produced. `null` when this block
   *  was not controlled. */
  control_max?: number | null;
  /** Whether the real occlusion beat its own control. `null` is "not
   *  controlled", which is not "did not clear it". */
  clears_control: boolean | null;
  control_draws?: number;
  /** The policy's attention on this patch, when the map was compared. */
  attention?: number | null;
}

/** A robot finding carried inside a `.mri`.
 *
 *  `/api/vla/share` wrote this from the day the feature landed and nothing
 *  served it back, so a shared robot finding opened as an empty text session.
 *  The reader (`session._vla`) was there the whole time.
 */
export interface VlaFindingSection {
  provenance: {
    policy: string;
    dataset: string;
    camera: string;
    revision: string;
    episode: number;
    timestep: number;
  };
  frame?: string;
  frame_size?: [number, number];
  /** Stated, never inferred. `false` is the positive claim "this is the
   *  resolution the policy saw" — a causal map is drawn over this frame, so a
   *  silently shrunk one puts every block in the wrong place. */
  frame_downsampled?: boolean;
  frame_note?: string;
  attention?: number[][][];
  occlusion?: {
    blocks: VlaBlock[];
    baseline: string;
    means?: string;
    grid?: number[];
    stride?: number;
    n_blocks?: number;
    n_controlled?: number;
    passes?: number;
    /** Which attention map the agreement was measured against. `null` is
     *  "not compared", which a reader must be able to tell from "layer 0". */
    compared_layer: number | null;
    compared_head: number | null;
    scale?: number | null;
    /** Correlation between the policy's attention and what actually moved the
     *  action. Negative is a real and common finding, not an error. */
    attention_agreement?: number | null;
  };
}

/** The robot finding inside an opened `.mri`, or nothing.
 *
 *  `available: false` is a STATE and not an error: most sessions carry no
 *  robot finding.
 */
export const getVlaReplay = () =>
  fetch("/api/vla/replay").then((r) =>
    json<{ available: boolean } & Partial<VlaFindingSection>>(r),
  );

/** What a share would carry, before it is asked for.
 *
 *  Priced before it is spent like every other measurement here — except the
 *  currency is BYTES, because the run has already happened and the only
 *  remaining cost is the size of the file somebody is about to attach to an
 *  issue.
 */
export const getImageSharePlan = () =>
  DEMO || VIEWER
    ? noModelHere(
        "Pricing a share means reading the run this machine last made, and " +
          "there is no image model behind this page to have made one.",
      )
    : fetch("/api/image/share/plan").then((r) =>
        json<{
          available: boolean;
          kind?: string;
          repo?: string;
          n_frames?: number;
          png_bytes?: number;
          n_attention_steps?: number;
          n_readout_rows?: number;
          seed?: number | null;
          /** THE PROMPT THE RUN WAS MADE WITH, which is not the prompt in the
           *  box. The share button captioned files from the live field, so
           *  editing it after a capture shipped a `.mri` labelled with a
           *  prompt that was never run. */
          prompt?: string;
          means: string;
        }>(r),
      );

/** The last image run, as a `.mri` somebody opens with nothing installed.
 *
 *  Bytes and a Content-Disposition rather than a plain `<a download>`, for
 *  the same reason `shareVlaFinding` is: the server answers a refusal as JSON
 *  with a 409 — nothing has been run yet — and a link would cheerfully save
 *  that sentence to disk as a `.mri` the recipient then cannot open.
 *
 *  NOTHING IN THIS BODY BECOMES A CLAIM IN THE FILE. The section is built
 *  from what the server measured; `note` is a caption and is the only field.
 */
export const shareImageRun = (body: { note?: string }) =>
  fetch("/api/image/share", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then(async (r) => {
    if (!r.ok) throw new ApiError(r.status, await r.text());
    return r.blob();
  });
