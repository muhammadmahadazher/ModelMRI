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
 *  prediction — measured on gpt2 layer 0, the twelve per-head scores sum to
 *  1.995 while ablating the whole layer gives 0.208.
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
  /** Resample only — the spread across draws. Head 10 on gpt2 layer 0 ranged
   *  0.027 to 0.335 around a median of 0.036, so one draw could have reported
   *  any of those as the head's score. */
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
  elapsed_s: number;
  ranked: HeadScore[];
  means: string;
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
  fetch(`/api/attention/baselines?layer=${layer}`).then((r) =>
    json<BaselineComparison>(r),
  );

/** How far masking one token moves the answer at one position.
 *
 *  Same units as `HeadScore.kl` — both come from `kl_nats` in ablate.py, so a
 *  head score and a token score on one screen mean the same thing. What they
 *  are NOT is comparable in behaviour: `modelmri/attribute.py` measured the
 *  singles over-stating one joint mask by 1.82x on gpt2 over the rows this
 *  list shows, and under-stating it by 0.35x on gemma-3-270m-it over the typed
 *  span. The panel prints both live numbers rather than a factor.
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
   *  it put gpt2's own words under a heading reading "chat template scaffold",
   *  on a model whose span_note says in the same breath that it has no chat
   *  template. `unknown` is "the server could not locate your words" — folded
   *  into `typed` it put the chat template under "what you typed". */
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
   *  gpt2, Qwen3-0.6B and gemma-3-270m-it; anything at or below it is
   *  arithmetic rather than the model. */
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
   *  and scored nothing. Measured on gpt2 with a 73-token prompt: 64 of 71
   *  candidates tested, and indices 1..7 rendered with no mark at all. */
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
  const r = await fetch(
    `/api/session/export?layer=${layer}&head=${head}&note=${encodeURIComponent(note)}`,
  );
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

export interface SAEStatus {
  loaded: boolean;
  repo: string | null;
  hook: string | null;
  layer: number | null;
  d_in: number | null;
  d_sae: number | null;
  calibration?: SAECalibration | null;
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

/** A KL small enough that fixed decimals would print it as zero.
 *
 *  Not cosmetic. Feature scores span from 0.4174529 down to -3e-08 on the one
 *  prompt this was measured on, and five decimal places render most of that
 *  list as 0.00000 — a measured value displayed as nothing. Whether a number
 *  that small MEANS anything is what `below_resolution` is for.
 */
export const fmtKL = (kl: number) =>
  kl !== 0 && Math.abs(kl) < 0.001 ? kl.toExponential(2) : kl.toFixed(5);

// ------------------------------------------------------- feature ablation

/** One feature's causal effect, exactly as `modelmri/feature_ablate.py`
 *  reports it. Nothing here is recomputed on this side.
 */
export interface FeatureScore {
  feature_id: number;
  /** Peak activation over the positions that were edited — re-encoded in
   *  float32 by the ablation, NOT read from the float16 cache the bar chart
   *  plots. Measured on gpt2: fp16 rounding moved the top feature's KL by
   *  0.09%, with a max activation error of 0.0916, so the two numbers can
   *  differ in the first decimal and neither is wrong. */
  activation: number;
  /** Every token index the feature was removed at. One entry at
   *  `scope="position"`. At `scope="prompt"` this is the field that lets the
   *  panel show a feature which fires nowhere near the token being attributed
   *  and still reaches the answer through attention — measured on gpt2, 4 of
   *  the causal top-8 across the prompt fire only at earlier tokens. */
  positions: number[];
  kl: number;
  /** What a RANDOM direction of the same norm, subtracted at the same tokens,
   *  cost. It is not zero and it is not small: at the top feature's norm of
   *  35.5 on gpt2, five draws spanned 0.0666-0.1093 nats against that
   *  feature's own 0.4175. A row below its own control has a score that is the
   *  size of its edit rather than the identity of its feature. */
  control_kl: number;
  /** `kl > control_kl`. Measured on gpt2 at the attributed token, 34 of 43
   *  rows clear it — and two that do not, #22852 and #1288, sit 5th and 6th in
   *  the bar chart above. */
  clears_control: boolean;
  /** Share of this feature's original activation the SAE's ENCODER still
   *  reports after the feature's own contribution was subtracted, at the worst
   *  of the edited positions. Not a failure of the edit — the stream moves by
   *  exactly one rank-1 term, which `removal_verified` checks — but a property
   *  of this SAE: W_enc[:,f] and W_dec[f] are not dual, so the encoder reads
   *  other features' contributions through f's direction. Measured on gpt2
   *  blocks.8.hook_resid_pre it runs 0% to 60.3%, above 1% on 38 of 43 rows. */
  encoder_residual: number;
  p_top_before: number;
  p_top_after: number;
  flips_top: boolean;
  /** The server's verdict against `resolution_kl`, and never recomputed here
   *  against `noise_floor_kl`. The floor is exactly 0.0 on this path and two
   *  measured scores came back NEGATIVE (-1e-08, -3e-08) — float32 summation
   *  over 50257 vocabulary entries — so a client greying out "at or below the
   *  floor" would grey out nothing. Measured: 2 of 43 scores are at or below
   *  the floor, 8 of 43 are below the resolution. */
  below_resolution: boolean;
}

/** One run of the feature ranking, exactly as the server reports it.
 *
 *  Every caveat rendered by the panel is either a field here or is computed
 *  from two fields here. Nothing is remembered: the additivity direction in
 *  particular is READ OFF THIS RUN, because features miss in the opposite
 *  direction from heads — the head panel's singles over-count 8x on gpt2 layer
 *  0 while these under-count 3.2x — and a remembered direction would be
 *  exactly backwards.
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
   *  token (0.2036 on gpt2); at prompt scope it is the worst of eleven tokens
   *  (0.4253, at token 3). Null when the stream has no norm there. */
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
  cameras?: string[];
  video_key?: string;
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

export interface TraceDoc {
  id: string;
  name: string;
  started_at: string;
  steps: TraceStep[];
}

export const getTraces = () => fetch("/api/traces").then((r) => json<TraceSummary[]>(r));
export const getTrace = (id: string) =>
  fetch(`/api/traces/${id}`).then((r) => json<TraceDoc>(r));

/** What the last generation cost, including what watching it cost.
 *
 *  Every field may be null and none is ever faked — CPU has no allocator to
 *  ask, and a 0 in a memory column is a claim that nothing was used. */
export interface TelemetryReport {
  available: boolean;
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
  fetch("/api/quantdiff/behaviour", {
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
  fetch("/api/graph").then((r) => json<GraphView>(r));

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
  fetch("/api/lens/tune", {
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
export const scanFolder = (path: string) =>
  fetch("/api/custom/scan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  }).then((r) =>
    json<{
      added: string;
      adapters: CustomCandidate[];
      torchscript: CustomCandidate[];
      roots: string[];
    }>(r),
  );
