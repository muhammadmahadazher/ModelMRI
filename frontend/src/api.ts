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
}

export interface AttentionData {
  layer: number;
  head: number;
  tokens: string[];
  matrix: number[][];
}

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json() as Promise<T>;
}

export const getSession = () =>
  fetch("/api/session").then((r) => json<SessionInfo>(r));

export const loadModel = (hf_id?: string) =>
  fetch("/api/model/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(hf_id ? { hf_id } : {}),
  }).then((r) => json<ModelStatus>(r));

export const getAttentionMeta = () =>
  fetch("/api/attention/meta").then((r) => json<AttentionMeta>(r));

export const getAttention = (layer: number, head: number) =>
  fetch(`/api/attention?layer=${layer}&head=${head}`).then((r) =>
    json<AttentionData>(r),
  );

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
) =>
  fetch("/api/model/prompt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, max_new_tokens, temperature }),
  }).then((r) => json<{ generation: string }>(r));

export type StreamHandlers = {
  onToken: (text: string) => void;
  onDone: () => void;
  onError: (message: string) => void;
};

export function streamGenerate(prompt: string, h: StreamHandlers): () => void {
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

  ws.onopen = () => ws.send(JSON.stringify({ prompt }));
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
