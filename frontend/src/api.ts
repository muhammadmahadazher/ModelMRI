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

export type StreamHandlers = {
  onToken: (text: string) => void;
  onDone: () => void;
  onError: (message: string) => void;
};

export function streamGenerate(prompt: string, h: StreamHandlers): () => void {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/generate`);
  ws.onopen = () => ws.send(JSON.stringify({ prompt }));
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data as string);
    if (msg.type === "token") h.onToken(msg.text);
    else if (msg.type === "done") {
      h.onDone();
      ws.close();
    } else if (msg.type === "error") h.onError(msg.message);
  };
  ws.onerror = () => h.onError("websocket error");
  return () => ws.close();
}
