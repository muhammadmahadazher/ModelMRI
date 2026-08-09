/** Viewer build: open a `.mri` with nothing installed at all.
 *
 *  `modelmri open` works, but it imports torch and transformers first —
 *  measured at 26 seconds on a warm machine — to display a 54 KB recording
 *  that needs neither. Someone sent an analysis is not someone who wanted a
 *  Python environment.
 *
 *  So: the same React app, built with VITE_VIEWER=1, with the API answered
 *  from a file the user dropped. Patching `fetch` (see main.tsx) is what the
 *  demo build already does, and it has the same virtue here — every call
 *  site stays identical to the real app, so the viewer cannot drift from the
 *  product it is showing.
 *
 *  Nothing is uploaded. There is no server to upload to; the file is read in
 *  the page and never leaves it.
 *
 *  The parsing below is a deliberate mirror of `modelmri/session.py`. Both
 *  sides are tested against the same fixtures (tests/test_viewer_parity.py)
 *  because a viewer that renders a *slightly* different matrix than the tool
 *  is worse than no viewer.
 */

export const VIEWER = import.meta.env.VITE_VIEWER === "1";

/** A `.mri` named in the URL, for `modelmri open` to point us at.
 *
 *  Only a same-origin relative path is honoured. A viewer that fetched any
 *  URL a link handed it would be a way to make someone's browser retrieve
 *  arbitrary addresses by sending them a link, which is not a thing a file
 *  reader should do.
 */
export function autoOpenPath(): string | null {
  if (!VIEWER) return null;
  const raw = new URLSearchParams(location.search).get("f");
  if (!raw) return null;
  if (/^[a-z][a-z0-9+.-]*:/i.test(raw) || raw.startsWith("//")) return null;
  return raw;
}

const FORMAT = "modelmri-session";
const FORMAT_VERSION = 1;

interface Doc {
  format?: string;
  format_version?: number;
  created_at?: string;
  modelmri?: string;
  meta?: Record<string, unknown>;
  prompt?: string;
  generation?: string;
  tokens?: string[];
  n_layers?: number;
  n_heads?: number;
  attention?: Record<string, { q: string; scale: number }>;
  lens?: unknown[];
}

let open: Doc | null = null;

export class ViewerError extends Error {}

/** gunzip via the platform. No library, and no bundle weight. */
async function gunzip(data: ArrayBuffer): Promise<string> {
  const bytes = new Uint8Array(data);
  const gzipped = bytes[0] === 0x1f && bytes[1] === 0x8b;
  if (!gzipped) return new TextDecoder().decode(bytes);
  if (typeof DecompressionStream === "undefined") {
    throw new ViewerError(
      "this browser cannot decompress gzip. Chrome 80+, Firefox 113+ or " +
        "Safari 16.4+ can, or run `modelmri open <file>` locally.",
    );
  }
  const stream = new Blob([bytes]).stream().pipeThrough(
    new DecompressionStream("gzip"),
  );
  return new Response(stream).text();
}

export async function parse(data: ArrayBuffer): Promise<Doc> {
  if (!data.byteLength) throw new ViewerError("the file is empty");
  let text: string;
  try {
    text = await gunzip(data);
  } catch (err) {
    if (err instanceof ViewerError) throw err;
    throw new ViewerError("could not decompress the file — it may be damaged");
  }
  let doc: Doc;
  try {
    doc = JSON.parse(text);
  } catch {
    throw new ViewerError(
      "this file is not a ModelMRI session — a .mri is written by " +
        "'Share this view' in the attention panel",
    );
  }
  if (!doc || doc.format !== FORMAT) {
    throw new ViewerError(
      "this is not a ModelMRI session file (no 'modelmri-session' marker)",
    );
  }
  if (typeof doc.format_version !== "number" || doc.format_version > FORMAT_VERSION) {
    throw new ViewerError(
      `this session is format version ${doc.format_version}, and this viewer ` +
        `reads up to ${FORMAT_VERSION}. Open it with a newer ModelMRI.`,
    );
  }
  if (!Array.isArray(doc.tokens) || !doc.tokens.every((t) => typeof t === "string")) {
    throw new ViewerError("the session's token list is missing or malformed");
  }
  return doc;
}

/** base64 uint8 -> [S,S] floats. The mirror of session._dequantise. */
function dequantise(blob: string, scale: number, n: number): number[][] {
  const binary = atob(blob);
  if (binary.length !== n * n) {
    throw new ViewerError(
      `attention block is ${binary.length} bytes but the token count says ` +
        `${n}x${n}=${n * n} — the file is truncated or not a session`,
    );
  }
  const rows: number[][] = [];
  for (let r = 0; r < n; r++) {
    const row = new Array<number>(n);
    for (let c = 0; c < n; c++) {
      // Round to 5dp exactly as the Python side does, so a viewer and a
      // local run never disagree in the last digit.
      row[c] = Math.round(binary.charCodeAt(r * n + c) * scale * 1e5) / 1e5;
    }
    rows.push(row);
  }
  return rows;
}

function state() {
  if (!open) return { open: false };
  return {
    open: true,
    meta: {
      ...(open.meta ?? {}),
      created_at: open.created_at ?? null,
      modelmri: open.modelmri ?? null,
    },
    prompt: open.prompt ?? "",
    generation: open.generation ?? "",
    n_tokens: (open.tokens ?? []).length,
    n_slices: Object.keys(open.attention ?? {}).length,
    slices: Object.keys(open.attention ?? {}).sort(),
  };
}

/** Endpoints that need a machine. Refused with the reason, not a 404. */
const NEEDS_A_MACHINE =
  "This is the viewer — it reads a shared analysis and nothing else. " +
  "Install ModelMRI (`pip install modelmri`) to point these instruments at " +
  "your own models.";

export async function viewerFetch(
  path: string,
  /** JSON body, when the caller sent one. Nothing here needs it yet — every
   *  endpoint the viewer answers is a read — but the signature mirrors
   *  demoFetch so the two shims stay comparable. */
  _body?: unknown,
  raw?: BodyInit | null,
): Promise<{ status: number; payload: unknown }> {
  const url = new URL(path, "http://viewer.local");
  const p = url.pathname;
  const ok = (payload: unknown) => ({ status: 200, payload });

  if (p === "/api/session/open") {
    const buffer =
      raw instanceof ArrayBuffer
        ? raw
        : raw instanceof Blob
          ? await raw.arrayBuffer()
          : new ArrayBuffer(0);
    try {
      open = await parse(buffer);
    } catch (err) {
      return {
        status: 422,
        payload: { error: err instanceof Error ? err.message : String(err) },
      };
    }
    return ok(state());
  }
  if (p === "/api/session/close") {
    open = null;
    return ok(state());
  }
  if (p === "/api/session/state") return ok(state());

  if (p === "/api/session") {
    return ok({
      app: "modelmri",
      version: `${open?.modelmri ?? ""} viewer`.trim(),
      model: { loaded: false, hf_id: null, device: null, dtype: null, n_params: null },
    });
  }

  if (p === "/api/attention/meta") {
    if (!open) return ok({ available: false });
    return ok({
      available: Object.keys(open.attention ?? {}).length > 0,
      n_layers: open.n_layers ?? 0,
      n_heads: open.n_heads ?? 0,
      n_tokens: (open.tokens ?? []).length,
      replay: true,
    });
  }

  if (p === "/api/attention") {
    if (!open) return { status: 409, payload: { error: "No session open." } };
    const layer = Number(url.searchParams.get("layer") ?? 0);
    const head = Number(url.searchParams.get("head") ?? 0);
    const block = (open.attention ?? {})[`${layer}:${head}`];
    if (!block) {
      const have = Object.keys(open.attention ?? {}).sort().slice(0, 6);
      return {
        status: 422,
        payload: {
          error:
            `this session does not contain layer ${layer} head ${head}. It has ` +
            `${Object.keys(open.attention ?? {}).length} slices, e.g. ` +
            `${have.join(", ")}. A session stores what was captured, not every ` +
            `combination.`,
        },
      };
    }
    const tokens = open.tokens ?? [];
    try {
      return ok({
        layer,
        head,
        tokens,
        matrix: dequantise(block.q, block.scale, tokens.length),
        replay: true,
      });
    } catch (err) {
      return {
        status: 422,
        payload: { error: err instanceof Error ? err.message : String(err) },
      };
    }
  }

  // Everything the viewer deliberately does not have. Answered honestly so
  // each panel shows its own resting state rather than an error.
  if (p === "/api/accelerator") {
    return ok({
      kind: "cpu",
      name: "viewer",
      vram_gb: null,
      reason: "nothing runs here — this page only reads a recording",
    });
  }
  if (p === "/api/model/progress") return ok({ active: false, stage: "", detail: "" });
  if (p === "/api/models/local") return ok([]);
  if (p === "/api/models/discovered") {
    return ok({ models: [], roots: [], truncated: false });
  }
  if (p === "/api/ollama") return ok({ up: false, models: [], installed: [], suggested: [] });
  if (p === "/api/hub/auth") return ok({ signed_in: false, user: null, source: null });
  if (p === "/api/hub/models") return ok([]);
  if (p === "/api/sae") return ok({ loaded: false, repo: null, hook: null });
  if (p === "/api/sae/available") return ok({ options: [] });
  if (p === "/api/custom") return ok({ loaded: false, roots: [] });
  if (p === "/api/vla") return ok({ loaded: false });
  if (p === "/api/vla/datasets") return ok({ datasets: [] });
  if (p === "/api/traces") return ok({ traces: [] });
  if (p === "/api/paths") {
    return ok({
      override: null,
      data: "(nothing is written — this page is read-only)",
      config: "(nothing is written)",
      cache: "(nothing is written)",
      hf_home: "(no models are downloaded by the viewer)",
      hf_hub_cache: "(no models are downloaded by the viewer)",
      trace_db: "(none)",
      hub_token: "(none — the viewer never asks for credentials)",
      undelivered_traces: "(none)",
      models_dirs: [],
      cwd: "(your browser)",
      legacy: null,
      platform: "browser",
    });
  }

  return { status: 409, payload: { error: NEEDS_A_MACHINE } };
}
