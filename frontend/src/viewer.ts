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
 *  Only a file the page is already serving. A viewer that fetched whatever
 *  URL a link handed it would be a way to make someone's browser retrieve
 *  arbitrary addresses — including LAN and localhost ones the sender cannot
 *  reach — just by getting them to click a link.
 *
 *  The first version of this tried to spot absolute URLs by pattern:
 *  reject anything matching `scheme:` or starting with `//`. That is exactly
 *  the wrong shape of check, and it was bypassed by a backslash —
 *  `?f=\\evil.com/x` is not caught by either test, and the URL parser then
 *  resolves it as protocol-relative. Do not pattern-match URLs. Resolve
 *  them and compare the resolved origin, which is the only thing that
 *  cannot be spelled around.
 */
export function autoOpenPath(): string | null {
  if (!VIEWER) return null;
  const raw = new URLSearchParams(location.search).get("f");
  if (!raw) return null;
  // Backslashes and control characters have no business in a filename here
  // and every business in a bypass. Refuse them before parsing, so the rule
  // is legible without knowing URL-parser trivia.
  // eslint-disable-next-line no-control-regex
  if (/[\\\u0000-\u001f\u007f\s]/.test(raw)) return null;

  let resolved: URL;
  try {
    resolved = new URL(raw, location.href);
  } catch {
    return null;
  }
  if (resolved.origin !== location.origin) return null;
  // ...and inside the directory the viewer itself was served from, so a
  // same-origin host cannot be walked with `../`.
  const dir = new URL(".", location.href).pathname;
  if (!resolved.pathname.startsWith(dir)) return null;
  return resolved.pathname + resolved.search;
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
  /** Where the prompt ends; 0 when the file predates the field. */
  n_prompt?: number;
  n_layers?: number;
  n_heads?: number;
  attention?: Record<string, { q: string; scale: number }>;
  lens?: unknown[];
  /** An attribution graph THIS TOOL DID NOT COMPUTE, read from a
   *  circuit-tracer file. Optional and additive, like `patch`: a file written
   *  before it simply has no key, which is why the format version does not
   *  move. `provenance` is not optional -- see the shim below. */
  graph?: {
    n_nodes?: number;
    edges?: { source: number; target: number; weight: number }[];
    provenance?: Record<string, unknown>;
    prompt?: string;
    summary?: Record<string, unknown>;
    notes?: string[];
  };
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
  // Split, matching session.py. The single check interpolated the value before
  // establishing it was a number, so a hand-made `.mri` could put arbitrary
  // text into the message — and this copy runs in the RECIPIENT'S browser on a
  // file a stranger forwarded, which is the worse half of the same bug. It
  // also drifted in shape: Python printed a repr where `${}` prints
  // "[object Object]", and viewer_check compares only numbers, so nothing
  // would have caught the two implementations disagreeing here.
  if (typeof doc.format_version !== "number") {
    throw new ViewerError(
      "this session does not say which format version it is, so it is " +
        "damaged or it is not a .mri",
    );
  }
  if (doc.format_version > FORMAT_VERSION) {
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

/** base64 uint8 -> [S,S] floats. The mirror of session._dequantise.
 *
 *  Exported because the demo bundle stores attention the same way, for the
 *  same reason: Qwen3-0.6B's 28 x 16 slices are 3.9 MB as raw JSON. One
 *  decoder for both surfaces means they cannot disagree about what a byte
 *  meant. */
export function dequantise(blob: string, scale: number, n: number): number[][] {
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

  if (p === "/api/graph") {
    if (!open) return { status: 409, payload: { error: "No session open." } };
    const g = open.graph;
    if (!g || !g.n_nodes) return ok({ available: false });
    // Refused here as well as in session.py, because this copy runs in the
    // RECIPIENT'S browser on a file a stranger forwarded — and the claim it
    // guards is the one the whole feature rests on. A graph rendered under
    // ModelMRI's chrome without saying who computed it is the confusion the
    // section exists to prevent, so an absent provenance is an error rather
    // than a missing caption.
    if (!g.provenance || !g.provenance.measured_by) {
      return {
        status: 422,
        payload: {
          error:
            "this session carries an attribution graph with no provenance. A " +
            "graph ModelMRI did not compute must say who did, so it is not " +
            "rendered rather than shown as if it were ours.",
        },
      };
    }
    return ok({
      available: true,
      n_nodes: g.n_nodes,
      edges: g.edges ?? [],
      provenance: g.provenance,
      prompt: g.prompt ?? "",
      summary: g.summary ?? {},
      notes: g.notes ?? [],
    });
  }

  if (p === "/api/attention/meta") {
    if (!open) return ok({ available: false });
    return ok({
      available: Object.keys(open.attention ?? {}).length > 0,
      n_prompt: open.n_prompt ?? 0,
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
        // So a shared analysis rests on a token too, instead of opening
        // on the empty canvas the resting state exists to replace.
        n_prompt: open.n_prompt ?? 0,
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
