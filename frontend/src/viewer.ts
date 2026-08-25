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
  /** An image run: a denoising strip, the cross-attention over the prompt
   *  that produced it, or a detector's readout. Optional and additive like
   *  `graph`.
   *
   *  A6, and the last unbuilt item in Theme A. Every other result this tool
   *  produces could be sent to somebody; the one that is a PICTURE could not,
   *  so an image finding was the only kind that had to be screenshot to be
   *  shared -- and a screenshot carries no provenance, no seed and no
   *  statement of what was shrunk. */
  image?: {
    provenance?: Record<string, unknown>;
    prompt?: string;
    /** `null` is NO SEED WAS FIXED, which is not seed 0. */
    seed?: number | null;
    scheduler?: string;
    frames?: Record<string, unknown>[];
    steps_requested?: number;
    steps_run?: number;
    decoded_steps?: number[];
    skipped_steps?: number[];
    steps_never_reached?: number[];
    attention?: Record<string, unknown>;
    readout?: Record<string, unknown>;
    means?: string;
  };
  /** A robot finding: one frame of one episode, the policy's attention over
   *  it, and the occlusion map that says which patch actually MOVED the
   *  action. Optional and additive like `graph`.
   *
   *  `/api/vla/share` has written this since the feature landed and nothing
   *  ever served it back, so a shared robot finding opened as an empty text
   *  session. */
  vla?: {
    provenance?: Record<string, unknown>;
    frame?: string;
    frame_size?: number[];
    frame_downsampled?: boolean;
    frame_note?: string;
    attention?: number[][][];
    occlusion?: Record<string, unknown>;
  };
  /** The agent run this analysis belongs to, and which step failed. Optional
   *  and additive like `graph`.
   *
   *  This is the half no hosted platform can ship: every competitor's share
   *  artefact is a link into their own trace UI, which dies when the account
   *  lapses. Here the recipient opens the file with nothing installed, clicks
   *  the failing tool call, and lands in the attention view of the generation
   *  that produced the bad argument. */
  trace?: {
    id?: string;
    name?: string;
    started_at?: string;
    steps?: Record<string, unknown>[];
    n_steps_total?: number;
    truncated?: number;
    step_ref?: string;
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
    // THE CARRIED RUN, in the shape `runtime.session_info` publishes. This is
    // how the agents panel learns a run exists, and it is the path that opens
    // on `step_ref` — the step the bundle was built AROUND, which is the
    // reason it was sent. Serving the run through `/api/traces` instead made
    // it an ordinary store row: it rendered, and it opened on step one.
    trace: {
      available: Boolean(open.trace?.steps?.length),
      id: open.trace?.id ?? "",
      name: open.trace?.name ?? "",
      n_steps: open.trace?.steps?.length ?? 0,
      n_steps_total: open.trace?.n_steps_total ?? open.trace?.steps?.length ?? 0,
      truncated: open.trace?.truncated ?? 0,
      step_ref: open.trace?.step_ref || null,
    },
  };
}

/** Endpoints that need a machine. Refused with the reason, not a 404. */
const NEEDS_A_MACHINE =
  "This is the viewer — it reads a shared analysis and nothing else. " +
  "Install ModelMRI (`pip install modelmri`) to point these instruments at " +
  "your own models.";

/** The step shape the panels read, from the smaller one a `.mri` carries.
 *
 *  MIRRORS `server.py`'s `/api/session/trace`, which does exactly this for the
 *  app. A bundle stores a deliberately reduced step — no `seq`, no cache or
 *  reasoning token counts, no `adoptable` — and the panels treat a MISSING key
 *  and a null one very differently:
 *
 *    step.tokens_cache_read !== null   is TRUE for undefined, and the ledger
 *                                      printed "undefined cache read"
 *    step {sel.seq}                    printed "step undefined"
 *
 *  `null` is the value that already means "the recorder said nothing", and
 *  `seq` is positional so it is derived rather than left blank. Fixed in the
 *  app first; the viewer served the raw steps and kept the bug, which is what
 *  a fix on one side of a wire looks like.
 */
function viewerSteps(steps: Record<string, unknown>[]): Record<string, unknown>[] {
  return steps.map((step, i) => ({
    truncated_in: 0,
    truncated_out: 0,
    tokens_cache_read: null,
    tokens_cache_write: null,
    tokens_reasoning: null,
    // Never adoptable, and false rather than absent: a `.mri` carries a run's
    // shape and not the token ids underneath it, so there is nothing here for
    // the mechanistic panels to reopen.
    adoptable: false,
    ...step,
    seq: i,
  }));
}

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
    // A non-empty STRING. `measured_by: true` passes a truthiness test and
    // React renders a boolean as nothing, so the graph would appear under
    // ModelMRI's chrome with a blank disclaimer.
    const claim = g.provenance?.measured_by;
    if (typeof claim !== "string" || !claim.trim()) {
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

  if (p === "/api/image/replay") {
    if (!open) return { status: 409, payload: { error: "No session open." } };
    const img = open.image;
    // `available: false` is a STATE and not an error: most sessions carry no
    // image run, and a 409 for the ordinary case would render as "this
    // measurement is broken".
    if (!img || !img.provenance) return ok({ available: false });
    // Refused here as well as in `session._image`, because THIS copy runs in
    // the recipient's browser on a file a stranger forwarded, and the claim
    // it guards is the one that makes the picture worth anything: which
    // checkpoint drew it. A strip rendered under ModelMRI's chrome with no
    // model behind it is the confusion the section exists to prevent.
    const prov = img.provenance as Record<string, unknown>;
    // `repo`, `family` and `kind` — the three `session._image` requires to be
    // non-empty. `architecture` and `revision` are checked for PRESENCE only,
    // because "" is a claim there ("the checkpoint published none") and the
    // reader accepts it; absence is silence and it does not.
    const named = ["repo", "family", "kind"].every(
      (k) => typeof prov[k] === "string" && (prov[k] as string).trim() !== "",
    );
    const stated = ["architecture", "revision"].every(
      (k) => typeof prov[k] === "string",
    );
    if (!named || !stated) {
      return {
        status: 422,
        payload: {
          error:
            "this session carries an image run that does not say which " +
            "checkpoint drew it, so it is not rendered. A picture without " +
            "its model is not a measurement anybody can repeat.",
        },
      };
    }
    return ok({ available: true, ...img });
  }

  if (p === "/api/vla/replay") {
    if (!open) return { status: 409, payload: { error: "No session open." } };
    const vla = open.vla;
    // `available: false` is a STATE and not an error, exactly as for an image
    // run: most sessions carry no robot finding.
    if (!vla || !vla.provenance) return ok({ available: false });
    // Refused here as well as in `session._vla`, because THIS copy runs in the
    // recipient's browser on a file a stranger forwarded — and the claim it
    // guards is the one that makes the heat map mean anything. A map without
    // its policy, dataset, episode, timestep and camera is, in that
    // validator's own words, a picture of nothing in particular.
    const prov = vla.provenance as Record<string, unknown>;
    const named = ["policy", "dataset", "camera", "revision"].every(
      (k) => typeof prov[k] === "string" && (prov[k] as string).trim() !== "",
    );
    const placed = ["episode", "timestep"].every(
      (k) => typeof prov[k] === "number" && Number.isInteger(prov[k]),
    );
    if (!named || !placed) {
      return {
        status: 422,
        payload: {
          error:
            "this session carries a robot finding that does not say which " +
            "policy, dataset, episode, timestep and camera produced it, so " +
            "it is not rendered. A heat map without those is a picture of " +
            "nothing in particular.",
        },
      };
    }
    return ok({ available: true, ...vla });
  }

  if (p === "/api/attention/meta") {
    // `reason` on both branches, mirroring `Session.attention_meta`. The panel
    // prints it and renders NOTHING without one, so a bundle exported around
    // an agent run — which carries no attention slices — made the whole
    // attention section vanish with nothing saying why. That was fixed in the
    // Python half and this shim, which serves the same route to the same
    // panel, kept the old shape: the fix reached the app and not the page it
    // was written for.
    if (!open) {
      return ok({
        available: false,
        reason:
          "no file is open yet. Drop a `.mri` on this page and its attention " +
          "maps, if it carries any, appear here.",
      });
    }
    const slices = Object.keys(open.attention ?? {}).length;
    return ok({
      available: slices > 0,
      n_prompt: open.n_prompt ?? 0,
      n_layers: open.n_layers ?? 0,
      n_heads: open.n_heads ?? 0,
      n_tokens: (open.tokens ?? []).length,
      replay: true,
      ...(slices > 0
        ? {}
        : {
            reason:
              "this session carries no attention maps. A `.mri` stores the " +
              "slices that were captured, and this one was exported for what " +
              "it does carry rather than for a layer and head.",
          }),
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
  // THE CARRIED RUN, through the same route the app serves it on. The agents
  // panel asks for this before the store list, and a bundle IS a carried run —
  // it is the one source this page has. `available: false` when the file holds
  // none, which is the ordinary answer and not an error.
  if (p === "/api/session/trace") {
    const t = open?.trace;
    if (!t?.steps?.length) return ok({ available: false });
    return ok({
      available: true,
      id: t.id || "bundled",
      name: t.name || "the bundled run",
      started_at: t.started_at || "",
      steps: viewerSteps(t.steps),
      step_ref: t.step_ref || "",
      truncated: t.truncated || 0,
      n_steps_total: t.n_steps_total ?? t.steps.length,
      // Rolled up server-side in the app, by `ledger.roll_up`. This page has
      // no Python, and a second implementation of the token arithmetic is
      // exactly the drift `viewer_check.py` exists to prevent — so the keys
      // are ABSENT rather than computed here, and `TokenLedger` renders
      // nothing for an absent rollup rather than a table of zeroes.
    });
  }
  // A store this page does not have. Every one of these writes to, searches,
  // or re-runs something the viewer has no machine for, and each refusal says
  // which — a 404 would read as "the viewer is broken" rather than "this is
  // not a thing a file can do".
  if (p === "/api/traces/search") {
    return {
      status: 409,
      payload: {
        error:
          "search runs across the trace store on a machine with ModelMRI " +
          "installed. This page holds one run — the one inside the file you " +
          "opened — so there is nothing to search across.",
      },
    };
  }
  if (p.startsWith("/api/traces/import")) {
    return {
      status: 409,
      payload: {
        error:
          "importing a run writes it to a store, and this page writes " +
          "nothing. Install ModelMRI to import an Inspect log or a trace of " +
          "your own.",
      },
    };
  }
  if (p.endsWith("/bundle/preview")) {
    return {
      status: 409,
      payload: {
        error:
          "you are already reading the bundle this would produce. Exporting " +
          "one is done from the tool that recorded the run.",
      },
    };
  }
  if (p.endsWith("/adopt")) {
    return {
      status: 409,
      payload: {
        error:
          "a `.mri` carries a run's shape and not the token ids underneath " +
          "it, so there is nothing here to reopen in the mechanistic panels " +
          "— and this page holds no weights to reopen it with.",
      },
    };
  }
  if (p === "/api/rubric/score") {
    return {
      status: 409,
      payload: {
        error:
          "a rubric scores every run in the store against rules you wrote. " +
          "This page holds one run and no store, so there is nothing to rank " +
          "it against.",
      },
    };
  }
  // A LIST, not `{traces: []}`. The real route returns a bare array and
  // `api.ts` types it as one, so the object this used to return made the
  // agents panel call `.map` on a non-array in the standalone viewer.
  // ALWAYS EMPTY. This is the trace STORE — what a machine recorded or
  // imported — and this page has no store. The run inside the file is served
  // as the CARRIED run above, which is what it is: not in a store, not
  // deletable, and gone when the file closes. Listing it here promised all
  // three, and cost the `step_ref` the bundle was built around.
  if (p === "/api/traces") {
    return ok([]);
  }
  if (p.startsWith("/api/traces/")) {
    const t = open?.trace;
    if (!t?.steps?.length) {
      // 404, matching the app's own route for an id it does not hold. A
      // refusal here would read as "the viewer cannot do this", when the
      // truth is that this particular file carries no run.
      return {
        status: 404,
        payload: { error: "this file carries no agent run" },
      };
    }
    // Patterns and token rollups are computed server-side in the app. The
    // viewer has no Python, so it serves the steps and omits those keys
    // rather than shipping a second implementation that could disagree with
    // the first — the exact drift `viewer_check.py` exists to prevent.
    return ok({
      id: t.id || "bundled",
      name: t.name || "the bundled run",
      started_at: t.started_at || "",
      steps: viewerSteps(t.steps),
      step_ref: t.step_ref || "",
      truncated: t.truncated || 0,
      n_steps_total: t.n_steps_total ?? t.steps.length,
    });
  }
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
