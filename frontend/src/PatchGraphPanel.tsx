import { CSSProperties, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  errorText,
  patchGraph,
  PatchGraphEdge,
  PatchGraphNode,
  PatchGraphView,
} from "./api";
import ReceiptLine from "./ReceiptLine";
import { useScanOnData } from "./useScanOnData";
import { useThemeVersion } from "./theme";

/** A PATCHING graph — what wrote the thing that wrote the answer.
 *
 *  The node grid says WHERE the answer is carried and a path trace says what
 *  wrote into ONE site. This asks that second question again of the senders
 *  that survived their controls, which is the question a circuit view is
 *  actually opened for.
 *
 *  ## It is a patching graph, and the heading says so
 *
 *  circuit-tracer's attribution graphs are built from transcoders, which exist
 *  for a handful of models. `GraphPanel` renders one of THEIRS under a
 *  provenance banner; this renders one of OURS, from nothing but the model
 *  that was loaded. Borrowing the more famous name for a thing this is not
 *  would be the claim the whole project exists not to make, so the word
 *  "attribution" appears here exactly once — in the sentence denying it.
 *
 *  ## No graph library
 *
 *  The geometry is `ArcCanvas`'s, applied in two dimensions: quadratic Bézier
 *  curves between element centres, on a canvas sized in device pixels and
 *  drawn in CSS pixels, repainted when the theme version moves because painted
 *  pixels do not re-cascade. Nodes are laid out on a grid — layer up the page,
 *  position across it — so the picture is the model's own geometry rather than
 *  whatever a force simulation settled on. Two runs of the same prompt draw
 *  the same picture, which a spring layout cannot promise.
 *
 *  ## Every edge here was controlled, and one that LOST is still drawn
 *
 *  Two rules that are easy to confuse. The server prunes any sender it never
 *  ran controls against, because an edge with a score and no verdict has
 *  nothing behind it to click. But an edge that WAS controlled and lost is
 *  kept and drawn dashed: "we tested this and it did not survive" and "we
 *  never saw this" are different findings, and only one of them is in this
 *  picture.
 */

/** Layout, in CSS pixels. Chips are ~86px wide, so the column pitch has to
 *  clear that or neighbours collide on a wide prompt. */
const COL_W = 116;
const ROW_H = 62;
const PAD_X = 74;
const PAD_Y = 34;

/** Sideways offset between nodes that share a (layer, position) cell. Several
 *  heads of one layer writing into one position is the COMMON case, not an
 *  edge case, so they are fanned rather than stacked. */
const SUB_W = 30;

/** How close the pointer has to come to a curve to pick it up. Generous,
 *  because a 1px stroke is not a click target anybody can hit. */
const HIT_PX = 9;

const CLEAN_DEFAULT = "The Eiffel Tower is located in the city of";
const CORRUPT_DEFAULT = "The Colosseum is located in the city of";

/** Point on a quadratic Bézier at `t`. The same curve `ArcCanvas` strokes;
 *  written out here because hit-testing needs to sample it. */
function at(t: number, p0: number, p1: number, p2: number): number {
  const u = 1 - t;
  return u * u * p0 + 2 * u * t * p1 + t * t * p2;
}

export default function PatchGraphPanel({
  epoch,
  recorded,
}: {
  epoch: number;
  /** Set when a `.mri` is open and carries a graph: the prompts it was built
   *  on, so the panel offers to draw the recording instead of a button whose
   *  only outcome is a refusal. */
  recorded?: { clean: string; corrupt: string };
}) {
  const [clean, setClean] = useState(recorded?.clean || CLEAN_DEFAULT);
  const [corrupt, setCorrupt] = useState(recorded?.corrupt || CORRUPT_DEFAULT);
  const [depth, setDepth] = useState(2);
  const [maxReceivers, setMaxReceivers] = useState(2);
  const [g, setG] = useState<PatchGraphView | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  /** The edge whose controls are open. An index, not the object: a re-fetch
   *  replaces every edge and an object held across it is a stale one. */
  const [picked, setPicked] = useState(-1);
  const [hovered, setHovered] = useState(-1);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const scanRef = useScanOnData(g);
  const themeV = useThemeVersion();

  async function run() {
    setBusy(true);
    setErr("");
    setG(null);
    setPicked(-1);
    setHovered(-1);
    try {
      setG(await patchGraph({ clean, corrupt, depth, max_receivers: maxReceivers }));
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  /** Defensive by type, not by presence. This renders a payload that can come
   *  out of a `.mri` a stranger forwarded, and `frontend/src` has no error
   *  boundary: one TypeError here unmounts the whole viewer and the reader
   *  gets a white screen with no explanation. */
  const nodes: PatchGraphNode[] = useMemo(
    () =>
      (Array.isArray(g?.nodes) ? g!.nodes : []).filter(
        (n) =>
          n &&
          typeof n.id === "string" &&
          Number.isFinite(n.layer) &&
          Number.isFinite(n.position),
      ),
    [g],
  );

  const edges: PatchGraphEdge[] = useMemo(
    () =>
      (Array.isArray(g?.edges) ? g!.edges : []).filter(
        (e) =>
          e &&
          typeof e.source === "string" &&
          typeof e.target === "string" &&
          typeof e.recovery === "number" &&
          Number.isFinite(e.recovery),
      ),
    [g],
  );

  /** Layer up the page and position across it, both on the values present
   *  rather than on their ranges: a graph over layers 9, 10 and 11 of a 12
   *  layer model should be three rows, not twelve with nine empty. */
  const layout = useMemo(() => {
    const layers = [...new Set(nodes.map((n) => n.layer))].sort((a, b) => a - b);
    const positions = [...new Set(nodes.map((n) => n.position))].sort((a, b) => a - b);

    // A cell is (layer, position) and a node is (layer, HEAD, position), so
    // several heads of one layer routinely land in the same cell — 4 of them
    // in one cell on a float32 run, which without this drew four chips at
    // identical coordinates: one visible, three hidden underneath, and four
    // edges converging on what looked like a single node.
    const cell = new Map<string, string[]>();
    for (const n of nodes) {
      const key = `${n.layer}:${n.position}`;
      cell.set(key, [...(cell.get(key) ?? []), n.id]);
    }
    const widest = Math.max(1, ...[...cell.values()].map((ids) => ids.length));

    const place = new Map<string, { x: number; y: number }>();
    for (const n of nodes) {
      const peers = cell.get(`${n.layer}:${n.position}`)!;
      const k = peers.indexOf(n.id);
      place.set(n.id, {
        // Fanned around the column's centre line, so the cell still reads as
        // one position and its occupants are individually clickable.
        x:
          PAD_X +
          positions.indexOf(n.position) * COL_W +
          (k - (peers.length - 1) / 2) * SUB_W,
        // Deepest layer at the TOP, so the picture reads the way the model
        // runs down the page — the same order the patching grid uses.
        y: PAD_Y + (layers.length - 1 - layers.indexOf(n.layer)) * ROW_H,
      });
    }
    return {
      place,
      positions,
      // The fan can push the outermost chip past the last column, so the
      // stage has to account for it or the rightmost node is clipped by the
      // scroller rather than reachable inside it.
      width:
        PAD_X * 2 +
        Math.max(0, positions.length - 1) * COL_W +
        (widest - 1) * SUB_W,
      height: PAD_Y * 2 + Math.max(0, layers.length - 1) * ROW_H,
    };
  }, [nodes]);

  /** The curve for one edge, as the three control values the canvas and the
   *  hit test both need. One function so they cannot disagree about where a
   *  line is — a click that selects a different edge from the one under the
   *  cursor is worse than no click at all. */
  const curveOf = useCallback(
    (e: PatchGraphEdge) => {
      const a = layout.place.get(e.source);
      const b = layout.place.get(e.target);
      if (!a || !b) return null;
      // Bowed sideways by the vertical gap, so two edges between the same pair
      // of columns do not lie on top of each other.
      const bow = Math.min(70, 16 + Math.abs(a.y - b.y) * 0.35);
      const mx = (a.x + b.x) / 2 + (a.x === b.x ? bow : 0);
      const my = (a.y + b.y) / 2;
      return { a, b, mx, my };
    },
    [layout],
  );

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, layout.width, layout.height);

    const css = getComputedStyle(document.documentElement);
    const pick = (name: string, fallback: string) =>
      css.getPropertyValue(name).trim() || fallback;
    // The same two hues the patching grid uses for its cells, so an edge here
    // and a cell there can be read together without a second colour key.
    const up = pick("--color-cobalt", "#1a5fd0");
    const down = pick("--crimson-500", "#c1121f");

    const peak = edges.reduce((m, e) => Math.max(m, Math.abs(e.recovery)), 1e-6);

    edges.forEach((e, i) => {
      const c = curveOf(e);
      if (!c) return;
      const on = i === picked || i === hovered;
      const scale = Math.abs(e.recovery) / peak;
      ctx.beginPath();
      ctx.moveTo(c.a.x, c.a.y);
      ctx.quadraticCurveTo(c.mx, c.my, c.b.x, c.b.y);
      ctx.lineWidth = (1.2 + 5 * scale) * (on ? 1.9 : 1);
      ctx.strokeStyle = e.recovery >= 0 ? up : down;
      // An edge its controls beat is DASHED rather than dropped. The reader
      // can see it was tested; they cannot mistake it for one that passed.
      ctx.setLineDash(e.clears_control === false ? [4, 4] : []);
      ctx.globalAlpha = on ? 1 : e.clears_control === false ? 0.42 : Math.min(1, 0.3 + scale * 0.7);
      ctx.stroke();
    });
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }, [edges, curveOf, layout, picked, hovered]);

  // Size in DEVICE pixels and draw in CSS pixels, exactly as ArcCanvas does:
  // at 1:1 the thin curves — which are the whole point — get upscaled by the
  // compositor and come out blurry on every retina-class screen.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !layout.width) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    canvas.width = Math.round(layout.width * dpr);
    canvas.height = Math.round(layout.height * dpr);
    canvas.style.width = `${layout.width}px`;
    canvas.style.height = `${layout.height}px`;
    canvas.getContext("2d")?.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }, [layout, draw]);

  // Painted pixels do not re-cascade when the theme flips.
  useEffect(() => {
    draw();
  }, [draw, themeV]);

  /** Which edge is under the pointer, by sampling each curve. Nearest wins,
   *  not first: overlapping curves would otherwise always resolve to whichever
   *  happened to be earlier in the list. */
  const edgeAt = useCallback(
    (x: number, y: number): number => {
      let best = -1;
      let bestD = HIT_PX;
      edges.forEach((e, i) => {
        const c = curveOf(e);
        if (!c) return;
        for (let s = 0; s <= 20; s++) {
          const t = s / 20;
          const dx = at(t, c.a.x, c.mx, c.b.x) - x;
          const dy = at(t, c.a.y, c.my, c.b.y) - y;
          const d = Math.hypot(dx, dy);
          if (d < bestD) {
            bestD = d;
            best = i;
          }
        }
      });
      return best;
    },
    [edges, curveOf],
  );

  const open = picked >= 0 ? edges[picked] : undefined;
  const num = (v: unknown): number | null =>
    typeof v === "number" && Number.isFinite(v) ? v : null;

  return (
    <div className="panel pgraph" key={epoch} ref={scanRef}>
      <div className="sect">
        <span className="dot d-patch" />
        <h2 className="h-patch">PATCHING GRAPH — WHAT WROTE THE THING THAT WROTE IT</h2>
        <span className="rule" />
      </div>
      <p className="meta">
        The grid above says where the answer is carried; clicking one cell says
        what wrote into that site. This asks that second question again of the
        senders that survived, and draws the result. It is a{" "}
        <b>patching graph</b>, built from nothing but the model already loaded
        — not a transcoder attribution graph, which needs transcoders that
        exist for a handful of models.
      </p>

      <div className="patch-inputs">
        <label>
          <span className="meta">clean — the run that knows</span>
          <input value={clean} onChange={(e) => setClean(e.target.value)} spellCheck={false} />
        </label>
        <label>
          <span className="meta">corrupt — one fact changed</span>
          <input
            value={corrupt}
            onChange={(e) => setCorrupt(e.target.value)}
            spellCheck={false}
          />
        </label>
      </div>

      <div className="row pgraph-dials">
        <label className="meta">
          levels back
          <input
            type="number"
            min={1}
            max={3}
            value={depth}
            onChange={(e) => setDepth(Math.max(1, Math.min(3, +e.target.value || 1)))}
          />
        </label>
        <label className="meta">
          receivers per level
          <input
            type="number"
            min={1}
            max={6}
            value={maxReceivers}
            onChange={(e) =>
              setMaxReceivers(Math.max(1, Math.min(6, +e.target.value || 1)))
            }
          />
        </label>
        <button className="cta" onClick={() => void run()} disabled={busy}>
          {busy ? "Walking backwards…" : recorded ? "Show the recorded graph" : "Build the graph"}
        </button>
        <span className="spacer" />
        {/* Every level is one path trace per receiver, and each of those is a
            forward pass per earlier component plus its controls. Saying so
            before the click is the difference between a wait and a surprise.
            The arithmetic is `estimate`'s own — at most `max_receivers` per
            level for `depth` levels — because a panel that quotes a different
            number from the projection is a third answer to one question.
            NO DURATION: this said "a minute or so on a laptop", which was a
            constant nobody measured, printed as guidance. The same dials take
            119s on Qwen3-1.7B and a different time on every other model, and
            `estimate` refuses to quote seconds for exactly that reason. */}
        <span className="meta">
          each level is one path trace per receiver — at most{" "}
          {depth * maxReceivers} traces, several hundred forward passes each
        </span>
      </div>

      {err && <div className="hint err">{err}</div>}

      {g && !edges.length && (
        <p className="meta">
          Nothing survived. {typeof g.seeding === "string" ? g.seeding : ""}
        </p>
      )}

      {g && edges.length > 0 && (
        <>
          <div className="patch-answers meta">
            <span>
              <b>{nodes.length}</b> nodes
            </span>
            <span>
              <b>{edges.length}</b> edges
            </span>
            <span>
              <b>{(num(g.n_scored) ?? 0).toLocaleString()}</b> senders scored
            </span>
            <span className="spacer" />
            <span>
              {num(g.passes) ?? 0} passes · {num(g.seconds) ?? 0}s
            </span>
          </div>

          <div className="pgraph-stage-wrap">
            <div
              className="pgraph-stage"
              style={{ width: layout.width, height: layout.height } as CSSProperties}
            >
              <canvas
                ref={canvasRef}
                className="pgraph-canvas"
                onMouseMove={(ev) => {
                  const r = ev.currentTarget.getBoundingClientRect();
                  setHovered(edgeAt(ev.clientX - r.left, ev.clientY - r.top));
                }}
                onMouseLeave={() => setHovered(-1)}
                onClick={(ev) => {
                  const r = ev.currentTarget.getBoundingClientRect();
                  const i = edgeAt(ev.clientX - r.left, ev.clientY - r.top);
                  if (i >= 0) setPicked(i === picked ? -1 : i);
                }}
              />
              {nodes.map((n) => {
                const at2 = layout.place.get(n.id);
                if (!at2) return null;
                const touching =
                  open && (open.source === n.id || open.target === n.id);
                return (
                  <span
                    key={n.id}
                    className={`pgraph-node ${n.role === "seed" ? "seed" : ""} ${
                      touching ? "on" : ""
                    }`}
                    style={{ left: at2.x, top: at2.y } as CSSProperties}
                    title={`layer ${n.layer}, position ${n.position}`}
                  >
                    {n.head === null || n.head === undefined
                      ? `L${n.layer} MLP`
                      : `L${n.layer}H${n.head}`}
                  </span>
                );
              })}
            </div>
          </div>

          <div className="meta pgraph-key">
            Thickness is recovery, on the same scale as the grid above. Blue
            recovered the clean answer, red pushed it further away.{" "}
            <b>Dashed edges were tested and did not beat their controls</b> —
            they are drawn rather than dropped, because "we tested this and it
            did not survive" and "we never saw this" are different findings.
            Ringed nodes are the sites the node grid flagged, which is where
            the walk started. <b>Click an edge</b> for the draws behind it.
          </div>

          {/* ─── the controls behind one edge ───────────────────────────── */}
          {open && (
            <div className="patch-path pgraph-controls">
              <div className="row">
                <span className="meta">
                  <b>{open.source}</b> wrote into <b>{open.target}</b>
                </span>
                <span className="spacer" />
                <button className="ghost sm" onClick={() => setPicked(-1)}>
                  close
                </button>
              </div>

              <p className="meta">
                Recovery <b>{open.recovery.toFixed(4)}</b> against{" "}
                {num(open.control_max) === null ? (
                  "no control"
                ) : (
                  <>
                    a strongest control of <b>{open.control_max!.toFixed(4)}</b>
                  </>
                )}{" "}
                over {open.control_draws} same-norm random draws —{" "}
                {open.clears_control
                  ? "it beats them"
                  : "it does NOT beat them, which is why it is dashed"}
                .{" "}
                {open.clears_position === null
                  ? "The shifted-position control was not run for this edge, which is not the same as its having failed one."
                  : open.clears_position
                    ? "It also beats the same edit taken from a different position, so it is this sender at this position rather than a loud layer."
                    : "It does NOT beat the same edit taken from a different position, so this may be the layer rather than this site."}
              </p>

              {Array.isArray(open.controls) && open.controls.length > 0 ? (
                <>
                  <div className="meta cand-head">
                    the {open.controls.length} draws, each a random patch of the
                    same norm at the same site
                  </div>
                  <ol className="path-list stagger pgraph-draws">
                    {open.controls.map((c, i) => (
                      <li key={i} style={{ "--i": i } as CSSProperties}>
                        <span className="mid path-name">draw {i + 1}</span>
                        <span className="path-track">
                          <span
                            className="path-bar"
                            style={{
                              width: `${Math.min(100, (Math.abs(c) / Math.max(Math.abs(open.recovery), 1e-6)) * 100)}%`,
                              background:
                                c >= 0 ? "var(--color-mute)" : "var(--crimson-500)",
                            }}
                          />
                        </span>
                        <span className="mid path-val">{c.toFixed(4)}</span>
                      </li>
                    ))}
                    <li className="pgraph-actual" style={{ "--i": open.controls.length } as CSSProperties}>
                      <span className="mid path-name">this edge</span>
                      <span className="path-track">
                        <span
                          className="path-bar"
                          style={{
                            width: "100%",
                            background:
                              open.recovery >= 0
                                ? "var(--color-cobalt)"
                                : "var(--crimson-500)",
                          }}
                        />
                      </span>
                      <span className="mid path-val">{open.recovery.toFixed(4)}</span>
                    </li>
                  </ol>
                </>
              ) : (
                <p className="meta">
                  This file carries the verdict and the strongest draw, but not
                  the individual ones — it was written before they travelled.
                  The verdict rests on the strongest, which is here.
                </p>
              )}
            </div>
          )}

          {/* The seeding rule is not a footnote. Edge count is quadratic in
              sites, so every graph here is a subset by construction, and one
              whose rule for choosing edges is unstated is a picture. */}
          <p className="meta">{typeof g.seeding === "string" ? g.seeding : ""}</p>
          <p className="meta">{typeof g.means === "string" ? g.means : ""}</p>
          <ReceiptLine receipt={g.receipt} />
        </>
      )}
    </div>
  );
}
