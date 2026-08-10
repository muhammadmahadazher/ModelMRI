import { CSSProperties, useCallback, useEffect, useRef, useState } from "react";
import { useThemeVersion } from "./theme";

interface Props {
  tokens: string[];
  matrix: number[][];
  /** The matrix is a DIFFERENCE, so values run both ways. Arcs are ranked
   *  and scaled by magnitude, and coloured by direction: one hue for "this
   *  run attended more here", another for "less". Without this a diff would
   *  render only its increases and quietly drop half the answer. */
  signed?: boolean;
  /** How many leading tokens are the prompt. Used to pick the token the
   *  panel rests on, and to mark where the model's own output begins. */
  nPrompt?: number;
  /** Told which chip is pinned, INCLUDING the -1 written when the strip is
   *  re-sized for a new generation.
   *
   *  That second call is the one that matters. Token attribution is measured
   *  AT a position, so an owner holding "attribute at index 14" while the
   *  strip underneath it becomes a different sequence is holding a stale
   *  index — and index 14 of the new generation is a different word. Telling
   *  the owner about the reset is what returns it to the default position
   *  instead of stranding it there.
   */
  onPin?: (i: number) => void;
  /** The index a ranking was measured at. That chip is marked, and — while
   *  scores are being shown — every chip AFTER it is marked too: they are
   *  outside the causal cone, so they were never candidates, and a bare chip
   *  beside scored ones reads as "scored zero" rather than "not asked". */
  attrPos?: number;
  /** Per-chip score for the ramp, 0..1, already normalised by the caller
   *  against the largest tested score. This component does not know what a
   *  KL is and must not invent a scale for one.
   *
   *  `null` means NOT TESTED and renders nothing at all. A tested token whose
   *  score is 0 still renders the bar's floor, because absence and "too small
   *  to see" are different answers and only one of them is a measurement. */
  scores?: (number | null)[];
  /** First index the ranking actually tested. Everything from index 1 up to
   *  here was a candidate the run had no budget for, and it is a THIRD state:
   *  not scored, but also not "outside the causal cone" the way `after-attr`
   *  chips are. Left unmarked they were simply blank — measured on gpt2 with
   *  a 73-token prompt, 64 of 71 candidates tested, indices 1..7 rendered
   *  with no mark of any kind while the panel's caveat offered two reasons
   *  for a blank chip and neither was the true one. */
  testedFrom?: number;
}

/** Canvas height in CSS pixels. The backing store is this times the DPR. */
const CANVAS_H = 110;

/** Token chips with hover/pin-driven attention arcs drawn on a canvas below. */
export default function ArcCanvas({
  tokens,
  matrix,
  signed,
  nPrompt = 0,
  onPin,
  attrPos,
  scores,
  testedFrom,
}: Props) {
  const rowRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const attrRef = useRef<HTMLSpanElement>(null);
  const [pinned, setPinned] = useState(-1);
  // Through a ref so `pin` below keeps ONE identity for the life of the
  // component. The sizing effect calls it, and an owner that rebuilds its
  // handler every render would otherwise put that effect in the dependency
  // list of its own re-renders: the canvas would be re-measured and the pin
  // silently dropped every time anything above changed.
  const onPinRef = useRef(onPin);
  onPinRef.current = onPin;
  // Arcs are painted pixels, not styled elements: a theme change re-cascades
  // the CSS but leaves this canvas holding the old palette until something
  // else happens to redraw it. Depending on the version forces the repaint.
  const themeV = useThemeVersion();

  /** What the panel shows when you are not pointing at anything.
   *
   *  It used to show nothing: a ~250px empty rectangle, with the sentence
   *  explaining that you should hover printed UNDERNEATH it. So the first
   *  thing a visitor met was a void, and the instruction for escaping the
   *  void was the last thing they read.
   *
   *  The LAST PROMPT token is the right default, for two reasons. Its row
   *  answers the question the panel exists for — what did the model look at
   *  to decide its first word — and it is near the left of the strip, so the
   *  arcs are on screen. Resting on the final token instead looks identical
   *  to the old empty panel on any real generation: the strip is a scroll
   *  container 24,820px wide here, and the arcs were drawn past the right
   *  edge of a viewport showing the first 800px of it.
   */
  const resting = nPrompt > 0 && nPrompt <= matrix.length ? nPrompt - 1 : -1;

  /** Move the pin and tell the owner. One place, so the reset in the sizing
   *  effect cannot drift from the click and Enter handlers — which is how the
   *  owner would end up describing a position the strip no longer shows. */
  const pin = useCallback((next: number) => {
    setPinned(next);
    onPinRef.current?.(next);
  }, []);

  const draw = useCallback(
    (i: number) => {
      const canvas = canvasRef.current;
      const row = rowRef.current;
      if (!canvas || !row) return;
      const ctx = canvas.getContext("2d")!;
      // The context is transformed to CSS pixels by the sizing effect, so
      // clear in those units rather than the device-pixel backing size.
      ctx.clearRect(0, 0, row.scrollWidth, CANVAS_H);
      // -1 means "nothing hovered or pinned", which is a resting state, not
      // an empty one. Hover still overrides it and leaving returns here, so
      // exploration is unchanged — there is simply never a blank panel.
      if (i < 0) i = resting;
      if (i < 0 || !matrix[i]) return;

      const rowRect = row.getBoundingClientRect();
      const chips = Array.from(row.children) as HTMLElement[];
      const centers = chips.map((el) => {
        const r = el.getBoundingClientRect();
        return r.left - rowRect.left + r.width / 2;
      });

      // In signed mode the interesting values go both ways, so rank and
      // threshold on magnitude. Ranking on the raw value would show only
      // the increases and silently drop every place attention moved AWAY —
      // which is half of what a comparison is for.
      const edges = matrix[i]
        .map((w, j) => ({ w, j }))
        .filter(({ w, j }) => j <= i && (signed ? Math.abs(w) : w) >= 0.02)
        .sort((a, b) => (signed ? Math.abs(b.w) - Math.abs(a.w) : b.w - a.w))
        .slice(0, 12);

      // --model is not a variable this stylesheet defines. getPropertyValue
      // returned "", canvas ignores an unparseable strokeStyle, and the arcs
      // silently drew in the default black instead of the attention colour.
      const css = getComputedStyle(document.documentElement);
      const pick = (name: string, fallback: string) =>
        css.getPropertyValue(name).trim() || fallback;
      // --sem-error is the palette's "divergence" hue and --sem-base its
      // "baseline"; both are already contrast-checked in light and dark.
      const color = pick("--color-attn", "#1a5fd0");
      const up = pick("--sem-error", "#c1121f");
      const down = pick("--sem-base", "#0f766e");
      const peak = edges.length ? Math.abs(edges[0].w) : 1;
      for (const { w, j } of edges) {
        const x1 = centers[i];
        const x2 = centers[j];
        const depth = Math.min(100, 18 + Math.abs(x1 - x2) * 0.12);
        ctx.beginPath();
        ctx.moveTo(x1, 4);
        ctx.quadraticCurveTo((x1 + x2) / 2, depth, x2, 4);
        const mag = signed ? Math.abs(w) : w;
        // Scaled against this diff's own maximum: differences are small in
        // absolute terms (a 0.05 shift is large for attention), so the fixed
        // 0..1 scale used for weights would draw every arc hairline.
        const scale = signed ? mag / Math.max(peak, 1e-6) : mag;
        ctx.lineWidth = 1 + 7 * scale;
        ctx.strokeStyle = signed ? (w >= 0 ? up : down) : color;
        ctx.globalAlpha = Math.min(1, 0.18 + scale);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    },
    [matrix, signed, resting],
  );

  // Size the canvas to the token row after layout; redraw pin if any.
  // The backing store is sized in DEVICE pixels and the element in CSS
  // pixels: at 1:1 the thin arcs -- which are the whole point of the panel --
  // were being upscaled by the compositor and came out blurry on every
  // retina-class screen, including the laptop this is developed on.
  useEffect(() => {
    const canvas = canvasRef.current;
    const row = rowRef.current;
    if (!canvas || !row) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    // Collapse the canvas before measuring. .attn-inner is width:max-content
    // and the canvas is its widest child, so measuring the row while the
    // canvas still carries the PREVIOUS generation's width just returns that
    // width again -- the strip could grow but never shrink, and a 23-token
    // generation kept rendering into a 12,645px box left over from a 267-token
    // one, which looked like an empty panel.
    canvas.style.width = "0px";
    const w = row.scrollWidth;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(CANVAS_H * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${CANVAS_H}px`;
    // Draw in CSS pixels; the transform maps to the device grid.
    canvas.getContext("2d")!.setTransform(dpr, 0, 0, dpr, 0, 0);
    pin(-1);
    draw(-1);
  }, [tokens, draw, pin]);

  useEffect(() => {
    draw(pinned);
  }, [pinned, draw, themeV]);

  // Bring the attributed chip on screen when a ranking arrives. The strip is
  // a horizontal scroller and the position being attributed is usually the
  // last prompt token, which on a long generation is far off the right edge:
  // measured on a 96-token gpt2 run, the ringed chip sat 4872px into a 965px
  // window at scrollLeft 0, so the panel opened on 19 chips of which the only
  // one carrying a bar was the sink at index 0 — while the text above it
  // pointed at the strip four times. FeaturesPanel.tsx does the same thing
  // for its peak activation; this is that rule, applied to the one view whose
  // whole claim is "at THIS token".
  useEffect(() => {
    if (attrPos === undefined || attrPos < 0 || !scores) return;
    attrRef.current?.scrollIntoView({ block: "nearest", inline: "center" });
  }, [attrPos, scores]);

  return (
    <div className="attn-scroll">
      <div className="attn-inner">
        <div className="tokens" ref={rowRef}>
          {tokens.map((t, i) => {
            // undefined = no ranking on screen; null = this chip was not
            // tested. Only a number earns a bar.
            const score = scores?.[i];
            const scored = typeof score === "number";
            const marked = attrPos !== undefined && attrPos >= 0;
            // A candidate the run had no budget for. Index 0 is always tested
            // and is never in this state, so the bound starts at 1.
            const untested =
              marked &&
              !!scores &&
              testedFrom !== undefined &&
              i >= 1 &&
              i < testedFrom &&
              !scored;
            return (
              // Hover and click were the only ways in, so the arc view was
              // pointer-only. Focus mirrors hover and Enter/Space mirrors
              // click, which also makes the strip arrow-free but fully
              // tabbable.
              <span
                key={i}
                ref={marked && i === attrPos ? attrRef : undefined}
                // `gen` marks the model's own output. The strip was one
                // undifferentiated row, so "what you typed" and "what it
                // produced" looked identical — and which is which is the first
                // thing anyone needs to read an attention map.
                className={`tok ${pinned === i ? "pin" : ""} ${
                  nPrompt > 0 && i >= nPrompt ? "gen" : ""
                } ${nPrompt > 0 && i === nPrompt ? "gen-start" : ""} ${
                  marked && i === attrPos ? "attr" : ""
                } ${marked && scores && i > attrPos ? "after-attr" : ""} ${
                  scored ? "scored" : ""
                } ${untested ? "untested" : ""}`}
                style={
                  scored
                    ? ({ "--tok-score": score } as CSSProperties)
                    : undefined
                }
                tabIndex={0}
                role="button"
                aria-pressed={pinned === i}
                aria-label={
                  `token ${i + 1} of ${tokens.length}` +
                  (nPrompt > 0 ? (i >= nPrompt ? ", generated" : ", prompt") : "") +
                  (marked && i === attrPos ? ", the position being attributed" : "") +
                  (marked && scores && i > attrPos
                    ? ", after that position, so not a candidate"
                    : "") +
                  (untested ? ", a candidate this run did not have budget to test" : "") +
                  `: ${t.trim() || "space"}`
                }
                onMouseEnter={() => pinned < 0 && draw(i)}
                onMouseLeave={() => pinned < 0 && draw(-1)}
                onFocus={() => pinned < 0 && draw(i)}
                onBlur={() => pinned < 0 && draw(-1)}
                onClick={() => pin(pinned === i ? -1 : i)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    pin(pinned === i ? -1 : i);
                  }
                }}
              >
                {t.replace(/ /g, "·") || "·"}
              </span>
            );
          })}
        </div>
        <canvas ref={canvasRef} style={{ display: "block" }} />
      </div>
    </div>
  );
}
