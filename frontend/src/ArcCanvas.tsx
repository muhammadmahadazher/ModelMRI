import { useCallback, useEffect, useRef, useState } from "react";

interface Props {
  tokens: string[];
  matrix: number[][];
}

/** Canvas height in CSS pixels. The backing store is this times the DPR. */
const CANVAS_H = 110;

/** Token chips with hover/pin-driven attention arcs drawn on a canvas below. */
export default function ArcCanvas({ tokens, matrix }: Props) {
  const rowRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [pinned, setPinned] = useState(-1);

  const draw = useCallback(
    (i: number) => {
      const canvas = canvasRef.current;
      const row = rowRef.current;
      if (!canvas || !row) return;
      const ctx = canvas.getContext("2d")!;
      // The context is transformed to CSS pixels by the sizing effect, so
      // clear in those units rather than the device-pixel backing size.
      ctx.clearRect(0, 0, row.scrollWidth, CANVAS_H);
      if (i < 0 || !matrix[i]) return;

      const rowRect = row.getBoundingClientRect();
      const chips = Array.from(row.children) as HTMLElement[];
      const centers = chips.map((el) => {
        const r = el.getBoundingClientRect();
        return r.left - rowRect.left + r.width / 2;
      });

      const edges = matrix[i]
        .map((w, j) => ({ w, j }))
        .filter(({ w, j }) => j <= i && w >= 0.02)
        .sort((a, b) => b.w - a.w)
        .slice(0, 12);

      // --model is not a variable this stylesheet defines. getPropertyValue
      // returned "", canvas ignores an unparseable strokeStyle, and the arcs
      // silently drew in the default black instead of the attention colour.
      const color =
        getComputedStyle(document.documentElement)
          .getPropertyValue("--color-attn")
          .trim() || "#1a5fd0";
      for (const { w, j } of edges) {
        const x1 = centers[i];
        const x2 = centers[j];
        const depth = Math.min(100, 18 + Math.abs(x1 - x2) * 0.12);
        ctx.beginPath();
        ctx.moveTo(x1, 4);
        ctx.quadraticCurveTo((x1 + x2) / 2, depth, x2, 4);
        ctx.lineWidth = 1 + 7 * w;
        ctx.strokeStyle = color;
        ctx.globalAlpha = Math.min(1, 0.18 + w);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    },
    [matrix],
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
    const w = row.scrollWidth;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(CANVAS_H * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${CANVAS_H}px`;
    // Draw in CSS pixels; the transform maps to the device grid.
    canvas.getContext("2d")!.setTransform(dpr, 0, 0, dpr, 0, 0);
    setPinned(-1);
    draw(-1);
  }, [tokens, draw]);

  useEffect(() => {
    draw(pinned);
  }, [pinned, draw]);

  return (
    <div className="attn-scroll">
      <div className="attn-inner">
        <div className="tokens" ref={rowRef}>
          {tokens.map((t, i) => (
            // Hover and click were the only ways in, so the arc view was
            // pointer-only. Focus mirrors hover and Enter/Space mirrors click,
            // which also makes the strip arrow-free but fully tabbable.
            <span
              key={i}
              className={`tok ${pinned === i ? "pin" : ""}`}
              tabIndex={0}
              role="button"
              aria-pressed={pinned === i}
              aria-label={`token ${i + 1} of ${tokens.length}: ${t.trim() || "space"}`}
              onMouseEnter={() => pinned < 0 && draw(i)}
              onMouseLeave={() => pinned < 0 && draw(-1)}
              onFocus={() => pinned < 0 && draw(i)}
              onBlur={() => pinned < 0 && draw(-1)}
              onClick={() => setPinned((p) => (p === i ? -1 : i))}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  setPinned((p) => (p === i ? -1 : i));
                }
              }}
            >
              {t.replace(/ /g, "·") || "·"}
            </span>
          ))}
        </div>
        <canvas ref={canvasRef} style={{ display: "block" }} />
      </div>
    </div>
  );
}
