import { useCallback, useEffect, useRef, useState } from "react";

interface Props {
  tokens: string[];
  matrix: number[][];
}

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
      ctx.clearRect(0, 0, canvas.width, canvas.height);
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

      const color = getComputedStyle(document.documentElement)
        .getPropertyValue("--model")
        .trim();
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
  useEffect(() => {
    const canvas = canvasRef.current;
    const row = rowRef.current;
    if (!canvas || !row) return;
    canvas.width = row.scrollWidth;
    canvas.style.width = `${row.scrollWidth}px`;
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
            <span
              key={i}
              className={`tok ${pinned === i ? "pin" : ""}`}
              onMouseEnter={() => pinned < 0 && draw(i)}
              onMouseLeave={() => pinned < 0 && draw(-1)}
              onClick={() => setPinned((p) => (p === i ? -1 : i))}
            >
              {t.replace(/ /g, "·") || "·"}
            </span>
          ))}
        </div>
        <canvas ref={canvasRef} height={110} style={{ display: "block" }} />
      </div>
    </div>
  );
}
