import { useEffect, useRef } from "react";

/** Live ASCII-dither field — flowing gradient forms rendered as a character
 *  grid (the machine's eye rendering itself). Cheap: ~10fps, one canvas,
 *  pauses when hidden, static single frame under prefers-reduced-motion. */

const RAMP = [" ", " ", ".", "·", ":", ";", "+", "=", "*", "x", "#", "@"];
const CELL = 14;
const FPS = 24;

// Pointer influence: the field leans toward the cursor, so the hero feels
// like an instrument you're touching rather than a looping background.
let px = -1e4;
let py = -1e4;
let pInfluence = 0;

function field(x: number, y: number, t: number, w: number, h: number): number {
  // three drifting wave systems + a slow radial pulse from the centre
  const a = Math.sin(x * 0.010 + t * 0.30) + Math.cos(y * 0.012 - t * 0.20);
  const b = Math.sin((x + y) * 0.006 + t * 0.16);
  const r = Math.hypot(x - w * 0.5, y - h * 0.55);
  const c = Math.sin(r * 0.020 - t * 0.9) * 0.8;
  let v = (a + b + c + 3) / 6;

  // a soft lens that follows the cursor, brightening nearby cells
  if (pInfluence > 0.01) {
    const d = Math.hypot(x - px, y - py);
    v += pInfluence * Math.exp(-(d * d) / 9000) * 0.55;
  }
  return v;
}

export default function AsciiField() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext("2d")!;
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    let last = 0;
    let running = true;
    let hovering = false;

    const size = () => {
      // CSS owns the canvas box — we only match the pixel buffer to it.
      // (Never write style.width/height here: sizing from the parent and
      // styling ourselves once caused an unbounded grow loop.)
      const r = canvas.getBoundingClientRect();
      const dpr = Math.min(devicePixelRatio, 2);
      const w = Math.round(r.width * dpr);
      const h = Math.round(r.height * dpr);
      if (canvas.width === w && canvas.height === h) return;
      canvas.width = w;
      canvas.height = h;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.font = `12px ui-monospace, Consolas, monospace`;
      ctx.textBaseline = "top";
    };

    const draw = (t: number) => {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);
      // ease the pointer lens in/out so entering and leaving both feel soft
      pInfluence += ((hovering ? 1 : 0) - pInfluence) * 0.08;

      for (let y = 0; y < h; y += CELL) {
        for (let x = 0; x < w; x += CELL) {
          const v = field(x, y, t, w, h);
          const glyph = RAMP[Math.min(RAMP.length - 1, Math.max(0, (v * RAMP.length) | 0))];
          if (glyph === " ") continue;
          // cobalt -> magenta as density rises: the ink "heats up" where the
          // field is strongest, echoing the attention heatmaps below
          const heat = Math.max(0, Math.min(1, (v - 0.55) * 1.9));
          const r = 25 + heat * 192;
          const g = 55 - heat * 12;
          const b = 224 - heat * 133;
          ctx.fillStyle = `rgba(${r | 0},${g | 0},${b | 0},${0.16 + v * 0.62})`;
          ctx.fillText(glyph, x, y);
        }
      }
    };

    const loop = (now: number) => {
      if (!running) return;
      if (now - last >= 1000 / FPS) {
        last = now;
        draw(now / 1000);
      }
      raf = requestAnimationFrame(loop);
    };

    const onVis = () => {
      running = !document.hidden && !reduced;
      if (running) raf = requestAnimationFrame(loop);
    };

    const onMove = (e: PointerEvent) => {
      const r = canvas.getBoundingClientRect();
      px = e.clientX - r.left;
      py = e.clientY - r.top;
      hovering = true;
    };
    const onLeave = () => {
      hovering = false;
    };

    size();
    const ro = new ResizeObserver(() => {
      size();
      draw(1.7); // repaint after any resize so the field is never blank
    });
    ro.observe(canvas);
    document.addEventListener("visibilitychange", onVis);
    if (!reduced) {
      canvas.addEventListener("pointermove", onMove);
      canvas.addEventListener("pointerleave", onLeave);
    }

    draw(1.7); // paint frame 1 synchronously — no blank hero before rAF fires
    if (!reduced) {
      raf = requestAnimationFrame(loop);
    }

    return () => {
      running = false;
      cancelAnimationFrame(raf);
      ro.disconnect();
      document.removeEventListener("visibilitychange", onVis);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
    };
  }, []);

  return <canvas ref={ref} className="ascii" aria-hidden="true" />;
}
