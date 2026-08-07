import { useEffect, useRef } from "react";
import { cssColor, toRgb, useThemeVersion } from "./theme";

/** Live ASCII-dither field — flowing gradient forms rendered as a character
 *  grid (the machine's eye rendering itself). Cheap: ~10fps, one canvas,
 *  pauses when hidden, static single frame under prefers-reduced-motion. */

const RAMP = [" ", " ", ".", "·", ":", ";", "+", "=", "*", "x", "#", "@"];
const CELL = 14;
const FPS = 24;

/* SIGNATURE — the field is a fingerprint, not a screensaver.
 *
 * The composition is seeded deterministically from the loaded checkpoint, so
 * every model gets its own stable field: load Qwen3 and you always get the
 * same one, load Gemma and you get a different one, and a reload does not
 * reroll it. An introspection tool whose hero is unseeded noise is announcing
 * that its visuals and its data have nothing to do with each other.
 *
 * Every parameter is drawn inside a hand-checked envelope rather than from an
 * open range, so no seed can produce a composition nobody has ever seen. */
const fnv1a = (s: string) => {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
};

const mulberry32 = (a: number) => () => {
  a |= 0;
  a = (a + 0x6d2b79f5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

export interface FieldParams {
  fx: number; fy: number; fd: number; fr: number;
  phase: number; cx: number; cy: number;
}

export function paramsFor(modelId: string | null | undefined): FieldParams {
  const rng = mulberry32(fnv1a(modelId ? `mri:${modelId}` : "vantage:0xC0FFEE"));
  return {
    fx: 0.008 + rng() * 0.005,
    fy: 0.010 + rng() * 0.005,
    fd: 0.004 + rng() * 0.004,
    fr: 0.016 + rng() * 0.008,
    phase: rng() * Math.PI * 2,
    cx: 0.38 + rng() * 0.24, // centre stays in the middle quarter
    cy: 0.46 + rng() * 0.18,
  };
}

// Pointer influence: the field leans toward the cursor, so the hero feels
// like an instrument you're touching rather than a looping background.
let px = -1e4;
let py = -1e4;
let pInfluence = 0;

function field(
  x: number, y: number, t: number, w: number, h: number, P: FieldParams,
): number {
  // three drifting wave systems + a slow radial pulse, all seeded
  const a = Math.sin(x * P.fx + t * 0.30 + P.phase) + Math.cos(y * P.fy - t * 0.20);
  const b = Math.sin((x + y) * P.fd + t * 0.16);
  const r = Math.hypot(x - w * P.cx, y - h * P.cy);
  const c = Math.sin(r * P.fr - t * 0.9) * 0.8;
  let v = (a + b + c + 3) / 6;

  // a soft lens that follows the cursor, brightening nearby cells
  if (pInfluence > 0.01) {
    const d = Math.hypot(x - px, y - py);
    v += pInfluence * Math.exp(-(d * d) / 9000) * 0.55;
  }
  return v;
}

export default function AsciiField({ modelId }: { modelId?: string | null }) {
  const themeV = useThemeVersion();
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

    // Both ends of the ink ramp come from the palette, resolved once per
    // effect run -- this loop paints ~2500 glyphs a frame and cannot afford
    // a getComputedStyle call inside it.
    const cold = toRgb(cssColor("--acc", "#2743e0"), [39, 67, 224]);
    const hot = toRgb(cssColor("--sem-feat", "#6c4ee0"), [108, 78, 224]);

    const P = paramsFor(modelId);

    const draw = (t: number) => {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);
      // ease the pointer lens in/out so entering and leaving both feel soft
      pInfluence += ((hovering ? 1 : 0) - pInfluence) * 0.08;

      for (let y = 0; y < h; y += CELL) {
        for (let x = 0; x < w; x += CELL) {
          const v = field(x, y, t, w, h, P);
          const glyph = RAMP[Math.min(RAMP.length - 1, Math.max(0, (v * RAMP.length) | 0))];
          if (glyph === " ") continue;
          // cobalt -> magenta as density rises: the ink "heats up" where the
          // field is strongest, echoing the attention heatmaps below
          const heat = Math.max(0, Math.min(1, (v - 0.55) * 1.9));
          const r = cold[0] + (hot[0] - cold[0]) * heat;
          const g = cold[1] + (hot[1] - cold[1]) * heat;
          const b = cold[2] + (hot[2] - cold[2]) * heat;
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
  }, [themeV, modelId]);

  return <canvas ref={ref} className="ascii" aria-hidden="true" />;
}
