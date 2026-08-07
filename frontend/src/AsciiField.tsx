import { useEffect, useRef } from "react";

/** Live ASCII-dither field — flowing gradient forms rendered as a character
 *  grid (the machine's eye rendering itself). Cheap: ~10fps, one canvas,
 *  pauses when hidden, static single frame under prefers-reduced-motion. */

const RAMP = [" ", " ", ".", "·", ":", ";", "+", "=", "x", "#", "@"];
const CELL = 17;
const FPS = 10;

function field(x: number, y: number, t: number): number {
  // layered trig flow — organic enough, no noise lib needed
  const a = Math.sin(x * 0.011 + t * 0.35) + Math.cos(y * 0.013 - t * 0.22);
  const b = Math.sin((x + y) * 0.007 + t * 0.18);
  const c = Math.sin(Math.hypot(x - 900, y - 80) * 0.004 - t * 0.4);
  return (a + b + c + 3) / 6; // 0..1
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

    const size = () => {
      const r = canvas.parentElement!.getBoundingClientRect();
      const dpr = Math.min(devicePixelRatio, 2);
      canvas.width = r.width * dpr;
      canvas.height = r.height * dpr;
      canvas.style.width = `${r.width}px`;
      canvas.style.height = `${r.height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.font = `12px ui-monospace, Consolas, monospace`;
      ctx.textBaseline = "top";
    };

    const draw = (t: number) => {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);
      for (let y = 0; y < h; y += CELL) {
        for (let x = 0; x < w; x += CELL) {
          const v = field(x, y, t);
          const glyph = RAMP[Math.min(RAMP.length - 1, (v * RAMP.length) | 0)];
          if (glyph === " ") continue;
          // cobalt-on-paper (the VANTAGE recipe): deep blue, density = ink
          const hueMix = (Math.sin(x * 0.004 - t * 0.12) + 1) / 2;
          const r = 25 + hueMix * 40; //  cobalt band
          const g = 55 + hueMix * 30;
          const b = 190 + hueMix * 50;
          ctx.fillStyle = `rgba(${r | 0},${g | 0},${b | 0},${0.18 + v * 0.6})`;
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

    size();
    const ro = new ResizeObserver(size);
    ro.observe(canvas.parentElement!);
    document.addEventListener("visibilitychange", onVis);

    draw(1.7); // paint frame 1 synchronously — no blank hero before rAF fires
    if (!reduced) {
      raf = requestAnimationFrame(loop);
    }

    return () => {
      running = false;
      cancelAnimationFrame(raf);
      ro.disconnect();
      document.removeEventListener("visibilitychange", onVis);
    };
  }, []);

  return <canvas ref={ref} className="ascii" aria-hidden="true" />;
}
