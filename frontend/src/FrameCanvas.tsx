import { useEffect, useRef } from "react";

interface Props {
  src: string;
  heat: number[][] | null;
  alpha?: number;
  scale?: number;
}

/** Robot camera frame with the policy's attention painted over it.
 *  Base layer: nearest-neighbour upscale so 96px pixels stay crisp.
 *  Heat layer: G x G drawn into an offscreen canvas, then bilinear-stretched. */
export default function FrameCanvas({ src, heat, alpha = 0.55, scale = 4 }: Props) {
  const baseRef = useRef<HTMLCanvasElement>(null);
  const heatRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = baseRef.current;
    if (!canvas || !src) return;
    let live = true;
    const img = new Image();
    img.onload = () => {
      if (!live) return;
      const w = img.width * scale;
      const h = img.height * scale;
      canvas.width = w;
      canvas.height = h;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      const ctx = canvas.getContext("2d")!;
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(img, 0, 0, w, h);
      const overlay = heatRef.current;
      if (overlay) {
        overlay.width = w;
        overlay.height = h;
        overlay.style.width = `${w}px`;
        overlay.style.height = `${h}px`;
      }
    };
    img.src = src;
    return () => {
      live = false;
    };
  }, [src, scale]);

  useEffect(() => {
    const overlay = heatRef.current;
    if (!overlay) return;
    const ctx = overlay.getContext("2d")!;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (!heat || !heat.length) return;

    const rows = heat.length;
    const cols = heat[0].length;
    const off = document.createElement("canvas");
    off.width = cols;
    off.height = rows;
    const octx = off.getContext("2d")!;
    const data = octx.createImageData(cols, rows);
    const css = getComputedStyle(document.documentElement);
    const cool = css.getPropertyValue("--color-vla").trim() || "#1e7a46";
    const hot = css.getPropertyValue("--color-pop").trim() || "#d92b5b";
    const c1 = hexToRgb(cool);
    const c2 = hexToRgb(hot);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = Math.max(0, Math.min(1, heat[r][c]));
        const i = (r * cols + c) * 4;
        data.data[i] = c1[0] + (c2[0] - c1[0]) * v;
        data.data[i + 1] = c1[1] + (c2[1] - c1[1]) * v;
        data.data[i + 2] = c1[2] + (c2[2] - c1[2]) * v;
        // low attention stays nearly transparent so the frame reads through
        data.data[i + 3] = Math.round(255 * alpha * Math.pow(v, 1.4));
      }
    }
    octx.putImageData(data, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, 0, 0, overlay.width, overlay.height);
  }, [heat, alpha, src]);

  return (
    <div className="vla-frame">
      <canvas ref={baseRef} />
      <canvas ref={heatRef} className="vla-heat" />
    </div>
  );
}

function hexToRgb(hex: string): [number, number, number] {
  const m = hex.replace("#", "");
  const n = parseInt(m.length === 3 ? m.replace(/(.)/g, "$1$1") : m, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
