/** Theme state, and the one thing everyone forgets: telling the canvases.
 *
 *  Three of our surfaces paint to <canvas> by reading CSS custom properties
 *  at draw time (attention arcs, the VLA frame heat, the hero field). CSS
 *  variables re-cascade for free when the theme flips; a canvas does not --
 *  its pixels are already rasterised. So a theme switch that only swaps the
 *  stylesheet leaves three panels painted in the old palette until something
 *  unrelated happens to trigger a redraw.
 *
 *  `useThemeVersion()` returns a counter that changes on every theme change.
 *  Put it in a draw effect's dependency array and the canvas repaints.
 */

import { useEffect, useState } from "react";

export type ThemeChoice = "light" | "dark" | "system";
export type Resolved = "light" | "dark";

const KEY = "modelmri:theme";

export function storedChoice(): ThemeChoice {
  try {
    const v = localStorage.getItem(KEY);
    if (v === "light" || v === "dark" || v === "system") return v;
  } catch {
    /* private mode, or storage disabled -- fall through to the default */
  }
  return "system";
}

export function systemPrefersDark(): boolean {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
}

export function resolve(choice: ThemeChoice): Resolved {
  return choice === "system" ? (systemPrefersDark() ? "dark" : "light") : choice;
}

/** Write the theme to the document. `color-scheme` is not decoration: it is
 *  what makes form controls, scrollbars and the canvas backdrop follow. */
export function apply(choice: ThemeChoice): Resolved {
  const mode = resolve(choice);
  const root = document.documentElement;
  root.dataset.theme = mode;
  root.style.colorScheme = mode;
  try {
    localStorage.setItem(KEY, choice);
  } catch {
    /* not worth failing a theme switch over */
  }
  window.dispatchEvent(new CustomEvent("modelmri:theme", { detail: mode }));
  return mode;
}

export function useTheme() {
  const [choice, setChoice] = useState<ThemeChoice>(storedChoice);
  const [mode, setMode] = useState<Resolved>(() => resolve(storedChoice()));

  useEffect(() => {
    setMode(apply(choice));
  }, [choice]);

  // Following the OS live is the whole point of "system"; without this the
  // page only tracks it on reload.
  useEffect(() => {
    if (choice !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => setMode(apply("system"));
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [choice]);

  return { choice, setChoice, mode };
}

/** A counter that bumps whenever the theme changes. Canvases depend on it. */
export function useThemeVersion(): number {
  const [v, setV] = useState(0);
  useEffect(() => {
    const bump = () => setV((n) => n + 1);
    window.addEventListener("modelmri:theme", bump);
    return () => window.removeEventListener("modelmri:theme", bump);
  }, []);
  return v;
}

/** Read a themed colour for canvas painting. Canvases cannot use var(). */
export function cssColor(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

let probe: CanvasRenderingContext2D | null = null;

/** Any CSS colour -> [r,g,b], by asking the browser to rasterise one pixel.
 *
 *  The obvious version of this parses "#rrggbb" with a regex, and it works
 *  right up until the palette moves to oklch() -- at which point it silently
 *  returns black and a heatmap turns into a grey smear. Canvas already knows
 *  how to parse every colour syntax the browser supports; use it. */
export function toRgb(colour: string, fallback: [number, number, number] = [0, 0, 0]) {
  if (!probe) {
    const c = document.createElement("canvas");
    c.width = c.height = 1;
    probe = c.getContext("2d", { willReadFrequently: true });
  }
  if (!probe) return fallback;
  try {
    probe.clearRect(0, 0, 1, 1);
    probe.fillStyle = "#000";
    probe.fillStyle = colour; // ignored if unparseable, leaving the sentinel
    probe.fillRect(0, 0, 1, 1);
    const [r, g, b] = probe.getImageData(0, 0, 1, 1).data;
    return [r, g, b] as [number, number, number];
  } catch {
    return fallback;
  }
}
