// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

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

/** THREE AXES, NOT TWO.
 *
 *  A PALETTE is which hues. A THEME is light or dark. A CONTRAST LEVEL is how
 *  far the ink sits from the ground. All three are independent, and getting
 *  that right is the difference between forty-eight real combinations and
 *  four-plus-one-that-does-not-compose.
 *
 *  It used to be two, and the second was a lie: "High contrast" shipped as a
 *  PALETTE, so it existed in exactly one hue family. Somebody who needs high
 *  contrast AND finds cobalt hard to look at had nowhere to go, and somebody
 *  who wanted amber at higher contrast had nowhere to go either — the two
 *  requests were mutually exclusive for no reason but the data model.
 *
 *  Material defines contrast as its own axis with exactly three levels,
 *  "automatically applied to both light and dark themes". So the document now
 *  carries three attributes: `data-palette` for the hues, `data-theme` for the
 *  mode, and `data-contrast` for the level. Every existing rule keys off the
 *  first two exactly as it did. */
export type Palette =
  | "paper"
  | "slate"
  | "amber"
  | "forest"
  | "ocean"
  | "plum"
  | "rose"
  | "graphite";

export const PALETTES: { key: Palette; label: string; why: string }[] = [
  { key: "paper", label: "Paper", why: "Warm neutral. The original." },
  { key: "slate", label: "Slate", why: "Cool blue-grey, lower colour temperature." },
  { key: "amber", label: "Amber", why: "Warm terminal. Easier late at night." },
  { key: "forest", label: "Forest", why: "Green-leaning neutral, softer accents." },
  { key: "ocean", label: "Ocean", why: "Cold blue-green. The furthest from paper." },
  { key: "plum", label: "Plum", why: "Violet-leaning. The one that is not a neutral pretending." },
  { key: "rose", label: "Rose", why: "Warm without yellow, for long sessions on a bright screen." },
  {
    key: "graphite",
    label: "Graphite",
    why: "Almost colourless, so the panels' own hues are the only colour on the page.",
  },
];

/** Material's three levels, and only these three. `standard` is the current
 *  behaviour; `medium` guarantees at least 3:1 for all text (it exists for
 *  readers who need more than standard but for whom high contrast halates);
 *  `high` reaches 7:1 for body text. Measured worst cases live beside the
 *  token blocks in styles.css. */
export type Contrast = "standard" | "medium" | "high";

export const CONTRASTS: { key: Contrast; label: string; why: string; glyph: string }[] = [
  {
    key: "standard",
    label: "Standard contrast",
    why: "Mixed levels of contrast, to reduce cognitive load.",
    glyph: "◔",
  },
  {
    key: "medium",
    label: "Medium contrast",
    why: "More separation, without the halation that high contrast can cause.",
    glyph: "◑",
  },
  {
    key: "high",
    label: "High contrast",
    why: "7:1 for body text, for low vision or bright sun. Works with every palette.",
    glyph: "●",
  },
];

const KEY = "modelmri:theme";
const PALETTE_KEY = "modelmri:palette";
const CONTRAST_KEY = "modelmri:contrast";

export function storedPalette(): Palette {
  try {
    const v = localStorage.getItem(PALETTE_KEY);
    if (PALETTES.some((p) => p.key === v)) return v as Palette;
  } catch {
    /* private mode, or storage disabled — fall through to the default */
  }
  return "paper";
}

/** Anyone who had chosen the old "contrast" PALETTE chose an accessibility
 *  setting, not a hue family, and dropping them back to Paper/standard would
 *  silently take that setting away from the one group who cannot afford to
 *  lose it. The stored value is migrated instead: hues go to the default,
 *  the intent goes to the axis that now carries it. */
export function storedContrast(): Contrast {
  try {
    if (localStorage.getItem(PALETTE_KEY) === "contrast") return "high";
    const v = localStorage.getItem(CONTRAST_KEY);
    if (v === "standard" || v === "medium" || v === "high") return v;
  } catch {
    /* private mode, or storage disabled — fall through to the default */
  }
  return "standard";
}

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
export function apply(
  choice: ThemeChoice,
  palette?: Palette,
  contrast?: Contrast,
): Resolved {
  const mode = resolve(choice);
  const root = document.documentElement;
  root.dataset.theme = mode;
  // All three are always stamped, including for the defaults — a rule can
  // then be written against `[data-palette="paper"]` or
  // `[data-contrast="standard"]` without having to also handle the attribute
  // being absent, which is the kind of asymmetry that produces one theme with
  // a subtly different rule set.
  root.dataset.palette = palette ?? storedPalette();
  root.dataset.contrast = contrast ?? storedContrast();
  root.style.colorScheme = mode;
  try {
    localStorage.setItem(KEY, choice);
    if (palette) localStorage.setItem(PALETTE_KEY, palette);
    if (contrast) localStorage.setItem(CONTRAST_KEY, contrast);
  } catch {
    /* not worth failing a theme switch over */
  }
  repaint(mode);
  return mode;
}

/** Set the hue family, and nothing else. */
export function applyPalette(palette: Palette): void {
  document.documentElement.dataset.palette = palette;
  try {
    localStorage.setItem(PALETTE_KEY, palette);
  } catch {
    /* not worth failing a palette switch over */
  }
  repaint(resolve(storedChoice()));
}

/** Set the contrast level, and nothing else. Persisted exactly as palette and
 *  theme are — an accessibility setting that has to be re-chosen every visit
 *  is not a setting. */
export function applyContrast(contrast: Contrast): void {
  document.documentElement.dataset.contrast = contrast;
  try {
    localStorage.setItem(CONTRAST_KEY, contrast);
  } catch {
    /* not worth failing a contrast switch over */
  }
  repaint(resolve(storedChoice()));
}

/** Canvases read CSS variables at DRAW time and their pixels are already
 *  rasterised, so a palette or contrast change has to bump this exactly as a
 *  mode change does — otherwise three panels stay painted in the previous
 *  palette until something unrelated triggers a redraw. */
function repaint(mode: Resolved): void {
  window.dispatchEvent(new CustomEvent("modelmri:theme", { detail: mode }));
}

export function useTheme() {
  const [choice, setChoice] = useState<ThemeChoice>(storedChoice);
  const [palette, setPalette] = useState<Palette>(storedPalette);
  const [contrast, setContrast] = useState<Contrast>(storedContrast);
  const [mode, setMode] = useState<Resolved>(() => resolve(storedChoice()));

  // ONE EFFECT PER AXIS, and each one passes only its own value.
  //
  // This used to be a single effect on [choice, palette] calling
  // apply(choice, palette), and that quietly reverted settings: every caller
  // of useTheme() holds its OWN copy of all three values, so after the
  // palette picker switched to Amber, the theme toggle still believed the
  // palette was Paper — and the next click on light/dark wrote that stale
  // Paper back over it. With a third axis the same bug becomes a three-way
  // round robin where no two controls can be used in sequence. Writing only
  // the axis that actually changed makes the collision impossible, and the
  // resync below keeps every copy honest.
  useEffect(() => {
    setMode(apply(choice));
  }, [choice]);

  useEffect(() => {
    applyPalette(palette);
  }, [palette]);

  useEffect(() => {
    applyContrast(contrast);
  }, [contrast]);

  // Every control that changes an axis dispatches `modelmri:theme`; every
  // copy of this hook re-reads storage when it fires. Setting state to the
  // value it already holds is a no-op in React, so this cannot loop.
  useEffect(() => {
    const sync = () => {
      setChoice(storedChoice());
      setPalette(storedPalette());
      setContrast(storedContrast());
      setMode(resolve(storedChoice()));
    };
    window.addEventListener("modelmri:theme", sync);
    return () => window.removeEventListener("modelmri:theme", sync);
  }, []);

  // Following the OS live is the whole point of "system"; without this the
  // page only tracks it on reload.
  useEffect(() => {
    if (choice !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => setMode(apply("system"));
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [choice]);

  return { choice, setChoice, mode, palette, setPalette, contrast, setContrast };
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
