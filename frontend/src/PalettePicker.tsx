import { CONTRASTS, PALETTES, useTheme } from "./theme";

/**
 * Pick the hues, and pick the contrast level. The light/dark toggle beside
 * this picks the mode.
 *
 * THREE controls rather than one list, because they are three orthogonal
 * axes: somebody who needs high contrast needs it in the dark too, and in
 * whichever hue family they can actually look at. This shipped as two, and
 * the second was a lie — "High contrast" was one of the palettes, so choosing
 * it meant giving up amber, and choosing amber meant giving up the
 * accessibility setting. Nobody should have to trade those against each
 * other, and with the contrast axis split out nobody does: all eight palettes
 * work at all three levels in both modes.
 *
 * Each swatch shows that palette's own ground and accent — and now shows them
 * in the CURRENT mode and contrast level, because the swatch is composed from
 * the same tokens the page is. Picking "Amber" while in dark high-contrast
 * shows you amber in dark high-contrast rather than a light-mode chip of it.
 *
 * The contrast control is labelled with a filling disc rather than words:
 * three words do not fit a phone topbar, and the amount of ink in the glyph
 * is the quantity being chosen. It is never colour alone — the selected item
 * carries a filled plate, its own elevation and the accent colour, and the
 * radio group reports the choice to a screen reader regardless.
 *
 * The `title` on each carries WHY it exists rather than repeating its name.
 * "Amber" tells you nothing the swatch does not; "warm terminal, easier late
 * at night" tells you when to pick it.
 */
export default function PalettePicker() {
  const { palette, setPalette, contrast, setContrast } = useTheme();

  return (
    <>
      <div
        className="palette-pick"
        role="radiogroup"
        aria-label={`Colour palette, currently ${palette}`}
      >
        {PALETTES.map((p) => (
          <button
            key={p.key}
            type="button"
            role="radio"
            aria-checked={palette === p.key}
            aria-label={`${p.label}. ${p.why}`}
            title={`${p.label} — ${p.why}`}
            data-p={p.key}
            className={`palette-dot${palette === p.key ? " on" : ""}`}
            onClick={() => setPalette(p.key)}
          />
        ))}
      </div>

      <div
        className="theme-seg contrast-seg"
        role="radiogroup"
        aria-label={`Contrast level, currently ${contrast}`}
      >
        {CONTRASTS.map((c) => (
          <button
            key={c.key}
            type="button"
            role="radio"
            aria-checked={contrast === c.key}
            aria-label={`${c.label}. ${c.why}`}
            title={`${c.label} — ${c.why}`}
            className={contrast === c.key ? "on" : ""}
            onClick={() => setContrast(c.key)}
          >
            <span className="glyph" aria-hidden="true">
              {c.glyph}
            </span>
          </button>
        ))}
      </div>
    </>
  );
}
