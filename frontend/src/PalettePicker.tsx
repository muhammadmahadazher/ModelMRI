import { PALETTES, useTheme } from "./theme";

/**
 * Pick the hues. The light/dark toggle beside this picks the mode.
 *
 * Two controls rather than one list of ten, because they are orthogonal:
 * somebody who needs high contrast needs it in the dark too, and somebody who
 * likes the amber terminal still switches to light at midday. A single list
 * would make "Amber" quietly mean "Amber, dark", and the first person to want
 * amber-light would have nowhere to go.
 *
 * Each swatch shows that palette's own ground and accent, so the control is a
 * sample of the thing rather than a label for it — you can see what you are
 * choosing before you choose it, which is the whole reason a colour picker is
 * swatches and not a dropdown.
 *
 * The `title` carries WHY each exists rather than repeating its name. "Amber"
 * tells you nothing you cannot see in the swatch; "warm terminal, easier late
 * at night" tells you when to pick it.
 */
export default function PalettePicker() {
  const { palette, setPalette } = useTheme();

  return (
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
  );
}
