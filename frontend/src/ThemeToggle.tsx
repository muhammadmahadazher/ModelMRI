import { ThemeChoice, useTheme } from "./theme";

const ORDER: ThemeChoice[] = ["light", "system", "dark"];

const GLYPH: Record<ThemeChoice, string> = {
  light: "☀",
  system: "◐",
  dark: "☾",
};

const LABEL: Record<ThemeChoice, string> = {
  light: "Light",
  system: "Match system",
  dark: "Dark",
};

/** Three states, not two. A binary toggle silently opts you out of the OS
 *  setting the first time you touch it, and there is then no way back. */
export default function ThemeToggle() {
  const { choice, setChoice, mode } = useTheme();

  return (
    <div
      className="theme-seg"
      role="radiogroup"
      aria-label={`Colour theme, currently ${mode}`}
    >
      {ORDER.map((t) => (
        <button
          key={t}
          role="radio"
          aria-checked={choice === t}
          aria-label={LABEL[t]}
          title={LABEL[t]}
          className={choice === t ? "on" : ""}
          onClick={() => setChoice(t)}
        >
          <span aria-hidden="true">{GLYPH[t]}</span>
        </button>
      ))}
    </div>
  );
}
