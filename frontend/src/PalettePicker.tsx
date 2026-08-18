import { useCallback, useEffect, useId, useRef, useState } from "react";
import { CONTRASTS, PALETTES, useTheme } from "./theme";

/**
 * The hues and the contrast level, behind a disclosure. The light/dark toggle
 * beside this stays in the open, because that is the control people reach for.
 *
 * WHY IT IS HIDDEN NOW, HAVING BEEN OPEN
 *
 * Three orthogonal axes is the right MODEL — somebody who needs high contrast
 * needs it in the dark too, and in whichever hue family they can actually look
 * at — but rendering all three at once put fourteen controls in a topbar that
 * also carries a wordmark, an accelerator badge, a model pill and two actions.
 * The maintainer's word for the result was "congested", and they were right:
 * mode is a daily choice, hue and contrast are set once and left, and giving
 * all three the same prominence spent the most valuable strip on the page on
 * the two nobody touches twice.
 *
 * So the axes stay three. Only their PLACEMENT changes: mode in the topbar,
 * the other two one click away. Nothing became unreachable and nothing became
 * a mode of something else.
 *
 * WHY THE TRIGGER IS A SWATCH RATHER THAN A GEAR
 *
 * Collapsing a control usually hides its state along with it, and then the
 * only way to answer "which palette am I on?" is to open the thing. The
 * trigger is the current palette's own swatch, composed from the same tokens
 * the page is — so the answer is on screen while the panel is shut, in one
 * 18px disc instead of eight.
 *
 * The panel is absolutely positioned, so opening it moves nothing underneath.
 * A topbar that reflows when you look at it is the congestion complaint again
 * in a different form.
 */
export default function PalettePicker() {
  const { palette, setPalette, contrast, setContrast } = useTheme();
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  const panelId = useId();

  const current = PALETTES.find((p) => p.key === palette);
  const level = CONTRASTS.find((c) => c.key === contrast);

  /** Shut it, and put focus back where it came from.
   *
   *  The focus half is not optional. Dismissing a panel that holds focus and
   *  leaving it on the removed node drops the caret to the top of the
   *  document, so the next Tab starts from the wordmark — a keyboard user
   *  loses their place every time they change a colour. */
  const close = useCallback((refocus: boolean) => {
    setOpen(false);
    if (refocus) trigger.current?.focus();
  }, []);

  useEffect(() => {
    if (!open) return;

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        close(true);
      }
    };
    // `pointerdown` rather than `click`: a click that starts inside the panel
    // and finishes outside it (dragging off a swatch, or a text selection)
    // fires `click` on the document and would shut the panel mid-gesture.
    const onDown = (e: PointerEvent) => {
      if (!wrap.current?.contains(e.target as Node)) close(false);
    };
    // Focus leaving the panel by keyboard is a dismissal too, and one that
    // `pointerdown` cannot see. Without this, tabbing past the last swatch
    // leaves an open panel floating over a page you are no longer in.
    const onFocus = (e: FocusEvent) => {
      if (!wrap.current?.contains(e.target as Node)) setOpen(false);
    };

    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("focusin", onFocus);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("focusin", onFocus);
    };
  }, [open, close]);

  /** Open with focus on the palette you are already on, not on the first one.
   *
   *  A keyboard user arriving at "Paper" when they are running Amber has to
   *  count across to find themselves; arriving ON Amber means the arrow keys
   *  move relative to the truth. */
  useEffect(() => {
    if (!open) return;
    const marked = panel.current?.querySelector<HTMLElement>(
      '.palette-dot[aria-checked="true"]',
    );
    (marked ?? panel.current?.querySelector<HTMLElement>(".palette-dot"))?.focus();
  }, [open]);

  /** Arrow keys move within the swatches, the way a radio group is expected
   *  to. `PALETTES` is the source of order here rather than the DOM, so this
   *  cannot drift from what is rendered. */
  const onSwatchKey = (e: React.KeyboardEvent) => {
    const keys = ["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", "Home", "End"];
    if (!keys.includes(e.key)) return;
    e.preventDefault();
    const at = PALETTES.findIndex((p) => p.key === palette);
    const last = PALETTES.length - 1;
    // One expression rather than a seeded `let` and four assignments: the
    // chain is exhaustive, so the seed was written and never read. Both
    // wrapping arms also survive `at === -1` — a stored palette this build no
    // longer ships — landing on the first or last swatch rather than
    // indexing off the end.
    const next =
      e.key === "Home"
        ? 0
        : e.key === "End"
          ? last
          : e.key === "ArrowRight" || e.key === "ArrowDown"
            ? at >= last
              ? 0
              : at + 1
            : at <= 0
              ? last
              : at - 1;
    setPalette(PALETTES[next].key);
    panel.current
      ?.querySelectorAll<HTMLElement>(".palette-dot")
      [next]?.focus();
  };

  return (
    <div className="theme-more" ref={wrap}>
      <button
        ref={trigger}
        type="button"
        className={`theme-more-btn${open ? " open" : ""}`}
        aria-expanded={open}
        aria-controls={panelId}
        // Names the current state, because the collapsed control has to
        // answer "what am I on?" to a screen reader as well as to an eye.
        // The contrast labels already end in the word "contrast", so this
        // does not add one — "contrast Standard contrast" is what happens
        // when a template assumes it is naming a bare value.
        aria-label={`Advanced themes. Palette ${current?.label ?? palette}, ${level?.label ?? contrast}`}
        title={`Advanced themes — ${current?.label ?? palette}, ${level?.label ?? contrast}`}
        onClick={() => (open ? close(true) : setOpen(true))}
      >
        <span className="palette-dot mini" data-p={palette} aria-hidden="true" />
        <span className="theme-more-arrow" aria-hidden="true">
          ▾
        </span>
      </button>

      {open && (
        <div className="theme-panel" id={panelId} ref={panel}>
          <p className="theme-panel-head">Advanced themes</p>

          <div
            className="palette-pick"
            role="radiogroup"
            aria-label={`Colour palette, currently ${current?.label ?? palette}`}
            onKeyDown={onSwatchKey}
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
                // Roving tabindex: the group is one tab stop and the arrows
                // move inside it, which is what a radiogroup promises.
                tabIndex={palette === p.key ? 0 : -1}
                className={`palette-dot${palette === p.key ? " on" : ""}`}
                onClick={() => setPalette(p.key)}
              />
            ))}
          </div>
          {/* The name of the hue you are on, spelled out. Eight discs with no
              text is a colour-only control, and the swatch cannot be read by
              somebody who cannot separate forest from ocean. */}
          <p className="theme-panel-now">{current?.label ?? palette}</p>

          <div
            className="theme-seg contrast-seg"
            role="radiogroup"
            aria-label={`Contrast level, currently ${level?.label ?? contrast}`}
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
            <span className="theme-panel-now inline">
              {level?.label ?? contrast}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
