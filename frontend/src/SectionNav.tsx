import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Jump between the sections of a page that has grown past one screen.
 *
 * WHY THIS READS THE DOM INSTEAD OF CARRYING A LIST OF SECTIONS
 *
 * The obvious version of this component is an array of `{id, label}` written
 * by hand. It would be wrong here within a week, because the set of sections
 * on this page is not a constant — it is decided at runtime by five separate
 * conditions:
 *
 *   - VIEWER drops Custom, Robot policy and Agents (there is no machine
 *     behind a shared `.mri`, and three panels that can only say "install
 *     ModelMRI" are worse than three absent panels)
 *   - `epoch > 0` gates Attention, Features, Patching — nothing has been run
 *     yet, so there is nothing for them to be about
 *   - `introspectable` gates Patching separately: an Ollama model runs as
 *     text and has no residual stream to patch
 *   - `replay` swaps several panels for their recorded twins
 *   - SessionBar and GraphPanel return null unless a session is open and
 *     actually carries a graph
 *
 * A hand-written list would show entries that jump to nothing, and would
 * silently omit whatever panel is added next. Reading the page instead means
 * the nav is correct by construction: a section exists here exactly when it
 * exists on screen, and a panel added later appears in the rail without
 * anybody remembering to come back to this file.
 *
 * The colour is taken the same way. Every section already marks itself with a
 * `.dot.d-*` class that the stylesheet colours, so the rail reuses that class
 * verbatim rather than restating the mapping — the one place a new section's
 * colour is defined stays the one place.
 */

/** Label and description are one heading: "ATTENTION — WHERE EACH TOKEN LOOKED". */
const SEPARATOR = /\s+[—–]\s+/;

/**
 * How far down the viewport counts as "the section you are reading".
 *
 * Near the top, and measured rather than guessed. The first version put this
 * line at 30% of the viewport, which is the usual advice and is wrong for a
 * page of panels this size: jumping to Patching left the rail lit on Custom
 * Model, the section BELOW the one just jumped to. With Patching scrolled to
 * the top, Custom Model's top measured 283px against a line at 285px — it had
 * crossed the line by two pixels, so the "last section past the line" was the
 * wrong one, and every panel shorter than a third of the window had the same
 * defect.
 *
 * A line near the top means the section you jumped to stays lit until it has
 * actually left, which is also what survives the page shifting under you —
 * and it does shift: the hero's field animates, and the same jump was landing
 * ~69px off by the time it settled.
 */
function readingLine(): number {
  return Math.min(window.innerHeight * 0.18, 140);
}

/**
 * Whether to write the shortcut as ⌘K or Ctrl+K.
 *
 * `navigator.platform` is deprecated and lies under iPadOS, which reports
 * "MacIntel". `userAgentData.platform` is the replacement but exists only in
 * Chromium, so both are consulted and the answer defaults to Ctrl — the wrong
 * hint on a Mac is a shortcut that appears broken, and Ctrl is the safer
 * default because it is what the larger share of readers actually press.
 */
const APPLE = (() => {
  const ua = navigator as Navigator & {
    userAgentData?: { platform?: string };
  };
  const platform = ua.userAgentData?.platform ?? navigator.platform ?? "";
  return /mac|iphone|ipad|ipod/i.test(platform);
})();

type Section = {
  /** Anchor id, applied to the panel so the browser's own history works. */
  id: string;
  /** The `d-*` class, reused so the rail needs no colour table of its own. */
  tone: string;
  /** "ATTENTION" */
  label: string;
  /** "WHERE EACH TOKEN LOOKED", or "" for a heading with no dash. */
  detail: string;
  el: HTMLElement;
};

function slug(text: string): string {
  return (
    text
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "") || "section"
  );
}

/** Every section currently on the page, in document order. */
function scan(): Section[] {
  const found: Section[] = [];
  const used = new Set<string>();

  document.querySelectorAll<HTMLElement>(".sect").forEach((sect) => {
    const heading = sect.querySelector("h2");
    const text = heading?.textContent?.trim();
    if (!text) return;

    // The panel is what we scroll to, not the little header strip inside it,
    // so the whole card lands in view. SessionBar wraps its .sect in a
    // div.panel and GraphPanel in a section.panel; anything that grows a
    // .sect without a .panel around it still resolves to something sensible
    // rather than being dropped.
    const target =
      sect.closest<HTMLElement>(".panel") ?? sect.parentElement ?? sect;

    // A panel React has unmounted can still be in a stale NodeList, and a
    // display:none panel is not somewhere you can jump. Both have no box.
    if (target.getClientRects().length === 0) return;

    let tone = "";
    for (const cls of Array.from(sect.querySelector(".dot")?.classList ?? [])) {
      if (cls.startsWith("d-")) {
        tone = cls;
        break;
      }
    }

    const [head, ...rest] = text.split(SEPARATOR);
    const label = head.trim();

    // Two panels CAN legitimately be on screen at once under the same tone —
    // Features renders a second card in replay. Suffixing keeps the anchor a
    // real id rather than a duplicate, which is invalid HTML and makes
    // scrollIntoView pick whichever the browser saw first.
    const base = tone ? `sec-${tone.slice(2)}` : `sec-${slug(label)}`;
    let id = base;
    for (let n = 2; used.has(id); n++) id = `${base}-${n}`;
    used.add(id);
    if (target.id !== id) target.id = id;

    found.push({ id, tone, label, detail: rest.join(" — ").trim(), el: target });
  });

  return found;
}

/**
 * The section being read: the last one whose top has passed the reading line.
 *
 * Deliberately not IntersectionObserver. Attention is routinely taller than
 * the viewport — a 167-token strip is — and a panel taller than the viewport
 * never reaches a high intersection ratio, so a ratio-ranked observer picks a
 * short neighbour and the rail highlights the wrong thing exactly when the
 * long panel fills the screen. A position test has no such blind spot.
 */
function readingIndex(list: Section[]): number {
  if (list.length === 0) return -1;

  // At the bottom of the document the last section is what you are reading
  // even if its top never crossed the line — which is the normal case for a
  // short final panel, and would otherwise leave the rail stuck one above.
  const bottom = window.innerHeight + window.scrollY;
  const height = document.documentElement.scrollHeight;
  if (bottom >= height - 2) return list.length - 1;

  const line = readingLine();
  let index = 0;
  for (let i = 0; i < list.length; i++) {
    if (list[i].el.getBoundingClientRect().top <= line) index = i;
    else break;
  }
  return index;
}

export default function SectionNav() {
  const [sections, setSections] = useState<Section[]>([]);
  const [active, setActive] = useState(0);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);

  const sectionsRef = useRef<Section[]>([]);
  sectionsRef.current = sections;
  const inputRef = useRef<HTMLInputElement>(null);
  // Where focus was before the palette took it, so Escape puts it back
  // instead of dropping it on <body> and stranding a keyboard user at the
  // top of the document.
  const restoreTo = useRef<HTMLElement | null>(null);

  // ---- discovery -------------------------------------------------------

  useEffect(() => {
    let frame = 0;
    const resync = () => {
      cancelAnimationFrame(frame);
      // Panels mount in bursts — one generation can add three at once — and
      // a scan per mutation would be three full-document queries for one
      // visual change. Coalescing to the next frame makes it one.
      frame = requestAnimationFrame(() => {
        const next = scan();
        setSections((prev) => {
          const same =
            prev.length === next.length &&
            prev.every((p, i) => p.id === next[i].id && p.label === next[i].label);
          // Returning `prev` keeps React from re-rendering the rail on every
          // unrelated DOM change on the page, of which there are many: a
          // token strip animating is a mutation too.
          return same ? prev : next;
        });
        setActive(readingIndex(next));
      });
    };

    resync();
    const mo = new MutationObserver(resync);
    mo.observe(document.body, { childList: true, subtree: true });
    return () => {
      mo.disconnect();
      cancelAnimationFrame(frame);
    };
  }, []);

  // ---- which one is being read ----------------------------------------

  useEffect(() => {
    let frame = 0;
    const onScroll = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() =>
        setActive(readingIndex(sectionsRef.current)),
      );
    };
    // passive: this listener never calls preventDefault, and saying so is
    // what keeps scrolling off the main thread's critical path.
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      cancelAnimationFrame(frame);
    };
  }, []);

  // ---- jumping ---------------------------------------------------------

  const jump = useCallback((section: Section) => {
    // A reader who has asked their OS for less motion is asking about this
    // too: a 900px smooth scroll is the most motion on the page.
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    section.el.scrollIntoView({
      behavior: still ? "auto" : "smooth",
      block: "start",
    });
    // The URL becomes shareable and the back button works, without the jump
    // that setting location.hash would cause on top of the smooth scroll.
    history.replaceState(null, "", `#${section.id}`);
  }, []);

  // ---- palette ---------------------------------------------------------

  const matches = query.trim()
    ? sections.filter((s) =>
        `${s.label} ${s.detail}`.toLowerCase().includes(query.trim().toLowerCase()),
      )
    : sections;

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        restoreTo.current = document.activeElement as HTMLElement | null;
        setQuery("");
        setCursor(0);
        setOpen((was) => !was);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    if (open) inputRef.current?.focus();
    else restoreTo.current?.focus?.();
  }, [open]);

  function onPaletteKey(event: React.KeyboardEvent) {
    if (event.key === "Escape") {
      event.preventDefault();
      setOpen(false);
      return;
    }
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (matches.length === 0) return;
      const step = event.key === "ArrowDown" ? 1 : -1;
      setCursor((c) => (c + step + matches.length) % matches.length);
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      const pick = matches[cursor];
      if (pick) {
        jump(pick);
        setOpen(false);
      }
    }
  }

  // One destination is not navigation. Below two sections the rail is
  // furniture, and the shortcut would open a palette listing one entry.
  if (sections.length < 2) return null;

  return (
    <>
      <nav className="secnav" aria-label="Sections on this page">
        <ul>
          {sections.map((section, i) => (
            <li key={section.id}>
              <button
                type="button"
                className={`secnav-item${i === active ? " on" : ""}`}
                aria-current={i === active ? "true" : undefined}
                onClick={() => jump(section)}
                title={
                  section.detail
                    ? `${section.label} — ${section.detail}`
                    : section.label
                }
              >
                <i className={`secnav-dot dot ${section.tone}`} />
                <span className="secnav-label">{section.label}</span>
              </button>
            </li>
          ))}
        </ul>
        <button
          type="button"
          className="secnav-find"
          onClick={() => {
            restoreTo.current = document.activeElement as HTMLElement | null;
            setQuery("");
            setCursor(0);
            setOpen(true);
          }}
        >
          <span className="secnav-label">jump to…</span>
          {/* Written as the key this machine actually has. A Mac reader told
              to press Ctrl+K presses Ctrl+K and nothing happens. */}
          <kbd>{APPLE ? "⌘" : "Ctrl"}K</kbd>
        </button>
      </nav>

      {open && (
        <div
          className="palette-scrim"
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) setOpen(false);
          }}
        >
          <div
            className="palette"
            role="dialog"
            aria-modal="true"
            aria-label="Jump to a section"
          >
            <input
              ref={inputRef}
              className="palette-input"
              placeholder="jump to a section…"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setCursor(0);
              }}
              onKeyDown={onPaletteKey}
              aria-label="Filter sections"
            />
            <ul className="palette-list">
              {matches.map((section, i) => (
                <li key={section.id}>
                  <button
                    type="button"
                    className={`palette-row${i === cursor ? " on" : ""}`}
                    onMouseEnter={() => setCursor(i)}
                    onClick={() => {
                      jump(section);
                      setOpen(false);
                    }}
                  >
                    <i className={`secnav-dot dot ${section.tone}`} />
                    <span className="palette-label">{section.label}</span>
                    {section.detail && (
                      <span className="palette-detail">{section.detail}</span>
                    )}
                  </button>
                </li>
              ))}
              {matches.length === 0 && (
                <li className="palette-empty">
                  nothing on this page matches “{query.trim()}”
                </li>
              )}
            </ul>
          </div>
        </div>
      )}
    </>
  );
}
