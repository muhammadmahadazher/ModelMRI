import { useCallback, useEffect, useState } from "react";

/**
 * Filter the page by what you came to inspect.
 *
 * WHY THIS EXISTS
 *
 * The page is one long column of panels in the order they were built. Somebody
 * who came to look at a diffusion model scrolls past the SAE browser, the
 * agent timeline and the model-diff panel to reach it, and reads the whole
 * page as "a lot of things I do not want". That is the navigation half of the
 * maintainer's own finding: the tool is hard to move around in.
 *
 * So the panels are grouped by WHAT THEY INSPECT — text-to-text, text-to-image,
 * robot policy, agents — rather than by which one happened to be written next.
 *
 * WHY IT READS THE DOM RATHER THAN CARRYING A LIST
 *
 * The same reason `SectionNav` does, and the reason is worth repeating because
 * the obvious implementation is a hand-written array. Which panels exist is
 * decided at runtime by five separate conditions — VIEWER drops three, `epoch`
 * gates three more, `introspectable` gates patching, a replay swaps several
 * for their recorded twins, and two panels return null unless a session
 * carries the right section. A hand-written category list would offer a
 * "Text → Image" button on a page with no image panel, and would silently omit
 * whatever is added next.
 *
 * Reading the page means the bar is correct by construction: a category exists
 * exactly when a panel in it is on screen.
 *
 * WHY IT FILTERS INSTEAD OF ROUTING
 *
 * Deliberate, and chosen over sub-pages. One page means no router, no URL
 * scheme to keep working, no existing link to break, and — the part that
 * matters for a measurement tool — Ctrl-F still finds everything, and "share
 * this view" still means one thing. The categories can prove themselves before
 * anybody pays the price of routing.
 *
 * Nothing is unmounted. Hiding is `display: none`, so a panel that was
 * mid-measurement when you switched category is still mid-measurement when you
 * switch back — unmounting would silently cancel a forward pass somebody is
 * waiting on.
 */

/** A category, and the panels that belong to it.
 *
 *  Keyed by the section-dot class each panel already carries, which is the
 *  same trick `SectionNav` uses for colour: the one place a section declares
 *  what it is stays the one place. A panel added later with a known dot lands
 *  in the right category without anybody editing this file; one with an
 *  unknown dot lands in `other`, which is visible rather than lost. */
const CATEGORY_BY_DOT: Record<string, string> = {
  // The playground. It IS a text-to-text surface — you type a prompt and a
  // language model answers — so it filters with the rest of them rather than
  // being pinned visible. Pinning it would mean the robot view showed a text
  // prompt box that does nothing for the panel underneath it.
  "d-run": "text",
  "d-mdiff": "text",
  "d-attn": "text",
  "d-feat": "text",
  "d-lens": "text",
  "d-patch": "text",
  "d-scope": "text",
  "d-probe": "text",
  "d-ground": "text",
  "d-base": "text",
  "d-custom": "text",
  "d-image": "image",
  "d-vla": "robot",
  "d-agent": "agents",
};

const LABELS: Record<string, string> = {
  all: "All",
  text: "Text → Text",
  image: "Text → Image",
  robot: "Robot policy",
  agents: "Agents",
  other: "Other",
};

/** The order they appear in, when present. Categories not listed here sort
 *  after these, alphabetically — so a new one shows up rather than vanishing
 *  because nobody added it to an order array. */
const ORDER = ["all", "text", "image", "robot", "agents", "other"];

type Group = { key: string; count: number };

export default function CategoryBar() {
  const [groups, setGroups] = useState<Group[]>([]);
  const [active, setActive] = useState("all");

  /** Read the page and work out which categories are actually on it. */
  const survey = useCallback(() => {
    const panels = Array.from(document.querySelectorAll<HTMLElement>(".panel"));
    const counts = new Map<string, number>();
    for (const panel of panels) {
      const dot = panel.querySelector(".dot");
      const key = dot
        ? (Array.from(dot.classList).find((c) => c.startsWith("d-")) ?? "")
        : "";
      const category = CATEGORY_BY_DOT[key] ?? "other";
      counts.set(category, (counts.get(category) ?? 0) + 1);
      // Stamped on the panel so the CSS can hide it without React re-rendering
      // the whole tree — and so a human reading the DOM can see the grouping.
      panel.dataset.mriCategory = category;
    }
    const found: Group[] = [...counts.entries()]
      .map(([key, count]) => ({ key, count }))
      .sort((a, b) => {
        const ai = ORDER.indexOf(a.key);
        const bi = ORDER.indexOf(b.key);
        if (ai === -1 && bi === -1) return a.key.localeCompare(b.key);
        if (ai === -1) return 1;
        if (bi === -1) return -1;
        return ai - bi;
      });
    setGroups(found);
  }, []);

  useEffect(() => {
    survey();
    // The panel set changes as models load, sessions open and runs finish, so
    // the bar has to re-survey rather than read the page once on mount. An
    // observer rather than a poll: this fires when the tree actually changes,
    // and costs nothing when it does not.
    const observer = new MutationObserver(() => survey());
    observer.observe(document.body, { childList: true, subtree: true });
    return () => observer.disconnect();
  }, [survey]);

  /** Apply the filter by attribute, not by unmounting. */
  useEffect(() => {
    for (const panel of document.querySelectorAll<HTMLElement>(".panel")) {
      const category = panel.dataset.mriCategory ?? "other";
      panel.dataset.mriHidden = active !== "all" && category !== active ? "1" : "";
    }
  }, [active, groups]);

  // One category is not a choice. A bar offering only "All" is furniture, and
  // this page has enough of that already.
  if (groups.length < 2) return null;

  const total = groups.reduce((n, g) => n + g.count, 0);

  return (
    <nav className="cat-bar" aria-label="Filter panels by what they inspect">
      <button
        className={`cat-chip${active === "all" ? " on" : ""}`}
        onClick={() => setActive("all")}
        aria-pressed={active === "all"}
      >
        {LABELS.all}
        <span className="cat-count">{total}</span>
      </button>
      {groups.map((g) => (
        <button
          key={g.key}
          className={`cat-chip${active === g.key ? " on" : ""}`}
          onClick={() => setActive(g.key)}
          aria-pressed={active === g.key}
        >
          {LABELS[g.key] ?? g.key}
          <span className="cat-count">{g.count}</span>
        </button>
      ))}
      {/* The count is not decoration. "Text → Image 0" would be a button that
          does nothing, so a category with no panels is never rendered at all —
          and the number tells you how much is behind a chip before you press
          it, which is the difference between a filter and a guess. */}
    </nav>
  );
}
