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
 * THE SECOND ROW
 *
 * Grouping by modality left one category holding eleven panels — Text → Text
 * is most of the tool — so the same pile the bar was built to fix reappeared
 * one level down. A second row of chips subdivides the CHOSEN category by what
 * its panels do: run, attention, concepts, causal, grounding, compare.
 *
 * It is the primary/secondary relationship Material describes and not two
 * independent filters: the second row belongs to the first row's selection,
 * changes when it changes, and resets to "All" with it. It is surveyed the
 * same way, from the same dot classes, so it is correct by construction for
 * the same reason — and it is absent, rather than empty, wherever the chosen
 * category has fewer than two sub-groups to offer.
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
  // Saved sweeps. A sweep is the same text-to-text measurement run over many
  // prompts, so it belongs with the panels that make those measurements
  // rather than in `other`.
  "d-sweep": "text",
  "d-image": "image",
  "d-vision": "vision",
  "d-vla": "robot",
  "d-agent": "agents",
};

/** Sub-groups WITHIN a category, keyed by the same dot classes.
 *
 *  "Text → Text" alone holds eleven panels, which is the pile the row above
 *  was built to fix, one level down: eleven is not a choice either. So the
 *  chosen category subdivides by what the panel DOES to the model — run it,
 *  read its attention, name a concept, cut a path, check it against a source,
 *  hold it against another model.
 *
 *  Keyed by dot for the same reason the categories are: the one place a
 *  section declares what it is stays the one place, so a panel added later
 *  with a known dot lands in the right sub-group with nobody editing this
 *  file. A dot that is not here lands in `rest`, which is rendered and
 *  labelled "More" — visible rather than lost, exactly like `other` above.
 *
 *  `d-lens` and `d-base` have no panel on the page today and are listed
 *  anyway, for the same reason they are listed in the category map: the cost
 *  of a key that matches nothing is zero (nothing is counted, so no chip is
 *  drawn), and the cost of a missing one is a panel in the wrong group. */
const SUB_BY_DOT: Record<string, string> = {
  "d-run": "run",
  "d-base": "run",
  "d-custom": "run",
  // "Run" rather than "compare": what the panel lists is runs and what they
  // cost to finish. The comparison a sweep enables happens inside its own
  // output file, not here.
  "d-sweep": "run",
  "d-attn": "attn",
  "d-feat": "concepts",
  "d-lens": "concepts",
  "d-probe": "concepts",
  "d-patch": "causal",
  "d-scope": "causal",
  "d-ground": "ground",
  "d-mdiff": "compare",
};

const LABELS: Record<string, string> = {
  all: "All",
  text: "Text → Text",
  image: "Text → Image",
  vision: "Vision",
  robot: "Robot policy",
  agents: "Agents",
  other: "Other",
};

const SUB_LABELS: Record<string, string> = {
  all: "All",
  run: "Run",
  attn: "Attention",
  concepts: "Concepts",
  causal: "Causal",
  ground: "Grounding",
  compare: "Compare",
  // Deliberately not "Other". That word is a CATEGORY on the row above, and
  // the two rows are on screen together — the same label at two levels reads
  // as the same filter twice.
  rest: "More",
};

/** The order they appear in, when present. Categories not listed here sort
 *  after these, alphabetically — so a new one shows up rather than vanishing
 *  because nobody added it to an order array. */
const ORDER = ["all", "text", "image", "vision", "robot", "agents", "other"];

/** Roughly the order you would actually work in: run something, look at where
 *  it looked, name what it was thinking, cut the path, check it against the
 *  world, then hold it against another model. `rest` last, always. */
const SUB_ORDER = ["run", "attn", "concepts", "causal", "ground", "compare", "rest"];

type Group = { key: string; count: number };

/** Sort by a fixed order, with anything unlisted after it alphabetically. */
function byOrder(order: string[]) {
  return (a: Group, b: Group) => {
    const ai = order.indexOf(a.key);
    const bi = order.indexOf(b.key);
    if (ai === -1 && bi === -1) return a.key.localeCompare(b.key);
    if (ai === -1) return 1;
    if (bi === -1) return -1;
    return ai - bi;
  };
}

/** Counted entries of a Map, in the order given. */
function ranked(counts: Map<string, number>, order: string[]): Group[] {
  return [...counts.entries()]
    .map(([key, count]) => ({ key, count }))
    .sort(byOrder(order));
}

/**
 * Arrow keys move focus along a row; Home and End go to its ends.
 *
 * Roving focus, so the whole bar is ONE tab stop rather than seven — a
 * keyboard reader on a page with this many controls should not have to press
 * Tab six times to get past a filter they are not using. The chip that is on
 * is the one Tab reaches, which is the convention and also the useful one:
 * the thing you land on is the state you are in.
 *
 * Focus moves, selection does not follow it. Arrowing along a row that
 * re-filtered the page under you at every step would run the filter six times
 * to reach the seventh chip; Enter and Space already choose, for free, because
 * these are real buttons.
 *
 * Wrap-around and `preventDefault` match the palette in `SectionNav`, which is
 * the other keyboard surface on this page — two list-walking behaviours that
 * disagree is worse than either.
 */
function onRowKey(event: React.KeyboardEvent<HTMLElement>) {
  if (
    event.key !== "ArrowLeft" &&
    event.key !== "ArrowRight" &&
    event.key !== "Home" &&
    event.key !== "End"
  ) {
    return;
  }
  const chips = Array.from(
    event.currentTarget.querySelectorAll<HTMLButtonElement>("button"),
  );
  const here = chips.indexOf(event.target as HTMLButtonElement);
  if (here === -1 || chips.length === 0) return;
  event.preventDefault();
  const next =
    event.key === "ArrowLeft"
      ? (here - 1 + chips.length) % chips.length
      : event.key === "ArrowRight"
        ? (here + 1) % chips.length
        : event.key === "Home"
          ? 0
          : chips.length - 1;
  chips[next].focus();
}

export default function CategoryBar() {
  const [groups, setGroups] = useState<Group[]>([]);
  /** Sub-groups per category. Keyed by category, so switching the row above
   *  changes the row below without re-reading the page. There is deliberately
   *  no "all" entry — see the render for why the second row is absent there. */
  const [subs, setSubs] = useState<Record<string, Group[]>>({});
  const [active, setActive] = useState("all");
  const [activeSub, setActiveSub] = useState("all");

  /** Read the page and work out which categories are actually on it. */
  const survey = useCallback(() => {
    const panels = Array.from(document.querySelectorAll<HTMLElement>(".panel"));
    const counts = new Map<string, number>();
    // Counted per category rather than globally: "Attention 1" under Text →
    // Text has to mean one TEXT panel, and a sub-group's number is only the
    // truth about the category you are standing in.
    const subCounts = new Map<string, Map<string, number>>();
    for (const panel of panels) {
      const dot = panel.querySelector(".dot");
      const key = dot
        ? (Array.from(dot.classList).find((c) => c.startsWith("d-")) ?? "")
        : "";
      const category = CATEGORY_BY_DOT[key] ?? "other";
      const sub = SUB_BY_DOT[key] ?? "rest";
      counts.set(category, (counts.get(category) ?? 0) + 1);
      let bucket = subCounts.get(category);
      if (!bucket) subCounts.set(category, (bucket = new Map()));
      bucket.set(sub, (bucket.get(sub) ?? 0) + 1);
      // Stamped on the panel so the CSS can hide it without React re-rendering
      // the whole tree — and so a human reading the DOM can see the grouping.
      panel.dataset.mriCategory = category;
      panel.dataset.mriSub = sub;
    }
    setGroups(ranked(counts, ORDER));
    const bySub: Record<string, Group[]> = {};
    for (const [category, tally] of subCounts) {
      bySub[category] = ranked(tally, SUB_ORDER);
    }
    setSubs(bySub);
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

  /** Apply the filter by attribute, not by unmounting.
   *
   *  AND MARK WHERE EACH GROUP STARTS. The page was fourteen identical peers
   *  the same 24px apart, so nothing on it said the image panels are a
   *  different modality from the attention panels — only these chips did, and
   *  they scroll away. Proximity is the cheapest grouping signal there is
   *  (https://m3.material.io/foundations/layout/understanding-layout/spacing)
   *  and it was carrying zero bits.
   *
   *  Computed over the VISIBLE panels, in the same pass that decides which
   *  those are: filter to Robot and the first robot panel becomes the first
   *  panel, so a marker stamped on the unfiltered order would leave a group
   *  heading floating above nothing.
   */
  useEffect(() => {
    let run = "";
    for (const panel of document.querySelectorAll<HTMLElement>(".panel")) {
      const category = panel.dataset.mriCategory ?? "other";
      const sub = panel.dataset.mriSub ?? "rest";
      const hidden =
        (active !== "all" && category !== active) ||
        (activeSub !== "all" && sub !== activeSub);
      panel.dataset.mriHidden = hidden ? "1" : "";
      if (hidden) continue;
      if (category === run) {
        panel.dataset.mriGroupStart = "";
        delete panel.dataset.mriGroupLabel;
        continue;
      }
      run = category;
      // The FIRST group needs no extra air above it — there is nothing above
      // it to be separated from — but it still gets its label.
      panel.dataset.mriGroupStart = panel.previousElementSibling?.classList.contains(
        "panel",
      )
        ? "1"
        : "first";
      panel.dataset.mriGroupLabel = LABELS[category] ?? category;
    }
  }, [active, activeSub, groups, subs]);

  /**
   * A selection can stop existing under you.
   *
   * The panel set is not stable — unload drops the custom-model panel, a reset
   * takes `epoch` back to zero and three panels with it, opening a replay swaps
   * several for their recorded twins. A filter still pointing at what left is
   * an empty page with a lit chip that has nothing behind it, which is the one
   * failure the counts were supposed to make impossible.
   *
   * So the selection is checked against what the survey just found, not
   * against what was true when it was clicked.
   */
  useEffect(() => {
    if (active !== "all" && !groups.some((g) => g.key === active)) {
      setActive("all");
      setActiveSub("all");
      return;
    }
    if (activeSub !== "all" && !(subs[active] ?? []).some((s) => s.key === activeSub)) {
      setActiveSub("all");
    }
  }, [groups, subs, active, activeSub]);

  // One category is not a choice. A bar offering only "All" is furniture, and
  // this page has enough of that already.
  if (groups.length < 2) return null;

  const total = groups.reduce((n, g) => n + g.count, 0);

  /**
   * The second row: the sub-groups of the chosen category, and only that one.
   *
   * `subs` has no "all" key by construction, so "All" carries no second row —
   * and that is the intended answer, not a gap. The sub-groups of EVERYTHING
   * are the categories themselves, which is the row above; and the fallback
   * chip under "All" would hold every image, robot and agent panel at once, a
   * filter meaning "not text" with no honest name to put on it.
   *
   * Below two sub-groups there is no row either — the same judgement the
   * category row makes for itself twenty lines up. Text → Image, Robot policy
   * and Agents each have too few panels to subdivide today, so they get
   * nothing rather than a fabricated hierarchy; when one of them grows a
   * second kind of panel, the row appears on its own.
   */
  const within = subs[active] ?? [];
  const showSubs = within.length >= 2;
  const subTotal = within.reduce((n, g) => n + g.count, 0);
  // "Text → Text" is read aloud as "text right-arrow text". The label on
  // screen keeps its arrow; the one that is spoken says the word.
  const spoken = (LABELS[active] ?? active).replace("→", "to");

  return (
    <div className="cat-nav">
      <nav
        className="cat-bar"
        aria-label="Filter panels by what they inspect"
        onKeyDown={onRowKey}
      >
        <button
          type="button"
          className={`cat-chip${active === "all" ? " on" : ""}`}
          onClick={() => {
            setActive("all");
            setActiveSub("all");
          }}
          aria-pressed={active === "all"}
          tabIndex={active === "all" ? 0 : -1}
        >
          {LABELS.all}
          <span className="cat-count">{total}</span>
        </button>
        {groups.map((g) => (
          <button
            key={g.key}
            type="button"
            className={`cat-chip${active === g.key ? " on" : ""}`}
            // Switching category resets the row below it. Carrying "Attention"
            // across to Robot policy would filter to nothing, and carrying it
            // back to Text → Text later would answer a question nobody asked
            // twice — the second row belongs to the selection above it.
            onClick={() => {
              setActive(g.key);
              setActiveSub("all");
            }}
            aria-pressed={active === g.key}
            tabIndex={active === g.key ? 0 : -1}
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

      {/* Always in the tree, even closed: the slot is what the row grows
          inside, so opening interpolates a height instead of teleporting the
          page down by one row. See the stylesheet for why only the OPENING
          transitions. */}
      <div className="cat-subs-slot" data-open={showSubs ? "1" : ""}>
        <div className="cat-subs-clip">
          {showSubs && (
            <nav
              className="cat-subs"
              // Names the category it belongs to, because out of context
              // "Filter panels by kind" is the third nav on this page with a
              // label that could describe any of them.
              aria-label={`Filter ${spoken} panels by kind`}
              onKeyDown={onRowKey}
            >
              <button
                type="button"
                className={`cat-chip cat-sub${activeSub === "all" ? " on" : ""}`}
                onClick={() => setActiveSub("all")}
                aria-pressed={activeSub === "all"}
                tabIndex={activeSub === "all" ? 0 : -1}
              >
                {SUB_LABELS.all}
                <span className="cat-count">{subTotal}</span>
              </button>
              {within.map((s) => (
                <button
                  key={s.key}
                  type="button"
                  className={`cat-chip cat-sub${activeSub === s.key ? " on" : ""}`}
                  onClick={() => setActiveSub(s.key)}
                  aria-pressed={activeSub === s.key}
                  tabIndex={activeSub === s.key ? 0 : -1}
                >
                  {SUB_LABELS[s.key] ?? s.key}
                  <span className="cat-count">{s.count}</span>
                </button>
              ))}
            </nav>
          )}
        </div>
      </div>
    </div>
  );
}
