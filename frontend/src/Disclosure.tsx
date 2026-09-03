// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

import { ReactNode, useEffect, useId, useState } from "react";

/** A sub-measurement, folded to one line until you want it.
 *
 *  THE COMPLAINT THIS ANSWERS: "advanced tools look way too complicated to
 *  use, because it looks congested." The attention panel had grown four
 *  sub-panels under the arcs — what a head is wired to do, what it does on
 *  your own corpus, what holds the answer, what the answer was sensitive to —
 *  and each one opens with a heading, a paragraph and a row of dials. Four of
 *  those stacked is a wall, and a wall reads as difficulty even when every
 *  individual control is simple.
 *
 *  Folding them is not hiding them. The heading stays, and so does a
 *  ONE-LINE statement of the question that block answers, because the thing a
 *  reader needs from a closed block is what it is FOR — the dials are only
 *  useful once they have decided to run it. Closed, the panel reads as a
 *  short index of questions; open, it is exactly what it was.
 *
 *  A RESULT OPENS IT AND KEEPS IT OPEN. A measurement that has arrived must
 *  never be behind a fold — that would be the panel hiding its own answer,
 *  which is worse than any amount of clutter. `hasResult` going true opens
 *  the block; a reader may still close it afterwards, and their choice
 *  sticks.
 *
 *  It is a real `<button>` with `aria-expanded` and `aria-controls`, not a
 *  clickable div, so it is reachable and announced. The chevron is
 *  `aria-hidden` because the button's state is already in `aria-expanded`,
 *  and reading both aloud says the same thing twice.
 */
export default function Disclosure({
  dot,
  title,
  asks,
  hasResult = false,
  disabled = false,
  children,
}: {
  /** The section-dot class, e.g. `d-attn`. The same hook `CategoryBar` and
   *  `SectionNav` read, so a folded block still declares what it belongs to. */
  dot: string;
  title: string;
  /** The question this block answers, in one line. Shown whether it is open
   *  or closed — closed, it is the only thing telling a reader what is in
   *  there. */
  asks: string;
  /** True once this block has a measurement on screen. */
  hasResult?: boolean;
  disabled?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const id = useId().replace(/:/g, "");

  // Opens ON the transition into having a result, rather than being pinned
  // open by it — so a reader who folds a finished block away keeps it folded.
  useEffect(() => {
    if (hasResult) setOpen(true);
  }, [hasResult]);

  return (
    <div className={`fold${open ? " open" : ""}`}>
      <button
        type="button"
        className="fold-head"
        aria-expanded={open}
        aria-controls={`fold-${id}`}
        onClick={() => setOpen((v) => !v)}
        disabled={disabled}
      >
        <span className={`dot ${dot}`} />
        <h3>{title}</h3>
        <span className="rule" />
        <span className="fold-mark" aria-hidden="true" />
      </button>
      <p className="meta fold-asks">{asks}</p>
      <div id={`fold-${id}`} className="fold-body" hidden={!open}>
        {children}
      </div>
    </div>
  );
}
