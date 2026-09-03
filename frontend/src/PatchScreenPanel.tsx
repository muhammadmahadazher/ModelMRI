// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

import { useState } from "react";
import { errorText, PatchScreen, patchScreen, ScreenSite } from "./api";
import { measured } from "./measured";
import Disclosure from "./Disclosure";

/** The route's own floor (`patch_screen.MIN_VERIFY`). One site verified
 *  is not an agreement between two rankings, it is a coincidence, so the
 *  route refuses below this — and a dial that offers it offers a control
 *  that can only fail. */
const MIN_VERIFY = 2;

/** Rank the patching grid in two passes instead of hundreds — and measure the
 *  screen against the exact grid on the few it shortlists.
 *
 *  A SCREEN WHOSE AGREEMENT WAS NEVER MEASURED IS A GUESS WITH A LEADERBOARD.
 *  That is the whole reason this is a separate control rather than a faster
 *  mode of the patching panel: the saving is real, the ranking is
 *  approximate, and the only thing that makes the approximation worth
 *  anything is checking it — so the verification is not optional here and its
 *  result is rendered above the ranking rather than under it.
 *
 *  The numbers are structurally distinguishable from `/api/patch`'s on
 *  purpose: `attribution` rather than `recovery`, and an `approximate` flag
 *  in the payload. A reader who copies a number out of this panel and into a
 *  sentence about what patching found should be able to tell which they have.
 */

function Site({ s }: { s: ScreenSite }) {
  return (
    <li className="ps-site">
      <code>{s.name}</code>
      <span className="meta">
        L{s.layer} · pos {s.position}
      </span>
      <b>{measured(s.attribution, 4)}</b>
      {/* `null` is "the exact grid never ran here", which is a different
          thing from "it ran and recovered nothing". */}
      <span className="meta">
        {s.exact_recovery === null ? (
          "not verified"
        ) : (
          <>
            exact {measured(s.exact_recovery, 4)}
            {s.exact_error !== null && <> ± {measured(s.exact_error, 4)}</>}
          </>
        )}
      </span>
    </li>
  );
}

export default function PatchScreenPanel({ disabled }: { disabled?: boolean }) {
  const [clean, setClean] = useState("The Eiffel Tower is in the city of");
  const [corrupt, setCorrupt] = useState("The Colosseum is in the city of");
  const [shortlist, setShortlist] = useState(12);
  const [verifyWanted, setVerifyWanted] = useState(6);
  // What will actually be sent. Dragging `shortlist` below `verify` used to
  // leave the two disagreeing until the server said so.
  const verify = Math.min(verifyWanted, Math.max(MIN_VERIFY, shortlist));
  const setVerify = setVerifyWanted;
  const [data, setData] = useState<PatchScreen | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  async function run() {
    if (busy || !clean.trim() || !corrupt.trim()) return;
    setBusy(true);
    setErr("");
    try {
      setData(await patchScreen({ clean, corrupt, shortlist, verify }));
    } catch (e) {
      setData(null);
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  const agree = data?.agreement;

  return (
    <div className="patch-screen">
      <Disclosure
        dot="d-attn"
        title="SCREEN THE GRID, THEN CHECK THE SCREEN"
        asks="Rank every patching site in two passes instead of hundreds — and measure the ranking against the exact grid on the few it shortlists."
        hasResult={data !== null}
      >

      <div className="ps-prompts">
        <label className="meta" htmlFor="ps-clean">
          clean
        </label>
        <input
          id="ps-clean"
          className="share-note"
          value={clean}
          onChange={(e) => setClean(e.target.value)}
          spellCheck={false}
          disabled={disabled || busy}
        />
        <label className="meta" htmlFor="ps-corrupt">
          corrupt
        </label>
        <input
          id="ps-corrupt"
          className="share-note"
          value={corrupt}
          onChange={(e) => setCorrupt(e.target.value)}
          spellCheck={false}
          disabled={disabled || busy}
        />
      </div>
      <div className="row">
        <label className="meta" htmlFor="ps-shortlist">
          shortlist
        </label>
        <input
          id="ps-shortlist"
          type="range"
          min={2}
          max={40}
          value={shortlist}
          onChange={(e) => setShortlist(Number(e.target.value))}
          disabled={disabled || busy}
        />
        <span className="meta">{shortlist}</span>
        {/* MIN 2, NOT 0. The route refuses anything under two — one site
            verified is not an agreement, it is a coincidence — so a dial
            offering 0 and 1 offered two positions that can only fail.

            And the clamp used to be on the DISPLAY only: the slider showed
            `min(verify, shortlist)` while the request carried the raw value,
            so dragging shortlist below verify produced a 422 naming a number
            the control had never shown. It is clamped in state now. */}
        <label className="meta" htmlFor="ps-verify">
          verify
        </label>
        <input
          id="ps-verify"
          type="range"
          min={MIN_VERIFY}
          max={Math.max(MIN_VERIFY, shortlist)}
          value={verify}
          onChange={(e) => setVerify(Number(e.target.value))}
          disabled={disabled || busy}
        />
        <span className="meta">{verify} patched for real</span>
        <button
          className="ghost sm"
          onClick={() => void run()}
          disabled={disabled || busy || !clean.trim() || !corrupt.trim()}
        >
          {busy ? "screening…" : data ? "screen again" : "screen the grid"}
        </button>
      </div>

      {err && <div className="hint err">{err}</div>}

      {data && (
        <>
          <div className="row ps-chips">
            <span className="pill">
              <code>{data.clean.answer.text}</code> vs{" "}
              <code>{data.corrupt.answer.text}</code>
            </span>
            <span className="meta">
              {data.n_sites_scored} sites over {data.n_layers} layers ×{" "}
              {data.n_positions} positions
            </span>
            {/* Not a footnote. These are not `/api/patch`'s numbers. */}
            {data.approximate && (
              <span className="meta warn">
                first-order approximation — not the exact recovery the patching
                panel reports
              </span>
            )}
            {data.n_sites_nonfinite > 0 && (
              <span className="meta warn">
                {data.n_sites_nonfinite} site(s) scored non-finite and were left
                out rather than ranked as zero
              </span>
            )}
          </div>

          {/* THE AGREEMENT, ABOVE THE RANKING IT QUALIFIES. */}
          {agree && (
            <div className="ps-agreement">
              {/* THE ANSWER, at answer size. The whole panel asks one
                  question — does the cheap screen rank the sites the way the
                  exact grid does — and this coefficient is the reply. It sat
                  in an 11.5px pill between two others, on a page where the
                  largest text in any panel is the 15px body, so the reply
                  looked exactly like the caveats qualifying it.

                  The null arm is `.unmeasured` and carries the route's own
                  `spearman_reason`: an unmeasurable correlation printed as a
                  number would be the loudest thing here and false. */}
              <p
                className={`answer${agree.spearman === null ? " unmeasured" : ""}`}
              >
                <span className="answer-n">
                  {agree.spearman === null
                    ? "no rank correlation"
                    : measured(agree.spearman, 3)}
                </span>
                <span className="answer-of">
                  {agree.spearman === null
                    ? (agree.spearman_reason ??
                      `over the ${agree.verified} site${agree.verified === 1 ? "" : "s"} patched for real`)
                    : `Spearman, screen against exact grid, over the ${agree.verified} site${agree.verified === 1 ? "" : "s"} patched for real`}
                </span>
              </p>

              <div className="row">
                {agree.sign_flips > 0 && (
                  <span className="meta warn">
                    {agree.sign_flips} sign flip
                    {agree.sign_flips === 1 ? "" : "s"} — the screen and the
                    exact grid disagreed about the DIRECTION
                  </span>
                )}
                {agree.near_zero_probed > 0 && (
                  <span className="meta">
                    {agree.near_zero_probed} near-zero site
                    {agree.near_zero_probed === 1 ? "" : "s"} probed too — a
                    screen checked only where it already agrees has not been
                    checked
                  </span>
                )}
              </div>
              <div className="hint">{agree.means}</div>
            </div>
          )}

          <ul className="ps-sites">
            {data.shortlist.map((s) => (
              <Site key={s.name} s={s} />
            ))}
          </ul>
          {/* `shortlist_capped_from` is how many sites were SCORED, not how
              many were asked for — the route sets it to `len(rows)`. Read as
              the request it said "12 of 486 requested" on a dial set to 12,
              which is a cap reported as the opposite of what happened. Both
              numbers now, each named. */}
          <p className="meta">
            {data.shortlist_size} shortlisted of{" "}
            {data.shortlist_capped_from.toLocaleString()} sites scored
            {data.shortlist_requested !== data.shortlist_size && (
              <> · {data.shortlist_requested} requested</>
            )}{" "}
            — the cap is reported rather than silently applied.
          </p>

          {/* The strongest NEGATIVE site, which a top-N list sorted by
              magnitude can drop entirely. A site that pushes the answer the
              other way is a finding, not a small positive one. */}
          {data.strongest_negative && !data.strongest_negative_on_shortlist && (
            <div className="ps-negative">
              <span className="meta">
                strongest site pushing the OTHER way, off the shortlist:
              </span>
              <ul className="ps-sites">
                <Site s={data.strongest_negative} />
              </ul>
              {data.strongest_negative_reason && (
                <span className="meta">{data.strongest_negative_reason}</span>
              )}
            </div>
          )}

          <div className="meta ps-cost">
            {data.cost.screen_forward_passes} forward ·{" "}
            {data.cost.screen_backward_passes} backward ·{" "}
            {data.cost.verification_passes} verification passes, against{" "}
            {data.cost.exact_grid_passes.toLocaleString()} for the exact grid —{" "}
            {data.cost.passes_saved_against_exact_grid.toLocaleString()} saved.
            {data.cost.seconds !== null && (
              <> {measured(data.cost.seconds, 2)} s.</>
            )}
          </div>
          <div className="hint">{data.cost.means}</div>
          {data.notes.map((n, i) => (
            <div className="meta" key={i}>
              {n}
            </div>
          ))}
          <div className="hint">{data.means}</div>
        </>
      )}
    </Disclosure>
    </div>
  );
}
