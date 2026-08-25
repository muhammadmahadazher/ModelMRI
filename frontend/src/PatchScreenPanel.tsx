import { useState } from "react";
import { errorText, PatchScreen, patchScreen, ScreenSite } from "./api";
import { measured } from "./measured";

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
  const [verify, setVerify] = useState(6);
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
      <div className="sect sub">
        <span className="dot d-attn" />
        <h3>SCREEN THE GRID, THEN CHECK THE SCREEN</h3>
        <span className="rule" />
      </div>
      <p className="meta">
        One gradient pass ranks every site at once, then the shortlisted few
        are patched for real and compared against what the screen said. The
        saving is the point and the checking is what makes it worth anything —
        so both are measured, not claimed.
      </p>

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
        <label className="meta" htmlFor="ps-verify">
          verify
        </label>
        <input
          id="ps-verify"
          type="range"
          min={0}
          max={Math.max(2, shortlist)}
          value={Math.min(verify, shortlist)}
          onChange={(e) => setVerify(Number(e.target.value))}
          disabled={disabled || busy}
        />
        <span className="meta">{Math.min(verify, shortlist)} patched for real</span>
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
              <div className="row">
                <span className="pill">
                  {agree.verified} site
                  {agree.verified === 1 ? "" : "s"} patched for real
                </span>
                <span className="pill">
                  {agree.spearman === null
                    ? "no rank correlation"
                    : `Spearman ${measured(agree.spearman, 3)}`}
                </span>
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
              {agree.spearman === null && agree.spearman_reason && (
                <div className="meta">{agree.spearman_reason}</div>
              )}
              <div className="hint">{agree.means}</div>
            </div>
          )}

          <ul className="ps-sites">
            {data.shortlist.map((s) => (
              <Site key={s.name} s={s} />
            ))}
          </ul>
          {data.shortlist_capped_from !== null && (
            <p className="meta">
              {data.shortlist_size} of {data.shortlist_capped_from} requested —
              the cap is reported rather than silently applied.
            </p>
          )}

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
    </div>
  );
}
