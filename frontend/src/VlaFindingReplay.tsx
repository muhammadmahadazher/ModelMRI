// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

import { useEffect, useState } from "react";
import { errorText, getVlaReplay, VlaBlock, VlaFindingSection } from "./api";
import { counted, measured, signed } from "./measured";

/** A robot finding somebody sent you, opened with nothing installed.
 *
 *  `/api/vla/share` has written this section since the robot work landed.
 *  `session._vla` has validated it to `_patch`'s standard the whole time, and
 *  `mcp_server` has advertised it in its `has` dict. Nothing served it back.
 *  So the Share button in `VLACausal` produced a file whose recipient opened
 *  an empty text session — "1 tokens · 0 attention maps" — with no mention of
 *  the policy, the frame, or the map. A writer does not imply a reader, and
 *  this is the third section in this project to prove it: the agent trace and
 *  the image run were the first two.
 *
 *  WHAT THIS PANEL REFUSES TO LET A READER CONCLUDE:
 *
 *    attention is not cause     the policy attending to a patch and the
 *                               action MOVING when that patch is covered are
 *                               two different measurements, and they
 *                               routinely disagree. The occlusion is the
 *                               causal one; the agreement between them is
 *                               printed as a number, negative and all.
 *    a shift is not a finding   until it beats a random occlusion of the same
 *                               size. `clears_control` is the test, and its
 *                               `null` means "nobody controlled this block" —
 *                               which is not "it failed".
 *    a map is not a picture     the map is drawn over ONE frame of ONE
 *                               episode at ONE timestep from ONE camera, and
 *                               a frame shrunk on the way in puts every block
 *                               in the wrong place.
 *
 *  It renders NOTHING when there is no robot finding, which is most sessions.
 */
/** Whether a block's control verdict has its numbers behind it.
 *
 *  `clears_control` and the two control numbers are read INDEPENDENTLY by the
 *  session reader: a null `control_max` survives on purpose, because an
 *  uncontrolled block genuinely has none. So a file can arrive carrying a
 *  verdict with nothing behind it, and `?? 0` rendered that as "0.0000 the
 *  best of 0 random occlusions managed" — a measured-looking zero over an
 *  absence, which is the one substitution this panel exists to prevent.
 *
 *  A type predicate rather than a boolean, so the branches it guards can
 *  print the numbers without asking the compiler to take `?? 0` for an
 *  answer.
 */
function controlRead(
  b: VlaBlock,
): b is VlaBlock & { control_max: number; control_draws: number } {
  return (
    b.clears_control !== null &&
    typeof b.control_max === "number" &&
    Number.isFinite(b.control_max) &&
    typeof b.control_draws === "number"
  );
}

export default function VlaFindingReplay({
  /** Bumped whenever the page resets. */
  epoch,
  /** WHICH `.mri` is open — the `created_at`, not a boolean. Opening a second
   *  file without closing the first leaves `open` true the whole way through,
   *  so a boolean would leave the previous finding on screen under the new
   *  file's name. */
  sessionKey,
}: {
  epoch: number;
  sessionKey: string;
}) {
  const [run, setRun] = useState<VlaFindingSection | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    let live = true;
    setErr("");
    // Cleared first: a panel showing nothing while it loads is honest; one
    // showing the previous file's frame under this file's name is not.
    setRun(null);
    void getVlaReplay()
      .then((got) => {
        if (!live) return;
        setRun(got.available ? (got as VlaFindingSection) : null);
      })
      .catch((e) => {
        if (!live) return;
        // NOT swallowed into `null`. A section that failed to load is not a
        // session without one, and rendering nothing for both would hide a
        // file the recipient cannot open behind a page that looks fine.
        setRun(null);
        setErr(errorText(e));
      });
    return () => {
      live = false;
    };
  }, [epoch, sessionKey]);

  if (err) {
    return (
      <div className="panel">
        <div className="hint err">{err}</div>
      </div>
    );
  }
  if (!run) return null;

  // Every shape is checked. Only one of the two paths here is validated: the
  // server serves what `session._vla` accepted, and the VIEWER build serves
  // the section straight out of the file with only the provenance checked.
  const list = <T,>(v: T[] | undefined | null): T[] => (Array.isArray(v) ? v : []);
  const occ = run.occlusion;
  const blocks = list(occ?.blocks).filter(
    (b) => b && typeof b.shift === "number" && Number.isFinite(b.shift),
  );
  const sized =
    Array.isArray(run.frame_size) && run.frame_size.length === 2
      ? run.frame_size
      : null;
  // A `.mri` NEVER FETCHES. `session._vla` refuses a frame that is not a data
  // URL in as many words; the viewer shim does not, so a hostile file could
  // carry a link and announce the recipient opening it.
  const frame =
    typeof run.frame === "string" && run.frame.startsWith("data:image/")
      ? run.frame
      : null;
  const linked = run.frame && !frame;

  // THE ANSWER: the block that moved the action most, and whether it beat its
  // own control. A shift with no control behind it is a number, not a finding.
  const strongest = blocks.reduce<VlaBlock | null>(
    (best, b) => (best === null || b.shift > best.shift ? b : best),
    null,
  );
  // The verdict AND its numbers. Counting a block as controlled on the
  // strength of a bare `clears_control` inflates this against occlusions
  // nothing was ever measured beside.
  const controlled = blocks.filter(controlRead);
  const cleared = blocks.filter((b) => b.clears_control === true);
  // ONE reading of the control, shared by the badge and the label below.
  // Three parallel ternaries over the same three-state field is how they
  // drift apart -- and "a verdict with no numbers behind it" is a fourth
  // state that only appears if something names it.
  const verdict =
    strongest === null || !controlRead(strongest)
      ? strongest !== null && strongest.clears_control !== null
        ? "no-numbers"
        : "uncontrolled"
      : strongest.clears_control === true
        ? "cleared"
        : "not-cleared";

  const agreement = occ?.attention_agreement;
  const where = (b: VlaBlock) =>
    typeof b.row === "number" && typeof b.col === "number"
      ? `row ${b.row}, col ${b.col}`
      : "an unplaced patch";

  return (
    <div className="panel" data-mri-group-label="robot">
      <div className="row">
        <span className="dot d-vla" />
        <h2>ROBOT FINDING — SHARED, NOT RE-RUN</h2>
      </div>

      {/* THE PROVENANCE FIRST, and for this section it is five things rather
          than one. Every other section here describes the model the file
          names; this one describes a policy AND a dataset AND one frame of
          one episode of it. `session._vla` refuses the section without all
          five, in its own words, because a heat map without them is a picture
          of nothing in particular. */}
      <p className="meta">
        <b>{run.provenance.policy}</b> on {run.provenance.dataset} · episode{" "}
        {run.provenance.episode} · timestep {run.provenance.timestep} ·{" "}
        {run.provenance.camera}
        {" · "}
        {run.provenance.revision ? (
          <>
            revision <code>{run.provenance.revision}</code>
          </>
        ) : (
          <>this policy published no revision</>
        )}
      </p>

      {/* THE ANSWER SLOT. Not the biggest shift — the biggest shift that beat
          a random occlusion of the same size. Occlusion is out of
          distribution, so covering ANYTHING moves the action somewhat, and a
          map without its control is a map of that. */}
      {strongest === null ? (
        <p className="answer unmeasured">
          <span className="answer-n">nothing measured</span>
          <span className="answer-of">
            this finding carries a map with no scored blocks in it
          </span>
        </p>
      ) : (
        <p
          className={`answer${verdict === "cleared" ? "" : " unmeasured"}`}
        >
          <span className="answer-n">
            {verdict === "cleared"
              ? `${where(strongest)} moved the action`
              : verdict === "not-cleared"
                ? "no patch beat its control"
                : verdict === "no-numbers"
                  ? "control not recorded"
                  : "uncontrolled"}
          </span>
          <span className="answer-of">
            {controlRead(strongest) && strongest.clears_control === true ? (
              <>
                by {measured(strongest.shift, 4)} when covered — more than the{" "}
                {measured(strongest.control_max, 4)} the best of{" "}
                {counted(strongest.control_draws, "random occlusion")} managed,
                which is what makes it a finding rather than a number
              </>
            ) : controlRead(strongest) && strongest.clears_control === false ? (
              <>
                the largest shift was {measured(strongest.shift, 4)} at{" "}
                {where(strongest)} and a random occlusion of the same size
                managed {measured(strongest.control_max, 4)}. Covering anything
                moves a policy that never saw a covered frame
              </>
            ) : strongest.clears_control !== null ? (
              <>
                the largest shift was {measured(strongest.shift, 4)} at{" "}
                {where(strongest)}, and this file records a verdict for it
                without the control numbers behind it — so there is nothing
                here to check the shift against. A verdict is not evidence on
                its own
              </>
            ) : (
              <>
                the largest shift was {measured(strongest.shift, 4)} at{" "}
                {where(strongest)}, and no random occlusion was run beside it —
                so nothing here separates it from what covering any patch would
                have done. Not a failed test: a test nobody ran
              </>
            )}
          </span>
        </p>
      )}

      {frame && (
        <div className="row vfr-frame-row">
          <figure className="vfr-frame">
            <img
              src={frame}
              alt={`episode ${run.provenance.episode}, timestep ${run.provenance.timestep}`}
              {...(sized ? { width: sized[0], height: sized[1] } : {})}
            />
            <figcaption className="meta">
              {/* The frame's OWN resolution, never the element's: an <img>
                  scales to its box and the box is the CSS's choice. */}
              {sized ? (
                <>
                  {sized[0]}×{sized[1]}
                </>
              ) : (
                <span className="warn">states no resolution</span>
              )}
              {run.frame_downsampled && (
                <>
                  <br />
                  <span className="warn">
                    {run.frame_note ||
                      "downsampled to fit the file; the original resolution is not recorded"}
                  </span>
                </>
              )}
            </figcaption>
          </figure>
        </div>
      )}

      {linked && (
        <p className="meta warn">
          The frame in this file points at a picture somewhere else instead of
          carrying one, and was not rendered. A `.mri` never fetches: opening
          one must not tell whoever wrote it that you did.
        </p>
      )}

      {occ && (
        <>
          <h3 className="irr-h">WHAT MOVED THE ACTION, PATCH BY PATCH</h3>
          <p className="meta">
            Each block is one patch of the frame, covered, with the policy
            re-run. The shift is how far the action moved. Occlusion is OUT OF
            DISTRIBUTION — the policy never saw a covered frame — so the
            control column is the measurement and the shift alone is not.
          </p>
          <ul className="irr-rows vfr-blocks">
            {blocks.map((b, i) => (
              <li key={i}>
                <b>{where(b)}</b>{" "}
                <span className="meta">shift {measured(b.shift, 4)}</span>
                {!controlRead(b) ? (
                  <span className="meta">
                    {" "}
                    ·{" "}
                    <span className="warn">
                      {b.clears_control === null
                        ? "not controlled, so this shift is not yet a finding"
                        : "a verdict with no control numbers behind it, so there is nothing to check it against"}
                    </span>
                  </span>
                ) : (
                  <span className="meta">
                    {" "}
                    · control {measured(b.control_max, 4)} over{" "}
                    {counted(b.control_draws, "draw")} ·{" "}
                    {b.clears_control ? (
                      "clears it"
                    ) : (
                      <span className="warn">does not clear it</span>
                    )}
                  </span>
                )}
                {typeof b.attention === "number" && (
                  <span className="meta">
                    {" "}
                    · attention {measured(b.attention, 4)}
                  </span>
                )}
              </li>
            ))}
          </ul>
          <p className="meta">
            {blocks.length} block(s) · {controlled.length} controlled ·{" "}
            {cleared.length} cleared their control
            {occ.baseline && <> · filled with {occ.baseline}</>}
            {Array.isArray(occ.grid) && occ.grid.length === 2 && (
              <>
                {" "}
                · a {occ.grid[0]}×{occ.grid[1]} grid
              </>
            )}
            {typeof occ.stride === "number" && occ.stride > 1 && (
              <> · stride {occ.stride}</>
            )}
            {typeof occ.passes === "number" && occ.passes > 0 && (
              <> · {occ.passes} forward pass(es)</>
            )}
          </p>

          {/* ATTENTION IS NOT A CAUSE, and this is the number that says so.
              A negative agreement is a real and common result — the policy
              looking hardest where covering changes least — not an error, so
              it is printed with its sign rather than as a magnitude. */}
          <p className="meta">
            {agreement === null || agreement === undefined ? (
              <span className="warn">
                the policy&apos;s attention was not compared with what actually
                moved the action in this file, so nothing here says whether the
                two agree
              </span>
            ) : (
              <>
                attention agrees with cause at {signed(agreement, 3)}
                {occ.compared_layer === null || occ.compared_head === null ? (
                  <>
                    {" "}
                    ·{" "}
                    <span className="warn">
                      against an attention map this file does not identify
                    </span>
                  </>
                ) : (
                  <>
                    {" "}
                    · measured against layer {occ.compared_layer}, head{" "}
                    {occ.compared_head}
                  </>
                )}
                . Attention is where the policy LOOKED; the shifts above are
                what changed the action when covered. They are different
                measurements and a low or negative agreement is a finding, not
                a fault.
              </>
            )}
          </p>
          {occ.means && <div className="hint">{occ.means}</div>}
        </>
      )}
    </div>
  );
}
