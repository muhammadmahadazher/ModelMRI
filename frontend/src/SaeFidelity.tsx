// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

import { useEffect, useRef, useState } from "react";
import {
  CEFidelity,
  CEFidelityCost,
  CEFidelityPrice,
  errorText,
  humanSeconds,
  saeFidelity,
  saeFidelityCost,
  saeFidelityPrice,
  SAEStatus,
} from "./api";
import { measured, percent } from "./measured";
import CorpusPicker from "./CorpusPicker";
import Disclosure from "./Disclosure";
import ReceiptLine from "./ReceiptLine";

/** How much of the model's predictive loss survives this SAE's reconstruction.
 *
 *  THE OTHER HALF OF THE BANNER ABOVE IT. `SAECalibration` reports FVU and L0,
 *  and both are ACTIVATION-space numbers: they ask whether the reconstruction
 *  is close to the vector the SAE was handed. The model does not care about
 *  that vector, it cares about the logits, and the directions carrying the
 *  residual stream's variance are not the directions the next token depends
 *  on. So an SAE can post an excellent FVU and still cost the model most of
 *  its predictive loss, and nothing on this panel could previously notice.
 *
 *  FOUR THINGS THIS CARD REFUSES TO LET A READER MISREAD:
 *
 *    the floor is on the card   "92%" alone is not a measurement. The same
 *                               reconstruction scores differently against
 *                               mean-ablation and zero-ablation, so the floor
 *                               is named beside the number and the select that
 *                               chose it starts EMPTY — there is no house
 *                               answer to a question the reader has to be told
 *                               the answer to.
 *    zero is not "not measured" Before the first run there is no percentage,
 *                               and the slot says so in words. A 0% here would
 *                               be a broken SAE reported as a fact.
 *    the number is not clamped  Below zero means the reconstruction predicts
 *                               WORSE than destroying the activation does.
 *                               That is a real answer and the one this
 *                               measurement most exists to find, so nothing
 *                               here takes a `Math.max(0, x)` and the drawing
 *                               puts the marker outside the span rather than
 *                               pinning it to the end.
 *    the price comes first      `3n + 2` forward passes, quoted from the
 *                               server before the button will run — the count
 *                               is arithmetic and free, so there is no excuse
 *                               for finding out afterwards.
 */
export default function SaeFidelityCard({
  sae,
  epoch,
  disabled,
}: {
  sae: SAEStatus;
  /** Bumped after each generation and on a model swap. A CE-recovered is a
   *  claim about the SAE against THE MODEL THAT WAS LOADED, so a swap has to
   *  clear it rather than leave one model's percentage under another's. */
  epoch: number;
  disabled?: boolean;
}) {
  const [text, setText] = useState("");
  // An id from `GET /api/corpus/available`, or a path — `resolve_any` takes
  // either under the one `file` field. It BEATS the box above on the server
  // (`/api/sae/fidelity` reads the file and never looks at `texts`), so the
  // box is disabled while this holds something rather than silently ignored.
  const [corpus, setCorpus] = useState("");
  // NO PRE-SELECTED VALUE. This empty string is the whole argument of the
  // measurement arriving in the DOM: a `<select>` that opened on "mean_ablate"
  // would answer the question for the reader and label the result as though
  // they had chosen.
  const [floor, setFloor] = useState("");
  const [cap, setCap] = useState("");
  // The free half of the price: `3n + 2`, arithmetic, no model involved. Only
  // the pasted arm can have it — a file's sequence count is not knowable until
  // somebody opens the file.
  const [price, setPrice] = useState<CEFidelityPrice | null>(null);
  // The measured half, and it is NOT free: three real forward passes — a
  // warm-up, a capture and a probe — so it is behind its own button and never
  // runs on mount or on a keystroke. It opens the corpus, so it is also the
  // only way this card can learn how many sequences a chosen FILE holds.
  const [timed, setTimed] = useState<CEFidelityCost | null>(null);
  const [pricing, setPricing] = useState(false);
  const [got, setGot] = useState<CEFidelity | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const lines = text
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
  const file = corpus.trim();
  const readable = file !== "" || lines.length > 0;
  // Blank means the whole corpus, and so does anything that is not a count.
  // `Number("")` is 0 and `Number("x")` is NaN; both would reach the server as
  // a cap — JSON.stringify writes NaN as null — and quietly change what was
  // scored while the label still said the whole file.
  const typedCap = Number(cap);
  const maxSequences =
    cap.trim() === "" || !Number.isFinite(typedCap) || typedCap < 1
      ? null
      : Math.floor(typedCap);

  useEffect(() => {
    setGot(null);
    setErr("");
  }, [epoch, sae.repo, sae.hook]);

  // Bumped whenever the corpus moves, and captured by both prices before they
  // await. Clearing the state below is not enough on its own: each price is a
  // round trip, and the picker and the box stay usable while one is in flight,
  // so a reply that lands late would repopulate the count from the corpus that
  // has just been replaced.
  const generation = useRef(0);

  // The price follows the corpus. Left standing, "32 passes" would sit beside
  // a button about to spend 3,002 — which is the cost preflight lying, and a
  // lying preflight is worse than none because it is the number people act on.
  useEffect(() => {
    generation.current += 1;
    setPrice(null);
    setTimed(null);
  }, [text, corpus, cap]);

  // What the reader has actually been shown on the control they are about to
  // click, and therefore what a click on it consents to. Null means nobody has
  // been told the cost yet, and the button says so rather than quoting a
  // number from a corpus that has since changed: the effect above clears both
  // prices the moment the corpus moves, and the generation it bumps is what
  // stops a reply still in flight from putting the old one back.
  const passes = timed?.passes ?? price?.passes ?? null;

  async function quote() {
    // NOT FOR THE FILE ARM. A file's sequence count is not knowable on this
    // side — only the server can open it — and the server reads `file` and
    // discards `texts`, so pricing the box here would quote a count the run
    // will not spend. It would also be taken as consent: the button carries
    // whatever count it has and sends `confirm` with it, so a stale 32 would
    // start a fifteen-thousand-pass run nobody was shown the size of.
    if (file !== "" || !lines.length) return;
    const mine = generation.current;
    const n =
      maxSequences === null
        ? lines.length
        : Math.min(lines.length, Math.max(1, maxSequences));
    try {
      const priced = await saeFidelityPrice(n);
      // The corpus moved while this was in flight, and the effect above has
      // already cleared the price it replaced. This one is about the old one.
      if (generation.current !== mine) return;
      setPrice(priced);
    } catch {
      // A price nobody could fetch is not worth an error banner over: the run
      // reports its own `passes` either way, and the server refuses a corpus
      // over the gate on its own.
      setPrice(null);
    }
  }

  async function timeIt() {
    if (!readable || pricing || busy) return;
    const mine = generation.current;
    setPricing(true);
    setErr("");
    try {
      const measured = await saeFidelityCost({
        ...(file ? { file } : { texts: lines }),
        max_sequences: maxSequences,
      });
      // Three real passes take real seconds, and neither the picker nor the
      // box is locked while they run. A count for the corpus that WAS chosen
      // is the same lie as a count for the pasted text — worse, because this
      // is the arm that learns a file's size, and the button would confirm it.
      if (generation.current !== mine) return;
      setTimed(measured);
    } catch (e) {
      setTimed(null);
      setErr(errorText(e));
    } finally {
      setPricing(false);
    }
  }

  async function run() {
    if (!readable || busy || !floor) return;
    setBusy(true);
    setErr("");
    try {
      setGot(
        await saeFidelity({
          // No `label` on the file arm: the server names the corpus from the
          // file it opened, and passing our word for it alongside would put
          // it on somebody else's measurement.
          ...(file ? { file } : { texts: lines, label: "pasted text" }),
          floor,
          max_sequences: maxSequences,
          // The button carries the pass count when there is one, so a click on
          // it IS the confirmation. With no count shown — a file nobody has
          // opened yet — this goes unconfirmed and the server's own gate
          // decides, in its own words, which is the arm that catches a corpus
          // file with twenty thousand lines in it.
          confirm: passes !== null,
        }),
      );
    } catch (e) {
      // A refusal replaces the previous answer rather than sitting beside it.
      setGot(null);
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  const cal = got?.calibration ?? null;

  return (
    <div className="sae-fidelity">
      <Disclosure
        dot="d-feat"
        title="WHAT THIS SAE COSTS THE MODEL'S ANSWERS"
        asks="The reconstruction is close to the activations — but does the model still predict as well through it? A different question from the FVU above, with a different answer."
        hasResult={got !== null}
        disabled={disabled}
      >
        <textarea
          className="corpus-box"
          rows={4}
          placeholder={"one line per sequence\nnothing is uploaded"}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onBlur={() => void quote()}
          /* Disabled, not merely overridden. The server reads the file and
             discards `texts` when both arrive, so leaving this typeable would
             let a reader edit lines that were never going to be read. */
          disabled={disabled || busy || file !== ""}
          spellCheck={false}
        />

        <CorpusPicker
          id="ce-corpus"
          value={corpus}
          onChange={setCorpus}
          disabled={disabled || busy}
        />

        <div className="row ce-dials">
          <label className="meta" htmlFor="ce-floor">
            floor
          </label>
          <select
            id="ce-floor"
            value={floor}
            onChange={(e) => setFloor(e.target.value)}
            disabled={disabled || busy}
            title={
              "What the reconstruction is scored against. These are two " +
              "different questions and they give different percentages for " +
              "the same SAE, which is why there is no default: mean-ablation " +
              "replaces the activation with this corpus's own mean vector at " +
              "this hook, zero-ablation with the zero vector — a point the " +
              "stream never visits."
            }
          >
            <option value="" disabled>
              choose one…
            </option>
            <option value="mean_ablate">mean-ablation floor</option>
            <option value="zero_ablate">zero-ablation floor</option>
          </select>

          <label className="meta" htmlFor="ce-cap">
            first
          </label>
          <input
            id="ce-cap"
            className="ce-cap"
            type="number"
            min={1}
            step={1}
            inputMode="numeric"
            placeholder="all"
            value={cap}
            onChange={(e) => setCap(e.target.value)}
            onBlur={() => void quote()}
            disabled={disabled || busy}
            title="Score only the first N sequences. Blank is the whole corpus; the result says how many of it was covered and marks itself truncated when some was not."
          />
          <span className="meta">sequences</span>

          {/* Its own button, because it is not free: three real forward
              passes to time one. It is also the only way this card can learn
              how many sequences a chosen FILE holds, which is why it is the
              one control the file arm needs before the price means anything. */}
          <button
            className="ghost sm"
            onClick={() => void timeIt()}
            disabled={disabled || busy || pricing || !readable}
            title="Runs the model three times — a warm-up, a capture and one probe pass — and projects the whole sweep from that. The pass count transfers between machines; the seconds do not."
          >
            {pricing ? "timing…" : "Time it here (3 passes)"}
          </button>

          <button
            className="ghost sm"
            onClick={() => void run()}
            disabled={disabled || busy || pricing || !readable || !floor}
            title={
              floor
                ? "Three forward passes per sequence — the model's own, the reconstruction spliced in, the floor spliced in — plus two taken once for the resolution every difference is read against."
                : "Choose a floor first. The percentage is not interpretable without one."
            }
          >
            {busy
              ? "measuring…"
              : passes !== null
                ? `Run — ${passes.toLocaleString()} passes`
                : file
                  ? "Open that corpus and run"
                  : "Run"}
          </button>
        </div>

        {/* Priced before it is spent, like every other sweep here — and the
            two prices are different things. `3n+2` is arithmetic and arrives
            the moment the box loses focus; the seconds cost three real passes
            and are behind their own button. A FILE has neither until it is
            opened, and that is SAID rather than left blank: a missing price
            beside a control that quotes one everywhere else reads as free. */}
        {price && <p className="meta">{price.means}</p>}
        {!price && file && !timed && (
          <p className="meta">
            three forward passes per sequence plus two — but how many sequences
            that file holds is not known until it is opened, so there is no
            total here yet. Time it, or run it and let the server refuse a
            corpus too big to start unasked.
          </p>
        )}
        {timed && (
          <p className="meta">
            {timed.means}{" "}
            {timed.estimate.seconds === null ? (
              /* An unmeasured wait is not a wait of zero seconds. */
              <>
                How long that takes here was not measured:{" "}
                {timed.estimate.unmeasured || "this machine would not say."}
              </>
            ) : (
              <>
                About {humanSeconds(timed.estimate.seconds)} on this machine,
                projected from one pass over {timed.probed_sequence_length}{" "}
                tokens.
              </>
            )}
          </p>
        )}

        {/* The refusal, in the server's own words. Every sentence this route
            answers with names what is missing and what to do — "load an SAE
            first", "measure at a hook point this text's predictions depend
            on" — and collapsing it into "error" throws away the only part a
            reader can act on. */}
        {err && <div className="hint err">{err}</div>}

        {busy && <FidelitySkeleton />}

        {!busy && !got && !err && (
          /* NOT 0%. Before the first run there is no percentage, and a zero
             here would report a broken SAE as a measured fact — the loudest
             thing on the card being false. */
          <p className="answer unmeasured">
            <span className="answer-n">not measured</span>
            <span className="answer-of">
              nothing has been run against this SAE yet. Paste a few lines or
              pick a corpus, choose a floor, and this says what fraction of the
              model's own predictive loss survives the reconstruction.
            </span>
          </p>
        )}

        {!busy && got && (
          <div className="ov-readout">
            <p className="answer">
              <span className="answer-n">
                <CountUp value={got.ce_recovered} />
              </span>
              <span className="answer-of">
                of the model's lost loss is recovered, against the{" "}
                <b>
                  {got.floor === "mean_ablate" ? "mean-ablation" : "zero-ablation"}{" "}
                  floor
                </b>
                {got.ce_recovered < 0
                  ? " — below zero, which means the reconstruction predicts worse than destroying the activation does"
                  : ""}{" "}
                · measured on {got.corpus_label} ·{" "}
                {got.n_tokens.toLocaleString()} predicted tokens in{" "}
                {got.n_sequences.toLocaleString()} of{" "}
                {got.n_sequences_given.toLocaleString()} sequences
              </span>
            </p>

            <LossScale got={got} />

            <div className="row">
              {/* `.at(-1)` and not `[1]`: an SAE repo is usually `owner/name`,
                  but a locally named one need not carry a slash at all, and
                  indexing past the end would print "undefined" as the name of
                  the thing being measured. */}
              <span className="pill violet">
                {got.repo.split("/").at(-1) ?? got.repo} · L{got.layer} ·{" "}
                {sae.d_sae === null ? "?" : sae.d_sae.toLocaleString()} features
              </span>
              {cal && (
                <span className="meta">
                  l0 {cal.l0.toFixed(1)} firing per token · fvu{" "}
                  {measured(cal.fvu, 4)} in the {cal.convention} convention
                </span>
              )}
              {got.truncated && (
                <span className="meta warn">
                  the rest of the corpus was NOT MEASURED, which is not the
                  same as measured and found not to matter
                </span>
              )}
            </div>

            <p className="meta ce-sha">
              corpus sha256 <code>{got.corpus_sha256.slice(0, 16)}</code> — the
              label is what you call it, this is what actually ran
            </p>

            <Disclosure
              dot="d-feat"
              title="THE THREE LOSSES UNDER IT"
              asks="A different floor gives a different percentage from these same three numbers, which is why all three are here."
              hasResult
            >
              <ul className="ov-list">
                <li>
                  <span>the model&rsquo;s own</span>
                  <span>{measured(got.ce_clean, 6)}</span>
                </li>
                <li>
                  <span>with the reconstruction spliced in</span>
                  <span>{measured(got.ce_recon, 6)}</span>
                </li>
                <li>
                  <span>with the activation replaced by the floor</span>
                  <span>{measured(got.ce_ablate, 6)}</span>
                </li>
                <li>
                  <span>what the floor cost the model (the denominator)</span>
                  <span>{measured(got.denominator, 6)}</span>
                </li>
                <li>
                  <span>what the reconstruction saved (the numerator)</span>
                  <span>{measured(got.numerator, 6)}</span>
                </li>
              </ul>
              <p className="meta">
                Nats per predicted token. {got.floor_means}{" "}
                {/* An unknown is not a zero: the zero floor is not averaged
                    from anything, and `n_floor_tokens` is null rather than 0
                    for exactly that reason. */}
                {got.n_floor_tokens === null
                  ? "It is not an average of anything, so there is no token count under it."
                  : `Averaged over ${got.n_floor_tokens.toLocaleString()} activations.`}
              </p>
              <p className="meta">
                The resolution every difference above is read against: writing
                the model&rsquo;s own stream back unchanged moved its loss by{" "}
                {measured(got.splice_deviation_nats, 6)} nats per token, and
                running the same pass again with no hook at all moved it{" "}
                {measured(got.replay_deviation_nats, 6)}.{" "}
                {got.calibrated_here
                  ? "The convention was calibrated on this corpus's first sequence."
                  : "The convention was calibrated elsewhere; the token count beside it says on what."}
              </p>
              <p className="meta">{got.means}</p>
            </Disclosure>

            <p className="meta">
              {got.passes.toLocaleString()} forward passes in {got.elapsed_s}s.
            </p>
            <ReceiptLine receipt={got.receipt} />
          </div>
        )}
      </Disclosure>
    </div>
  );
}

/** The percentage, counted up to where it landed.
 *
 *  A number that arrives already at rest reads as though it had always been
 *  there; watching it move says a measurement just finished. It is decoration
 *  and nothing else — the final value is what the server sent, to the digit,
 *  and a reader who has asked their OS for less motion gets it immediately.
 */
const stillPlease = () =>
  window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

function CountUp({ value }: { value: number }) {
  // Starts at the value itself when motion is off, and at zero when it is not.
  // Initialising to `value` unconditionally paints the final number for one
  // frame, drops it to zero when the effect runs, and then counts back up to
  // where it already was — which reads as a glitch rather than as an arrival.
  const [shown, setShown] = useState(() => (stillPlease() ? value : 0));
  const frame = useRef(0);

  useEffect(() => {
    if (stillPlease()) {
      setShown(value);
      return;
    }
    const started = performance.now();
    const span = 620;
    const tick = (now: number) => {
      const t = Math.min(1, (now - started) / span);
      // Cubic ease-out: fast at the start, settling rather than stopping.
      setShown(value * (1 - (1 - t) ** 3));
      if (t < 1) frame.current = requestAnimationFrame(tick);
      // The last frame is set from `value` itself rather than from the eased
      // expression, so what comes to rest is the measurement and not a value
      // 0.0001 short of it.
      else setShown(value);
    };
    frame.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame.current);
  }, [value]);

  // `percent` and not `(x * 100).toFixed(1)`: a share below 0.0005 renders in
  // exponent form rather than as "0.0%", which is a real measurement printed
  // as none of it.
  return <>{percent(shown, 1)}</>;
}

/** The three losses on one axis, which is what the ratio is a picture of.
 *
 *  The percentage is the position of the reconstruction between the floor and
 *  the model's own loss. Drawn rather than described because the relationship
 *  is spatial, and because a reconstruction that lands OUTSIDE that span — the
 *  negative case — is instantly visible as a marker off the end and is not
 *  visible at all in a number that was clamped to keep a bar inside its box.
 */
function LossScale({ got }: { got: CEFidelity }) {
  const values = [got.ce_clean, got.ce_recon, got.ce_ablate];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo;
  // A flat scale is a real state — three losses that agree — and dividing by
  // it would put every marker at NaN. Centre them instead and let the numbers
  // beside the axis say so.
  const at = (v: number) => (span > 0 ? 6 + ((v - lo) / span) * 88 : 50);

  const marks: { v: number; label: string; cls: string }[] = [
    { v: got.ce_clean, label: "model's own", cls: "ce-clean" },
    { v: got.ce_recon, label: "reconstructed", cls: "ce-recon" },
    { v: got.ce_ablate, label: `${got.floor} floor`, cls: "ce-ablate" },
  ];

  return (
    <>
      <svg
        className="ce-scale"
        viewBox="0 0 100 20"
        preserveAspectRatio="none"
        role="img"
        aria-label={`Cross-entropy in nats per token: the model's own ${got.ce_clean}, with the reconstruction spliced in ${got.ce_recon}, with the ${got.floor} floor spliced in ${got.ce_ablate}.`}
      >
        <line className="ce-axis" x1="4" y1="10" x2="96" y2="10" />
        {/* The span the ratio is taken over: floor to the model's own loss. */}
        <line
          className="ce-span"
          x1={at(got.ce_ablate)}
          y1="10"
          x2={at(got.ce_clean)}
          y2="10"
        />
        {marks.map((m) => (
          <line
            key={m.label}
            className={`ce-mark ${m.cls}`}
            x1={at(m.v)}
            y1="3"
            x2={at(m.v)}
            y2="17"
          />
        ))}
      </svg>
      {/* The key is HTML rather than `<text>` inside the drawing. The viewBox
          is stretched to the panel's width — only the horizontal positions
          carry information — and stretched type is unreadable at exactly the
          narrow widths where a legend matters most. */}
      <p className="meta ce-legend">
        {marks.map((m) => (
          <span key={m.label} className="ce-key">
            <span className={`ce-swatch ${m.cls}`} aria-hidden="true" />
            {m.label} {measured(m.v, 4)}
          </span>
        ))}
      </p>
    </>
  );
}

/** While the sweep runs. Low-contrast on purpose: a skeleton that shimmers as
 *  brightly as real content trains you to read it, and then you read nothing.
 */
function FidelitySkeleton() {
  return (
    <div className="ce-skel" aria-hidden="true">
      <div className="skel-bar ce-skel-answer" />
      <div className="skel-bar ce-skel-line" />
      <div className="skel-bar ce-skel-short" />
    </div>
  );
}
