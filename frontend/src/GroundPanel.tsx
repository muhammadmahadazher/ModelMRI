import { CSSProperties, useEffect, useState } from "react";
import { measured, percent } from "./measured";
import RunsOn, { useModelReady } from "./RunsOn";
import { errorText, GroundScore, Grounding, groundAnswer } from "./api";
import ReceiptLine from "./ReceiptLine";
import { useScanOnData } from "./useScanOnData";

/**
 * Did the answer come from the document, or from the weights?
 *
 * Every local-LLM app with RAG shows you WHICH CHUNKS WERE RETRIEVED. None of
 * them shows whether the answer used them — and a retriever that pulled the
 * right paragraph beside a model that ignored it and answered from memory
 * looks identical to a working system in all of them. That combination is
 * what a confident hallucination looks like from outside.
 *
 * So this panel is two measurements per passage, drawn SIDE BY SIDE, and its
 * whole layout is built around the case where they disagree:
 *
 *   dependence   how far the answer moved when the passage was masked out
 *   attention    how much of the answer position's attention landed on it
 *
 * Three things this renders that a retrieval panel cannot:
 *
 *   - LOOKED AT, NOT DEPENDED ON gets its own row treatment and its own line
 *     in the summary. It is the finding, not a footnote.
 *   - The two bars have SEPARATE SCALES and each says so, because nats and an
 *     attention share are different quantities. A shared axis would invite
 *     exactly the comparison the panel exists to prevent.
 *   - Nothing is a percentage of the answer. Masking a whole passage is a big
 *     intervention and the effects are not additive — the joint mask is
 *     printed next to the parts so a reader can see they do not sum.
 */

/** A short document that produces a real finding on arrival: one passage
 *  carries the date the question asks for and three do not. Short enough to
 *  read in the box, so nobody has to take the split on trust. */
const DOC_DEFAULT = [
  "The Antikythera mechanism was recovered from a shipwreck in 1901.",
  "It is an ancient Greek geared device used to predict astronomical positions and eclipses decades in advance.",
  "The device was found off the coast of the island of Antikythera, between Crete and the Peloponnese.",
  "Unrelated paragraph about coffee. Beans grown at high altitude ripen more slowly and are usually more acidic than beans grown lower down.",
  "Another unrelated paragraph. The train from the coast leaves every twenty minutes on weekdays and hourly at the weekend.",
].join("\n\n");

const QUESTION_DEFAULT =
  "Question: In which year was the mechanism recovered?\nAnswer:";

/** How a passage is labelled, in the order the labels have to be decided.
 *
 *  `looked_not_used` comes FIRST when it is true. A passage can be both "did
 *  not clear the floor" and "took a third of the attention", and reporting
 *  the first is true while throwing away the only interesting thing about it.
 *
 *  `null` is a third state and it is NOT "false". It means the reading could
 *  not be taken on this run, so nothing here is allowed to imply it was. */
function verdict(c: GroundScore): { text: string; cls: string } {
  if (c.looked_not_used)
    return { text: "looked at, not depended on", cls: "g-looked" };
  // "the answer moved" rather than "depended on". Under a zero floor that is
  // the strongest thing that is actually true of a passage here, and the
  // banner above says so — a row reading "depended on" would restate the
  // claim the banner just withdrew.
  if (c.depended_on) return { text: "the answer moved", cls: "g-used" };
  // Not clearing the floor is not clearing the floor, whether or not the
  // "looked at" half was measurable. An undecidable `looked_not_used` only
  // means the qualifier cannot be ADDED, and the block above already says
  // that once rather than on every row.
  return { text: "no measurable effect", cls: "g-neither" };
}

export default function GroundPanel({
  epoch,
  recorded,
}: {
  epoch: number;
  /** Set when a `.mri` is open and carries a grounding: the question it
   *  answered, so the panel shows the recording's own finding rather than a
   *  form whose only button can refuse.
   *
   *  The DOCUMENT is deliberately not here. A `.mri` carries ~120-character
   *  passage previews and not the passages, because a grounded document is
   *  usually the private half of the pair — so a recording can show what was
   *  measured and can never re-measure it. */
  recorded?: { available: boolean; question: string };
}) {
  // Nothing loaded means every button here can only be refused. Shares
  // `RunsOn`'s cached session, so the badge and the control it disables
  // read one answer rather than two requests that can disagree.
  const ready = useModelReady(epoch);
  const [doc, setDoc] = useState(DOC_DEFAULT);
  const [file, setFile] = useState("");
  const [question, setQuestion] = useState(
    recorded?.question || QUESTION_DEFAULT,
  );
  const [data, setData] = useState<Grounding | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const scanRef = useScanOnData(data);

  // Grounding runs its own document-and-question and does not commit the
  // prompt, so it does not depend on the current generation — but it does
  // depend on the loaded model, and the epoch moves on load and unload.
  useEffect(() => {
    setData(null);
    setErr("");
  }, [epoch]);

  async function run() {
    setBusy(true);
    setErr("");
    setData(null);
    try {
      setData(
        await groundAnswer(
          recorded
            ? // The server serves the recording and ignores the body, but
              // sending a document it will not use would invite a reader to
              // believe it was measured against one.
              { document: "(recorded)", question: recorded.question }
            : file.trim()
              ? { file: file.trim(), question }
              : { document: doc, question },
        ),
      );
    } catch (e) {
      // The refusals carry the feature's terms: the passage cap quotes what
      // the run would cost, and the tokenizer refusal explains why a slow
      // tokenizer cannot be worked around.
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  // Each bar is scaled against the longest of ITS OWN quantity. Nats and an
  // attention share have no common axis, and putting them on one would make
  // "this bar is longer than that one" a sentence about nothing.
  const widestDep = data
    ? data.chunks.reduce((m, c) => Math.max(m, c.dependence), 0) || 1
    : 1;
  const widestAttn = data
    ? data.chunks.reduce((m, c) => Math.max(m, c.attention ?? 0), 0) || 1
    : 1;
  const looked = data ? data.chunks.filter((c) => c.looked_not_used) : [];
  // Three-valued, so `!looked.length` is not the same question. A run where
  // the reading could not be taken and a run where no passage had the problem
  // produce the same empty list and mean opposite things.
  const undecidable = !!data && data.chunks.some((c) => c.looked_not_used === null);
  const partsSum = data
    ? data.chunks.reduce((s, c) => s + c.dependence, 0)
    : 0;

  return (
    <div className="panel ground" ref={scanRef}>
      <div className="sect">
        <span className="dot d-ground" />
        <h2 className="h-ground">
          GROUNDING — THE DOCUMENT, OR THE WEIGHTS?
        </h2>
        <span className="rule" />
      </div>
      <RunsOn epoch={epoch} />
      <p className="meta">
        Attach a passage of your own text and ask a question about it. Every
        RAG interface shows you which chunks were <i>retrieved</i>; this
        measures whether the answer actually <b>depended</b> on them — one
        forward pass per passage, with that passage masked out of the model's
        attention. Nothing is downloaded, nothing is indexed and nothing is
        embedded: every passage you paste is in the prompt.
      </p>

      {recorded && (
        <p className="meta ground-recorded">
          This is a recording. It carries what was measured — the dependence
          and attention of each passage — and <b>not the document</b>: a
          `.mri` holds a short preview of each passage on purpose, because the
          text somebody grounds an answer in is usually the half they do not
          want forwarded. Nothing here can be re-run without that text and the
          model it was measured on.
        </p>
      )}

      {!recorded && (
      <div className="ground-inputs">
        <label>
          <span className="meta">
            the document — a blank line starts a new passage
          </span>
          <textarea
            value={doc}
            onChange={(e) => setDoc(e.target.value)}
            spellCheck={false}
            rows={8}
            disabled={
            ready === false ||!!file.trim()}
          />
        </label>
        <div className="ground-side">
          <label>
            <span className="meta">…or a local .txt / .jsonl</span>
            <input
              value={file}
              onChange={(e) => setFile(e.target.value)}
              spellCheck={false}
              placeholder="notes.txt — or a path to one"
            />
          </label>
          <label>
            <span className="meta">
              the question — it goes last, and the answer is read at the end
            </span>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              spellCheck={false}
              rows={3}
            />
          </label>
        </div>
      </div>
      )}

      <div className="row" style={{ margin: "10px 0" }}>
        <button
          className="cta"
          onClick={() => void run()}
          disabled={
            busy || (!recorded && ((!doc.trim() && !file.trim()) || !question.trim()))
          }
        >
          {busy
            ? recorded
              ? "Reading the recording…"
              : "Masking every passage…"
            : recorded
              ? "Show the recorded grounding"
              : "Ask where the answer came from"}
        </button>
        <span className="meta">
          {recorded
            ? recorded.question || "the question this file answered"
            : file.trim()
              ? "read from disk, never sent anywhere"
              : "one forward pass per passage, plus four"}
        </span>
      </div>

      {err && <div className="hint err">{err}</div>}

      {data && (
        <>
          <div className="ground-answer meta">
            the model's next token here was{" "}
            <b>{JSON.stringify(data.answer)}</b> at p={measured(data.answer_p, 4)}{" "}
            · {data.n_chunks} passages over {data.n_prompt_tokens} tokens ·{" "}
            {data.passes} passes, {data.seconds}s
          </div>

          {/* The verdict, in the order the claims can actually be made.
              Ungrounded first — it is the strongest thing this can say and
              the reason the panel exists. */}
          <div
            className={`ground-verdict ${
              data.ungrounded ? "none" : data.floor_degenerate ? "weak" : "ok"
            }`}
          >
            {data.ungrounded ? (
              <>
                <b>No passage cleared the noise floor.</b> Removing any one of
                them moved the answer no further than a pass that changed
                nothing, so on this evidence the answer did not depend on the
                document you attached. That is a measurement, not a verdict on
                whether the answer is correct.
              </>
            ) : data.floor_degenerate ? (
              <>
                <b>The noise floor here is exactly zero</b> — this model
                reproduced its own answer bit for bit, so every passage that
                moved it at all counts as clearing, and{" "}
                {data.chunks.filter((c) => c.depended_on).length} of{" "}
                {data.n_chunks} did. Read the nats below, not the labels:
                there is no significance test on this run.
              </>
            ) : (
              <>
                {data.chunks.filter((c) => c.depended_on).length} of{" "}
                {data.n_chunks} passages moved the answer further than this
                model's own run-to-run spread ({measured(data.noise_floor, 4)}{" "}
                nats).
              </>
            )}
          </div>

          {/* The finding, given its own block rather than left as a badge to
              be spotted in a list — and its ABSENCE given a block too when
              the reading could not be taken. A panel that simply shows
              nothing there is telling the reader everything is fine. */}
          {undecidable && (
            <div className="ground-flag undecided">
              <b>
                Looked at but not depended on could not be decided on this
                run.
              </b>{" "}
              {!data.attention_available
                ? "The attention half did not run, so there is no “looked at” to pair with anything."
                : "The noise floor is exactly zero, so every passage that moved the answer at all counts as depended-on and the flag could never fire."}{" "}
              No passage is flagged below, and that is the absence of a test
              rather than a clean result.
            </div>
          )}
          {looked.length > 0 && (
            <div className="ground-flag">
              <b>
                Looked at, not depended on:{" "}
                {looked.map((c) => `#${c.index}`).join(", ")}.
              </b>{" "}
              The answer position put attention on{" "}
              {looked.length === 1 ? "that passage" : "those passages"} and
              removing {looked.length === 1 ? "it" : "them"} changed nothing
              measurable. Attention is where the model looked, not what it
              used — and this pair disagreeing is what an answer coming from
              the weights looks like from outside.
            </div>
          )}

          {/* TWO SCALES, each labelled with its own unit and its own longest
              bar. One shared axis would invite reading a long attention bar
              as a large dependence, which is precisely the mistake. */}
          {/* Decorative once every value carries its own unit — a screen
              reader hearing "dependence 3.2526 nats" does not also need the
              column title, and the two together read as a stutter. */}
          <div className="ground-heads meta" aria-hidden="true">
            <span />
            <span>
              dependence · nats · longest {measured(widestDep, 3)}
            </span>
            <span>
              {data.attention_available
                ? `attention · share · longest ${measured(widestAttn, 3)}`
                : "attention · not measurable on this model"}
            </span>
            <span />
          </div>

          <ol className="ground-rows stagger">
            {data.chunks.map((c, i) => {
              const v = verdict(c);
              return (
                <li
                  key={c.index}
                  className={v.cls}
                  style={{ "--i": i } as CSSProperties}
                >
                  <span className="mid ground-id">#{c.index}</span>
                  <span className="ground-track">
                    <span
                      className="ground-bar dep"
                      style={{ width: `${(c.dependence / widestDep) * 100}%` }}
                    />
                    {/* The unit rides with the VALUE, not only with the
                        column header. A list has no header-to-cell
                        association, so with the header alone a screen reader
                        reads "#0, 3.2526, 0.3775" — two numbers in different
                        units, announced identically. */}
                    <span
                      className="mid ground-num"
                      aria-label={`dependence ${measured(c.dependence, 4)} nats`}
                    >
                      {measured(c.dependence, 4)}
                    </span>
                  </span>
                  <span className="ground-track">
                    {/* Blank, not a zero-width bar. A model that never
                        returned attention scores and a passage nothing looked
                        at must not draw the same. */}
                    {c.attention === null ? (
                      <span className="meta ground-unknown">not measured</span>
                    ) : (
                      <>
                        <span
                          className="ground-bar attn"
                          style={{
                            width: `${(c.attention / widestAttn) * 100}%`,
                          }}
                        />
                        <span
                          className="mid ground-num"
                          aria-label={`attention share ${measured(c.attention, 4)}`}
                        >
                          {measured(c.attention, 4)}
                        </span>
                      </>
                    )}
                  </span>
                  <span className="ground-body">
                    <span className="meta ground-verd">{v.text}</span>
                    <span className="ground-preview">{c.preview}</span>
                  </span>
                </li>
              );
            })}
          </ol>

          {/* Says what is missing and why, and stops there — the block above
              already reported that the looked-at reading could not be taken,
              and repeating it put the same sentence on screen twice. */}
          {!data.attention_available && (
            <p className="meta ground-warn">
              {data.attention_note} Every passage above still carries a
              dependence score and no attention share. Blank is what was
              measured; a zero would have been a claim.
            </p>
          )}

          {/* The one arithmetic fact that keeps this honest, printed rather
              than asserted: the parts do not sum to the whole, so no number
              here is a share of the answer. */}
          <p className="meta ground-joint">
            Masking <b>every</b> passage at once moved the answer by{" "}
            <b>{measured(data.joint, 4)}</b> nats. The passages above sum to{" "}
            {measured(partsSum, 4)}. They are not the same number and they never
            will be — masking a whole passage is a large intervention and the
            effects are not additive, which is why nothing here is a
            percentage of the answer.
            {data.attention_available && data.attention_share !== null && (
              <>
                {" "}
                The attention shares cover{" "}
                {percent(data.attention_share, 1)} of the answer
                position's mass; the rest went to the question, the template
                and any position not inside a passage.
              </>
            )}
          </p>

          <ReceiptLine receipt={data.receipt} />
        </>
      )}
    </div>
  );
}
