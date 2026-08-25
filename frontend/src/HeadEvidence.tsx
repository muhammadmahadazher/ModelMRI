import { useState } from "react";
import {
  errorText,
  headEvidence,
  headEvidenceCost,
  HeadEvidence as Evidence,
} from "./api";
import { measured } from "./measured";
import Disclosure from "./Disclosure";
import CorpusPicker from "./CorpusPicker";

/** What this head does on YOUR text — beside what it did on one prompt.
 *
 *  The ranking above answers "did this head matter for the prompt in the box",
 *  and the number moves with the prompt. `HeadWiring` answers "what is it
 *  wired to do", from weights alone. This answers the third question, which
 *  neither of them can: where does it act in a body of text you chose.
 *
 *  THREE THINGS THIS PANEL REFUSES TO LET A READER MISREAD, each of which
 *  cost a defect in the module before it was caught:
 *
 *    a zero write is not a span   a head that wrote once is not twenty
 *                                 findings, so the list only holds real writes
 *    sparse is not absent         a head writing once in ten positions has a
 *                                 median of zero and is not "not seen here"
 *    the pair, not the position   "wrote at 41" without "reading `Paris`" is
 *                                 the half that gets quoted
 *
 *  The cost is shown before the run, because a corpus sweep is two forward
 *  passes per sequence and nobody should discover that afterwards.
 */
export default function HeadEvidencePanel({
  layer,
  head,
  disabled,
}: {
  layer: number;
  head: number;
  disabled?: boolean;
}) {
  const [text, setText] = useState("");
  // An id from `GET /api/corpus/available`, or a path. The route takes either
  // one under the same `file` field, and it takes precedence over the pasted
  // box on the SERVER — `head_evidence` overwrites `texts` when `file` is set
  // — so the box has to be visibly out of play here rather than quietly
  // ignored, or a reader watches their pasted lines not be read.
  const [corpusFile, setCorpusFile] = useState("");
  const [readAttention, setReadAttention] = useState(true);
  const [out, setOut] = useState<Evidence | null>(null);
  const [price, setPrice] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const lines = text
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
  const file = corpusFile.trim();
  const readable = file !== "" || lines.length > 0;

  async function quote() {
    if (!lines.length) return;
    try {
      setPrice((await headEvidenceCost(lines.length, readAttention)).means);
    } catch {
      // A price nobody could fetch is not worth an error banner over: the run
      // below reports its own `passes` either way.
      setPrice("");
    }
  }

  async function run() {
    if (!readable || busy) return;
    setBusy(true);
    setErr("");
    try {
      setOut(
        await headEvidence({
          // No `label` on the file arm: `load_corpus` names the corpus from
          // the file it opened, and passing "pasted text" alongside would put
          // our word for it on somebody else's measurement.
          ...(file ? { file } : { texts: lines, label: "pasted text" }),
          layer,
          head,
          read_attention: readAttention,
        }),
      );
    } catch (e) {
      setOut(null);
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  // The result is of the head that was current when it ran. Moving the dial
  // has to invalidate it rather than leave one head's spans under another
  // head's label — the same staleness rule the heat map above follows.
  const stale =
    out !== null && (out.corpus.layer !== layer || out.corpus.head !== head);
  const corpus = out?.corpus;

  return (
    <div className="head-evidence">
      <Disclosure
        dot="d-attn"
        title="WHERE THIS HEAD ACTS IN YOUR TEXT"
        asks="Where does this head write, across a body of text you choose? The prompt above is not involved."
        hasResult={out !== null}
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
        disabled={disabled || file !== ""}
        spellCheck={false}
      />

      <CorpusPicker
        id="he-corpus"
        value={corpusFile}
        onChange={setCorpusFile}
        disabled={disabled || busy}
      />

      <div className="row">
        <label className="meta">
          <input
            type="checkbox"
            checked={readAttention}
            onChange={(e) => {
              setReadAttention(e.target.checked);
              setPrice("");
            }}
            disabled={disabled || busy}
          />{" "}
          also read what it was attending to
        </label>
        <button
          className="ghost sm"
          onClick={() => void run()}
          disabled={disabled || busy || !readable}
        >
          {busy
            ? "reading…"
            : file
              ? "Read that corpus"
              : `Read ${lines.length || "…"} line(s)`}
        </button>
        {/* Priced before it is spent, like every other sweep here. A FILE
            cannot be priced from here: `/cost` prices `n_sequences`, and how
            many sequences a file holds is not known until it is opened. That
            is said rather than left blank — a missing price beside a control
            that quotes one everywhere else reads as free. */}
        {file ? (
          <span className="meta">
            {readAttention ? "two forward passes" : "one forward pass"} per
            sequence — but how many sequences that file holds is not known
            until it is opened, so there is no total here.
          </span>
        ) : (
          price && <span className="meta">{price.split(".")[0]}.</span>
        )}
      </div>

      {err && <div className="hint err">{err}</div>}

      {corpus && !stale && (
        <div className="ov-readout">
          {/* THE ANSWER, at answer size. This count used to sit in an 11.5px
              pill among four others, which on a page where nothing is larger
              than the 15px body means the one figure the sweep exists to
              produce looked exactly like the caveats beside it.

              THE COUNT AND NOT THE SHARE. `n_wrote` alone says nothing — 200
              writes is a busy head over 300 positions and a silent one over
              20,000 — so the denominator travels in `.answer-of`, at meta
              size but never dropped. A share would have hidden both numbers
              behind a percentage the reader cannot check. */}
          <p className={`answer${corpus.n_wrote === 0 ? " unmeasured" : ""}`}>
            <span className="answer-n">
              {corpus.n_wrote === 0
                ? "wrote nowhere"
                : corpus.n_wrote.toLocaleString()}
            </span>
            <span className="answer-of">
              {corpus.n_wrote === 0
                ? `not one of the ${corpus.n_positions.toLocaleString()} positions in this corpus carried a write from this head — which is a measurement, not a failed run`
                : `of ${corpus.n_positions.toLocaleString()} positions carried a write · largest ${measured(corpus.write_norm_max, 3)} · median ${measured(corpus.write_norm_median, 3)}`}
            </span>
          </p>

          <div className="row">
            {corpus.truncated && (
              <span className="meta warn">
                the corpus was cut — a larger write further in is not ruled out
              </span>
            )}
            {!corpus.attention_read && (
              <span className="meta warn">attention not read</span>
            )}
          </div>

          {corpus.spans.length === 0 ? (
            /* NOT an empty table. A head that wrote nowhere in this text is a
               measurement, and rendering it as a blank list reads as a failed
               run rather than as the answer. */
            <div className="hint">{corpus.means}</div>
          ) : (
            <>
              <ol className="span-list">
                {corpus.spans.map((s, i) => (
                  <li key={i}>
                    <div className="span-line">
                      <code>{s.text}</code>
                    </div>
                    <div className="meta">
                      wrote at {s.position} (<b>{s.token}</b>) ·{" "}
                      {measured(s.write_norm, 3)}
                      {/* The pair. Absent is said, not left blank. */}
                      {s.source_token !== null ? (
                        <>
                          {" "}
                          · reading <b>{s.source_token}</b> at {s.source_position}
                          {s.source_share !== null &&
                            ` (${measured(s.source_share * 100, 0)}% of its attention)`}
                        </>
                      ) : (
                        " · what it was reading was not measured on this run"
                      )}
                    </div>
                  </li>
                ))}
              </ol>
              <div className="hint">{corpus.means}</div>
            </>
          )}

          {out?.pushes_at && (
            <div className="meta pad">
              And from its weights alone, reading{" "}
              <code>{out.pushes_at.source_token}</code> pushes toward{" "}
              {out.pushes_at.promotes.slice(0, 3).map((t, i) => (
                <span key={i}>
                  {i > 0 && ", "}
                  <code>{t.token}</code>
                </span>
              ))}
              . That is arithmetic on the checkpoint and did not use your text —
              the two halves can disagree, and a head wired to promote a token it
              never got to read here is exactly what that separation is for.
            </div>
          )}

          {/* The leg that was NOT run, named. Omitting it silently is what
              turns a two-legged answer into a claim it cannot support. */}
          {out && !out.causal.available && (
            <div className="hint">
              {out.causal.why} <code>{out.causal.how}</code>
            </div>
          )}
        </div>
      )}

      {stale && (
        <div className="meta">
          that reading was taken on another head — read it again for this one
        </div>
      )}
    </Disclosure>
    </div>
  );
}
