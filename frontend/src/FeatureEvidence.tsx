import { CSSProperties, useEffect, useState } from "react";
import { errorText, FeatureEvidence, featureEvidence } from "./api";
import ReceiptLine from "./ReceiptLine";

/**
 * What one feature fires on — IN YOUR TEXT — and what it promotes.
 *
 * The dashboards this competes with show features from a model and an SAE
 * somebody else chose, on text somebody else picked, under a natural-language
 * label a third model wrote. Three layers of somebody else's choices between
 * you and the claim. Here:
 *
 *   - The corpus is yours and NOTHING IS DOWNLOADED. Its name and token count
 *     travel with every number, because a top activation is a top activation
 *     in this text and nowhere else.
 *   - The share of features that NEVER FIRED is printed. A feature that did
 *     not fire is not seen in your corpus, which is a fact about the corpus;
 *     calling it dead would be a claim about the model.
 *   - There is NO LABEL. The server returns `label: null` deliberately and
 *     this renders that as a sentence, because a generated label would be the
 *     one thing on the page that nothing measured.
 *
 * Two readouts sit side by side and they are different KINDS of number. The
 * spans are a sample: change the corpus and they change. The logit weights are
 * weight arithmetic — no corpus, no sampling, the same every time.
 */

const CORPUS_HINT =
  "one sequence per line — your own text, nothing is downloaded";

export default function FeatureEvidencePanel({
  feature,
  epoch,
}: {
  feature: number;
  epoch: number;
}) {
  const [text, setText] = useState("");
  const [file, setFile] = useState("");
  const [data, setData] = useState<FeatureEvidence | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  // A sweep is about one feature of one SAE on one corpus. Changing the
  // feature makes the spans below belong to a different one, and the epoch
  // moves when the model or SAE changes underneath.
  useEffect(() => {
    setData(null);
    setErr("");
  }, [feature, epoch]);

  const lines = text
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);

  async function run() {
    setBusy(true);
    setErr("");
    setData(null);
    try {
      setData(
        await featureEvidence(
          file.trim()
            ? { file: file.trim(), feature }
            : { texts: lines, label: "pasted text", feature },
        ),
      );
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  const ev = data?.evidence;
  const lw = data?.logit_weights;
  const peak = ev ? Math.max(...ev.histogram, 1) : 1;

  return (
    <div className="feat-evidence">
      <div className="row">
        <span className="meta">
          evidence for <b>#{feature}</b> — what it fires on in your text
        </span>
      </div>

      <div className="fe-inputs">
        <label>
          <span className="meta">{CORPUS_HINT}</span>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            spellCheck={false}
            rows={5}
            disabled={!!file.trim()}
            placeholder={
              "The lawyer cited Brown v. Board of Education in her brief.\n" +
              "The train leaves the station every twenty minutes."
            }
          />
        </label>
        <label>
          <span className="meta">…or a local .txt / .jsonl</span>
          {/* A RELATIVE example. An absolute one names somebody's machine —
              the drive letter that was here was mine, and the leak test that
              guards every shipped file caught it rather than review. */}
          <input
            value={file}
            onChange={(e) => setFile(e.target.value)}
            spellCheck={false}
            placeholder="notes.txt — or a path to one"
          />
        </label>
      </div>

      <div className="row" style={{ margin: "8px 0" }}>
        <button
          className="cta"
          onClick={() => void run()}
          disabled={busy || (!lines.length && !file.trim())}
        >
          {busy ? "Sweeping your corpus…" : "Sweep for evidence"}
        </button>
        <span className="meta">
          {file.trim()
            ? "read from disk, never sent anywhere"
            : `${lines.length} sequence${lines.length === 1 ? "" : "s"}`}
        </span>
      </div>

      {err && <div className="hint err">{err}</div>}

      {data && (
        <>
          {/* The corpus, attached to the numbers rather than mentioned near
              them. Every claim below is scoped by this line. */}
          <p className="meta fe-corpus">
            <b>{data.corpus.corpus_label}</b> · {data.corpus.n_tokens} tokens
            over {data.corpus.n_sequences} sequences ·{" "}
            {(data.corpus.never_fired_share * 100).toFixed(1)}% of this SAE's{" "}
            {data.corpus.n_features.toLocaleString()} features never fired here
            — not seen in this corpus, not dead.
            {data.corpus.truncated && (
              <b> The sweep was cut short, so part of your text was not read.</b>
            )}
          </p>

          {ev && (
            <>
              <div className="fe-head">
                <span className="mid">
                  fired on {ev.n_fired} of {ev.n_tokens} tokens (
                  {(ev.firing_rate * 100).toFixed(1)}%) · peak{" "}
                  {ev.max_activation.toFixed(2)}
                </span>
                {!ev.selective && (
                  <span className="fe-warn">
                    NOT A CONCEPT — it fires on too much of your text for its
                    top spans to be read as one
                  </span>
                )}
              </div>

              {ev.spans.length > 0 ? (
                <ol className="fe-spans stagger">
                  {ev.spans.map((s, i) => (
                    <li key={i} style={{ "--i": i } as CSSProperties}>
                      <span className="mid fe-act">
                        {s.activation.toFixed(2)}
                      </span>
                      {/* The firing token marked at ITS position, from the
                          offset the server measured — not by searching the
                          span for the word. "The appeals court disagreed with
                          the trial court's" contains "court" twice and only
                          one of them fired; splitting on the string lit both
                          and claimed two firings that were not there. */}
                      <span className="fe-text">
                        {s.text.slice(0, s.offset)}
                        <mark className="fe-tok">
                          {s.text.slice(s.offset, s.offset + s.token.length)}
                        </mark>
                        {s.text.slice(s.offset + s.token.length)}
                      </span>
                    </li>
                  ))}
                </ol>
              ) : (
                <p className="meta">
                  It never fired in this text. That is a fact about the corpus
                  you handed it, not about the feature.
                </p>
              )}

              {/* A histogram of ONE firing is not a distribution, it is the
                  span already printed above drawn as a rectangle. Measured on
                  gpt2 layer 8: feature 5302 fired once in 98 tokens and the
                  chart was 19 empty bins and a block. Saying so is shorter
                  and truer than drawing it. */}
              {ev.n_fired > 1 ? (
                <>
                  <div className="fe-hist" aria-label="activation distribution">
                    {ev.histogram.map((n, i) => (
                      <span
                        key={i}
                        style={{ height: `${(n / peak) * 100}%` }}
                        title={`${n} firings between ${ev.bin_edges[i].toFixed(2)} and ${ev.bin_edges[i + 1].toFixed(2)}`}
                      />
                    ))}
                  </div>
                  {/* A SIBLING, not a child. Absolutely positioning this
                      inside the flex row laid it across the bars it was
                      describing and ran off the right edge. */}
                  <p className="meta fe-hist-cap">
                    all {ev.n_fired} firings, from{" "}
                    {ev.bin_edges[0].toFixed(2)} to{" "}
                    {ev.bin_edges[ev.bin_edges.length - 1].toFixed(2)} — the{" "}
                    {ev.spans.length} span{ev.spans.length === 1 ? "" : "s"}{" "}
                    above are its right tail, not the whole of it.
                  </p>
                </>
              ) : (
                ev.n_fired === 1 && (
                  <p className="meta fe-hist-cap">
                    One firing in the whole corpus, so there is no distribution
                    to draw — the span above is all of it. Hand it more text
                    before reading anything into the shape.
                  </p>
                )
              )}

              {/* Rendered as a sentence rather than left blank. A missing
                  label reads as an omission; a stated refusal reads as the
                  position it is. */}
              <p className="meta fe-nolabel">
                No name is generated for this feature. What it responds to is
                the reader's call from the evidence on this page — a label
                would be the one thing here that nothing measured.
              </p>
            </>
          )}

          {lw && (
            <div className="fe-logits">
              <div className="fe-col">
                <span className="meta">promotes</span>
                <ol>
                  {lw.promotes.map((t, i) => (
                    <li key={i}>
                      <code>{t.token.trim() || "␣"}</code>
                      <span className="mid">{t.logit.toFixed(2)}</span>
                    </li>
                  ))}
                </ol>
              </div>
              <div className="fe-col">
                <span className="meta">suppresses</span>
                <ol>
                  {lw.suppresses.map((t, i) => (
                    <li key={i}>
                      <code>{t.token.trim() || "␣"}</code>
                      <span className="mid">{t.logit.toFixed(2)}</span>
                    </li>
                  ))}
                </ol>
              </div>
              {/* The one exact number on this page, and it says so. The spans
                  above are a sample of your corpus; this is weight
                  arithmetic and does not move between runs. */}
              <p className="meta fe-exact">
                No corpus and no sampling — this is the feature's decoder
                direction read through the unembedding, identical on every run.
                It ranks tokens rather than predicting logit amounts.
              </p>
            </div>
          )}

          <ReceiptLine receipt={data.receipt} />
        </>
      )}
    </div>
  );
}
