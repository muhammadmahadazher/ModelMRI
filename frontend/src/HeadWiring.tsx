import { useState } from "react";
import {
  errorText,
  getHeadOv,
  getHeadOvSpectrum,
  HeadOv,
  HeadOvSpectrum,
} from "./api";
import { measured } from "./measured";

/** What a head is WIRED to do, beside what it did on this generation.
 *
 *  Every other control in the attention panel answers a question about the
 *  current run and gives a different answer for the next one. This reads the
 *  head's own weights, so it is the same every time — and that difference is
 *  the whole reason it is a separate block with its own heading rather than
 *  another number in the ranking. A reader who does not know which kind of
 *  claim they are looking at will read one as the other.
 *
 *  Two readouts, and neither is a verdict:
 *
 *    what it writes   the tokens this head pushes toward when it attends to a
 *                     token you name. Exact weight arithmetic, no corpus.
 *    its spectrum     the fraction of sampled eigenvalues with a positive real
 *                     part. Near chance is what an ordinary head looks like,
 *                     and the panel says so rather than labelling anything.
 */
export default function HeadWiring({
  layer,
  head,
  disabled,
}: {
  layer: number;
  head: number;
  disabled?: boolean;
}) {
  const [token, setToken] = useState("");
  const [ov, setOv] = useState<HeadOv | null>(null);
  const [spectrum, setSpectrum] = useState<HeadOvSpectrum | null>(null);
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");

  async function readWriting() {
    if (!token.trim() || busy) return;
    setBusy("ov");
    setErr("");
    try {
      setOv(await getHeadOv(layer, head, token, 8));
    } catch (e) {
      setOv(null);
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }

  async function readSpectrum() {
    if (busy) return;
    setBusy("spectrum");
    setErr("");
    try {
      setSpectrum(await getHeadOvSpectrum(layer, head));
    } catch (e) {
      setSpectrum(null);
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }

  // Both readouts are of the head that was current when they were taken, so
  // moving the dial has to invalidate them rather than leave last head's
  // numbers under this head's label — the same staleness rule the heat map
  // above already follows.
  const stale = ov !== null && (ov.layer !== layer || ov.head !== head);
  const spectrumStale =
    spectrum !== null && (spectrum.layer !== layer || spectrum.head !== head);

  return (
    <div className="head-wiring">
      <div className="sect sub">
        <span className="dot d-attn" />
        <h3>WHAT THIS HEAD IS WIRED TO DO</h3>
        <span className="rule" />
      </div>
      <p className="meta">
        Read off the weights, not off this generation — so it is the same every
        time, and it is about head {head} rather than about what you just typed.
      </p>

      <div className="row">
        <label className="meta" htmlFor="ov-token">
          when it attends to
        </label>
        <input
          id="ov-token"
          className="share-note"
          placeholder="a token, e.g. Paris"
          value={token}
          onChange={(e) => setToken(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && void readWriting()}
          spellCheck={false}
          disabled={disabled}
        />
        <button
          className="ghost sm"
          onClick={() => void readWriting()}
          disabled={disabled || busy !== "" || !token.trim()}
        >
          {busy === "ov" ? "reading…" : "what does it write?"}
        </button>
        <button
          className="ghost sm"
          onClick={() => void readSpectrum()}
          disabled={disabled || busy !== ""}
          title="Eigenvalues of this head's OV circuit over a sample of the vocabulary"
        >
          {busy === "spectrum" ? "reading…" : "spectrum"}
        </button>
      </div>

      {err && <div className="hint err">{err}</div>}

      {ov && !stale && (
        <div className="ov-readout">
          <div className="row">
            <span className="pill">
              L{ov.layer} H{ov.head} · reads value head {ov.kv_head} of{" "}
              {ov.geometry.n_kv_heads}
            </span>
            {/* The grouping, said out loud. On Qwen3-1.7B two query heads share
                each value head, so "head 3" and "head 2" are not as separate as
                the dial suggests, and a reader comparing them deserves to know. */}
            {ov.geometry.group_size > 1 && (
              <span className="meta">
                {ov.geometry.group_size} query heads share each value head here
              </span>
            )}
            {/* Not a footnote. A head reads ONE token, so which one this is
                changes the answer entirely. */}
            {ov.source_token_count > 1 && (
              <span className="meta warn">
                “{token}” is {ov.source_token_count} tokens — this is the first
              </span>
            )}
          </div>
          <div className="ov-columns">
            <div>
              <span className="meta">pushes toward</span>
              <ol className="ov-list">
                {ov.promotes.map((t, i) => (
                  <li key={`p${i}`}>
                    <code>{t.token}</code>
                    <span className="meta">{measured(t.score, 2)}</span>
                  </li>
                ))}
              </ol>
            </div>
            <div>
              <span className="meta">pushes away from</span>
              <ol className="ov-list">
                {ov.suppresses.map((t, i) => (
                  <li key={`s${i}`}>
                    <code>{t.token}</code>
                    <span className="meta">{measured(t.score, 2)}</span>
                  </li>
                ))}
              </ol>
            </div>
          </div>
          <div className="hint">{ov.means}</div>
        </div>
      )}

      {spectrum && !spectrumStale && (
        <div className="ov-readout">
          <div className="row">
            <span className="pill">
              {spectrum.positive} of {spectrum.n_sampled} eigenvalues positive
            </span>
            <span className="meta">
              sampled from {spectrum.n_vocab.toLocaleString()} tokens at seed{" "}
              {spectrum.seed}
            </span>
            {spectrum.sample_capped && (
              <span className="meta warn">
                the vocabulary is smaller than the sample asked for
              </span>
            )}
          </div>
          {/* The imaginary mass is not decoration either: past about half, the
              sign of an eigenvalue is describing rotation as much as direction,
              and "positive fraction" means less than it looks like. */}
          {spectrum.imaginary_mass > 0.5 && (
            <div className="meta">
              {measured(spectrum.imaginary_mass * 100, 0)}% of the spectrum sits
              off the real line, so this fraction describes rotation as much as
              sign.
            </div>
          )}
          <div className="hint">{spectrum.means}</div>
        </div>
      )}

      {(stale || spectrumStale) && (
        <div className="meta">
          the readouts above were taken on another head — read them again for
          this one
        </div>
      )}
    </div>
  );
}
