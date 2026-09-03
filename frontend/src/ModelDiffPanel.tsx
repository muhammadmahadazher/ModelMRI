// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

import { CSSProperties, useEffect, useState } from "react";
import { measured, percent, signed } from "./measured";
import { diffModels, errorText, ModelDiffReport } from "./api";
import ReceiptLine from "./ReceiptLine";
import { useScanOnData } from "./useScanOnData";

/**
 * What your finetune changed, over a PROMPT SET.
 *
 * The panel this sits beside compares one model against its own quantisation
 * on one prompt, and one prompt is a fair sample there because both sides are
 * the same weights. A finetune is not that: it changed the model on purpose,
 * in some places and not others, so one prompt's diff presented as a property
 * of the finetune is the error the whole feature refuses.
 *
 * Which is why this panel has no single headline number anywhere. Every
 * quantity is drawn as a MEDIAN WITH ITS MIDDLE HALF, and when the middle
 * half is wider than the median the summary says there is no single amount
 * the finetune moved the answer by and points at the per-prompt rows.
 *
 * Three things it draws that a diff view normally would not:
 *
 *   - The per-layer cosine as a band, not a line. The band is the spread
 *     across prompts; a line would be a median pretending to be a curve.
 *   - A PLURALITY marked as a plurality. When the divergence layer is first
 *     on fewer than half the prompts, the point of divergence moves between
 *     them and naming one layer is picking the commonest of several.
 *   - Both sides of every head, never the difference alone. A head that went
 *     from 0.02 to 0.06 and one that went from 4.00 to 4.04 moved by the same
 *     amount and are not the same finding.
 */

const PROMPTS_DEFAULT = [
  "The capital of France is",
  "Water boils at a temperature of",
  "The largest planet in the solar system is",
  "Photosynthesis converts sunlight into",
  "The author of Hamlet was",
  "A triangle has this many sides:",
].join("\n");

function lines(text: string): string[] {
  return text
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
}

export default function ModelDiffPanel({ epoch }: { epoch: number }) {
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [prompts, setPrompts] = useState(PROMPTS_DEFAULT);
  const [heads, setHeads] = useState(false);
  const [tokens, setTokens] = useState(false);
  const [data, setData] = useState<ModelDiffReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const scanRef = useScanOnData(data);

  // A comparison is about two OTHER models and survives anything happening to
  // the one loaded here — but the panel is remounted with the page, and a
  // result left over from a previous session would have no context on screen.
  useEffect(() => {
    setData(null);
    setErr("");
  }, [epoch]);

  const n = lines(prompts).length;

  async function run() {
    setBusy(true);
    setErr("");
    setData(null);
    try {
      setData(
        await diffModels({
          a: a.trim(),
          b: b.trim(),
          prompts: lines(prompts),
          include_heads: heads,
          include_tokens: tokens,
        }),
      );
    } catch (e) {
      // The refusals carry the feature's terms: a mismatched pair names both
      // sides and both numbers, and one prompt is refused with the reason a
      // spread cannot be taken over it.
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  const stable =
    data?.kl != null &&
    (data.kl.median === 0
      ? data.kl.high === 0
      : data.kl.high - data.kl.low <= Math.abs(data.kl.median) * 0.5);
  const majority = (data?.consensus_share ?? 0) >= 0.5;
  // The cosine band is drawn against its own floor rather than against 0:
  // every value sits between about 0.97 and 1.0 on a real pair, and a 0-1
  // axis would draw a flat line across the top of the chart.
  const floor = data?.layers.length
    ? Math.min(...data.layers.map((l) => l.low))
    : 0;
  // IDENTICAL MODELS DEGENERATE THE AXIS. Every cosine is 1.0, so the floor
  // is 1.0 and the span is zero — and `(v - floor) / tiny` puts every band at
  // 0%, drawn as a sliver against the LEFT edge, which reads as a cosine of
  // nothing. The exact opposite of what was measured. Two streams that agree
  // perfectly belong at the right-hand end of the axis.
  const degenerate = 1 - floor < 1e-9;
  const span = Math.max(1e-9, 1 - floor);
  const place = (v: number) => (degenerate ? 100 : ((v - floor) / span) * 100);

  return (
    <div className="panel mdiff" ref={scanRef}>
      <div className="sect">
        <span className="dot d-mdiff" />
        <h2 className="h-mdiff">FINETUNE DIFF — WHAT YOUR TRAINING CHANGED</h2>
        <span className="rule" />
      </div>
      <p className="meta">
        Two model ids — a base and its finetune, or two checkpoints of your own
        run — and a set of prompts. Each side is loaded <b>once</b>, in
        sequence, because 8&nbsp;GB will not hold both and the models worth
        comparing are exactly the ones near that limit. Every number below is a
        median over your prompts with its middle half beside it.
      </p>

      <div className="mdiff-inputs">
        <label>
          <span className="meta">base — the model you started from</span>
          <input
            value={a}
            onChange={(e) => setA(e.target.value)}
            spellCheck={false}
            placeholder="an id, or a path to a checkpoint"
          />
        </label>
        <label>
          <span className="meta">finetune — the model you trained</span>
          <input
            value={b}
            onChange={(e) => setB(e.target.value)}
            spellCheck={false}
            placeholder="an id, or a path to a checkpoint"
          />
        </label>
      </div>
      <label className="mdiff-prompts">
        <span className="meta">
          the prompt set — one per line · {n} prompt{n === 1 ? "" : "s"}
        </span>
        <textarea
          value={prompts}
          onChange={(e) => setPrompts(e.target.value)}
          spellCheck={false}
          rows={6}
        />
      </label>

      <div className="row mdiff-opts">
        {/* Both cost far more than the comparison itself, so both are opted
            into and both say what they cost. */}
        <label className="meta">
          <input
            type="checkbox"
            checked={heads}
            onChange={(e) => setHeads(e.target.checked)}
          />{" "}
          rank every head on both sides
          <span className="mdiff-cost">
            n_layers × n_heads passes per prompt, per side
          </span>
        </label>
        <label className="meta">
          <input
            type="checkbox"
            checked={tokens}
            onChange={(e) => setTokens(e.target.checked)}
          />{" "}
          attribute every prompt token
          <span className="mdiff-cost">~31 passes per prompt, per side</span>
        </label>
      </div>

      <div className="row" style={{ margin: "10px 0" }}>
        <button
          className="cta"
          onClick={() => void run()}
          disabled={busy || !a.trim() || !b.trim() || n < 4}
        >
          {busy ? "Loading each side in turn…" : "Compare them"}
        </button>
        <span className="meta">
          {n < 4
            ? `${n} prompt${n === 1 ? "" : "s"} — this needs at least 4, because the whole output is a spread across them`
            : "neither model is held in memory while the other is loaded"}
        </span>
      </div>

      {err && <div className="hint err">{err}</div>}

      {data && (
        <>
          <div className="mdiff-head meta">
            <b>{data.model_a}</b> → <b>{data.model_b}</b> ·{" "}
            {data.n_prompts} prompts over {data.n_layers} layers ·{" "}
            {data.seconds}s
          </div>

          {data.kl && (
            <div className={`mdiff-verdict ${stable ? "ok" : "wide"}`}>
              The answers differ by a median{" "}
              <b>{measured(data.kl.median, 4)}</b> nats per position, middle half{" "}
              {measured(data.kl.low, 4)} to {measured(data.kl.high, 4)}.{" "}
              {stable ? (
                <>The prompts agree with each other about how much moved.</>
              ) : (
                <b>
                  The prompts disagree by more than the median, so there is no
                  single amount this finetune moved the answer by — read the
                  per-prompt rows.
                </b>
              )}
            </div>
          )}

          {/* The cosine, as a band. The band IS the finding when it is wide:
              a median line alone would draw a curve nobody measured. */}
          {/* The SAME grid template as a row below, so the two ends of the
              axis sit over the track they describe. As a flex row with a
              `.spacer` between them they landed at x=185 and x=354 while the
              track ran 239 to 810 — labels describing an axis they were not
              over. `.spacer` is `flex: 0 1 auto` and does not grow. */}
          <div className="mdiff-scale meta" aria-hidden="true">
            <span />
            <span className="mdiff-axis">
              <span className="mdiff-axis-lo">
                {degenerate ? "identical at every layer" : floor.toFixed(6)}
              </span>
              <span className="mdiff-axis-hi">1.000000</span>
            </span>
            <span />
            <span>residual cosine</span>
          </div>
          <ol className="mdiff-layers stagger">
            {data.layers.map((row, i) => (
              <li
                key={row.layer}
                className={
                  row.layer === data.consensus_layer ? "mdiff-turn" : undefined
                }
                style={{ "--i": i } as CSSProperties}
              >
                <span className="mid mdiff-l">L{row.layer}</span>
                <span className="mdiff-track">
                  {/* Clamped inside the track. A layer at cosine 1.0 places
                      at 100% and still gets the 0.6% minimum width, so the
                      band began AT the right edge and ran past it — every row
                      overflowed its own track by 4px and the panel grew a
                      horizontal scrollbar. */}
                  <span
                    className="mdiff-band"
                    style={(() => {
                      const width = Math.min(
                        100,
                        Math.max(0.6, place(row.high) - place(row.low)),
                      );
                      return {
                        left: `${Math.min(place(row.low), 100 - width)}%`,
                        width: `${width}%`,
                      };
                    })()}
                    title={`middle half ${row.low.toFixed(6)} to ${row.high.toFixed(6)} over ${row.n} prompts`}
                  />
                  {/* The marker is 2px wide and centred on its position, so
                      at 100% it hangs 1px past the edge. Pulled in by its own
                      width at the extremes rather than letting the track
                      scroll. */}
                  <span
                    className="mdiff-median"
                    style={{ left: `min(${place(row.median)}%, 100% - 2px)` }}
                  />
                </span>
                <span className="mid mdiff-val">{row.median.toFixed(5)}</span>
                <span className="meta mdiff-first">
                  {row.n_first
                    ? `steepest fall on ${row.n_first}/${data.n_prompts}`
                    : ""}
                </span>
              </li>
            ))}
          </ol>

          <div className={`mdiff-verdict ${majority ? "ok" : "wide"}`}>
            {data.consensus_layer === null ? (
              <>
                <b>The cosine never falls</b> on any prompt: the two residual
                streams stay as aligned at the end as they were at the start.
                Whatever changed did not show up as a rotation of the stream on
                this prompt set.
              </>
            ) : (
              <>
                The cosine falls furthest at <b>layer {data.consensus_layer}</b>{" "}
                on {percent(data.consensus_share, 0)} of prompts.{" "}
                {majority ? (
                  <>That is a majority, so it is where this finetune starts to differ on this set.</>
                ) : (
                  <b>
                    That is a plurality and not a majority — the point of
                    divergence moves between your prompts, and naming one layer
                    would be picking the commonest of several.
                  </b>
                )}
              </>
            )}
          </div>

          {data.heads.length > 0 && (
            <>
              <p className="meta mdiff-sub">
                heads whose ablation score moved most · both sides shown,
                because a head that went from 0.02 to 0.06 and one that went
                from 4.00 to 4.04 moved by the same amount and are not the same
                finding
              </p>
              <ol className="mdiff-heads stagger">
                {/* Capped at ten, and the count follows below — a list that
                    stops without saying so reads as the whole finding. */}
                {data.heads.slice(0, 10).map((h, i) => (
                  <li key={`${h.layer}-${h.head}`} style={{ "--i": i } as CSSProperties}>
                    <span className="mid">
                      L{h.layer}H{h.head}
                    </span>
                    {/* Through `measured`, like the KL sentence at the top
                        of this panel. `toFixed(4)` floored any ablation score
                        under 5e-5 to "0.0000", which annihilates exactly the
                        both-sides comparison the caption argues for: a head
                        that went 2e-5 to 6e-5 read "0.0000 → 0.0001". */}
                    <span className="mid">{measured(h.median_a, 4)}</span>
                    <span className="meta">→</span>
                    <span className="mid">{measured(h.median_b, 4)}</span>
                    <span
                      className={`mid mdiff-shift ${h.shift > 0 ? "up" : "down"}`}
                    >
                      {/* `signed` is this expression exactly, and it keeps
                          the sign on a value small enough to escape to an
                          exponent — where `"+" + "0.0000"` read as "+0". */}
                      {signed(h.shift, 4)}
                    </span>
                    <span className="meta">
                      {h.top_a === h.top_b
                        ? `top-8 on ${h.top_a}/${data.n_prompts} both sides`
                        : `top-8 on ${h.top_a} → ${h.top_b} of ${data.n_prompts}`}
                    </span>
                  </li>
                ))}
              </ol>
              {data.heads.length > 10 && (
                <p className="meta">
                  {data.heads.length - 10} more head(s) were compared and moved
                  less. Every one was measured; this is the top ten.
                </p>
              )}
            </>
          )}

          {data.tokens.length > 0 && (
            <>
              <p className="meta mdiff-sub">
                prompt tokens that crossed a noise floor · each side against{" "}
                <b>its own</b> floor, because the two models have different
                ones and a shared threshold would compare one model's signal
                against the other's noise
              </p>
              <ol className="mdiff-tokens stagger">
                {data.tokens
                  .filter((t) => t.newly_used || t.newly_ignored)
                  .slice(0, 10)
                  .map((t, i) => (
                    <li
                      key={`${t.prompt_index}-${t.index}`}
                      className={t.newly_used ? "gained" : "lost"}
                      style={{ "--i": i } as CSSProperties}
                    >
                      <code>{t.token.trim() || "␣"}</code>
                      <span className="meta">
                        prompt {t.prompt_index + 1}, position {t.index}
                      </span>
                      {/* This list is filtered to tokens where one side
                          crossed its own noise floor and the other did not,
                          so by construction one of the two KLs sits just
                          below a floor — and `model_diff` rounds these to six
                          places, so every nonzero value under 5e-5 printed
                          "0.0000 → 0.0000" on the row claiming the token is
                          newly depended on. */}
                      <span className="mid">
                        {measured(t.kl_a, 4)} → {measured(t.kl_b, 4)}
                      </span>
                      <span className="meta mdiff-cross">
                        {t.newly_used ? "newly depended on" : "no longer used"}
                      </span>
                    </li>
                  ))}
              </ol>
            </>
          )}

          <p className="meta mdiff-means">{data.means}</p>
          <ReceiptLine receipt={data.receipt} />
        </>
      )}
    </div>
  );
}
