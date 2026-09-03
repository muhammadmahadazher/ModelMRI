// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

import { useEffect, useState } from "react";
import RunsOn, { useModelIdentity, useModelReady } from "./RunsOn";
import { errorText, getSession, Patchscope, runPatchscope } from "./api";
import ReceiptLine from "./ReceiptLine";
import { useScanOnData } from "./useScanOnData";

/**
 * Ask the model to describe one of its own hidden states, in words.
 *
 * ITS OWN SURFACE, not a column beside the logit lens. The lens reads a state
 * through the unembedding and reports tokens; this hands the state to the
 * model inside a second prompt and reports a SENTENCE. Rendered side by side
 * they would read as two measurements of one thing, and they are not.
 *
 * The method's known failure is that a good target prompt describes ANYTHING
 * fluently — hand it a random vector and it still produces a confident
 * sentence. So this panel never shows a decode alone. The two controls sit
 * beside it at the same size, in the same typeface:
 *
 *   identity   the target prompt with its own activation, untouched
 *   random     the target prompt with a same-norm random vector
 *
 * A decode matching either is the TARGET PROMPT TALKING. And because a decode
 * can differ from a control as a string while using none of its own words, the
 * vocabulary overlap is printed as a number rather than left to a string
 * comparison.
 */

const SOURCE_DEFAULT = "The Eiffel Tower is located in the city of Paris";

/** How much of the DECODE's vocabulary this control already used, as a phrase
 *  rather than a bare ratio. 1.0 means the decode said nothing new, which is
 *  the failure this whole panel exists to catch.
 *
 *  The subject is named in every branch. "every word of it was already in the
 *  control" sits under a column headed "control" and reads, at a glance, as a
 *  statement about the control rather than about the decode — which inverts
 *  the finding. */
function overlapText(v: number): string {
  const pct = Math.round(v * 100);
  if (v >= 1) return "the decode said nothing this control had not";
  if (v <= 0) return "the decode shares no word with this control";
  return `${pct}% of the decode's words were already here`;
}

export default function PatchscopePanel({ epoch }: { epoch: number }) {
  // Nothing loaded means every button here can only be refused. Shares
  // `RunsOn`'s cached session, so the badge and the control it disables
  // read one answer rather than two requests that can disagree.
  const ready = useModelReady(epoch);
  const model = useModelIdentity(epoch);
  const [prompt, setPrompt] = useState(SOURCE_DEFAULT);
  const [layer, setLayer] = useState(0);
  const [position, setPosition] = useState(-1);
  const [targetLayer, setTargetLayer] = useState<number | null>(null);
  // Empty until the first run, then filled with the target the SERVER used.
  // Not a copy of the default kept over here: a duplicated constant drifts,
  // and the one number on the page that is allowed to be wrong is none of
  // them. The placeholder says what will happen.
  const [target, setTarget] = useState("");
  const [tokens, setTokens] = useState(12);
  const [layers, setLayers] = useState(0);
  const [data, setData] = useState<Patchscope | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const scanRef = useScanOnData(data);

  // From the SESSION, not from `/api/attention/meta`. Attention metadata only
  // exists after a generation, and a patchscope needs none — it runs its own
  // source prompt. Reading the layer count from there left this dropdown empty
  // on a freshly loaded model, so the one control the panel cannot work
  // without was dead until the user generated something they did not need.
  useEffect(() => {
    let live = true;
    void (async () => {
      const s = await getSession().catch(() => null);
      if (!live || !s?.model?.n_layers) return;
      setLayers(s.model.n_layers);
      setLayer(Math.floor(s.model.n_layers / 2));
    })();
    return () => {
      live = false;
    };
  }, [epoch, model]);

  useEffect(() => {
    setData(null);
    setErr("");
  }, [epoch, model]);

  async function run() {
    setBusy(true);
    setErr("");
    setData(null);
    try {
      const out = await runPatchscope({
        prompt,
        layer,
        position,
        target: target.trim(),
        target_layer: targetLayer,
        max_new_tokens: tokens,
      });
      setData(out);
      // The field now holds what was actually used, so editing it edits a
      // real thing rather than an assumption about one.
      setTarget(out.target.prompt);
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  const layerOptions = Array.from({ length: layers }, (_, i) => (
    <option key={i} value={i}>
      {i}
    </option>
  ));

  return (
    <div className="panel scope" ref={scanRef}>
      <div className="sect">
        <span className="dot d-scope" />
        <h2 className="h-scope">PATCHSCOPE — THE STATE, DESCRIBED IN WORDS</h2>
        <span className="rule" />
      </div>
      <RunsOn epoch={epoch} />
      <p className="meta">
        Take a hidden state from one run and splice it into a second prompt
        built to make the model describe whatever it is holding. Everything
        else on this page asks the model a question in numbers; this asks it in
        words — and words are the easiest thing on the page to over-read, which
        is why a decode never appears here without both controls beside it.
      </p>

      <div className="scope-inputs">
        <label className="scope-wide">
          <span className="meta">source — the prompt whose state is read</span>
          <input
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            spellCheck={false}
          />
        </label>
      </div>

      <div className="row scope-controls">
        <label className="meta" htmlFor="scope-layer">
          read at layer
        </label>
        {/* A dropdown when the block count is known, a number field when it
            is not — an exotic config need not publish `num_hidden_layers`,
            and a permanently empty select is a control that cannot be used.
            The server refuses an out-of-range layer by name either way. */}
        {layers ? (
          <select
            id="scope-layer"
            value={layer}
            onChange={(e) => setLayer(Number(e.target.value))}
          >
            {layerOptions}
          </select>
        ) : (
          <input
            id="scope-layer"
            className="scope-num"
            type="number"
            min={0}
            value={layer}
            onChange={(e) => setLayer(Number(e.target.value))}
          />
        )}
        <label className="meta" htmlFor="scope-pos">
          position
        </label>
        <input
          id="scope-pos"
          className="scope-num"
          type="number"
          value={position}
          onChange={(e) => setPosition(Number(e.target.value))}
        />
        <label className="meta" htmlFor="scope-tlayer">
          splice into layer
        </label>
        <select
          id="scope-tlayer"
          value={targetLayer === null ? "same" : targetLayer}
          onChange={(e) =>
            setTargetLayer(
              e.target.value === "same" ? null : Number(e.target.value),
            )
          }
        >
          <option value="same">the same one</option>
          {layerOptions}
        </select>
        <label className="meta" htmlFor="scope-tok">
          decode
        </label>
        <input
          id="scope-tok"
          className="scope-num"
          type="number"
          min={1}
          max={64}
          value={tokens}
          onChange={(e) => setTokens(Number(e.target.value))}
        />
        <span className="meta">tokens, greedily</span>
      </div>

      {/* The target prompt is an INPUT and a RESULT. It is shown at full size
          rather than behind a disclosure, because a decode taken under a
          different target is a different experiment and the reader has to be
          able to see which one they are looking at. */}
      <label className="scope-target">
        <span className="meta">
          target — the prompt the state is spliced into
          {data && data.target.tokens.length
            ? ` · ${data.target.tokens.length} tokens, spliced at ${data.target.position}`
            : ""}
        </span>
        <textarea
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          spellCheck={false}
          rows={3}
          placeholder={
            "left empty, the server's identity target is used — and the exact " +
            "text it used appears here after the run"
          }
        />
      </label>

      <div className="row" style={{ margin: "10px 0" }}>
        <button
          className="cta"
          onClick={() => void run()}
          disabled={busy || ready === false}
          title={
            ready === false
              ? "Load a model in Run at the top of the page first — this measurement runs it."
              : undefined
          }
        >
          {busy ? "Decoding three times…" : "Describe the state"}
        </button>
        <span className="meta">
          three greedy decodes — the patched one and both controls
        </span>
      </div>

      {err && <div className="hint err">{err}</div>}

      {data && (
        <>
          <div
            className={`scope-verdict ${data.informative ? "informative" : "flat"}`}
          >
            {data.informative ? (
              <>
                The decode differs from <b>both</b> controls, so it is at least
                responding to what was patched.
              </>
            ) : (
              <>
                <b>This decode is not about the state.</b>{" "}
                {data.same_as_identity
                  ? "It is identical to the untouched target prompt — the patch changed nothing."
                  : data.same_as_random
                    ? "It is identical to what a same-norm random vector produced — the target prompt says this whatever it is handed."
                    : "It says nothing its controls did not already say."}
              </>
            )}
          </div>

          {/* Three columns, one size. The decode is not the headline and the
              controls are not footnotes — that layout is the claim. */}
          <div className="scope-decodes">
            <div className="scope-decode patched">
              <span className="meta">
                patched — layer {data.source.layer}, position{" "}
                {data.source.position}
                {data.source.tokens[data.source.position]
                  ? ` (${data.source.tokens[data.source.position].trim() || "␣"})`
                  : ""}
              </span>
              <p>{data.decode || <em className="meta">nothing at all</em>}</p>
            </div>
            <div className="scope-decode">
              <span className="meta">
                control — the target untouched
                {data.same_as_identity ? " · IDENTICAL" : ""}
              </span>
              <p>{data.controls.identity}</p>
              <span className="meta scope-ov">
                {overlapText(data.overlap_identity)}
              </span>
            </div>
            <div className="scope-decode">
              <span className="meta">
                control — a random vector, same norm ({data.source.norm})
                {data.same_as_random ? " · IDENTICAL" : ""}
              </span>
              {data.controls.random.map((r, i) => (
                <p key={i}>{r}</p>
              ))}
              <span className="meta scope-ov">
                {overlapText(data.overlap_random)}
              </span>
            </div>
          </div>

          {data.cross_layer && (
            <p className="meta scope-warn">
              Layer {data.source.layer} into layer {data.target.layer}: the two
              streams are only comparable where the model treats them alike,
              and nothing here checks that they do.
            </p>
          )}

          <p className="meta scope-means">{data.means}</p>
          <ReceiptLine receipt={data.receipt} />
        </>
      )}
    </div>
  );
}
