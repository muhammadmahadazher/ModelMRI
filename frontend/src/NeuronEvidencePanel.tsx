import { useEffect, useState } from "react";
import { errorText, getSession, NeuronEvidence, neuronEvidence } from "./api";
import { measured } from "./measured";

/** A string as CODE POINTS, which is the unit the route counts offsets
 *  in. `Array.from` is the one string split that respects surrogate
 *  pairs. */
const points = (t: string): string[] => Array.from(t);

/** What one MLP neuron fires on, across text the reader supplies.
 *
 *  For the models with no published sparse autoencoder — which is most of
 *  them. The feature panel beside this needs an SAE in the registry and
 *  answers nothing without one; this reads the neurons directly.
 *
 *  BLUNTER, AND IT SAYS SO. A neuron is polysemantic: it routinely responds
 *  to several unrelated things, and that is the entire reason sparse
 *  autoencoders exist. The route ships that caveat in the payload rather than
 *  in a tooltip, and it is rendered here at the top of the answer rather than
 *  the bottom, because a reader who scrolls off before reaching it has been
 *  handed a monosemantic story.
 *
 *  THE FIRING RATE NEEDS ITS LAYER BESIDE IT. "Fires on 29% of tokens" is
 *  meaningless alone: on a layer whose median neuron fires on 51% it is
 *  quiet, and on one whose median is 3% it is constantly on. Both numbers,
 *  always, on one line.
 */
export default function NeuronEvidencePanel({
  epoch,
  disabled,
}: {
  /** Bumped when the loaded model changes, so the layer dial follows it. */
  epoch: number;
  disabled?: boolean;
}) {
  // The layer count off the MODEL, not off `/api/attention/meta` — that only
  // exists after a generation, and reading a neuron needs none. Taking it
  // from there would leave the one dial this panel cannot work without dead
  // on a freshly loaded model, which is the bug `PatchscopePanel` records.
  const [nLayers, setNLayers] = useState<number | null>(null);
  useEffect(() => {
    let live = true;
    // The answer below is about the model that WAS loaded. Leaving it up
    // after a swap puts the old model's spans, firing rate and — worst —
    // its `layer_width` beside a neuron dial that now indexes a different
    // network.
    setData(null);
    setErr("");
    void getSession()
      .then((s) => {
        if (!live || !s?.model?.n_layers) return;
        setNLayers(s.model.n_layers);
        setLayer((l) => Math.min(l, s.model.n_layers! - 1));
      })
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [epoch]);

  const [text, setText] = useState(
    "The Eiffel Tower is in Paris.\nThe Colosseum is in Rome.\nBerlin is the capital of Germany.",
  );
  const [layer, setLayer] = useState(0);
  const [neuron, setNeuron] = useState(0);
  const [data, setData] = useState<NeuronEvidence | null>(null);
  // WHICH LAYER THIS ANSWER IS OF. The route publishes no `layer`, so the
  // panel has to remember what it asked for — reading the live dial instead
  // relabelled the histogram, the spans and "this layer's median" the moment
  // the dial moved, with no re-read.
  const [readLayer, setReadLayer] = useState(0);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const texts = text
    .split("\n")
    .map((t) => t.trim())
    .filter(Boolean);

  async function run() {
    if (busy || texts.length === 0) return;
    setBusy(true);
    setErr("");
    try {
      const got = await neuronEvidence({ texts, layer, neuron, top_k: 8 });
      setReadLayer(layer);
      setData(got);
    } catch (e) {
      setData(null);
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  const peak = data
    ? Math.max(...data.spans.map((s) => Math.abs(s.activation)), 0)
    : 0;
  const bins = data ? Math.max(...data.histogram, 0) : 0;

  return (
    <div className="neuron-evidence">
      <div className="sect sub">
        <span className="dot d-feat" />
        <h3>WHAT ONE NEURON FIRES ON</h3>
        <span className="rule" />
      </div>
      <p className="meta">
        For a model with no published sparse autoencoder, which is most of
        them. One line of text per sequence — nothing is downloaded, and the
        corpus is yours.
      </p>

      <textarea
        className="corpus-box"
        rows={4}
        value={text}
        onChange={(e) => setText(e.target.value)}
        spellCheck={false}
        disabled={disabled || busy}
        aria-label="corpus, one sequence per line"
      />

      <div className="row">
        <label className="meta" htmlFor="ne-layer">
          layer
        </label>
        <input
          id="ne-layer"
          type="range"
          min={0}
          max={Math.max(0, (nLayers ?? 1) - 1)}
          value={layer}
          onChange={(e) => setLayer(Number(e.target.value))}
          disabled={disabled || busy || !nLayers}
        />
        <span className="meta">
          {layer}
          {nLayers ? ` of ${nLayers - 1}` : ""}
        </span>
        <label className="meta" htmlFor="ne-neuron">
          neuron
        </label>
        <input
          id="ne-neuron"
          className="share-note ne-index"
          type="number"
          min={0}
          value={neuron}
          onChange={(e) => setNeuron(Math.max(0, Number(e.target.value) || 0))}
          disabled={disabled || busy}
        />
        {/* The width, from the answer rather than from a config guess. It is
            only known after a read, so it appears after one. */}
        {data && (
          <span className="meta">of {data.layer_width.toLocaleString()}</span>
        )}
        <button
          className="ghost sm"
          onClick={() => void run()}
          disabled={disabled || busy || texts.length === 0}
        >
          {busy ? "reading…" : data ? "read again" : `read ${texts.length} sequence${texts.length === 1 ? "" : "s"}`}
        </button>
      </div>

      {err && <div className="hint err">{err}</div>}

      {data && (
        <>
          {/* THE CAVEAT FIRST. A reader who scrolls off before reaching it has
              been handed a monosemantic story about a polysemantic unit. */}
          <div className="ne-caveat">{data.polysemantic}</div>

          <div className="row ne-chips">
            <span className="pill">
              neuron {data.neuron} · layer {readLayer}
            </span>
            <span className="pill">
              {/* Never one without the other. */}
              {data.firing_rate === null
                ? "firing rate not measured"
                : `fires on ${measured(data.firing_rate * 100, 1)}% of ${data.n_tokens} tokens`}
              {data.layer_median_firing_rate !== null && (
                <span className="meta">
                  {" "}
                  · this layer's median is{" "}
                  {measured(data.layer_median_firing_rate * 100, 1)}%
                </span>
              )}
            </span>
            {data.n_nonfinite > 0 && (
              <span className="meta warn">
                {data.n_nonfinite} of {data.n_tokens} activations were non-finite
                and left out rather than counted as zero
              </span>
            )}
            {data.n_fired === 0 && (
              <span className="meta warn">
                never fired on this text — which is a fact about the text as
                much as about the neuron
              </span>
            )}
          </div>

          {/* Both tails. A neuron's negative side is a real response, and a
              histogram cropped at zero would say it has none. */}
          {bins > 0 && (
            <div className="ne-hist" aria-hidden="true">
              {data.histogram.map((c, i) => (
                <i
                  key={i}
                  className={data.bin_edges[i] < 0 ? "neg" : "pos"}
                  style={{ height: `${((c / bins) * 100).toFixed(1)}%` }}
                  title={`${c} token(s) between ${measured(data.bin_edges[i], 3)} and ${measured(data.bin_edges[i + 1] ?? data.bin_edges[i], 3)}`}
                />
              ))}
            </div>
          )}
          {data.min_activation !== null && data.max_activation !== null && (
            <div className="row ne-range">
              <span className="meta">{measured(data.min_activation, 4)}</span>
              <span className="meta">
                {data.n_negative} below zero · {data.n_fired} above
              </span>
              <span className="meta">{measured(data.max_activation, 4)}</span>
            </div>
          )}

          <ul className="ne-spans">
            {data.spans.map((s, i) => (
              <li key={i}>
                <span className="ne-bar">
                  <i
                    style={{
                      width: `${peak > 0 ? ((Math.abs(s.activation) / peak) * 100).toFixed(1) : 0}%`,
                    }}
                  />
                </span>
                <b>{measured(s.activation, 4)}</b>
                {/* The offset, not a search. A window can hold the same token
                    twice and only one of them fired. */}
                {/* CODE POINTS, not UTF-16 units. `neurons.py` computes the
                    offset as `len("".join(tokens[:position]))` in Python,
                    where an emoji is one character; `String.slice` counts it
                    as two. One astral character earlier in the window shifted
                    the highlight by one per character, so the mark landed on
                    the wrong token — a wrong reading of which token fired. */}
                <span className="ne-line">
                  {points(s.text).slice(0, s.offset).join("")}
                  <mark>
                    {points(s.text)
                      .slice(s.offset, s.offset + points(s.token).length)
                      .join("")}
                  </mark>
                  {points(s.text)
                    .slice(s.offset + points(s.token).length)
                    .join("")}
                </span>
              </li>
            ))}
          </ul>
          {data.n_spans_shown < data.n_spans_available && (
            <p className="meta">
              {data.n_spans_shown} of {data.n_spans_available} firing positions
              shown — the cap is on this list, and every count above is over all
              of them.
            </p>
          )}

          <div className="hint">{data.means}</div>
        </>
      )}
    </div>
  );
}
