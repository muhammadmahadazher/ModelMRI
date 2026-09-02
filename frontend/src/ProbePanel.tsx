import { CSSProperties, useEffect, useState } from "react";
import { percent } from "./measured";
import RunsOn, { useModelIdentity, useModelReady } from "./RunsOn";
import { errorText, LayerProbe, ProbeReport, runProbe } from "./api";
import ReceiptLine from "./ReceiptLine";
import { useScanOnData } from "./useScanOnData";

/**
 * A linear probe at every layer — drawn against the two things that make an
 * accuracy mean anything.
 *
 * Every probing dashboard shows the curve. The curve is the easy part, and on
 * its own it is close to worthless: 0.83 at layer 7 sounds like a finding
 * until you learn that fitting the SAME probe to SHUFFLED labels reaches 0.81,
 * or that always answering with the commoner class gets 0.80. So this panel is
 * built around the two references rather than around the line:
 *
 *   - the PERMUTATION NULL, shaded behind every bar. A bar inside it did not
 *     beat chance for this fit at this many examples.
 *   - the MAJORITY line, a vertical mark. A bar left of it is beaten by
 *     guessing the commoner class.
 *
 * And two things it says that a curve cannot:
 *
 *   - NULL SATURATION. When the shuffled fit reaches the top of the scale, no
 *     accuracy could have cleared it — the layer is untestable at this sample
 *     size, which is different from uninformative, and drawing it as "inside
 *     the null" would quietly call it a negative result.
 *   - THE MULTIPLE COMPARISON. Sweeping N layers against a 95th-percentile
 *     band gives 0.05·N layers clearing by chance. With 12 layers that is
 *     0.6 — so ONE clearing layer is what noise looks like, and the panel says
 *     the expected number beside the observed one instead of letting the
 *     reader count peaks.
 */

/** A starting pair that CAN actually run.
 *
 *  Sized off the server's floor rather than off what looks tidy: a quarter of
 *  the examples are held out and the fit refuses below 12 held-out examples,
 *  because an accuracy measured on 8 has a resolution of 12 percentage points.
 *  Two classes of 12 gave 8 and the panel's own default hit its own refusal —
 *  a default that can only ever be rejected is a broken default, however
 *  correct the message.
 *
 *  Both classes are the same length, so the majority line lands at 0.5 and is
 *  visible from the first run instead of hugging an edge. */
const A_DEFAULT = [
  "The film was a masterpiece from start to finish.",
  "I loved every minute of it.",
  "An absolute delight — I would watch it again tomorrow.",
  "Warm, funny and beautifully made.",
  "The best thing I have seen all year.",
  "It left me grinning the whole way home.",
  "Wonderful performances and a perfect ending.",
  "Genuinely moving and very well judged.",
  "A joy throughout.",
  "Sharp, clever and completely charming.",
  "I recommend it without reservation.",
  "Everything about it worked.",
  "The pacing was flawless and the score lifted every scene.",
  "I came out of the cinema lighter than I went in.",
  "Beautifully shot, and it earns its ending.",
  "Every character felt like somebody real.",
  "It is generous, and it never once talks down to you.",
  "I have thought about the last ten minutes all week.",
  "The script is quietly brilliant.",
  "A rare thing — it gets better as it goes.",
  "I would happily sit through it a third time.",
  "Confident, tender and completely sure of itself.",
  "The performances carry it and then some.",
  "It made the whole room laugh at the same moment.",
  "Small, careful and much bigger than it looks.",
  "One of the best I have seen in a long time.",
].join("\n");

const B_DEFAULT = [
  "The film was a waste of two hours.",
  "I hated every minute of it.",
  "An absolute chore — I would not sit through it again.",
  "Cold, humourless and badly made.",
  "The worst thing I have seen all year.",
  "It left me checking my watch the whole way through.",
  "Wooden performances and a pointless ending.",
  "Genuinely tedious and very poorly judged.",
  "A slog throughout.",
  "Blunt, obvious and completely charmless.",
  "I would not recommend it to anyone.",
  "Nothing about it worked.",
  "The pacing dragged and the score buried every scene.",
  "I came out of the cinema heavier than I went in.",
  "Ugly to look at, and it fumbles its ending.",
  "Not one character felt like anybody real.",
  "It is mean, and it talks down to you throughout.",
  "I had forgotten the last ten minutes by the car park.",
  "The script is quietly awful.",
  "A rare thing — it gets worse as it goes.",
  "I would not sit through it a second time.",
  "Timid, cold and completely unsure of itself.",
  "The performances sink it and then some.",
  "It made the whole room check their phones at once.",
  "Loud, careless and much smaller than it looks.",
  "One of the worst I have seen in a long time.",
].join("\n");

function lines(text: string): string[] {
  return text
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
}

/** The verdict for one layer, in the order that decides what is drawn.
 *
 *  Saturation comes FIRST. A saturated layer is also inside the null — every
 *  accuracy is — and reporting it as "inside the null" would present an
 *  untestable layer as a measured negative. */
function verdict(p: LayerProbe): { text: string; cls: string } {
  if (p.null_saturated)
    return { text: "untestable here", cls: "probe-untestable" };
  if (p.inside_null) return { text: "inside the null", cls: "probe-null" };
  if (!p.beats_majority)
    return { text: "under the majority class", cls: "probe-null" };
  return { text: "clears both", cls: "probe-clears" };
}

export default function ProbePanel({ epoch }: { epoch: number }) {
  // Nothing loaded means every button here can only be refused. Shares
  // `RunsOn`'s cached session, so the badge and the control it disables
  // read one answer rather than two requests that can disagree.
  const ready = useModelReady(epoch);
  const model = useModelIdentity(epoch);
  const [a, setA] = useState(A_DEFAULT);
  const [b, setB] = useState(B_DEFAULT);
  const [name, setName] = useState("");
  const [data, setData] = useState<ProbeReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const scanRef = useScanOnData(data);

  const nA = lines(a).length;
  const nB = lines(b).length;

  async function run() {
    setBusy(true);
    setErr("");
    setData(null);
    try {
      const examples = [
        ...lines(a).map((text) => ({ text, label: 0 })),
        ...lines(b).map((text) => ({ text, label: 1 })),
      ];
      setData(await runProbe({ examples, save_as: name.trim() }));
    } catch (e) {
      // The refusals ARE the panel on a small set: "8 per class" and "12 in
      // the held-out half" are the difference between a probe and a number.
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  // A probe is fitted to YOUR examples and does not depend on the current
  // generation — but it does depend on the loaded model, and the epoch moves
  // on load and unload. Keeping a curve across that would put Qwen3-1.7B's
  // layers under a different model's name.
  useEffect(() => {
    setData(null);
    setErr("");
  }, [epoch, model]);

  return (
    <div className="panel probe" ref={scanRef}>
      <div className="sect">
        <span className="dot d-probe" />
        <h2 className="h-probe">PROBES — WHERE A PROPERTY BECOMES READABLE</h2>
        <span className="rule" />
      </div>
      <RunsOn epoch={epoch} />
      <p className="meta">
        Two groups of your own sentences, and a linear fit at every layer that
        tries to tell them apart from the residual stream. The accuracy is the
        easy part. What this draws behind it is the same fit on{" "}
        <b>shuffled labels</b> — because a probe that cannot beat its own
        shuffled null found nothing, however high the number looks.
      </p>

      <div className="probe-inputs">
        <label>
          <span className="meta">
            group A — one per line · {nA} example{nA === 1 ? "" : "s"}
          </span>
          <textarea
            value={a}
            onChange={(e) => setA(e.target.value)}
            spellCheck={false}
            rows={7}
          />
        </label>
        <label>
          <span className="meta">
            group B — one per line · {nB} example{nB === 1 ? "" : "s"}
          </span>
          <textarea
            value={b}
            onChange={(e) => setB(e.target.value)}
            spellCheck={false}
            rows={7}
          />
        </label>
      </div>

      <div className="row" style={{ margin: "10px 0" }}>
        <button
          className="cta"
          onClick={() => void run()}
          disabled={busy || !nA || !nB || ready === false}
        >
          {busy ? "Fitting every layer…" : "Fit the probe"}
        </button>
        <label className="meta" htmlFor="probe-save">
          save the best layer's direction as
        </label>
        <input
          id="probe-save"
          className="probe-name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="optional — steerable name"
          spellCheck={false}
        />
      </div>

      {err && <div className="hint err">{err}</div>}

      {data && (
        <>
          {/* The headline is the VERDICT, not the peak. `best_layer` is null
              when nothing cleared anywhere, and that is a result — a panel
              that always names a best layer would name one here too. */}
          <div className={`probe-verdict ${data.best_layer === null ? "none" : ""}`}>
            {data.best_layer === null ? (
              <>
                <b>Nothing cleared.</b> No layer beat both its own shuffled
                null and the majority class, so this property was not linearly
                readable anywhere in this model at this sample size.
              </>
            ) : (
              <>
                Readable from <b>layer {data.best_layer}</b> ·{" "}
                {data.n_readable_layers} of {data.layers.length} layers cleared
                both references
              </>
            )}
          </div>

          {/* The multiple comparison, in the same breath as the count. */}
          <p className="meta probe-fp">
            {data.layers.length} layers were each tested against a{" "}
            {data.n_permutations}-permutation null, so about{" "}
            <b>{data.expected_false_positives.toFixed(1)}</b> would clear by
            chance alone.{" "}
            {data.n_readable_layers <= data.expected_false_positives ? (
              <b>
                That is at or above what cleared, so this curve is what noise
                looks like.
              </b>
            ) : (
              <>
                {data.n_readable_layers} cleared, and a run of adjacent layers
                is harder to get by chance than the same count scattered.
              </>
            )}
            {data.n_underpowered_layers > 0 && (
              <>
                {" "}
                {data.n_underpowered_layers} layer
                {data.n_underpowered_layers === 1 ? " is" : "s are"} untestable:
                the shuffled fit reached the top of the scale there, so no
                accuracy could have cleared it.
              </>
            )}
          </p>

          {/* Same grid template as a row below, so the axis sits over the
              track it describes rather than over the whole panel. */}
          <div className="probe-scale meta" aria-hidden="true">
            <span />
            <span className="probe-axis">
              <span className="probe-axis-lo">0</span>
              <span
                className="probe-majority-key"
                style={{ left: `${data.majority * 100}%` }}
              >
                majority {percent(data.majority, 0)}
                {data.counts && (
                  <span className="meta">
                    {" "}
                    ({Object.entries(data.counts)
                      .map(([k, v]) => `${v} of class ${k}`)
                      .join(", ")})
                  </span>
                )}
              </span>
              <span className="probe-axis-hi">100%</span>
            </span>
          </div>

          <ol className="probe-rows stagger">
            {[...data.layers]
              .reverse()
              .map((p, ri) => {
                const v = verdict(p);
                return (
                  <li
                    key={p.layer}
                    className={v.cls}
                    style={{ "--i": ri } as CSSProperties}
                  >
                    <span className="mid probe-l">L{p.layer}</span>
                    <span className="probe-track">
                      {/* The null band, drawn FIRST and behind — it is the
                          thing the bar has to clear, so it is scenery the bar
                          sits in rather than an annotation beside it. */}
                      <span
                        className="probe-band"
                        style={{
                          left: `${p.null_low * 100}%`,
                          width: `${Math.max(0, p.null_high - p.null_low) * 100}%`,
                        }}
                        title={`shuffled labels reached ${(p.null_low * 100).toFixed(0)}–${percent(p.null_high, 0)}`}
                      />
                      <span
                        className="probe-majority"
                        style={{ left: `${data.majority * 100}%` }}
                      />
                      <span
                        className="probe-bar"
                        style={{ width: `${p.accuracy * 100}%` }}
                      />
                    </span>
                    <span className="mid probe-acc">
                      {percent(p.accuracy, 0)}
                    </span>
                    <span className="meta probe-verd">{v.text}</span>
                  </li>
                );
              })}
          </ol>

          <p className="meta">
            {data.n_train} examples fitted, <b>{data.n_test} held out</b>, and
            every accuracy above is on the held-out half. The shaded band is
            where the same fit landed on shuffled labels; the vertical mark is
            the majority class. A bar has to clear both to mean anything.
          </p>

          {data.saved && (
            <p className="meta">
              {data.saved.replaced ? "Replaced" : "Saved"}{" "}
              <b>{data.saved.name}</b> — the layer-{data.best_layer} direction,{" "}
              {data.saved.dims} dimensions, in the same space the steering panel
              pushes through.
              {data.saved.replaced &&
                " A direction was already stored under that name; this one is now in its place."}{" "}
              It is a direction that separates your two groups, which is not the
              same as a direction that causes the difference; steering it is how
              you find out.
            </p>
          )}

          <p className="meta probe-means">{data.means}</p>
          <ReceiptLine receipt={data.receipt} />
        </>
      )}
    </div>
  );
}
