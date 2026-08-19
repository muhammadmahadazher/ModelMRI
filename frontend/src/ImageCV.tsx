import { useEffect, useState } from "react";
import {
  errorText,
  imageCvAttribute,
  imageCvCost,
  imageCvPredict,
  imageCvReadout,
  ImageCvAttribution,
  ImageCvCost,
  ImageCvPrediction,
  ImageCvReadout,
} from "./api";

/**
 * What a classifier, detector or segmenter actually says — and where it looked.
 *
 * WHY THIS IS A SEPARATE CONTROL FROM THE OCCLUSION SWEEP
 *
 * The panel could already cover parts of an image and report what moved, which
 * answers "what supports this answer". It could not answer "what IS the
 * answer". So a ViT loaded here offered an attribution map over a prediction
 * the reader was never shown — a saliency picture with no caption.
 *
 * WHY THE LABELS CARRY A NOTE
 *
 * Class names come off the checkpoint's own `id2label`. A checkpoint that
 * publishes none gets INDICES and a sentence saying so, rather than borrowing
 * ImageNet's names because the shape happens to be 1000 — a wrong class name
 * is read as the model's answer, which is worse than a number nobody can
 * interpret. `labels_read` is the flag; the note is the server's own words.
 *
 * WHY THE READOUT CAN REFUSE
 *
 * A ViT has attention; a convolutional backbone does not. The response says
 * which, and this renders the refusal rather than an empty grid — a blank
 * heat map is indistinguishable from one that measured nothing but zero.
 */
export default function ImageCV({
  picture,
  canReadout,
}: {
  /** A data URL, or "" when nothing is picked yet. */
  picture: string;
  canReadout: boolean;
}) {
  const [pred, setPred] = useState<ImageCvPrediction | null>(null);
  const [readout, setReadout] = useState<ImageCvReadout | null>(null);
  const [attr, setAttr] = useState<ImageCvAttribution | null>(null);
  const [cost, setCost] = useState<ImageCvCost | null>(null);
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");
  const [topK, setTopK] = useState(5);
  const [layer, setLayer] = useState(0);

  /** Attribute a NAMED class, not just the argmax.
   *
   *  The prediction list raises the question this answers — "why that one?"
   *  and "then what supports the second one?" are different questions, and
   *  the server reports which of the two a given map is by setting
   *  `region_chosen_by` to "model" or "caller". Passing the index the reader
   *  clicked is what makes it the second kind.
   */
  async function run(what: "predict" | "readout" | "attribute", target?: number) {
    if (!picture) return;
    setBusy(what);
    setErr("");
    try {
      if (what === "predict") {
        setPred(await imageCvPredict({ image: picture, top_k: topK }));
      } else if (what === "readout") {
        const got = await imageCvReadout({ image: picture, top_k: topK });
        setReadout(got);
        setLayer(0);
      }
      else {
        setAttr(
          await imageCvAttribute({
            image: picture,
            // `undefined` means "the model's own top answer" and the response
            // says so. Not defaulted to 0 here: index 0 is a real class.
            target: target ?? null,
          }),
        );
      }
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }

  // Priced before any button is pressed, like every other sweep here. The
  // occlusion pass is the expensive one — hundreds of forward passes — and a
  // reader who is told afterwards was told too late.
  useEffect(() => {
    if (!pred) return;
    let live = true;
    void imageCvCost(pred.height, pred.width)
      .then((c) => live && setCost(c))
      // A preflight that cannot be fetched must not block the measurement it
      // was going to describe; the cost line simply does not appear.
      .catch(() => live && setCost(null));
    return () => {
      live = false;
    };
  }, [pred]);

  const grid = readout?.layers?.[layer];
  /** The strongest cell in THIS layer, so the shading has a scale.
   *
   *  Per layer rather than across the whole readout: early and late layers
   *  differ by orders of magnitude, and one shared scale renders every early
   *  layer as uniformly blank. */
  const hi = grid
    ? grid.values.reduce((m, row) => row.reduce((n, v) => (v > n ? v : n), m), 0) || 1
    : 1;

  return (
    <div className="icv">
      <div className="row istep-controls">
        <label className="meta">
          top
          <input
            type="number"
            className="istep-num"
            value={topK}
            min={1}
            max={100}
            onChange={(e) => setTopK(Math.max(1, Number(e.target.value) || 1))}
            aria-label="How many classes to report"
          />
        </label>
        <button
          className="green"
          onClick={() => void run("predict")}
          disabled={busy !== "" || !picture}
        >
          {busy === "predict" ? "Asking…" : "What does it say?"}
        </button>
        {canReadout && (
          <button
            className="ghost"
            onClick={() => void run("readout")}
            disabled={busy !== "" || !picture}
          >
            {busy === "readout" ? "Reading…" : "Where did it look?"}
          </button>
        )}
        {!picture && <span className="meta">pick a picture first</span>}
      </div>

      {pred && (
        <div className="icv-pred">
          <p className="meta">
            {pred.task_label} · {pred.classes} class
            {pred.classes === 1 ? "" : "es"} · {pred.width}x{pred.height} ·{" "}
            {pred.dtype}
          </p>
          <ol className="icv-classes">
            {pred.classes_top.map((c) => (
              <li key={c.index}>
                {/* A BUTTON, because the list raises the question. "Why that
                    one?" and "then what supports the second one?" are
                    different questions, and until now only the first was
                    reachable — the sweep always took the argmax. */}
                <button
                  className="mid icv-label icv-pick"
                  onClick={() => void run("attribute", c.index)}
                  disabled={busy !== ""}
                  title={`Cover the picture and measure what supports "${c.label}"`}
                >
                  {c.label}
                </button>
                <span className="icv-track">
                  <span
                    className="icv-bar"
                    style={{ width: `${Math.max(c.probability * 100, 0.6)}%` }}
                  />
                </span>
                <span className="mid icv-p">
                  {(c.probability * 100).toFixed(1)}%
                </span>
              </li>
            ))}
          </ol>
          {/* The server's own sentence about where the names came from. Not
              re-worded here: this panel must not be the thing that decides
              whether a class name is trustworthy. */}
          <p className="meta icv-note">{pred.labels_note}</p>
          {cost && (
            <p className="meta">
              Click a class to see what supports it.{" "}
              {String(cost.attribution?.means ?? "")}
            </p>
          )}
          {pred.boxes && pred.boxes.length > 0 && (
            <p className="meta">
              {pred.boxes.length} box{pred.boxes.length === 1 ? "" : "es"} above
              the score threshold.
            </p>
          )}
        </div>
      )}

      {readout && readout.kind !== "attention" && (
        <p className="meta icv-note">
          {readout.reason ||
            "This architecture has no per-layer attention to read, so there is nothing to draw."}
        </p>
      )}

      {readout && readout.kind === "attention" && readout.layers.length > 0 && (
        <div className="icv-readout">
          <div className="row istep-controls">
            <label className="meta">
              layer
              <input
                type="range"
                min={0}
                max={readout.layers.length - 1}
                value={layer}
                onChange={(e) => setLayer(Number(e.target.value))}
                aria-label="Which layer to show"
              />
            </label>
            <span className="mid">
              {grid?.layer} of {readout.layers.length - 1}
            </span>
          </div>
          {grid && (
            <div
              className="icv-grid"
              style={{ gridTemplateColumns: `repeat(${grid.cols}, 1fr)` }}
              role="img"
              aria-label={`Attention over image patches at layer ${grid.layer}`}
            >
              {grid.values.flatMap((row, r) =>
                row.map((v, c) => (
                  <i
                    key={`${r}:${c}`}
                    style={{ opacity: Math.min(v / hi, 1) }}
                    title={`patch ${r},${c} · ${v.toFixed(5)}`}
                  />
                )),
              )}
            </div>
          )}
          <p className="meta">{readout.means}</p>
        </div>
      )}

      {attr && (
        <div className="icv-attr">
          {/* The server's own sentence. It states WHICH answer the map is of
              and whether the tool or the reader chose it — "explaining the
              answer given" and "auditing one you supplied" are different
              claims and only the response knows which this was. */}
          <p className="meta icv-note">{attr.means}</p>
          {attr.attribution === null && (
            <p className="meta">
              The occluder produced no map for that choice.
            </p>
          )}
        </div>
      )}

      {err && <p className="err">{err}</p>}
    </div>
  );
}
