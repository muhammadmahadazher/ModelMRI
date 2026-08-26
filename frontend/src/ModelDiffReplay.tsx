import { useEffect, useState } from "react";
import { errorText, getDiffReplay, ModelDiffSection } from "./api";
import { counted, measured, percent } from "./measured";

/** A comparison of two models somebody sent you, opened with nothing installed.
 *
 *  `runtime._model_diff_for_export` wrote this section into every export that
 *  had a comparison behind it, and `session._model_diff` validated it -- with
 *  two rules of its own about what a diff may claim. Nothing read it back on
 *  any surface: not a route, not `/api/session`, not a panel, not the CLI, not
 *  `modelmri diff`. It was the one section in the format with no reader
 *  anywhere, which is why it is the last of five to get one.
 *
 *  SEPARATE FROM `ModelDiffPanel`, which RUNS a comparison against two
 *  checkpoints on this disk. A recording has neither, and a recipient with no
 *  weights is exactly who this is for.
 *
 *  WHAT THIS PANEL REFUSES TO LET A READER CONCLUDE:
 *
 *    it is not about this file  a diff can ride in a `.mri` about a THIRD
 *                               model, so the two names it compared are the
 *                               first thing on screen rather than a caption.
 *    a median is not a curve    every number here is a distribution over a
 *                               prompt set. The middle half travels with the
 *                               median, and when the spread is wider than the
 *                               median itself the panel says there is no
 *                               single amount rather than printing one.
 *    no divergence is a result  `consensus_layer: null` means the cosine
 *                               never fell. That is an answer -- these two
 *                               checkpoints agree -- and not a missing field.
 *
 *  It renders NOTHING when there is no comparison, which is most sessions.
 */
export default function ModelDiffReplay({
  /** Bumped whenever the page resets. */
  epoch,
  /** WHICH `.mri` is open — the `created_at`, not a boolean. */
  sessionKey,
}: {
  epoch: number;
  sessionKey: string;
}) {
  const [run, setRun] = useState<ModelDiffSection | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    let live = true;
    setErr("");
    // Cleared first: a panel showing the previous file's comparison under this
    // file's name is the worst thing it could do.
    setRun(null);
    void getDiffReplay()
      .then((got) => {
        if (!live) return;
        setRun(got.available ? (got as ModelDiffSection) : null);
      })
      .catch((e) => {
        if (!live) return;
        setRun(null);
        setErr(errorText(e));
      });
    return () => {
      live = false;
    };
  }, [epoch, sessionKey]);

  if (err) {
    return (
      <div className="panel">
        <div className="hint err">{err}</div>
      </div>
    );
  }
  if (!run) return null;

  // Every shape is checked: the viewer build serves this section straight out
  // of the file with only the two model names verified.
  const list = <T,>(v: T[] | undefined | null): T[] => (Array.isArray(v) ? v : []);
  const prompts = list(run.prompts);
  const layers = list(run.layers);
  const heads = list(run.heads);
  const kl = run.kl;

  // THE SPREAD DECIDES WHETHER THERE IS AN ANSWER. `ModelDiffPanel` applies
  // the same test to the live measurement: when the middle half is wider than
  // the median, the prompts disagree by more than the amount being reported
  // and a single number would be a fiction.
  const tight =
    kl !== undefined &&
    (kl.median === 0
      ? kl.high - kl.low === 0
      : kl.high - kl.low <= Math.abs(kl.median) * 0.5);

  return (
    <div className="panel" data-mri-group-label="diff">
      <div className="row">
        <span className="dot d-mdiff" />
        <h2>TWO MODELS COMPARED — SHARED, NOT RE-RUN</h2>
      </div>

      {/* THE TWO NAMES FIRST, and not as a caption. `session._model_diff`
          requires them because a diff can ride in a file about a third model,
          and one that does not name its own sides is read as being about the
          model the rest of the page describes. */}
      <p className="meta">
        <b>{run.model_a}</b> → <b>{run.model_b}</b>
        {typeof run.n_prompts === "number" && (
          <> · over {run.n_prompts} prompt(s)</>
        )}
        {typeof run.n_layers === "number" && <> · {run.n_layers} layers</>}
      </p>

      {/* THE ANSWER SLOT: how far apart the two answers are, or the statement
          that the prompts do not agree on an amount. */}
      {kl === undefined ? (
        <p className="answer unmeasured">
          <span className="answer-n">no spread recorded</span>
          <span className="answer-of">
            this file carries per-prompt rows without the distribution over
            them, so there is no single amount to report
          </span>
        </p>
      ) : (
        <p className={`answer${tight ? "" : " unmeasured"}`}>
          <span className="answer-n">
            {tight ? `${measured(kl.median, 4)} nats` : "no single amount"}
          </span>
          <span className="answer-of">
            {tight ? (
              <>
                is the median distance between the two models&apos; answers per
                position, over {kl.n} prompt(s) — middle half{" "}
                {measured(kl.low, 4)} to {measured(kl.high, 4)}
                {typeof kl.n_nonzero === "number" && (
                  <>
                    , and {kl.n_nonzero} of {kl.n} moved at all
                  </>
                )}
              </>
            ) : (
              <>
                the prompts disagree by more than the median: {measured(kl.low, 4)}{" "}
                to {measured(kl.high, 4)} around {measured(kl.median, 4)} over{" "}
                {kl.n} prompt(s). A single number here would be a median
                pretending to be a result
              </>
            )}
          </span>
        </p>
      )}

      {/* WHERE THEY PART. `null` is a RESULT: the cosine never fell, so these
          two checkpoints agree all the way down. */}
      <p className="meta">
        {run.consensus_layer === null || run.consensus_layer === undefined ? (
          <>
            no layer diverged on a majority of prompts — on this prompt set the
            two checkpoints do not part company anywhere in particular, which
            is an answer rather than a missing one
          </>
        ) : (
          <>
            the cosine falls furthest at <b>layer {run.consensus_layer}</b>
            {typeof run.consensus_share === "number" && (
              <> on {percent(run.consensus_share, 0)} of prompts</>
            )}
            {typeof run.consensus_share === "number" &&
              run.consensus_share < 0.5 && (
                <>
                  {" "}
                  ·{" "}
                  <span className="warn">
                    which is a minority of them, so this is where they part
                    MOST OFTEN rather than where they part
                  </span>
                </>
              )}
          </>
        )}
      </p>

      {layers.length > 0 && (
        <>
          <h3 className="irr-h">PER LAYER, ACROSS THE PROMPT SET</h3>
          <ul className="irr-rows">
            {layers.slice(0, 40).map((row, i) => (
              <li key={i}>
                <b>layer {row.layer}</b>{" "}
                <span className="meta">
                  median {measured(row.median, 5)}, middle half{" "}
                  {measured(row.low, 5)}–{measured(row.high, 5)} over{" "}
                  {counted(row.n, "prompt")}
                  {typeof row.n_first === "number" && row.n_first > 0 && (
                    <> · first to diverge on {row.n_first}</>
                  )}
                </span>
              </li>
            ))}
          </ul>
          {layers.length > 40 && (
            <p className="meta">
              {layers.length - 40} further layer(s) are in the file and not
              listed here — a cap on what is SHOWN, not on what was measured.
            </p>
          )}
        </>
      )}

      {heads.length > 0 && (
        <>
          <h3 className="irr-h">HEADS THAT MOVED</h3>
          <ul className="irr-rows">
            {heads.slice(0, 40).map((h, i) => (
              <li key={i}>
                <b>
                  L{h.layer}H{h.head}
                </b>{" "}
                <span className="meta">
                  {measured(h.median_a, 4)} → {measured(h.median_b, 4)}{" "}
                  (shift {measured(h.shift, 4)} over {counted(h.n, "prompt")})
                  {(h.top_a || h.top_b) && (
                    <>
                      {" "}
                      · looked at <code>{h.top_a || "—"}</code> and now{" "}
                      <code>{h.top_b || "—"}</code>
                    </>
                  )}
                </span>
              </li>
            ))}
          </ul>
        </>
      )}

      {prompts.length > 0 && (
        <p className="meta">
          {prompts.length} prompt(s) recorded
          {typeof run.head_passes === "number" && run.head_passes > 0 && (
            <> · {run.head_passes} head pass(es)</>
          )}
          {typeof run.seconds === "number" && run.seconds > 0 && (
            <> · measured in {measured(run.seconds, 1)}s</>
          )}
        </p>
      )}

      {run.means && <div className="hint">{run.means}</div>}
    </div>
  );
}
