import { useEffect, useState } from "react";
import {
  Ablation,
  ControlRanking,
  HeadScore,
  controlRanking,
  errorText,
} from "./api";

/**
 * THE CONTROL. The same ranking, on an untrained model of the same shape.
 *
 * This tool argues on nearly every surface that a measurement without a
 * control is not a measurement — and its own control had no way in. The
 * server has answered `/api/attention/control` since the null model landed
 * and nothing on the page could call it.
 *
 * It answers the question underneath every ranking here: **would this have
 * produced a confident, ordered list anyway?** A transformer with random
 * weights still has heads, still has a next-token distribution, and removing
 * one of its heads still moves that distribution — so it still yields a
 * ranked list of five heads with numbers beside them, and that list looks
 * exactly as convincing as this one. The only way to tell the two apart is to
 * run both and correlate them.
 *
 * WHY IT SITS INSIDE THE RANKING BLOCK
 *
 * Because it is an action on the ranking already on screen, not a separate
 * instrument. It appears once there is something to control for, it names the
 * layer it ran on, and its verdict is printed beside the numbers it is about.
 * In its own panel it would read as a second opinion; here it reads as what
 * it is — the ranking's own error bar.
 *
 * WHAT IT DOES NOT CLAIM
 *
 * The twin is one architecture at one seed on one prompt. A high correlation
 * says this ranking is mostly reporting the architecture *on this prompt*; it
 * does not say the head is uninteresting, and a low one does not certify
 * anything. The server writes that sentence itself (`verdict`) with the
 * thresholds it used, and it is rendered verbatim rather than summarised —
 * the caveats are the sentence.
 */

/**
 * A measured number, without printing a measurement as zero.
 *
 * Three decimals is the panel's own format and it is right for the heads that
 * matter. It is wrong for both halves of this comparison, and MEASURED wrong
 * rather than suspected: on Qwen3-0.6B layer 0, four of the five model rows
 * came back at 1.9e-4, 1.6e-4, 1.4e-4 and 1.3e-4 and every one rendered as
 * `KL 0.000`. Nothing greys them — they are genuinely above the noise floor,
 * which measured exactly 0.0 — so they simply read as "removing this head
 * does nothing", which is not what was measured. The untrained twin is worse:
 * its own top token carries 9e-5 of the mass, so every row of its probability
 * column read `0.000 → 0.000`, i.e. a whole column of nothing.
 *
 * So anything that would round to zero without being zero prints in exponent
 * form instead. An exact 0 still prints as 0.000, because that one IS the
 * measurement — the same distinction this file keeps everywhere else.
 */
function num(v: number): string {
  if (v === 0) return v.toFixed(3);
  return Math.abs(v) < 0.0005 ? v.toExponential(1) : v.toFixed(3);
}

/** Both sides of the comparison carry their own noise floor, their own target
 *  token and their own pass count. Nothing is shared between the columns
 *  except the tokens they ran over, so nothing is factored out of them. */
function ControlRow({
  r,
  floor,
  token,
}: {
  r: HeadScore;
  floor: number;
  token: string;
}) {
  // Same forward pass twice with nothing ablated. At or below it, the number
  // is arithmetic moving rather than the model, and it is greyed rather than
  // hidden — a list that quietly drops its bottom reads as complete.
  const noise = r.kl <= floor;
  /** A spread bound that is not there is UNKNOWN, never 0.000. */
  const bound = (v: number | undefined) => (v == null ? "not reported" : num(v));
  return (
    <li className={noise ? "faint" : ""}>
      <span className="mid ctl-head">
        L{r.layer} H{r.head}
      </span>
      <span className="mid ctl-kl">
        {noise ? "below the noise floor" : `KL ${num(r.kl)}`}
      </span>
      <span className="meta ctl-note">
        {/* The median is not the measurement, the spread is: a single donor
            sentence could have reported any point inside it as this head's
            score. */}
        {r.draws != null && (
          <>
            median of {r.draws}, {bound(r.kl_min)}–{bound(r.kl_max)} ·{" "}
          </>
        )}
        p({JSON.stringify(token)}) {num(r.p_top_before)} →{" "}
        {/* Null under resample: there is no single "after" across eight draws,
            so there is no honest number to print. */}
        {r.p_top_after == null ? "varies by draw" : num(r.p_top_after)}
        {r.flips_top &&
          (r.draws != null
            ? " · changes the top token under every draw"
            : " · changes the top token")}
      </span>
    </li>
  );
}

function ControlColumn({
  title,
  what,
  a,
}: {
  title: string;
  what: string;
  a: Ablation;
}) {
  const top = a.ranked.slice(0, 5);
  return (
    <div className="ctl-col">
      <div className="ctl-col-head">
        <strong>{title}</strong>
        <span className="meta">{what}</span>
      </div>
      <span className="meta ctl-col-facts">
        {a.passes} forward passes · {a.elapsed_s}s · {a.baseline}-ablation ·
        noise floor {a.noise_floor_kl} · its top token{" "}
        {JSON.stringify(a.target_token)}
        {/* The corpus is part of a resample measurement: the same head scores
            differently against different donor sentences. */}
        {a.corpus && (
          <>
            {" "}
            · {a.draws} draws from {a.corpus}
          </>
        )}
      </span>
      {top.length === 0 ? (
        // Not an empty list. "Nothing scored" and "nothing was ranked" are
        // different answers, and only the second one is true here.
        <div className="hint">
          This side produced no ranked heads at all, so there is nothing to
          list under it.
        </div>
      ) : (
        <ol className="ranking-list ctl-list">
          {top.map((r) => (
            <ControlRow
              key={`${r.layer}.${r.head}`}
              r={r}
              floor={a.noise_floor_kl}
              token={a.target_token}
            />
          ))}
        </ol>
      )}
      {/* The cap, said. Five rows out of however many were ranked. */}
      {a.ranked.length > top.length && (
        <span className="meta">
          showing the top {top.length} of {a.ranked.length} ranked head(s)
        </span>
      )}
    </div>
  );
}

export default function ControlTwin({
  layer,
  baseline,
  wholeModel,
}: {
  /** The layer the control runs on. The route takes ONE layer — it does not
   *  accept a whole-model scope — so this is the panel's current layer and
   *  the block says so when the ranking above it covers more. */
  layer: number;
  /** Taken from the on-screen ranking's own response, not from the panel's
   *  select: the select can have moved since, and a control run against a
   *  different baseline is a control for a ranking nobody is looking at. */
  baseline: string;
  /** True when the ranking above came from a whole-model sweep. */
  wholeModel: boolean;
}) {
  const [out, setOut] = useState<ControlRanking | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  /**
   * A control belongs to ONE layer under ONE baseline, and the response does
   * not echo which — so the only place that fact lives is these props, and a
   * result left on screen after they move would be labelled with a layer it
   * was never about.
   *
   * That is not hypothetical. A whole-model sweep ranks every layer and the
   * rows are buttons: clicking `L6 H2` moves the panel to layer 6 while the
   * ranking stays, so without this the block would keep the layer-14 numbers
   * and print "this control ran on layer 6 only" over them. The panel above
   * already discards its baseline comparison for the same reason when a new
   * ranking lands.
   */
  useEffect(() => {
    setOut(null);
    setErr("");
  }, [layer, baseline]);

  async function run() {
    setBusy(true);
    setErr("");
    try {
      // No seed is sent, so the server's own default answers and `out.seed`
      // is the number that was actually used rather than one echoed back from
      // here.
      setOut(await controlRanking(layer, baseline));
    } catch (e) {
      // The server's own sentence. Every refusal this route can raise already
      // says what would make it work — a recording carries no second model,
      // Ollama has no forward pass to intervene in, an architecture whose
      // blocks cannot be found has no twin — and anything in front of those
      // is the client guessing.
      setErr(errorText(e));
      setOut(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="ctl-twin">
      <div className="row">
        <button
          className="ghost sm"
          onClick={() => void run()}
          disabled={busy}
          title={
            "Build this architecture again with random weights, run the " +
            "identical ranking over the same tokens, and report how far the " +
            "two orders agree. Costs a second model in memory for the " +
            "duration, and two sweeps."
          }
        >
          {busy
            ? "running the control…"
            : out
              ? "run the control again"
              : "control: the same ranking on an untrained twin"}
        </button>
        <span className="meta">
          layer {layer} · {baseline}-ablation
        </span>
      </div>

      {err && <div className="hint err">{err}</div>}

      {out && (
        <div className="ctl-result">
          {/* The server's sentence, whole. It carries its own thresholds and
              is written to be argued with beside the number it came from, so
              summarising it would remove the part that matters. */}
          <div className="hint ok ctl-verdict">{out.verdict}</div>

          <span className="meta ctl-facts">
            {out.spearman == null ? (
              // NOT 0.00. "The two are uncorrelated" and "one side is not a
              // ranking at all" are different statements about the data.
              <>rank correlation not defined — one side has no ranking</>
            ) : (
              <>Spearman {out.spearman.toFixed(2)}</>
            )}
            {" · "}
            {out.top_k > 0 ? (
              <>
                sharing {out.top_k_shared} of the top {out.top_k}
              </>
            ) : (
              // Defensive and honest: "sharing 0 of the top 0" is a
              // conclusion drawn from nothing.
              <>no top heads were compared</>
            )}
            {" · "}seed {out.seed} · {out.baseline}-ablation
          </span>

          {/* The scope mismatch, stated rather than left to be noticed. The
              list above can be a whole-model sweep; the control is one layer,
              because that is what the route measures. */}
          {wholeModel && (
            <div className="hint warn">
              The ranking above covers every layer in the model. This control
              ran on layer {layer} only — it is a control for that layer's
              heads, not for the whole-model order.
            </div>
          )}

          <div className="ctl-cols">
            <ControlColumn
              title="this model"
              what="the weights you loaded"
              a={out.model}
            />
            <ControlColumn
              title="untrained twin"
              what={`same architecture, random weights, seed ${out.seed}`}
              a={out.untrained}
            />
          </div>

          {/* Verbatim, and said once because both sides came out of the same
              function and carry the same sentence. */}
          <div className="hint">{out.untrained.means}</div>
          <div className="hint">
            Both columns were produced by the same code over the same tokens at
            the same position — deliberately, because a second implementation
            of the measurement could differ from the one being checked, and
            then agreement would mean nothing in either direction. The twin's
            weights come from <code>config.json</code> alone, so nothing was
            downloaded to build it.
          </div>
        </div>
      )}
    </div>
  );
}
