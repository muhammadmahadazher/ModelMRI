import { TokenCount, TokenRollup, TraceCost } from "./api";

/**
 * Token counts a provider reported — and a cost column only when the user
 * brought their own prices.
 *
 * Two rules, and this file is both of them drawn:
 *
 *   - **null is not zero.** Anthropic returns cache counts only when a cache
 *     was in play and reasoning tokens only from models that reason. A field
 *     nobody reported renders "not reported by provider", greyed, never "0".
 *     Drawing a 0 there would make a silent provider indistinguishable from
 *     one that answered zero.
 *   - **a partial total says it is partial.** A run where 3 of 11 calls have
 *     no price on file shows the figure as a FLOOR with the count of unpriced
 *     calls beside it, never as a total that looks complete.
 *
 * There is no bundled price map. Every competitor derives cost from one with
 * regex model matching, and a regex matching the wrong model produces a
 * plausible dollar figure with no signal it is wrong.
 */

/** The five, in the order a reader wants them, with the label each gets. */
const LABELS: Array<[string, string]> = [
  ["tokens_in", "input"],
  ["tokens_out", "output"],
  ["tokens_cache_read", "cache read"],
  ["tokens_cache_write", "cache write"],
  ["tokens_reasoning", "reasoning"],
];

function Cell({ count }: { count?: TokenCount }) {
  // Absent from the payload entirely and reported-by-nobody are the same
  // fact to a reader, and both are "not reported".
  if (!count || count.total === null) {
    return <span className="tok-none">not reported by provider</span>;
  }
  return (
    <>
      <b className="mid">{count.total.toLocaleString()}</b>
      {/* Some reported and some did not. Printing the sum alone would be the
          same lie as storing 0, one level up. */}
      {count.silent > 0 && (
        <span className="tok-partial">
          {" "}
          from {count.reported} of {count.reported + count.silent}
        </span>
      )}
    </>
  );
}

export function TokenTable({
  rollup,
  title,
}: {
  rollup?: TokenRollup;
  title: string;
}) {
  if (!rollup || !rollup.n_llm_steps) {
    return (
      <p className="meta">
        {title}: no LLM calls here, so there are no tokens to count.
      </p>
    );
  }
  return (
    <div className="tok-table">
      <div className="meta tok-title">
        {title} · {rollup.n_llm_steps} LLM call
        {rollup.n_llm_steps === 1 ? "" : "s"}
      </div>
      <dl className="tok-grid">
        {LABELS.map(([key, label]) => (
          <div key={key} className="tok-row">
            <dt className="meta">{label}</dt>
            <dd>
              <Cell count={rollup.counts?.[key]} />
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export function CostBanner({ cost }: { cost?: TraceCost }) {
  if (!cost) return null;
  // A price file that could not be read is a field, not a failed request: the
  // token counts above are complete and useful without it.
  if (cost.error) {
    return <div className="tok-cost err">{cost.error}</div>;
  }
  if (!cost.n_calls) return null;
  const state = cost.total === null ? "unpriced" : cost.partial ? "partial" : "full";
  return (
    <div className={`tok-cost ${state}`}>
      {cost.total !== null && (
        <b className="mid">
          {cost.currency}
          {cost.total.toLocaleString(undefined, {
            minimumFractionDigits: 4,
            maximumFractionDigits: 4,
          })}
        </b>
      )}{" "}
      {/* The sentence is authored server-side beside the arithmetic, so the
          panel cannot drift from what was actually counted. */}
      <span className="meta">{cost.means}</span>
    </div>
  );
}

/** One step's own counts, for the inspector header. */
export function StepTokens({
  step,
}: {
  step: {
    tokens_in: number | null;
    tokens_out: number | null;
    tokens_cache_read: number | null;
    tokens_cache_write: number | null;
    tokens_reasoning: number | null;
  };
}) {
  const parts: string[] = [];
  // `in` and `out` are drawn as a pair only when BOTH are present. The old
  // line tested `tokens_in != null` alone and rendered "100→null tok" on a
  // provider that reported one and not the other.
  if (step.tokens_in !== null && step.tokens_out !== null) {
    parts.push(`${step.tokens_in}→${step.tokens_out} tok`);
  } else if (step.tokens_in !== null) {
    parts.push(`${step.tokens_in} tok in, output not reported`);
  } else if (step.tokens_out !== null) {
    parts.push(`${step.tokens_out} tok out, input not reported`);
  }
  if (step.tokens_cache_read !== null) parts.push(`${step.tokens_cache_read} cache read`);
  if (step.tokens_cache_write !== null) parts.push(`${step.tokens_cache_write} cache write`);
  if (step.tokens_reasoning !== null) parts.push(`${step.tokens_reasoning} reasoning`);
  if (!parts.length) return null;
  return <> · {parts.join(" · ")}</>;
}
