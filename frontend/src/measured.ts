/**
 * Printing a measurement without printing it as zero.
 *
 * WHY THIS IS ONE MODULE AND NOT A HABIT
 *
 * The rule was understood and independently reinvented four times, with three
 * different thresholds and four different decimal counts: `ControlTwin.num`
 * (< 0.0005), `ImagePanel.distance` and `ImagePanel.drop` (both < 0.0001), and
 * `FeaturesPanel.fmtFVU`, which scales its decimals by magnitude and still
 * floors at four — so an FVU of 1e-5 printed `0.0000`. Four copies of one idea
 * is four places for it to be missing, and it WAS missing: the sites listed
 * below were all printing live measurements as `0.000`.
 *
 * THE FAILURE THIS EXISTS TO STOP
 *
 * MEASURED on Qwen3-0.6B, layer 0: four head rows came back at 1.9e-4, 1.6e-4,
 * 1.4e-4 and 1.3e-4 and every one rendered `KL 0.000`. The noise floor for that
 * run measured exactly 0.0, so nothing greyed them out — they simply read as
 * "removing this head does nothing", which is the opposite of what was
 * measured. An untrained control model was worse: its top token carries 9e-5 of
 * the mass, so a whole probability column read `0.000 → 0.000`.
 *
 * That is the project's central prohibition arriving through the formatter
 * rather than through the maths. A number that rounds to zero without being
 * zero is a fabricated measurement, and it does not matter that the fabrication
 * happened in `toFixed`.
 *
 * AN EXACT ZERO STILL PRINTS AS ZERO
 *
 * `0.000`, not `0.0e+0`. That one IS the measurement — an ablation that changed
 * nothing, a noise floor that really is flat — and dressing it in exponent
 * notation would suggest a precision nobody asked for.
 */

/** The magnitude below which `decimals` fixed places would render zero. */
const floorFor = (decimals: number) => 0.5 * 10 ** -decimals;

/**
 * A measured quantity, at `decimals` fixed places, escaping to exponent form
 * rather than rounding away to nothing.
 *
 * @param v        the measurement
 * @param decimals fixed places for the ordinary case
 */
export function measured(v: number, decimals = 3): string {
  if (!Number.isFinite(v)) return "—";
  if (v === 0) return v.toFixed(decimals);
  return Math.abs(v) < floorFor(decimals) ? v.toExponential(1) : v.toFixed(decimals);
}

/**
 * A signed movement, with the sign always shown.
 *
 * The sign is the reading — a window that pushed the logit DOWN is a different
 * finding from one that pushed it up — so it is never dropped and never left to
 * a colour alone.
 */
export function signed(v: number, decimals = 4): string {
  if (!Number.isFinite(v)) return "—";
  if (v === 0) return "0";
  const a = Math.abs(v);
  const body =
    a < floorFor(decimals) ? a.toExponential(1) : a.toFixed(decimals);
  return (v > 0 ? "+" : "-") + body;
}


/**
 * A FRACTION rendered as a percentage, without rendering a real share as 0%.
 *
 * The same failure as `measured`, one multiplication along, and it was in
 * twenty-odd places: `(x * 100).toFixed(1)` prints "0.0%" for every share
 * below 0.0005. MEASURED on Qwen3-1.7B, whose context window is 40,960
 * tokens — every prompt under about twenty tokens rendered
 * "context 20 / 40,960 (0.0%)", a measured fraction of a real window shown
 * as none of it.
 *
 * Ordinary values are untouched: 0.5 is still "50.0%". Only a share that is
 * nonzero and would round away escapes to exponent form, and an exact zero
 * still prints "0.0%" because that one IS the measurement.
 */
export function percent(fraction: number, decimals = 1): string {
  if (!Number.isFinite(fraction)) return "—";
  const scaled = fraction * 100;
  if (fraction === 0) return `${scaled.toFixed(decimals)}%`;
  return Math.abs(scaled) < floorFor(decimals)
    ? `${scaled.toExponential(1)}%`
    : `${scaled.toFixed(decimals)}%`;
}
/**
 * A before-and-after pair, at enough precision to show that it moved.
 *
 * `measured` fixes the small end and does nothing for the other failure, which
 * is a pair of numbers near 1 whose DIFFERENCE is small. MEASURED on
 * Qwen3-1.7B, ablating head 10 of layer 0: p went 0.99993 → 0.99990, and three
 * fixed decimals printed `1.000 → 1.000`. The KL in the same row said the
 * distribution changed. One row, two formatters, two contradictory answers —
 * and the one the eye believes is the pair, because a pair is what movement
 * looks like.
 *
 * So the decimals grow until the two sides differ, or until the cap. The cap
 * matters: two values that are genuinely equal must print equal rather than
 * being chased to fifteen places, because "it did not move" is a real reading
 * and this must not manufacture a difference to avoid printing one.
 */
export function pair(before: number, after: number, decimals = 3): [string, string] {
  if (!Number.isFinite(before) || !Number.isFinite(after)) {
    return [measured(before, decimals), measured(after, decimals)];
  }
  let d = decimals;
  // 9 places: past float64's honest precision for numbers of this size, so
  // anything still indistinguishable here is indistinguishable.
  while (d < 9 && before !== after && before.toFixed(d) === after.toFixed(d)) d += 1;
  return [measured(before, d), measured(after, d)];
}

/**
 * A quantity whose scale is not known in advance, so the decimals follow the
 * magnitude.
 *
 * FVU spans five orders of magnitude between a working sparse autoencoder and a
 * wrong one — 0.0010 against 13579.24 on the default release — and attention
 * mass depends entirely on the latent resolution, single figures at 16x16 and
 * hundreds at 64x64. One fixed width is either noise or a column of zeroes.
 *
 * The small end still goes through `measured`, which is the half the previous
 * versions of this got wrong: scaling the decimals up to four is not the same
 * as never printing a measurement as zero.
 */
export function scaled(v: number): string {
  if (!Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(2);
  return measured(v, 4);
}


/** A KL small enough that fixed decimals would print it as zero.
 *
 *  Not cosmetic. Feature scores span from 0.4174529 down to -3e-08 on the one
 *  prompt this was measured on, and five decimal places render most of that
 *  list as 0.00000 — a measured value displayed as nothing. Whether a number
 *  that small MEANS anything is what `below_resolution` is for.
 *
 *  A SEPARATE FLOOR from `measured`, deliberately. `measured(kl, 5)` escapes
 *  only below 5e-6, where five places genuinely round to zero; this escapes at
 *  1e-3, because a ranked list of KLs is read by comparing its entries and
 *  0.00003 against 0.00007 loses the comparison that the list exists to make.
 *  One rule for one quantity, in one place — it had two identical definitions,
 *  in `api.ts` and inside `AttentionPanel`, which is a disagreement waiting to
 *  happen rather than one that had happened yet.
 */
export const fmtKL = (kl: number) =>
  kl !== 0 && Math.abs(kl) < 0.001 ? kl.toExponential(2) : kl.toFixed(5);


/** A byte count in the unit that keeps its significant digits.
 *
 *  The client half of `modelmri/fmt.py`'s `bytes_si`, and it exists because
 *  the two halves disagreed on screen. `{n / 1e6} MB` alone is what turned a
 *  real 400 kB checkpoint into "0 MB" on a row whose Load button was enabled,
 *  and `{n / 1e9:,.1f} GB` did the same to a 4 MB cache in the sentence above
 *  it — one quantity, two formatters, one screen.
 *
 *  The thresholds are `bytes_si`'s, exactly, so a size named here and the same
 *  size named in a server sentence read identically rather than nearly. `gb`
 *  in `LoadBar` delegates to this rather than keeping a second opinion.
 */
export function bytesSI(n: number): string {
  if (!Number.isFinite(n)) return "unknown";
  if (n >= 1e9) return `${(n / 1e9).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6).toLocaleString()} MB`;
  if (n >= 1e3) return `${Math.round(n / 1e3).toLocaleString()} kB`;
  return `${Math.max(0, Math.round(n)).toLocaleString()} bytes`;
}


/** How long something took, in the unit a reader can act on.
 *
 *  Moved here from `RubricPanel`, where it was the better of two duration
 *  formatters this app had. The agents list used `(ms / 1000).toFixed(1)}s`,
 *  which prints a real 1 ms step as "0.0s" — the rounds-to-zero rule, in a
 *  column read as a measurement — and a five-minute run as "312.4s", which is
 *  a number you have to do arithmetic on before it means anything.
 *
 *  `null` is UNKNOWN and returns "", never "0ms": a duration nobody recorded
 *  is not a duration of nothing, which is the whole reason `duration_ms` is
 *  nullable in the store.
 */
export function howLong(ms: number | null): string {
  if (ms === null || !Number.isFinite(ms)) return "";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  return `${m}m ${Math.round((ms % 60_000) / 1000)}s`;
}

/**
 * `n` with the suffix English actually uses, and thousands separated.
 *
 * The mirror of `fmt.ordinal` in the Python half, and it exists for the same
 * reason and against the same bug: appending "th" to an integer is right for
 * 4 through 10 and wrong for every other digit. The robot panel printed
 * "every 2th eligible row" beside a reference set of 12,745 rows — a typo in
 * the sentence a reader is being asked to trust a measurement through.
 *
 * `ordinal(1)` → "1st", `ordinal(11)` → "11th", `ordinal(1013)` → "1,013th".
 */
export function ordinal(n: number): string {
  const whole = Math.trunc(n);
  const last2 = Math.abs(whole) % 100;
  const last1 = Math.abs(whole) % 10;
  // 11, 12 and 13 take "th" despite their last digit — and so do 111 and
  // 1013, which is what makes the last digit alone the wrong thing to read.
  const suffix =
    last2 === 11 || last2 === 12 || last2 === 13
      ? "th"
      : last1 === 1
        ? "st"
        : last1 === 2
          ? "nd"
          : last1 === 3
            ? "rd"
            : "th";
  return `${whole.toLocaleString()}${suffix}`;
}
