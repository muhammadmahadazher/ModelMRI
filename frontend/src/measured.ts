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
