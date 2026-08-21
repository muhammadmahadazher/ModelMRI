import { LoadProgress } from "./api";
import { bytesSI } from "./measured";

/**
 * The meter for any in-flight model load, text or image.
 *
 * Lifted out of `Playground.tsx` unchanged rather than reimplemented for the
 * image panel. The image side needed the same thing — a named stage, bytes
 * when they are known, an indeterminate sweep when they are not, and a Stop —
 * and a second copy would have been a second place for the two corrections
 * already baked in here to be missing: the clamp that stopped "5.0 GB / 2.5
 * GB" from being printed beside a full bar, and showing the id the SERVER is
 * loading rather than whatever the picker now has selected.
 *
 * The stage vocabulary is shared for the same reason. Text and image loads do
 * not run the same steps — a diffusion pipeline is scanned for live pickle
 * opcodes and a language model is not — so the map holds both, and an
 * unrecognised stage falls back to "Loading" instead of rendering a raw
 * protocol word at somebody.
 */

const STAGES: Record<string, string> = {
  // Text loads.
  resolving: "Resolving on the Hub",
  weights: "Fetching weights",
  device: "Moving to the accelerator",
  // Image loads. The names come from the order `image_runtime.load` runs in,
  // where every step refuses before the next one costs anything.
  identify: "Reading the checkpoint",
  scan: "Scanning the weights",
  open: "Opening the pipeline",
  // Both.
  ready: "Ready",
  error: "Failed",
  // Its own word, not "Failed". The server answers a stopped load 200 with a
  // sentence precisely so the panel does not report a deliberate act as a
  // fault, and a stage label reading "Failed" would undo that on its own.
  cancelled: "Stopped"
};

/** A byte count in the unit that keeps its significant digits.
 *
 *  The `MB` arm alone rounded every real size under half a megabyte to
 *  "0 MB" — measured while chasing a bar that read "0 MB / 0 MB" against a
 *  live download. That denominator turned out to be wrong for its own reason
 *  (see `_expected_files`), but a formatter that can print a nonzero quantity
 *  as zero is a second bug hiding behind the first, and this project's rule is
 *  that a number the reader is waiting on may not round away to nothing. */
// Delegated rather than duplicated. This had its own three arms and no kB
// one, so 400,000 bytes printed as "400,000 bytes" here while the server's
// `fmt.bytes_si` called the same number "400 kB". Kept as an export because a
// dozen call sites name it; the rule itself lives in `measured.ts` beside
// every other formatter that mirrors a server-side one.
export const gb = (n: number) => bytesSI(n);

/** A duration somebody can act on. Seconds under a minute, then minutes, then
 *  hours and minutes — "312 minutes" is a number you have to do arithmetic on
 *  before it means anything. */
export function remaining(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`;
}

/** Progress for an in-flight load: named stage, bytes when we know them,
 *  and an indeterminate sweep when we don't. */
export default function LoadBar({
  p,
  id,
  onStop,
}: {
  p: LoadProgress | null;
  id: string;
  onStop: () => void;
}) {
  const total = p?.bytes_total ?? 0;
  // Clamped, and not only in the bar. The width was already capped at 100%
  // while the text beside it was not, so a mis-count showed as a full bar
  // labelled "5.0 GB / 2.5 GB" — the number that gave the bug away.
  const done = Math.min(p?.bytes_done ?? 0, total || Infinity);
  const pct = total > 0 ? Math.min(100, (done / total) * 100) : null;
  const stopping = (p?.detail ?? "").startsWith("stopping");
  // The model the server is loading, which is not necessarily the one the
  // picker is showing: pick a second model while the first is still loading
  // and `id` is already the new one, so the running load's bytes and elapsed
  // time appeared under a model that had not started.
  const loading = p?.hf_id ?? id;
  return (
    <div className="loadbar glass-inset" role="status" aria-live="polite">
      <div className="loadbar-row">
        <span className="loadbar-stage">{STAGES[p?.stage ?? ""] ?? "Loading"}</span>
        <span className="mid loadbar-id">{loading}</span>
        <span className="spacer" />
        <span className="meta">
          {pct !== null && `${gb(done)} / ${gb(total)} · ${gb(total - done)} left · `}
          {(p?.elapsed_s ?? 0).toFixed(0)}s
          {/* Only when the server is willing to estimate. It withholds the
              number until there is enough history to divide by, because a
              countdown that opens with "4 hours" and settles at "40 seconds"
              is one the reader learns to ignore. */}
          {p?.eta_s != null && ` · ~${remaining(p.eta_s)} left`}
        </span>
        {/* The whole reason this component was revisited. A minutes-long
            download with no way out is a trap, and this one could run for
            days before failing. */}
        <button className="ghost sm stop" onClick={onStop} disabled={stopping}>
          {stopping ? "stopping…" : "Stop"}
        </button>
      </div>
      <div className={`loadbar-track ${pct === null ? "indeterminate" : ""}`}>
        <div
          className="loadbar-fill"
          style={pct === null ? undefined : { width: `${pct}%` }}
        />
      </div>
      {p?.detail && <div className="meta loadbar-detail">{p.detail}</div>}
    </div>
  );
}
