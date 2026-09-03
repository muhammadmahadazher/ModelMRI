// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

import { useEffect, useState } from "react";
import { percent } from "./measured";
import { getTelemetry, TelemetryReport } from "./api";

/** Bytes at a unit that does not round them away.
 *
 *  A fixed "GB" printed 259,200 bytes of attention scores as "0.00 GB", which
 *  reads as nothing at all — the exact failure this bar exists to avoid, since
 *  the whole point of the introspection line is that the number is real. The
 *  unit follows the magnitude, and anything under a kilobyte is shown in bytes
 *  rather than as a rounded 0.0.
 */
const bytes = (n: number) => {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)} kB`;
  return `${n} B`;
};

/** What the last run cost, and how much of it was this tool watching.
 *
 *  Live telemetry is table stakes — TextGen shows tokens/sec, LM Studio has a
 *  developer page, llama-server exposes /metrics. The differentiated line is
 *  the introspection cost: ModelMRI forces eager attention and asks for the
 *  scores, which a runner never allocates, and that is the honest answer to
 *  "why is this slower than Ollama".
 *
 *  Every cell can read "could not measure". None of them ever reads 0 to mean
 *  that — a zero in a memory column is a claim that nothing was used.
 */
export default function TelemetryBar({ epoch }: { epoch: number }) {
  const [t, setT] = useState<TelemetryReport | null>(null);

  useEffect(() => {
    let live = true;
    void getTelemetry()
      .then((r) => live && setT(r))
      // A failed read is not an empty run. Keeping the last report is the
      // difference between "nothing measured" and "the server blinked".
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [epoch]);

  if (!t || !t.available) return null;

  const pct =
    t.context_fraction != null ? `${percent(t.context_fraction, 1)}` : null;

  return (
    <div className="telemetry">
      <span className="tl-item">
        <b>{t.tokens_per_s != null ? t.tokens_per_s.toFixed(1) : "—"}</b> tok/s
        {/* One generation is one sample, so the thing it depended on travels
            with it rather than being quoted as a property of the model. */}
        <span className="meta">
          {" "}
          over {t.generated_tokens} tokens, {t.prompt_tokens}-token prompt
        </span>
      </span>

      <span className="tl-item">
        prompt{" "}
        <b>{t.prompt_ms != null ? `${t.prompt_ms.toFixed(0)}ms` : "—"}</b>
        <span className="meta"> · decode </span>
        <b>{t.decode_ms != null ? `${t.decode_ms.toFixed(0)}ms` : "—"}</b>
      </span>

      <span className="tl-item">
        peak{" "}
        <b>{t.peak_bytes != null ? bytes(t.peak_bytes) : "could not measure"}</b>
        {/* Not "VRAM used". The allocator's view is not the driver's, and
            other processes are invisible to it. */}
        <span className="meta"> allocated by PyTorch</span>
      </span>

      <span className="tl-item">
        context <b>{t.context_used.toLocaleString()}</b>
        {t.context_limit != null ? (
          <span className="meta">
            {" "}
            / {t.context_limit.toLocaleString()} ({pct})
          </span>
        ) : (
          <span className="meta"> · no usable limit reported</span>
        )}
      </span>

      {/* The differentiated line. Charged to this tool, not to the model. */}
      {t.introspection_bytes != null && (
        <span className="tl-item tl-cost" title={t.introspection_note}>
          introspection <b>{bytes(t.introspection_bytes)}</b>
          <span className="meta"> attention scores ModelMRI asks for</span>
        </span>
      )}

      <span className="spacer" />
      <span className="meta">
        {t.device} · {t.dtype}
      </span>
      {t.notes.length > 0 && (
        <span className="meta tl-note" title={t.notes.join(" · ")}>
          {t.notes.length} note{t.notes.length === 1 ? "" : "s"}
        </span>
      )}
    </div>
  );
}
