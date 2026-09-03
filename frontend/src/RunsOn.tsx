// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

import { useEffect, useState } from "react";
import { getSession, SessionInfo } from "./api";

/**
 * Which model this panel is about.
 *
 * WHY IT HAD TO BE ADDED AT ALL
 *
 * The maintainer asked, of five panels in a row: "what model is it using? its
 * not clear." It was not clear because it was not said. The model is chosen
 * once at the top of the page and every measurement panel below silently
 * inherits it — they are handed `epoch`, a counter that changes when a model
 * loads, and nothing else, so they could not have named it even if they tried.
 *
 * The one panel that ever mentioned a model was Features, and only because it
 * has to refuse: "No sparse autoencoder exists for Qwen/Qwen3-1.7B." A tool
 * whose entire argument is that a number without its conditions cannot be
 * checked should not make the reader infer the most important condition of
 * all from a control two screens up.
 *
 * WHY ONE REQUEST RATHER THAN ONE PER PANEL
 *
 * Six panels mounting this is six identical `/api/session` calls on every
 * load and on every model change. The answer is shared and keyed on `epoch`,
 * so the panels re-read together when the model actually changes and not
 * otherwise.
 */

/** Shared across every mounted copy.
 *
 *  KEYED ON EPOCH ALONE, THIS WAS WRONG IN BOTH DIRECTIONS. `epoch` counts
 *  GENERATIONS — `Playground` bumps it when one finishes — and it is set to
 *  the literal 0 after a load. On a fresh page nothing has generated, so epoch
 *  is already 0, `setEpoch(0)` is a no-op, no `[epoch]` effect re-runs, and
 *  `sessionFor(0)` hands back the promise resolved before the model existed.
 *
 *  MEASURED: fresh page, press Load, wait for "Loaded ✓" in the RUN panel —
 *  and six panels below it still read "Nothing is loaded, so pick one in Run
 *  at the top of the page first." The page telling you to do the thing you
 *  just did.
 *
 *  The mirror is worse: after Unload, `App` bumps `resetKey` and the remount
 *  lands on epoch 0 again, but this cache is module-level and survives it — so
 *  the panels advertise "measures <model> · cuda · bfloat16" with live buttons
 *  over freed memory.
 *
 *  `epoch` cannot simply be made monotonic: `epoch > 0` gates the telemetry
 *  bar and the attention panel, and bumping it on a load would mount both for
 *  a run that never happened. So a model change says so explicitly, and every
 *  mounted copy re-reads when it does.
 */
let cached: { key: string; promise: Promise<SessionInfo> } | null = null;
let version = 0;
const listeners = new Set<() => void>();

/** The resident model changed — drop the shared answer and re-read.
 *
 *  Called wherever a model is loaded, unloaded or swapped. Cheap: it is one
 *  request shared by every mounted panel, which is the reason the cache
 *  exists at all.
 */
export function invalidateSession(): void {
  version += 1;
  cached = null;
  listeners.forEach((notify) => notify());
}

function sessionFor(epoch: number): Promise<SessionInfo> {
  const key = `${epoch}:${version}`;
  if (!cached || cached.key !== key) {
    cached = { key, promise: getSession() };
  }
  return cached.promise;
}

/** Re-render when ANY resident model changes — text, image or robot policy.
 *
 *  Exported because the header needs it too. It reads `/api/session`, which
 *  reports all three, and nothing was telling it to re-read: loading a
 *  diffusion pipeline left "no model loaded" in the top bar with 3.3 GB
 *  resident and every control in that panel live.
 *
 *  A cleared cache alone is not enough: the effects below are keyed on
 *  `[epoch]`, so nothing would re-run to notice it was cleared.
 */
export function useSessionVersion(): number {
  const [seen, setSeen] = useState(version);
  useEffect(() => {
    const notify = () => setSeen(version);
    listeners.add(notify);
    // Between render and subscribe, an invalidation can land — pick it up
    // rather than waiting for the next one.
    notify();
    return () => {
      listeners.delete(notify);
    };
  }, []);
  return seen;
}

/**
 * Is there a live model these panels can measure?
 *
 * Shares `RunsOn`'s cache, so asking costs no extra request — the badge and
 * the button it disables are reading one answer.
 *
 * WHY THE BUTTONS NEEDED IT. `Playground` gates these panels on
 * `introspectable = model?.device !== "ollama"`, which is TRUE when `model`
 * is null: `undefined !== "ollama"`. So with nothing loaded at all, four
 * measurement panels mounted with their forms filled in and their buttons
 * live, and every click answered "No model loaded — pick one first".
 *
 * The panels stay mounted deliberately. Hiding them would make the tool look
 * emptier and harder to learn, which is the opposite of what is wanted —
 * `RunsOn` already prints the sentence, and this is what stops the control
 * beneath it from being clickable anyway.
 */
/** WHICH model is resident, or null when none is.
 *
 *  `epoch` counts GENERATIONS, and the note above records why it cannot be
 *  made to count loads instead: `epoch > 0` gates the telemetry bar and the
 *  attention panel, so bumping it on a load would mount both for a run that
 *  never happened.
 *
 *  That left every panel holding a MEASUREMENT with no signal to clear on.
 *  `Playground.ensureLoaded` ends with `setEpoch(0)`, which is a no-op when
 *  epoch is already 0 — and it is, for every measurement that needs no
 *  generation: a probe is fitted to your own examples, a patchscope reads a
 *  residual stream directly. So load A, probe, load B left A's curve on
 *  screen under B's name, which is the exact substitution those panels'
 *  own clear-effects were written to prevent.
 *
 *  The GGUF marker is part of the identity: loading one swaps the resident
 *  model while `hf_id` can stay exactly where it was.
 */
export function useModelIdentity(epoch: number): string | null {
  const [id, setId] = useState<string | null>(null);
  const seen = useSessionVersion();
  useEffect(() => {
    let live = true;
    void sessionFor(epoch)
      .then((s) => {
        if (!live) return;
        const m = s.model;
        setId(m?.loaded ? `${m.hf_id ?? ""}|${m.gguf?.plan.path ?? ""}` : null);
      })
      // Unknown stays unknown, deliberately: throwing a measurement away
      // because one status request failed would lose real work over a
      // network blip. `useModelReady` answers `null` for the same reason.
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [epoch, seen]);
  return id;
}

export function useModelReady(epoch: number): boolean | null {
  const [ready, setReady] = useState<boolean | null>(null);
  const seen = useSessionVersion();
  useEffect(() => {
    let live = true;
    void sessionFor(epoch)
      // `null` while unknown, so a button is never disabled on a failed
      // fetch — that would take a working control away over a network blip.
      .then((s) => live && setReady(!!s.model?.loaded))
      .catch(() => live && setReady(null));
    return () => {
      live = false;
    };
  }, [epoch, seen]);
  return ready;
}

export default function RunsOn({
  epoch,
  /** What the panel does when nothing is loaded. Some refuse outright; some
   *  read a recorded `.mri` and are perfectly useful with no model at all,
   *  and telling those to "load a model" would be wrong. */
  needsModel = true,
}: {
  epoch: number;
  needsModel?: boolean;
}) {
  const [info, setInfo] = useState<SessionInfo | null>(null);
  const seen = useSessionVersion();

  useEffect(() => {
    let live = true;
    void sessionFor(epoch)
      .then((s) => live && setInfo(s))
      // A panel must not break because the badge could not resolve. Silence
      // here means the line is absent, not that the model is unknown-and-said
      // -to-be — those are different claims and only one of them is safe to
      // make from a failed fetch.
      .catch(() => live && setInfo(null));
    return () => {
      live = false;
    };
  }, [epoch, seen]);

  if (!info) return null;
  const m = info.model;

  if (!m?.loaded) {
    if (!needsModel) return null;
    return (
      <p className="meta runs-on">
        Nothing is loaded — this measures a model, so pick one in{" "}
        <b>Run</b> at the top of the page first.
      </p>
    );
  }

  return (
    <p className="meta runs-on">
      measures{" "}
      <b className="runs-on-id" title={m.hf_id ?? undefined}>
        {m.hf_id}
      </b>
      {/* Device and dtype ride along because they are half of what makes a
          number reproducible — the same head on the same prompt gives
          different digits in float32 and bfloat16, and this project refuses
          to publish a figure without them. */}
      {m.device && ` · ${m.device}`}
      {m.dtype && ` · ${m.dtype}`}
    </p>
  );
}
