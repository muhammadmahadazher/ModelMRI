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

/** Shared across every mounted copy. Keyed by epoch so a model change
 *  invalidates it and nothing else does. */
let cached: { epoch: number; promise: Promise<SessionInfo> } | null = null;

function sessionFor(epoch: number): Promise<SessionInfo> {
  if (!cached || cached.epoch !== epoch) {
    cached = { epoch, promise: getSession() };
  }
  return cached.promise;
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
export function useModelReady(epoch: number): boolean | null {
  const [ready, setReady] = useState<boolean | null>(null);
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
  }, [epoch]);
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
  }, [epoch]);

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
