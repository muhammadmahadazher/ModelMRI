import { useEffect, useRef } from "react";

/** Flash a panel's specular scan when genuinely NEW data lands.
 *
 *  Keyed on a payload identity, not on render. A panel that animates every
 *  time React re-runs is not telling you anything — you learn to ignore it,
 *  which is worse than no motion at all. Pass something that changes exactly
 *  when the content does: a generation epoch, a frame index, a trace id.
 *
 *  Returns a ref to put on the panel element.
 */
export function useScanOnData<T>(payloadId: T) {
  const ref = useRef<HTMLDivElement>(null);
  const seen = useRef<T | null>(null);

  useEffect(() => {
    if (payloadId == null || seen.current === payloadId) return;
    const first = seen.current === null;
    // Record the payload BEFORE checking the element. These panels render
    // null until their metadata arrives, so on the first pass ref.current is
    // null; bailing here without bookkeeping left seen===null, and the next
    // payload was then mistaken for the first sight and skipped too. The
    // result was that the first genuinely-new data never scanned.
    seen.current = payloadId;
    const el = ref.current;
    if (first || !el) return;

    el.dataset.scan = "";
    const done = () => {
      delete el.dataset.scan;
    };
    el.addEventListener("animationend", done, { once: true });
    return () => el.removeEventListener("animationend", done);
  }, [payloadId]);

  return ref;
}
