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
    const el = ref.current;
    // Skip the first sight of a payload: that is the panel appearing, which
    // its own entrance already covers. The scan is for the second onward.
    if (!el || payloadId == null || seen.current === payloadId) return;
    const first = seen.current === null;
    seen.current = payloadId;
    if (first) return;

    el.dataset.scan = "";
    const done = () => {
      delete el.dataset.scan;
    };
    el.addEventListener("animationend", done, { once: true });
    return () => el.removeEventListener("animationend", done);
  }, [payloadId]);

  return ref;
}
