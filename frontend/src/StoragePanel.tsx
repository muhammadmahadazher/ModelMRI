import { useEffect, useRef, useState } from "react";
import { errorText, getPaths, PathInfo } from "./api";

/** Where ModelMRI reads and writes, on demand.
 *
 *  A tool that puts gigabytes on your disk should be able to say where. This
 *  used to require reading the source: every location was spelled out at its
 *  point of use, and the HuggingFace cache was derived six separate times.
 *
 *  Collapsed by default — most sessions never need it — and it fetches only
 *  when opened, so the page load stays as inert as every other panel.
 */
const LABELS: Record<string, string> = {
  hf_hub_cache: "models download to",
  hf_home: "huggingface root",
  data: "traces database",
  config: "settings & token",
  cache: "scratch",
  cwd: "started in",
  legacy: "older location, still read",
  override: "MODELMRI_HOME",
};

const ORDER = ["hf_hub_cache", "hf_home", "data", "config", "cache", "cwd", "legacy", "override"];

export default function StoragePanel() {
  const [open, setOpen] = useState(false);
  const [info, setInfo] = useState<PathInfo | null>(null);
  const [err, setErr] = useState("");
  const panel = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open || info) return;
    let live = true;
    void getPaths()
      .then((p) => live && setInfo(p))
      .catch((e) => live && setErr(errorText(e)));
    return () => {
      live = false;
    };
  }, [open, info]);

  // Escape closes it, like every other transient surface here.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <>
      <button
        className="ghost sm"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        title="Every directory ModelMRI reads or writes"
      >
        {open ? "Hide storage" : "Where things live"}
      </button>

      {open && (
        <div className="storage" ref={panel} role="region" aria-label="storage locations">
          {err && <div className="hint err">{err}</div>}
          {!info && !err && <div className="meta">reading…</div>}
          {info && (
            <>
              <dl className="readout">
                {ORDER.map((key) => {
                  const value = (info as unknown as Record<string, string | null>)[key];
                  if (!value) return null;
                  return (
                    <div className="storage-row" key={key}>
                      <dt>{LABELS[key] ?? key}</dt>
                      <dd title={value}>{value}</dd>
                    </div>
                  );
                })}
              </dl>
              <div className="hint">
                change any of it with <code>MODELMRI_HOME</code>,{" "}
                <code>MODELMRI_MODELS_DIR</code>, <code>MODELMRI_TRACE_DIR</code>, or
                HuggingFace's own <code>HF_HOME</code> / <code>HF_HUB_CACHE</code> ·
                same list from <code>modelmri where</code>
              </div>
            </>
          )}
        </div>
      )}
    </>
  );
}
