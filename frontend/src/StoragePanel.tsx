// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

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
  models_dirs: "your own models",
  trace_db: "trace database",
  hub_token: "huggingface token",
  undelivered_traces: "undelivered traces",
  data: "data directory",
  config: "config directory",
  cache: "scratch",
  cwd: "started in",
  legacy: "older location, still read",
  override: "MODELMRI_HOME",
  // Both were in the payload and in neither list, so they were dropped by the
  // renderer without a word — and this is the panel somebody opens to answer
  // "where did my download actually go".
  models_home: "models kept in",
  inherited_caches: "also read from",
};

const ORDER = [
  "hf_hub_cache",
  "hf_home",
  // `models_home` is where MODELMRI_MODELS_DIR points and `inherited_caches`
  // is every other root this process will read from. `paths.py` has published
  // both all along; `ORDER` decides what is drawn, and neither was in it, so
  // the one panel that answers "where do my models live" answered with two of
  // its own fields missing. The renderer already joins a list and already
  // skips an empty value, so nothing else had to change.
  "models_home",
  "inherited_caches",
  "models_dirs",
  "trace_db",
  "hub_token",
  "undelivered_traces",
  "data",
  "config",
  "cache",
  "cwd",
  "legacy",
  "override",
];

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
                  const raw = (
                    info as unknown as Record<string, string | string[] | null>
                  )[key];
                  // MODELMRI_MODELS_DIR takes a list; an empty one means the
                  // user never set it, so the row is noise rather than news.
                  const value = Array.isArray(raw) ? raw.join(" · ") : raw;
                  if (!value) return null;
                  return (
                    <div className="storage-row" key={key}>
                      <dt>{LABELS[key] ?? key}</dt>
                      <dd title={value}>{value}</dd>
                    </div>
                  );
                })}
              </dl>
              {/* The static demo bakes placeholders for every row above and
                  has always sent this sentence explaining that. It was typed
                  nowhere and rendered nowhere, so a visitor saw seven rows of
                  `<your data dir>` with nothing to say why. */}
              {info.demo_note && <div className="hint">{info.demo_note}</div>}
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
