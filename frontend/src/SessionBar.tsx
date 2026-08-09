import { useEffect, useRef, useState } from "react";
import { closeSession, errorText, openSession, SessionState } from "./api";
import { autoOpenPath, VIEWER } from "./viewer";

/** Open someone else's analysis, or put yours down.
 *
 *  Until now everything here was ephemeral. You find the head that moves the
 *  subject token and the only way to show anyone is a screenshot, which they
 *  cannot explore. A `.mri` is the observation without the model — tokens,
 *  attention, the generation — so it opens on a laptop with no GPU and the
 *  same panels drive it.
 *
 *  Quiet when nothing is open: one small button, because most sessions never
 *  need it. Loud when something is, because looking at a recording and
 *  thinking it is your own live model is the single worst thing this feature
 *  could cause.
 */
export default function SessionBar({
  session,
  onChange,
}: {
  session: SessionState;
  onChange: (s: SessionState) => void;
}) {
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const take = async (file: File | undefined) => {
    if (!file) return;
    setErr("");
    setBusy(true);
    try {
      onChange(await openSession(await file.arrayBuffer()));
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  // `modelmri open somebody.mri` serves the file next to the viewer and
  // links to it, so the analysis is already on screen when the tab opens —
  // nobody should have to find and drop a file they just named on the
  // command line.
  useEffect(() => {
    const path = autoOpenPath();
    if (!path || session.open) return;
    let live = true;
    setBusy(true);
    void fetch(path)
      .then((r) => {
        if (!r.ok) throw new Error(`could not read ${path} (HTTP ${r.status})`);
        return r.arrayBuffer();
      })
      .then((buf) => openSession(buf))
      .then((s) => live && onChange(s))
      .catch((e) => live && setErr(errorText(e)))
      .finally(() => live && setBusy(false));
    return () => {
      live = false;
    };
    // Once, on mount. Re-running on `session` would reopen it after a close.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // A file dropped anywhere on the page should open, not navigate away from
  // the app — which is what the browser does by default, discarding whatever
  // was on screen.
  useEffect(() => {
    const over = (e: DragEvent) => {
      if (!e.dataTransfer?.types.includes("Files")) return;
      e.preventDefault();
      setDragging(true);
    };
    const leave = (e: DragEvent) => {
      if (e.relatedTarget === null) setDragging(false);
    };
    const drop = (e: DragEvent) => {
      if (!e.dataTransfer?.types.includes("Files")) return;
      e.preventDefault();
      setDragging(false);
      void take(e.dataTransfer.files[0]);
    };
    window.addEventListener("dragover", over);
    window.addEventListener("dragleave", leave);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragover", over);
      window.removeEventListener("dragleave", leave);
      window.removeEventListener("drop", drop);
    };
  }, []);

  const shut = async () => {
    setBusy(true);
    try {
      onChange(await closeSession());
      setErr("");
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  const meta = session.meta;

  return (
    <>
      {dragging && (
        <div className="drop-veil" aria-hidden="true">
          <span>drop a .mri to open it</span>
        </div>
      )}

      <input
        ref={input}
        type="file"
        accept=".mri"
        hidden
        onChange={(e) => {
          void take(e.target.files?.[0]);
          e.target.value = ""; // so re-picking the same file fires again
        }}
      />

      {session.open ? (
        <div className="panel replay" role="status">
          <div className="sect">
            <span className="dot d-replay" />
            <h2 className="h-replay">VIEWING A SHARED SESSION</h2>
            <span className="rule" />
            <button className="ghost sm" onClick={() => void shut()} disabled={busy}>
              Close
            </button>
          </div>

          <dl className="readout">
            <div className="storage-row">
              <dt>model</dt>
              <dd>
                {meta?.model ?? "unknown"}
                {meta?.dtype ? ` · ${meta.dtype}` : ""}
                {meta?.device ? ` · ${meta.device}` : ""}
              </dd>
            </div>
            {meta?.note && (
              <div className="storage-row">
                <dt>note</dt>
                <dd>{meta.note}</dd>
              </div>
            )}
            <div className="storage-row">
              <dt>captured</dt>
              <dd>
                {meta?.created_at ? new Date(meta.created_at).toLocaleString() : "—"}
                {meta?.modelmri ? ` · ModelMRI ${meta.modelmri}` : ""}
              </dd>
            </div>
            <div className="storage-row">
              <dt>contains</dt>
              <dd>
                {session.n_tokens} tokens · {session.n_slices} attention maps
                {meta?.scope ? ` — ${meta.scope}` : ""}
              </dd>
            </div>
          </dl>

          <div className="replay-text">
            <span className="meta">prompt</span>
            <p>{session.prompt || "—"}</p>
            <span className="meta">generation</span>
            <p>{session.generation || "—"}</p>
          </div>

          <div className="hint">
            These numbers were recorded on someone else's machine. Nothing is
            loaded here — you are reading their run, not producing one.
            {meta?.precision ? ` ${meta.precision}.` : ""}
          </div>
          {err && <div className="hint err">{err}</div>}
        </div>
      ) : (
        VIEWER ? (
          // The viewer's entire purpose. A ghost button in a corner would be
          // burying the one action this page exists for.
          <button
            className="dropzone"
            onClick={() => input.current?.click()}
            disabled={busy}
          >
            <span className="dropzone-mark" aria-hidden="true">
              ⌁
            </span>
            <strong>{busy ? "opening…" : "Drop a .mri here, or click to choose one"}</strong>
            <span className="meta">
              read in your browser · never uploaded · no model needed
            </span>
            {err && <span className="hint err">{err}</span>}
          </button>
        ) : (
          <div className="session-open-row">
            <button
              className="ghost sm"
              onClick={() => input.current?.click()}
              disabled={busy}
              title="Open a .mri someone shared with you — no model required"
            >
              {busy ? "opening…" : "Open a shared analysis (.mri)"}
            </button>
            {err && <span className="hint err">{err}</span>}
          </div>
        )
      )}
    </>
  );
}
