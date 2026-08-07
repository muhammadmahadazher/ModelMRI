import { useEffect, useRef, useState } from "react";
import {
  Discovery,
  getDiscovered,
  getHubAuth,
  getHubModels,
  getOllama,
  HubAuth,
  HubModel,
  hubSignIn,
  hubSignOut,
  OllamaState,
  pullOllama,
} from "./api";

interface Props {
  open: boolean;
  onClose: () => void;
  onPick: (id: string, source: "hf" | "ollama") => void;
  current: string;
}

/** Model browser: search HuggingFace, sign in for gated repos, or pick /
 *  pull an Ollama model. Presented as a liquid-glass sheet. */
export default function ModelPicker({ open, onClose, onPick, current }: Props) {
  const [tab, setTab] = useState<"local" | "hf" | "ollama">("local");
  const [disco, setDisco] = useState<Discovery | null>(null);
  const [auth, setAuth] = useState<HubAuth | null>(null);
  const [tokenInput, setTokenInput] = useState("");
  const [showSignIn, setShowSignIn] = useState(false);
  const [q, setQ] = useState("");
  const [models, setModels] = useState<HubModel[] | null>(null);
  const [ollama, setOllama] = useState<OllamaState | null>(null);
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");
  const debounce = useRef<number | undefined>(undefined);
  const sheetRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  // onClose is an inline arrow in the parent, so its identity changes on every
  // parent render. Depending on it directly would re-run the modal effect --
  // re-capturing the opener, re-locking scroll and yanking focus back to the
  // search box mid-keystroke. The effect depends on `open` alone.
  const closeRef = useRef(onClose);
  closeRef.current = onClose;

  useEffect(() => {
    if (!open) return;
    let live = true;
    void getHubAuth().then((a) => live && setAuth(a));
    void getOllama().then((o) => live && setOllama(o));
    void getDiscovered()
      .then((d) => live && setDisco(d))
      .catch(() => live && setDisco({ models: [], roots: [], truncated: false }));
    return () => {
      live = false;
    };
  }, [open]);

  useEffect(() => {
    if (!open || tab !== "hf") return;
    let live = true;
    setModels(null);
    window.clearTimeout(debounce.current);
    debounce.current = window.setTimeout(() => {
      void getHubModels(q)
        .then((m) => live && setModels(m))
        .catch((e) => {
          if (!live) return;
          setErr(e instanceof Error ? e.message : String(e));
          // models===null is the "searching" sentinel, so a failure that only
          // sets the error leaves the list spinning forever. An empty list is
          // the honest terminal state.
          setModels([]);
        });
    }, 280);
    return () => {
      live = false;
    };
  }, [open, tab, q, auth?.signed_in]);

  // Modal behaviour, per the ARIA dialog pattern: Esc closes, Tab cannot
  // escape into the page behind the scrim, the page behind cannot scroll,
  // and focus goes back where it came from on close. Without the trap, one
  // Tab past the last model row landed on the topbar links underneath.
  useEffect(() => {
    if (!open) return;
    // Captured before we move focus ourselves. React's autoFocus fires during
    // commit, i.e. before this effect, so an autoFocus'd input would have made
    // this read the input instead of the button that opened the sheet -- and
    // closing would then restore focus to a node that no longer exists, which
    // means body. Hence the explicit focus() below rather than autoFocus.
    const opener = document.activeElement as HTMLElement | null;
    searchRef.current?.focus();
    const { overflow } = document.body.style;
    document.body.style.overflow = "hidden";

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        closeRef.current();
        return;
      }
      if (e.key !== "Tab") return;
      const focusable = sheetRef.current?.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])',
      );
      if (!focusable?.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const here = document.activeElement;
      if (e.shiftKey && (here === first || !sheetRef.current?.contains(here))) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && here === last) {
        e.preventDefault();
        first.focus();
      }
    };

    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = overflow;
      opener?.focus?.();
    };
  }, [open]);

  if (!open) return null;

  async function doSignIn() {
    setBusy("auth");
    setErr("");
    try {
      setAuth(await hubSignIn(tokenInput.trim()));
      setTokenInput("");
      setShowSignIn(false);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  }

  async function doPull(name: string) {
    setBusy(`pull:${name}`);
    setErr("");
    try {
      await pullOllama(name);
      setOllama(await getOllama());
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  }

  return (
    <div className="sheet-scrim" onClick={onClose}>
      <div
        className="sheet glass"
        ref={sheetRef}
        role="dialog"
        aria-modal="true"
        aria-label="Choose a model"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="sheet-head">
          <div className="seg">
            <button
              className={tab === "local" ? "on" : ""}
              onClick={() => setTab("local")}
            >
              On this machine{disco ? ` · ${disco.models.length}` : ""}
            </button>
            <button className={tab === "hf" ? "on" : ""} onClick={() => setTab("hf")}>
              HuggingFace
            </button>
            <button
              className={tab === "ollama" ? "on" : ""}
              onClick={() => setTab("ollama")}
            >
              Ollama {ollama?.up ? `· ${ollama.installed?.length ?? 0}` : "· off"}
            </button>
          </div>
          <span className="spacer" />
          {tab === "hf" &&
            (auth?.signed_in ? (
              <span className="pill accel gpu" title={`token source: ${auth.source}`}>
                <i className="accel-dot" />
                {auth.user}
                <button
                  className="linkish"
                  onClick={() => void hubSignOut().then(setAuth)}
                >
                  sign out
                </button>
              </span>
            ) : (
              <button className="ghost sm" onClick={() => setShowSignIn((s) => !s)}>
                Sign in for gated models
              </button>
            ))}
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>

        {tab === "hf" && showSignIn && !auth?.signed_in && (
          <div className="signin glass-inset">
            <p className="meta">
              Paste a <b>read</b> access token from{" "}
              <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">
                huggingface.co/settings/tokens
              </a>
              . It is stored only on this machine and unlocks gated models you have
              accepted (Gemma, Llama…) plus your private repos.
            </p>
            <div className="row">
              <input
                className="combo grow"
                type="password"
                placeholder="hf_..."
                value={tokenInput}
                onChange={(e) => setTokenInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && void doSignIn()}
              />
              <button className="cta" onClick={() => void doSignIn()} disabled={busy !== ""}>
                {busy === "auth" ? "Checking…" : "Sign in"}
              </button>
            </div>
          </div>
        )}

        {tab === "local" ? (
          <div className="model-list stagger" role="listbox" aria-label="Models on this machine">
            {disco === null && <div className="meta pad">scanning…</div>}
            {disco?.models.length === 0 && (
              <div className="meta pad">
                Nothing found under {disco.roots.join(", ") || "the working directory"}.
                Start ModelMRI from the folder your models live in, or set
                MODELMRI_MODELS_DIR.
              </div>
            )}
            {disco?.models.map((m, i) => (
              <button
                key={m.path}
                style={{ ["--i" as string]: i }}
                className={`model-row ${m.id === current ? "sel" : ""} ${m.loadable ? "" : "locked"}`}
                role="option"
                aria-selected={m.id === current}
                onClick={() => m.loadable && onPick(m.id, "hf")}
                title={m.loadable ? m.path : `${m.path} — ${m.note}`}
              >
                <span className="mid">{m.name}</span>
                <span className={`chip ${m.kind === "gguf" ? "warn" : ""}`}>
                  {m.kind === "hf-cache" ? "cached" : m.kind === "folder" ? "folder" : "gguf"}
                </span>
                <span className="spacer" />
                <span className="meta">{m.size_gb.toFixed(2)} GB</span>
              </button>
            ))}
            {disco?.truncated && (
              <div className="meta pad">
                Scan stopped early — this drive is large. Set MODELMRI_MODELS_DIR to
                point straight at your models folder.
              </div>
            )}
            {disco && disco.models.length > 0 && (
              <div className="meta pad">
                looked in: {disco.roots.join(" · ")}
              </div>
            )}
          </div>
        ) : tab === "hf" ? (
          <>
            <input
              className="combo search"
              placeholder="Search HuggingFace models…  (empty = curated picks that fit your GPU)"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              ref={searchRef}
            />
            <div className="model-list stagger" role="listbox" aria-label="Models">
              {/* Results arrive asynchronously, so a screen reader is told the
                  count rather than left guessing whether anything happened. */}
              <div className="sr-only" role="status" aria-live="polite">
                {models === null
                  ? "Searching models"
                  : `${models.length} model${models.length === 1 ? "" : "s"} found`}
              </div>
              {models === null && <div className="meta pad">searching…</div>}
              {models?.length === 0 && (
                <div className="meta pad">
                  {err ? "search failed — see the message below" : "no matches"}
                </div>
              )}
              {models?.map((m) => {
                // Trust the server's verified answer. `gated && !signed_in`
                // was the same wrong assumption the API used to make: being
                // signed in is not the same as having accepted the licence.
                const locked = !m.usable;
                return (
                  <button
                    key={m.id}
                    className={`model-row ${m.id === current ? "sel" : ""} ${locked ? "locked" : ""}`}
                    role="option"
                    aria-selected={m.id === current}
                    onClick={() =>
                      locked
                        ? // A dead row tells you nothing. Send people to the
                          // page where the licence is actually accepted.
                          window.open(
                            `https://huggingface.co/${m.id}`,
                            "_blank",
                            "noopener,noreferrer",
                          )
                        : onPick(m.id, "hf")
                    }
                    title={
                      locked
                        ? auth?.signed_in
                          ? `Accept the licence on huggingface.co/${m.id}, then search again`
                          : "Sign in, then accept this model's licence"
                        : m.id
                    }
                  >
                    <span className="mid">{m.id}</span>
                    {m.params && <span className="chip">{m.params}</span>}
                    {m.gated && (
                      <span className={`chip ${locked ? "warn" : "ok"}`}>
                        {locked
                          ? auth?.signed_in
                            ? "accept licence ↗"
                            : "sign in"
                          : "gated ✓"}
                      </span>
                    )}
                    <span className="spacer" />
                    {m.downloads > 0 && (
                      <span className="meta">{fmt(m.downloads)} downloads</span>
                    )}
                  </button>
                );
              })}
            </div>
          </>
        ) : (
          <div className="model-list">
            {!ollama?.up && (
              <div className="meta pad">
                Ollama is not running. Install it from{" "}
                <a href="https://ollama.com" target="_blank" rel="noreferrer">
                  ollama.com
                </a>
                , start it, then reopen this panel. Ollama runs any open model as
                text — attention and features need a HuggingFace model.
              </div>
            )}
            {ollama?.installed?.map((m) => (
              <button
                key={m.name}
                className={`model-row ${m.name === current ? "sel" : ""}`}
                onClick={() => onPick(m.name, "ollama")}
              >
                <span className="mid">{m.name}</span>
                {m.params && <span className="chip">{m.params}</span>}
                {m.quant && <span className="chip">{m.quant}</span>}
                <span className="spacer" />
                <span className="meta">{m.size_gb} GB</span>
              </button>
            ))}
            {ollama?.up && (
              <>
                <div className="meta pad">not installed — pull one:</div>
                {ollama.suggested
                  ?.filter((s) => !ollama.installed?.some((i) => i.name === s.name))
                  .map((s) => (
                    <div key={s.name} className="model-row static">
                      <span className="mid">{s.name}</span>
                      <span className="chip">{s.size}</span>
                      <span className="meta">{s.note}</span>
                      <span className="spacer" />
                      <button
                        className="ghost sm"
                        onClick={() => void doPull(s.name)}
                        disabled={busy !== ""}
                      >
                        {busy === `pull:${s.name}` ? "Pulling…" : "Pull"}
                      </button>
                    </div>
                  ))}
              </>
            )}
          </div>
        )}

        {err && <div className="hint err">{err}</div>}
      </div>
    </div>
  );
}

function fmt(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}k`;
  return String(n);
}
