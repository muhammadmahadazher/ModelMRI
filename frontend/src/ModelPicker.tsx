import { useEffect, useRef, useState } from "react";
import {
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
  const [tab, setTab] = useState<"hf" | "ollama">("hf");
  const [auth, setAuth] = useState<HubAuth | null>(null);
  const [tokenInput, setTokenInput] = useState("");
  const [showSignIn, setShowSignIn] = useState(false);
  const [q, setQ] = useState("");
  const [models, setModels] = useState<HubModel[] | null>(null);
  const [ollama, setOllama] = useState<OllamaState | null>(null);
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");
  const debounce = useRef<number | undefined>(undefined);

  useEffect(() => {
    if (!open) return;
    let live = true;
    void getHubAuth().then((a) => live && setAuth(a));
    void getOllama().then((o) => live && setOllama(o));
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
        .catch((e) => live && setErr(String(e)));
    }, 280);
    return () => {
      live = false;
    };
  }, [open, tab, q, auth?.signed_in]);

  // Esc closes, like every other sheet on the platform
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

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
      <div className="sheet glass" onClick={(e) => e.stopPropagation()}>
        <div className="sheet-head">
          <div className="seg">
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

        {tab === "hf" ? (
          <>
            <input
              className="combo search"
              placeholder="Search HuggingFace models…  (empty = curated picks that fit your GPU)"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              autoFocus
            />
            <div className="model-list">
              {models === null && <div className="meta pad">searching…</div>}
              {models?.length === 0 && <div className="meta pad">no matches</div>}
              {models?.map((m) => {
                const locked = m.gated && !auth?.signed_in;
                return (
                  <button
                    key={m.id}
                    className={`model-row ${m.id === current ? "sel" : ""} ${locked ? "locked" : ""}`}
                    onClick={() => !locked && onPick(m.id, "hf")}
                    title={locked ? "Gated - sign in and accept the license" : m.id}
                  >
                    <span className="mid">{m.id}</span>
                    {m.params && <span className="chip">{m.params}</span>}
                    {m.gated && (
                      <span className={`chip ${locked ? "warn" : "ok"}`}>
                        {locked ? "gated" : "gated ✓"}
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
