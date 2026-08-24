import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  Discovery,
  errorText,
  getDiscovered,
  getHubAuth,
  getHubModels,
  getOllama,
  getPullProgress,
  getOllamaSuggested,
  HubAuth,
  HubModel,
  hubSignIn,
  hubSignOut,
  OllamaResolved,
  OllamaState,
  LoadProgress,
  OllamaSuggestion,
  pullOllama,
  resolveOllama,
} from "./api";
// `gb` as well as `remaining`. This file took the duration helper from
// LoadBar and re-implemented the byte one, reproducing the bug LoadBar's own
// comment records as fixed: a pull under half a megabyte, and the whole first
// stretch of any live download, read "0 MB". Two meters for the same download
// cannot disagree if there is one formatter.
import { gb, remaining } from "./LoadBar";

/** What a pull is doing, right now. Bytes, a bar, and a time — the three
 *  things somebody watching a download wants and the picker used to answer
 *  with the word "Pulling…". */
function PullProgress({ p }: { p: LoadProgress | null }) {
  if (!p || !p.active) return null;
  const pct = p.bytes_total > 0 ? (100 * p.bytes_done) / p.bytes_total : null;
  return (
    <div className="pull-progress" role="status" aria-live="polite">
      <div className={`pull-track ${pct === null ? "indeterminate" : ""}`}>
        <div
          className="pull-fill"
          style={pct === null ? undefined : { width: `${pct}%` }}
        />
      </div>
      <span className="meta">
        {pct !== null
          ? `${gb(p.bytes_done)} of ${gb(p.bytes_total)} · ${gb(
              p.bytes_total - p.bytes_done,
            )} left`
          : p.detail || "starting"}
        {` · ${p.elapsed_s.toFixed(0)}s`}
        {p.eta_s != null && ` · ~${remaining(p.eta_s)} left`}
      </span>
    </div>
  );
}

/** Is THIS row the thing currently downloading?
 *
 *  Either because this tab clicked Pull, or because the server says a pull of
 *  that name is in flight — which is what makes the bar survive a refresh and
 *  show up in a second tab. */
function isPulling(busy: string, p: LoadProgress | null, name: string): boolean {
  if (busy === `pull:${name}`) return true;
  return Boolean(p?.active && p.hf_id === name);
}

interface Props {
  open: boolean;
  onClose: () => void;
  onPick: (id: string, source: "hf" | "ollama") => void;
  current: string;
}

/** Placeholder rows while an async list loads.
 *
 *  The sheet is a fixed height, so "scanning…" alone left ~600px of empty
 *  glass for as long as the disk walk takes — up to its 6s budget on a synced
 *  drive. Showing the shape of the answer makes the wait legible, and the
 *  widths are staggered so it reads as a list rather than a loading bar.
 */
export function ModelSkeleton({ label }: { label: string }) {
  const widths = [62, 44, 71, 38, 55, 67, 41, 58];
  return (
    <>
      <div className="meta pad skel-label">{label}…</div>
      <div aria-hidden="true">
        {widths.map((w, i) => (
          <div className="skel-row" key={i} style={{ ["--i" as string]: i }}>
            <span className="skel-bar" style={{ width: `${w}%` }} />
            <span className="skel-bar skel-size" />
          </div>
        ))}
      </div>
    </>
  );
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
  // A pull runs inside this sheet, and the sheet covers the load meter on the
  // page behind it. Without this, the one place you can start a nine gigabyte
  // download was also the one place you could not watch it.
  const [pulling, setPulling] = useState<LoadProgress | null>(null);
  const [err, setErr] = useState("");
  // A pull the size guard refused, held with its model so "anyway" retries
  // the right one.
  const [pullWarning, setPullWarning] = useState<{
    name: string;
    message: string;
  } | null>(null);
  // Switching tabs clears the error and the size-guard warning.
  //
  // There is one `err` for the whole sheet, so without this the message from
  // the tab you just left stays on screen under the tab you just opened. On
  // the hosted demo that was plainly wrong rather than merely untidy: search
  // HuggingFace, get "searching HuggingFace is a live call … this page is a
  // static recording", switch to Ollama, and that same HuggingFace sentence
  // sat under the Ollama tab as if it were Ollama's own explanation. Both
  // tabs do refuse here, and they refuse for different reasons — the whole
  // point of writing separate messages was that a visitor could tell which
  // limitation they had hit.
  const openTab = (next: "local" | "hf" | "ollama") => {
    setErr("");
    setPullWarning(null);
    setTab(next);
  };

  // Ollama's "search": a name, resolved against the registry.
  const [ollamaName, setOllamaName] = useState("");
  // Curated Ollama picks, sized live. Fetched only when that tab is opened —
  // eight registry lookups is not a cost to pay for a panel you never visit.
  const [suggestions, setSuggestions] = useState<OllamaSuggestion[]>([]);
  const [resolved, setResolved] = useState<OllamaResolved | null>(null);

  async function doResolve() {
    const name = ollamaName.trim();
    if (!name) return;
    setBusy("resolve");
    setErr("");
    setPullWarning(null);
    try {
      setResolved(await resolveOllama(name));
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }
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

  // Only when the Ollama tab is actually opened, and only once.
  useEffect(() => {
    if (!open || tab !== "ollama" || suggestions.length) return;
    let live = true;
    void getOllamaSuggested()
      .then((s) => live && setSuggestions(s))
      // Offline, the name box still works and the installed list still
      // renders. An empty curated strip is a smaller loss than an error.
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [open, tab, suggestions.length]);

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
          setErr(errorText(e));
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

  // ABOVE `if (!open) return null` — see below. Polled while the sheet is
  // OPEN, not while this tab happens to be the one
  // that started a pull. A download lives on the server: reload the page or
  // open a second tab and the bytes keep arriving, so a bar that only exists
  // in the tab that clicked is a bar that vanishes exactly when somebody
  // refreshes to check on it.
  useEffect(() => {
    if (!open) {
      setPulling(null);
      return;
    }
    let live = true;
    const tick = () => {
      void getPullProgress()
        .then((p) => live && setPulling(p))
        // A failed poll is not a failed pull. Keep the last numbers rather
        // than blanking the bar, which would read as "it stopped".
        .catch(() => undefined);
    };
    tick();
    const id = window.setInterval(tick, 700);
    return () => {
      live = false;
      window.clearInterval(id);
    };
  }, [open]);

  // EVERY hook above this line, without exception. This early return is
  // why: a hook placed below it does not run when the sheet is closed, so
  // opening the sheet renders MORE hooks than the previous render did and
  // React tears the tree down (#310) rather than showing it. Symptom is a
  // picker that never appears, which reads like a dead button rather than
  // like a crash. tests/ui_check.py waits for `.sheet` and is what caught
  // it; it has now caught it twice, in two different components.
  if (!open) return null;

  async function doSignIn() {
    setBusy("auth");
    setErr("");
    try {
      setAuth(await hubSignIn(tokenInput.trim()));
      setTokenInput("");
      setShowSignIn(false);
    } catch (e) {
      setErr(errorText(e));
    } finally {
      setBusy("");
    }
  }


  async function doPull(name: string, confirm = false) {
    setBusy(`pull:${name}`);
    setErr("");
    setPullWarning(null);
    try {
      await pullOllama(name, confirm);
      setOllama(await getOllama());
    } catch (e) {
      const message = errorText(e);
      // The size guard refused. Offer the override only when the server said
      // it is overridable — a disk with no room is not a matter of opinion.
      if (message.includes("Load it anyway")) {
        setPullWarning({ name, message });
      } else {
        setErr(message);
      }
    } finally {
      setBusy("");
    }
  }

  // Portalled to <body>. MEASURED: the scrim is `position: fixed; inset: 0`,
  // and it was rendering 935x546 at (36,136) inside a 1006x626 viewport --
  // the panel's own box. `.panel` carries `transform: matrix(1,0,0,1,0,0)`
  // and `filter: blur(0px)` left over from its entrance animation, and EITHER
  // of those makes a descendant's `fixed` resolve against that ancestor
  // instead of the viewport. So the dim-and-blur only ever covered the panel
  // it was opened from, which is exactly what "only blur in a small part of
  // the background" looks like. An identity transform still creates the
  // containing block, so there is nothing to "turn off" -- the sheet has to
  // leave the panel.
  return createPortal(
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
              onClick={() => openTab("local")}
            >
              On this machine{disco ? ` · ${disco.models.length}` : ""}
            </button>
            <button className={tab === "hf" ? "on" : ""} onClick={() => openTab("hf")}>
              HuggingFace
            </button>
            <button
              className={tab === "ollama" ? "on" : ""}
              onClick={() => openTab("ollama")}
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
                {/* Only for a token this tool owns. `sign_out` deletes
                    ModelMRI's own file and deliberately nothing else -- it has
                    no business removing huggingface-cli's token or unsetting a
                    parent process's environment variable, both of which are
                    shared with every other library in the ecosystem.

                    Offering the button regardless is what made it look broken:
                    the click deleted nothing, the server honestly answered
                    signed_in:true from the surviving source, and the pill
                    re-rendered identically with no feedback of any kind. On
                    the machine this was reported from, ModelMRI's own token
                    file did not exist at all. Where the credential is not
                    ours, name whose it is and how to remove it. */}
                {auth.source === "modelmri" ? (
                  <button
                    className="linkish"
                    onClick={() =>
                      void hubSignOut()
                        .then(setAuth)
                        .catch((e) => setErr(errorText(e)))
                    }
                    title="Remove the token ModelMRI stored"
                  >
                    sign out
                  </button>
                ) : (
                  <span
                    className="hub-source"
                    title={
                      auth.source === "env"
                        ? "This token comes from your environment. Unset HF_TOKEN and restart the server to sign out."
                        : "This token belongs to the HuggingFace CLI. Run `hf auth logout` to sign out."
                    }
                  >
                    via {auth.source === "env" ? "HF_TOKEN" : "huggingface-cli"}
                  </span>
                )}
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
            {disco === null && <ModelSkeleton label="scanning your working directory" />}
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
                {m.loadable ? (
                  <span className="chip">
                    {m.kind === "hf-cache" ? "cached" : m.kind === "folder" ? "folder" : "gguf"}
                  </span>
                ) : (
                  // The reason used to live only in a title attribute, so a
                  // segmentation model and a language model looked identical
                  // and one of them failed minutes later with a tokenizer
                  // traceback. Say it on the row.
                  <span className="why">{m.note}</span>
                )}
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
              {models === null && <ModelSkeleton label="searching the Hub" />}
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
                    {/* The size, before you click. Its absence is how a
                        click here started a 1.5 TB download on a laptop. */}
                    {m.size_gb != null && (
                      <span className={`chip ${m.size_gb > 40 ? "warn" : ""}`}>
                        {m.size_gb >= 1000
                          ? `${(m.size_gb / 1000).toFixed(1)} TB`
                          : `${m.size_gb.toFixed(m.size_gb < 10 ? 1 : 0)} GB`}
                      </span>
                    )}
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
                    {/* `!= null` before the comparison, because `null` is
                        UNKNOWN here and not a zero. With the Hub unreachable
                        every curated row carries one, and `null > 0` is
                        false in JS — so this rendered correctly by accident
                        while the type said it could not happen. */}
                    {m.downloads != null && m.downloads > 0 && (
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
                {/* Ollama has no search API, so this is a name box rather
                    than a result list — and it reaches strictly more models
                    than a list would: any tag, any namespace, anything
                    published since. It resolves against the registry before
                    offering the button, so the size is on screen first. */}
                <div className="ollama-find">
                  <input
                    className="share-note ollama-name"
                    placeholder="pull any model by name — qwen3:8b, user/model:tag"
                    value={ollamaName}
                    onChange={(e) => {
                      setOllamaName(e.target.value);
                      setResolved(null);
                    }}
                    onKeyDown={(e) => e.key === "Enter" && void doResolve()}
                  />
                  <button
                    className="ghost sm"
                    onClick={() => void doResolve()}
                    disabled={busy !== "" || !ollamaName.trim()}
                  >
                    {busy === "resolve" ? "looking…" : "Find"}
                  </button>
                </div>
                {resolved && !resolved.found && (
                  <div className="hint err pad">
                    {resolved.error ||
                      `The Ollama registry has no "${resolved.name}". Check the ` +
                        `name and tag at ollama.com/library.`}
                  </div>
                )}
                {resolved?.found && (
                  <>
                    <div className="model-row static">
                      <span className="mid">{resolved.name}</span>
                      <span className={`chip ${resolved.ok ? "" : "warn"}`}>
                        {resolved.bytes >= 1e12
                          ? `${(resolved.bytes / 1e12).toFixed(1)} TB`
                          : `${(resolved.bytes / 1e9).toFixed(1)} GB`}
                      </span>
                      <span className="spacer" />
                      <button
                        className="ghost sm"
                        onClick={() => void doPull(resolved.name, !resolved.ok)}
                        // A pull that cannot fit the disk will be refused by
                        // the server no matter what. Offering the button
                        // anyway is a promise the next click breaks.
                        disabled={
                          busy !== "" || (!resolved.ok && !resolved.overridable)
                        }
                        title={resolved.ok ? undefined : resolved.warning}
                      >
                        {busy === `pull:${resolved.name}`
                          ? "Pulling…"
                          : !resolved.ok && !resolved.overridable
                            ? "Won't fit"
                            : !resolved.ok
                              ? "Pull anyway"
                              : "Pull"}
                      </button>
                    </div>
                    {isPulling(busy, pulling, resolved.name) && (
                      <PullProgress p={pulling} />
                    )}
                    {resolved.warning && (
                      <div className="hint err pad">{resolved.warning}</div>
                    )}
                  </>
                )}

                <div className="meta pad">or one of these:</div>
                {pullWarning && (
                  <div className="oversize" role="alert">
                    <span className="oversize-mark" aria-hidden="true">
                      !
                    </span>
                    <div>
                      <p>{pullWarning.message}</p>
                      <div className="row">
                        <button
                          className="ghost sm"
                          onClick={() => void doPull(pullWarning.name, true)}
                        >
                          Pull it anyway
                        </button>
                        <button
                          className="ghost sm"
                          onClick={() => setPullWarning(null)}
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  </div>
                )}
                {/* Sizes resolved live against the registry, and marked
                    against this GPU — the same annotation the HuggingFace tab
                    puts on its curated picks. They used to be strings typed
                    into the source ("2.6 GB"), which is a number nobody
                    rechecks against tags that get republished. */}
                {suggestions
                  .filter((s) => !ollama.installed?.some((i) => i.name === s.name))
                  .map((s) => (
                    <div key={s.name} className="model-row static">
                      <span className="mid">{s.name}</span>
                      {s.size_gb > 0 ? (
                        <span className={`chip ${s.fits === false ? "warn" : ""}`}>
                          {s.size_gb} GB
                        </span>
                      ) : (
                        <span className="meta">size unknown offline</span>
                      )}
                      {s.fits === false && (
                        <span className="meta">bigger than this GPU</span>
                      )}
                      <span className="spacer" />
                      <button
                        className="ghost sm"
                        onClick={() => void doPull(s.name)}
                        disabled={busy !== ""}
                      >
                        {busy === `pull:${s.name}` ? "Pulling…" : "Pull"}
                      </button>
                      {isPulling(busy, pulling, s.name) && (
                        <PullProgress p={pulling} />
                      )}
                    </div>
                  ))}
              </>
            )}
          </div>
        )}

        {err && <div className="hint err">{err}</div>}
      </div>
    </div>,
    document.body,
  );
}

function fmt(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}k`;
  return String(n);
}
