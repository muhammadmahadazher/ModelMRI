import { useEffect, useState } from "react";
import { AvailableCorpora, errorText, getAvailableCorpora } from "./api";
import { bytesSI } from "./measured";

/** Pick a corpus off this machine, or type a path — both still work.
 *
 *  WHY A LIST AND A BOX, RATHER THAN EITHER ALONE. Every panel that reads a
 *  corpus used to offer one text field, which is a fine control if you already
 *  know the path and a dead end if you do not: there is no way to find out
 *  from the page what this machine actually has. `GET /api/corpus/available`
 *  walks the corpus roots and names what it found, and `corpus_index` accepts
 *  either one of those ids or a typed path, so NOTHING IS TAKEN AWAY here. The
 *  box below is the same box; the list is new, and choosing from it fills the
 *  box with an id.
 *
 *  THE LIST IS NOT READ UNTIL IT IS ASKED FOR. The walk is real work — MEASURED
 *  at 2.2–3.8s and 3,707 directories on the machine this was written on — and
 *  two panels carry this control, so mounting it would have spent that twice
 *  on every page load for a reader who never opened either. It is a button,
 *  and the button says what it is about to do.
 *
 *  WHAT IT REFUSES TO LET A READER CONCLUDE:
 *
 *    a capped list is not the disk   `truncated_files` means the walk stopped
 *                                    counting, not that your file is gone
 *    a failed listing is not "none"  a request that never answered renders as
 *                                    a refusal with a next step, never as an
 *                                    empty dropdown
 *    an id is not a filename         a 32-character hash in the box says
 *                                    nothing, so the file it stands for is
 *                                    named underneath it
 */

/** The last listing, shared by every picker on the page.
 *
 *  Two panels each holding one would otherwise walk the same directories twice
 *  for the same answer within a second of each other. A rejection is NOT kept:
 *  caching a failure would turn one bad moment into a control that never works
 *  again until the tab is reloaded.
 */
let pending: Promise<AvailableCorpora> | null = null;

function listCorpora(force: boolean): Promise<AvailableCorpora> {
  if (force || pending === null) {
    const p = getAvailableCorpora();
    pending = p;
    p.catch(() => {
      if (pending === p) pending = null;
    });
  }
  return pending;
}

export default function CorpusPicker({
  id,
  value,
  onChange,
  disabled,
  placeholder = "notes.txt — or a path to one",
}: {
  /** Unique per instance: two of these share a page, and a duplicated `id`
   *  points both labels at whichever input rendered first. */
  id: string;
  value: string;
  onChange: (next: string) => void;
  disabled?: boolean;
  placeholder?: string;
}) {
  const [found, setFound] = useState<AvailableCorpora | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  // A listing another picker already took is reused on mount without asking
  // for a new walk — the cost this control defers is the SCAN, and there is
  // none to spend when the answer is already in hand.
  useEffect(() => {
    let live = true;
    if (pending === null) return;
    void pending
      .then((got) => {
        if (live) setFound(got);
      })
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, []);

  async function load(force: boolean) {
    if (busy) return;
    setBusy(true);
    setErr("");
    try {
      setFound(await listCorpora(force));
    } catch (e) {
      // The list is gone, the box is not. Saying only "failed" would leave a
      // reader with a dead control and no idea they can still proceed.
      setFound(null);
      setErr(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  const typed = value.trim();
  const chosen = found?.corpora.find((c) => c.id === typed) ?? null;

  return (
    <div className="corpus-picker">
      <div className="row cp-head">
        <label className="meta" htmlFor={`${id}-path`}>
          …or a corpus on this machine — a .txt or .jsonl, read from disk and
          never sent anywhere
        </label>
        <button
          className="ghost sm"
          onClick={() => void load(found !== null)}
          disabled={disabled || busy}
          title={
            found === null
              ? "Walks this machine's corpus roots and lists the .txt and .jsonl under them."
              : "Walks them again — a file created since this list was taken is not in it."
          }
        >
          {busy
            ? "reading directories…"
            : found === null
              ? "list what's here"
              : "re-read"}
        </button>
      </div>

      {found !== null && found.corpora.length > 0 && (
        <select
          className="cp-select"
          aria-label="corpora found on this machine"
          /* Empty whenever the box does not hold one of these ids, so a path
             typed by hand cannot leave a stale filename selected above it. */
          value={chosen ? chosen.id : ""}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled || busy}
        >
          <option value="">— pick one, or type a path below —</option>
          {/* Grouped by root, because `relative` alone repeats across roots
              and the payload splits the two for exactly this reason. */}
          {found.roots
            .filter((root) => found.corpora.some((c) => c.root === root))
            .map((root) => (
              <optgroup key={root} label={root}>
                {found.corpora
                  .filter((c) => c.root === root)
                  .map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.relative} · {bytesSI(c.bytes)}
                    </option>
                  ))}
              </optgroup>
            ))}
        </select>
      )}

      <input
        id={`${id}-path`}
        className="cp-path"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        spellCheck={false}
        disabled={disabled}
        /* A RELATIVE example. An absolute one would name the machine this was
           written on, which is the leak the shipped-file test guards. */
        placeholder={placeholder}
      />

      {/* An id is a hash. Without this line the box holds 32 characters of hex
          and the reader has no way to check they picked the file they meant. */}
      {chosen && (
        <p className="meta cp-chosen">
          that id is <b>{chosen.name}</b> — <code>{chosen.relative}</code> under{" "}
          <code>{chosen.root}</code>, {bytesSI(chosen.bytes)}
        </p>
      )}

      {found !== null && found.corpora.length === 0 && (
        <p className="meta warn">
          no {found.suffixes.join(" or ")} was found under{" "}
          {found.roots.length ? found.roots.join(", ") : "any corpus root"} —{" "}
          {found.n_dirs_read.toLocaleString()} director
          {found.n_dirs_read === 1 ? "y" : "ies"} were read,{" "}
          {found.max_depth} levels deep at most. Name your corpus's directory in{" "}
          <code>MODELMRI_CORPUS_DIRS</code> and restart, or type its path above.
        </p>
      )}

      {/* Every cap on screen, never only applied. `means` says both of these
          too, at the bottom — but it is a paragraph, and a reader hunting for
          a file that is not in the dropdown needs the answer at the dropdown. */}
      {found?.truncated_files && (
        <p className="meta warn">
          THE LIST IS CAPPED at {found.n_found.toLocaleString()} files. There
          are more on this disk than these — a file missing from the list above
          is not a file missing from your machine. Type its path in the box, or
          narrow the search with <code>MODELMRI_CORPUS_DIRS</code>.
        </p>
      )}
      {found?.truncated_dirs && (
        <p className="meta warn">
          THE WALK STOPPED after {found.n_dirs_read.toLocaleString()}{" "}
          directories, so part of the tree was never read at all. Same remedy:
          name the directory you mean in <code>MODELMRI_CORPUS_DIRS</code>, or
          type the path above.
        </p>
      )}

      {err && (
        <div className="hint err">
          {err} Nothing can be picked from a list that was not read — the box
          above still takes a path, and the sweep will say for itself if it
          cannot read one.
        </div>
      )}

      {/* The route's own sentence, last: it names the roots, the depth, the two
          suffixes and why an id and a path are both accepted. */}
      {found && <div className="hint cp-means">{found.means}</div>}
    </div>
  );
}
