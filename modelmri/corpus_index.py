"""The corpora on this machine — listed by the server, opened by descent.

WHY THIS EXISTS AT ALL. The routes that read a corpus used to take a path:
you typed `~/data/wiki.txt` and the server opened it. That is the obvious
design and it is a path-injection primitive — a local tool that will read any
file on the filesystem on request, reachable from a browser. Five successive
attempts to guard it with a normalise-and-check barrier all failed to satisfy
CodeQL (#418, #431, #432), and reading the reported flow explains why:
`Path.resolve()` READS THE FILESYSTEM, so it is itself the sink, and no check
placed around a value that reaches it counts as a barrier in that model.

So the taint is removed rather than fenced, and it is removed the same way
for both of the things a caller may send:

  an id      the server walks the roots, mints an id per file, and the client
             names one. `resolve()` is a dictionary lookup.
  a path     the server DESCENDS to it. The typed string is split into
             components and used only to compare against the names
             `os.scandir` returns; each step's path comes out of that scan.
             `resolve_typed()` never passes a caller's string to a path API.

Both end at a `Path` this process built from its own directory reads, so
nothing a request sends is ever used to construct the thing that gets opened.
A typed path still works — you can point this at your own corpus exactly as
before — but the file that opens is one the server independently found, and a
path outside the roots has nothing to find.

This is the same shape `/api/models/discovered` already has, which is the
argument that it is the right one here: a local tool that FINDS what you have
is nicer to use than one you type paths into, and it is the version that
cannot be pointed at `/etc/shadow`.

THE CLI IS NOT BEHIND THIS. `modelmri sweep --prompts <path>` still takes a
path, because that is the person at the keyboard naming their own file; they
can already read anything that process can, and refusing a path its own user
just typed protects nobody.

WHAT THE WALK COSTS, AND WHAT IT REFUSES TO SPEND. A recursive scan of a home
directory can be millions of files, so every bound here is explicit and every
one of them is REPORTED in the payload rather than silently applied — a
listing that quietly stopped early is a reader concluding a file is not there.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import paths
from .errors import BadRequest

#: Suffixes a corpus may have. The same two `sweep.load_prompts` reads, and it
#: is the reader that decides — offering a `.csv` here would list files that
#: then refuse.
SUFFIXES = (".txt", ".jsonl")

#: How deep below a root to look. Four is `~/projects/thing/data/corpus.txt`,
#: which is where corpora actually live; unbounded is a scan of a whole home
#: directory on every listing.
MAX_DEPTH = 4

#: Files listed. A machine with 50,000 text files is not helped by a dropdown
#: with 50,000 entries, and the cap is in the payload so a reader can see the
#: list is partial and narrow the roots.
MAX_FILES = 500

#: Directories entered. The stop that actually bounds the walk: `MAX_FILES`
#: alone would still descend forever through a tree with no `.txt` in it.
MAX_DIRS = 4_000

#: Never entered. Not a security boundary — everything here is inside a root
#: you already allowed — but a package tree holds tens of thousands of files
#: and none of them is your corpus.
SKIP = frozenset(
    {
        "node_modules",
        "site-packages",
        "__pycache__",
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist-info",
        "AppData",
        "Library",
        "Applications",
        "Windows",
        "Program Files",
        "Program Files (x86)",
        "$RECYCLE.BIN",
        "System Volume Information",
    }
)


@dataclass
class Corpus:
    """One readable corpus, named by an id the SERVER minted."""

    id: str
    name: str
    #: Which root it was found under, so two files with one name are
    #: distinguishable without printing an absolute path at the reader.
    root: str
    #: Where it sits below that root. Relative, because an absolute path is a
    #: fact about this machine and this list is rendered in a browser.
    relative: str
    bytes: int

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "root": self.root,
            "relative": self.relative,
            "bytes": self.bytes,
        }


@dataclass
class Listing:
    corpora: list[Corpus] = field(default_factory=list)
    roots: list[str] = field(default_factory=list)
    n_found: int = 0
    n_dirs_read: int = 0
    #: True when the walk stopped on a cap rather than because it was done.
    truncated_files: bool = False
    truncated_dirs: bool = False
    max_depth: int = MAX_DEPTH

    def to_dict(self) -> dict:
        return {
            "corpora": [c.to_dict() for c in self.corpora],
            "roots": self.roots,
            "n_found": self.n_found,
            "n_dirs_read": self.n_dirs_read,
            "truncated_files": self.truncated_files,
            "truncated_dirs": self.truncated_dirs,
            "max_depth": self.max_depth,
            "suffixes": list(SUFFIXES),
            "means": self.means(),
        }

    def means(self) -> str:
        where = ", ".join(self.roots) or "nowhere — no root resolved"
        parts = [
            f"{self.n_found} corpus file(s) found under {where}, "
            f"{MAX_DEPTH} directories deep at most, reading "
            f"{self.n_dirs_read:,} directories. Only {' and '.join(SUFFIXES)} "
            f"are listed, because those are the two `load_prompts` reads."
        ]
        if self.truncated_files:
            parts.append(
                f"THE LIST IS CAPPED at {MAX_FILES}. There are more; these are "
                f"the first {MAX_FILES} the walk reached, and a file missing "
                f"from this list is not a file that is missing from your disk. "
                f"Narrow it with MODELMRI_CORPUS_DIRS."
            )
        if self.truncated_dirs:
            parts.append(
                f"THE WALK STOPPED after {MAX_DIRS:,} directories, so part of "
                f"the tree was never read. Same remedy: name the directory you "
                f"mean in MODELMRI_CORPUS_DIRS."
            )
        parts.append(
            "You can name one of these ids or type a path; both work, and "
            "neither is used to BUILD a path. An id is a lookup in this list, "
            "and a typed path is walked down to one directory entry at a time "
            "— so the file that opens is one this server found, not a string a "
            "request supplied. A path outside the directories above has "
            "nothing here to find."
        )
        return " ".join(parts)


def _id_for(absolute: str) -> str:
    """A stable id for a path, minted from the path the SERVER walked to.

    sha256 rather than the path itself, so nothing a client holds is a
    filesystem location — an id that leaked into a log or a screenshot says
    nothing about the machine's directory layout.
    """
    return hashlib.sha256(absolute.encode("utf-8", "surrogateescape")).hexdigest()[:32]


def scan() -> Listing:
    """Walk the corpus roots. Every bound is reported, never only applied."""
    roots = paths.corpus_roots()
    out = Listing(roots=[str(r) for r in roots])
    seen: set[str] = set()
    dirs_read = 0

    for root in roots:
        # (directory, depth), breadth-first so a corpus sitting at the top of
        # a root is found before the walk spends its budget deep in a tree.
        queue: list[tuple[Path, int]] = [(root, 0)]
        while queue:
            if dirs_read >= MAX_DIRS:
                out.truncated_dirs = True
                break
            here, depth = queue.pop(0)
            try:
                entries = list(os.scandir(here))
            except (OSError, ValueError):
                # Unreadable, gone, or a name the filesystem rejects. Skipping
                # narrows what can be listed and never widens it.
                continue
            dirs_read += 1
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if depth + 1 <= MAX_DEPTH and entry.name not in SKIP:
                            if not entry.name.startswith("."):
                                queue.append((Path(entry.path), depth + 1))
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    if not entry.name.lower().endswith(SUFFIXES):
                        continue
                    absolute = os.path.normcase(os.path.abspath(entry.path))
                    if absolute in seen:
                        continue
                    if len(out.corpora) >= MAX_FILES:
                        out.truncated_files = True
                        break
                    seen.add(absolute)
                    out.corpora.append(
                        Corpus(
                            id=_id_for(absolute),
                            name=entry.name,
                            root=str(root),
                            relative=os.path.relpath(entry.path, root),
                            bytes=entry.stat().st_size,
                        )
                    )
                except (OSError, ValueError):
                    continue
            if out.truncated_files:
                break
        if out.truncated_files or out.truncated_dirs:
            # Keep going to the next root only if there is budget; otherwise
            # the flags above already say the listing is partial.
            if out.truncated_files:
                break

    out.corpora.sort(key=lambda c: (c.root, c.relative.lower()))
    out.n_found = len(out.corpora)
    out.n_dirs_read = dirs_read
    return out


#: The last walk, so choosing a corpus does not re-scan a home directory. Not
#: a security boundary — the walk is redone when an id is not in it, so a file
#: created since the last listing is still reachable.
_INDEX: dict[str, Path] = {}


def _rebuild() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for corpus in scan().corpora:
        index[corpus.id] = Path(corpus.root) / corpus.relative
    _INDEX.clear()
    _INDEX.update(index)
    return index


def resolve(corpus_id: object) -> Path:
    """The path the server walked to for this id, or a refusal.

    THE ID IS NEVER USED TO BUILD A PATH. It is a dictionary key, and the
    values in that dictionary were produced by `scan()` walking directories
    this process chose. A caller who sends `../../etc/shadow` gets "no corpus
    with that id" for the same reason they would get it for `banana`.
    """
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise BadRequest(
            "`corpus` is the id of a file from `GET /api/corpus/available`, "
            "and this was empty. That route lists what is readable on this "
            "machine; a path typed here is not accepted."
        )
    key = corpus_id.strip()

    found = _INDEX.get(key) or _rebuild().get(key)
    if found is None:
        raise BadRequest(
            f"no corpus on this machine has the id {key[:16]!r}. Re-read "
            f"`GET /api/corpus/available` — the file may have been moved or "
            f"renamed since that list was taken, or it may sit outside the "
            f"directories a corpus is read from. A PATH IS NOT ACCEPTED HERE: "
            f"the server lists what it can read and a request names one of "
            f"those, which is what stops this field being a way to read any "
            f"file on the machine."
        )
    return found


def _root_for(lexical: str) -> Path | None:
    """The corpus root `lexical` sits under, or `None`.

    The comparison is INLINE rather than behind a helper on purpose: a taint
    tracker models a barrier by reading the guarding condition, and it does
    not follow a boolean back through a function call. That is why five
    earlier versions of this guard were invisible to it.
    """
    for root in paths.corpus_roots():
        prefix = os.path.normcase(str(root))
        if lexical == prefix or lexical.startswith(prefix.rstrip(os.sep) + os.sep):
            return root
    return None


def resolve_typed(text: object) -> Path:
    """A path a person typed, resolved by DESCENT from a root.

    THE TYPED STRING NEVER REACHES A PATH API. It is normalised with pure
    string operations, checked against the roots, split into components, and
    from then on it is only ever the right-hand side of a name comparison.
    Every path this function touches — every `scandir` argument, and the
    `Path` it returns — came out of a directory read that started at a root
    this process chose.

    That is what makes a typed path safe to keep. The alternative designs
    were to fence the string with a barrier (five attempts, all invisible to
    the analyser because `Path.resolve()` is itself the sink) or to drop typed
    paths entirely, which would take away the way people actually point this
    at their own corpus.

    One `scandir` per component, so depth costs a directory read each and
    there is no depth cap here — the cap on `scan()` bounds a LISTING, which
    is a different job.
    """
    if not isinstance(text, (str, Path)) or not str(text).strip():
        raise BadRequest(
            "a corpus is a path to a .txt or .jsonl on this machine, or an id "
            "from `GET /api/corpus/available`. This was empty."
        )
    raw = str(text).strip()
    if chr(0) in raw:
        raise BadRequest(_unreadable(raw))

    try:
        # Pure string work: `expanduser`, `abspath` and `normpath` ask the
        # filesystem nothing, and `normcase` folds case so the comparison
        # below means the same thing on Windows as on POSIX.
        lexical = os.path.normcase(
            os.path.normpath(os.path.abspath(os.path.expanduser(raw)))
        )
    except (OSError, ValueError, RuntimeError):
        raise BadRequest(_unreadable(raw)) from None

    root = _root_for(lexical)
    if root is None:
        raise BadRequest(_outside(os.path.basename(lexical) or lexical))

    # What the caller asked for, below the root, as plain names.
    rest = lexical[len(os.path.normcase(str(root))) :].strip(os.sep)
    if os.altsep:
        rest = rest.replace(os.altsep, os.sep)
    wanted = [part for part in rest.split(os.sep) if part not in ("", ".")]
    if any(part == ".." for part in wanted):
        # `normpath` already collapsed these; one surviving here means the
        # path climbed above its own root, which `_root_for` would have
        # caught. Belt and braces, and it costs nothing.
        raise BadRequest(_outside(os.path.basename(lexical) or lexical))

    here = root
    for part in wanted:
        step = None
        try:
            with os.scandir(here) as entries:
                for entry in entries:
                    if os.path.normcase(entry.name) == part:
                        # `entry.path`, not `here / part`. At runtime the two
                        # are the same file — `part` only reaches here by
                        # matching a real entry — so no test can tell them
                        # apart, and mutating this line is correctly reported
                        # as an equivalent mutation. The difference is the
                        # DATA FLOW: one is built from a directory read, the
                        # other from the request's own string, and that is the
                        # entire property this module exists to hold.
                        step = Path(entry.path)
                        break
        except (OSError, ValueError):
            raise BadRequest(_unreadable(raw)) from None
        if step is None:
            raise BadRequest(
                f"there is no {part!r} in {here.name or str(here)!r}. The "
                f"directories a corpus is read from are "
                f"{', '.join(str(r) for r in paths.corpus_roots())}; check the "
                f"spelling, or add its directory to MODELMRI_CORPUS_DIRS."
            )
        here = step

    # A symlink is fine as long as it does not LEAD out. `here` came from
    # `scandir`, so resolving it is a read of a path this process built.
    final = Path(os.path.realpath(here))
    if _root_for(os.path.normcase(str(final))) is None:
        raise BadRequest(
            f"{here.name!r} is a link that leads outside the directories a "
            f"corpus is read from. The path itself is inside one of them; "
            f"what it points at is not."
        )
    return final


def resolve_any(value: object) -> Path:
    """An id from the listing, or a path someone typed. Both are supported.

    Ids are checked first because they are unambiguous — a 32-character hex
    string is not a filename anybody types — and a miss falls through to the
    path reading rather than refusing, so a caller never has to say which kind
    of thing they are sending.
    """
    if isinstance(value, str):
        key = value.strip()
        if len(key) == 32 and all(c in "0123456789abcdef" for c in key.lower()):
            found = _INDEX.get(key.lower()) or _rebuild().get(key.lower())
            if found is not None:
                return found
    return resolve_typed(value)


def _unreadable(raw: str) -> str:
    return (
        f"{raw!r} is not a path this machine can read. Check the drive, and "
        f"that no link in it points at itself."
    )


def _outside(name: str) -> str:
    roots = ", ".join(str(r) for r in paths.corpus_roots())
    return (
        f"{name!r} is outside the directories a corpus is read from over "
        f"HTTP: {roots}. Move the file under one of those, or name its "
        f"directory in MODELMRI_CORPUS_DIRS and restart. (`modelmri sweep "
        f"--prompts` has no such boundary — it is you naming your own file.)"
    )
