# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Where things live, resolved rather than assumed.

Every location this program reads or writes was previously spelled out at its
point of use, which produced two problems.

**The HuggingFace cache was re-derived six times**, each as
`HF_HOME or ~/.cache/huggingface`. That expression is not how the library
resolves it. `huggingface_hub` honours `HF_HUB_CACHE` first, then `HF_HOME`,
then `XDG_CACHE_HOME`, and only then falls back to `~/.cache/huggingface`. So
anyone who set `HF_HUB_CACHE` — the variable the docs recommend — had their
models downloaded to one place and looked for in another. We ask the library
now. It owns the answer and cannot drift from itself.

**Application state went to `~/.modelmri` on every platform.** That is a Unix
convention wearing a Windows costume. Each OS has a place for this, and users
expect their home directory not to accumulate dotfiles from every tool they
try.

  Linux    $XDG_DATA_HOME/modelmri, $XDG_CONFIG_HOME/modelmri
  macOS    ~/Library/Application Support/ModelMRI
  Windows  %LOCALAPPDATA%\\ModelMRI, %APPDATA%\\ModelMRI

`MODELMRI_HOME` overrides all of it with one directory, for anyone who wants
everything in one place — a USB stick, a synced drive, a container.

Nothing here creates a directory as a side effect of being asked a question.
Call `ensure()` at the point of writing, so a read-only probe stays read-only.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

APP = "ModelMRI"  # display-cased; lowercased on Linux by convention


def _expand(raw: str) -> Path | None:
    """`Path(raw).expanduser()`, or None when it cannot be resolved.

    `expanduser` raises **RuntimeError** — not OSError — when a path contains
    `~` and there is no home directory to expand it against. POSIX rarely gets
    there because it falls back to the passwd database; Windows has nothing to
    fall back on, so `HF_HOME=~/models` with no USERPROFILE raises instead of
    returning something usable.

    This lived as four separate try/excepts, in `_env_path`, in the models-dir
    override, in `_hub_constant` and in `capacity.ollama_models_dir`. All four
    caught `(OSError, ValueError)`; none caught RuntimeError — while `_home()`
    directly below documents that exact failure and catches it. The
    windows-latest CI job failed on the `_hub_constant` one; the other three
    were the same bug waiting for a different environment variable to be set.
    One definition, so a fifth caller cannot get it wrong again.

    Every caller is resolving an optional override, so an unusable value means
    "not configured" rather than "stop".
    """
    try:
        return Path(raw).expanduser()
    except (OSError, ValueError, RuntimeError):
        return None


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    if not raw or not raw.strip():
        return None
    return _expand(raw)


def _home() -> Path | None:
    """`Path.home()`, or None where there is no answer.

    `Path.home()` does not degrade — it raises RuntimeError. That happens on
    a Linux container running as an arbitrary UID with no passwd entry (the
    OpenShift default), on distroless images, and on a Windows service
    account with no USERPROFILE. Every use of a home directory here is a
    fallback for when nothing was configured, so the right behaviour when
    there is no home is to keep going, not to bring down the import.
    """
    try:
        return Path.home()
    except (RuntimeError, OSError):
        return None


def corpus_roots() -> list[Path]:
    """Directories a TEXT CORPUS may be read from when the request came over
    HTTP.

    WIDER THAN `custom.allowed_roots`, ON PURPOSE. That boundary guards
    IMPORT — a file under it gets executed — so it is the working directory
    and the model directories and nothing else. This one guards READING a
    `.txt` or a `.jsonl`, which IS the feature: pointing the tuned lens or a
    neuron sweep at your own corpus is the thing people install this for.

    So it is the working directory, your home directory, and the system
    temporary directory, plus anything named in `MODELMRI_CORPUS_DIRS`
    (`os.pathsep`-separated, the same convention `MODELMRI_MODELS_DIR` uses).

    WHAT THIS IS AND IS NOT. It is a boundary against TRAVERSAL — `..` out to
    `/etc/shadow`, to the Windows system directory, or into another account's
    home — reached
    from an HTTP route. It is NOT a sandbox around your own files, and it
    cannot be one: reading the file you named is the whole feature, and a
    corpus lives in Documents far more often than in the directory you
    happened to launch from.

    The CLI does not go through this at all. `modelmri sweep --prompts` is the
    person at the keyboard naming their own file, and they can already read
    anything that process can — a boundary there would refuse a file its own
    user just typed while protecting nobody.

    Temp is in the list because it is where a downloaded corpus lands, where
    an editor writes a scratch file, and where every test fixture on a Linux
    runner is created. It is user-writable and holds nothing the caller could
    not already write themselves.
    """
    import tempfile

    roots: list[Path] = [Path.cwd()]
    home = _home()
    if home is not None:
        roots.append(home)
    try:
        roots.append(Path(tempfile.gettempdir()))
    except (OSError, ValueError):
        # No temp directory the OS will name. Narrows what may be read and
        # never widens it, which is the safe direction for a boundary.
        pass
    # The same parsing `models_dirs` does, and for the same reason its
    # docstring records: two modules once parsed that variable independently
    # and neither expanded `~`, so `~/corpora` became a literal directory
    # named `~` under the cwd and every file in it was refused.
    raw = os.environ.get("MODELMRI_CORPUS_DIRS") or ""
    for part in raw.split(os.pathsep):
        part = part.strip()
        if not part:
            continue
        extra = _expand(os.path.expandvars(part))
        if extra is not None:
            roots.append(extra)

    out: list[Path] = []
    for r in roots:
        try:
            resolved = r.resolve(strict=False)
        except OSError:
            continue
        if resolved not in out:
            out.append(resolved)
    return out


def override() -> Path | None:
    """One directory for everything, if the user asked for that."""
    return _env_path("MODELMRI_HOME")


def models_dirs() -> list[Path]:
    """Extra directories to look for models in, from MODELMRI_MODELS_DIR.

    Split on the platform separator, with `~` and `%VARS%`/`$VARS` expanded.
    Two modules used to parse this variable independently and neither
    expanded `~`, so `MODELMRI_MODELS_DIR=~/models` became the literal
    directory `<cwd>/~/models`: the scanner silently dropped it, and the
    adapter loader refused every file under it as outside the allowed roots.
    """
    raw = os.environ.get("MODELMRI_MODELS_DIR") or ""
    out: list[Path] = []
    for part in raw.split(os.pathsep):
        part = part.strip()
        if not part:
            continue
        resolved = _expand(os.path.expandvars(part))
        if resolved is None:
            continue
        if resolved not in out:
            out.append(resolved)
    return out


def _no_home() -> Path:
    """Somewhere absolute to stand when there is no home directory.

    `os.path.expanduser("~")` is NOT the answer, and it was: unlike
    `Path.home()` it does not raise when it cannot resolve, it returns the
    string `"~"` unchanged. Every path built on it was therefore RELATIVE,
    rooted at a directory literally named `~`. Measured with the home
    variables cleared: `data_dir()` returned `~/AppData/Local/ModelMRI` with
    `is_absolute()` False, so `modelmri serve` created a junk directory named
    `~` inside whatever directory it was started in, wrote the trace database
    there, and `modelmri where` answered the question "where is my stuff" with
    a relative path. On a read-only working directory it did not start at all.

    This is the trap `models_dirs()` documents two functions above, arriving
    by a different door. The temporary directory is not a good home, but it is
    an absolute one, and `describe()` says so rather than letting the fallback
    pass for a real answer.
    """
    return Path(tempfile.gettempdir()) / "modelmri-no-home"


def _platform_dir(kind: str) -> Path:
    """kind: data | config | cache"""
    home = _home() or _no_home()
    if sys.platform == "win32":
        # LOCALAPPDATA for data and cache (machine-local, not roamed);
        # APPDATA for config, which is small and worth roaming.
        local = _env_path("LOCALAPPDATA") or home / "AppData" / "Local"
        roaming = _env_path("APPDATA") or home / "AppData" / "Roaming"
        base = roaming if kind == "config" else local
        return base / APP / ("Cache" if kind == "cache" else "")
    if sys.platform == "darwin":
        if kind == "cache":
            return home / "Library" / "Caches" / APP
        return home / "Library" / "Application Support" / APP
    # Linux and the other unixes: XDG.
    name = APP.lower()
    if kind == "config":
        return (_env_path("XDG_CONFIG_HOME") or home / ".config") / name
    if kind == "cache":
        return (_env_path("XDG_CACHE_HOME") or home / ".cache") / name
    return (_env_path("XDG_DATA_HOME") or home / ".local" / "share") / name


def data_dir() -> Path:
    """Things the user would be upset to lose: the trace database."""
    root = override()
    return (root / "data") if root else _platform_dir("data")


def config_dir() -> Path:
    """Settings and credentials. Small, and worth backing up."""
    root = override()
    return (root / "config") if root else _platform_dir("config")


def cache_dir() -> Path:
    """Regenerable. Safe to delete."""
    root = override()
    return (root / "cache") if root else _platform_dir("cache")


def ensure(path: Path) -> Path:
    """Create a directory at the moment of writing, not of asking."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ------------------------------------------------------------ legacy location


def legacy_root() -> Path | None:
    """`~/.modelmri`, or None where there is no home to hang it off.

    Deliberately a function. As a module-level constant it was evaluated at
    import, which meant `import modelmri` died with a RuntimeError on any
    machine with no resolvable home — before MODELMRI_HOME, the documented
    fix for exactly that situation, could be read.

    MODELMRI_HOME switches it off entirely. That variable is documented as
    "all of it under one directory", and it was not: on a machine that had
    ever run 0.5.1 or earlier, a surviving `~/.modelmri/traces.sqlite` still
    won, so the same command produced different storage on two machines
    depending on their upgrade history. An explicit instruction beats a
    compatibility fallback.
    """
    if override() is not None:
        return None
    home = _home()
    return (home / ".modelmri") if home else None


def legacy_file(name: str) -> Path | None:
    """An existing file from before this module, if there is one.

    Versions up to 0.5.1 wrote `~/.modelmri/`. Moving the default without
    reading the old place would silently lose someone's traces and sign them
    out, so every caller checks here before creating anything new.
    """
    root = legacy_root()
    if root is None:
        return None
    candidate = root / name
    try:
        return candidate if candidate.is_file() else None
    except OSError:
        return None


def trace_db_path() -> Path:
    """The trace database the server actually opens."""
    return legacy_file("traces.sqlite") or (data_dir() / "traces.sqlite")


def token_path() -> Path:
    """The file the HuggingFace token actually lives in."""
    return legacy_file("hub.json") or (config_dir() / "hub.json")


def undelivered_traces() -> Path:
    """Where the recorder parks traces when the server is unreachable."""
    return _env_path("MODELMRI_TRACE_DIR") or (data_dir() / "undelivered")


# -------------------------------------------------------- HuggingFace cache


def _hub_constant(name: str) -> Path | None:
    """A path constant from huggingface_hub, if it is usable.

    The blankness check is not paranoia. `constants.HF_HUB_CACHE` is
    `os.getenv("HF_HUB_CACHE", <default>)`, so an empty-but-set variable —
    `ENV HF_HUB_CACHE=` in a Dockerfile, a shell that exported it before the
    value was computed — makes the library itself hand back `""`, and
    `Path("")` is the current working directory. ModelMRI would then scan
    the CWD for models and watch it for downloads that are landing elsewhere.
    """
    try:
        from huggingface_hub import constants

        raw = getattr(constants, name, "")
    except Exception:
        return None
    if not raw or not str(raw).strip():
        return None
    return _expand(str(raw))


def hf_hub_cache() -> Path:
    """The directory HuggingFace actually downloads into.

    Resolution order matches huggingface_hub: HF_HUB_CACHE, then HF_HOME/hub,
    then XDG_CACHE_HOME/huggingface/hub, then ~/.cache/huggingface/hub. The
    hand-rolled version this replaced knew only the last two, so a machine
    using HF_HUB_CACHE downloaded models to one directory and searched another.

    Read live, not once at import.
    """
    # Environment first, at CALL time. `huggingface_hub.constants.HF_HUB_CACHE`
    # is a module-level constant evaluated on import, so reading it alone
    # returns a snapshot from process start and ignores anything set since —
    # which made a test that sets HF_HOME look at the developer's real cache.
    if direct := _env_path("HF_HUB_CACHE"):
        return direct
    # The pre-rename variable, still honoured by the library itself:
    # `HF_HUB_CACHE = os.getenv("HF_HUB_CACHE", HUGGINGFACE_HUB_CACHE)`. Anyone
    # who set it years ago and never revisited it downloads into this path,
    # and we would have searched a different one.
    if legacy := _env_path("HUGGINGFACE_HUB_CACHE"):
        return legacy
    if home_env := _env_path("HF_HOME"):
        return home_env / "hub"
    if xdg := _env_path("XDG_CACHE_HOME"):
        return xdg / "huggingface" / "hub"
    if constant := _hub_constant("HF_HUB_CACHE"):
        return constant
    home = _home()
    base = home / ".cache" if home else _no_home() / ".cache"
    return base / "huggingface" / "hub"


# What HF_HOME said before ModelMRI pointed it somewhere else. Kept so
# discovery can still see models that are already downloaded: redirecting new
# downloads must not hide the old ones, or somebody with 50 GB cached opens
# the picker and finds it empty.
_INHERITED: list[Path] = []


def models_home() -> Path | None:
    """Where ModelMRI downloads models, when you have asked for somewhere.

    None means "wherever HuggingFace would put them anyway", which is the
    default and stays the default: a tool that silently relocates a cache
    other tools share would strand every model you already have.

    Set `MODELMRI_MODELS_HOME` to keep the weights somewhere you choose —
    beside the project, on a bigger disk, off a synced drive. This exists
    because an ambient `HF_HOME` had quietly put 50 GB of weights inside a
    Google Drive folder on the machine this was written on, and nothing in
    the tool said so. Saying so is `modelmri where`'s job; moving them is
    this variable's.
    """
    if explicit := _env_path("MODELMRI_MODELS_HOME"):
        return explicit
    if root := override():
        return root / "models"
    return None


def adopt_models_home() -> dict:
    """Point HuggingFace's downloader at the store, IF one was configured.

    Called before anything imports huggingface_hub, whose cache constants are
    module-level and computed on import, so this has to win that race. Every
    reader inside ModelMRI re-reads the environment at call time and follows.

    Does nothing at all unless `MODELMRI_MODELS_HOME` or `MODELMRI_HOME` is
    set. An `HF_HOME` somebody configured deliberately is theirs, and
    overriding it by default would be this tool deciding it knows better about
    a directory shared with transformers, datasets and every other library in
    the ecosystem.

    When it does act, the previous location is not discarded — it goes into
    the discovery roots, so models already downloaded stay visible and
    loadable rather than appearing to have vanished.
    """
    target = models_home()
    if target is None:
        return {"adopted": False, "reason": "no MODELMRI_MODELS_HOME set"}

    previous = os.environ.get("HF_HOME") or None
    previous_hub = os.environ.get("HF_HUB_CACHE") or None
    for raw in (previous, previous_hub):
        if not raw or not raw.strip():
            continue
        path = _expand(raw)
        if path is not None and path not in _INHERITED:
            _INHERITED.append(path)

    os.environ["HF_HOME"] = str(target)
    # HF_HUB_CACHE wins over HF_HOME inside huggingface_hub, so a stale one
    # would send the download straight back to where it came from.
    os.environ.pop("HF_HUB_CACHE", None)
    return {
        "adopted": True,
        "target": str(target),
        "previous": previous,
        "inherited": [str(p) for p in _INHERITED],
    }


def inherited_roots() -> list[Path]:
    """Caches that existed before ModelMRI took over the download location.

    Search-only. Nothing is written here.
    """
    return list(_INHERITED)


def ensure_models_home() -> Path | None:
    """Create the store and make it ignore itself. Called before downloading.

    Deliberately not part of `adopt_models_home`: that runs for every command
    including `modelmri inspect`, which reads one file and exits, and a
    read-only command that creates a directory is a side effect nobody asked
    for.
    """
    target = models_home()
    if target is None:
        return None
    try:
        _protect(target)
    except OSError:
        # Not a reason to refuse to run — it is a reason to have said so, and
        # `modelmri where` prints the location either way.
        pass
    return target


def _protect(store: Path) -> None:
    """Make the model store ignore itself.

    The default store lives in the working directory, and a working directory
    is very often a git repository. Multi-gigabyte weights staged by an
    absent-minded `git add -A` is a mistake that is genuinely hard to undo
    once pushed, so the store carries its own `.gitignore` saying `*`. That
    works regardless of whether the surrounding project has one, and needs
    nobody to remember anything.
    """
    store.mkdir(parents=True, exist_ok=True)
    marker = store.parent / ".gitignore"
    if not marker.exists():
        marker.write_text(
            "# ModelMRI's local model store. Weights do not belong in git.\n*\n",
            encoding="utf-8",
        )


def hf_home() -> Path:
    """The HuggingFace root — the parent of `hub/`, and where LeRobot sits."""
    if home_env := _env_path("HF_HOME"):
        return home_env
    if xdg := _env_path("XDG_CACHE_HOME"):
        return xdg / "huggingface"
    if constant := _hub_constant("HF_HOME"):
        return constant
    home = _home()
    return (home / ".cache" if home else _no_home() / ".cache") / "huggingface"


def describe() -> dict:
    """Every location, for the UI and for `modelmri where`.

    A tool that writes to your disk should be able to tell you where, without
    you having to read its source.
    """
    legacy = legacy_root()
    try:
        legacy_shown = str(legacy) if legacy is not None and legacy.is_dir() else None
    except OSError:
        legacy_shown = None
    return {
        "override": str(override()) if override() else None,
        "data": str(data_dir()),
        "config": str(config_dir()),
        "cache": str(cache_dir()),
        "models_home": str(models_home()) if models_home() else None,
        "inherited_caches": [str(p) for p in inherited_roots()],
        "hf_home": str(hf_home()),
        "hf_hub_cache": str(hf_hub_cache()),
        # The files, not just the directories that contain them. Reporting
        # `config_dir()` while the caller reads a surviving `~/.modelmri`
        # file made `modelmri where` name a path nothing was using.
        "trace_db": str(trace_db_path()),
        "hub_token": str(token_path()),
        "undelivered_traces": str(undelivered_traces()),
        "models_dirs": [str(p) for p in models_dirs()],
        "cwd": str(Path.cwd()),
        "legacy": legacy_shown,
        "platform": sys.platform,
    }


def validate_repo_id(raw: str, *, kind: str = "repository") -> str:
    """One repository id, checked once, or a Refusal naming what is wrong.

    THREE RESOLVERS EACH ROLLED THEIR OWN CHECK, and a dependency that already
    does this properly sat installed with zero call sites. Measured, before
    this existed:

      `vla_data.snapshot_path("pusht")`   ValueError: not enough values to
                                          unpack -> HTTP 500, CLI traceback
      `modelmri audit` with no argument   AttributeError: 'NoneType' has no
                                          attribute 'split'
      `vla._snapshot("../../etc/passwd")` 409 "is not cached. Download it first
                                          (huggingface-cli download
                                          ../../etc/passwd)" -- a command that
                                          cannot run, for a string that is not
                                          an id at all

    `huggingface_hub.utils.validate_repo_id` catches the traversal, the empty
    string, `a/b/c`, `a//b` and anything with a space. It does NOT catch
    `pusht`, because a bare name IS a valid Hub id for a canonical repo like
    `gpt2` -- so the slash is required separately here. The callers build
    `datasets--{owner}--{name}` and `models--{owner}--{name}` directory names,
    which structurally need a namespace; that requirement belongs to them
    rather than to the Hub's grammar, and stating it here keeps the two
    sentences apart.

    The wording is `vla._snapshot`'s, which already read well and was the only
    one of the three that said anything useful.
    """
    from .errors import Refusal

    text = (raw or "").strip()
    try:
        from huggingface_hub.utils import HFValidationError
        from huggingface_hub.utils import validate_repo_id as _hf
    except ImportError:  # pragma: no cover - the library is a hard dependency
        # If the validator is ever unavailable the slash rule below still
        # stands. Refusing to run because a checker could not be imported
        # would be a worse answer than the weaker check.
        HFValidationError = None
    if HFValidationError is not None:
        try:
            _hf(text)
        except HFValidationError:
            raise Refusal(
                f"`{text or '(nothing)'}` is not a {kind} id. A HuggingFace id "
                f"is `owner/name` — `lerobot/smolvla_base`, not "
                f"`smolvla_base` — and this one is not a name the Hub allows "
                f"at all."
            ) from None

    if "/" not in text:
        raise Refusal(
            f"`{text or '(nothing)'}` is not a {kind} id. A HuggingFace id is "
            f"`owner/name` — `lerobot/smolvla_base`, not `smolvla_base` — so "
            f"there is no owner here to look under."
        )
    return text
