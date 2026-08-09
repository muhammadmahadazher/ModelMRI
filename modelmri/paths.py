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
from pathlib import Path

APP = "ModelMRI"  # display-cased; lowercased on Linux by convention


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    if not raw or not raw.strip():
        return None
    try:
        return Path(raw).expanduser()
    except (OSError, ValueError):
        return None


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
        try:
            resolved = Path(os.path.expandvars(part)).expanduser()
        except (OSError, ValueError):
            continue
        if resolved not in out:
            out.append(resolved)
    return out


def _platform_dir(kind: str) -> Path:
    """kind: data | config | cache"""
    home = _home() or Path(os.path.expanduser("~"))
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
    try:
        return Path(str(raw)).expanduser()
    except (OSError, ValueError):
        return None


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
    base = home / ".cache" if home else Path(".cache")
    return base / "huggingface" / "hub"


def hf_home() -> Path:
    """The HuggingFace root — the parent of `hub/`, and where LeRobot sits."""
    if home_env := _env_path("HF_HOME"):
        return home_env
    if xdg := _env_path("XDG_CACHE_HOME"):
        return xdg / "huggingface"
    if constant := _hub_constant("HF_HOME"):
        return constant
    home = _home()
    return (home / ".cache" if home else Path(".cache")) / "huggingface"


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
