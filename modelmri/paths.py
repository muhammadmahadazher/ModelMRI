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


def override() -> Path | None:
    """One directory for everything, if the user asked for that."""
    return _env_path("MODELMRI_HOME")


def _platform_dir(kind: str) -> Path:
    """kind: data | config | cache"""
    home = Path.home()
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

LEGACY = Path.home() / ".modelmri"


def legacy_file(name: str) -> Path | None:
    """An existing file from before this module, if there is one.

    Versions up to 0.5.1 wrote `~/.modelmri/`. Moving the default without
    reading the old place would silently lose someone's traces and sign them
    out, so every caller checks here before creating anything new.
    """
    candidate = LEGACY / name
    try:
        return candidate if candidate.is_file() else None
    except OSError:
        return None


# -------------------------------------------------------- HuggingFace cache


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
    if home_env := _env_path("HF_HOME"):
        return home_env / "hub"
    if xdg := _env_path("XDG_CACHE_HOME"):
        return xdg / "huggingface" / "hub"
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def hf_home() -> Path:
    """The HuggingFace root — the parent of `hub/`, and where LeRobot sits."""
    if home_env := _env_path("HF_HOME"):
        return home_env
    if xdg := _env_path("XDG_CACHE_HOME"):
        return xdg / "huggingface"
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HOME)
    except Exception:
        return Path.home() / ".cache" / "huggingface"


def describe() -> dict:
    """Every location, for the UI and for `modelmri where`.

    A tool that writes to your disk should be able to tell you where, without
    you having to read its source.
    """
    return {
        "override": str(override()) if override() else None,
        "data": str(data_dir()),
        "config": str(config_dir()),
        "cache": str(cache_dir()),
        "hf_home": str(hf_home()),
        "hf_hub_cache": str(hf_hub_cache()),
        "cwd": str(Path.cwd()),
        "legacy": str(LEGACY) if LEGACY.is_dir() else None,
        "platform": sys.platform,
    }
