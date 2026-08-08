"""Find models that are already sitting on this machine.

The HuggingFace cache is only one place models live. People also just keep
a folder of them, next to the project or on a second drive, and having to
retype a path for something already on disk is a small daily insult.

So this walks a root and recognises three shapes:

  models--org--name/   a HuggingFace cache entry
  some-folder/         config.json plus weights -- a plain from_pretrained dir
  something.gguf       llama.cpp / Ollama format, which transformers cannot
                       open; it is listed but marked unloadable rather than
                       hidden, because "why isn't my model here" is worse
                       than "here it is, and here is why it won't load".

Walking a synced drive is not free, so this prunes the directories that are
never models (node_modules, .git, venvs, caches), stops descending once a
directory *is* a model, bounds its depth, and gives up after a time budget.
If the budget runs out it says so in the result instead of quietly returning
a short list -- a truncated scan that looks complete is how you end up
believing a model isn't there.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Directories that never contain models and cost a fortune to walk.
SKIP = {
    "node_modules",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "site-packages",
    "dist",
    "build",
    ".next",
    ".cache",
    "$RECYCLE.BIN",
    "System Volume Information",
}

WEIGHTS = (".safetensors", ".bin", ".pth")
MAX_DEPTH = 6
BUDGET_S = 6.0


@dataclass
class Found:
    id: str  # what to put in the load box
    name: str  # short label for the UI
    path: str
    kind: str  # hf-cache | folder | gguf
    size_gb: float
    loadable: bool
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _size_of(paths: list[Path]) -> float:
    total = 0
    for p in paths:
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return round(total / 1e9, 2)


# The playground runs AutoModelForCausalLM. A repo whose architecture is not
# one of those cannot be loaded there, however valid a model it is — and the
# picker used to offer every cached repo as loadable, so choosing SAM or a
# diffusion model produced a minutes-long wait and then a HuggingFace
# tokenizer traceback. Say what a thing is instead of promising it.
_CAUSAL = ("ForCausalLM", "LMHeadModel")

# Repos ModelMRI does use, elsewhere. Naming the right panel beats "no".
_ELSEWHERE = {
    "sae": "a sparse autoencoder — load it from the features panel",
    "smolvla": "a robot policy — open it in the robot panel",
    "diffusion_pusht": "a robot policy — open it in the robot panel",
}

_KIND_BY_MODEL_TYPE = {
    "sam3_video": "an image/video segmentation model",
    "sam3": "an image segmentation model",
    "whisper": "a speech-recognition model",
    "clip": "an image-text embedding model",
    "vit": "an image model",
}


def _describe(config: dict | None, repo: str) -> tuple[bool, str]:
    """Can the playground load this, and if not, what is it?"""
    low = repo.lower()
    for token, note in _ELSEWHERE.items():
        if token in low:
            return False, note

    if config is None:
        return False, "not a transformers model (no config.json)"

    archs = config.get("architectures") or []
    if any(a.endswith(_CAUSAL) for a in archs):
        return True, "cached, loads offline"

    model_type = config.get("model_type") or ""
    if model_type in _KIND_BY_MODEL_TYPE:
        return False, f"{_KIND_BY_MODEL_TYPE[model_type]}, not a text model"
    if any(a.endswith("ForConditionalGeneration") for a in archs):
        return False, "a vision-language model — not a causal LM"
    if archs:
        return False, f"{archs[0]} is not a causal language model"
    return False, "no architecture in config.json — not a transformers model"


def _read_config(directory: Path) -> dict | None:
    """config.json from a model dir or the newest snapshot of a cache entry."""
    import json

    candidates = [directory / "config.json"]
    snapshots = directory / "snapshots"
    if snapshots.is_dir():
        try:
            for snap in sorted(
                snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
            ):
                candidates.append(snap / "config.json")
        except OSError:
            pass
    for path in candidates:
        try:
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return None


def _looks_like_model_dir(entries: list[os.DirEntry]) -> bool:
    """config.json plus at least one weight file is a from_pretrained dir."""
    names = {e.name for e in entries if e.is_file()}
    if "config.json" not in names:
        return False
    return any(
        n.endswith(WEIGHTS) or n.endswith(".safetensors.index.json") for n in names
    )


def _hf_cache_entry(root: Path, name: str) -> Found:
    """models--org--name -> the repo id transformers expects."""
    repo = "/".join(name.removeprefix("models--").split("--"))
    d = root / name
    # max, not sum: snapshots/ is a full copy of blobs/ wherever symlinks are
    # unavailable, so summing double-counts and the size sort goes wrong.
    size = max(
        _size_of([p for p in (d / "blobs").rglob("*") if p.is_file()]),
        _size_of([p for p in (d / "snapshots").rglob("*") if p.is_file()]),
    )
    loadable, note = _describe(_read_config(d), repo)
    return Found(
        id=repo,
        name=repo,
        path=str(d),
        kind="hf-cache",
        size_gb=size,
        loadable=loadable,
        note=note,
    )


def scan(root: str | Path, budget_s: float = BUDGET_S) -> tuple[list[Found], bool]:
    """Walk `root` for models. Returns (found, hit_budget)."""
    root = Path(root)
    out: list[Found] = []
    seen: set[str] = set()
    started = time.monotonic()
    truncated = False

    def walk(d: Path, depth: int) -> None:
        nonlocal truncated
        if truncated or depth > MAX_DEPTH:
            return
        if time.monotonic() - started > budget_s:
            truncated = True
            return
        try:
            entries = list(os.scandir(d))
        except OSError:
            return

        # A HuggingFace cache directory: take the repos, do not descend.
        cache_repos = [
            e for e in entries if e.is_dir() and e.name.startswith("models--")
        ]
        for e in cache_repos:
            if e.path not in seen:
                seen.add(e.path)
                out.append(_hf_cache_entry(d, e.name))

        if _looks_like_model_dir(entries):
            files = [Path(e.path) for e in entries if e.is_file()]
            if str(d) not in seen:
                seen.add(str(d))
                loadable, note = _describe(_read_config(d), d.name)
                out.append(
                    Found(
                        id=str(d),
                        name=d.name,
                        path=str(d),
                        kind="folder",
                        size_gb=_size_of(files),
                        loadable=loadable,
                        note="local folder" if loadable else note,
                    )
                )
            return  # a model dir has no models inside it

        for e in entries:
            if e.is_file() and e.name.endswith(".gguf"):
                if e.path in seen:
                    continue
                seen.add(e.path)
                out.append(
                    Found(
                        id=e.path,
                        name=e.name,
                        path=e.path,
                        kind="gguf",
                        size_gb=_size_of([Path(e.path)]),
                        loadable=False,
                        note="GGUF - run it through Ollama; transformers cannot open it",
                    )
                )
            elif e.is_dir(follow_symlinks=False) and e.name not in SKIP:
                if not e.name.startswith(".") and not e.name.startswith("models--"):
                    walk(Path(e.path), depth + 1)

    walk(root, 0)
    # Loadable first. Sorting by size alone scattered the seven models you
    # can actually run among eleven you cannot.
    out.sort(key=lambda f: (not f.loadable, -f.size_gb, f.name.lower()))
    return out, truncated


def roots() -> list[Path]:
    """Where to look. The working directory, plus anything explicitly set."""
    found: list[Path] = []
    if env := os.environ.get("MODELMRI_MODELS_DIR"):
        found += [Path(p) for p in env.split(os.pathsep) if p.strip()]
    found.append(Path.cwd())
    hf = os.environ.get("HF_HOME")
    if hf:
        found.append(Path(hf))
    out: list[Path] = []
    for p in found:
        try:
            rp = p.resolve()
        except OSError:
            continue
        # Drop a root that is inside one we are already scanning.
        if rp.is_dir() and not any(rp == o or o in rp.parents for o in out):
            out.append(rp)
    return out


def discover() -> dict:
    """Everything loadable on this machine, with the roots we looked in."""
    models: list[Found] = []
    seen: set[str] = set()
    truncated = False
    looked: list[str] = []
    for root in roots():
        looked.append(str(root))
        found, cut = scan(root)
        truncated = truncated or cut
        for f in found:
            if f.path not in seen:
                seen.add(f.path)
                models.append(f)
    # Same ordering as scan(): what you can load, first. This second sort
    # across roots silently undid the first one.
    models.sort(key=lambda f: (not f.loadable, -f.size_gb, f.name.lower()))
    return {
        "models": [m.to_dict() for m in models],
        "roots": looked,
        "truncated": truncated,
    }
