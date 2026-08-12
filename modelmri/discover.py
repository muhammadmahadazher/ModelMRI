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
    # Agent worktrees are full COPIES of the repo, so scanning them listed the
    # same adapter_template.py three times with three different paths -- a
    # picker that appears to have found three models when it has found one.
    ".claude",
    ".worktrees",
}

WEIGHTS = (".safetensors", ".bin", ".pth")

# STANDALONE CHECKPOINTS, and what is honestly true of each one.
#
# Only `.gguf` used to be matched here, so a directory of somebody's own
# training output scanned to nothing at all -- the exact "why isn't my model
# here" this module's docstring says it exists to avoid. Finding a file is not
# the same as supporting it, so every entry carries the reason it will or will
# not open, and `loadable` is False wherever the loader would refuse.
#
# `.pt` and `.pth` are the same pickle container and say nothing about their
# contents: a TorchScript archive, a bare state_dict, or a whole pickled
# model. The loader tells them apart by reading the file; the scanner cannot,
# so the note covers both cases rather than guessing one.
LOOSE_WEIGHTS: dict[str, tuple[bool, str]] = {
    ".gguf": (
        False,
        "GGUF - inspectable here, but transformers cannot run it; use Ollama "
        "for that",
    ),
    # `loadable` is False for both, and that is not pessimism. A `.pt`/`.pth`
    # is one of three unrelated things and none of them can be inspected as a
    # bare file: a TorchScript archive loads but cannot be hooked (PyTorch
    # removes the hooks on RecursiveScriptModule), a state_dict is weights
    # with no model to put them in, and a pickled module executes code on
    # load. All three want the same answer -- an adapter -- so the note says
    # that rather than offering a button that will refuse.
    ".pt": (
        False,
        "PyTorch checkpoint - needs an adapter with a load() that builds the "
        "model; TorchScript archives run but cannot be instrumented",
    ),
    ".pth": (
        False,
        "PyTorch checkpoint - needs an adapter with a load() that builds the "
        "model; TorchScript archives run but cannot be instrumented",
    ),
    ".onnx": (
        False,
        "ONNX - not readable yet; ModelMRI hooks PyTorch modules and an "
        "ONNX graph has none. Export the PyTorch model instead",
    ),
    ".ckpt": (
        False,
        "Lightning checkpoint - weights only; point ModelMRI at an adapter "
        "that builds your model and loads this into it",
    ),
    ".h5": (
        False,
        "Keras weights - this tool instruments PyTorch modules",
    ),
    ".msgpack": (
        False,
        "Flax weights - this tool instruments PyTorch modules",
    ),
    ".pkl": (
        False,
        "a pickle - it can execute arbitrary code on load, so ModelMRI will "
        "not open it. Load it yourself in an adapter",
    ),
}
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
            # A file that vanished between the scandir and the stat
            # (FileNotFoundError), an unmaterialised Google Drive or OneDrive
            # placeholder (WinError 1920 — the module docstring says this walks
            # synced drives), a broken symlink, EACCES.
            #
            # What makes continuing safe is that `total` accumulates: one
            # unreadable file makes the answer an undercount, not a zero. A
            # zero is the outcome that would matter, because this number is
            # the picker's only warning about what a click is going to cost.
            continue
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


def _describe(
    config: dict | None, repo: str, unreadable: bool = False
) -> tuple[bool, str]:
    """Can the playground load this, and if not, what is it?"""
    low = repo.lower()
    for token, note in _ELSEWHERE.items():
        if token in low:
            return False, note

    if config is None:
        # "not a transformers model" is a claim about the repo, and we are
        # only entitled to it when we actually looked. A config we could not
        # READ — no permission, a cloud placeholder that never materialised, a
        # cache entry being rewritten underneath the scan — used to reach this
        # same sentence, so the picker told people their model was not a model
        # because their sync client had not finished.
        if unreadable:
            return False, (
                "could not read its config.json, so ModelMRI cannot say what "
                "this is — check permissions, or let a synced folder finish"
            )
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


def _read_config(directory: Path) -> tuple[dict | None, bool]:
    """(config, unreadable) — from a model dir or a cache entry's newest snapshot.

    Two ways to come back with no config, and they are not the same fact:
    this directory has none, or it has one we could not open. `_describe`
    turns the first into a statement about what the repo *is*, so the second
    has to be distinguishable or that statement gets made about a repo nobody
    ever managed to look inside.
    """
    import json

    unreadable = False
    candidates = [directory / "config.json"]
    snapshots = directory / "snapshots"
    if snapshots.is_dir():
        try:
            for snap in sorted(
                snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
            ):
                candidates.append(snap / "config.json")
        except OSError:
            # `iterdir` raises FileNotFoundError, PermissionError or
            # NotADirectoryError when a HuggingFace cache entry is written or
            # removed while we are walking it; the `stat` sort key raises
            # FileNotFoundError for a snapshot deleted mid-sort.
            #
            # Carrying on with only `directory/config.json` is right — there
            # may be one — but a cache entry usually keeps its config inside a
            # snapshot, so the likely outcome is finding nothing. That is
            # precisely why it has to be recorded as "could not read" instead
            # of being reported as "has none".
            unreadable = True
    for path in candidates:
        try:
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8")), unreadable
        except (OSError, ValueError):
            # The file is there and we still have nothing: PermissionError or
            # a placeholder that will not materialise (OSError), or bytes that
            # are not JSON (ValueError, via JSONDecodeError and
            # UnicodeDecodeError). Try the next snapshot, an older one may be
            # intact — but a config did exist here, so "no config.json" is now
            # the wrong thing to say about this directory.
            unreadable = True
            continue
    return None, unreadable


def _looks_like_model_dir(entries: list[os.DirEntry]) -> bool:
    """config.json plus at least one weight file is a from_pretrained dir."""
    names = {e.name for e in entries if e.is_file()}
    if "config.json" not in names:
        return False
    return any(
        n.endswith(WEIGHTS) or n.endswith(".safetensors.index.json") for n in names
    )


# The file extensions that mean "the weights are actually here".
_WEIGHTS = (".safetensors", ".bin", ".pth", ".pt", ".gguf", ".ckpt", ".msgpack", ".h5")


def has_weights(repo_dir: Path) -> bool:
    """Does this cache directory hold weights, or only metadata?

    Asking the Hub what a repo *is* downloads its `config.json`, and a
    refused or abandoned load leaves that behind. The result is a directory
    that looks exactly like a cached model and is not one: the picker listed
    `zai-org/GLM-5.2` at "0.00 GB" on a machine that had refused to download
    it, and clicking it would have started the whole 1.5 TB again.

    Cheap: stops at the first weight file rather than walking the tree.
    """
    for sub in ("blobs", "snapshots"):
        try:
            for path in (repo_dir / sub).rglob("*"):
                # Blobs are content-addressed, so the name carries no
                # extension — size is the only signal, and metadata files are
                # tiny. 1 MB is far below any real shard and far above any
                # config or tokenizer.
                if not path.is_file():
                    continue
                if path.suffix.lower() in _WEIGHTS:
                    return True
                if sub == "blobs" and path.stat().st_size > 1_000_000:
                    return True
        except OSError:
            # The subdirectory is missing or unwalkable, or a blob vanished
            # between the walk and the stat. Answering False for it is the
            # cautious direction and the one this function exists for: an
            # entry we cannot confirm holds weights is not offered as "already
            # on this machine", so the worst case is a model you have being
            # left out rather than a 1.5 TB download you thought you had.
            continue
    return False


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
    config, unreadable = _read_config(d)
    loadable, note = _describe(config, repo, unreadable)
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
            if e.path in seen:
                continue
            seen.add(e.path)
            # "On this machine" has to mean it. A metadata-only directory is
            # a download waiting to happen, not a model you already have.
            if not has_weights(Path(e.path)):
                continue
            out.append(_hf_cache_entry(d, e.name))

        if _looks_like_model_dir(entries):
            files = [Path(e.path) for e in entries if e.is_file()]
            if str(d) not in seen:
                seen.add(str(d))
                config, unreadable = _read_config(d)
                loadable, note = _describe(config, d.name, unreadable)
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
            suffix = Path(e.name).suffix.lower()
            if e.is_file() and suffix in LOOSE_WEIGHTS:
                if e.path in seen:
                    continue
                seen.add(e.path)
                loadable, note = LOOSE_WEIGHTS[suffix]
                out.append(
                    Found(
                        id=e.path,
                        name=e.name,
                        path=e.path,
                        # The extension IS the kind here. A `.pth` and a
                        # `.onnx` fail for different reasons and the panel
                        # should be able to say which.
                        kind=suffix.lstrip("."),
                        size_gb=_size_of([Path(e.path)]),
                        loadable=loadable,
                        note=note,
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
    from . import paths

    found: list[Path] = list(paths.models_dirs())
    found.append(Path.cwd())
    # Ask huggingface_hub where it actually caches, rather than reading one of
    # the four environment variables it honours and hoping that is the one set.
    # The cache directory itself, not its parent: scan() already recognises
    # `models--*` children of whatever it is handed, and `HF_HUB_CACHE=D:\hf`
    # made the parent `D:\` — a whole-drive walk that burned the six-second
    # budget before reaching the models it was sent to find.
    # ...plus anywhere models were downloaded BEFORE ModelMRI took over the
    # download location. Redirecting new downloads must not hide the old ones:
    # somebody with 50 GB already cached should not open the picker and find
    # it empty. Search-only — nothing is ever written to these.
    for candidate in (
        paths.hf_home(),
        paths.hf_hub_cache(),
        *paths.inherited_roots(),
    ):
        if candidate not in found:
            found.append(candidate)
    out: list[Path] = []
    for p in found:
        try:
            rp = p.resolve()
        except OSError:
            # A configured root that cannot be resolved — a disconnected
            # network drive, an unmounted volume, a symlink loop. Dropping it
            # is right: it is one of several roots and the rest still get
            # walked. It is also not hidden — `discover()` reports the roots
            # this function returned, under "roots", so a directory dropped
            # here is never counted as one that was searched.
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
