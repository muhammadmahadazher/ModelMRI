"""Bring your own model.

ModelMRI's other panels are transformer-shaped: attention heads, residual
streams, sparse autoencoders. None of that applies to the small networks
people actually train themselves — an MLP on tabular data, a CNN on CIFAR, a
two-layer regressor. This module serves those.

What it gives you is a layer-by-layer map of one real forward pass: what shape
comes out of every module, what the activations look like, how many units are
dead, whether anything has gone non-finite, and where the time goes. Those are
the questions you have at 2am when the loss is nan and you don't know which
layer did it.

Three ways in:

  adapter.py     a Python file exposing load() -> nn.Module. Most flexible;
                 works for any architecture, including one that needs your
                 own class definitions.
  model.pt       TorchScript, from torch.jit.save. Self-contained.
  weights.pth    a state_dict — REFUSED, with an explanation. A state_dict is
                 numbers without an architecture, and no amount of guessing
                 reconstructs the class that produced it.

Loading an adapter imports and runs Python, by design: that is how you point
at a network only your code knows how to build. Paths are confined to roots
you configured, and nothing is ever fetched from the network. See SECURITY.md.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .errors import BadRequest

log = logging.getLogger("modelmri")

# Saturation is distance from an activation's REAL bounds, so the bounds have
# to be written down. Measuring against the tensor's own max instead rescales
# the threshold to whatever the data happens to be, and inverts the answer:
# 9,000 sigmoid units pinned at 0 (gradient ~0, textbook saturated) reported
# 10% while the 1,000 healthy units at 0.5 were the ones counted.
#
# Softmax and LogSigmoid are deliberately absent. A softmax is saturated when
# it is peaked, not when its elements are near a bound — per-element
# saturation over a distribution is not a meaningful quantity, and reporting
# one made a maximum-entropy uniform softmax read as 100% saturated.
_BOUNDS: dict[str, tuple[float, float]] = {
    "Sigmoid": (0.0, 1.0),
    "Hardsigmoid": (0.0, 1.0),
    "Tanh": (-1.0, 1.0),
    "Hardtanh": (-1.0, 1.0),
}
_BOUNDED = frozenset(_BOUNDS)

# Modules whose output is the interesting thing. Everything else with no
# children is still hooked; this set only drives the "activation" flag used to
# decide whether a dead-unit count is worth showing.
_ACTIVATIONS = {
    "ReLU",
    "LeakyReLU",
    "ELU",
    "GELU",
    "SiLU",
    "Mish",
    "Softplus",
    "PReLU",
    "ReLU6",
    "Hardswish",
    *_BOUNDED,
}

MAX_LAYERS = 512  # a map nobody can read is not a map


@dataclass
class LayerStat:
    """One module's contribution to a forward pass."""

    order: int
    name: str
    kind: str
    out_shape: list[int]
    n_params: int
    trainable: bool
    ms: float
    # Activation statistics. None when the output held no float tensor.
    # WHICH CALL, when a module runs more than once in a single forward pass.
    # A shared encoder applied to two inputs -- a siamese or two-branch model
    # -- fires every one of its leaves twice, so the table showed two rows
    # named `enc.0` with nothing to tell them apart. Correct data, read as a
    # duplication bug, which is exactly the failure this project keeps
    # finding: a true row that looks false. 1 for a module that ran once.
    call: int = 1
    calls_total: int = 1
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    pct_zero: float | None = None
    pct_saturated: float | None = None  # bounded activations only
    n_nonfinite: int = 0
    is_activation: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CustomStatus:
    loaded: bool = False
    path: str | None = None
    source: str | None = None  # adapter | torchscript
    name: str | None = None
    device: str = "cpu"
    # `None`, not 0, and this object already got that right for `path`,
    # `source`, `name`, `input_shape` and `labels`. With `loaded: false`
    # nothing has been measured, so "this model has 0 parameters" is a
    # measurement nobody took — reported beside five fields correctly saying
    # they do not know.
    n_params: int | None = None
    n_trainable: int | None = None
    n_modules: int | None = None
    input_shape: list[int] | None = None
    input_origin: str = ""  # adapter | inferred | user
    input_reason: str = ""
    labels: list[str] | None = None
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class AdapterError(BadRequest):
    """Something about the file or its contents is wrong, and we say what.

    A `BadRequest`, and therefore still a ValueError, so every handler that
    caught it before catches it unchanged. The classification is the point:
    each of these is a fact about the path in the request or the file it names
    — not a Python module, no `load()`, a state_dict where a model was
    expected — which is 422 with the sentence, exactly what `/api/custom/load`
    and `/api/custom/run` answer today.

    What it is NOT is the exception raised *by* the adapter. That one is the
    user's own code failing, and server.py deliberately names its class rather
    than hiding it behind the generic 500 — see the note there.
    """


# ---------------------------------------------------------------- path safety


# Folders the user pointed at during this run. The scan used to be limited to
# the directory the server was launched in, which is the wrong question to ask
# somebody whose model lives on another drive: their answer is "it is over
# there", and the tool's answer was "restart me somewhere else".
#
# These are ADDED to the allowed roots rather than bypassing them. A local tool
# that will import any path handed to it is a nastier primitive than it looks —
# `_resolve` still refuses anything outside this list, so the boundary moves
# deliberately, once, when a person asks it to, and it does not survive a
# restart.
_SESSION_ROOTS: list[Path] = []


def add_root(raw: str) -> Path:
    """Let this run also look in `raw`. Returns the resolved directory."""
    if not raw or not raw.strip():
        raise AdapterError("no folder given")
    candidate = Path(raw.strip()).expanduser()
    try:
        resolved = candidate.resolve(strict=False)
    except OSError as err:
        raise AdapterError(
            f"that path cannot be resolved ({type(err).__name__})"
        ) from err
    if not resolved.exists():
        raise AdapterError(f"{resolved} does not exist")
    if not resolved.is_dir():
        # A file is what the NEXT step takes; this one widens where to look.
        raise AdapterError(
            f"{resolved} is a file, not a folder — give the folder it is in "
            f"and ModelMRI will find it"
        )
    # A filesystem or drive root is not a folder somebody means. Adding one
    # would make every .py on the machine loadable, and the adapter loader
    # IMPORTS what it loads, so "widen the search" would quietly become "run
    # anything on this disk". Naming a real directory is the whole point.
    if resolved.parent == resolved:
        raise AdapterError(
            f"{resolved} is the root of a filesystem, not a folder of models. "
            f"Name the directory your model is actually in."
        )
    if resolved not in _SESSION_ROOTS:
        _SESSION_ROOTS.append(resolved)
    return resolved


def clear_roots() -> None:
    """Forget the folders added during this run. Used by the tests."""
    _SESSION_ROOTS.clear()


def allowed_roots() -> list[Path]:
    """Directories an adapter may be loaded from.

    The working directory you launched in, plus MODELMRI_MODELS_DIR. Anything
    else is refused — a local tool that will import any path on the filesystem
    on request is a nastier primitive than it looks.
    """
    from . import paths

    roots = [Path.cwd(), *paths.models_dirs(), *_SESSION_ROOTS]
    out: list[Path] = []
    for r in roots:
        try:
            out.append(r.resolve(strict=False))
        except OSError:
            # `strict=False` already tolerates a path that does not exist, so
            # what is left is a path that cannot be resolved at all: a symlink
            # loop (ELOOP), a disconnected network drive, a name the
            # filesystem rejects. Dropping it is the safe direction for a
            # *security* boundary — a root that is not in this list is one
            # that `resolve_under_roots` will refuse to load from, so a
            # failure here narrows what may be imported and never widens it.
            continue
    return out


def resolve_under_roots(path: str | Path) -> Path:
    """Resolve `path` to an existing FILE, or raise if it escapes the roots."""
    return _under_roots(path, want="file")


def resolve_dir_under_roots(path: str | Path) -> Path:
    """The same boundary, for a directory.

    A full-precision checkpoint is a directory of safetensors, so the
    quantisation comparison needs to name one. Splitting the check by kind
    rather than relaxing `resolve_under_roots` to accept both: a caller that
    wants a file and is handed a directory has a bug, and the two messages say
    different things about what to do next.
    """
    return _under_roots(path, want="dir")


def _under_roots(path: str | Path, *, want: str) -> Path:
    try:
        p = Path(path).expanduser().resolve(strict=False)
    except OSError as err:
        raise AdapterError(f"cannot resolve {path!r} ({type(err).__name__})") from err

    roots = allowed_roots()
    for root in roots:
        try:
            p.relative_to(root)
        except ValueError:
            # Not an error being swallowed — this IS the test. `relative_to`
            # raises ValueError precisely when `p` is not under `root`, which
            # is the question being asked, so a raise here means "try the next
            # root" and falling out of the loop means "under none of them",
            # which the `else` below turns into a refusal.
            continue
        break
    else:
        listed = ", ".join(str(r) for r in roots)
        raise AdapterError(
            f"{p} is outside the directories ModelMRI may load from ({listed}). "
            "Start the server where your model lives, or set MODELMRI_MODELS_DIR."
        )

    if not p.exists():
        raise AdapterError(f"{p} does not exist")
    if want == "file" and not p.is_file():
        raise AdapterError(f"{p} is not a file")
    if want == "dir" and not p.is_dir():
        raise AdapterError(f"{p} is not a directory")
    return p


# ------------------------------------------------------------------- loading


def _import_adapter(path: Path):
    """Import a .py file as a throwaway module. This runs its top level."""
    mod_name = f"modelmri_adapter_{uuid.uuid4().hex[:12]}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise AdapterError(f"{path.name} could not be imported as a Python module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    # The adapter's own directory on sys.path, so `from my_net import Net`
    # works the way it does when you run your training script.
    parent = str(path.parent)
    added = parent not in sys.path
    if added:
        sys.path.insert(0, parent)
    try:
        spec.loader.exec_module(module)
    except Exception as err:
        sys.modules.pop(mod_name, None)
        raise AdapterError(
            f"{path.name} raised while being imported: "
            f"{type(err).__name__}: {err}"  # leak-ok: the reader's own adapter code
        ) from err
    finally:
        if added:
            try:
                sys.path.remove(parent)
            except ValueError:
                # Already the narrowest type there is: `list.remove` raises
                # ValueError and nothing else, and only when the value is
                # absent. It can be absent because `exec_module` two lines up
                # ran the adapter's top level — the user's code, which is free
                # to rebind or clean `sys.path` on its way past.
                #
                # Continuing is right in the strongest sense: the entry we
                # added is gone, which is the state this `finally` exists to
                # reach. What it does NOT cover is the mirror case — an
                # adapter that appended the same directory itself, because
                # `.remove` deletes only the first equal entry and leaves the
                # duplicate behind. That is a leak in `sys.path`, not a crash,
                # and it belongs to the file that put it there.
                pass
    return module


def load_from_adapter(path: Path):
    """Return (model, example_input, labels, module) from an adapter file.

    The MODULE comes back too, because the causal sweep needs the adapter's
    declared `TASK` and its `sample_inputs()`, and re-importing the file to
    read them would run the reader's top-level code a second time — and could
    hand back a different model from the one being measured.
    """
    import torch

    module = _import_adapter(path)
    loader = getattr(module, "load", None)
    if loader is None:
        raise AdapterError(
            f"{path.name} has no load() function. An adapter is a Python file "
            "with `def load(): return your_model` — see the template in "
            "examples/adapter_template.py."
        )
    if not callable(loader):
        raise AdapterError(f"{path.name}: load is not callable")

    try:
        model = loader()
    except Exception as err:
        raise AdapterError(
            f"{path.name}: load() raised {type(err).__name__}: {err}"  # leak-ok: the reader's own adapter code
        ) from err

    if not isinstance(model, torch.nn.Module):
        raise AdapterError(
            f"{path.name}: load() returned {type(model).__name__}, not a "
            "torch.nn.Module. Return the model itself, not a state_dict, a "
            "path, or a (model, optimizer) tuple."
        )

    example = None
    maker = getattr(module, "example_input", None)
    if callable(maker):
        try:
            example = maker()
        except Exception as err:
            raise AdapterError(
                f"{path.name}: example_input() raised {type(err).__name__}: {err}"  # leak-ok: the reader's own adapter code
            ) from err

    labels = getattr(module, "LABELS", None)
    if labels is not None:
        try:
            labels = [str(x) for x in labels]
        except TypeError:
            # LABELS is whatever the adapter author assigned. TypeError is the
            # one thing this can raise — a non-iterable — and None is the
            # value the caller already means by "this model has no label
            # names", so the axis is simply unlabelled rather than the load
            # failing over a cosmetic attribute.
            labels = None

    return model, example, labels, module


# WHOSE EXCEPTION IT IS, WHICH IS THE ONLY QUESTION THAT MATTERS HERE.
#
# Four messages in this file relay a caught exception's text, and they are
# marked `leak-ok`. Every one of them is the READER'S OWN code failing: their
# adapter module failing to import, their `load()` raising, their
# `example_input()` raising, their model's forward pass raising. That text was
# written by them or by the library they chose to call, it is on their machine,
# and it is the entire content of "why did my adapter not work" — suppressing
# it would leave them with a class name and no way forward.
#
# The messages that do NOT relay text are the ones where ModelMRI made the
# call: `torch.load` on a checkpoint the reader only pointed at, and path
# resolution. There the exception is a library talking about this machine to
# somebody who was not asking about this machine, and it carried an absolute
# weights path plus a site-packages frame into the browser. Measured.
def nearby_model_classes(weights: Path) -> list[tuple[str, str]]:
    """`nn.Module` subclasses defined in .py files beside a checkpoint.

    Returns (file name, class name) pairs. Read with `ast`, never imported:
    importing an arbitrary module to find out what it contains means executing
    it, and "this file is not something I will execute" is the rule the whole
    adapter contract rests on. Reading the text costs nothing and cannot bite.

    The point is the sentence it lets the refusal say. A state_dict genuinely
    cannot be loaded on its own -- but "write an adapter" is a poor answer when
    the class is one file away, and this is how the tool notices.
    """
    import ast

    out: list[tuple[str, str]] = []
    try:
        siblings = sorted(weights.parent.glob("*.py"))
    except OSError:
        return out

    for f in siblings[:40]:  # a folder, not a source tree
        if f.name.startswith("_") or f.name.endswith("_adapter.py"):
            continue
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError, ValueError):
            # A file that will not parse is not a candidate, and is also not
            # this function's problem to report.
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                # nn.Module, torch.nn.Module, Module -- match the attribute
                # name so an aliased import still counts.
                name = (
                    base.attr
                    if isinstance(base, ast.Attribute)
                    else base.id
                    if isinstance(base, ast.Name)
                    else ""
                )
                if name == "Module":
                    out.append((f.name, node.name))
                    break
    return out


def load_torchscript(path: Path):
    """Load a TorchScript archive, or explain why a plain checkpoint can't be.

    Scanned BEFORE `torch.jit.load` touches it. A TorchScript archive is a zip
    that can carry pickles, and `jit.load` unpickles them — so the window
    between "the user chose this file" and "arbitrary code has run" is this
    call. The scan closes it, and a dangerous file is a refusal with the
    finding named rather than a warning printed after the fact.
    """
    import torch

    from . import weights_scan

    weights_scan.guard(path)

    try:
        return torch.jit.load(str(path), map_location="cpu")
    except Exception as err:  # a probe, not a load; see below
        # Deliberately broad: the question this try asks is "is this file
        # TorchScript", and every way of answering no is a no. torch raises
        # RuntimeError for most of them (a zip with no `constants.pkl`, a
        # plain pickle, a text file) but the set is not enumerable and not
        # stable across torch versions, and a type this missed would replace
        # a diagnosis with a traceback.
        #
        # Logged, though, because swallowing it is only free when the answer
        # really is "not TorchScript". A file that IS TorchScript and failed
        # for some other reason — a truncated archive, a version mismatch —
        # falls through to the checkpoint reader below and gets described by
        # *that* error instead, which is the wrong one. This line is where
        # the right one survives.
        log.debug(
            "%s did not load as TorchScript (%s: %s)",
            path.name,
            type(err).__name__,
            err,
        )

    # Not TorchScript. Say precisely what it is instead of "load failed".
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as err:
        # The class, not the text. An AdapterError is relayed to the browser
        # in its own words, and torch's text here is not the project's:
        # measured, a PytorchStreamReader failure pasted both the caller's
        # absolute weights path and a site-packages serialization.py frame
        # into the body. The class still says which KIND of unreadable it is,
        # which is what the sentence needs; the rest is in the terminal.
        log.warning("reading %s as a checkpoint failed", path.name, exc_info=err)
        raise AdapterError(
            f"{path.name} is neither TorchScript nor a readable checkpoint "
            f"({type(err).__name__} — the full error is in the terminal). If "
            "it needs pickle to load, it can execute arbitrary code — load it "
            "yourself in an adapter instead."
        ) from err

    kind = type(obj).__name__
    if isinstance(obj, dict):
        keys = list(obj)[:4]
        # BEFORE refusing, look for the half that is missing. The class
        # is very often in the same folder as the weights -- that is how
        # people lay a project out -- and telling somebody to go and write an
        # adapter while their model class sits next to the checkpoint is the
        # tool failing to look.
        nearby = nearby_model_classes(path)
        found = ""
        if nearby:
            listed = ", ".join(f"{cls} in {fn}" for fn, cls in nearby[:3])
            first_file, first_cls = nearby[0]
            found = (
                f"\n\nThere is a model class next to it: {listed}. "
                f"An adapter that uses it is six lines:\n\n"
                f"    import torch\n"
                f"    from {first_file[:-3]} import {first_cls}\n\n"
                f"    def load():\n"
                f"        m = {first_cls}()\n"
                f'        m.load_state_dict(torch.load("{path.name}", '
                f'map_location="cpu", weights_only=True))\n'
                f"        return m.eval()\n\n"
                f"Save that beside them and point ModelMRI at it. If the "
                f"forward pass takes anything other than one tensor, add an "
                f"example_input() returning what it does take."
            )
        raise AdapterError(
            f"{path.name} is a state_dict ({len(obj)} tensors, e.g. "
            f"{', '.join(map(str, keys))}), which is weights without an "
            "architecture — nothing can reconstruct your model class from it. "
            "Write a six-line adapter that builds the model and loads these "
            "weights into it; see examples/adapter_template.py." + found
        )
    raise AdapterError(
        f"{path.name} contains a {kind}, not a model. Point ModelMRI at a "
        "TorchScript file (torch.jit.save) or an adapter .py."
    )


# ------------------------------------------------------------- introspection


def count_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _model_device(model) -> str:
    """Where this model's parameters actually are."""
    try:
        return str(next(model.parameters()).device)
    except (StopIteration, AttributeError):
        return "cpu"  # a model with no parameters is on no device in particular


def leaf_modules(model) -> list[tuple[str, object]]:
    """Modules with no children, in declaration order, named as you named them."""
    out = []
    for name, mod in model.named_modules():
        if name and not list(mod.children()):
            out.append((name, mod))
    return out


def suggest_input(model) -> tuple[list[int] | None, str]:
    """A starting shape for the forward pass, and why.

    This is a suggestion the user confirms, never a silent guess. Getting it
    wrong produces a confusing traceback rather than a wrong number, but the
    project's rule is that nothing runs on an assumption the user hasn't seen.
    """
    import torch

    for _, mod in leaf_modules(model):
        if isinstance(mod, torch.nn.Linear):
            return [1, mod.in_features], (
                f"first Linear takes {mod.in_features} features"
            )
        if isinstance(mod, (torch.nn.Conv1d,)):
            return [1, mod.in_channels, 64], (
                f"first Conv1d takes {mod.in_channels} channels; length 64 is a guess"
            )
        if isinstance(mod, (torch.nn.Conv2d,)):
            return [1, mod.in_channels, 32, 32], (
                f"first Conv2d takes {mod.in_channels} channels; 32x32 is a guess"
            )
        if isinstance(mod, torch.nn.Conv3d):
            return [1, mod.in_channels, 8, 32, 32], (
                f"first Conv3d takes {mod.in_channels} channels; 8x32x32 is a guess"
            )
        if isinstance(mod, torch.nn.Embedding):
            return [1, 16], (
                f"first Embedding has {mod.num_embeddings} entries; "
                "input is token ids, so give it a length"
            )
    return None, (
        "no Linear, Conv or Embedding found to infer a shape from — give the "
        "input shape yourself, or add example_input() to your adapter"
    )


def _first_tensor(value):
    """Find the tensor in whatever a module decided to return."""
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def tensor_stats(t, kind: str) -> dict:
    """Activation statistics for one output tensor.

    Non-finite values are counted and then excluded, so one nan doesn't turn
    every other number on the row into nan and hide where the problem started.
    """
    import torch

    out: dict = {"out_shape": list(t.shape)}
    if not t.is_floating_point():
        out["note"] = (
            f"{str(t.dtype).replace('torch.', '')} output — no activation stats"
        )
        return out

    f = t.detach().float().reshape(-1)
    finite = torch.isfinite(f)
    n_bad = int((~finite).sum().item())
    out["n_nonfinite"] = n_bad
    good = f[finite]
    if good.numel() == 0:
        out["note"] = "every value is nan or inf"
        return out

    out["mean"] = round(float(good.mean()), 6)
    out["std"] = round(float(good.std(unbiased=False)), 6)
    out["min"] = round(float(good.min()), 6)
    out["max"] = round(float(good.max()), 6)
    out["pct_zero"] = round(float((good == 0).float().mean()) * 100, 2)
    bounds = _BOUNDS.get(kind)
    if bounds is not None:
        # Within 1% of the activation's own range of EITHER bound. Both ends
        # matter: a sigmoid pinned at 0 has gradient s(1-s) ~ 0 just as surely
        # as one pinned at 1, and the magnitude-only test could not see it.
        lo, hi = bounds
        margin = 0.01 * (hi - lo)
        near = ((good <= lo + margin) | (good >= hi - margin)).float().mean()
        out["pct_saturated"] = round(float(near) * 100, 2)
    return out


def inspect(model, example) -> tuple[list[LayerStat], dict]:
    """Run one forward pass with hooks on every leaf, return the layer map."""
    import torch

    leaves = leaf_modules(model)
    if not leaves:
        raise AdapterError(
            "this model has no leaf modules — there is nothing to hook. A bare "
            "function is not an nn.Module; wrap it in one."
        )
    truncated = len(leaves) > MAX_LAYERS
    leaves = leaves[:MAX_LAYERS]

    rows: list[LayerStat] = []
    starts: dict[int, float] = {}
    order = [0]

    def pre_hook(mod, _inp):
        starts[id(mod)] = time.perf_counter()

    # How many times each module has fired during THIS pass.
    call_no: dict[str, int] = {}

    def make_hook(name: str, mod):
        def hook(_m, _inp, output):
            kind = type(mod).__name__
            call_no[name] = call_no.get(name, 0) + 1
            began = starts.pop(id(mod), None)
            ms = round((time.perf_counter() - began) * 1000, 3) if began else 0.0
            n_params = sum(p.numel() for p in mod.parameters(recurse=False))
            trainable = any(p.requires_grad for p in mod.parameters(recurse=False))
            t = _first_tensor(output)
            base = {
                "order": order[0],
                "name": name,
                "kind": kind,
                "n_params": n_params,
                "trainable": trainable,
                "ms": ms,
                "is_activation": kind in _ACTIVATIONS,
                "out_shape": [],
                "call": call_no[name],
            }
            order[0] += 1
            if t is None:
                base["note"] = f"returned {type(output).__name__} with no tensor"
                rows.append(LayerStat(**base))
                return
            base.update(tensor_stats(t, kind))
            rows.append(LayerStat(**base))

        return hook

    handles = []
    for name, mod in leaves:
        try:
            handles.append(mod.register_forward_pre_hook(pre_hook))
            handles.append(mod.register_forward_hook(make_hook(name, mod)))
        except RuntimeError as err:
            # A REFUSAL, NOT A 500. torch installs a generated `fail()` over
            # both hook APIs on RecursiveScriptModule, which is what
            # `torch.jit.load` returns -- so every TorchScript archive on disk
            # is un-hookable, all of them, not a corner case. Only an
            # in-memory `torch.jit.trace` result keeps its hooks and `load`
            # cannot produce one.
            #
            # This raised from the registration loop, ABOVE the try/finally
            # below, as a bare RuntimeError. server.py catches AdapterError
            # and nothing else, so it fell to the 500 arm and the reader was
            # told something inside ModelMRI had failed -- when the honest
            # answer is that this file format cannot carry what the panel
            # measures, and there is something they can do about it.
            for h in handles:
                h.remove()
            raise AdapterError(
                "This is a TorchScript archive, and TorchScript cannot be "
                "instrumented — PyTorch removes the forward hooks this panel "
                "reads activations through, so there is nothing here to "
                "measure. It loads and runs; only the layer-by-layer "
                "statistics are unavailable.\n\n"
                "To inspect it, point ModelMRI at the original nn.Module "
                "instead: an adapter with a `load()` that builds your model "
                "and loads the weights. `modelmri where` prints the folders "
                "that are scanned."
            ) from err

    was_training = model.training
    model.eval()
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            output = model(*example) if isinstance(example, tuple) else model(example)
    except Exception as err:
        raise AdapterError(
            f"the forward pass raised {type(err).__name__}: {err}. "  # leak-ok: the reader's own model
            "Most often the input shape is wrong — check the shape field, or "
            "add example_input() to your adapter so ModelMRI never has to guess."
        ) from err
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    # The TOTAL is only knowable once the pass is over, so it is stamped on
    # afterwards. A row saying "call 1" tells you nothing on its own; "call 1
    # of 2" tells you the module is shared and that a second reading of it
    # exists further down.
    for r in rows:
        r.calls_total = call_no.get(r.name, 1)

    total_ms = round((time.perf_counter() - t0) * 1000, 3)
    out_t = _first_tensor(output)
    meta = {
        "total_ms": total_ms,
        "n_layers": len(rows),
        # Modules that ran more than once. A two-branch model applies the same
        # encoder to two inputs, and without this the table simply looks like
        # it has repeated itself.
        "repeated": sorted({r.name for r in rows if r.calls_total > 1}),
        "truncated": truncated,
        "output_shape": list(out_t.shape) if out_t is not None else [],
        "output": _summarise_output(out_t),
    }
    return rows, meta


def _summarise_output(t) -> dict:
    """The model's answer, in the terms a small-model author cares about."""
    import torch

    if t is None:
        return {}
    flat = (
        t.detach().float().reshape(t.shape[0], -1)
        if t.dim() > 1
        else t.detach().float().reshape(1, -1)
    )
    row = flat[0]
    finite = torch.isfinite(row)
    n_nonfinite = int((~finite).sum())
    if row.numel() == 0 or not finite.any():
        return {
            "nonfinite": True,
            "n_nonfinite": n_nonfinite,
            "n_out": int(row.numel()),
        }

    # RANKED OVER THE FINITE VALUES. The guard above only fires when EVERY
    # value is unusable, so a partly-NaN output fell straight through -- and
    # torch ranks NaN as the largest thing there is. MEASURED on
    # [0.1, nan, 0.9, 0.3, inf]: argmax returns 1, and topk returns
    # [nan, inf, 0.9]. So "your model's top prediction is class 1" named the
    # slot that had no number in it, on the panel whose whole job is telling a
    # small-model author what their network answered.
    #
    # Masking to -inf keeps the indices honest: a finite value wins, and the
    # count of unusable slots travels beside the answer instead of replacing
    # it. A network that is half NaN is a finding, not a reason to say nothing.
    ranked = row.masked_fill(~finite, float("-inf"))
    top = torch.topk(ranked, k=min(5, row.numel()))
    keep = [
        (int(i), float(v))
        for i, v in zip(top.indices.tolist(), top.values.tolist(), strict=True)
        if v != float("-inf")
    ]
    return {
        "top_index": [i for i, _ in keep],
        "top_value": [round(v, 6) for _, v in keep],
        "argmax": int(torch.argmax(ranked)),
        "n_out": int(row.numel()),
        # 0 on a healthy model, so the panel can say "3 of 10 outputs are not
        # numbers" rather than leaving it to be inferred from a short list.
        "n_nonfinite": n_nonfinite,
    }


# ------------------------------------------------------------------- handle


class CustomHandle:
    """Server-side state for one user-supplied model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.model = None
        self.example = None
        self.status_ = CustomStatus()
        self.rows: list[LayerStat] = []
        self.meta: dict = {}
        # The adapter module itself, kept only for `.py` loads. The causal
        # sweep needs two things `load()` does not return -- the declared TASK
        # and `sample_inputs()` -- and re-importing the file to read them
        # would run the reader's top-level code a second time and could
        # produce a DIFFERENT model from the one being measured.
        self.module = None

    def status(self) -> CustomStatus:
        return self.status_

    def unload(self) -> CustomStatus:
        with self._lock:
            self.model = None
            self.example = None
            self.module = None
            self.rows = []
            self.meta = {}
            self.status_ = CustomStatus()
        return self.status_

    def load(self, path: str | Path) -> CustomStatus:
        """Import an adapter or a TorchScript file. Blocking — use a thread."""
        p = resolve_under_roots(path)
        suffix = p.suffix.lower()

        if suffix == ".py":
            model, example, labels, module = load_from_adapter(p)
            source = "adapter"
        elif suffix in (".pt", ".pth", ".ptc", ".torchscript"):
            model = load_torchscript(p)
            example, labels, module = None, None, None
            source = "torchscript"
        else:
            raise AdapterError(
                f"{p.name}: expected a .py adapter or a TorchScript .pt — got "
                f"{suffix or 'no extension'}."
            )

        total, trainable = count_params(model)
        if example is not None:
            shape = list(getattr(example, "shape", []) or [])
            origin, reason = "adapter", "from your example_input()"
        else:
            shape, reason = suggest_input(model)
            origin = "inferred" if shape else ""

        with self._lock:
            self.model = model
            self.example = example
            self.module = module
            self.rows = []
            self.meta = {}
            self.status_ = CustomStatus(
                loaded=True,
                path=str(p),
                source=source,
                name=type(model).__name__,
                # Read off the model, not asserted. A CustomStatus that says
                # "cpu" about a model whose parameters are on a GPU is a label
                # that contradicts the thing it describes.
                device=_model_device(model),
                n_params=total,
                n_trainable=trainable,
                n_modules=len(leaf_modules(model)),
                input_shape=shape,
                input_origin=origin,
                input_reason=reason,
                labels=labels,
            )
        return self.status_

    def run(self, shape: list[int] | None = None, seed: int = 0) -> dict:
        """One forward pass, hooked. `shape` overrides the adapter's example."""
        import torch

        with self._lock:
            model = self.model
            example = self.example
            status = self.status_
        if model is None:
            raise AdapterError("no custom model is loaded")

        if shape:
            if any(int(d) <= 0 for d in shape):
                raise AdapterError(f"input shape {shape} has a non-positive dimension")
            n = 1
            for d in shape:
                n *= int(d)
            if n > 64_000_000:
                raise AdapterError(
                    f"input shape {shape} is {n:,} elements — too large for a "
                    "single inspection pass."
                )
            gen = torch.Generator().manual_seed(int(seed))
            if _wants_integer_input(model):
                high = _embedding_size(model)
                example = torch.randint(
                    0, max(high, 1), tuple(int(d) for d in shape), generator=gen
                )
            else:
                example = torch.randn(*[int(d) for d in shape], generator=gen)
        elif example is None:
            raise AdapterError(
                "no input to run: give an input shape, or add example_input() "
                "to your adapter."
            )

        rows, meta = inspect(model, example)
        used = list(getattr(example, "shape", []) or [])
        with self._lock:
            self.rows = rows
            self.meta = meta
            self.status_.input_shape = used
            if shape:
                self.status_.input_origin = "user"
                self.status_.input_reason = "the shape you entered"
        return {
            "layers": [r.to_dict() for r in rows],
            "input_shape": used,
            "labels": status.labels,
            **meta,
        }

    def ablate(self, kind: str = "layers", *, grid: int = 0) -> dict:
        """Sweep this model causally. Blocking — use a thread.

        `custom.run` is descriptive: it says what each layer emitted. This
        says what the answer would be without it, which is a different
        question and the only one that supports the word "matters".
        """
        from . import custom_ablate as ablate_mod

        with self._lock:
            model = self.model
            module = self.module
            source = self.status_.source
        if model is None:
            raise AdapterError("no custom model is loaded")
        if module is None:
            raise AdapterError(
                f"a causal sweep needs the adapter that built this model, and "
                f"this one was loaded from {source or 'a file'} rather than a "
                f".py adapter. TorchScript carries weights and no way to say "
                f"what the model is for or what its real inputs look like — "
                f"both of which this measurement needs to be honest. Point "
                f"ModelMRI at an adapter instead."
            )

        task = ablate_mod.read_task(module)
        samples = ablate_mod.read_samples(module)
        if kind == "inputs":
            return ablate_mod.sweep_inputs(
                model,
                samples,
                task=task,
                grid=grid or ablate_mod.DEFAULT_PATCH_GRID,
            ).to_dict()
        if kind == "layers":
            return ablate_mod.sweep_layers(model, samples, task=task).to_dict()
        raise AdapterError(f"unknown sweep {kind!r} — expected 'layers' or 'inputs'.")


def _wants_integer_input(model) -> bool:
    """True when the first parameterised leaf is an Embedding.

    Feeding an Embedding a float tensor raises a type error that reads like a
    bug in ModelMRI rather than a mismatch, so we get this right up front.
    """
    import torch

    for _, mod in leaf_modules(model):
        if isinstance(mod, torch.nn.Embedding):
            return True
        if list(mod.parameters(recurse=False)):
            return False
    return False


def _embedding_size(model) -> int:
    import torch

    for _, mod in leaf_modules(model):
        if isinstance(mod, torch.nn.Embedding):
            return int(mod.num_embeddings)
    return 1


# ------------------------------------------------------------------ discovery

_ADAPTER_HINTS = ("modelmri", "adapter", "mri")

# `def load(` at column zero, with no `self` — a function, not a method.
_MODULE_LEVEL_LOAD = re.compile(r"^def\s+load\s*\(\s*(?!self\b)", re.MULTILINE)
_MODULE_LEVEL_EXAMPLE = re.compile(r"^def\s+example_input\s*\(", re.MULTILINE)


def find_adapters(root: str | Path | None = None, limit: int = 40) -> list[dict]:
    """Python files under the allowed roots that look like adapters.

    Cheap and text-only: reads at most 4 KB per candidate looking for a load()
    definition. It never imports anything — discovery must not run code.
    """
    roots = [Path(root)] if root else allowed_roots()
    skip = {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        "site-packages",
        "dist",
        "build",
        # Agent worktrees are full copies of the repo, so scanning them listed
        # the SAME adapter three times under three paths -- a panel that looks
        # like it found three models when it found one, and the reader has to
        # read three long paths to work out they are the same file.
        ".claude",
        ".worktrees",
        ".tox",
        ".idea",
        # Test suites are full of `def load()` in fixtures and docstrings, and
        # nobody's trained model lives in one. Type the path if yours does.
        "tests",
        "test",
        ".pytest_cache",
    }
    found: list[dict] = []
    seen: set[Path] = set()
    for base in roots:
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if len(found) >= limit:
                return found
            # Relative to the scan root, not the absolute path. `path.parts`
            # includes every ancestor above the root, so a repo that happens to
            # live under a directory named `build`, `dist`, `node_modules` or
            # `venv` had EVERY candidate skipped and the scan silently found
            # nothing — the result depending on where the user keeps their code
            # rather than on what is in it.
            if any(part in skip for part in path.relative_to(base).parts):
                continue
            # NOT ModelMRI'S OWN TEMPLATE. `examples/adapter_template.py` ships
            # inside this package as something to copy, and listing it beside
            # somebody's real model reads as "here is a model you have" when it
            # is a blank form -- worse when a checkout, a worktree and an
            # install all contribute a copy and the panel shows four. The
            # refusal that tells you to write an adapter already names the file
            # by path; that is where it belongs, not in a list of your models.
            if path.name == "adapter_template.py":
                continue
            if path.name.startswith("test_"):
                continue
            if path in seen:
                continue
            seen.add(path)
            try:
                head = path.read_text(encoding="utf-8", errors="replace")[:4096]
            except OSError:
                # PermissionError, a file removed since the glob, a cloud
                # placeholder that will not materialise. `errors="replace"`
                # already rules out a decode failure, so nothing but OSError
                # gets here. Skipping is right and cannot mislead: this is a
                # convenience list of candidates, and a file missing from it
                # can still be loaded by typing its path — which is the only
                # way anything gets loaded here anyway.
                continue
            # Module level only. A substring search for "def load(" matches
            # every `def load(self, ...)` method in the world, which offered
            # ModelMRI's own saes.py and vla.py as models you had trained.
            if not _MODULE_LEVEL_LOAD.search(head):
                continue
            score = sum(h in path.name.lower() for h in _ADAPTER_HINTS)
            found.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "dir": str(path.parent),
                    "has_example": bool(_MODULE_LEVEL_EXAMPLE.search(head)),
                    "hint": score > 0,
                }
            )
    found.sort(key=lambda f: (not f["hint"], not f["has_example"], f["name"]))
    return found


def checkpoint_kind(path: Path) -> str:
    """What a .pt/.pth actually IS, without executing a byte of it.

    The panel grouped these by extension under a heading reading TORCHSCRIPT,
    so somebody's state_dict was filed as TorchScript -- two different things
    that fail in two different ways, and the heading told them the wrong one
    before they clicked.

    Both are zip archives since torch 1.6, and the member names say which is
    which: a TorchScript archive carries `constants.pkl` and a `code/` tree
    (the serialised graph), while `torch.save` of a tensor dict writes
    `data.pkl` and a `data/` directory of storages. Reading the central
    directory is a read of the file's index -- no unpickling, no import, and
    nothing from the file is run.

    Returns "gguf" | "torchscript" | "checkpoint" | "legacy" | "unreadable".
    Never raises: this labels a row in a candidate list, and a file that cannot
    be inspected is still a file worth showing with an honest label on it.
    """
    import zipfile

    # GGUF is not a zip and never will be, so it is decided by its magic bytes
    # before the archive logic below gets a chance to call it unreadable.
    try:
        with open(path, "rb") as fh:
            if fh.read(4) == b"GGUF":
                return "gguf"
    except OSError:
        return "unreadable"

    try:
        # Explicitly, because `is_zipfile` answers False for a path that does
        # not exist rather than raising -- so a file that vanished between the
        # scan and this call was being labelled "legacy", which is a claim
        # about its format made about a file that is not there.
        if not path.is_file():
            return "unreadable"
        if not zipfile.is_zipfile(path):
            # Pre-1.6 torch.save, or not a torch file at all. Either way it is
            # not TorchScript, which only ever used the zip container.
            return "legacy"
        with zipfile.ZipFile(path) as z:
            names = z.namelist()[:400]
    except (OSError, zipfile.BadZipFile, ValueError):
        return "unreadable"

    tail = [n.rsplit("/", 1)[-1] for n in names]
    if "constants.pkl" in tail or any("/code/" in n for n in names):
        return "torchscript"
    if "data.pkl" in tail:
        return "checkpoint"
    return "unreadable"


def find_torchscript(limit: int = 40) -> list[dict]:
    """Loadable-looking checkpoints under the allowed roots.

    Named for history: it used to list only TorchScript. It lists every
    .pt/.pth candidate now, each labelled with what it actually is, and
    the file is READ (its zip index) but never executed.
    """
    out: list[dict] = []
    # ONE ROW PER FILE. The allowed roots overlap by design -- the working
    # directory, MODELMRI_MODELS_DIR and any folder added this session -- so
    # a checkpoint sitting under two of them was listed twice, with identical
    # names and identical paths. Two rows for one file reads as two models.
    seen: set[Path] = set()
    skip = {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        "site-packages",
        ".claude",
        ".worktrees",
    }
    for base in allowed_roots():
        if not base.is_dir():
            continue
        # `.gguf` is here because the scanner used to find the format most
        # people running models locally actually have and then not list it at
        # all — so the panel that exists to say "here is what is on your disk"
        # was silently omitting most of it. It still cannot be RUN, and
        # `checkpoint_kind` labels it so, but it can now be READ.
        for pattern in ("*.pt", "*.pth", "*.torchscript", "*.gguf"):
            for path in sorted(base.rglob(pattern)):
                if len(out) >= limit:
                    return out
                # Relative to the scan root, not the absolute path -- the
                # same fix `find_adapters` above already carries, in its own
                # words: "`path.parts` includes every ancestor above the root,
                # so a repo that happens to live under a directory named
                # `build`, `dist`, `node_modules` or `venv` had EVERY
                # candidate skipped and the scan silently found nothing".
                #
                # It was fixed there and left here, so the checkpoint scanner
                # still returned an empty list for anyone whose models sit
                # under a path with `venv` or `site-packages` anywhere above
                # them -- an answer about where they keep their files, printed
                # as an answer about what they have.
                if any(part in skip for part in path.relative_to(base).parts):
                    continue
                try:
                    key = path.resolve()
                except OSError:
                    key = path
                if key in seen:
                    continue
                seen.add(key)
                try:
                    size = path.stat().st_size
                except OSError:
                    # Same shrug as `find_adapters` above, and the same reason
                    # it is safe — a candidate list, not a capability. Skipping
                    # rather than listing it with a made-up size, because the
                    # size is the only thing this row adds over the filename.
                    continue
                out.append(
                    {
                        "path": str(path),
                        "name": path.name,
                        "dir": str(path.parent),
                        "mb": round(size / 1e6, 2),
                        # What it IS, read from the archive index rather than
                        # guessed from the extension. `.pt` and `.pth` are the
                        # same container and say nothing about the contents.
                        "kind": checkpoint_kind(path),
                    }
                )
    return out
