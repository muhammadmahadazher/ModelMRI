# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Open an attribution graph somebody else computed.

circuit-tracer builds these and nothing outside its own Neuronpedia flow opens
one. Reading is a fraction of the cost of building, and the read is the part
that makes a graph shareable — so this renders one in the same viewer as every
other finding, behind a banner naming the file, the tool and the model.

**Nothing here claims ModelMRI measured any of it.** That is not optional
chrome. A rendered graph this tool did not compute must never be mistakable
for one it did, so provenance is a required field of the result rather than a
caption the UI may forget to draw.

## Reading a pickle from a stranger

A `.pt` is a pickle, and unpickling runs code. `torch.load(weights_only=True)`
is the usual answer and it does not work here — measured, not assumed: a real
circuit-tracer graph stores `cfg` as a `UnifiedConfig` and `logit_targets` as
`LogitTarget` objects, and weights-only refuses the whole file on the first
one:

    UnpicklingError: Unsupported global: GLOBAL circuit_tracer...UnifiedConfig
    was not an allowed global by default

Refusing the file outright would mean never reading the model name, which the
banner requires. Loading with `weights_only=False` would mean executing
whatever the file says, which is the thing not to do.

So the unpickler is restricted instead. `_SafeUnpickler.find_class` allows
torch's tensor-rebuild machinery and nothing else; every other class named by
the file is answered with an inert stub *this module* defines, so the named
module is never imported and none of its code runs. The attributes still
arrive, because pickle sets them on the object it was handed — which is how
`cfg.model_name` reaches the banner without trusting the file.

## Size

A graph's adjacency matrix is nodes x nodes. circuit-tracer graphs for a 2B
model run to thousands of nodes, so the dense matrix is the largest thing in
the file by an order of magnitude — 10,000 nodes is 400 MB at float32, and
`.tolist()` on that is several gigabytes of Python floats and a dead machine.

Nothing here ever materialises it. The shape is checked before anything is
read, `summary()` reduces on the tensor, and `edges()` returns the strongest
`limit` edges found with `topk` on a flattened view. The cap is reported so a
pruned graph is never mistaken for a whole one.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import BadRequest, Refusal

# The layout this reader was written against: circuit-tracer's `Graph.to_pt`.
# There is no version field in the file, so the layout is pinned by the keys it
# must carry and an unrecognised one is refused by name rather than guessed at.
REQUIRED = ("input_tokens", "adjacency_matrix")
KNOWN = (
    "input_string",
    "adjacency_matrix",
    "cfg",
    "active_features",
    "logit_targets",
    "logit_probabilities",
    "vocab_size",
    "input_tokens",
    "selected_features",
    "activation_values",
    "scan_name",
    # `scan` is the pre-rename spelling of `scan_name`; circuit-tracer's own
    # loader still accepts it, so this does too.
    "scan",
)

# Above this the file is refused rather than read. Not a memory limit — the
# matrix is never materialised — but a sanity bound: a plausible attribution
# graph is thousands of nodes, and a header claiming millions is a corrupt or
# hostile file, and `nodes * nodes` on it overflows into a hang.
# 20,000 nodes is 400M elements -- 1.6 GB at float32 -- and already far past
# any published attribution graph. 200,000 was the first bound and it permits
# 4x10^10 elements: a ~2 KB file declaring `size=(200000,200000)` over a
# one-element storage (which is what `torch.zeros(1).expand(200_000,200_000)`
# writes) is 2-D, square and under the old limit, and then `torch.isfinite`
# materialises 40 GB. The comment claimed the bound existed to stop exactly
# that and the number did not deliver it.
MAX_NODES = 20_000

# How many edges a summary carries by default. A dense matrix has nodes^2 of
# them and almost all are ~0; the graph people read is the strong tail.
DEFAULT_EDGE_LIMIT = 2_000


class _Foreign:
    """A class the file named. Defined HERE; the named module is never imported.

    Pickle constructs one of these and sets the attributes the file carried, so
    `cfg.model_name` is readable without any circuit-tracer code running. The
    origin is kept so the reader can say what the file claimed the object was.
    """

    _origin = "?"

    def __init__(self, *args, **kwargs) -> None:
        """Accept anything and do nothing with it.

        `object.__init__` takes no arguments, so a foreign class pickled
        through `__reduce__`-with-args -- Enum members, pathlib.Path,
        functools.partial, datetime -- raised `TypeError: X() takes no
        arguments`, which surfaced as "could not be read as a torch archive",
        blaming the file for the wrong thing. Today's circuit-tracer graphs
        happen to use NEWOBJ; one new Enum-typed field would have broken the
        reader with a misleading refusal.
        """
        if args:
            self.__dict__["_args"] = args
        if kwargs:
            self.__dict__.update(kwargs)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<foreign {self._origin} {sorted(self.__dict__)}>"

    def as_dict(self) -> dict:
        """The attributes, JSON-safe, with nothing executed to get them."""
        out = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                out[key] = value
            elif isinstance(value, _Foreign):
                out[key] = value.as_dict()
        return out


# EXACT (module, name) pairs that may resolve to a real callable. Everything
# else -- including anything else under torch or numpy -- becomes an inert
# stub. This is the security boundary and it is a pair list, not a module list,
# because a module list does not work.
#
# The first version allowed whole roots: `module.split(".")[0] in ("torch",
# "collections", "numpy")`. That is arbitrary code execution, demonstrated
# rather than argued:
#
#     numpy/testing/_private/utils.py:  def runstring(astr, dict): exec(astr, dict)
#
# `numpy.testing._private.utils` passes a top-level check on "numpy", so a
# GLOBAL naming it plus one REDUCE runs `exec` on attacker text. A payload
# built that way wrote a file through `circuit.read` on this machine. The same
# hole admits `torch.hub.load` (clones and imports a GitHub repo) and, at
# pickle protocol >= 4 where the attacker writes the PROTO opcode,
# `_getattribute` splits a DOTTED name -- so GLOBAL "torch.serialization"
# "pickle.loads" hands back the real unrestricted loads and the restriction is
# gone in one hop.
#
# What a real torch.save actually needs was measured, not guessed: float32,
# integer, bool, float16, bfloat16, non-contiguous tensors, and nested
# str/int/list/dict/None all round-trip through exactly two pairs.
_ALLOWED_GLOBALS = frozenset(
    {
        ("collections", "OrderedDict"),
        ("torch._utils", "_rebuild_tensor_v2"),
        # Not exercised by the files measured, but unambiguously tensor-rebuild
        # machinery that older and sparser checkpoints do use. Each is a
        # constructor for tensor data and none of them executes caller text.
        ("torch._utils", "_rebuild_tensor"),
        ("torch._utils", "_rebuild_sparse_tensor"),
        ("torch._utils", "_rebuild_meta_tensor_no_storage"),
        ("torch._utils", "_rebuild_wrapper_subclass"),
        ("torch.storage", "_load_from_bytes"),
        ("torch", "Size"),
        ("torch", "device"),
        ("torch", "dtype"),
    }
    # The typed storages a legacy checkpoint names directly. Enumerated rather
    # than pattern-matched so the set stays finite and readable.
    | {
        ("torch", f"{t}Storage")
        for t in (
            "Float",
            "Double",
            "Half",
            "BFloat16",
            "Long",
            "Int",
            "Short",
            "Char",
            "Byte",
            "Bool",
            "ComplexFloat",
            "ComplexDouble",
        )
    }
)


class _SafeUnpickler(pickle.Unpickler):
    """Unpickle without importing anything the file asks for by name.

    The registry is a CLASS attribute, filled by `_reader_for` into a fresh
    per-call subclass. It cannot be per-instance: `torch.load` does not call
    `pickle_module.load` — it takes `pickle_module.Unpickler` and subclasses
    it, constructing that itself, so an instance this module made never
    exists. The first version recorded into `self.foreign` and every read
    reported zero foreign classes, which made `producer` say "unknown" for a
    genuine circuit-tracer graph. Found by reading a real one.
    """

    _foreign: dict[str, type] = {}

    def find_class(self, module: str, name: str):
        # A dotted NAME is an attribute walk that pickle protocol >= 4 performs
        # for us, and the attacker chooses the protocol. `GLOBAL
        # "torch.serialization" "os.system"` resolves through it, so the dot is
        # refused before the pair is even consulted.
        if "." not in name and (module, name) in _ALLOWED_GLOBALS:
            return super().find_class(module, name)
        origin = f"{module}.{name}"
        registry = type(self)._foreign
        if origin not in registry:
            # A CLASS, not a factory: pickle's NEWOBJ opcode requires a type
            # and raises "NEWOBJ class argument must be a type, not function"
            # otherwise. Found by trying the function first.
            registry[origin] = type(name, (_Foreign,), {"_origin": origin})
        return registry[origin]


def _reader_for(registry: dict[str, type]):
    """A pickle-module stand-in whose Unpickler records into `registry`.

    Built per call so two concurrent reads cannot write into one another's
    registry — the alternative, a module-level dict, is a data race the moment
    the server reads two graphs at once.
    """

    class _Unpickler(_SafeUnpickler):
        _foreign = registry

    class _Shim:
        Unpickler = _Unpickler

        @staticmethod
        def load(file, **kw):
            return _Unpickler(file, **kw).load()

    return _Shim


@dataclass
class Graph:
    """An attribution graph read from someone else's file."""

    path: str
    n_nodes: int
    n_tokens: int
    prompt: str
    # Everything below is what the FILE said, never what this tool measured.
    model: str | None
    scan: str | None
    producer: str
    foreign_classes: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Held as a tensor and never converted wholesale. See the module docstring.
    _adjacency: Any = None

    @property
    def provenance(self) -> dict:
        """Who computed this, on what. Required, not decorative.

        `measured_by` is a constant sentence rather than a flag the UI has to
        remember to interpret: a graph ModelMRI did not compute must never be
        mistakable for one it did, and the safest place for that claim is
        inside the payload.
        """
        return {
            "file": Path(self.path).name,
            "producer": self.producer,
            "model": self.model,
            "scan": self.scan,
            "measured_by": (
                "This graph was computed by another tool and read here. "
                "ModelMRI did not run the model, did not compute these "
                "attributions, and cannot vouch for them."
            ),
        }

    def _cached_edges(self, limit: int) -> list[dict]:
        """`edges()` memoised per limit.

        `open_graph` called `summary()` (which calls `edges()`), then
        `edges()`, then `to_session()` (which calls both again) -- five full
        `abs()` passes over the matrix for one command. Each pass allocates a
        nodes^2 temporary, so at the advertised 10,000-node size that was
        2 GB of transient allocation to print four rows.
        """
        cache = self.__dict__.setdefault("_edge_cache", {})
        if limit not in cache:
            cache[limit] = self._edges_uncached(limit=limit)
        return cache[limit]

    def summary(self, *, edge_limit: int = DEFAULT_EDGE_LIMIT) -> dict:
        """Shape and strength, reduced ON the tensor.

        Every statistic here is a torch reduction over the adjacency matrix.
        Nothing calls `.tolist()` on it, because at 10,000 nodes that is
        several gigabytes of Python floats.
        """
        import torch

        a = self._adjacency
        if a is None:
            return {"nodes": self.n_nodes, "means": "no adjacency matrix"}
        flat = a.reshape(-1)
        nonzero = int(torch.count_nonzero(a))
        total = int(flat.numel())
        strongest = self.edges(limit=edge_limit)
        return {
            "nodes": self.n_nodes,
            "tokens": self.n_tokens,
            "possible_edges": total,
            "nonzero_edges": nonzero,
            "density": round(nonzero / total, 8) if total else None,
            "max_abs_weight": float(flat.abs().max()) if total else None,
            "returned_edges": len(strongest),
            "edge_limit": edge_limit,
            "truncated": nonzero > len(strongest),
            "means": (
                f"An attribution graph has nodes x nodes possible edges and "
                f"almost all are zero, so the {len(strongest)} strongest by "
                f"absolute weight are returned rather than all "
                f"{total:,}. `truncated` says whether that cap bit. Every "
                f"number here is the FILE's, reduced on the tensor — ModelMRI "
                f"did not compute any of it."
            ),
        }

    def edges(self, *, limit: int = DEFAULT_EDGE_LIMIT) -> list[dict]:
        return self._cached_edges(limit)

    def _edges_uncached(self, *, limit: int = DEFAULT_EDGE_LIMIT) -> list[dict]:
        """The `limit` strongest edges, by absolute weight.

        `topk` on a flattened view, so the peak allocation is `limit` rather
        than the matrix. Sorting `nodes^2` values to take 2,000 of them would
        be the same answer at a thousand times the cost.
        """
        import torch

        a = self._adjacency
        if a is None or a.numel() == 0 or limit <= 0:
            return []
        flat = a.reshape(-1)
        k = min(int(limit), int(flat.numel()))
        values, indices = torch.topk(flat.abs(), k)
        cols = int(a.shape[1])
        out = []
        for value, index in zip(values.tolist(), indices.tolist(), strict=True):
            if value == 0:
                # topk pads with zeros once the real edges run out. A zero
                # edge is the absence of one, so it is dropped rather than
                # reported as an edge of no weight.
                break
            row, col = divmod(index, cols)
            out.append(
                {
                    "source": col,
                    "target": row,
                    "weight": float(flat[index]),
                }
            )
        return out

    def to_dict(self, *, edge_limit: int = DEFAULT_EDGE_LIMIT) -> dict:
        return {
            "path": self.path,
            "prompt": self.prompt,
            "tokens": self.tokens,
            "n_nodes": self.n_nodes,
            "n_tokens": self.n_tokens,
            "provenance": self.provenance,
            "foreign_classes": self.foreign_classes,
            "summary": self.summary(edge_limit=edge_limit),
            "edges": self.edges(limit=edge_limit),
            "notes": self.notes,
        }


def to_session(graph: Graph, *, edge_limit: int = DEFAULT_EDGE_LIMIT) -> bytes:
    """A `.mri` carrying this graph, so it travels like every other finding.

    The session's own fields are deliberately empty: there is no attention, no
    lens and no generation here, because ModelMRI ran nothing. `model_id` is
    left None rather than filled from the graph -- a `.mri` whose header names
    a model reads as one this tool loaded, and the model is already in the
    graph's provenance where it is labelled as the FILE's claim.
    """
    from . import session

    return session.build(
        model_id=None,
        device=None,
        dtype=None,
        n_params=None,
        tokens=list(graph.tokens),
        generation="",
        prompt=graph.prompt,
        attention={},
        lens=[],
        n_layers=0,
        n_heads=0,
        note=f"attribution graph read from {Path(graph.path).name}",
        graph={
            "n_nodes": graph.n_nodes,
            "edges": graph.edges(limit=edge_limit),
            "provenance": graph.provenance,
            "prompt": graph.prompt,
            "summary": graph.summary(edge_limit=edge_limit),
            "notes": graph.notes,
        },
    )


def read(path: str | Path, *, max_nodes: int = MAX_NODES) -> Graph:
    """Read an attribution graph, refusing anything that is not one.

    Same posture as `session.parse`: a ragged graph or an implausible node
    count stops here, not in the recipient's browser. Every refusal names what
    was wrong with the file rather than what went wrong inside this function.
    """
    import torch

    target = Path(path).expanduser()
    if not target.is_file():
        raise BadRequest(f"no such file: {target}")

    # BEFORE torch.load, because torch.load dispatches on this itself and the
    # dispatch happens ahead of `pickle_module`. `torch/serialization.py`:
    #
    #     if _is_torchscript_zip(opened_zipfile):
    #         if weights_only: raise RuntimeError(...)
    #         return torch.jit.load(opened_file, ...)
    #
    # `weights_only=False` is required here (see the module docstring), so that
    # guard is inert and a `.pt` carrying a record named `constants.pkl` would
    # be handed to the TorchScript loader -- the C++ unpickler, on a stranger's
    # archive, with the restricted reader discarded. Refused by name instead.
    import zipfile

    try:
        if zipfile.is_zipfile(target):
            with zipfile.ZipFile(target) as archive:
                names = archive.namelist()
            if any(n.rsplit("/", 1)[-1] == "constants.pkl" for n in names):
                raise Refusal(
                    f"{target.name} is a TorchScript archive, not an "
                    "attribution graph. torch.load hands those to "
                    "`torch.jit.load` before any restricted reader can see "
                    "them, so this refuses rather than opening it."
                )
    except zipfile.BadZipFile:
        # Not a zip; the legacy pickle path below handles it and reports.
        pass

    registry: dict[str, type] = {}
    shim = _reader_for(registry)

    try:
        # weights_only=False is deliberate AND safe here: the shim above is the
        # thing deciding what may be constructed, and it imports nothing the
        # file names. Passing weights_only=True would ignore the shim and
        # refuse the file, which is what this whole design exists to avoid.
        raw = torch.load(
            target, pickle_module=shim, weights_only=False, map_location="cpu"
        )
    except Exception as err:
        raise Refusal(
            f"{target.name} could not be read as a torch archive "
            f"({type(err).__name__}). An attribution graph is written by "
            "circuit-tracer's `Graph.to_pt`; a `.pt` from anything else will "
            "not have this shape."
        ) from err

    if not isinstance(raw, dict):
        raise Refusal(
            f"{target.name} holds a {type(raw).__name__}, not the dict that "
            "`Graph.to_pt` writes. This is a torch file, but not a graph."
        )

    missing = [k for k in REQUIRED if k not in raw]
    if missing:
        raise Refusal(
            f"{target.name} is missing {', '.join(missing)}, so it is not a "
            f"circuit-tracer attribution graph. It holds: "
            f"{', '.join(sorted(str(k) for k in raw)) or 'nothing'}."
        )

    notes: list[str] = []
    unknown = [str(k) for k in raw if k not in KNOWN]
    if unknown:
        # Not a refusal: a newer circuit-tracer adding a key should still open.
        # Said out loud so an unread field is never mistaken for an absent one.
        notes.append(
            f"keys this reader does not know and did not read: "
            f"{', '.join(sorted(unknown))}"
        )

    adjacency = raw.get("adjacency_matrix")
    if not isinstance(adjacency, torch.Tensor):
        raise Refusal(
            f"{target.name}'s adjacency_matrix is a "
            f"{type(adjacency).__name__}, not a tensor."
        )
    if adjacency.ndim != 2:
        raise Refusal(
            f"{target.name}'s adjacency_matrix is {adjacency.ndim}-dimensional "
            f"{tuple(adjacency.shape)}; an attribution graph's is a square "
            "matrix of node-to-node weights."
        )
    rows, cols = int(adjacency.shape[0]), int(adjacency.shape[1])
    if rows != cols:
        raise Refusal(
            f"{target.name}'s adjacency_matrix is {rows}x{cols}, which is not "
            "square, so it cannot be node-to-node. A ragged graph stops here "
            "rather than in a browser."
        )
    if rows > max_nodes:
        raise Refusal(
            f"{target.name} claims {rows:,} nodes, above the {max_nodes:,} "
            "this reads. A real attribution graph is thousands; a number this "
            "large is a corrupt or hostile header, and squaring it is how a "
            "reader hangs instead of refusing."
        )
    # Elements, not just the side length -- and BEFORE any full-tensor op.
    # `isfinite` below allocates a dense bool tensor of this size.
    if rows * cols > MAX_NODES * MAX_NODES:
        raise Refusal(
            f"{target.name}'s adjacency matrix declares {rows * cols:,} "
            "elements, which is more than this reads."
        )
    if not adjacency.is_contiguous():
        # An expanded or transposed view makes `reshape(-1)` copy the whole
        # matrix. Made contiguous once, here, rather than silently once per
        # reduction downstream.
        adjacency = adjacency.contiguous()
    if not torch.isfinite(adjacency).all():
        notes.append(
            "the adjacency matrix contains non-finite weights (nan or inf); "
            "they are reported as they are and not cleaned"
        )

    tokens_tensor = raw.get("input_tokens")
    n_tokens = (
        int(tokens_tensor.numel()) if isinstance(tokens_tensor, torch.Tensor) else 0
    )
    if isinstance(tokens_tensor, torch.Tensor) and tokens_tensor.ndim > 2:
        raise Refusal(
            f"{target.name}'s input_tokens is {tokens_tensor.ndim}-dimensional; "
            "expected a sequence of token ids."
        )

    cfg = raw.get("cfg")
    model = None
    if isinstance(cfg, _Foreign):
        # Several spellings across circuit-tracer versions. Tried in order and
        # left as None when none is present -- an unnamed model is reported as
        # unnamed, because inventing one would put a claim in the banner.
        for key in ("model_name", "model", "name", "tokenizer_name"):
            value = cfg.__dict__.get(key)
            if isinstance(value, str) and value:
                model = value
                break
    elif isinstance(cfg, str):
        model = cfg
    if model is None:
        notes.append(
            "the file does not name the model it was computed on, so the "
            "banner cannot either"
        )

    scan = raw.get("scan_name", raw.get("scan"))
    if isinstance(scan, (list, tuple)):
        scan = ", ".join(str(s) for s in scan)
    elif scan is not None and not isinstance(scan, str):
        scan = str(scan)

    foreign = sorted(registry)
    producer = (
        "circuit-tracer"
        if any("circuit_tracer" in f for f in foreign)
        else ("unknown — the file names no circuit-tracer classes")
    )
    if not foreign:
        notes.append(
            "no foreign classes in this file, so its producer is unattested; "
            "the shape matches a circuit-tracer graph but nothing proves it"
        )

    prompt = raw.get("input_string")
    if not isinstance(prompt, str):
        prompt = ""

    return Graph(
        path=str(target),
        n_nodes=rows,
        n_tokens=n_tokens,
        prompt=prompt,
        model=model,
        scan=scan,
        producer=producer,
        foreign_classes=foreign,
        tokens=[],
        notes=notes,
        _adjacency=adjacency,
    )
