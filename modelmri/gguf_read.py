"""Read what is inside a GGUF, without loading it and without llama.cpp.

The scanner has always found `.gguf` files and then said it could not open
them. That is the format most people running models on their own machine
actually have — Ollama, LM Studio, llama.cpp and Jan all use it — so the tool
was finding the majority of local weights and refusing them with a note.

This module still does not *run* one -- reading a table that describes tensors
and unpacking the tensors themselves are different jobs, and only the second
needs a GPU, a dependency and a memory budget. `gguf_load.py` does that half,
and prices it before it starts. Everything here is in the header alone:
architecture,
context length, rope settings, tokeniser, and a full tensor table with each
tensor's ggml type, shape and byte count.

**Stdlib only, and deliberately not the `gguf` pip package.** The roadmap
called for wrapping `gguf-py`, whose argument is memory-mapped tensor access —
but nothing here reads tensor DATA, only the table describing it, and that
table is a length-prefixed structure at the front of the file. `fit.py` makes
the same call about safetensors for the same reason: a reader that needs no
dependency works offline, adds no optional extra, and cannot be broken by a
release of a package versioned against llama.cpp. Bytes read for a 4 GB model:
the header, typically well under a megabyte.

**Bits-per-weight is arithmetic, not a label.** Every runner shows you `Q4_K_M`
and a file size. That label is a preset name, not a measurement, and it hides
the thing people actually want to know: the big tensors are 4.5 bpw while
`token_embd.weight` and `output.weight` are usually left much higher. So this
computes `bytes * 8 / elements` per tensor and rolls it up, and it names the
tensors that sit above the headline rather than averaging them into it.

**An unknown quant type stays unknown.** ggml adds types regularly. One this
table has not seen is reported as `ggml type 37 (unknown)` with its byte count
omitted rather than bucketed into the nearest familiar thing — a wrong
bits-per-weight computed confidently is worse than an absent one.
"""

from __future__ import annotations

import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .errors import BadRequest, Refusal

MAGIC = b"GGUF"

# A header larger than this is not a header. Guards against a corrupt length
# turning into a multi-gigabyte read.
MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_KV = 100_000
MAX_TENSORS = 500_000

# GGUF metadata value types.
(
    _UINT8,
    _INT8,
    _UINT16,
    _INT16,
    _UINT32,
    _INT32,
    _FLOAT32,
    _BOOL,
    _STRING,
    _ARRAY,
    _UINT64,
    _INT64,
    _FLOAT64,
) = range(13)

_SCALARS = {
    _UINT8: ("<B", 1),
    _INT8: ("<b", 1),
    _UINT16: ("<H", 2),
    _INT16: ("<h", 2),
    _UINT32: ("<I", 4),
    _INT32: ("<i", 4),
    _FLOAT32: ("<f", 4),
    _BOOL: ("<?", 1),
    _UINT64: ("<Q", 8),
    _INT64: ("<q", 8),
    _FLOAT64: ("<d", 8),
}

# ggml tensor type -> (name, elements per block, bytes per block).
#
# The pair is what makes byte counts exact: a quantised tensor stores whole
# blocks, so its size is `elements / block * bytes`, never `elements * bpw`.
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
}


@dataclass
class Tensor:
    name: str
    ggml_type: int
    type_name: str
    dims: list[int]
    elements: int
    # None when the ggml type is one this reader does not know. Not 0, and not
    # a guess: an unknown type has an unknown size, and inventing one would
    # make every roll-up containing it quietly wrong.
    bytes: int | None
    offset: int

    @property
    def bpw(self) -> float | None:
        if self.bytes is None or not self.elements:
            return None
        return self.bytes * 8 / self.elements

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bpw"] = round(self.bpw, 3) if self.bpw is not None else None
        return d


@dataclass
class Gguf:
    path: str
    version: int
    tensor_count: int
    metadata: dict
    tensors: list[Tensor]
    unknown_types: list[int] = field(default_factory=list)
    # Where the tensor BLOB begins: the end of the header, rounded up to
    # `general.alignment`. Every `Tensor.offset` is relative to this.
    #
    # Recorded because the parser is the only thing that knows it, and
    # discarding it forced `quantdiff` to reverse-engineer the base from the
    # file size -- `file_size - (last offset + size)`, which is only right if
    # the file ends exactly at the last tensor. GGUF writers pad to alignment
    # after it, so that inference ran long by the padding and then rounded UP
    # again, putting every read past the true start. 0 when unknown.
    data_offset: int = 0

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "version": self.version,
            "tensor_count": self.tensor_count,
            "metadata": self.metadata,
            "tensors": [t.to_dict() for t in self.tensors],
            "unknown_types": self.unknown_types,
            "summary": self.summary(),
        }

    # ------------------------------------------------------------- roll-ups

    def summary(self) -> dict:
        """Architecture, size, and where the bits actually went."""
        known = [t for t in self.tensors if t.bytes is not None]
        unmeasured = len(self.tensors) - len(known)

        # Parameters come from EVERY tensor. `elements` is read from `dims`
        # before the ggml type is consulted, so it is exactly as known for an
        # unknown type as for F32 — excluding those was the bug. A file whose
        # bulk tensors use a type this table has not seen (llama.cpp is at 39
        # while this stops at 30; shipping gpt-oss GGUFs use MXFP4 for their
        # FFN weights) reported 131,072 parameters for a 1.44B model, wrong by
        # 11,009x, with only an unrelated `unmeasured_tensors` field dissenting.
        elements = sum(t.elements for t in self.tensors)
        measured_elements = sum(t.elements for t in known)
        payload = sum(t.bytes or 0 for t in known)
        by_type: dict[str, dict] = {}
        for t in known:
            row = by_type.setdefault(
                t.type_name, {"tensors": 0, "elements": 0, "bytes": 0}
            )
            row["tensors"] += 1
            row["elements"] += t.elements
            row["bytes"] += t.bytes or 0
        for _name, row in by_type.items():
            row["bpw"] = (
                round(row["bytes"] * 8 / row["elements"], 3)
                if row["elements"]
                else None
            )

        # The tensors most often left at higher precision. Named rather than
        # averaged in, because "this model is 4.5 bpw" is false when the
        # embedding and output layers are 6 or 16 and they are large.
        headline = _dominant_type(by_type)
        outliers = [
            {"name": t.name, "type": t.type_name, "bpw": round(t.bpw or 0, 3)}
            for t in known
            if headline and t.type_name != headline and t.elements > 0
        ]
        outliers.sort(key=lambda r: -r["bpw"])

        meta = self.metadata
        arch = str(meta.get("general.architecture", "") or "")

        # Everything derived from BYTES is refused outright when any tensor
        # could not be sized. The module already refuses per tensor — `bytes`
        # and `bpw` are None for an unknown type — and then this method threw
        # that discipline away by averaging over the leftovers and printing the
        # result as the file's headline. A partial average presented as a whole
        # one is the confidently-wrong number the docstring says is worse than
        # an absent one.
        whole = unmeasured == 0
        why = (
            None
            if whole
            else (
                f"{unmeasured} of {len(self.tensors)} tensors use a ggml type "
                f"this reader does not know ({', '.join(str(t) for t in self.unknown_types)}), "
                "so their size is unknown and any byte total or bits-per-weight "
                "over this file would be an average of the parts that happened "
                "to be recognised."
            )
        )
        return {
            "architecture": arch or None,
            "name": meta.get("general.name"),
            "quantisation_label": meta.get("general.file_type_name"),
            # Exact regardless: element counts come from `dims`, not the type.
            "parameters": elements,
            "measured_parameters": measured_elements,
            "tensor_bytes": payload if whole else None,
            "effective_bpw": (
                round(payload * 8 / measured_elements, 3)
                if whole and measured_elements
                else None
            ),
            "by_type": by_type,
            "by_type_covers_whole_file": whole,
            "dominant_type": headline if whole else None,
            "why_unmeasured": why,
            # Capped at twelve, and the total travels with it. The panel
            # then cuts to six, so a file with forty tensors above its
            # headline showed six and said nothing — a list that reads as
            # "these are the ones sitting above the dominant type" when it is
            # 15% of them. Both cuts are disclosed now, from this one number.
            "higher_precision_tensors": outliers[:12],
            "n_higher_precision_tensors": len(outliers),
            "context_length": meta.get(f"{arch}.context_length") if arch else None,
            "block_count": meta.get(f"{arch}.block_count") if arch else None,
            "embedding_length": meta.get(f"{arch}.embedding_length") if arch else None,
            "head_count": meta.get(f"{arch}.attention.head_count") if arch else None,
            "head_count_kv": (
                meta.get(f"{arch}.attention.head_count_kv") if arch else None
            ),
            "tokenizer": meta.get("tokenizer.ggml.model"),
            "unmeasured_tensors": unmeasured,
            "means": (
                "Bits per weight is bytes x 8 / elements, computed per tensor "
                "from the file's own table. It is not the quantisation label, "
                "which is a preset name — the tensors listed as higher "
                "precision sit above the headline and are excluded from it."
                if whole
                else "Parameter count is exact — element counts are read from the "
                "tensor shapes and do not depend on the quantisation type. "
                "Byte totals and bits-per-weight are withheld: see "
                "why_unmeasured."
            ),
        }


def _dominant_type(by_type: dict[str, dict]) -> str | None:
    """The type holding the most elements — the one the label refers to."""
    if not by_type:
        return None
    return max(by_type.items(), key=lambda kv: kv[1]["elements"])[0]


# ------------------------------------------------------------------ parsing


class _Cursor:
    """A bounded reader over the header bytes. Every read is length-checked.

    A GGUF from the internet is a stranger's file, and this parses it. Running
    off the end has to be a refusal naming the file, never an IndexError
    surfacing as a 500.
    """

    def __init__(self, data: bytes, name: str) -> None:
        self.data = data
        self.at = 0
        self.name = name

    def take(self, n: int) -> bytes:
        if n < 0 or self.at + n > len(self.data):
            raise BadRequest(
                f"{self.name} ends in the middle of its header — it is "
                "truncated or not a GGUF file."
            )
        out = self.data[self.at : self.at + n]
        self.at += n
        return out

    def scalar(self, fmt: str, size: int):
        return struct.unpack(fmt, self.take(size))[0]

    def string(self) -> str:
        length = self.scalar("<Q", 8)
        if length > MAX_HEADER_BYTES:
            raise BadRequest(f"{self.name} declares an implausible string length")
        return self.take(length).decode("utf-8", errors="replace")

    def value(self, kind: int, depth: int = 0):
        if kind in _SCALARS:
            fmt, size = _SCALARS[kind]
            return self.scalar(fmt, size)
        if kind == _STRING:
            return self.string()
        if kind == _ARRAY:
            if depth:  # GGUF has no nested arrays; a claimed one is corruption
                raise BadRequest(f"{self.name} declares a nested array")
            inner = self.scalar("<I", 4)
            count = self.scalar("<Q", 8)
            if count > MAX_HEADER_BYTES:
                raise BadRequest(f"{self.name} declares an implausible array length")
            return [self.value(inner, depth + 1) for _ in range(count)]
        raise BadRequest(
            f"{self.name} uses metadata value type {kind}, which "
            "this reader does not know"
        )


def read(path: str | Path, *, max_array: int = 64) -> Gguf:
    """Parse a GGUF header. Reads bytes; never loads or executes anything.

    `max_array` truncates long metadata arrays — a tokeniser vocabulary is
    frequently 128,000 strings, and nobody wants that in a JSON response. The
    truncation is reported in the value rather than silent.
    """
    p = Path(path)
    if not p.is_file():
        raise Refusal(f"{p} is not a file.")

    with open(p, "rb") as fh:
        head = fh.read(4)
        if head != MAGIC:
            raise BadRequest(
                f"{p.name} does not start with the GGUF magic bytes, so it is "
                "not a GGUF file whatever its extension says."
            )
        fixed = fh.read(20)
        if len(fixed) < 20:
            raise BadRequest(f"{p.name} is too short to be a GGUF file.")
        version, tensor_count, kv_count = struct.unpack("<IQQ", fixed)
        if version not in (1, 2, 3):
            raise Refusal(
                f"{p.name} is GGUF version {version}, and this reader knows "
                "versions 1 to 3. Refusing rather than guessing at a layout "
                "that may have changed."
            )
        if tensor_count > MAX_TENSORS or kv_count > MAX_KV:
            raise BadRequest(
                f"{p.name} declares {tensor_count} tensors and {kv_count} "
                "metadata entries, which is not a file this will parse."
            )
        # The header is everything before the tensor data. We do not know its
        # length up front, so read a bounded window — generous for a metadata
        # block, tiny against the model itself.
        blob = fh.read(MAX_HEADER_BYTES)

    cur = _Cursor(blob, p.name)
    metadata: dict = {}
    for _ in range(kv_count):
        key = cur.string()
        kind = cur.scalar("<I", 4)
        value = cur.value(kind)
        if isinstance(value, list) and len(value) > max_array:
            kept = value[:max_array]
            value = {
                "truncated": True,
                "length": len(value),
                "shown": kept,
                "note": f"{len(value) - max_array} more not shown",
            }
        metadata[key] = value

    tensors: list[Tensor] = []
    unknown: set[int] = set()
    for _ in range(tensor_count):
        name = cur.string()
        n_dims = cur.scalar("<I", 4)
        if n_dims > 8:
            raise BadRequest(f"{p.name} declares a {n_dims}-dimensional tensor")
        dims = [cur.scalar("<Q", 8) for _ in range(n_dims)]
        ggml_type = cur.scalar("<I", 4)
        offset = cur.scalar("<Q", 8)

        elements = 1
        for d in dims:
            elements *= int(d)

        spec = GGML_TYPES.get(ggml_type)
        if spec is None:
            unknown.add(ggml_type)
            type_name, nbytes = f"ggml type {ggml_type} (unknown)", None
        else:
            type_name, block, per_block = spec
            # Whole blocks. `elements * bpw` would be wrong for every k-quant.
            nbytes = (elements // block) * per_block if block else 0
            if block and elements % block:
                nbytes += per_block

        tensors.append(
            Tensor(
                name=name,
                ggml_type=ggml_type,
                type_name=type_name,
                dims=[int(d) for d in dims],
                elements=elements,
                bytes=nbytes,
                offset=int(offset),
            )
        )

    # The cursor stops at the end of the tensor-info table, and the blob
    # starts at the next `alignment` boundary.
    alignment = int(metadata.get("general.alignment", 32) or 32) or 32
    blob_start = cur.at + ((-cur.at) % alignment)

    return Gguf(
        path=str(p),
        version=int(version),
        tensor_count=int(tensor_count),
        metadata=metadata,
        tensors=tensors,
        unknown_types=sorted(unknown),
        data_offset=int(blob_start),
    )
