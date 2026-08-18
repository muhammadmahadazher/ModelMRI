"""Will this model fit, and what is the longest context it can hold?

`capacity.guard` already answers "will this DOWNLOAD fit", from the repo's
published byte count. This answers the harder question people actually ask —
"why won't my 8 GB card run this" — and it answers it with arithmetic the
reader can check by hand rather than a verdict they have to trust.

Three terms, all shown:

  weights      read from the safetensors header, exactly. Not the file size,
               which includes the header and any padding, and not a parameter
               count multiplied by an assumed dtype.
  KV cache     2 x n_layers x n_kv_heads x head_dim x seq_len x dtype_bytes.
               The 2 is keys and values. Every factor comes from the config or
               the tensor shapes; none is assumed.
  attention    what ModelMRI ITSELF costs. This tool forces eager attention
               and `output_attentions=True` so it has something to show you,
               which materialises n_layers x n_heads x S x S scores. At S=4096
               on a 12-layer, 12-head model that is 4.6 GB — larger than the
               weights of most models this runs on, and invisible in every
               other fit calculator because no other tool asks for it.

**The weights term is exact about the checkpoint, and an upper bound on the
card.** It prices every tensor the header declares. Some checkpoints declare
tensors a modern loader never materialises — an older architecture can ship a
`[1, 1, N, N]` causal-mask buffer per layer, and transformers stopped loading
those. Measured on such a checkpoint (F32 on disk, bf16 on an RTX 4060), the
header declares more elements than the model has parameters by exactly the
size of those masks, so the term reads a few percent above what the allocator
reports.

Left alone, deliberately. Skipping tensors by name would mean a hardcoded list
of buffer names per architecture, which is the kind of special case that is
right until the day it is silently wrong; and erring high is the safe
direction for a number whose job is to answer "will this fit". The total is
what matters and it lands close: 283.5 MB predicted against a 287.4 MB peak,
1.014x.

**What is NOT predicted.** Activations, workspace, allocator fragmentation,
and the CUDA context itself. Those are named in `excluded` rather than folded
in with a fudge factor, because a fudge factor is a number nobody can check.
This is why `grade()` exists: after a real load, it puts the measured
allocation next to the prediction and calls the difference what it is —
unpredicted runtime overhead — instead of quietly tuning the formula until the
two agree.

**Architectures this formula is wrong for are refused by name.** The KV
expression above is exact for standard MHA and GQA. It is wrong for MLA
(DeepSeek compresses KV through a low-rank projection and the cache is
`kv_lora_rank` wide, not `n_kv_heads x head_dim`), wrong for sliding-window
attention (the cache is capped at the window, so the honest answer stops
growing with seq_len), and meaningless for hybrid SSM stacks (Mamba-style
layers carry a fixed-size state and no KV at all). Each is detected from the
config and refused with the reason. An approximation here would be confidently
wrong in the direction that says "it fits" — see `budget.py` for why this
package treats that direction as the dangerous one.

**head_dim is read, not divided.** `ablate.head_geometry` documents that
`hidden_size // n_heads` is wrong by 2x on Qwen3-0.6B and wrong on
gemma-3-270m-it. It has a loaded model to read the projection width from; this
module runs BEFORE any load, so it reads the same width out of the safetensors
header — `k_proj.weight` is `[n_kv_heads * head_dim, hidden_size]` — and only
falls back to the quotient when there is no header to read, saying so.
"""

from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .errors import BadRequest, Refusal

# safetensors dtype -> bytes per element. Anything absent is refused rather
# than guessed: a dtype this table has not seen is one whose width we do not
# know, and assuming 2 would silently halve or double the answer.
DTYPE_BYTES = {
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "U16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}

# The header is a length prefix plus JSON. A malformed file could claim a
# gigabyte of it; this is a sanity bound, not a format limit.
MAX_HEADER_BYTES = 100_000_000


class UnsupportedArchitecture(Refusal):
    """The KV formula does not describe this model, and we will not pretend."""


@dataclass
class Term:
    """One factor in the arithmetic, so the reader can redo it."""

    name: str
    bytes: int
    formula: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Fit:
    seq_len: int
    dtype_bytes: int
    weights_bytes: int
    kv_bytes: int
    attention_bytes: int
    total_bytes: int
    budget_bytes: int | None
    fits: bool | None
    terms: list[Term] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["terms"] = [t.to_dict() for t in self.terms]
        return d


# ------------------------------------------------------------- safetensors


def read_header(path: Path) -> dict:
    """The tensor table from a .safetensors file. No torch, no mmap of weights.

    Format: 8 bytes of little-endian uint64 giving the JSON header's length,
    then that many bytes of JSON mapping tensor name -> {dtype, shape,
    data_offsets}. We read the header and stop — the weights are never touched,
    which is what makes this cheap enough to run before a download finishes.
    """
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            raise BadRequest(f"{path.name} is too short to be a safetensors file")
        (length,) = struct.unpack("<Q", raw)
        if not 0 < length <= MAX_HEADER_BYTES:
            raise BadRequest(
                f"{path.name} claims a {length}-byte header, which is not a "
                "safetensors file this can read"
            )
        blob = fh.read(length)
    if len(blob) < length:
        raise BadRequest(f"{path.name} ends inside its own header")
    try:
        header = json.loads(blob)
    except json.JSONDecodeError as err:
        raise BadRequest(f"{path.name}'s header is not valid JSON") from err
    header.pop("__metadata__", None)
    return header


# Dtypes a `dtype=` load converts. An integer or boolean tensor is not touched
# by `from_pretrained(dtype=bfloat16)`, so its width on the card is its width on
# disk — folding those into the conversion would understate a quantised
# checkpoint's real footprint.
FLOATING = {"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"}


@dataclass
class Weights:
    """What the checkpoint holds, on disk and after a dtype conversion.

    These are two different numbers and conflating them is what the first
    version of this module did. Measured (F32 on disk, loaded bf16 on
    an RTX 4060), the file payload was twice what was actually allocated,
    because every float was halved on the way in. A calculator that quotes the
    disk figure as "what this needs on your card" is wrong by 2x on the most
    common case there is — a float32 checkpoint on a GPU that prefers bf16.
    """

    disk_bytes: int
    card_bytes: int
    elements: int
    by_dtype: dict[str, int]
    source: str
    converted: bool

    def to_dict(self) -> dict:
        return asdict(self)


def weights_bytes(model_dir: Path, *, dtype_bytes: int | None = None) -> Weights:
    """What the tensor table says, on disk and at the dtype we would load.

    Sums `data_offsets` spans across every shard for the disk figure — the real
    payload, not `st.stat().st_size`, which also counts the header and any
    alignment padding. The card figure re-prices the floating tensors at
    `dtype_bytes` and leaves integer and boolean tensors at their own width.

    `dtype_bytes=None` means "loaded as stored", and the two figures agree.
    """
    shards = sorted(Path(model_dir).glob("*.safetensors"))
    if not shards:
        raise Refusal(
            f"no .safetensors file under {model_dir}, so there is no tensor "
            "table to read the exact weight size from. This calculator does "
            "not estimate from file sizes."
        )

    disk = card = elements = 0
    by_dtype: dict[str, int] = {}
    for shard in shards:
        for name, spec in read_header(shard).items():
            try:
                start, end = spec["data_offsets"]
                dtype = spec["dtype"]
                shape = spec["shape"]
            except (KeyError, TypeError, ValueError) as err:
                raise BadRequest(
                    f"{shard.name} describes tensor {name!r} in a shape this "
                    "reader does not recognise"
                ) from err
            if dtype not in DTYPE_BYTES:
                raise Refusal(
                    f"{shard.name} holds a {dtype} tensor, a dtype this "
                    "calculator does not know the width of. Refusing rather "
                    "than assuming one."
                )
            span = int(end) - int(start)
            count = 1
            for dim in shape:
                count *= int(dim)

            disk += span
            elements += count
            width = (
                dtype_bytes
                if dtype_bytes is not None and dtype in FLOATING
                else DTYPE_BYTES[dtype]
            )
            card += count * width
            by_dtype[dtype] = by_dtype.get(dtype, 0) + span

    where = f"{len(shards)} safetensors shard{'s' if len(shards) > 1 else ''}"
    return Weights(
        disk_bytes=disk,
        card_bytes=card,
        elements=elements,
        by_dtype=by_dtype,
        source=where,
        converted=dtype_bytes is not None and card != disk,
    )


# ------------------------------------------------------------- KV geometry


@dataclass
class KVGeometry:
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    head_dim_source: str

    def to_dict(self) -> dict:
        return asdict(self)


def _reject_unsupported(config: dict) -> None:
    """Refuse architectures the KV formula does not describe. By name."""
    model_type = str(config.get("model_type", "")).lower()

    if config.get("kv_lora_rank") is not None:
        raise UnsupportedArchitecture(
            f"{model_type or 'this model'} uses multi-head latent attention: "
            f"its cache is a {config['kv_lora_rank']}-wide compressed vector "
            "per token, not n_kv_heads x head_dim. The formula this "
            "calculator shows would overstate the cache several times over, "
            "so it is not applied here."
        )

    window = config.get("sliding_window")
    # `is not False` and not a truthiness test: absent means the window is in
    # force (most configs never write the flag), while an explicit False means
    # the architecture has one and has turned it off.
    if window and config.get("use_sliding_window") is not False:
        raise UnsupportedArchitecture(
            f"{model_type or 'this model'} uses sliding-window attention with "
            f"a window of {window} tokens, so its KV cache stops growing once "
            "the context passes the window. A number that keeps growing with "
            "sequence length would be wrong in the direction that says it "
            "does not fit."
        )

    ssm = ("mamba", "jamba", "zamba", "bamba", "falcon_h1", "recurrentgemma", "rwkv")
    if any(tag in model_type for tag in ssm) or config.get("ssm_cfg") is not None:
        raise UnsupportedArchitecture(
            f"{model_type or 'this model'} is a hybrid state-space stack. Its "
            "recurrent layers carry a fixed-size state and no KV cache at all, "
            "so a per-layer KV figure would describe layers that do not exist."
        )


def _merged(config: dict) -> dict:
    """One config with `text_config` folded in, so every guard sees it.

    Multimodal configs (gemma-3, qwen2-vl, llava) nest the language model under
    `text_config`. `need()` already fell through to it for layer and head
    counts, but `_reject_unsupported`, `num_key_value_heads` and `head_dim` all
    read the top level only — so a nested gemma-3 was NOT refused for its
    sliding window, and its KV was overstated 5.8x at 32k and 6.5x at 128k,
    while `longest_context` returned ~3.4k tokens for a model that holds 131k.
    The identical FLAT config was correctly refused, which is the tell that the
    guard was reading the wrong dict rather than the wrong rule.

    Nested keys win: they describe the language model, which is what the KV
    cache belongs to.
    """
    nested = config.get("text_config")
    if not isinstance(nested, dict):
        return config
    merged = dict(config)
    merged.update(nested)
    # `model_type` too — it is what `_reject_unsupported` matches on, and the
    # outer one names the wrapper ("gemma3") rather than the decoder.
    if nested.get("model_type"):
        merged["model_type"] = nested["model_type"]
    return merged


def kv_geometry(config: dict, header: dict | None = None) -> KVGeometry:
    """The four numbers the KV formula needs, each read rather than assumed.

    A missing key is named and refused. That is the whole discipline here: a
    default for `num_key_value_heads` would turn a GQA model into an MHA one
    and multiply the predicted cache by the group size, silently.
    """
    config = _merged(config)
    _reject_unsupported(config)

    def need(key: str, *alts: str) -> int:
        for name in (key, *alts):
            value = config.get(name)
            if value is not None:
                return int(value)
            raw = config.get("text_config", {})
            if isinstance(raw, dict) and raw.get(name) is not None:
                return int(raw[name])
        raise BadRequest(
            f"this model's config.json has no {key!r}, so the KV cache size "
            "cannot be computed from it. Refusing rather than substituting a "
            "default."
        )

    n_layers = need("num_hidden_layers", "n_layer")
    n_heads = need("num_attention_heads", "n_head")
    # GQA: absent means every head has its own KV, which IS the definition of
    # MHA rather than a guess. Stated in `head_dim_source` either way.
    n_kv_heads = int(config.get("num_key_value_heads") or n_heads)

    head_dim, source = _head_dim(config, header, n_heads, n_kv_heads)
    return KVGeometry(n_layers, n_heads, n_kv_heads, head_dim, source)


def _head_dim(
    config: dict, header: dict | None, n_heads: int, n_kv_heads: int
) -> tuple[int, str]:
    """head_dim, preferring what is written down over what can be divided.

    Order: the config's own `head_dim`; then the k_proj row count from the
    safetensors header divided by n_kv_heads, which is the same width
    `ablate.head_geometry` reads off a loaded model; then the quotient, which
    is measurably wrong on Qwen3 and Gemma-3 and is labelled as a fallback.
    """
    stated = config.get("head_dim")
    if stated:
        return int(stated), "config.json head_dim"

    if header:
        for name, spec in header.items():
            if not name.endswith("k_proj.weight"):
                continue
            shape = spec.get("shape") or []
            if len(shape) == 2 and n_kv_heads:
                rows = int(shape[0])
                if rows % n_kv_heads == 0:
                    return rows // n_kv_heads, f"{name} rows / num_key_value_heads"
            break

    hidden = config.get("hidden_size") or config.get("n_embd")
    if not hidden:
        raise BadRequest(
            "this model's config.json states neither `head_dim` nor "
            "`hidden_size`, so head_dim cannot be determined."
        )
    return int(hidden) // n_heads, "hidden_size / num_attention_heads (a fallback)"


# --------------------------------------------------------------- the totals


def kv_cache_bytes(geo: KVGeometry, seq_len: int, dtype_bytes: int) -> int:
    """2 x layers x kv_heads x head_dim x seq_len x dtype_bytes. The 2 is K and V."""
    return 2 * geo.n_layers * geo.n_kv_heads * geo.head_dim * seq_len * dtype_bytes


def attention_bytes(geo: KVGeometry, seq_len: int, dtype_bytes: int) -> int:
    """What ModelMRI's own eager attention materialises: layers x heads x S x S.

    Quadratic in sequence length, and the term that surprises people. It is
    charged to this tool rather than to the model, because a runner that never
    shows you attention never pays it.
    """
    return geo.n_layers * geo.n_heads * seq_len * seq_len * dtype_bytes


def plan(
    model_dir: str | Path,
    *,
    seq_len: int,
    dtype_bytes: int = 2,
    budget_bytes: int | None = None,
) -> Fit:
    """The whole calculation for one sequence length, every term visible."""
    if seq_len < 1:
        raise BadRequest("seq_len must be at least 1")

    directory = Path(model_dir)
    config_path = directory / "config.json"
    if not config_path.exists():
        raise Refusal(
            f"no config.json under {directory}, so the KV geometry cannot be "
            "read. This calculator does not guess an architecture."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))

    shards = sorted(directory.glob("*.safetensors"))
    header = read_header(shards[0]) if shards else None

    geo = kv_geometry(config, header)
    w = weights_bytes(directory, dtype_bytes=dtype_bytes)
    weights = w.card_bytes
    kv = kv_cache_bytes(geo, seq_len, dtype_bytes)
    attn = attention_bytes(geo, seq_len, dtype_bytes)
    total = weights + kv + attn

    weights_formula = (
        f"{w.elements:,} elements re-priced at {dtype_bytes} bytes "
        f"({w.disk_bytes / 1e6:,.1f} MB on disk as {'/'.join(sorted(w.by_dtype))})"
        if w.converted
        else f"exact, summed from {w.source}"
    )
    terms = [
        Term("weights", weights, weights_formula),
        Term(
            "kv_cache",
            kv,
            f"2 x {geo.n_layers} layers x {geo.n_kv_heads} kv_heads x "
            f"{geo.head_dim} head_dim x {seq_len} tokens x {dtype_bytes} bytes",
        ),
        Term(
            "eager_attention",
            attn,
            f"{geo.n_layers} layers x {geo.n_heads} heads x {seq_len}^2 x "
            f"{dtype_bytes} bytes — what ModelMRI itself adds",
        ),
    ]

    return Fit(
        seq_len=seq_len,
        dtype_bytes=dtype_bytes,
        weights_bytes=weights,
        kv_bytes=kv,
        attention_bytes=attn,
        total_bytes=total,
        budget_bytes=budget_bytes,
        fits=None if budget_bytes is None else total <= budget_bytes,
        terms=terms,
        excluded=[
            "activations and workspace",
            "allocator fragmentation",
            "the CUDA/driver context",
        ],
        notes=[
            f"head_dim from {geo.head_dim_source}",
            f"weight dtypes on disk: {', '.join(sorted(w.by_dtype))}",
        ]
        + (
            [
                f"this checkpoint is stored as {'/'.join(sorted(w.by_dtype))} and "
                f"would load at {dtype_bytes} bytes per float, so it needs "
                f"{weights / 1e6:,.0f} MB on the accelerator rather than the "
                f"{w.disk_bytes / 1e6:,.0f} MB it occupies on disk"
            ]
            if w.converted
            else []
        ),
    )


def longest_context(
    model_dir: str | Path,
    *,
    budget_bytes: int,
    dtype_bytes: int = 2,
    ceiling: int = 1 << 20,
) -> int:
    """The largest seq_len whose predicted total still fits the budget.

    Binary search over `plan`, which is exact arithmetic rather than a
    measurement, so this is cheap. Returns 0 when even one token does not fit —
    never a negative or a fabricated minimum.
    """
    if plan(model_dir, seq_len=1, dtype_bytes=dtype_bytes).total_bytes > budget_bytes:
        return 0

    low, high = 1, ceiling
    while low < high:
        mid = (low + high + 1) // 2
        total = plan(model_dir, seq_len=mid, dtype_bytes=dtype_bytes).total_bytes
        if total <= budget_bytes:
            low = mid
        else:
            high = mid - 1
    return low


def grade(prediction: Fit, measured_bytes: int) -> dict:
    """Put the measured allocation beside the prediction and name the gap.

    The gap is not an error to be tuned away. `plan` deliberately excludes
    activations, workspace and fragmentation, so a positive gap is those
    things showing up — which is worth reporting as its own number rather than
    hidden inside a corrected total.
    """
    gap = measured_bytes - prediction.total_bytes
    ratio = (
        (measured_bytes / prediction.total_bytes) if prediction.total_bytes else None
    )
    return {
        "predicted_bytes": prediction.total_bytes,
        "measured_bytes": measured_bytes,
        "gap_bytes": gap,
        "ratio": round(ratio, 3) if ratio else None,
        "gap_is": (
            "unpredicted runtime overhead — activations, workspace and "
            "allocator fragmentation, which this calculator excludes on purpose"
            if gap >= 0
            else "less than predicted: the allocator had not yet grown to the "
            "full cache, or the run was shorter than the sequence length priced"
        ),
        "excluded": prediction.excluded,
    }
