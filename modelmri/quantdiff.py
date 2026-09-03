# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""What did quantising this model actually cost, tensor by tensor?

Everyone picks a quantisation by reputation. `Q4_K_M` is "the good one",
`Q3_K_S` is "too far", and the evidence is a perplexity number somebody posted
once. `llama-perplexity --kl-divergence-base` measures one number end to end
and wants an 11-37 GiB FP16 logit dump before it will say anything;
`llama-imatrix --show-statistics` frames activation statistics as calibration
rather than as damage. Nothing joins **which weights were damaged** to **how
much**.

This does the weight half: dequantise the GGUF back to floats, line each tensor
up against the same tensor in the full-precision original, and report four
things per tensor —

    rms          root mean square of (quantised - original)
    max_abs      the single worst weight
    cosine       direction agreement, which is scale-free
    sign_flips   the fraction of weights whose sign changed

`cosine` and `sign_flips` are there because `rms` alone is misleading in
opposite directions. A tensor of large weights can absorb a large RMS without
its direction moving; a tensor of tiny weights can have a small RMS and be
mostly noise. A sign flip is the crispest damage there is — the weight now
pushes the other way.

**One tensor resident at a time.** Both sides are streamed: the GGUF through
its own table, the original through the safetensors header's offsets. Peak
memory is the largest single tensor, not the model, so this runs on any machine
regardless of how big the pair is.

**An unmapped tensor is listed, never dropped.** GGUF names (`blk.0.attn_q`)
and HF names (`model.layers.0.self_attn.q_proj`) are different vocabularies,
and the mapping is not total. A tensor this cannot pair up appears in
`not_compared` with the reason. Silently omitting it would shrink the
denominator of every aggregate and make a worse quantisation look better.

**This is the quantiser's damage, not llama.cpp's end-to-end damage.** It
compares stored weights. What llama.cpp then does with them at inference — its
own kernels, its own accumulation order — is a different question, and this
number must not be read as answering it.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .errors import BadRequest, Refusal

# GGUF name -> HF name. transformers keeps a mapping for loading; this is the
# inverse direction and is kept here rather than reached for privately, because
# a mapping that silently misses is the failure this module reports rather than
# suffers.
_LAYER = re.compile(r"^blk\.(\d+)\.(.+)$")

_PER_LAYER = {
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
}

_GLOBAL = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}


def hf_name(gguf_name: str) -> str | None:
    """The HuggingFace name for a GGUF tensor, or None when unknown.

    None rather than a guess: pairing two tensors that are not the same tensor
    produces a damage number about nothing, and it would look exactly like a
    real one.
    """
    if gguf_name in _GLOBAL:
        return _GLOBAL[gguf_name]
    m = _LAYER.match(gguf_name)
    if not m:
        return None
    index, rest = m.group(1), m.group(2)
    tail = _PER_LAYER.get(rest)
    return f"model.layers.{index}.{tail}" if tail else None


@dataclass
class TensorDamage:
    name: str
    hf_name: str
    quant_type: str
    elements: int
    rms: float
    max_abs: float
    # Scale-free: two tensors can differ a lot in magnitude and still point the
    # same way, which is what a good quantiser preserves.
    cosine: float
    sign_flips: float
    # RMS relative to the original's own RMS, so tensors of different scales
    # are comparable to each other.
    relative_rms: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Report:
    quantised: str
    original: str
    tensors: list[TensorDamage] = field(default_factory=list)
    not_compared: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "quantised": self.quantised,
            "original": self.original,
            "tensors": [t.to_dict() for t in self.tensors],
            "not_compared": self.not_compared,
            "summary": self.summary(),
            "notes": self.notes,
        }

    def summary(self) -> dict:
        if not self.tensors:
            return {
                "compared": 0,
                "not_compared": len(self.not_compared),
                "means": (
                    "No tensor could be paired between the two files, so there "
                    "is nothing to report. Check that these are the same model."
                ),
            }
        worst_rel = max(self.tensors, key=lambda t: t.relative_rms)
        worst_cos = min(self.tensors, key=lambda t: t.cosine)
        worst_flip = max(self.tensors, key=lambda t: t.sign_flips)
        total = sum(t.elements for t in self.tensors)
        return {
            "compared": len(self.tensors),
            "not_compared": len(self.not_compared),
            "elements_compared": total,
            # Element-weighted, so one tiny norm tensor cannot dominate.
            "mean_relative_rms": round(
                sum(t.relative_rms * t.elements for t in self.tensors) / total, 5
            ),
            "mean_sign_flips": round(
                sum(t.sign_flips * t.elements for t in self.tensors) / total, 5
            ),
            "worst_relative_rms": {
                "name": worst_rel.name,
                "value": worst_rel.relative_rms,
            },
            "worst_cosine": {"name": worst_cos.name, "value": worst_cos.cosine},
            "worst_sign_flips": {
                "name": worst_flip.name,
                "value": worst_flip.sign_flips,
            },
            "means": (
                "Error between the stored weights of the two files, per tensor. "
                "This is the quantiser's damage, NOT what llama.cpp's kernels "
                "then do with those weights at inference — a different question "
                "this does not answer. Tensors that could not be paired are "
                "listed in not_compared and are excluded from every average "
                "above rather than counted as undamaged."
            ),
        }


def _require_gguf():
    try:
        from gguf.constants import GGMLQuantizationType
        from gguf.quants import dequantize
    except Exception as err:
        raise Refusal(
            "measuring quantisation damage needs the `gguf` package to "
            "dequantise the weights — install it with `pip install "
            "modelmri[gguf]`. The header reader in `modelmri.gguf_read` needs "
            "nothing and still works."
        ) from err
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import dequantize

    return dequantize, GGMLQuantizationType


def _read_gguf_tensor(path: Path, tensor, dequantize, QuantType):
    """One tensor's floats, read from the file and dequantised. Nothing else."""
    import numpy as np

    with open(path, "rb") as fh:
        fh.seek(tensor["data_offset"])
        raw = fh.read(tensor["bytes"])
    if len(raw) < tensor["bytes"]:
        raise BadRequest(
            f"{path.name} ends before tensor {tensor['name']!r} does — the file "
            "is truncated."
        )
    try:
        quant = QuantType(tensor["ggml_type"])
    except ValueError:
        return None
    try:
        values = dequantize(np.frombuffer(raw, dtype=np.uint8), quant)
    except Exception:
        # A type the installed gguf knows by name but cannot unpack. Reported
        # as not-compared rather than approximated.
        return None
    # GGUF stores dims fastest-first; the HF tensor is the transpose ordering.
    return values.astype(np.float32).reshape(tuple(reversed(tensor["dims"])))


def compare_tensor(quantised, original) -> dict:
    """The four numbers, for one pair of already-loaded arrays."""
    import numpy as np

    qa = np.asarray(quantised, dtype=np.float64)
    oa = np.asarray(original, dtype=np.float64)
    # BEFORE ravelling. Flattening first made (2,3) and (3,2) both (6,), so the
    # guard never fired and two genuinely different tensors with the same
    # element count were compared as though they were the same one — which is
    # precisely the mispairing this module refuses to do elsewhere.
    if qa.shape != oa.shape:
        raise BadRequest(
            f"shape mismatch: {tuple(qa.shape)} against {tuple(oa.shape)}. "
            "These are not the same tensor."
        )
    q = qa.ravel()
    o = oa.ravel()
    diff = q - o
    rms = float(np.sqrt(np.mean(diff * diff)))
    o_rms = float(np.sqrt(np.mean(o * o)))
    qn = float(np.linalg.norm(q))
    on = float(np.linalg.norm(o))
    return {
        "rms": rms,
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        # A zero-norm tensor has no direction, so cosine is undefined rather
        # than 0.0 — and 0.0 would read as "completely orthogonal", a strong
        # and false claim.
        "cosine": float(np.dot(q, o) / (qn * on)) if qn and on else None,
        "sign_flips": float(np.mean(np.sign(q) != np.sign(o))) if q.size else 0.0,
        # Relative to the original's own scale, so tensors of wildly different
        # magnitudes can be compared to one another.
        "relative_rms": (rms / o_rms) if o_rms else None,
        "elements": int(q.size),
    }


def compare(
    quantised_path: str | Path, original_dir: str | Path, *, limit: int = 0
) -> Report:
    """Damage report between a GGUF and the full-precision original.

    `original_dir` holds the HuggingFace checkpoint (safetensors). Streams both
    sides one tensor at a time, so peak memory is the largest single tensor.
    `limit` caps how many tensors are compared, 0 for all — and when it bites,
    the cap is recorded in `notes` rather than leaving a partial report looking
    complete.
    """
    import numpy as np
    from safetensors import safe_open

    from . import fit, gguf_read

    dequantize, QuantType = _require_gguf()

    qpath = Path(quantised_path)
    odir = Path(original_dir)
    header = gguf_read.read(qpath)

    shards = sorted(odir.glob("*.safetensors"))
    if not shards:
        raise Refusal(
            f"no .safetensors under {odir}, so there is no full-precision side "
            "to compare against."
        )
    # Which shard holds which tensor, from the headers alone.
    where: dict[str, Path] = {}
    for shard in shards:
        for name in fit.read_header(shard):
            where.setdefault(name, shard)

    # Where the tensor data starts: the GGUF header is followed by padding to
    # `general.alignment` (32 unless stated), then the tensor blob.
    align = int(header.metadata.get("general.alignment", 32) or 32)
    with open(qpath, "rb") as fh:
        fh.seek(0, 2)
        file_size = fh.tell()
    # The blob begins after the header; the last tensor's offset+size must fit,
    # which is what pins the base down without re-parsing.
    sized = [t for t in header.tensors if t.bytes is not None]
    if not sized:
        raise Refusal(
            f"{qpath.name} uses only ggml types this reader cannot size, so no "
            "tensor can be located in the file."
        )
    # READ FROM THE HEADER, not inferred from the file size. This used to do
    # `base = file_size - max(offset + size)`, which is only correct when the
    # file ends exactly at the last tensor -- and GGUF writers pad to
    # `general.alignment` after it. Trailing padding made the inferred base
    # run long by that many bytes, and `base += (-base) % align` then rounded
    # it UP again, so every tensor was read from past its true start. Not a
    # crash: dequantising misaligned bytes produces numbers, and the
    # quantisation-damage report would have been computed on them.
    base = int(getattr(header, "data_offset", 0) or 0)
    if not base:
        # A header parsed before `data_offset` existed. The old inference,
        # kept as a fallback and named as one.
        end = max(t.offset + (t.bytes or 0) for t in sized)
        base = file_size - end
        base += (-base) % align

    # Whatever the source, the blob has to actually fit.
    end = max(t.offset + (t.bytes or 0) for t in sized)
    if base + end > file_size:
        raise Refusal(
            f"{qpath.name} says its tensors end at byte {base + end:,} and the "
            f"file is {file_size:,} bytes. Refusing to read past the end "
            f"rather than returning numbers from whatever is there."
        )

    report = Report(quantised=str(qpath), original=str(odir))
    if limit:
        report.notes.append(f"compared the first {limit} tensors only")

    done = 0
    for t in header.tensors:
        if limit and done >= limit:
            break
        target = hf_name(t.name)
        if target is None:
            report.not_compared.append(
                {"name": t.name, "why": "no HuggingFace name is known for this tensor"}
            )
            continue
        if t.bytes is None:
            report.not_compared.append(
                {"name": t.name, "why": f"{t.type_name} — this reader cannot size it"}
            )
            continue
        shard = where.get(target)
        if shard is None:
            report.not_compared.append(
                {"name": t.name, "why": f"{target} is not in the original checkpoint"}
            )
            continue

        values = _read_gguf_tensor(
            qpath,
            {
                "name": t.name,
                "data_offset": base + t.offset,
                "bytes": t.bytes,
                "ggml_type": t.ggml_type,
                "dims": t.dims,
            },
            dequantize,
            QuantType,
        )
        if values is None:
            report.not_compared.append(
                {
                    "name": t.name,
                    "why": f"the installed gguf cannot dequantise {t.type_name}",
                }
            )
            continue

        with safe_open(str(shard), framework="numpy") as fh:
            original = fh.get_tensor(target)
        original = np.asarray(original, dtype=np.float32)
        if tuple(original.shape) != tuple(values.shape):
            report.not_compared.append(
                {
                    "name": t.name,
                    "why": (
                        f"shape {tuple(values.shape)} against {tuple(original.shape)} "
                        "in the original — not the same tensor"
                    ),
                }
            )
            continue

        stats = compare_tensor(values, original)
        if stats["cosine"] is None or stats["relative_rms"] is None:
            report.not_compared.append(
                {
                    "name": t.name,
                    "why": "the original tensor is all zeros; error is undefined",
                }
            )
            continue
        report.tensors.append(
            TensorDamage(
                name=t.name,
                hf_name=target,
                quant_type=t.type_name,
                elements=stats["elements"],
                rms=round(stats["rms"], 8),
                max_abs=round(stats["max_abs"], 8),
                cosine=round(stats["cosine"], 6),
                sign_flips=round(stats["sign_flips"], 6),
                relative_rms=round(stats["relative_rms"], 6),
            )
        )
        done += 1

    return report
