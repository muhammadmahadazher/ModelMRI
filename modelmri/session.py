"""`.mri` — an analysis you can send to someone who has no GPU.

Everything ModelMRI shows you is currently ephemeral. You find the head that
moves the subject token, and the only way to show anyone is a screenshot,
which they cannot explore. Reproducing it themselves means downloading the
model, matching your prompt, and finding the same head.

A `.mri` file is the observation, not the model: the tokens, the attention
that was actually captured, the generation, and the run's settings. It opens
in any ModelMRI with nothing loaded, and every panel reads it exactly as it
reads a live model — because the runtime serves it through the same methods.

**Size is the design constraint.** A 24-layer, 14-head, 141-token attention
tensor is 6.7 million numbers; as JSON at four decimals that is tens of
megabytes for something meant to be attached to a message. Two decisions fix
it:

  * uint8 with a per-matrix scale. Attention rows sum to 1 and are dominated
    by near-zero entries, so a linear quantisation against each matrix's own
    maximum keeps the visible structure — the arcs are drawn from relative
    weight — while costing one byte per value instead of eight.
  * gzip. Attention is highly structured (the sink column, the causal
    triangle of zeros) and compresses hard.

The quantisation is lossy and the file says so in `precision`, because a
number that has silently lost precision is exactly the kind of thing this
project refuses to ship.
"""

from __future__ import annotations

import base64
import gzip
import json
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

FORMAT = "modelmri-session"
FORMAT_VERSION = 1

# Bounds on untrusted input. A `.mri` is meant to be forwarded, so `parse`
# takes bytes a stranger sent — and every one of these was reachable.
#
# MAX_FILE is on the compressed bytes; MAX_INFLATED stops a gzip bomb, which
# the server's 64 MB body cap did not (64 MB of zeros inflates to ~69 GB).
# MAX_CELLS bounds n^2, because the expensive thing is per-slice cells rather
# than tokens. MAX_DIM keeps layer/head counts to something that is a shape.
MAX_FILE = 256 * 1024 * 1024
MAX_INFLATED = 512 * 1024 * 1024
MAX_CELLS = 24_000_000
MAX_DIM = 4096


def _inflate(data: bytes) -> bytes:
    """gunzip, refusing to keep going past MAX_INFLATED.

    `gzip.decompress` has no bound at all: it allocates whatever the stream
    tells it to. Decompressing incrementally and checking `eof` is the only
    way to tell "the file ended" from "the file is still going and we have
    stopped listening".
    """
    engine = zlib.decompressobj(31)  # 31 = gzip wrapper
    raw = engine.decompress(data, MAX_INFLATED)
    if not engine.eof:
        raise SessionError(
            f"this file expands to more than {MAX_INFLATED // 1024 // 1024} MB. "
            f"A session holds an observation, not a model — that is not one."
        )
    return raw


class SessionError(ValueError):
    """The file is not a session we can open, and we say why."""


def _quantise(matrix: Any) -> tuple[str, float]:
    """[S,S] floats -> (base64 uint8, scale). value ~= byte * scale.

    Takes a torch tensor or a list of lists. The tensor path is not an
    optimisation for its own sake: a 141-token, 24x14 export is 6.7 million
    values, and quantising those one at a time in Python takes long enough
    that a user would assume the button was broken.
    """
    if hasattr(matrix, "clamp") and hasattr(matrix, "contiguous"):
        import torch

        # float64, not float32: the two paths must agree bit for bit. A value
        # of exactly 0.1 lands on 26.0 in double and 25.999998 in single, and
        # truncation turns that into two different bytes -- so the "fast" path
        # would quietly export a different matrix than the portable one.
        m = matrix.detach().to(torch.float64)
        # NaN loses every comparison, so `max()` returns nan, the scale
        # becomes nan, and every cell quantises to 0: a smooth, plausible,
        # entirely blank heat map with nothing on screen saying the numbers
        # were never there. Refuse instead.
        if m.numel() and not bool(torch.isfinite(m).all()):
            raise SessionError(
                "this attention map contains non-finite values (nan or inf), "
                "so there is nothing honest to export. That usually means the "
                "model produced nan during the forward pass — the custom-model "
                "panel reports which layer first goes non-finite."
            )
        peak = float(m.max()) if m.numel() else 0.0
        scale = (peak / 255.0) if peak > 0 else 1.0
        q = (m / scale + 0.5).clamp(0, 255).to(torch.uint8).contiguous()
        return base64.b64encode(q.numpy().tobytes()).decode("ascii"), scale

    peak = 0.0
    for row in matrix:
        for v in row:
            if v != v or v in (float("inf"), float("-inf")):
                raise SessionError(
                    "this attention map contains non-finite values (nan or "
                    "inf), so there is nothing honest to export."
                )
            if v > peak:
                peak = v
    scale = (peak / 255.0) if peak > 0 else 1.0
    flat = bytearray()
    for row in matrix:
        for v in row:
            q = int(v / scale + 0.5) if scale else 0
            flat.append(255 if q > 255 else (0 if q < 0 else q))
    return base64.b64encode(bytes(flat)).decode("ascii"), scale


def _dequantise(blob: str, scale: float, n: int) -> list[list[float]]:
    raw = base64.b64decode(blob)
    if len(raw) != n * n:
        raise SessionError(
            f"attention block is {len(raw)} bytes but the token count says "
            f"{n}x{n}={n * n} — the file is truncated or not a session"
        )
    return [[round(raw[r * n + c] * scale, 5) for c in range(n)] for r in range(n)]


@dataclass
class Session:
    """A recorded analysis. Read-only by construction."""

    meta: dict = field(default_factory=dict)
    tokens: list[str] = field(default_factory=list)
    generation: str = ""
    prompt: str = ""
    # (layer, head) -> {"q": base64, "scale": float}
    attention: dict[str, dict] = field(default_factory=dict)
    lens: list[dict] = field(default_factory=list)
    n_layers: int = 0
    n_heads: int = 0

    # -------------------------------------------------- the runtime's shape
    def attention_meta(self) -> dict:
        return {
            "available": bool(self.attention),
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_tokens": len(self.tokens),
            "replay": True,
        }

    def attention_slice(self, layer: int, head: int) -> dict:
        key = f"{layer}:{head}"
        block = self.attention.get(key)
        if block is None:
            have = sorted(self.attention)[:6]
            raise SessionError(
                f"this session does not contain layer {layer} head {head}. "
                f"It has {len(self.attention)} slices, e.g. {', '.join(have)}. "
                "A session stores what was captured, not every combination."
            )
        return {
            "layer": layer,
            "head": head,
            "tokens": self.tokens,
            "matrix": _dequantise(block["q"], block["scale"], len(self.tokens)),
            "replay": True,
        }


def build(
    *,
    model_id: str | None,
    device: str | None,
    dtype: str | None,
    n_params: int | None,
    tokens: list[str],
    prompt: str,
    generation: str,
    attention: dict[tuple[int, int], list[list[float]]],
    n_layers: int,
    n_heads: int,
    lens: list[dict] | None = None,
    note: str = "",
    scope: str = "",
) -> bytes:
    """Serialise one analysis into a gzipped `.mri`."""
    from . import __version__

    blocks: dict[str, dict] = {}
    for (layer, head), matrix in attention.items():
        q, scale = _quantise(matrix)
        blocks[f"{layer}:{head}"] = {"q": q, "scale": scale}

    doc = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "modelmri": __version__,
        "meta": {
            "model": model_id,
            "device": device,
            "dtype": dtype,
            "n_params": n_params,
            "note": note,
            # What was captured, spelled out. A session that quietly holds a
            # slice of the cube looks identical to one that holds all of it
            # until you click the head it does not have.
            "scope": scope,
            # Stated, not implied. The arcs are drawn from relative weight, so
            # this is lossless for what you see and lossy for what you'd
            # compute — anyone doing arithmetic on these numbers should know.
            "precision": "attention quantised to uint8 against each matrix's "
            "own maximum; about 0.4% of that maximum per step",
        },
        "prompt": prompt,
        "generation": generation,
        "tokens": tokens,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "attention": blocks,
        "lens": lens or [],
    }
    return gzip.compress(json.dumps(doc, separators=(",", ":")).encode("utf-8"), 6)


def parse(data: bytes) -> Session:
    """Read a `.mri`, refusing anything that is not one, with the reason.

    Every bound below exists because this function takes bytes a stranger
    sent you. The whole premise of the format is that it travels.
    """
    if not data:
        raise SessionError("the file is empty")
    if len(data) > MAX_FILE:
        raise SessionError(
            f"this file is {len(data) / 1e6:,.0f} MB. A session is the "
            f"observation, not the model — a large one is tens of megabytes, "
            f"so this is almost certainly not one."
        )
    try:
        raw = _inflate(data) if data[:2] == b"\x1f\x8b" else data
    except (OSError, EOFError, zlib.error) as err:
        raise SessionError(f"could not decompress the file: {err}") from err

    try:
        doc: Any = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        # Deliberately not the exception class name. "JSONDecodeError" told a
        # user nothing except that something internal had gone wrong, which is
        # the wrong impression: the file is fine, it is just not one of ours.
        raise SessionError(
            "this file is not a ModelMRI session — a .mri is written by "
            "'Share this view' in the attention panel"
        ) from err

    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        raise SessionError(
            "this is not a ModelMRI session file (no 'modelmri-session' marker)"
        )
    version = doc.get("format_version")
    if not isinstance(version, int) or version > FORMAT_VERSION:
        raise SessionError(
            f"this session is format version {version}, and this ModelMRI "
            f"reads up to {FORMAT_VERSION}. Upgrade with `pip install -U modelmri`."
        )

    tokens = doc.get("tokens") or []
    if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
        raise SessionError("the session's token list is missing or malformed")

    # Cells, not tokens. Cost is n^2 per slice, so the token count is the
    # wrong thing to bound: a 31 KB file claiming 10,000 tokens asks for a
    # hundred million Python floats the moment a layer/head dial is clicked,
    # and the identical loop runs in the browser viewer for whoever you
    # forwarded it to.
    size = len(tokens)
    if size * size > MAX_CELLS:
        raise SessionError(
            f"this session claims {size:,} tokens, which is {size * size / 1e6:,.0f} "
            f"million attention cells per map — more than ModelMRI will render. "
            f"The file is either damaged or not a session."
        )

    # Validated before anything is built from it. `attention` is indexed by
    # string keys and iterated by the panels; a list or a dict with non-string
    # keys got past this and turned every later request into a 500.
    attention = doc.get("attention")
    if attention is None:
        attention = {}
    # `or {}` here turned a malformed value into an empty one: a file whose
    # attention was `[]` opened as a session with no maps rather than being
    # refused, which is a damaged file presented as an intact empty one.
    if not isinstance(attention, dict) or not all(
        isinstance(k, str) and isinstance(v, dict) for k, v in attention.items()
    ):
        raise SessionError("the session's attention index is missing or malformed")

    # These reach the UI as loop bounds. A float, a negative, or 1e20 is not
    # a shape — it is a hang or a crash in whatever renders it.
    counts = {}
    for key in ("n_layers", "n_heads"):
        value = doc.get(key) or 0
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= MAX_DIM
        ):
            raise SessionError(f"the session's {key} is not a sensible number")
        counts[key] = value

    return Session(
        meta={
            **(doc.get("meta") or {}),
            "created_at": doc.get("created_at"),
            "modelmri": doc.get("modelmri"),
        },
        tokens=tokens,
        prompt=doc.get("prompt") or "",
        generation=doc.get("generation") or "",
        attention=attention,
        lens=doc.get("lens") or [],
        n_layers=counts["n_layers"],
        n_heads=counts["n_heads"],
    )
