"""What a fine-tune actually changed — read off the adapter, not guessed.

People download a LoRA and get a name, a file size and a thumbnail. What they
do not get is WHICH PART OF THE MODEL IT TOUCHES: whether it moved the UNet or
only the text encoder, whether it reached the cross-attention that reads the
prompt or only the self-attention that does not, and how hard. That is the
question this answers, and it is answerable exactly because a LoRA carries its
own targets in its tensor names.

## Why this is a weight diff and not a behaviour diff

`model_diff.py` answers the same question for text models by RUNNING both
sides over a prompt set, because two language models that differ everywhere by
a little are indistinguishable from two that differ in one place by a lot
unless you look at what they say. That is the right tool there and the wrong
one here for a practical reason: a diffusion pair is two multi-gigabyte
pipelines resident at once, and the common case — an 80 MB LoRA — does not
need it. The adapter file already states its targets.

So this is deliberately the CHEAP half. It reads a file and reports structure
and magnitude. `image_steps.filmstrip` is where you go to see what a fine-tune
does to a picture; the two answer different questions and neither is a
substitute for the other.

## A norm is a MAGNITUDE, never an effect

`||ΔW||` says how far a weight matrix moved. It does not say the image
changed, and it cannot: a large move in a layer the sampler barely exercises
can do less than a small move in one it leans on. Every number here is
reported as what it is — the size of a delta — and the response says so rather
than letting a big number be read as a big effect.

RELATIVE magnitude is the number people actually want (`||ΔW|| / ||W||`), and
it needs the base weights. When the base model is not resident this reports
the absolute norm and says the relative one is unavailable, rather than
dividing by a stand-in.

## The scaling is real and is applied

A LoRA stores `A` and `B` with `ΔW = B @ A * (alpha / rank)`. Reporting
`||B @ A||` without the scale would over-state every adapter whose alpha is
below its rank, which is most of them. The scale is read from the adapter's
own config where it publishes one, from its `.alpha` tensors where it does
not, and where NEITHER exists the norm is reported unscaled and flagged —
because assuming `alpha == rank` is a guess that silently doubles some
adapters.
"""

from __future__ import annotations

import json
import logging
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

from .errors import BadRequest, Refusal

log = logging.getLogger(__name__)

#: A safetensors header is a length-prefixed JSON blob. Bounded before the
#: read, because the length comes from the file and a hostile one can claim to
#: be exabytes. Same constant and same reasoning as `imaging`.
MAX_HEADER_BYTES = 100 * 1024 * 1024

#: How many modules to report individually. The rest are summed into their
#: group and the count is REPORTED — an adapter touching 700 modules is a real
#: thing and a list truncated in silence would read as a small one.
TOP_MODULES = 40

#: The two halves of a LoRA, in the spellings the ecosystem actually ships.
#: diffusers, PEFT, kohya and ComfyUI all differ, and a reader that knows one
#: of them silently reports "no LoRA tensors" for the other three.
_DOWN = ("lora_A", "lora_down", "lora.down", "lora_a")
_UP = ("lora_B", "lora_up", "lora.up", "lora_b")

#: Where in a diffusion pipeline a tensor lives. Ordered: the first match
#: wins, so `text_encoder_2` is not read as `text_encoder`.
_COMPONENTS = (
    ("text_encoder_2", "text encoder 2"),
    ("text_encoder", "text encoder"),
    ("te2_", "text encoder 2"),
    ("te_", "text encoder"),
    ("unet", "UNet"),
    ("transformer", "transformer"),
    ("vae", "VAE"),
)

#: What the module DOES, which is the part that matters for a diffusion model:
#: cross-attention is where the prompt enters, and a LoRA that never touches it
#: cannot be changing how words are read.
_ROLES = (
    (re.compile(r"attn2|cross_attn|\.to_k\b.*attn2|encoder_attn"), "cross-attention"),
    (re.compile(r"attn1|self_attn"), "self-attention"),
    # `ff` does NOT work here and looked like it did. kohya spells a
    # module `..._transformer_blocks_9_ff_net_0_proj`, and `_` is a word
    # character, so there is no word boundary around `ff` — every
    # feed-forward module in latent-consistency/lcm-lora-sdxl fell through
    # to "other", which then reported as the LARGEST group in that adapter
    # (162 modules, and the top five deltas). The underscore spelling needs
    # a separator class, not a word boundary.
    (re.compile(r"(?:^|[._])ff(?:[._]|$)|feed_forward|mlp|proj_mlp"), "feed-forward"),
    (re.compile(r"conv|resnet"), "convolution"),
    (re.compile(r"attn|to_q|to_k|to_v|to_out"), "attention"),
)


class NotAnAdapter(Refusal):
    """The file is readable and is not a LoRA.

    Its own kind of refusal because the remedy differs from every other
    failure here: a full fine-tune, a merged checkpoint and a VAE are all
    legitimate files that this cannot decompose, and the answer is "point it
    at the adapter" rather than "the file is broken".
    """


@dataclass
class Module:
    """One targeted module, and how far the adapter moves it."""

    name: str
    component: str
    role: str
    rank: int | None
    #: `alpha / rank`. `None` when the adapter published neither, in which
    #: case `delta_norm` is UNSCALED and `scaled` says so.
    scale: float | None
    scaled: bool
    #: Frobenius norm of the delta this adapter applies. A MAGNITUDE.
    delta_norm: float
    #: `||ΔW|| / ||W||`, or None when the base weights were not available.
    #: Never approximated: the point of the ratio is the denominator.
    relative: float | None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "component": self.component,
            "role": self.role,
            "rank": self.rank,
            "scale": self.scale,
            "scaled": self.scaled,
            "delta_norm": self.delta_norm,
            "relative": self.relative,
        }


@dataclass
class Group:
    """Every module sharing a component and a role, summed."""

    component: str
    role: str
    modules: int
    delta_norm: float

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "role": self.role,
            "modules": self.modules,
            "delta_norm": self.delta_norm,
        }


@dataclass
class AdapterReport:
    path: str
    #: Ranks seen. A list, not a number: an adapter may mix ranks per module
    #: and reporting one would be picking which.
    ranks: list[int]
    modules_total: int
    modules_listed: int
    components: list[str]
    roles: list[str]
    groups: list[Group]
    top: list[Module]
    #: True when every reported norm carries its alpha/rank scale.
    all_scaled: bool
    base_model: str | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "ranks": self.ranks,
            "modules_total": self.modules_total,
            "modules_listed": self.modules_listed,
            "components": self.components,
            "roles": self.roles,
            "groups": [g.to_dict() for g in self.groups],
            "top": [m.to_dict() for m in self.top],
            "all_scaled": self.all_scaled,
            "base_model": self.base_model,
            "notes": self.notes,
            "means": self.means(),
        }

    def means(self) -> str:
        where = ", ".join(self.components) or "components this does not name"
        what = ", ".join(self.roles) or "no role it could classify"
        head = (
            f"{self.modules_total} module(s) changed across {where}, reaching {what}. "
        )
        if self.modules_listed < self.modules_total:
            head += (
                f"The {self.modules_listed} largest are listed individually and "
                f"the rest are in the group totals — none are omitted. "
            )
        head += (
            "Every number here is the SIZE of a weight delta, not its effect: "
            "a large move in a layer the sampler barely exercises can matter "
            "less than a small move in one it leans on. "
        )
        if not self.all_scaled:
            head += (
                "Some modules published neither an alpha nor a rank, so their "
                "norms are UNSCALED and not comparable with the rest. "
            )
        if all(m.relative is None for m in self.top):
            head += (
                "Relative size needs the base weights, which are not loaded, "
                "so these are absolute norms only."
            )
        return head.strip()


def _read_header(path: Path) -> dict:
    """The safetensors header, without reading the tensors.

    A few kilobytes of a file that may be gigabytes. Bounded before the read
    for the same reason `imaging.read_tensor_names` is.
    """
    try:
        with path.open("rb") as fh:
            raw = fh.read(8)
            if len(raw) < 8:
                raise NotAnAdapter(
                    f"{path.name} is too short to be a safetensors file — its "
                    f"header length field is incomplete."
                )
            (length,) = struct.unpack("<Q", raw)
            if not 0 < length <= MAX_HEADER_BYTES:
                raise NotAnAdapter(
                    f"{path.name} declares a {length}-byte header, which is "
                    f"outside anything this reads. The file is not a "
                    f"safetensors adapter, or it is damaged."
                )
            header = json.loads(fh.read(length).decode("utf-8"))
    except NotAnAdapter:
        raise
    except (OSError, ValueError, struct.error) as err:
        raise NotAnAdapter(
            f"{path.name} could not be read as safetensors "
            f"({type(err).__name__}). LoRA adapters ship as `.safetensors`; a "
            f"`.bin` or `.ckpt` is a pickle and is not opened here."
        ) from None
    if not isinstance(header, dict):
        raise NotAnAdapter(
            f"{path.name} has a safetensors header that is not an object."
        )
    return header


def _pair_key(name: str) -> tuple[str, str] | None:
    """Split a LoRA tensor name into (module, half), or None if it is neither.

    The four spellings are tried in order and the FIRST that matches wins.
    Substring order matters: `lora_down` contains `lora_d`, so the table is
    written longest-first rather than relying on the loop.
    """
    for token in _DOWN:
        if token in name:
            return name.replace(token, "\0"), "down"
    for token in _UP:
        if token in name:
            return name.replace(token, "\0"), "up"
    return None


def _classify(name: str) -> tuple[str, str]:
    component = next((label for tok, label in _COMPONENTS if tok in name), "unnamed")
    role = next((label for rx, label in _ROLES if rx.search(name)), "other")
    return component, role


def _alphas(header: dict) -> dict[str, float]:
    """Per-module alpha, from the `.alpha` tensors kohya-style adapters ship.

    Read from the header alone where the value is not there — an alpha tensor
    is a scalar and its value lives in the data segment, so this records WHICH
    modules publish one and the loader fills the values in.
    """
    return {
        k.rsplit(".", 1)[0]: 0.0
        for k in header
        if k.endswith(".alpha") or k.endswith("_alpha")
    }


def read(path: str | Path, *, base=None, top: int = TOP_MODULES) -> AdapterReport:
    """Decompose a LoRA adapter into what it targets and how hard.

    `base` is an optional loaded model. Given one, each module's delta is
    divided by the norm of the weight it applies to and `relative` is filled
    in; without one that field stays None rather than being approximated.
    """
    import torch
    from safetensors.torch import load_file

    p = Path(path)
    if p.is_dir():
        # PEFT writes `adapter_model.safetensors` beside `adapter_config.json`.
        found = next(
            (
                c
                for c in (
                    p / "adapter_model.safetensors",
                    p / "pytorch_lora_weights.safetensors",
                )
                if c.is_file()
            ),
            None,
        )
        if found is None:
            raise BadRequest(
                f"{p} is a directory with no adapter in it. Expected "
                f"`adapter_model.safetensors` or "
                f"`pytorch_lora_weights.safetensors`."
            )
        cfg_path, p = p / "adapter_config.json", found
    else:
        cfg_path = p.parent / "adapter_config.json"

    header = _read_header(p)
    names = [k for k in header if k != "__metadata__"]
    pairs: dict[str, dict[str, str]] = {}
    for name in names:
        split = _pair_key(name)
        if split is None:
            continue
        key, half = split
        pairs.setdefault(key, {})[half] = name

    both = {k: v for k, v in pairs.items() if "down" in v and "up" in v}
    if not both:
        raise NotAnAdapter(
            f"{p.name} holds {len(names)} tensor(s) and no LoRA pairs among "
            f"them. This reads adapters that store a down/up factorisation "
            f"(lora_A/lora_B, lora_down/lora_up); a merged fine-tune or a full "
            f"checkpoint has no such pair to decompose, and comparing one to "
            f"its base is what `/api/diff` is for."
        )

    # Published config, where there is one. `lora_alpha` and `r` are PEFT's
    # names; a missing file is not an error, it just means the scale has to
    # come from the tensors or be flagged as absent.
    cfg = {}
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError):
            log.warning("adapter config beside %s is unreadable", p, exc_info=True)
            cfg = {}
    cfg_alpha = cfg.get("lora_alpha")
    cfg_rank = cfg.get("r")
    base_model = cfg.get("base_model_name_or_path") or None

    tensors = load_file(str(p))
    alpha_keys = _alphas(header)

    notes: list[str] = []
    modules: list[Module] = []
    unscaled = 0
    for key, halves in both.items():
        down = tensors.get(halves["down"])
        up = tensors.get(halves["up"])
        if down is None or up is None:
            continue
        rank = int(down.shape[0]) if down.ndim >= 1 else None

        stem = key.replace("\0", "").rstrip(".")
        alpha = None
        for cand in (stem, stem.rstrip(".")):
            if cand in alpha_keys and cand in tensors:
                alpha = float(tensors[cand].reshape(-1)[0])
                break
        if alpha is None and cfg_alpha is not None:
            alpha = float(cfg_alpha)
        if rank is None or rank == 0:
            scale, scaled = None, False
        elif alpha is None:
            # NOT assumed to be 1.0. Most adapters set alpha below rank, so
            # assuming parity silently inflates them.
            scale, scaled = None, False
            unscaled += 1
        else:
            scale, scaled = alpha / rank, True

        # `B @ A` is the delta. Computed in float32 whatever the file stores:
        # a norm accumulated in float16 over a 4096-wide matrix loses digits
        # that matter for ordering modules against each other.
        # BOTH halves flatten from their FIRST axis, and that is the whole
        # trick for convolutional LoRAs. A linear pair is `down (rank, in)`
        # and `up (out, rank)`; a conv pair is `down (rank, in, kh, kw)` and
        # `up (out, rank, 1, 1)`. Flattening each from axis 0 gives
        # `(rank, in*kh*kw)` and `(out, rank)`, which compose for both — and
        # `B @ A` is then the delta to the flattened conv kernel.
        #
        # Measured on latent-consistency/lcm-lora-sdxl: reshaping `up` from
        # its LAST axis instead turned `(1280, 64, 1, 1)` into `(81920, 1)`,
        # and every one of that adapter's ~230 conv modules was dropped with a
        # "do not compose" note. The note was honest; the arithmetic was not.
        a = down.to(torch.float32).reshape(down.shape[0], -1)
        b = up.to(torch.float32).reshape(up.shape[0], -1)
        try:
            delta = b @ a if b.shape[1] == a.shape[0] else (b.T @ a)
        except RuntimeError:
            notes.append(
                f"{stem}: the two halves do not compose "
                f"({tuple(up.shape)} against {tuple(down.shape)}), so its "
                f"delta was not computed rather than reshaped into agreement."
            )
            continue
        norm = float(torch.linalg.norm(delta))
        if scale is not None:
            norm *= scale

        component, role = _classify(stem)
        modules.append(
            Module(
                name=stem,
                component=component,
                role=role,
                rank=rank,
                scale=scale,
                scaled=scaled,
                delta_norm=norm,
                relative=_relative(base, stem, delta, scale),
            )
        )

    if not modules:
        raise NotAnAdapter(
            f"{p.name} has LoRA-shaped names but no pair whose halves could be "
            f"multiplied. Nothing here is a decomposable adapter."
        )

    if unscaled:
        notes.append(
            f"{unscaled} module(s) published neither an alpha tensor nor a "
            f"`lora_alpha` in a config beside the file. Their norms are "
            f"UNSCALED — assuming alpha equals rank would silently inflate "
            f"any adapter that set it lower, which most do."
        )
    if cfg_rank is not None and any(m.rank != cfg_rank for m in modules):
        notes.append(
            f"The config declares r={cfg_rank} and the tensors do not all "
            f"agree. The per-module rank is read from the tensors, which is "
            f"what the maths uses."
        )

    grouped: dict[tuple[str, str], Group] = {}
    for m in modules:
        g = grouped.setdefault(
            (m.component, m.role), Group(m.component, m.role, 0, 0.0)
        )
        g.modules += 1
        g.delta_norm += m.delta_norm

    modules.sort(key=lambda m: -m.delta_norm)
    listed = modules[: max(1, int(top))]
    return AdapterReport(
        path=str(p),
        ranks=sorted({m.rank for m in modules if m.rank is not None}),
        modules_total=len(modules),
        modules_listed=len(listed),
        components=sorted({m.component for m in modules}),
        roles=sorted({m.role for m in modules}),
        groups=sorted(grouped.values(), key=lambda g: -g.delta_norm),
        top=listed,
        all_scaled=all(m.scaled for m in modules),
        base_model=base_model,
        notes=notes,
    )


def _relative(base, stem: str, delta, scale: float | None) -> float | None:
    """`||ΔW|| / ||W||`, or None.

    None whenever the base weight cannot be found and matched — an
    approximation here would be a ratio against a denominator nobody chose,
    and the ratio is the whole reason to want the number.
    """
    if base is None:
        return None
    import torch

    # LoRA names are the module path with separators mangled by whichever tool
    # wrote them. Try the spellings rather than one.
    candidates = {
        stem,
        stem.replace("lora_unet_", "").replace("lora_te_", ""),
        stem.replace("_", "."),
        stem.replace("$$", "."),
    }
    named = dict(base.named_parameters()) if hasattr(base, "named_parameters") else {}
    for cand in candidates:
        for suffix in ("", ".weight"):
            w = named.get(cand + suffix)
            if w is None:
                continue
            try:
                d = delta.to(torch.float32)
                if scale is not None:
                    d = d * scale
                base_norm = float(torch.linalg.norm(w.detach().to(torch.float32)))
                if base_norm == 0:
                    return None
                return float(torch.linalg.norm(d)) / base_norm
            except RuntimeError:
                return None
    return None
