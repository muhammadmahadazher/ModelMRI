"""Sparse autoencoder loading and math.

Loads SAEs directly from the HuggingFace Hub without depending on the sae-lens
package — the formats are stable and the math is small:

    features = relu(prepare(x) @ W_enc + b_enc)
    recon    = features @ W_dec + b_dec

`prepare` is the interesting part, and it is measured rather than assumed. An
SAE is trained on activations in one particular convention — centered along
d_model or not, with b_dec subtracted from the input or not — and feeding it
the other kind does not error. It returns features, in the right shape, for a
vector the SAE never saw.

That is what was happening. cfg.json for the default release has no
`apply_b_dec_to_input` key and nothing anywhere declares centering, so the
panel fed the model's raw HuggingFace residual stream to an SAE trained on
TransformerLens activations, which are centered. Measured at
blocks.8.hook_resid_pre, prompt "The Eiffel Tower is located in the city of",
fraction of variance unexplained went from 13579.24 to 0.0010 once the
convention was right, and the number of features firing per token from 7491.5
to 60.5. The panel's top-8 overlapped the correct top-8 two of eight.

So `calibrate` runs all four conventions against the model the SAE is actually
attached to and keeps the one that reconstructs, because "which convention"
has no universal answer — Gemma Scope SAEs are trained on raw HuggingFace
activations and must not be centered. See CONVENTIONS below.

Default release: jbloom/GPT2-Small-SAEs-Reformatted, one SAE per GPT-2
residual-stream hook point (e.g. blocks.8.hook_resid_pre, d_sae=24576).

## Two on-disk layouts, one in-memory object

SAELens ships `cfg.json` + `sae_weights.safetensors` per hook point. Gemma
Scope ships one `params.npz` per (layer, width, average L0) — measured on
google/gemma-scope-2b-pt-res, `layer_20/width_16k/average_l0_71/params.npz` is
302,131,416 bytes holding five float32 arrays: W_enc [2304, 16384], W_dec
[16384, 2304], b_enc [16384], b_dec [2304] and `threshold` [16384].

`_read_sae_lens` and `_read_gemma_scope` are the entire difference between
them. Each returns the same `_Loaded`, so everything below — calibration,
encode, decode, steering — is written once and cannot drift apart. Forking the
encode path per format would mean the convention search that makes this module
trustworthy existed in two copies, one of which would eventually stop being
the one that runs.

## `threshold` is a gate, not decoration

Gemma Scope SAEs are JumpReLU: a feature emits its pre-activation when that
pre-activation clears the feature's OWN learned threshold, and zero otherwise.
The thresholds are large and per-feature — on the release above they span
4.5164 to 30.2257 — so a plain ReLU passes almost everything.

Measured on google/gemma-2-2b, blocks.20.hook_resid_post, prompt "The Eiffel
Tower is located in the city of", 11 tokens: the JumpReLU gate fires 66.6
features per token, against 1795.0 for the same weights through a plain ReLU.
The release advertises an average L0 of 71. That is the same class of silent
wrongness as the input-convention bug above — right shape, plausible
magnitudes, a gate that was never applied — so `_activate` is the one place
either rule lives.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors.torch import load_file

from .errors import BadRequest, Refusal

DEFAULT_SAE_REPO = "jbloom/GPT2-Small-SAEs-Reformatted"
DEFAULT_SAE_HOOK = "blocks.8.hook_resid_pre"

# The two layouts this module can open. Named rather than sniffed by repo id:
# a repo called "gemma-scope-something" that ships safetensors is a thing that
# could exist, and matching on names would open the wrong reader for it.
LAYOUT_SAE_LENS = "sae_lens"
LAYOUT_GEMMA_SCOPE = "gemma_scope"

# The ONLY reader of a Gemma Scope path. `embedding/width_4k/...` exists in the
# same repo and deliberately does not match: this module addresses transformer
# blocks through `blocks.N.hook_*` names, and the embedding stream has no such
# name. Listing it as a layer would offer a release nothing here can hook.
GEMMA_SCOPE_FILE = re.compile(
    r"^layer_(\d+)/(width_[\w.]+)/average_l0_(\d+)/params\.npz$"
)

# Gemma Scope's `layer_N` directory is the stream LEAVING block N — the release
# was trained on the output of `model.model.layers[N]`. There is no resid_pre
# release, so a resid_pre hook against this layout is refused rather than
# quietly served the other side of the block.
GEMMA_SCOPE_POINT = "resid_post"


# The four ways an SAE's input can be prepared, and the reason there are four.
#
# `b_dec` is declared by a cfg key that the default release does not have, so
# `cfg.get("apply_b_dec_to_input", False)` resolved to False for an SAE whose
# training forward always subtracted it. Centering is not declared at all:
# SAELens SAEs are trained on TransformerLens activations, and TransformerLens
# loads with center_writing_weights=True, which makes the residual stream
# mean-zero along d_model. HuggingFace's stream is not centered — its d_model
# mean is nowhere near zero — so an SAE trained on a centered stream
# was being asked about vectors it never saw.
#
# Measured at blocks.8.hook_resid_pre, prompt "The Eiffel Tower is located
# in the city of", 11 tokens, float32, fraction of variance unexplained:
#
#     raw, no b_dec  (what shipped)   FVU 13579.24    L0 7491.5
#     raw + b_dec                     FVU 12908.35    L0 2745.4
#     centered, no b_dec              FVU     0.4219  L0 1344.0
#     centered + b_dec                FVU     0.0010  L0   60.5
#
# The top-8 features the panel plots overlapped the correct ones 2 of 8.
#
# Neither convention is universally right — Gemma Scope SAEs are trained on raw
# HuggingFace activations and must not be centered — so this is not a new
# default to hardcode. It is measured per SAE against the model it is actually
# attached to, which is the only form of the answer that is true on someone
# else's machine with someone else's SAE.
# Ordered LEAST-transforming first, and the order is load-bearing. `sorted` is
# stable, so conventions that score identically keep this order and the winner
# is the one that touched the activations least. An SAE the data cannot
# distinguish should not be silently centered: applying a transform that buys
# no measurable reconstruction is a change with nothing behind it.
CONVENTIONS: tuple[tuple[str, bool, bool], ...] = (
    ("raw", False, False),
    ("b_dec", False, True),
    ("centered", True, False),
    ("centered+b_dec", True, True),
)

# Worse than predicting the mean vector. Not a taste threshold: at FVU >= 1 the
# "reconstruction" carries less of the activation than a constant would, so the
# features are not a decomposition of anything and must not be plotted.
FVU_UNUSABLE = 1.0


# --------------------------------------------------------- which release, and why
#
# Gemma Scope does not publish "the SAE for layer 20". It publishes 316 files,
# and layer 20 alone is ten of them: two dictionary widths crossed with five
# sparsities each. Those are real choices — a width_65k release has four times
# the features, and average_l0 22 against 294 is the difference between a
# terse decomposition and a dense one. Picking one silently would hand the
# panel a number that looks like a property of the model and is actually a
# property of a directory name nobody was shown.
#
# So the coordinates are arguments, the defaults are RULES rather than values,
# and both the rule and what it resolved to travel back in `SAERelease`.


@dataclass
class SAERelease:
    """Which published SAE this is, and how each coordinate got chosen.

    `chosen_by` is the part that matters. Every coordinate is either "caller"
    or a sentence naming the rule and the alternatives it beat, so a reader of
    the panel can tell a deliberate choice from a default without reading this
    file.

    `available` is the index for THIS layer only, and `None` in it never means
    "none exist": it means the Hub listing could not be read, which is a
    different answer and stays a different answer. `width` and `advertised_l0`
    are `None` for a layout that has no such coordinate — SAELens addresses a
    release by hook point and nothing else — rather than 0 or "".
    """

    repo: str
    layout: str
    #: Path within the repo. None when the release is several files, as
    #: SAELens releases are (cfg.json beside sae_weights.safetensors).
    file: str | None
    layer: int
    point: str
    width: str | None
    #: What the directory name CLAIMS the average L0 is. The measured one
    #: arrives separately, in `SAECalibration.l0`, and the two are worth
    #: reading side by side — they are computed on different corpora.
    advertised_l0: int | None
    chosen_by: dict[str, str]
    available: dict[str, list[int]] | None


@dataclass
class _Loaded:
    """The tensors, however they were stored. Both readers return this.

    This is the seam that keeps one encode path: by the time anything
    downstream sees an SAE, the difference between a `.safetensors` release
    and a `.npz` one has already been spent, and calibration, encode, decode
    and steering are written once.
    """

    W_enc: torch.Tensor
    b_enc: torch.Tensor
    W_dec: torch.Tensor
    b_dec: torch.Tensor
    #: JumpReLU gate, per feature. None means plain ReLU — an SAE that has no
    #: thresholds, not one whose thresholds are zero.
    threshold: torch.Tensor | None
    declared_b_dec: bool | None
    release: SAERelease


# repo -> {layer: {width: [average_l0, ...]}}. One listing per repo per
# process: 300-odd filenames that do not change between two loads, against a
# network round trip that choosing a default cannot skip.
_INDEX_CACHE: dict[str, dict[int, dict[str, list[int]]]] = {}


def release_index(repo: str) -> dict[int, dict[str, list[int]]] | None:
    """Every (layer, width, average L0) this repo publishes. None if unread.

    None is not an empty index, and the distinction is the whole point. `{}`
    means the listing was read and holds no Gemma Scope releases — which is
    how a SAELens repo answers. `None` means nobody managed to ask, and a
    caller that flattened the two would report "this repo publishes nothing"
    about a repo it never reached.
    """
    if repo in _INDEX_CACHE:
        return _INDEX_CACHE[repo]
    try:
        files = HfApi().list_repo_files(repo)
    except Exception:
        # Deliberately not cached: a listing that failed because the network
        # was down must not become this process's permanent answer about the
        # repo. The exception itself is dropped rather than wrapped — every
        # caller that needs the index says so in its own words, and this one
        # has no reader to write a sentence for.
        return None
    index: dict[int, dict[str, list[int]]] = {}
    for name in files:
        m = GEMMA_SCOPE_FILE.match(name)
        if not m:
            continue
        layer, width, l0 = int(m.group(1)), m.group(2), int(m.group(3))
        index.setdefault(layer, {}).setdefault(width, []).append(l0)
    for widths in index.values():
        for l0s in widths.values():
            l0s.sort()
    _INDEX_CACHE[repo] = index
    return index


# Google's own label, e.g. `width_16k` for a 16384-feature dictionary. Parsed
# only to ORDER the widths — the dictionary size this module reports is read
# off W_enc, because the label is a name and the tensor is the measurement.
_WIDTH_LABEL = re.compile(r"^width_(\d+)([km]?)$")
_WIDTH_SCALE = {"": 1, "k": 1024, "m": 1024 * 1024}


def _width_order(label: str) -> tuple[int, int, str]:
    """Sort key for a width label. Unparseable labels sort last, by name.

    A label this cannot read is not given a size of 0 — that would make it the
    narrowest and therefore the default, choosing a release BECAUSE it could
    not be understood.
    """
    m = _WIDTH_LABEL.match(label)
    if not m:
        return (1, 0, label)
    return (0, int(m.group(1)) * _WIDTH_SCALE[m.group(2)], label)


def _pick_width(available: dict[str, list[int]], layer: int) -> tuple[str, str]:
    """The narrowest published width, and the sentence saying so.

    Narrowest because it is the smallest download and the only width Google
    publishes at EVERY layer — measured on gemma-scope-2b-pt-res, width_16k
    and width_65k are the two present for all 26 layers, and the wider sweeps
    exist only at layers 5, 12 and 19. A default that is missing for most
    layers is not a default.
    """
    ordered = sorted(available, key=_width_order)
    return ordered[0], (
        f"default: the narrowest of the {len(ordered)} width"
        f"{'s' if len(ordered) != 1 else ''} published for layer {layer} "
        f"({', '.join(ordered)})"
    )


def _pick_l0(l0s: list[int], width: str) -> tuple[int, str]:
    """The median published sparsity, and the sentence saying so.

    Median rather than a fixed target: the sparsities Google trains are not
    the same numbers at every layer (layer 20 at width_16k is 22/38/71/139/294,
    layer 0 is 13/25/46/105/226), so any constant would land between published
    releases at most layers and have to be rounded to one anyway. The middle of
    what exists is a rule that resolves at every layer without a table. Lower
    of the two middles when the count is even, so it is deterministic.
    """
    ordered = sorted(l0s)
    return ordered[(len(ordered) - 1) // 2], (
        f"default: the median of the {len(ordered)} average-L0 release"
        f"{'s' if len(ordered) != 1 else ''} at {width} "
        f"({', '.join(str(v) for v in ordered)})"
    )


def _parse_hook(hook: str) -> tuple[int, str]:
    """`blocks.20.hook_resid_post` -> (20, "resid_post"). Raises on anything else.

    Split out of `load` so the two readers share one answer about what a hook
    name means, and so a malformed one is rejected before any network call.
    """
    m = re.search(r"blocks\.(\d+)\.hook_(\w+)", hook)
    if not m:
        raise BadRequest(f"Cannot parse layer index from hook name: {hook!r}")
    point = m.group(2)
    # The hook POINT was previously discarded, so every SAE was fed the
    # residual stream ENTERING the block. For a resid_post SAE that is the
    # wrong side of the block: it produces plausible-looking features that
    # describe activations the SAE was never trained on. Reject what we
    # cannot place rather than quietly using the wrong tensor.
    #
    # BadRequest rather than Refusal, and it is a close call: the sentence
    # reads like a refusal ("we will not use the wrong tensor"), but what
    # it rejects is a value in the request, and it names the two that work.
    # 422 is also what this answered before the split, and no test pins it.
    if point not in ("resid_pre", "resid_post"):
        raise BadRequest(
            f"Unsupported hook point {point!r} in {hook!r}. ModelMRI reads the "
            f"residual stream: use a hook_resid_pre or hook_resid_post SAE."
        )
    return int(m.group(1)), point


def _read_sae_lens(repo: str, hook: str, layer: int, point: str) -> _Loaded | None:
    """SAELens layout: `{hook}/cfg.json` + `{hook}/sae_weights.safetensors`.

    Returns None — not an exception — when the repo has no cfg.json at that
    hook, because "this is not that layout" is an ordinary answer that `load`
    handles by trying the other reader. A missing file and a Hub that cannot
    be reached are different events and only the first one gets to be None.
    """
    try:
        cfg_path = hf_hub_download(repo, f"{hook}/cfg.json")
        weights_path = hf_hub_download(repo, f"{hook}/sae_weights.safetensors")
    except EntryNotFoundError:
        return None
    except Exception as err:
        raise Refusal(
            f"Could not fetch the SAE for {hook} from {repo} "
            f"({type(err).__name__}). Check the repo id and that the Hub is "
            f"reachable."
        ) from err

    # read_text rather than open(...).read(): the latter leaves the handle
    # to the garbage collector, which is fine on CPython and not guaranteed
    # anywhere else.
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    tensors = load_file(weights_path)
    try:
        W_enc, b_enc = tensors["W_enc"], tensors["b_enc"]
        W_dec, b_dec = tensors["W_dec"], tensors["b_dec"]
    except KeyError as err:
        raise Refusal(
            f"The SAE at {hook} in {repo} is missing {err.args[0]}, so it is "
            f"not a weight file this can encode with."
        ) from err

    return _Loaded(
        W_enc=W_enc.float(),
        b_enc=b_enc.float(),
        W_dec=W_dec.float(),
        b_dec=b_dec.float(),
        # SAELens releases are plain ReLU. None rather than a zero vector: a
        # threshold of zero is a gate that is always open, which is what ReLU
        # already does, and materialising d_sae zeros to say so would cost
        # megabytes to express nothing.
        threshold=None,
        # None when absent. `cfg.get(key, False)` is what made an undeclared
        # SAE look like one that had declined the subtraction.
        declared_b_dec=(
            bool(cfg["apply_b_dec_to_input"]) if "apply_b_dec_to_input" in cfg else None
        ),
        release=SAERelease(
            repo=repo,
            layout=LAYOUT_SAE_LENS,
            file=None,
            layer=layer,
            point=point,
            width=None,
            advertised_l0=None,
            chosen_by={"hook": "caller"},
            available=None,
        ),
    )


def _read_gemma_scope(
    repo: str,
    hook: str,
    layer: int,
    point: str,
    width: str | None,
    average_l0: int | None,
) -> _Loaded:
    """Gemma Scope layout: one `params.npz` per (layer, width, average L0).

    Resolves the two coordinates first — from the caller, or from the rules in
    `_pick_width` / `_pick_l0` against the repo's real listing — and records
    which of the two it was, because a default nobody was told about is the
    defect this function exists to avoid.
    """
    if point != GEMMA_SCOPE_POINT:
        raise BadRequest(
            f"Gemma Scope's layer_{layer} release is the residual stream "
            f"LEAVING block {layer}, and there is no release for the stream "
            f"entering it. Ask for blocks.{layer}.hook_{GEMMA_SCOPE_POINT} "
            f"rather than {hook!r}."
        )
    # `isinstance(True, int)` is True, so a stray boolean would sail through as
    # average_l0=1 and be reported as a deliberate sparsity choice.
    if isinstance(average_l0, bool):
        raise BadRequest("average_l0 is a published sparsity such as 71, not a flag.")
    if average_l0 is not None:
        average_l0 = int(average_l0)
    if width is not None:
        width = str(width)
        # The panel would reasonably show "16k"; the repo spells it
        # "width_16k". Accept either and normalise, rather than 404 on the
        # shorter form the reader was shown.
        if not width.startswith("width_"):
            width = f"width_{width}"

    index = release_index(repo)
    chosen_by: dict[str, str] = {"layer": "caller (from the hook name)"}
    available: dict[str, list[int]] | None = None

    if index is not None:
        if not index:
            raise Refusal(
                f"{repo} publishes no SAE this can open. ModelMRI reads two "
                f"layouts: SAELens (cfg.json beside sae_weights.safetensors "
                f"under a hook name) and Gemma Scope (params.npz under "
                f"layer_N/width_.../average_l0_...). This repo has neither."
            )
        if layer not in index:
            published = ", ".join(str(n) for n in sorted(index))
            raise BadRequest(
                f"{repo} publishes no SAE for layer {layer}. It has layers {published}."
            )
        available = {w: list(v) for w, v in sorted(index[layer].items())}
        if width is None:
            width, chosen_by["width"] = _pick_width(available, layer)
        elif width not in available:
            offered = ", ".join(sorted(available, key=_width_order))
            raise BadRequest(
                f"{repo} publishes no {width} SAE for layer {layer}. At that "
                f"layer it has {offered}."
            )
        else:
            chosen_by["width"] = "caller"
        if average_l0 is None:
            average_l0, chosen_by["average_l0"] = _pick_l0(available[width], width)
        elif average_l0 not in available[width]:
            offered = ", ".join(str(v) for v in available[width])
            raise BadRequest(
                f"{repo} publishes no average-L0 {average_l0} SAE at "
                f"{width} on layer {layer}. At that width it has {offered}."
            )
        else:
            chosen_by["average_l0"] = "caller"
    else:
        # No listing. A caller who named both coordinates does not need one —
        # the path is fully determined — so this is only fatal when a default
        # would have had to be invented.
        if width is None or average_l0 is None:
            raise Refusal(
                "Gemma Scope publishes one SAE per layer, dictionary width and "
                "average L0, and the list of what exists lives on the "
                "HuggingFace Hub, which could not be read just now. Name a "
                "width and an average L0 to load one without the list, or try "
                "again once the Hub is reachable."
            )
        chosen_by["width"] = "caller"
        chosen_by["average_l0"] = "caller"
        chosen_by["available"] = (
            "not listed: the Hub index could not be read, so what else exists "
            "at this layer is unknown rather than empty"
        )

    name = f"layer_{layer}/{width}/average_l0_{average_l0}/params.npz"
    try:
        path = hf_hub_download(repo, name)
    except EntryNotFoundError as err:
        raise BadRequest(
            f"{repo} has no release at layer {layer}, {width}, average L0 {average_l0}."
        ) from err
    except Exception as err:
        raise Refusal(
            f"Could not fetch the layer {layer} {width} average-L0 "
            f"{average_l0} SAE from {repo} ({type(err).__name__}). Check the "
            f"repo id and that the Hub is reachable."
        ) from err

    tensors = _read_npz(path, repo, name)
    return _Loaded(
        W_enc=tensors["W_enc"],
        b_enc=tensors["b_enc"],
        W_dec=tensors["W_dec"],
        b_dec=tensors["b_dec"],
        threshold=tensors["threshold"],
        # Gemma Scope ships no config declaring the input convention, so there
        # is nothing to declare — which is None, not False. `calibrate`
        # measures it, and a False here would have been a claim the release
        # never made.
        declared_b_dec=None,
        release=SAERelease(
            repo=repo,
            layout=LAYOUT_GEMMA_SCOPE,
            file=name,
            layer=layer,
            point=point,
            width=width,
            advertised_l0=average_l0,
            chosen_by=chosen_by,
            available=available,
        ),
    )


#: What a Gemma Scope params.npz must hold. Read off the real file rather than
#: from documentation — see the module docstring for the measured shapes.
_NPZ_KEYS = ("W_enc", "b_enc", "W_dec", "b_dec", "threshold")


def _read_npz(path: str, repo: str, name: str) -> dict[str, torch.Tensor]:
    """The five arrays, as float32 torch tensors, one at a time.

    numpy is imported here rather than at module scope: it arrives with
    transformers rather than as a declared dependency of this package, and the
    SAELens path has no use for it. `torch.from_numpy` shares the array's
    buffer instead of copying, and each array is released as soon as it is
    wrapped, so peak resident stays at one W_enc rather than two — which is
    150,994,944 bytes on the 16k release and four times that on the 65k one.
    """
    try:
        import numpy as np
    except ImportError as err:  # pragma: no cover - numpy ships with transformers
        raise Refusal(
            "Reading a Gemma Scope SAE needs numpy, which is not installed "
            "here. Install numpy, or load a SAELens-format SAE instead."
        ) from err

    out: dict[str, torch.Tensor] = {}
    with np.load(path) as archive:
        held = set(archive.files)
        missing = [k for k in _NPZ_KEYS if k not in held]
        if missing:
            raise Refusal(
                f"{name} in {repo} is missing {', '.join(missing)}, so it is "
                f"not a Gemma Scope parameter file this can encode with."
            )
        for key in _NPZ_KEYS:
            array = archive[key]
            # float32 already on every release measured; `.float()` on a
            # float32 tensor returns the same object, so this costs nothing
            # when it is already right and is correct if a release ever ships
            # float16.
            out[key] = torch.from_numpy(array).float()
            del array
    return out


def _activate(pre: torch.Tensor, threshold: torch.Tensor | None) -> torch.Tensor:
    """Pre-activations -> feature activations. The one place the gate lives.

    ReLU when there is no threshold, JumpReLU when there is: the feature keeps
    its pre-activation if it clears its own learned threshold and is zero
    otherwise. The `relu` is redundant on every Gemma Scope release measured
    (every threshold is positive, so clearing one implies being positive) and
    is kept because it is what makes the two branches the same function of
    `threshold` — a release with a negative threshold would otherwise emit
    negative "activations" through this line alone.
    """
    if threshold is None:
        return torch.relu(pre)
    return torch.relu(pre) * (pre > threshold)


@dataclass
class SAECalibration:
    """Which input convention this SAE turned out to want, and how well it did.

    `declared` is what cfg.json claimed about b_dec — None when the key is
    absent, which is the case that started this. Kept beside the measured
    answer so a disagreement is visible rather than silently overridden.
    """

    convention: str
    center: bool
    subtract_b_dec: bool
    fvu: float
    l0: float
    rel_err: float
    n_tokens: int
    declared_b_dec: bool | None
    ranked: list[tuple[str, float]]  # every convention tried, by FVU
    # Fields, not properties: the server returns this through
    # `dataclasses.asdict`, which skips properties. As a property `usable`
    # would never reach the browser and the panel would have to carry its own
    # copy of the threshold — putting the decision about whether a measurement
    # can be trusted in the one place that cannot measure it.
    usable: bool
    unusable_at: float = FVU_UNUSABLE


@dataclass
class SAEStatus:
    loaded: bool
    repo: str | None = None
    hook: str | None = None
    layer: int | None = None
    d_in: int | None = None
    d_sae: int | None = None
    # Absent until the first encode, because calibration needs real activations
    # from the model the SAE is attached to.
    calibration: SAECalibration | None = None
    #: "relu" or "jumprelu". None when nothing is loaded — an unloaded panel
    #: does not have a plain-ReLU SAE, it has no SAE.
    activation: str | None = None
    #: [min, max] of the JumpReLU thresholds. None for a ReLU SAE, which has
    #: no thresholds at all rather than thresholds of zero. Two measured
    #: numbers so the gate is visible as a fact about the loaded weights.
    threshold_span: list[float] | None = None
    #: Which published release this is, and which coordinates were defaulted.
    release: SAERelease | None = None


class SAEHandle:
    """One loaded SAE: weights on CPU float32, encode/decode helpers."""

    def __init__(
        self,
        repo: str,
        hook: str,
        point: str,
        layer: int,
        W_enc: torch.Tensor,
        b_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_dec: torch.Tensor,
        apply_b_dec_to_input: bool | None,
        threshold: torch.Tensor | None = None,
        release: SAERelease | None = None,
    ) -> None:
        self.repo, self.hook, self.layer = repo, hook, layer
        self.point = point  # resid_pre | resid_post
        self.W_enc, self.b_enc = W_enc, b_enc
        self.W_dec, self.b_dec = W_dec, b_dec
        # Per-feature JumpReLU gate, or None for a plain-ReLU SAE. Not a zero
        # vector for the ReLU case: "no gate" and "a gate that never closes"
        # are the same arithmetic but different facts, and only one of them is
        # true of a SAELens release.
        self.threshold = threshold
        self.release = release
        # What cfg.json declared. None means the key was absent — which is not
        # the same as "False", and treating it as False is the bug this class
        # now measures its way out of.
        self.declared_b_dec = apply_b_dec_to_input
        self.calibration: SAECalibration | None = None
        self.d_in, self.d_sae = W_enc.shape

    @classmethod
    def load(
        cls,
        repo: str = DEFAULT_SAE_REPO,
        hook: str = DEFAULT_SAE_HOOK,
        *,
        width: str | None = None,
        average_l0: int | None = None,
    ) -> SAEHandle:
        """One SAE, from either layout, as one object.

        `width` and `average_l0` address a Gemma Scope release and are None for
        a SAELens one, which has no such coordinates. Naming either of them
        says which layout is meant; naming neither makes this try SAELens
        first and fall through, because the layout is a property of the repo
        and asking the repo is more honest than matching on its name.

        Keyword-only on purpose: `load(repo, hook, "width_16k")` would be a
        third positional argument that reads like a hook variant, and the two
        coordinates are meaningless in isolation from each other.
        """
        layer, point = _parse_hook(hook)

        if width is not None or average_l0 is not None:
            got = _read_gemma_scope(repo, hook, layer, point, width, average_l0)
        else:
            found = _read_sae_lens(repo, hook, layer, point)
            got = (
                found
                if found is not None
                else _read_gemma_scope(repo, hook, layer, point, None, None)
            )

        return cls(
            repo=repo,
            hook=hook,
            point=point,
            layer=layer,
            W_enc=got.W_enc,
            b_enc=got.b_enc,
            W_dec=got.W_dec,
            b_dec=got.b_dec,
            apply_b_dec_to_input=got.declared_b_dec,
            threshold=got.threshold,
            release=got.release,
        )

    def status(self) -> SAEStatus:
        return SAEStatus(
            loaded=True,
            repo=self.repo,
            hook=self.hook,
            layer=self.layer,
            d_in=self.d_in,
            d_sae=self.d_sae,
            calibration=self.calibration,
            activation="relu" if self.threshold is None else "jumprelu",
            threshold_span=(
                None
                if self.threshold is None
                else [
                    round(self.threshold.min().item(), 6),
                    round(self.threshold.max().item(), 6),
                ]
            ),
            release=self.release,
        )

    def _prepare(self, x: torch.Tensor, center: bool, subtract: bool) -> torch.Tensor:
        if center:
            x = x - x.mean(-1, keepdim=True)
        if subtract:
            x = x - self.b_dec
        return x

    @torch.no_grad()
    def calibrate(self, x: torch.Tensor) -> SAECalibration:
        """Decide which input convention this SAE wants, by reconstructing.

        An SAE is a claim that these activations decompose into these features.
        That claim is checkable: encode, decode, and see how much of the
        variance comes back. So rather than trusting a config key that may be
        absent — or a convention inherited from whichever library trained it —
        run all four and keep the one that reconstructs.

        On the default SAE the margin is five orders of magnitude — FVU 0.0010
        against 13579.24 — so the choice is never close in practice. It can be
        exactly tied in principle, though: an SAE that reconstructs the identity
        does so under every convention. `CONVENTIONS` is therefore ordered
        least-transforming first and the sort is stable, so a tie resolves to
        the convention that touched the activations least rather than to
        whichever happened to be listed first.

        Note which tensor is the target: when a convention centers the input,
        the SAE is reconstructing the CENTERED stream, because that is the
        stream it was trained on. Scoring it against the raw one would fail
        every centered SAE for being correct.
        """
        x = x.float()
        # Deliberately a plain ValueError and NOT a BadRequest: no request
        # reaches this check. `calibrate` is called from `encode`, with a
        # tensor this package built out of the model's own residual stream, so
        # a wrong shape here is a ModelMRI bug and belongs on the 500 path with
        # its traceback in the log — not in front of a user who cannot act on
        # it. tests/test_saes.py calls it directly and expects ValueError.
        if x.ndim != 2 or x.shape[-1] != self.d_in:
            raise ValueError(
                f"Calibration needs [S, {self.d_in}] activations, got {tuple(x.shape)}"
            )

        scored: list[tuple[str, float, bool, bool, float, float]] = []
        for name, center, subtract in CONVENTIONS:
            target = x - x.mean(-1, keepdim=True) if center else x
            feats = _activate(
                self._prepare(x, center, subtract) @ self.W_enc + self.b_enc,
                self.threshold,
            )
            recon = feats @ self.W_dec + self.b_dec
            resid = target - recon
            denom = (target - target.mean(0)).pow(2).sum()
            fvu = (resid.pow(2).sum() / denom).item() if denom > 0 else float("inf")
            scored.append(
                (
                    name,
                    fvu,
                    center,
                    subtract,
                    feats.gt(0).float().sum(-1).mean().item(),
                    (resid.norm() / (target.norm() + 1e-12)).item(),
                )
            )

        scored.sort(key=lambda row: row[1])
        name, fvu, center, subtract, l0, rel_err = scored[0]
        self.calibration = SAECalibration(
            convention=name,
            center=center,
            subtract_b_dec=subtract,
            fvu=round(fvu, 6),
            l0=round(l0, 2),
            rel_err=round(rel_err, 6),
            n_tokens=int(x.shape[0]),
            declared_b_dec=self.declared_b_dec,
            ranked=[(row[0], round(row[1], 6)) for row in scored],
            usable=fvu < FVU_UNUSABLE,
        )
        return self.calibration

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[S, d_in] residual activations -> [S, d_sae] feature activations.

        Calibrates on first use. The activations passed here are exactly the
        ones calibration needs, so there is nothing for a caller to arrange and
        no way to forget.
        """
        x = x.float()
        if self.calibration is None:
            self.calibrate(x)
        cal = self.calibration
        assert cal is not None
        return _activate(
            self._prepare(x, cal.center, cal.subtract_b_dec) @ self.W_enc + self.b_enc,
            self.threshold,
        )

    @torch.no_grad()
    def encode_feature(self, x: torch.Tensor, feature_id: int) -> torch.Tensor:
        """[S, d_in] -> [S] activations of ONE feature. Same arithmetic as encode.

        One column of `W_enc` instead of all 24,576, which is the difference
        between 768 multiply-adds and 19 million. `feature_ablate` asks this
        once per scored row to report how much of a feature the SAE still reads
        after that feature's contribution has been subtracted from the stream —
        a full `encode` per row would have made a free check cost seconds.

        Requires an existing calibration and does not create one: a single
        feature's activation is not enough to choose a convention, and silently
        calibrating from it would pick one on evidence nobody asked for.
        """
        cal = self.calibration
        if cal is None:
            raise ValueError(
                "encode_feature needs a calibration; call encode() on the full "
                "activations first so the input convention is chosen on all of "
                "them rather than on one feature."
            )
        prepared = self._prepare(x.float(), cal.center, cal.subtract_b_dec)
        return _activate(
            prepared @ self.W_enc[:, feature_id] + self.b_enc[feature_id],
            # One scalar, not the whole [d_sae] vector: the gate is per
            # feature, so restricting the encoder to one column has to
            # restrict the threshold to the same one or the column would be
            # judged against 16,384 other features' thresholds.
            None if self.threshold is None else self.threshold[feature_id],
        )

    @torch.no_grad()
    def decode(self, feats: torch.Tensor) -> torch.Tensor:
        """[S, d_sae] feature activations -> [S, d_in] reconstruction."""
        return feats.float() @ self.W_dec + self.b_dec

    def steering_vector(self, feature_id: int) -> torch.Tensor:
        """Unit-norm decoder direction for one feature ([d_in])."""
        v = self.W_dec[feature_id]
        return v / (v.norm() + 1e-8)
