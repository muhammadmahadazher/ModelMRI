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

No default release. `SAEHandle.load` requires a repo and a hook, because a
default that names one model is one model's SAE answering for every model —
`sae_registry.for_model` is where "which release belongs to this model" is
answered, and "none" is one of its answers.

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

## The gate is named by the release, not inferred from its tensors

`threshold is None` used to mean "plain ReLU", and for a while that was true of
every SAE this could open: Gemma Scope shipped thresholds, SAELens shipped none.
It is false now. SAELens registers four inference architectures — standard,
gated, topk, jumprelu — and a TopK release ships EXACTLY the four tensors a
standard release ships. Nothing in its weight file betrays it; the entire gate
is one cfg.json key. So a loader that parsed cfg.json for nothing loaded every
modern release wide open, and the failure looks like the two above: right
shape, plausible magnitudes, a rule that was never applied.

`_read_sae_lens` therefore reads the architecture, and `_activate` dispatches on
that name rather than on which tensors turned up. What cannot be encoded is
refused BY NAME — a Refusal naming the architecture is an ordinary answer, and
the one thing that must never happen is a gated SAE loading as a ReLU. The same
rule covers `normalize_activations`, whose scaling factor SAELens keeps in its
own bundled table rather than in the release, and `reshape_activations`, which
says the SAE reads per-head attention output rather than the residual stream.

## Two fidelity numbers, and only one of them was in activation space

`calibrate` answers "does this SAE reconstruct the stream it is attached to",
in FVU and in L0. Both are taken against the SAE's own input, and both can read
well while the model on top stops predicting: FVU is measured against the
directions carrying the residual stream's VARIANCE, and the directions the next
token depends on are not those directions.

`ce_recovered` at the bottom of this file asks the output-space question —
splice the reconstruction back in, run the model, and see how much of its
predictive loss survives against an ablation floor. Its long comment is where
the floor, the corpus and the shared splice are argued; the short version is
that the floor is half the number, it is named in the payload, and all three
raw losses come back so the other normalisation can be computed from them.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors.torch import load_file

from .errors import BadRequest, Refusal

# No module-level default repo or hook. They named one model's release, and a
# default that names a model is what `/api/sae/load` stopped doing: the
# registry knows which release belongs to which model, so the answer comes
# from there or the call is refused by name. A default here would put the
# assumption back one layer down, where it is harder to see.
#
# `SAEHandle.load` therefore REQUIRES both. A caller that does not know which
# SAE it wants is a caller that should be asking `sae_registry.for_model`.

# The two layouts this module can open. Named rather than sniffed by repo id:
# a repo called "gemma-scope-something" that ships safetensors is a thing that
# could exist, and matching on names would open the wrong reader for it.
LAYOUT_SAE_LENS = "sae_lens"
LAYOUT_GEMMA_SCOPE = "gemma_scope"

# The activation function an SAE's encoder ends in, and the one place each name
# is spelled. SAELens calls these "architectures"; this module calls the four
# it can apply activations, because "standard" names a release and `relu` names
# a rule, and the rule is what `_activate` has to run.
ACT_RELU = "relu"
ACT_JUMPRELU = "jumprelu"
ACT_TOPK = "topk"
ACT_GATED = "gated"

# SAELens's `architecture` strings mapped to the rule each one means. This is
# the whole of what this module encodes; every other string is refused by name.
# Refused rather than approximated because the nearest gate to an architecture
# nobody here has read is a plain ReLU, and a plain ReLU is precisely the wrong
# answer — read off the register_sae_class calls in sae_lens/__init__.py at
# 6.50.0, where these four are the architectures that have an inference class
# AND read and write the same stream.
ARCHITECTURE_ACTIVATION: dict[str, str] = {
    "standard": ACT_RELU,
    "gated": ACT_GATED,
    "topk": ACT_TOPK,
    "jumprelu": ACT_JUMPRELU,
}

# Registered for TRAINING only: `get_sae_class("batchtopk")` is a bare dict
# lookup in SAELens and raises. A cfg.json naming one of these is a training
# checkpoint rather than a release, and its gate ranks activations across a
# whole batch — which would make one token's features depend on which other
# tokens were in the request. A RELEASED BatchTopK SAE says "jumprelu" and
# ships the distilled threshold, and that one loads through the JumpReLU path,
# because at inference it IS JumpReLU.
ARCHITECTURES_TRAINING_ONLY = ("batchtopk", "matryoshka_batchtopk")

# Registered with inference classes and still refused. A transcoder maps
# between TWO hook points (`hook_name` -> `hook_name_out`), so it does not
# reconstruct the stream it reads, and `calibrate`'s "how much of the variance
# comes back" is not a question that can be asked of it at all.
ARCHITECTURES_TRANSCODER = (
    "transcoder",
    "skip_transcoder",
    "jumprelu_transcoder",
    "jumprelu_skip_transcoder",
)

# Which tensors a release must ship, per activation. Gated is the one to read
# twice: it has NO b_enc. SAELens assigns `self.b_enc = None` on a name it never
# registered as a parameter, so the key never reaches the file — asking for one
# would refuse every gated release, and defaulting one to zeros would load one
# wrong, which is worse.
_WEIGHT_KEYS: dict[str, tuple[str, ...]] = {
    ACT_RELU: ("W_enc", "b_enc", "W_dec", "b_dec"),
    ACT_TOPK: ("W_enc", "b_enc", "W_dec", "b_dec"),
    ACT_JUMPRELU: ("W_enc", "b_enc", "W_dec", "b_dec", "threshold"),
    ACT_GATED: ("W_enc", "W_dec", "b_dec", "b_gate", "r_mag", "b_mag"),
}

# cfg.json has been written by three schemas, and which one a file is decides
# what an ABSENT `architecture` key means — the whole point of telling them
# apart. In the modern schema the key is always written, so its absence is a
# broken file; in the two older ones it did not exist yet, so its absence is
# the schema and standard is what SAELens's own migration resolves it to.
SCHEMA_MODERN = "modern"  # >= 6.0.0-rc.0: SAEConfig fields + a nested metadata
SCHEMA_FLAT = "flat"  # v3-v5: flat keys, activation_fn_str, no metadata
SCHEMA_LEGACY = "legacy"  # pre-v3: hook_point, and no architecture at all

# SAELens's own cutover, from `handle_config_defaulting`: a file whose
# sae_lens_version is below 6.0.0-rc.0 goes through the legacy migration. Only
# the major number is compared, because that is the only question asked of it
# and parsing a full version needs a dependency this module does not have.
_SCHEMA_MODERN_MAJOR = 6

# What `normalize_activations` may say. Four values, and only one of them can
# be honoured here — see `_refuse_unencodable_preprocessing` for why each of
# the others is a refusal rather than a silent load.
NORMALIZE_NONE = "none"
NORMALIZE_EXPECTED_AVERAGE = "expected_average_only_in"
NORMALIZE_CONSTANT = "constant_norm_rescale"
NORMALIZE_LAYER_NORM = "layer_norm"

# `reshape_activations`. "hook_z" says the SAE was trained on per-head
# attention output and expects [..., n_heads, d_head] flattened before the
# encoder, which is not the tensor this module addresses.
RESHAPE_NONE = "none"
RESHAPE_HOOK_Z = "hook_z"

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
    #: The encoder bias, or None for a Gated release — which genuinely has
    #: none. SAELens assigns `self.b_enc = None` on a name it never registered
    #: as a parameter, so the key never reaches the file. None rather than a
    #: zero vector, for the same reason `threshold` is: "this architecture has
    #: no such tensor" and "its values happen to be zero" are different facts,
    #: and only the first one is true of a gated SAE.
    b_enc: torch.Tensor | None
    W_dec: torch.Tensor
    b_dec: torch.Tensor
    #: Which rule turns pre-activations into feature activations. READ from
    #: the release rather than inferred from which tensors turned up: a TopK
    #: release ships exactly the four tensors a standard one ships, so there
    #: is nothing to infer from and the guess would always be "relu".
    activation: str
    #: JumpReLU gate, per feature. None means plain ReLU — an SAE that has no
    #: thresholds, not one whose thresholds are zero.
    threshold: torch.Tensor | None
    declared_b_dec: bool | None
    release: SAERelease
    #: How many features may fire per token. TopK only, where it is the whole
    #: gate; None everywhere else, because no other architecture has one.
    k: int | None = None
    #: The three tensors only a Gated release ships: the gate's own bias, the
    #: per-feature log-scale on the magnitude path, and the magnitude bias.
    b_gate: torch.Tensor | None = None
    r_mag: torch.Tensor | None = None
    b_mag: torch.Tensor | None = None


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


def _cfg_schema(cfg: dict) -> str:
    """Which of the three cfg.json generations wrote this file.

    Exactly one question depends on the answer, and it is the load-bearing
    one: what an ABSENT `architecture` key means. SAELens 6 writes that key
    unconditionally — it is `SAEConfig.to_dict()` calling `self.architecture()`
    — so a modern file without one is a broken file and defaulting it would be
    inventing a claim. The two older schemas had no such key at all, so its
    absence there is the schema speaking, and standard is what SAELens's own
    migration resolves it to because standard was the only architecture that
    existed when those files were written.

    The version gate is SAELens's own, from `handle_config_defaulting`:
    `sae_lens_version` at top level, else `metadata.sae_lens_version`, else
    legacy, with the cutover at 6.0.0-rc.0.
    """
    metadata = cfg.get("metadata")
    version = cfg.get("sae_lens_version")
    if version is None and isinstance(metadata, dict):
        version = metadata.get("sae_lens_version")
    if isinstance(version, str):
        # Only the major number, and only compared for ">= 6". Telling
        # "6.0.0-rc.0" from "6.0.0" needs a PEP 440 parser this package does
        # not depend on and does not need one: every released 6.x is at or
        # above the cutover and nothing below 6 can reach it, so the major
        # number answers the only question being asked.
        major = version.split(".", 1)[0]
        if major.isdigit() and int(major) >= _SCHEMA_MODERN_MAJOR:
            return SCHEMA_MODERN
    # A nested `metadata` dict is the modern schema's own shape — the two
    # older ones are flat — so a modern file that lost its version is still
    # recognisable. `activation_fn_str` settles it the other way: it is a key
    # the modern schema deleted, so a file carrying one is not modern
    # whatever else it holds.
    if isinstance(metadata, dict) and "activation_fn_str" not in cfg:
        return SCHEMA_MODERN
    if "activation_fn_str" in cfg or "architecture" in cfg:
        return SCHEMA_FLAT
    return SCHEMA_LEGACY


def _refuse_unencodable_preprocessing(cfg: dict, repo: str, hook: str) -> None:
    """Everything cfg.json says happens BEFORE the encoder, or a Refusal.

    Two keys, and both describe a transform applied to the activations on
    their way in. An SAE trained on rescaled activations and fed the raw
    residual stream is the same failure as the input convention at the top of
    this file — right shape, plausible magnitudes, a rule that was never
    applied — except that neither of these is recoverable by measurement the
    way the b_dec convention is, because the number to undo them with is not
    in the release.

    Checked before the architecture, so a release that is wrong here is
    refused once rather than once per gate.
    """
    norm = cfg.get("normalize_activations", NORMALIZE_NONE)
    # SAELens wrote a bool here before it wrote a string, and its own
    # migration reads False as "none" and True as "expected_average_only_in".
    # The bool test has to come first: `isinstance(True, int)` is True, and a
    # True compared against the strings would fall through to the
    # unknown-value refusal below and name `True` at a reader who would then
    # have nothing to search for.
    if isinstance(norm, bool):
        norm = NORMALIZE_EXPECTED_AVERAGE if norm else NORMALIZE_NONE
    if norm is None:
        norm = NORMALIZE_NONE

    if norm == NORMALIZE_EXPECTED_AVERAGE:
        raise Refusal(
            f"The SAE at {hook} in {repo} declares normalize_activations="
            f"{NORMALIZE_EXPECTED_AVERAGE!r}, and the scaling factor that "
            f"needs is not in this release. SAELens keeps that number in its "
            f"own bundled release table, keyed by release name and SAE id, "
            f"and folds it into the weights when it loads — reading a repo "
            f"path alone cannot recover it, and without it every feature "
            f"magnitude is off by an unknown constant. Load this one through "
            f"sae-lens, or pick a release that needs no folding."
        )
    if norm in (NORMALIZE_CONSTANT, NORMALIZE_LAYER_NORM):
        raise Refusal(
            f"The SAE at {hook} in {repo} declares normalize_activations="
            f"{norm!r}, which rescales each activation vector on the way in "
            f"and has to be undone on the way out. `encode` and `decode` are "
            f"separate calls here with no state between them, so half of that "
            f"rule would hand back a reconstruction in units nothing else "
            f"uses. Pick a release trained on unnormalised activations."
        )
    if norm != NORMALIZE_NONE:
        raise Refusal(
            f"The SAE at {hook} in {repo} declares normalize_activations="
            f"{norm!r}, which is not one of the values SAELens defines "
            f"({NORMALIZE_NONE}, {NORMALIZE_EXPECTED_AVERAGE}, "
            f"{NORMALIZE_CONSTANT}, {NORMALIZE_LAYER_NORM}). Something "
            f"normalises this SAE's input and nothing here knows what."
        )

    reshape = cfg.get("reshape_activations", RESHAPE_NONE)
    if reshape is None:
        reshape = RESHAPE_NONE
    if reshape == RESHAPE_HOOK_Z:
        raise Refusal(
            f"The SAE at {hook} in {repo} declares reshape_activations="
            f"{RESHAPE_HOOK_Z!r}, which means it reads per-head attention "
            f"output — [..., n_heads, d_head], flattened before the encoder. "
            f"ModelMRI addresses the residual stream, whose vectors are "
            f"[..., d_model]: the right shape and the wrong content. Ask for "
            f"a hook_resid_pre or hook_resid_post release."
        )
    if reshape != RESHAPE_NONE:
        raise Refusal(
            f"The SAE at {hook} in {repo} declares reshape_activations="
            f"{reshape!r}, which is not a reshaping this knows. Something "
            f"rearranges this SAE's input before its encoder and nothing here "
            f"knows what."
        )


def _resolve_activation(
    cfg: dict, schema: str, repo: str, hook: str
) -> tuple[str, int | None, str]:
    """cfg.json -> (activation, k, the sentence saying how that was decided).

    The sentence travels with the release in `SAERelease.chosen_by`, because
    "this file says topk" and "this file is old enough that standard is the
    only thing it could be" are different facts and a reader of the panel has
    to be able to tell them apart. Same discipline as `_pick_width` and
    `_pick_l0`, one layer further in.
    """
    architecture = cfg.get("architecture")
    if architecture is None:
        if schema == SCHEMA_MODERN:
            raise Refusal(
                f"The SAE at {hook} in {repo} was written by SAELens 6 or "
                f"later, which always records an architecture, and this "
                f"cfg.json has no architecture key. Defaulting it would be "
                f"the same mistake as reading an absent apply_b_dec_to_input "
                f"as false: a claim the file never made, resolved in favour "
                f"of the gate that does nothing. Re-download the release, or "
                f"load it through sae-lens."
            )
        architecture = "standard"
        why = (
            f"read: this cfg.json names no architecture and its schema "
            f"({schema}) predates the key. Standard was the only inference "
            f"architecture SAELens had when files of this shape were written, "
            f"and standard is what SAELens's own migration resolves an absent "
            f"key to."
        )
    else:
        why = f"declared: cfg.json says architecture {architecture!r}."

    if architecture in ARCHITECTURES_TRAINING_ONLY:
        raise Refusal(
            f"The SAE at {hook} in {repo} declares architecture "
            f"{architecture!r}, which SAELens registers for TRAINING only — "
            f"it has no inference class, and SAELens's own lookup raises on "
            f"it. Its gate ranks activations across a whole batch, so a "
            f"feature's activation would depend on which other tokens were in "
            f"the request. A RELEASED SAE of this kind is saved as "
            f"{ACT_JUMPRELU!r} with the distilled threshold that gate becomes "
            f"at inference, and that one loads here; a cfg.json literally "
            f"saying {architecture!r} is a training checkpoint."
        )
    if architecture in ARCHITECTURES_TRANSCODER:
        raise Refusal(
            f"The SAE at {hook} in {repo} declares architecture "
            f"{architecture!r}. A transcoder maps between TWO hook points — "
            f"it reads one and writes another — so it does not reconstruct "
            f"the stream it was read from, and the reconstruction quality "
            f"every number in this panel is scaled against cannot be asked of "
            f"it at all. Load a release trained to reconstruct the hook it "
            f"reads."
        )
    activation = ARCHITECTURE_ACTIVATION.get(architecture)
    if activation is None:
        raise Refusal(
            f"The SAE at {hook} in {repo} declares architecture "
            f"{architecture!r}, which ModelMRI cannot encode with. It reads "
            f"{', '.join(sorted(ARCHITECTURE_ACTIVATION))}. Refused rather "
            f"than approximated, because the nearest gate to an architecture "
            f"nobody here has read is a plain ReLU — and a plain ReLU is "
            f"exactly the wrong answer for every architecture that has a gate."
        )

    # The pre-6.0 spelling. v3-v5 wrote the activation function as a string
    # beside `architecture` rather than as the architecture itself, and TopK
    # lived there: `activation_fn_str: "topk"` with the k in
    # `activation_fn_kwargs`. SAELens's own migration promotes exactly that
    # pair, and a release written by v5 is not a rarity — reading only the
    # modern key would load every one of them wide open.
    kwargs = cfg.get("activation_fn_kwargs") or {}
    fn = cfg.get("activation_fn_str", cfg.get("activation_fn"))
    if fn is not None and fn != ACT_RELU:
        if fn == ACT_TOPK:
            if kwargs.get("k") is None:
                raise Refusal(
                    f"The SAE at {hook} in {repo} declares activation_fn_str "
                    f"{ACT_TOPK!r} and no k in its activation_fn_kwargs, so "
                    f"it names a gate and withholds the one number that gate "
                    f"is. SAELens leaves this file as a standard SAE and "
                    f"drops the activation function, which loads it as a "
                    f"plain ReLU; this refuses instead."
                )
            activation = ACT_TOPK
            why += (
                f" Its activation_fn_str is {ACT_TOPK!r} with a k, which is "
                f"how v3-v5 spelled the TopK architecture, so the gate is "
                f"top-k whatever the architecture key says."
            )
        elif fn == "tanh-relu":
            raise Refusal(
                f"The SAE at {hook} in {repo} declares activation_fn_str "
                f"{fn!r}, and the two versions of SAELens disagree about what "
                f"that means: v5 computed tanh(relu(pre)), and v6 drops the "
                f"value and computes relu(pre). Neither answer can be trusted "
                f"without knowing which version trained this release, so this "
                f"names the disagreement rather than picking a side."
            )
        else:
            raise Refusal(
                f"The SAE at {hook} in {repo} declares activation_fn_str "
                f"{fn!r}, which is not an activation function SAELens ever "
                f"defined. Its encoder ends in something nothing here can "
                f"reproduce."
            )

    k = None
    if activation == ACT_TOPK:
        k = cfg.get("k", kwargs.get("k"))
        if k is None:
            raise Refusal(
                f"The SAE at {hook} in {repo} declares architecture "
                f"{architecture!r} and names no k. k IS the gate — it is how "
                f"many of the d_sae features may fire at each token — and "
                f"there is no defensible default for it: SAELens's dataclass "
                f"default of 100 is a placeholder, not a property of "
                f"anybody's release."
            )
        # `isinstance(True, int)` is True, so a stray boolean would sail
        # through as a top-1 gate and be reported as a published sparsity.
        if isinstance(k, bool) or not isinstance(k, int):
            raise Refusal(
                f"The SAE at {hook} in {repo} declares k={k!r}, which is not "
                f"a count of features. k is how many of this SAE's features "
                f"may fire at each token, so it is a whole number."
            )

    trained_as = None
    if isinstance(cfg.get("metadata"), dict):
        trained_as = cfg["metadata"].get("training_architecture")
    if trained_as and trained_as != architecture:
        why += (
            f" Its metadata records that it was TRAINED as {trained_as!r} and "
            f"saved for inference as {architecture!r}, so the threshold it "
            f"ships was distilled at save time out of a gate that ranked "
            f"activations — one number every feature shares, rather than a "
            f"per-feature learned bar. A flat threshold span on this release "
            f"is the shape of the release and not a fault."
        )
    return activation, k, why


def _refuse_impossible_k(k: int, d_sae: int, repo: str, hook: str) -> None:
    """k against the weight file it shipped beside, not against the cfg alone.

    Both bounds are silent otherwise: `torch.topk` answers a k of zero without
    complaining, and the shapes come from the tensors here, so a k that
    describes a wider dictionary than the one on disk would simply raise
    somewhere deeper with a message about tensor sizes.
    """
    if k < 1:
        raise Refusal(
            f"The SAE at {hook} in {repo} declares k={k}, and a gate that "
            f"lets {k} features fire is not a sparsity — it is an SAE that "
            f"emits nothing at all. torch.topk would answer that question "
            f"without complaining, which is why this asks it first."
        )
    if k > d_sae:
        raise Refusal(
            f"The SAE at {hook} in {repo} declares k={k}, and the weight file "
            f"beside that cfg.json holds {d_sae} features. k is how many of "
            f"them may fire at each token, so a k above {d_sae} describes a "
            f"different weight file than the one that shipped here."
        )


def _declared(cfg: dict, *names: str) -> object | None:
    """The first of these keys, looked for at the top level and in `metadata`.

    The three cfg generations moved a release's account of itself between
    those two places without renaming anything: SAELens 6 pushed `hook_name`
    and `model_name` down into the nested `metadata` and left `d_in` and
    `d_sae` flat, v3-v5 kept all four flat, and pre-v3 spelled the hook
    `hook_point`. A reader that looks in one place reads half the file and
    concludes the declaration is absent — which is the one answer that must
    not be invented here, because an absent declaration and a contradicted one
    get opposite treatment below.
    """
    metadata = cfg.get("metadata")
    for name in names:
        if name in cfg:
            return cfg[name]
        if isinstance(metadata, dict) and name in metadata:
            return metadata[name]
    return None


def _refuse_a_cfg_that_describes_another_release(
    cfg: dict, W_enc: torch.Tensor, repo: str, hook: str, layer: int, point: str
) -> None:
    """The cfg's own account of itself, against the file it was read beside.

    These checks are free and none of them was being made. cfg.json states the
    hook it reads and the two dimensions of the dictionary it describes; this
    function has the hook it asked the Hub for and the tensors that came back.
    So a directory whose cfg.json describes a DIFFERENT release loaded without
    a word, and every number after it — the convention search, the FVU, the
    ranked features — described weights that were never at that address.

    The hook is the dangerous one, and specifically resid_pre against
    resid_post: those two streams differ by one block's output, so an SAE fed
    the wrong side still reconstructs well enough to sit under `FVU_UNUSABLE`
    and be plotted. `_parse_hook` already refuses a hook name it cannot place;
    this is the same refusal one level in, about the name the release chose
    for itself rather than the name the caller asked by.

    Only a declaration that PARSES is compared, and only a dimension that is
    an integer. A cfg naming its hook in a spelling this module's regex cannot
    read is a shape nobody here understands, and refusing on it would reject
    releases for being written unfamiliarly rather than for being wrong.
    """
    for name, got in (("d_in", int(W_enc.shape[0])), ("d_sae", int(W_enc.shape[1]))):
        said = _declared(cfg, name)
        # `isinstance(True, int)` is True, and a bool is not a width.
        if isinstance(said, bool) or not isinstance(said, int):
            continue
        if said != got:
            raise Refusal(
                f"The SAE at {hook} in {repo} declares {name}={said}, and the "
                f"weight file beside that cfg.json holds {name}={got}. One of "
                f"those two files belongs to another release and there is no "
                f"way to tell which from here: every number this panel "
                f"reports is measured on the tensors, and every gate it "
                f"applies is read off the cfg, so a disagreement between them "
                f"is a disagreement about what was measured."
            )

    said_hook = _declared(cfg, "hook_name", "hook_point")
    if not isinstance(said_hook, str):
        return
    m = re.search(r"blocks\.(\d+)\.hook_(\w+)", said_hook)
    if m is not None and (int(m.group(1)), m.group(2)) != (layer, point):
        raise Refusal(
            f"The SAE at {hook} in {repo} says in its own cfg.json that it "
            f"reads {said_hook!r}, and it was fetched from — and would be "
            f"attached at — {hook!r}. A release addressed one block or one "
            f"side of a block away from where it was trained reconstructs "
            f"well enough to look calibrated, which is why nothing "
            f"downstream would have caught this. Ask for {said_hook!r}, or "
            f"use a repo whose directories are the hooks they name."
        )


def _read_sae_lens(repo: str, hook: str, layer: int, point: str) -> _Loaded | None:
    """SAELens layout: `{hook}/cfg.json` + `{hook}/sae_weights.safetensors`.

    Returns None — not an exception — when the repo has no cfg.json at that
    hook, because "this is not that layout" is an ordinary answer that `load`
    handles by trying the other reader. A missing file and a Hub that cannot
    be reached are different events and only the first one gets to be None.

    A cfg.json that EXISTS and names something unencodable is neither of
    those, and refuses. Returning None for it would send `load` on to the
    Gemma Scope reader, which would end in "this repo publishes no SAE this
    can open" — about a release that was found, read and understood.
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
    schema = _cfg_schema(cfg)
    _refuse_unencodable_preprocessing(cfg, repo, hook)
    activation, k, chose_architecture = _resolve_activation(cfg, schema, repo, hook)

    tensors = load_file(weights_path)
    # Asked for by ARCHITECTURE rather than by "the four everybody has", which
    # is what let a gated release load as a broken standard SAE. Gated is the
    # case to read twice: it ships no b_enc at all, so a reader that demanded
    # one would refuse every gated release and a reader that defaulted one to
    # zeros would load one wrong.
    needed = _WEIGHT_KEYS[activation]
    missing = [key for key in needed if key not in tensors]
    if missing:
        raise Refusal(
            f"The SAE at {hook} in {repo} declares a {activation} "
            f"architecture, whose weights are {', '.join(needed)}, and its "
            f"weight file is missing {', '.join(missing)}. It is not a weight "
            f"file this can encode with, and encoding it through the tensors "
            f"it does have would apply a gate the publisher did not train."
        )

    # Only ABSENT keys were checked, and a weight file can also carry a key
    # this does not read. One of the ones it can carry is load-bearing:
    # `scaling_factor` is a per-feature finetuning scale that older releases
    # shipped, and SAELens's own `read_sae_components_from_disk` deletes it
    # when it is all ones and otherwise renames it to
    # `finetuning_scaling_factor` and applies it — raising outright when the
    # cfg says there is none. Dropping a non-ones one here would report every
    # feature magnitude, every L0 and every reconstruction error for a decoder
    # the publisher did not ship: the same silence as reading a TopK release
    # as a ReLU, one file further down.
    #
    # Refused rather than folded, and the distinction is the usual one. What
    # is established is that SAELens applies this factor and that it renames
    # it first; the arithmetic it applies it WITH is not, and a per-feature
    # scale folded into the wrong side of the decoder is a wrong answer that
    # looks exactly like a right one.
    scale = tensors.get("scaling_factor")
    if scale is not None:
        scale = scale.float()
        if not bool(torch.allclose(scale, torch.ones_like(scale))):
            raise Refusal(
                f"The SAE at {hook} in {repo} ships a scaling_factor tensor "
                f"that is not all ones. It is a per-feature finetuning scale: "
                f"SAELens deletes an all-ones one and folds a real one in "
                f"before the decoder, and this loader applies neither. "
                f"Ignoring it would leave every feature magnitude off by a "
                f"factor the publisher trained, with nothing in the panel "
                f"saying so. Load this one through sae-lens, or pick a "
                f"release whose weight file carries only "
                f"{', '.join(needed)}."
            )

    # Anything else the file carries is dropped, and dropped tensors are said
    # out loud rather than left as nothing — the same rule as an unmeasured
    # coordinate. Recorded only when there IS something extra: a sentence
    # saying "no surprises" on every ordinary release is noise, and the
    # architecture sentence already names what was read.
    extra = sorted(set(tensors) - set(needed))
    chose_weights = None
    if extra:
        chose_weights = (
            f"read {', '.join(needed)}, which is what a {activation} release "
            f"ships. This weight file also carries {', '.join(extra)}, which "
            f"nothing here applies."
        )
        if "scaling_factor" in extra:
            chose_weights += (
                " Its scaling_factor is all ones, which is the one value "
                "SAELens itself deletes rather than folds, so dropping it "
                "changes no number; one that was not all ones is refused."
            )

    W_enc = tensors["W_enc"].float()
    W_dec = tensors["W_dec"].float()
    b_dec = tensors["b_dec"].float()
    # `.float()` on every one of them, not only on the four that were always
    # read: a float16 threshold compared against float32 pre-activations
    # promotes silently and gates at a slightly different bar than the one the
    # publisher trained, and `status` quotes that number as measured.
    b_enc = tensors["b_enc"].float() if "b_enc" in needed else None
    threshold = tensors["threshold"].float() if "threshold" in needed else None
    b_gate = tensors["b_gate"].float() if "b_gate" in needed else None
    r_mag = tensors["r_mag"].float() if "r_mag" in needed else None
    b_mag = tensors["b_mag"].float() if "b_mag" in needed else None

    _refuse_a_cfg_that_describes_another_release(cfg, W_enc, repo, hook, layer, point)

    chose_rescale = None
    if activation == ACT_TOPK:
        _refuse_impossible_k(k, W_enc.shape[1], repo, hook)
        # `rescale_acts_by_decoder_norm` multiplies the pre-activations by the
        # decoder row norms BEFORE the selection, so it decides which features
        # win and not only how big they are. Folded into the weights here,
        # exactly as SAELens folds it when it saves an inference model —
        # W_enc and b_enc multiplied by the norms, W_dec divided by them —
        # rather than carried as a flag and applied at encode time. After the
        # fold the plain arithmetic IS the rescaled arithmetic, so nothing
        # downstream has to know the flag existed: `feature_ablate`
        # subtracting act x W_dec[f], the feature corpus reading W_dec[f] as a
        # direction, and `steering_vector` all keep working on one convention
        # instead of two.
        #
        # Type-checked rather than coerced with `bool(...)`, for the same
        # reason k is a few lines up: `bool("false")` is True, so a string a
        # person reads as "off" would switch on a fold that rewrites three
        # tensors and moves which features win the selection. Every other
        # value this module takes out of cfg.json is checked before it is
        # believed; this one was the exception.
        rescale = cfg.get("rescale_acts_by_decoder_norm", False)
        if not isinstance(rescale, bool):
            raise Refusal(
                f"The SAE at {hook} in {repo} declares "
                f"rescale_acts_by_decoder_norm={rescale!r}, which is not true "
                f"or false. That flag multiplies the pre-activations by the "
                f"decoder row norms BEFORE the selection, so it decides which "
                f"features fire and not only how large they are — too much to "
                f"settle on whether a value happens to be truthy."
            )
        if rescale:
            norms = W_dec.norm(dim=-1)
            if bool((norms == 0).any()):
                raise Refusal(
                    f"The SAE at {hook} in {repo} asks for its activations to "
                    f"be rescaled by the decoder norms, and "
                    f"{int((norms == 0).sum())} of its {W_dec.shape[0]} "
                    f"decoder rows are exactly zero. The rescale divides by "
                    f"those norms, so folding it in would put infinities in "
                    f"the decoder and NaNs in every reconstruction. Load this "
                    f"one through sae-lens, which reads the flag at encode "
                    f"time."
                )
            W_enc = W_enc * norms
            b_enc = b_enc * norms
            W_dec = W_dec / norms.unsqueeze(-1)
            # Said out loud, because after this line the tensors held here are
            # not the tensors the publisher uploaded. The arithmetic is
            # invariant — a feature's activation times its decoder row is
            # unchanged, which is what `feature_ablate` subtracts — but a
            # reader comparing `W_dec[f].norm()` against the published file
            # would find every row unit-norm and no sentence explaining why.
            chose_rescale = (
                "folded at load: cfg.json sets rescale_acts_by_decoder_norm, "
                "which multiplies the pre-activations by the decoder row "
                "norms BEFORE the top-k selection and so decides which "
                "features fire. Folded into the weights exactly as SAELens "
                "folds it when it saves an inference model — W_enc and b_enc "
                "multiplied by the norms, W_dec divided by them — so the "
                "plain arithmetic downstream IS the rescaled arithmetic, and "
                "the loaded tensors are not the published ones."
            )

    # None when absent. `cfg.get(key, False)` is what made an undeclared SAE
    # look like one that had declined the subtraction, and the fix is that
    # absent stays a third answer — NOT that it acquires SAELens's default.
    # SAELens defaults an absent key to TRUE in every schema it has written,
    # which is worth telling the reader, so it is said in the sentence below
    # rather than substituted for the missing declaration.
    declared_b_dec = (
        bool(cfg["apply_b_dec_to_input"]) if "apply_b_dec_to_input" in cfg else None
    )
    if declared_b_dec is None:
        chose_b_dec = (
            "not declared: this cfg.json has no apply_b_dec_to_input key. "
            "SAELens defaults an absent one to true in every schema it has "
            "written, so the release behaves as though it said true — but it "
            "did not say it, and the convention that actually runs is the "
            "measured one."
        )
    else:
        chose_b_dec = (
            f"declared: cfg.json says apply_b_dec_to_input is "
            f"{str(declared_b_dec).lower()}. The convention that actually "
            f"runs is the measured one, and the two are worth reading side by "
            f"side."
        )

    # The one declaration this module can repeat and cannot check. An SAE is
    # attached to whichever model the session has loaded, and nothing anywhere
    # compares that against the model it was trained on — `runtime` checks
    # d_in and d_in alone, which two unrelated models of the same width share.
    # So the name is carried through as a fact the release stated, in both of
    # its states, because "this cfg does not say" and "this cfg says gpt2" are
    # different things to know when the features look wrong.
    said_model = _declared(cfg, "model_name")
    chosen_by = {
        "hook": "caller",
        "architecture": chose_architecture,
        "apply_b_dec_to_input": chose_b_dec,
        "model": (
            f"declared: cfg.json says this SAE was trained on {said_model!r}. "
            f"Nothing here checks it against the loaded model — only that the "
            f"widths agree — so read it against what the session has open."
            if isinstance(said_model, str)
            else "not declared: this cfg.json names no model_name, so which "
            "model these features describe is unrecorded rather than assumed."
        ),
    }
    if chose_weights is not None:
        chosen_by["weights"] = chose_weights
    if chose_rescale is not None:
        chosen_by["rescale_acts_by_decoder_norm"] = chose_rescale

    return _Loaded(
        W_enc=W_enc,
        b_enc=b_enc,
        W_dec=W_dec,
        b_dec=b_dec,
        activation=activation,
        # None for every architecture that ships no thresholds, rather than a
        # zero vector: a threshold of zero is a gate that is always open,
        # which is what ReLU already does, and materialising d_sae zeros to
        # say so would cost megabytes to express nothing.
        threshold=threshold,
        declared_b_dec=declared_b_dec,
        k=k,
        b_gate=b_gate,
        r_mag=r_mag,
        b_mag=b_mag,
        release=SAERelease(
            repo=repo,
            layout=LAYOUT_SAE_LENS,
            file=None,
            layer=layer,
            point=point,
            width=None,
            advertised_l0=None,
            chosen_by=chosen_by,
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
        # Named, not implied by the presence of the fifth array. Gemma Scope
        # publishes JumpReLU SAEs and always has, so this reader has always
        # known the answer — it just used to say it by handing over a tensor
        # and letting `_activate` infer the rest, which is the habit that made
        # every SAELens release a plain ReLU.
        activation=ACT_JUMPRELU,
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


def _activate(
    pre: torch.Tensor,
    activation: str,
    *,
    threshold: torch.Tensor | None = None,
    k: int | None = None,
    gate_pre: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pre-activations -> feature activations. The one place the gate lives.

    Dispatches on the NAME of the activation, which the release declared, and
    never on which tensors turned up. That distinction is the whole fix: a
    TopK release ships exactly the tensors a standard one ships, so "no
    threshold, therefore ReLU" answered "plain ReLU" for a gate that keeps two
    features out of sixteen thousand.

    The four rules, each verbatim from the SAELens class that owns it:

    relu      relu(pre). What a standard release does, and what a JumpReLU
              whose bar is below every pre-activation degenerates to — the
              branches below are one function of the gate rather than two
              implementations that happen to agree.
    jumprelu  relu(pre) * (pre > threshold), the comparison STRICTLY greater
              and the threshold per feature. The relu is redundant on every
              release measured (every threshold is positive, so clearing one
              implies being positive) and is kept so a release with a negative
              threshold cannot emit negative "activations" through this line.
    topk      the k largest PRE-activations of each token survive and the relu
              is applied to those alone, so a selected feature whose
              pre-activation is negative is written as exactly zero and L0 can
              be less than k. Selection is per token, over the whole feature
              axis — which is why it is the one rule here that is not a
              function of one feature at a time.
    gated     a separate gating pre-activation decides IF a feature fires and
              the magnitude pre-activation decides how much. Both come from
              the same W_enc — the magnitude path scales it per feature by
              exp(r_mag) — and the caller computes them, because this is the
              one architecture whose rule needs two.
    """
    if activation == ACT_RELU:
        return torch.relu(pre)
    if activation == ACT_JUMPRELU:
        return torch.relu(pre) * (pre > threshold)
    if activation == ACT_TOPK:
        # `sorted=False` because the scatter writes each winner to its own
        # column, so the order they come back in cannot reach the result.
        # Ties at the k-th place are therefore harmless here in a way worth
        # stating: torch does not specify which of two exactly equal values it
        # keeps, but whichever loses is left as the zero it was initialised to
        # and whichever wins carries the same value, so the returned
        # activations are the same either way.
        values, indices = torch.topk(pre, k, dim=-1, sorted=False)
        # A freshly allocated buffer, scattered into. Not `pre.scatter_`: the
        # pre-activations can be a view of a captured forward pass — the
        # splice in `ce_recovered` runs this inside a live model — and an
        # in-place write there would corrupt the run being measured.
        return torch.zeros_like(pre).scatter_(-1, indices, torch.relu(values))
    if activation == ACT_GATED:
        return (gate_pre > 0).to(pre.dtype) * torch.relu(pre)
    # Unreachable through `load`, which refuses an architecture it cannot name
    # before any tensor is read. A plain ValueError rather than a Refusal
    # because reaching it means this module handed itself a name it does not
    # implement, which is a bug here and not a fact about anybody's release.
    raise ValueError(f"No gate implemented for activation {activation!r}")


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
    #: "relu", "jumprelu", "topk" or "gated" — the rule the release named,
    #: not one inferred from its tensors. None when nothing is loaded: an
    #: unloaded panel does not have a plain-ReLU SAE, it has no SAE.
    activation: str | None = None
    #: [min, max] of the JumpReLU thresholds. None for every architecture that
    #: has no thresholds at all, rather than thresholds of zero. Two measured
    #: numbers so the gate is visible as a fact about the loaded weights.
    threshold_span: list[float] | None = None
    #: How many features a top-k gate lets fire per token. None for every
    #: other architecture, which have no such number — not 0, which would read
    #: as a gate that fires nothing.
    k: int | None = None
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
        # None for a Gated release, which registers no b_enc at all — the same
        # answer `_Loaded.b_enc` carries. Annotated as optional because it IS
        # optional: a signature promising a tensor for the one architecture
        # that never ships one is a claim the loader contradicts on every
        # gated load, and the None-discipline this module is built on is worth
        # exactly as much as its weakest statement of it.
        b_enc: torch.Tensor | None,
        W_dec: torch.Tensor,
        b_dec: torch.Tensor,
        apply_b_dec_to_input: bool | None,
        threshold: torch.Tensor | None = None,
        release: SAERelease | None = None,
        activation: str = ACT_RELU,
        k: int | None = None,
        b_gate: torch.Tensor | None = None,
        r_mag: torch.Tensor | None = None,
        b_mag: torch.Tensor | None = None,
    ) -> None:
        self.repo, self.hook, self.layer = repo, hook, layer
        self.point = point  # resid_pre | resid_post
        self.W_enc, self.b_enc = W_enc, b_enc
        self.W_dec, self.b_dec = W_dec, b_dec
        # Which rule turns pre-activations into activations, by name. Defaults
        # to ReLU rather than to "whatever the tensors suggest", and that is
        # deliberate even though it means handing this a `threshold` without
        # naming JumpReLU gets a ReLU: the gate is a fact about the release,
        # `load` is the only constructor that reads a release, and inferring
        # it from which arguments arrived is the exact habit that made every
        # modern SAELens SAE load wide open.
        self.activation = activation
        # Per-feature JumpReLU gate, or None for an architecture that has no
        # thresholds. Not a zero vector for the ReLU case: "no gate" and "a
        # gate that never closes" are the same arithmetic but different facts.
        self.threshold = threshold
        #: How many features a top-k gate may fire per token. None elsewhere.
        self.k = k
        # The three tensors only a gated release carries. Its `b_enc` is None
        # rather than zeros — SAELens never registers one for this
        # architecture, so a zero vector would be a tensor nobody published.
        self.b_gate, self.r_mag, self.b_mag = b_gate, r_mag, b_mag
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
        repo: str,
        hook: str,
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
            activation=got.activation,
            k=got.k,
            b_gate=got.b_gate,
            r_mag=got.r_mag,
            b_mag=got.b_mag,
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
            activation=self.activation,
            k=self.k,
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

    def _encode_prepared(self, prepared: torch.Tensor) -> torch.Tensor:
        """Prepared activations -> features. The encoder, once, for everyone.

        `calibrate` and `encode` both end here, so the four conventions are
        scored through exactly the arithmetic that will run afterwards. The
        gate itself lives in `_activate`; what this adds is the ONE thing that
        differs before it, which is that a gated release has two encoder
        passes rather than one.

        Both of a gated SAE's pre-activations come from the same W_enc — the
        magnitude path scales it per feature by exp(r_mag) and adds b_mag, the
        gating path adds b_gate — so this is one matrix read twice and not two
        encoders.
        """
        if self.activation == ACT_GATED:
            return _activate(
                prepared @ (self.W_enc * self.r_mag.exp()) + self.b_mag,
                self.activation,
                gate_pre=prepared @ self.W_enc + self.b_gate,
            )
        return _activate(
            prepared @ self.W_enc + self.b_enc,
            self.activation,
            threshold=self.threshold,
            k=self.k,
        )

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
            feats = self._encode_prepared(self._prepare(x, center, subtract))
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
        return self._encode_prepared(self._prepare(x, cal.center, cal.subtract_b_dec))

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

        THE COLUMN TRICK DOES NOT WORK FOR TOP-K, and the saving is given up
        rather than the answer. ReLU, JumpReLU and Gated are elementwise in
        the feature index — every operation on feature f reads only f's
        column, f's biases and f's threshold — so one column really is one
        column and the equality with `encode(x)[:, f]` is exact. A top-k gate
        is a RANK STATISTIC over the whole row: feature f fires if its
        pre-activation is among the k largest of all d_sae, which is a
        question about the other d_sae - 1 features and cannot be answered
        from f's column at any price. There is no per-feature threshold
        standing in for it either, because the effective cutoff is the k-th
        largest pre-activation and that is a different number at every token.
        So top-k pays for the full row and takes its column, and the caller in
        `feature_ablate` pays 19 million multiply-adds a row instead of 768
        for that architecture alone. The alternative was returning a positive
        number for a feature the full encode zeroed, which would put a feature
        the SAE does not read into the honesty column that exists to say
        whether it reads it.
        """
        cal = self.calibration
        if cal is None:
            raise ValueError(
                "encode_feature needs a calibration; call encode() on the full "
                "activations first so the input convention is chosen on all of "
                "them rather than on one feature."
            )
        prepared = self._prepare(x.float(), cal.center, cal.subtract_b_dec)
        if self.activation == ACT_TOPK:
            return self._encode_prepared(prepared)[..., feature_id]
        column = self.W_enc[:, feature_id]
        if self.activation == ACT_GATED:
            # BOTH paths restricted, or the gate would be read off one feature
            # and the magnitude off another. Each is elementwise in the
            # feature index, so this is the same arithmetic as the full encode
            # with everything but column f left uncomputed.
            return _activate(
                prepared @ (column * self.r_mag[feature_id].exp())
                + self.b_mag[feature_id],
                self.activation,
                gate_pre=prepared @ column + self.b_gate[feature_id],
            )
        return _activate(
            prepared @ column + self.b_enc[feature_id],
            self.activation,
            # One scalar, not the whole [d_sae] vector: the gate is per
            # feature, so restricting the encoder to one column has to
            # restrict the threshold to the same one or the column would be
            # judged against 16,384 other features' thresholds.
            threshold=None if self.threshold is None else self.threshold[feature_id],
        )

    @torch.no_grad()
    def decode(self, feats: torch.Tensor) -> torch.Tensor:
        """[S, d_sae] feature activations -> [S, d_in] reconstruction."""
        return feats.float() @ self.W_dec + self.b_dec

    def steering_vector(self, feature_id: int) -> torch.Tensor:
        """Unit-norm decoder direction for one feature ([d_in])."""
        v = self.W_dec[feature_id]
        return v / (v.norm() + 1e-8)


# --------------------------------------------------- fidelity, in the output space
#
# Everything above measures this SAE in ACTIVATION space. `fvu` is the share of
# the stream's variance the reconstruction misses, `rel_err` the size of the
# residual, `l0` how many features it took to get there — the SAE's own claim,
# checked against the SAE's own input. Those numbers can all read well while
# the model built on top of them stops predicting, because FVU is taken against
# the directions carrying the VARIANCE of the residual stream and the
# directions the next token depends on are not the same directions.
#
# CE-recovered asks the output-space question instead: splice the
# reconstruction back in where the activation was, run the model again, and see
# how much of its predictive loss survives — normalised against a floor that
# says what destroying the activation costs in the first place.
#
#     CE_recovered = (CE_ablate - CE_recon) / (CE_ablate - CE_clean)
#
# ## The floor is half the number, and it is usually unsaid
#
# Mean-ablation and zero-ablation are both defensible and they are different
# denominators, so the same reconstruction scores a different percentage under
# each. A bare "97% recovered" does not say which was used and cannot be
# converted into the other one after the fact. So `floor` is a required
# argument with no default — there is no house answer to a question the reader
# has to be told the answer to — it comes back BY NAME in the payload, and all
# three raw losses come back beside the ratio so a reader holding the other
# convention can normalise these numbers themselves.
#
# ## The corpus is the other half
#
# CE is a loss ON SOME TEXT, exactly the way a feature's top activation is a
# top activation IN SOME CORPUS (feature_corpus.py makes this argument at
# length). The label, the sha256 of the token ids, the sequence count and the
# number of PREDICTED tokens travel with the ratio, because "this SAE recovers
# 97%" and "this SAE recovered 97% of the loss on the 4,000 tokens you handed
# it" are different claims and only the second one was measured.
#
# ## One splice, not two
#
# The reconstruction is written back through `feature_ablate`'s own hook
# helpers, with the arithmetic `feature_ablate` uses for its residual baseline
# — `x_recon[p] = recon[p] + mu[p]`, here with the window being every token.
# Two implementations of one intervention is how they drift, and `+ mu[p]` is
# exactly the part that would drift: when the calibrated convention centers,
# the SAE reconstructs `x - mean(x)` and the stream handed back to the model is
# that reconstruction plus the ORIGINAL per-token mean.
#
# ## The convention is the one calibration already chose
#
# `SAECalibration` picks between four input conventions by lowest FVU, and this
# uses that choice rather than making its own. Two numbers on one panel taken
# under two conventions describe two different splices, and nothing in either
# number would say so.

#: Replace the activation with the mean activation vector measured over this
#: corpus at this hook, or with zeros. Both are published choices; neither is
#: the default, because the reader has to be told which one produced the
#: percentage they are reading.
FLOOR_MEAN = "mean_ablate"
FLOOR_ZERO = "zero_ablate"
CE_FLOORS: tuple[str, ...] = (FLOOR_MEAN, FLOOR_ZERO)

# Positions per chunk when the next-token loss is taken. A pass's logits are
# [S, vocab] in the model's dtype and `log_softmax` needs them in float32: on
# Qwen3-1.7B's 151,936-entry vocabulary that is 607,744 bytes per position, so
# a 512-token sequence upcast whole would materialise 311 MB beside the logits
# the model already produced. Sixty-four positions is 38.9 MB. That is
# arithmetic rather than a measurement, and it is why the number is a named
# constant rather than a comment about being careful.
NLL_CHUNK_TOKENS = 64

# Below this, the denominator `CE_ablate - CE_clean` is not a measurement of
# anything and the ratio divides by noise. Same basis as
# `feature_ablate.RESOLUTION_KL`: these losses are float32 sums over a
# vocabulary, and a difference of that size is the summation showing itself.
# It is a FLOOR under the real test, which is the write-back deviation measured
# on the corpus in hand — see `ce_recovered`.
MIN_DENOMINATOR_NATS = 1e-6

# How far writing the model's own residual stream back into the hook unchanged
# may move its loss. The reconstruction and the floor go in through that same
# hook, so if the write-back does not land — a resid_post SAE hooked on a
# block's input, a block that runs twice — all three losses are measuring the
# write-back. Same tolerance and same reasoning as
# `feature_ablate.WRITEBACK_TOLERANCE`, and like that one it is compared
# against a no-hook replay, so a nondeterministic accelerator raises its own
# bar rather than tripping this.
SPLICE_TOLERANCE_NATS = 1e-6


@dataclass
class CEFidelity:
    """CE-recovered, the three losses under it, and what they are losses OF.

    The ratio is not the first field on purpose. It is the only number here
    that depends on a choice the reader did not make — the floor — and the
    three losses it is built from are all present so that choice can be undone.
    """

    repo: str
    hook: str
    layer: int
    point: str

    #: "mean_ablate" or "zero_ablate", by name, because the percentage is not
    #: interpretable without it.
    floor: str
    floor_means: str
    #: Tokens the mean floor vector was averaged over. None for the zero floor,
    #: which is not averaged from anything — never 0, which would claim a mean
    #: taken over an empty corpus.
    n_floor_tokens: int | None

    #: Nats per predicted token, all three, on the corpus named below.
    ce_clean: float
    ce_recon: float
    ce_ablate: float
    numerator: float  # ce_ablate - ce_recon
    denominator: float  # ce_ablate - ce_clean
    #: NOT clamped, in either direction. Below zero is a real answer and it
    #: means the reconstruction predicts worse than destroying the activation
    #: does; clamping it to 0 would report a broken SAE as a merely useless one.
    ce_recovered: float

    corpus_label: str
    corpus_sha256: str
    n_sequences: int
    n_sequences_given: int
    truncated: bool
    #: Tokens the loss was taken over. The first token of a sequence is fed and
    #: never predicted, so this is `sum(len(seq) - 1)` and not the number of
    #: tokens that went in — which is `n_tokens_seen`, beside it.
    n_tokens: int
    n_tokens_seen: int

    #: Which input convention the splice used, and how well it reconstructs in
    #: activation space. Carried whole rather than summarised: a CE-recovered
    #: taken in a different convention is a different measurement, and `fvu`
    #: and `l0` beside it are the activation-space half of the same panel.
    calibration: SAECalibration
    #: True when this run calibrated the SAE, on the first sequence of this
    #: corpus. False when it reused a calibration taken somewhere else, in
    #: which case `calibration.n_tokens` says what that one was measured on.
    calibrated_here: bool

    #: What the same pass costs run twice with no hook at all, and what writing
    #: the captured stream back unchanged costs, both in nats per token on the
    #: first sequence. The second is the resolution of every difference above.
    replay_deviation_nats: float
    splice_deviation_nats: float

    passes: int
    elapsed_s: float

    def means(self) -> str:
        return (
            f"CE-recovered {self.ce_recovered:.4f} for {self.repo} at "
            f"{self.hook}, against the {self.floor} floor, measured on "
            f"{self.corpus_label}: {self.n_tokens:,} predicted tokens in "
            f"{self.n_sequences:,} sequences. The three losses it is built "
            f"from, in nats per token — the model's own {self.ce_clean:.6f}, "
            f"with the SAE's reconstruction spliced in {self.ce_recon:.6f}, "
            f"with the activation replaced by the floor {self.ce_ablate:.6f}. "
            f"{self.floor_means} A different floor gives a different "
            f"percentage from these same three losses, which is why all three "
            f"are here. The reconstruction was taken in the "
            f"'{self.calibration.convention}' input convention — the one "
            f"calibration measured the lowest FVU ({self.calibration.fvu:g}) "
            f"for, on {self.calibration.n_tokens:,} tokens — and a figure "
            f"taken in another convention describes another splice. Nothing "
            f"is clamped: below zero would mean the reconstruction predicts "
            f"worse than the floor does. The denominator this is divided by is "
            f"{self.denominator:.6f} nats/token, against the "
            f"{self.splice_deviation_nats:.3e} that writing the model's own "
            f"stream back unchanged moved the same loss by."
            + (
                f" {self.n_sequences:,} of {self.n_sequences_given:,} "
                f"sequences were scored; the rest were NOT MEASURED, which is "
                f"not the same as measured and found not to matter."
                if self.truncated
                else ""
            )
        )

    def to_dict(self) -> dict:
        return {**asdict(self), "means": self.means()}


def ce_recovered_passes(n_sequences: int) -> int:
    """Exactly what `ce_recovered` will spend, in forward passes: `3n + 2`.

    Three per sequence — the model's own pass, which also captures the stream
    the other two are built from; the reconstruction spliced in; the floor
    spliced in — plus two taken once on the first sequence: a plain replay with
    no hook at all, and the captured stream written back unchanged. Those two
    are the resolution every difference is read against, and they are the only
    reason this is not `3n`.

    Exact and portable, which is the half of "what will this cost" that
    transfers between machines. `estimate_ce_recovered_cost` is the other half.
    """
    # `isinstance(True, int)` is True, so a stray flag would price a two-pass
    # run and be reported as a corpus of one sequence.
    if isinstance(n_sequences, bool):
        raise BadRequest("n_sequences is a count of sequences, not a flag.")
    n_sequences = int(n_sequences)
    if n_sequences < 1:
        raise BadRequest(
            "CE-recovered needs at least one sequence to be a loss on "
            "anything; there is nothing to price for a corpus of 0."
        )
    return 3 * n_sequences + 2


def _sum_nll(
    logits: torch.Tensor, ids: torch.Tensor, *, chunk: int = NLL_CHUNK_TOKENS
) -> tuple[float, int]:
    """Summed next-token loss over one `[1, S]` sequence, and the tokens in it.

    A SUM and a count rather than a mean, because the caller accumulates over a
    corpus of unequal sequences and a mean of means would weight a six-token
    sequence like a six-hundred-token one.

    Position t's logits are scored against token t+1, so the first token of a
    sequence is fed and never predicted — which is why the count returned is
    `S - 1`. Summed in float64: a float32 running total over a corpus loses the
    low bits of every late token, and the quantity this file reports is a
    DIFFERENCE between two such totals.
    """
    targets = ids[0, 1:]
    n = int(ids.shape[1]) - 1
    total = 0.0
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        # float32 for the softmax, on a slice, for the reason
        # NLL_CHUNK_TOKENS is a named constant.
        log_probs = torch.log_softmax(logits[0, start:stop].float(), dim=-1)
        want = targets[start:stop].to(log_probs.device).unsqueeze(-1)
        total += float(log_probs.gather(-1, want).squeeze(-1).double().sum())
        del log_probs, want
    return -total, n


def _corpus_sha256(sequences: list[torch.Tensor]) -> str:
    """A stable identity for the exact token ids that were scored.

    The label is what the reader calls the corpus; this is what actually ran,
    so two results carrying the same label over different text are
    distinguishable. Hashed one sequence at a time with a separator between
    them, so [[1, 2], [3]] and [[1], [2, 3]] do not collide.
    """
    digest = hashlib.sha256()
    for ids in sequences:
        digest.update(b"\x00")
        digest.update(
            ",".join(str(int(t)) for t in ids.flatten().tolist()).encode("ascii")
        )
    return digest.hexdigest()


def _sequence_for_ce(value: object, index: int) -> torch.Tensor:
    """One `[1, S]` id tensor, or a sentence saying what arrived instead."""
    if not isinstance(value, torch.Tensor):
        raise BadRequest(
            f"sequence {index} is a {type(value).__name__}, not a tensor of "
            f"token ids. CE-recovered scores the model's own loss on ids it "
            f"can be fed; tokenize the text first."
        )
    ids = value.unsqueeze(0) if value.dim() == 1 else value
    if ids.dim() != 2 or int(ids.shape[0]) != 1:
        raise BadRequest(
            f"sequence {index} is shaped {tuple(value.shape)}; CE-recovered "
            f"reads one unbatched sequence at a time, [S] or [1, S]. Batching "
            f"pads, and a padded position contributes a loss on a token nobody "
            f"wrote."
        )
    if int(ids.shape[1]) < 2:
        raise BadRequest(
            f"sequence {index} is {int(ids.shape[1])} token(s) long. The first "
            f"token of a sequence is fed and never predicted, so a sequence "
            f"shorter than two tokens contributes no loss at all — it would be "
            f"counted in the corpus and score nothing."
        )
    return ids


@torch.no_grad()
def ce_recovered(
    model,
    block: torch.nn.Module,
    sequences,
    sae: SAEHandle,
    *,
    floor: str,
    corpus_label: str,
    max_sequences: int | None = None,
) -> CEFidelity:
    """How much of the model's predictive loss survives this SAE's reconstruction.

    `block` is the module the SAE is attached to (`runtime._block(sae.layer)`),
    `sequences` an ordered collection of token-id tensors — `[S]` or `[1, S]`,
    already on the model's device — and `floor` one of `CE_FLOORS`, by name,
    with no default.

    Three passes per sequence and two more on the first; the exact figure is
    `ce_recovered_passes(n)` and it can be asked BEFORE this is called.

    **Two sweeps, and the floor is the reason.** Mean-ablation replaces the
    activation with one vector averaged over the whole corpus, which is not
    known until every sequence has been read once. Sweep one takes the model's
    own loss, captures the stream, splices the reconstruction and accumulates
    that mean; sweep two splices the floor. `sequences` is therefore read twice
    and has to be a re-iterable collection rather than a generator. What is
    held between the sweeps is the token ids and one `[d_in]` accumulator —
    never the activations, which are recomputed rather than cached, because a
    corpus of those is `tokens x d_model x 4` bytes against eight bytes a token
    for the ids.

    **An SAE that reconstructs nothing is not refused here**, and that is the
    one place this parts company with `feature_ablate.rank_features`. That
    function refuses an unusable calibration because ranking the causal effects
    of a non-decomposition ranks arbitrary directions. Here a broken SAE has a
    real answer — a CE-recovered at or below zero — and refusing to print it
    would hide exactly the finding this measurement exists to make.
    `calibration.usable` and `calibration.fvu` come back either way.
    """
    started = time.perf_counter()
    passes = 0

    if floor not in CE_FLOORS:
        raise BadRequest(
            f"unknown floor {floor!r} — CE-recovered normalises against one of "
            f"{', '.join(CE_FLOORS)}, and there is no default because the two "
            f"give different percentages for the same reconstruction. "
            f"'{FLOOR_MEAN}' replaces the activation with the mean vector of "
            f"this corpus at this hook; '{FLOOR_ZERO}' replaces it with zeros."
        )
    if not corpus_label or not str(corpus_label).strip():
        raise BadRequest(
            "CE-recovered is a loss on some text, so the text has to be named. "
            "Pass corpus_label — the file name, or whatever the reader will "
            "recognise it by."
        )
    if isinstance(max_sequences, bool):
        raise BadRequest("max_sequences is a count of sequences, not a flag.")

    given = [_sequence_for_ce(s, i) for i, s in enumerate(sequences)]
    if not given:
        raise BadRequest(
            "CE-recovered needs at least one sequence — an empty corpus has no "
            "loss to recover."
        )
    if max_sequences is None:
        used = given
    else:
        limit = int(max_sequences)
        if limit < 1:
            raise BadRequest(
                f"max_sequences is {limit}; it caps how much of the corpus is "
                f"scored, so it has to be at least 1 or there is nothing left "
                f"to score."
            )
        used = given[:limit]
    truncated = len(used) < len(given)

    # Local import for the reason `ablate.estimate_cost` imports `budget`
    # locally: this is the only place in this module that needs the model-side
    # machinery, and `saes.py` is imported by plenty of code that never runs a
    # forward pass. The hooks are IMPORTED rather than rewritten —
    # `feature_ablate` already owns "write a tensor into this block's residual
    # stream", it already knows resid_pre is the block's input and resid_post
    # its output, and a second copy here would be a second answer to that.
    from .feature_ablate import _register_capture, _register_edit

    device = None
    dtype = None
    clean_nll = 0.0
    recon_nll = 0.0
    ablate_nll = 0.0
    n_tokens = 0
    n_tokens_seen = 0
    floor_sum = torch.zeros(sae.d_in, dtype=torch.float64)
    n_floor_tokens = 0
    calibrated_here = sae.calibration is None
    replay_deviation: float | None = None
    splice_deviation: float | None = None

    def spliced(ids: torch.Tensor, edited: torch.Tensor) -> tuple[float, int]:
        """One pass with `edited` ([S, d_in] cpu float32) written into the hook."""
        nonlocal passes
        xd = edited.to(device=device, dtype=dtype).unsqueeze(0)
        handle = _register_edit(block, sae.point, xd)
        try:
            out = model(ids).logits
        finally:
            handle.remove()
        passes += 1
        got = _sum_nll(out, ids)
        del out
        return got

    # ---- sweep one: the model's own loss, the reconstruction's, and the mean
    for index, ids in enumerate(used):
        sink: list[torch.Tensor] = []
        handle = _register_capture(block, sae.point, sink)
        try:
            logits = model(ids).logits
        finally:
            handle.remove()
        passes += 1
        if not sink:
            # A plain RuntimeError and a 500, for the reason `feature_ablate`
            # gives the same event: the caller hands in the block, `runtime`
            # builds it from the SAE's own layer index, so a hook that never
            # fires means this package contradicted itself and the traceback
            # belongs in the log rather than in front of a reader.
            raise RuntimeError(
                "the capture hook on the SAE's block never fired, so there is "
                "no residual stream to reconstruct."
            )
        captured = sink[0]
        device, dtype = captured.device, captured.dtype
        x = captured[0].detach().to("cpu").float()  # [S, d_in]
        if x.shape[-1] != sae.d_in:
            raise RuntimeError(
                f"captured a stream of width {x.shape[-1]} for an SAE with "
                f"d_in={sae.d_in}; runtime.py checks this at load time, so "
                "reaching it means the wrong block was handed in."
            )
        nll, n = _sum_nll(logits, ids)
        clean_nll += nll
        n_tokens += n
        n_tokens_seen += int(ids.shape[1])
        del logits, sink, captured

        # `encode` calibrates on first use — here on this corpus's first
        # sequence — and reuses an existing calibration otherwise. Which of the
        # two happened is `calibrated_here` in the payload, rather than
        # something the reader has to infer from `calibration.n_tokens`.
        feats = sae.encode(x)
        # feature_ablate.py's residual baseline with the window being every
        # token: `x_recon[p] = recon[p] + mu[p]`. `mu` is the per-token mean
        # the calibrated convention removed, and it is HELD rather than
        # recomputed — the reconstruction lives in the centered space, and the
        # stream handed back to the model is that plus the ORIGINAL mean. For
        # an uncentered convention it is zero and this is the reconstruction
        # alone.
        cal = sae.calibration
        assert cal is not None  # encode() always leaves one behind
        mu = x.mean(-1, keepdim=True) if cal.center else torch.zeros_like(x[:, :1])
        x_recon = sae.decode(feats) + mu
        del feats

        if index == 0:
            # The two passes that are not per-sequence. `replay` is the same
            # pass again with no hook at all — this model's own reproducibility
            # — and the write-back is the captured stream spliced back
            # unchanged, which is the only check that the edit lands where the
            # capture came from. Every difference below is read against them.
            replay_logits = model(ids).logits
            passes += 1
            replay_nll, _ = _sum_nll(replay_logits, ids)
            del replay_logits
            replay_deviation = abs(replay_nll - nll) / n
            writeback_nll, _ = spliced(ids, x.clone())
            splice_deviation = abs(writeback_nll - nll) / n

        recon_nll += spliced(ids, x_recon)[0]

        if floor == FLOOR_MEAN:
            # Accumulated in float64 without materialising a float64 copy of
            # `x`. This vector REPLACES the stream rather than being subtracted
            # from it, so its low bits are the floor's low bits.
            floor_sum += torch.sum(x, dim=0, dtype=torch.float64)
            n_floor_tokens += int(x.shape[0])
        del x, x_recon

    calibration = sae.calibration
    assert calibration is not None
    assert replay_deviation is not None and splice_deviation is not None

    if splice_deviation > max(replay_deviation, SPLICE_TOLERANCE_NATS):
        raise Refusal(
            f"writing this model's own residual stream back into {sae.hook} "
            f"unchanged moves its loss by {splice_deviation:.3e} nats per "
            f"token, and running the same pass again with no hook at all moves "
            f"it {replay_deviation:.3e}. The reconstruction and the floor go in "
            f"through that same hook, so all three losses would be measuring "
            f"the write-back rather than the SAE. It would work at a hook point "
            f"whose input can be replaced by the tensor read out of it."
        )

    if floor == FLOOR_MEAN:
        floor_vector = (floor_sum / n_floor_tokens).float()
        floor_means = (
            f"The floor is mean-ablation: at every token the stream at "
            f"{sae.hook} was replaced by a single vector, the mean of this "
            f"corpus's own {n_floor_tokens:,} activations there."
        )
        floor_tokens: int | None = n_floor_tokens
    else:
        floor_vector = torch.zeros(sae.d_in)
        floor_means = (
            f"The floor is zero-ablation: at every token the stream at "
            f"{sae.hook} was replaced by the zero vector, which is not a point "
            f"this stream ever visits."
        )
        floor_tokens = None

    # ---- sweep two: the floor, now that it is known ------------------------
    for ids in used:
        # `repeat` rather than `expand`: this tensor is handed to the model's
        # own kernels and a stride-0 view is not something every backend reads
        # the same way. One [S, d_in] float32 copy per sequence, freed with it.
        ablate_nll += spliced(ids, floor_vector.unsqueeze(0).repeat(ids.shape[1], 1))[0]

    ce_clean = clean_nll / n_tokens
    ce_recon = recon_nll / n_tokens
    ce_ablate = ablate_nll / n_tokens
    denominator = ce_ablate - ce_clean
    numerator = ce_ablate - ce_recon

    resolvable = max(replay_deviation, splice_deviation, MIN_DENOMINATOR_NATS)
    if denominator <= resolvable:
        # THE REFUSAL. At or below this the floor costs the model nothing this
        # run can resolve — the hook point does not matter for this text — and
        # the ratio becomes a small difference over a smaller one, which is
        # noise amplified without limit. A NEGATIVE denominator lands here too,
        # and it is the same statement said louder: destroying the activation
        # made the model predict better, so there is no lost loss to take a
        # percentage of. The three losses are real measurements and they are in
        # the sentence, because they are still the answer to a question the
        # reader can ask.
        raise Refusal(
            f"CE-recovered has no meaning for {sae.repo} at {sae.hook} on "
            f"{corpus_label}: replacing the activation with the {floor} floor "
            f"moves this model's loss by {denominator:+.3e} nats per token, at "
            f"or under the {resolvable:.3e} this run can resolve, so the ratio "
            f"would divide by noise. The three losses are real and here they "
            f"are, in nats per token over {n_tokens:,} predicted tokens — the "
            f"model's own {ce_clean:.6f}, with the reconstruction spliced in "
            f"{ce_recon:.6f}, with the floor spliced in {ce_ablate:.6f}. "
            f"Measure at a hook point this text's predictions depend on, or on "
            f"a corpus where they do."
        )

    return CEFidelity(
        repo=sae.repo,
        hook=sae.hook,
        layer=sae.layer,
        point=sae.point,
        floor=floor,
        floor_means=floor_means,
        n_floor_tokens=floor_tokens,
        ce_clean=round(ce_clean, 6),
        ce_recon=round(ce_recon, 6),
        ce_ablate=round(ce_ablate, 6),
        numerator=round(numerator, 6),
        denominator=round(denominator, 6),
        # Six places on the ratio, and it is computed from the UNROUNDED
        # losses: a reconstruction that recovers 0.999999 and one that recovers
        # 1.0 exactly are different objects, and the identity case has to come
        # back exact or the splice is not checkable at all.
        ce_recovered=round(numerator / denominator, 6),
        corpus_label=str(corpus_label),
        corpus_sha256=_corpus_sha256(used),
        n_sequences=len(used),
        n_sequences_given=len(given),
        truncated=truncated,
        n_tokens=n_tokens,
        n_tokens_seen=n_tokens_seen,
        calibration=calibration,
        calibrated_here=calibrated_here,
        replay_deviation_nats=replay_deviation,
        splice_deviation_nats=splice_deviation,
        passes=passes,
        elapsed_s=round(time.perf_counter() - started, 2),
    )


@torch.no_grad()
def estimate_ce_recovered_cost(
    model,
    block: torch.nn.Module,
    ids: torch.Tensor,
    sae: SAEHandle,
    *,
    n_sequences: int,
    device_kind: str = "cpu",
) -> dict:
    """What would `ce_recovered` cost here? The pass count exactly, the rest measured.

    `ce_recovered_passes` is exact and transfers between machines. What a pass
    costs does not, so this runs ONE real iteration — the edit hook, the
    forward, the chunked float32 loss — and lets `budget` project from it. The
    probe body is built here rather than by the caller because it has to match
    the loop: `budget.probe_pass` records what happens when a probe does less
    work than the loop it prices.

    `ids` is one representative sequence, and representative means the LENGTH —
    a pass over 64 tokens does not price a pass over 512. This spends three
    passes of its own: a warm-up, a capture, and the probe.

    Retained bytes are what the loop holds across passes: the captured stream
    and the reconstruction built from it, both `[S, d_in]` float32, plus the
    float64 accumulator behind the mean floor. Stated from the shapes rather
    than measured, because it is arithmetic and not an observation.
    """
    from . import budget
    from .feature_ablate import _register_capture, _register_edit

    projected = ce_recovered_passes(n_sequences)

    # Warm the kernels before anything is timed. The first pass after a load
    # pays device init and measured several times the steady rate elsewhere in
    # this package, so probing it would price the whole sweep at that rate.
    model(ids)

    sink: list[torch.Tensor] = []
    handle = _register_capture(block, sae.point, sink)
    try:
        model(ids)
    finally:
        handle.remove()
    if not sink:
        raise RuntimeError(
            "the capture hook on the SAE's block never fired, so there is no "
            "pass to price."
        )
    xd = sink[0][0].detach().unsqueeze(0)

    def one_iteration() -> None:
        edit = _register_edit(block, sae.point, xd)
        try:
            logits = model(ids).logits
        finally:
            edit.remove()
        _sum_nll(logits, ids)

    seq = int(ids.shape[1])
    probe = budget.probe_pass(one_iteration, device_kind)
    estimate = budget.project(
        probe,
        projected,
        retained_bytes=2 * seq * sae.d_in * 4 + sae.d_in * 8,
    )
    return {
        "estimate": estimate.to_dict(),
        "probe": probe.to_dict(),
        "passes": projected,
        "n_sequences": int(n_sequences),
        "probed_sequence_length": seq,
        "means": (
            f"{projected} forward passes — three per sequence (the model's "
            f"own, the reconstruction spliced in, the floor spliced in) plus "
            f"two taken once for the resolution. The count is exact and it "
            f"transfers; the seconds were measured from one pass over {seq} "
            f"tokens on this machine and do not."
        ),
    }
