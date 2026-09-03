# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Which sparse autoencoders exist, and which model each one fits.

A SAE is trained against one model's residual stream at one layer. There is no
such thing as a general one, and there is no way to make a model's features
appear if nobody has trained an autoencoder for it — that is GPU-months of
someone else's work, not a feature this tool can implement. Public SAEs exist
for only a handful of models; the table below is what this build knows of.

So this is a lookup table, not a capability. Its job is to stop you typing
repository names, and to say plainly when the answer for your model is "none
yet" rather than leaving the panel looking broken.

**The table is a convenience; the guarantee is elsewhere.** Nothing here is
trusted: SAEHandle.load refuses any SAE whose `d_in` does not equal the loaded
model's `hidden_size`, so a wrong entry fails loudly at load rather than
producing confident features describing the wrong model. That check is the
reason it is safe to ship a hand-maintained list at all.

**What this table does NOT hold is the release index.** Gemma Scope publishes
312 residual-stream SAEs for gemma-2-2b — 26 layers crossed with the
dictionary widths and average-L0 sparsities trained at each — and which ones
exist differs per layer. Writing one of them down here would be picking a
sparsity on the reader's behalf and calling it "the Gemma Scope SAE", which is
the kind of invisible choice this project treats as a defect. `indexed_by`
names the coordinates instead, and `saes.release_index` reads the real list off
the Hub at load time so it cannot go stale in a source file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SAEEntry:
    repo: str
    #: HuggingFace ids this SAE was trained against, lowercased.
    models: tuple[str, ...]
    d_in: int
    layers: tuple[int, ...]
    point: str  # resid_pre | resid_post
    label: str
    #: Which layer the one-click Load button opens, when the middle of
    #: `layers` is not it. `to_dict` derives `default_hook` from the middle
    #: otherwise, and for a repo whose releases are equally unexamined that
    #: rule is honest — it picks a layer nobody has looked at and says so by
    #: being a rule. It is the wrong answer for a repo this project has
    #: PUBLISHED numbers against: the button carries this row's label, so a
    #: reader pressing it is told they are getting the thing the README
    #: describes, and the middle of the range is not that thing.
    #: None means "use the middle", which stays the answer for every row with
    #: no measured layer of its own.
    default_layer: int | None = None
    #: Which on-disk layout `saes.SAEHandle.load` has to open. One of
    #: `saes.LAYOUT_SAE_LENS` (cfg.json + sae_weights.safetensors per hook) or
    #: `saes.LAYOUT_GEMMA_SCOPE` (params.npz per layer/width/average L0).
    #: Advisory: the loader detects the layout from the repo rather than from
    #: this field, so a wrong value here cannot open the wrong reader.
    layout: str = "sae_lens"
    #: Coordinates a release needs BEYOND the layer, in the order a picker
    #: should present them. Empty means the hook name is the whole address.
    #: The VALUES are not listed here on purpose — see the module docstring.
    indexed_by: tuple[str, ...] = ()
    #: Whether a release from this repo has been loaded and run here. Not
    #: "whether the format is readable": the Gemma Scope reader is one code
    #: path shared by every repo using that layout, so a `False` beside a
    #: `gemma_scope` layout means unverified, and the note says so.
    supported: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        """The three ways a hand-typed row can be wrong before anyone loads it.

        ValueError rather than Refusal, and at import time rather than at
        request time, because none of these is a reader's mistake or a
        missing measurement — they are a row in this file contradicting
        itself, and the person who can act on the sentence is the one editing
        the table. A Refusal here would put a maintainer's typo in front of a
        user as though the Hub had let them down.
        """
        if not self.layers:
            raise ValueError(
                f"{self.repo} lists no layers, so there is no hook to offer "
                f"and no default to derive."
            )
        if self.default_layer is not None and self.default_layer not in self.layers:
            raise ValueError(
                f"{self.repo} names layer {self.default_layer} as its default "
                f"but does not list it among the layers it publishes. The "
                f"default is what the one-click button downloads, so a layer "
                f"the repo has no release for is a button that can only 404."
            )
        wrong_case = [m for m in self.models if m != m.lower()]
        if wrong_case:
            raise ValueError(
                f"{self.repo} names {', '.join(wrong_case)} with capitals. "
                f"`for_model` lowercases what it is ASKED and compares it "
                f"against these verbatim, so a capital here is a row that can "
                f"never match anything."
            )

    def hook(self, layer: int) -> str:
        return f"blocks.{layer}.hook_{self.point}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["models"] = list(self.models)
        d["layers"] = list(self.layers)
        d["indexed_by"] = list(self.indexed_by)
        layer = (
            self.default_layer
            if self.default_layer is not None
            else self.layers[len(self.layers) // 2]
        )
        d["default_hook"] = self.hook(layer)
        return d


REGISTRY: tuple[SAEEntry, ...] = (
    SAEEntry(
        repo="jbloom/GPT2-Small-SAEs-Reformatted",
        models=("gpt2", "openai-community/gpt2"),
        d_in=768,
        # Read off the repo's own file listing rather than assumed from the
        # model's depth: it publishes `blocks.0..11.hook_resid_pre`, which is
        # every layer, AND one `blocks.11.hook_resid_post`. That last one
        # cannot be named by a row whose `point` is a single string, so this
        # row does not offer it — it is still loadable by naming the hook
        # outright, through `/api/sae/load` or `--hook`. Said here rather than
        # left as an unexplained absence, because a reader counting 13 release
        # directories against 12 layers deserves the reason.
        layers=tuple(range(12)),
        point="resid_pre",
        # NOT the middle of the range. Every figure this project publishes for
        # this SAE — the README's demo line, the convention table in saes.py,
        # feature_ablate.py's worked example — was measured at layer 8, and a
        # button labelled with this row that downloaded layer 6 would be
        # offering 151 MB of a release nothing here describes. An earlier
        # version of this row had no such field and did exactly that.
        default_layer=8,
        label="SAELens · GPT-2 small · residual stream",
        # EVERY FIGURE BELOW NAMES THE TEXT IT WAS TAKEN ON, and that is the
        # whole reason this note is as long as it is. An earlier version gave
        # the token counts alone — "on 19 tokens", "over 54 predicted tokens"
        # — while its own closing sentence said the figures move with the
        # corpus. Both statements were true, and together they described a
        # measurement nobody could repeat: re-running it here reproduced the
        # model and SAE facts exactly and none of the four calibration
        # numbers, because 19 tokens of one text are not 19 tokens of
        # another. A receipt that cannot be redeemed is not a receipt.
        note="Loaded and run here: blocks.8.hook_resid_pre against gpt2 "
        "(124,439,808 params, hidden_size 768), float32 on cuda. The SAE "
        "reported d_in 768 and d_sae 24576, and its cfg.json names no "
        "architecture at all — a pre-6.0 SAELens file, which resolves to "
        "standard ReLU, so there is no threshold and no k to gate with, and "
        "no apply_b_dec_to_input key either. Calibrated on the prompt the "
        'convention table in saes.py names, "The Eiffel Tower is located in '
        'the city of" (11 tokens), it chose the centered+b_dec input '
        "convention at FVU 0.000984 and 60.55 features per token, against "
        "0.421870 centered, 12908.35 b_dec and 13579.24 raw — the four "
        "figures that table records, re-measured. Spliced back into the "
        "model it recovers 0.878486 of what the mean_ablate floor costs: "
        "4.052836 nats per token clean, 4.520641 reconstructed, 7.902623 "
        "ablated, over 54 predicted tokens in 4 sequences (58 tokens seen, "
        "corpus sha256 4e38eed2…), with writing the model's own stream back "
        "unchanged moving that loss by 0.000e+00. Against the zero_ablate "
        "floor the same reconstruction scores 0.937484, which is why the "
        "floor is named rather than assumed. That corpus, in full, because a "
        'token count is not a corpus: "The Eiffel Tower is located in the '
        'city of Paris." / "A sparse autoencoder is trained against one '
        'model at one layer." / "The capital of Japan is Tokyo, and its '
        'largest port is Yokohama." / "A reconstruction that explains the '
        'variance can still cost the model its predictions."',
    ),
    SAEEntry(
        repo="google/gemma-scope-2b-pt-res",
        models=("google/gemma-2-2b", "google/gemma-2-2b-it"),
        d_in=2304,
        layers=tuple(range(26)),
        point="resid_post",
        label="Gemma Scope · gemma-2-2b · residual stream",
        # Both coordinates, together. `layout` is what tells a picker this
        # release is not addressed by its hook name alone, and `indexed_by`
        # is what it must ask for instead; a row carrying one without the
        # other describes a release nobody can address. Which widths and
        # which sparsities exist is deliberately absent — see the module
        # docstring, and `saes.release_index`, which reads them off the Hub.
        layout="gemma_scope",
        indexed_by=("width", "average_l0"),
        note="Loaded and run here: layer 20, width_16k, average_l0 71 against "
        "google/gemma-2-2b (2,614,341,888 params, hidden_size 2304). The SAE "
        "reported d_in 2304, d_sae 16384 and a JumpReLU threshold span of "
        "4.516486 to 30.225666. Widths and sparsities are read from the "
        "repo's own listing, not assumed.",
    ),
    SAEEntry(
        repo="google/gemma-scope-9b-pt-res",
        models=("google/gemma-2-9b", "google/gemma-2-9b-it"),
        d_in=3584,
        layers=tuple(range(42)),
        point="resid_post",
        label="Gemma Scope · gemma-2-9b · residual stream",
        # The same pair as the 2b row above, and for the same reason: this is
        # the same .npz layout, published across the same two coordinates.
        layout="gemma_scope",
        indexed_by=("width", "average_l0"),
        # `False` here means UNVERIFIED, not unreadable — see the field's own
        # docstring. It is the same layout and the same code path as the 2b
        # release above, which has been run; this one has not, because
        # gemma-2-9b does not fit the 8.6 GB card it would have been run on.
        # Claiming it works on that basis would be exactly the kind of
        # untested assertion the flag exists to prevent.
        supported=False,
        note="Same .npz layout as the 2b release, read by the same code path, "
        "which has been verified against that release. This one has not been "
        "run here — gemma-2-9b needs more memory than the machine it would "
        "have been tested on. Expected to work; not measured.",
    ),
    SAEEntry(
        repo="EleutherAI/sae-llama-3-8b-32x",
        models=("meta-llama/meta-llama-3-8b", "meta-llama/meta-llama-3-8b-instruct"),
        d_in=4096,
        layers=tuple(range(32)),
        point="resid_post",
        label="EleutherAI · Llama-3-8B · residual stream",
        # `layout` is left at its default here, and that default is not a
        # claim about this repo. There are two values, both naming a reader
        # `saes` actually has, and this release is neither — inventing a third
        # would put a string in the payload that no code path can dispatch on
        # and that a picker would have to guess the meaning of. The note is
        # where the real answer lives, and `supported=False` is what stops
        # anything acting on the field.
        supported=False,
        note="EleutherAI's sae layout differs from SAELens; not opened yet.",
    ),
)


def for_model(hf_id: str | None) -> list[dict]:
    """Entries trained against this model. Empty is a real, common answer."""
    if not hf_id:
        return []
    key = hf_id.lower()
    short = key.split("/")[-1]
    out = []
    for entry in REGISTRY:
        if key in entry.models or any(m.split("/")[-1] == short for m in entry.models):
            out.append(entry.to_dict())
    return out


def catalogue() -> list[dict]:
    """Everything known, for the "what else is out there" list."""
    return [e.to_dict() for e in REGISTRY]
