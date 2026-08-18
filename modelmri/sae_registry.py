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

    def hook(self, layer: int) -> str:
        return f"blocks.{layer}.hook_{self.point}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["models"] = list(self.models)
        d["layers"] = list(self.layers)
        d["indexed_by"] = list(self.indexed_by)
        d["default_hook"] = self.hook(self.layers[len(self.layers) // 2])
        return d


REGISTRY: tuple[SAEEntry, ...] = (
    SAEEntry(
        repo="google/gemma-scope-2b-pt-res",
        models=("google/gemma-2-2b", "google/gemma-2-2b-it"),
        d_in=2304,
        layers=tuple(range(26)),
        point="resid_post",
        label="Gemma Scope · gemma-2-2b · residual stream",
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
