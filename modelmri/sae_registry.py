"""Which sparse autoencoders exist, and which model each one fits.

A SAE is trained against one model's residual stream at one layer. There is no
such thing as a general one, and there is no way to make a model's features
appear if nobody has trained an autoencoder for it — that is GPU-months of
someone else's work, not a feature this tool can implement. Public SAEs exist
for maybe a dozen models in total.

So this is a lookup table, not a capability. Its job is to stop you typing
repository names, and to say plainly when the answer for your model is "none
yet" rather than leaving the panel looking broken.

**The table is a convenience; the guarantee is elsewhere.** Nothing here is
trusted: SAEHandle.load refuses any SAE whose `d_in` does not equal the loaded
model's `hidden_size`, so a wrong entry fails loudly at load rather than
producing confident features describing the wrong model. That check is the
reason it is safe to ship a hand-maintained list at all.
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
    #: SAELens-format `.safetensors` with W_enc/W_dec is what the loader opens.
    #: Anything else is listed so you know it exists, and marked unsupported
    #: rather than offered and then failing.
    supported: bool = True
    note: str = ""

    def hook(self, layer: int) -> str:
        return f"blocks.{layer}.hook_{self.point}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["models"] = list(self.models)
        d["layers"] = list(self.layers)
        d["default_hook"] = self.hook(self.layers[len(self.layers) // 2])
        return d


REGISTRY: tuple[SAEEntry, ...] = (
    SAEEntry(
        repo="jbloom/GPT2-Small-SAEs-Reformatted",
        models=("gpt2", "openai-community/gpt2"),
        d_in=768,
        layers=tuple(range(12)),
        point="resid_pre",
        label="GPT-2 small · residual stream · 24,576 features",
        note="The one this tool was built against, verified end to end.",
    ),
    SAEEntry(
        repo="google/gemma-scope-2b-pt-res",
        models=("google/gemma-2-2b", "google/gemma-2-2b-it"),
        d_in=2304,
        layers=tuple(range(26)),
        point="resid_post",
        label="Gemma Scope · gemma-2-2b · residual stream",
        supported=False,
        note="Gemma Scope ships .npz parameter files, which this loader does "
        "not open yet. Listed so you know it exists.",
    ),
    SAEEntry(
        repo="google/gemma-scope-9b-pt-res",
        models=("google/gemma-2-9b", "google/gemma-2-9b-it"),
        d_in=3584,
        layers=tuple(range(42)),
        point="resid_post",
        label="Gemma Scope · gemma-2-9b · residual stream",
        supported=False,
        note="Same .npz format as the 2b release.",
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
