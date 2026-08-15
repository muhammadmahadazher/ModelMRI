"""What a finetune changed — over a PROMPT SET, not over one prompt.

`behavdiff.py` compares two models on one prompt and was built for a different
question: what a quantisation cost you, where the two sides are the same
weights at two precisions and one prompt is a fair sample of the damage. A
finetune is not that. It changed the model on purpose, in some places and not
others, and the whole question is WHICH — so one prompt's diff presented as a
property of the finetune is exactly the error this project exists to refuse.

    "A number measured once is a sample, not a property."

So every number here is a distribution over your prompts: median, inter-
quartile range, n, and how often the thing happened at all. A layer that tops
the divergence on one prompt and sits mid-table on the other nineteen displays
as exactly that.

WHAT THIS IS NOT

Not model diffing in the crosscoder sense. crosscode and OpenMOSS train a
shared autoencoder over both models and can say a FEATURE moved; that is
GPU-months and neither ships a license file, which legally blocks reuse
anyway. This is a diff of BEHAVIOUR on a prompt set — where the answers differ
and where the residual stream has rotated — and it must never be described as
the other thing.

THE PAIR HAS TO BE COMPARABLE, AND THAT IS CHECKED

Two models with different layer counts, hidden sizes or vocabularies produce a
per-layer table that lines up layer 3 with layer 3 and means nothing. Depths
are never normalised to a 0-1 fraction to paper over it: layer 12 of 24 and
layer 12 of 32 are not the same place, and a fraction would say they were.
Tokenisers are checked PER PROMPT, because two models can share a tokeniser
config and still split one string differently.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field

from .errors import BadRequest

# One prompt is not a sample. Below this the spread this module exists to
# report is not a spread -- an IQR over three numbers is two of them.
MIN_PROMPTS = 4

# Each prompt costs one forward pass per side, plus the load of each model
# once. The load dominates, so the cap is about the table and the wait rather
# than about memory.
MAX_PROMPTS = 64

# How many heads the summary names. Every head still carries a row; this caps
# what the sentence reads out, the same way `sweep.py` caps its top-k.
TOP_HEADS = 8


class DiffError(BadRequest):
    """This pair cannot be diffed honestly, and we say why."""


@dataclass
class PromptResult:
    """One prompt, compared. Never reported on its own."""

    prompt: str
    n_tokens: int
    mean_kl: float
    max_kl: float
    flips: int
    # Where this prompt's residual cosine falls furthest in one step. No
    # threshold is involved -- see `steepest_drop`. None when the curve never
    # decreases, which is a result.
    first_divergent_layer: int | None = None
    # How far it fell there. Printed beside the layer so nobody has to guess
    # whether a turn in the curve is worth anything: 5e-04 and 0.4 are both
    # "the steepest drop" and only one of them is a change.
    drop: float = 0.0
    # layer -> cosine between the two models' residual streams, meaned over
    # positions. Empty when hidden states were not captured.
    cosine: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Spread:
    """One quantity across the prompt set. The unit of every claim here."""

    name: str
    median: float
    low: float
    high: float
    n: int
    # How many prompts this happened on at all, when the quantity is a count.
    n_nonzero: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def stable(self) -> bool:
        """Is the spread narrow enough that the median describes the set?

        The bar is a full spread of half the median — a quarter either side of
        it if the spread is symmetric — and is deliberately generous: the claim
        it gates is only "this number is typical", and a stricter threshold
        would call an honest measurement unstable.

        A median of exactly zero has no scale to be relative to, so it is
        stable only if the whole set is zero. `high` at 1e-9 around a zero
        median is a set with no typical value, not a tight one.
        """
        if self.median == 0.0:
            return self.high == 0.0
        return (self.high - self.low) <= abs(self.median) * 0.5


@dataclass
class HeadShift:
    """One head, and how far its ablation score moved between the two models.

    Both sides' medians are carried, not only the difference. A head that went
    from 0.02 to 0.06 and one that went from 4.00 to 4.04 have the same shift
    and are not the same finding, and a single delta column cannot tell them
    apart.
    """

    layer: int
    head: int
    median_a: float
    median_b: float
    shift: float
    # How many prompts this head was measured on. A head can be missing from
    # one prompt's ranking if that prompt's ranking was truncated.
    n: int
    # How often it landed in the top-k on each side. The rank is what a reader
    # acts on -- "layer 6 head 9 carries this" -- and it can change while
    # every score stays put.
    top_a: int = 0
    top_b: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TokenShift:
    """One prompt token, and how the two models' reliance on it moved.

    Per PROMPT, not pooled across the set. Token 4 of one prompt and token 4
    of another are different words, and averaging them would be arithmetic on
    a coincidence of position. The spread this module reports everywhere else
    is over prompts; here it is the prompts themselves that are the rows.
    """

    prompt_index: int
    index: int
    token: str
    kl_a: float
    kl_b: float
    shift: float
    # A token whose score cleared the noise floor on one side and not the
    # other. The headline: the finetune started, or stopped, depending on it.
    newly_used: bool = False
    newly_ignored: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LayerSpread:
    """One layer's residual cosine across the prompt set."""

    layer: int
    median: float
    low: float
    high: float
    n: int
    # On how many prompts this layer was the FIRST to diverge. The headline
    # number, and the one a single prompt cannot support.
    n_first: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelDiff:
    model_a: str
    model_b: str
    n_prompts: int
    n_layers: int
    prompts: list[PromptResult] = field(default_factory=list)
    layers: list[LayerSpread] = field(default_factory=list)
    kl: Spread | None = None
    flips: Spread | None = None
    # The layer that was first to diverge on the most prompts, and on how
    # many. None when no prompt diverged anywhere.
    consensus_layer: int | None = None
    consensus_share: float = 0.0
    # Ranked by how far the head's ablation score moved. Empty unless the
    # caller asked for it: it costs n_layers x n_heads forward passes PER
    # PROMPT PER SIDE, which is the most expensive thing in this module by two
    # orders of magnitude.
    heads: list[HeadShift] = field(default_factory=list)
    head_passes: int = 0
    # Prompt tokens whose scores crossed their own noise floor between the two
    # sides. Empty unless the caller asked for it -- it costs a token
    # attribution per prompt per side.
    tokens: list[TokenShift] = field(default_factory=list)
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["means"] = self.means()
        return out

    def means(self) -> str:
        parts = [
            f"{self.model_a} against {self.model_b} on {self.n_prompts} of "
            f"your prompts. Every number below is a MEDIAN over those prompts "
            f"with the middle half beside it, because a diff measured on one "
            f"prompt is a sample and this tool will not print it as a "
            f"property of the finetune."
        ]
        if self.kl:
            spread = "typical of the set" if self.kl.stable() else "NOT typical"
            parts.append(
                f"The answers differ by a median {self.kl.median:.4f} nats "
                f"per position (middle half {self.kl.low:.4f} to "
                f"{self.kl.high:.4f}), which is {spread}: "
                + (
                    "the prompts agree with each other about how much moved."
                    if self.kl.stable()
                    else "the prompts disagree with each other by more than "
                    "the median, so there is no single amount this finetune "
                    "moved the answer by. Read the per-prompt rows."
                )
            )
        if self.flips:
            parts.append(
                f"The top token changed on a median {self.flips.median:.0f} "
                f"positions per prompt, and on {self.flips.n_nonzero} of "
                f"{self.n_prompts} prompts at all."
            )
        drops = [p.drop for p in self.prompts if p.first_divergent_layer is not None]
        if self.consensus_layer is None:
            parts.append(
                "THE COSINE NEVER FALLS on any prompt: the two residual "
                "streams stay as aligned at the end as they were at the "
                "start. Whatever changed did not show up as a rotation of the "
                "stream on this prompt set."
            )
        else:
            share = self.consensus_share * 100
            typical = statistics.median(drops) if drops else 0.0
            parts.append(
                f"The cosine falls furthest at layer {self.consensus_layer} on "
                f"{share:.0f}% of prompts, by a median {typical:.2e} — read "
                f"that size before the layer, because the steepest drop in a "
                f"pair that barely differs is still the steepest drop"
                + (
                    ". That is a majority, so it is where this finetune "
                    "starts to differ on this set."
                    if self.consensus_share >= 0.5
                    else ". That is a plurality and not a majority, so the "
                    "point of divergence MOVES between your prompts and "
                    "naming one layer would be picking the commonest of "
                    "several."
                )
            )
        if self.heads:
            top = self.heads[0]
            moved = [h for h in self.heads if h.top_a != h.top_b]
            parts.append(
                f"The head whose ablation score moved most is L{top.layer}"
                f"H{top.head}: a median {top.median_a:.4f} nats in "
                f"{self.model_a} against {top.median_b:.4f} in {self.model_b}. "
                f"BOTH sides are printed rather than the difference alone — a "
                f"head that went from 0.02 to 0.06 and one that went from 4.00 "
                f"to 4.04 moved by the same amount and are not the same "
                f"finding."
            )
            if moved:
                names = ", ".join(f"L{h.layer}H{h.head}" for h in moved[:4])
                parts.append(
                    f"{len(moved)} heads changed how often they land in the "
                    f"top {TOP_HEADS} ({names}) — which is a change in WHICH "
                    f"head carries the answer, not in how much any of them "
                    f"matters, and the two can move independently."
                )
            else:
                parts.append(
                    f"No head changed how often it lands in the top "
                    f"{TOP_HEADS}: the same heads carry the answer on both "
                    f"sides of this comparison, whatever their scores did."
                )
        if self.tokens:
            gained = [t for t in self.tokens if t.newly_used]
            lost = [t for t in self.tokens if t.newly_ignored]
            if gained or lost:
                def _names(rows):
                    return ", ".join(f"{r.token!r}" for r in rows[:4])

                bits = []
                if gained:
                    bits.append(
                        f"{len(gained)} prompt token(s) the finetune NEWLY "
                        f"depends on ({_names(gained)})"
                    )
                if lost:
                    bits.append(
                        f"{len(lost)} it STOPPED depending on ({_names(lost)})"
                    )
                parts.append(
                    " and ".join(bits)
                    + ". Crossing a noise floor is not the same as mattering a "
                    "lot — these are tokens whose score moved from below one "
                    "side's floor to above the other's, which is a change in "
                    "KIND rather than in degree."
                )
            else:
                parts.append(
                    "No prompt token crossed a noise floor in either "
                    "direction: the two models depend on the same words, "
                    "whatever their scores did."
                )
        parts.append(
            "This is a diff of BEHAVIOUR on your prompts and of where the "
            "residual stream rotated. It cannot say that a shared feature "
            "moved — that needs a crosscoder trained over both models — and "
            "it is not model diffing in that sense."
        )
        if self.notes:
            parts.append(" ".join(self.notes))
        return " ".join(parts)


# ------------------------------------------------------- the compatibility gate


def _shape_from_config(config) -> dict:
    """The four numbers the gate and the cost estimate need.

    ONE builder for both callers. `_shape_of` reads a loaded model and
    `shape_without_loading` reads a config file, and when they were two
    literals the second silently lacked `n_heads` — which surfaced as a
    KeyError from the cost estimate, after a model had been saved and both
    sides were about to load.
    """
    return {
        "n_layers": int(getattr(config, "num_hidden_layers", 0) or 0),
        "hidden": int(getattr(config, "hidden_size", 0) or 0),
        "vocab": int(getattr(config, "vocab_size", 0) or 0),
        "n_heads": int(getattr(config, "num_attention_heads", 0) or 0),
    }


def _shape_of(model) -> dict:
    return _shape_from_config(getattr(model, "config", None))


def shape_without_loading(spec: str) -> dict | None:
    """Layer count, hidden size and vocabulary from the config alone.

    A few hundred bytes of JSON against several gigabytes of weights. Without
    this the compatibility gate fires only once BOTH models have been loaded
    and released — and the models worth comparing are exactly the ones near
    the limit of the machine, so refusing a mismatched pair after two full
    loads is the wrong way round.

    None when the config cannot be read at all: a local GGUF, a path this
    cannot resolve, no network for a Hub id. That is not a refusal — the
    post-load check below still runs and still catches it.
    """
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(spec)
    except Exception:
        # Deliberately broad and deliberately silent. Every failure here means
        # "the cheap path did not work", and every one of them is answered the
        # same way: fall through to the check that needs the model.
        return None
    return _shape_from_config(config)


def check_pair(shape_a: dict, shape_b: dict, label_a: str, label_b: str) -> None:
    """Refuse a pair whose per-layer table would line up the wrong things.

    NAMES BOTH SIDES in every message. "the models are incompatible" sends the
    reader to check both; "24 layers against 32" tells them which one they did
    not mean to pick.

    Depths are never normalised to a 0-1 fraction to make the table line up.
    Layer 12 of 24 and layer 12 of 32 are not the same place, and a fraction
    would quietly assert that they are.

    A shape the config never stated is REFUSED before any of that. It used to
    arrive as 0, and `0 != 0` is false — so two models that both failed to
    state their depth passed this gate as a match, and every check below it
    then compared a pair whose shapes had never actually been read.
    """
    for shape, label in ((shape_a, label_a), (shape_b, label_b)):
        missing = [
            name
            for name in ("n_layers", "hidden", "vocab")
            if not shape.get(name)
        ]
        if missing:
            raise DiffError(
                f"{label}'s config does not state its "
                f"{', '.join(missing)}. That is not a zero — it is a number "
                f"nothing has read, and comparing two models on shapes nobody "
                f"read would line up whatever happened to be there."
            )
    if shape_a["n_layers"] != shape_b["n_layers"]:
        # The example layer is picked from the SMALLER model so it exists in
        # both. Hardcoding "layer 12" produced "layer 12 of 12 is the same
        # place as layer 12 of 6" on gpt2 against distilgpt2 — an illustration
        # naming a layer one of the two models does not have.
        example = min(shape_a["n_layers"], shape_b["n_layers"]) // 2
        raise DiffError(
            f"{label_a} has {shape_a['n_layers']} layers and {label_b} has "
            f"{shape_b['n_layers']}. A per-layer comparison would line up "
            f"layer 3 with layer 3 and mean nothing, and normalising both to a "
            f"depth fraction would assert that layer {example} of "
            f"{shape_a['n_layers']} is the same place as layer {example} of "
            f"{shape_b['n_layers']}, which nothing here has measured."
        )
    if shape_a["hidden"] != shape_b["hidden"]:
        raise DiffError(
            f"{label_a} has a hidden size of {shape_a['hidden']} and "
            f"{label_b} has {shape_b['hidden']}. There is no cosine between "
            f"vectors of different lengths."
        )
    if shape_a["vocab"] != shape_b["vocab"]:
        raise DiffError(
            f"{label_a} has a vocabulary of {shape_a['vocab']:,} and "
            f"{label_b} has {shape_b['vocab']:,}. A KL between distributions "
            f"over different vocabularies is not a KL — and a finetune that "
            f"added tokens is a real and common case, which is why this says "
            f"both numbers rather than refusing anonymously."
        )


def check_tokens(ids_a: list[int], ids_b: list[int], prompt: str) -> None:
    """Refuse a prompt the two tokenisers split differently.

    PER PROMPT, not once for the pair. Two models can carry the same tokeniser
    config and still disagree on one string — an added special token, a
    different normaliser setting — and the disagreement is invisible until
    that string arrives. Position 7 of one run being a different word from
    position 7 of the other is silent and produces a table of nonsense.
    """
    if ids_a == ids_b:
        return
    where = next(
        (i for i, (x, y) in enumerate(zip(ids_a, ids_b)) if x != y),
        min(len(ids_a), len(ids_b)),
    )
    raise DiffError(
        f"the two tokenisers split this prompt differently — "
        f"{len(ids_a)} tokens against {len(ids_b)}, first differing at "
        f"position {where}. Every per-position number would be comparing one "
        f"model's word with the other model's different word. The prompt was: "
        f"{prompt[:80]!r}"
    )


# --------------------------------------------------------------- the numbers


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    """Median and the middle half, for any n.

    `statistics.quantiles` needs at least two points and raises below that.
    One prompt cannot reach here -- `MIN_PROMPTS` is four -- but a per-layer
    slice can be shorter than the prompt list if a layer went missing on one
    of them, and a crash there would lose the whole comparison over one row.
    """
    if not values:
        return 0.0, 0.0, 0.0
    if len(values) < 2:
        return values[0], values[0], values[0]
    cuts = statistics.quantiles(values, n=4, method="inclusive")
    return statistics.median(values), cuts[0], cuts[2]


def _spread(name: str, values: list[float]) -> Spread:
    median, low, high = _quartiles(values)
    return Spread(
        name=name,
        median=round(median, 6),
        low=round(low, 6),
        high=round(high, 6),
        n=len(values),
        n_nonzero=sum(1 for v in values if v),
    )


def cosine_per_layer(hidden_a: list, hidden_b: list) -> list[float]:
    """Cosine between the two residual streams at each layer, over positions.

    Cosine and not distance: the two models' streams can differ in SCALE
    without differing in direction — a finetune that changed a norm gain moves
    every vector's length and none of their meanings — and a distance would
    report that as the model having changed everywhere.
    """
    import torch

    out = []
    for a, b in zip(hidden_a, hidden_b):
        similarity = torch.nn.functional.cosine_similarity(
            a.float(), b.float(), dim=-1
        )
        out.append(round(float(similarity.mean()), 6))
    return out


def steepest_drop(cosines: list[float]) -> tuple[int | None, float]:
    """Where the two streams first come apart, and by how much. NO THRESHOLD.

    The first version of this compared each layer against a floor and the
    floor was the constant 0.999 — dressed in a docstring claiming it was
    measured on the pair, which it was not. MEASURED on gpt2 against a copy
    with one head zeroed in block 6: the cosine reads 1.000000000 through
    layer 6 and 0.999475 at layer 7, exactly where that block's output first
    appears, and 0.999475 sits ABOVE 0.999. A real divergence, correctly
    measured, reported as none.

    The largest single-step DECREASE needs no constant. It is the layer where
    the curve turns, which is the question a reader is asking, and it is
    scale-free: a pair that differs by 0.0005 and a pair that differs by 0.4
    both get an answer, and the size comes back beside it so nobody has to
    guess whether the answer is worth anything.

    Returns (layer, drop). `layer` is None only when nothing decreased
    anywhere — two streams that never come apart, which is a result.
    """
    best_layer, best_drop = None, 0.0
    for layer in range(1, len(cosines)):
        drop = cosines[layer - 1] - cosines[layer]
        if drop > best_drop:
            best_layer, best_drop = layer, drop
    return best_layer, best_drop


def summarise(
    model_a: str,
    model_b: str,
    results: list[PromptResult],
    *,
    n_layers: int,
    seconds: float = 0.0,
    notes: list[str] | None = None,
) -> ModelDiff:
    """Turn per-prompt results into the only shape this module reports in."""
    if not results:
        raise DiffError("nothing was compared.")

    layers: list[LayerSpread] = []
    firsts = [r.first_divergent_layer for r in results]
    for layer in range(n_layers):
        values = [r.cosine[layer] for r in results if layer < len(r.cosine)]
        if not values:
            continue
        median, low, high = _quartiles(values)
        layers.append(
            LayerSpread(
                layer=layer,
                median=round(median, 6),
                low=round(low, 6),
                high=round(high, 6),
                n=len(values),
                n_first=sum(1 for f in firsts if f == layer),
            )
        )

    diverged = [f for f in firsts if f is not None]
    consensus, share = None, 0.0
    if diverged:
        consensus = max(set(diverged), key=diverged.count)
        # Out of ALL prompts, not out of the ones that diverged. A layer that
        # was first on both of the two prompts that diverged out of twenty is
        # not "100% of prompts", and that is exactly how a rate becomes a
        # claim nobody measured.
        share = diverged.count(consensus) / len(results)

    return ModelDiff(
        model_a=model_a,
        model_b=model_b,
        n_prompts=len(results),
        n_layers=n_layers,
        prompts=results,
        layers=layers,
        kl=_spread("mean KL per position", [r.mean_kl for r in results]),
        flips=_spread("positions whose top token changed", [float(r.flips) for r in results]),
        consensus_layer=consensus,
        consensus_share=round(share, 4),
        seconds=round(seconds, 2),
        notes=list(notes or []),
    )


def head_pass_estimate(n_layers: int, n_heads: int, n_prompts: int) -> int:
    """What the head half costs, before it is run.

    `rank_heads` is one pass per head plus a base, a repeat for the noise
    floor and a joint check. Times two sides, times every prompt. On gpt2 with
    six prompts that is about 1,760 passes; on a 1.7B model with 448 heads it
    is about 5,400. Both are answerable in minutes and neither should start
    without the reader having seen the number.

    An unstated layer or head count is refused rather than quoted. With
    n_heads=0 this returns `3 * 2 * n_prompts` — 36 passes for six prompts,
    for a run that will do thousands. A preflight that under-quotes is worse
    than no preflight, because it is the number somebody plans around.
    """
    if n_layers <= 0 or n_heads <= 0:
        raise DiffError(
            f"cannot price the head comparison: this config states "
            f"{n_layers} layers and {n_heads} attention heads. One of those "
            f"was never read, and a cost estimate built on it would quote a "
            f"few dozen passes for a run of several thousand."
        )
    return (n_layers * n_heads + 3) * 2 * n_prompts


def _rank_one(model, tokenizer, prompt: str) -> dict:
    """One side's head ranking on one prompt, as {(layer, head): kl}."""
    import torch

    from . import ablate

    device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    config = model.config
    n_layers = int(config.num_hidden_layers)
    n_heads = int(getattr(config, "num_attention_heads", 0) or 0)
    if not n_heads:
        raise DiffError(
            "this model does not publish a head count, so its heads cannot be "
            "ranked. The rest of the comparison is unaffected."
        )

    def blocks(layer: int):
        return _block_of(model, layer)

    out = ablate.rank_heads(
        model,
        blocks,
        ids,
        position=int(ids.shape[-1]) - 1,
        layers=list(range(n_layers)),
        n_heads=n_heads,
        baseline="zero",
        decode=lambda t: tokenizer.decode([t]),
    )
    return {(r["layer"], r["head"]): float(r["kl"]) for r in out.get("ranked") or []}


def _block_of(model, layer: int):
    """The transformer block at `layer`, across the architectures this sees.

    Duplicated from `ModelRuntime._block` rather than imported: that method
    reaches through `self` for a model this module does not own, and the two
    lists of attribute names are the same three lines either way.
    """
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        node = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if node is not None:
            return node[layer]
    raise DiffError(
        "ModelMRI cannot find this model's transformer blocks, so its heads "
        "cannot be ranked. The rest of the comparison is unaffected."
    )


def summarise_heads(
    per_prompt_a: list[dict], per_prompt_b: list[dict]
) -> list[HeadShift]:
    """Turn two lists of per-prompt rankings into one spread per head."""
    heads = set()
    for row in per_prompt_a + per_prompt_b:
        heads.update(row)

    def top_set(row: dict) -> set:
        return set(sorted(row, key=lambda k: -row[k])[:TOP_HEADS])

    tops_a = [top_set(r) for r in per_prompt_a]
    tops_b = [top_set(r) for r in per_prompt_b]

    out: list[HeadShift] = []
    for key in sorted(heads):
        values_a = [r[key] for r in per_prompt_a if key in r]
        values_b = [r[key] for r in per_prompt_b if key in r]
        if not values_a or not values_b:
            continue
        median_a = statistics.median(values_a)
        median_b = statistics.median(values_b)
        out.append(
            HeadShift(
                layer=key[0],
                head=key[1],
                median_a=round(median_a, 6),
                median_b=round(median_b, 6),
                shift=round(median_b - median_a, 6),
                n=min(len(values_a), len(values_b)),
                top_a=sum(1 for t in tops_a if key in t),
                top_b=sum(1 for t in tops_b if key in t),
            )
        )
    # By the SIZE of the move, not by its sign: a head the finetune stopped
    # relying on is as much a finding as one it started relying on.
    out.sort(key=lambda h: -abs(h.shift))
    return out


def token_pass_estimate(n_tokens: int, n_prompts: int) -> int:
    """What the token half costs, before it is run.

    `rank_tokens` is one pass per tested token plus seven fixed ones. Times
    two sides, times every prompt. Far cheaper than the head half -- a
    24-token prompt is about 31 passes a side -- but still priced rather than
    assumed, because a 500-token prompt is not.
    """
    return (min(n_tokens, 64) + 7) * 2 * n_prompts


def _attribute_one(model, tokenizer, prompt: str) -> dict:
    """One side's token attribution on one prompt.

    Returns `{"scores": {index: kl}, "tokens": {index: text}, "floor": float}`.
    The FLOOR travels with the scores: this module compares two models'
    reliance on a token, and "cleared its own floor" is the only comparison
    that survives the two models having different noise floors -- which they
    do, because a finetune changes the arithmetic as well as the weights.
    """
    from . import ablate, attribute

    device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    if int(ids.shape[-1]) < 3:
        raise DiffError(
            f"a token attribution needs a prompt with something in front of "
            f"the position being attributed, and {prompt[:40]!r} tokenises to "
            f"{int(ids.shape[-1])}."
        )
    try:
        out = attribute.rank_tokens(
            model,
            ids,
            position=int(ids.shape[-1]) - 1,
            n_prompt=int(ids.shape[-1]),
            control_ids=attribute.control_token_ids(tokenizer),
            decode=lambda t: tokenizer.decode([t]),
        )
    except attribute.AttributionError as err:
        # Not a crash: a measurement that module cannot take honestly on this
        # model. The rest of the comparison is unaffected, so this becomes a
        # skipped prompt rather than a failed run.
        raise DiffError(str(err)) from err  # leak-ok: authored by attribute.py
    # The floor is REQUIRED, not defaulted. `summarise_tokens` calls a token
    # used when its KL clears its own side's floor, so a floor that fell back
    # to 0.0 would mark every token on that side as used and every crossing as
    # a finding. An absent floor is a broken measurement, not a permissive one.
    floor = out.get("noise_floor_kl")
    if floor is None:
        raise DiffError(
            "this model's token attribution returned no noise floor, so "
            "nothing here can tell a token that matters from one that does "
            "not. Refusing rather than treating every token as significant."
        )
    return {
        "scores": {r["index"]: float(r["kl"]) for r in out.get("ranked") or []},
        "tokens": {r["index"]: r["token"] for r in out.get("ranked") or []},
        "floor": float(floor),
    }


def summarise_tokens(
    per_prompt_a: list[dict], per_prompt_b: list[dict]
) -> list[TokenShift]:
    """Which tokens crossed a noise floor between the two models.

    EACH SIDE AGAINST ITS OWN FLOOR. The two models have different noise
    floors -- a finetune changes the arithmetic as well as the weights -- so
    "above 0.05 in both" would be comparing one model's signal against the
    other model's noise. "Cleared its own floor" is the only comparison that
    survives that.
    """
    out: list[TokenShift] = []
    for prompt_index, (a, b) in enumerate(zip(per_prompt_a, per_prompt_b)):
        for index in sorted(set(a["scores"]) & set(b["scores"])):
            kl_a, kl_b = a["scores"][index], b["scores"][index]
            used_a = kl_a > a["floor"]
            used_b = kl_b > b["floor"]
            out.append(
                TokenShift(
                    prompt_index=prompt_index,
                    index=index,
                    token=a["tokens"].get(index) or b["tokens"].get(index, ""),
                    kl_a=round(kl_a, 6),
                    kl_b=round(kl_b, 6),
                    shift=round(kl_b - kl_a, 6),
                    newly_used=bool(used_b and not used_a),
                    newly_ignored=bool(used_a and not used_b),
                )
            )
    # Crossings first, then by the size of the move. A token that changed KIND
    # is the finding; one that merely changed degree is context for it.
    out.sort(key=lambda t: (not (t.newly_used or t.newly_ignored), -abs(t.shift)))
    return out


def plan(prompts: list[str], *, min_prompts: int = MIN_PROMPTS) -> list[str]:
    """The prompt set, or a refusal that says why one prompt is not enough."""
    clean = [p.strip() for p in prompts if isinstance(p, str) and p.strip()]
    if len(clean) < min_prompts:
        raise DiffError(
            f"this needs at least {min_prompts} prompts and got {len(clean)}. "
            f"The whole output of this comparison is a SPREAD across your "
            f"prompts — a diff measured on one prompt is a sample, and "
            f"printing it as a property of the finetune is the error this "
            f"module exists to avoid."
        )
    if len(clean) > MAX_PROMPTS:
        raise DiffError(
            f"that is {len(clean)} prompts and each costs a forward pass on "
            f"both sides. The cap is {MAX_PROMPTS} — cut the set rather than "
            f"having it cut for you, so you know which prompts the answer is "
            f"about."
        )
    return clean


def compare(
    load_side,
    model_a: str,
    model_b: str,
    prompts: list[str],
    *,
    min_prompts: int = MIN_PROMPTS,
    include_heads: bool = False,
    include_tokens: bool = False,
    on_stage=None,
) -> ModelDiff:
    """Load each side ONCE, capture every prompt, release, compare.

    `load_side(spec)` is supplied by the caller and returns `(model,
    tokenizer)`; releasing is the caller's job through the context it hands
    back. Taking it as an argument keeps this module free of load policy —
    `behavdiff` already owns that — and makes the sequencing testable without
    a model.

    ONE LOAD PER SIDE, not one per prompt. The models worth comparing are
    exactly the ones near the limit of the machine, and reloading per prompt
    would turn a 20-prompt comparison into 40 loads.
    """
    import torch

    from . import ablate

    if model_a == model_b:
        raise DiffError(
            "both sides are the same model, so every difference would be zero "
            "by construction."
        )
    wanted = plan(prompts, min_prompts=min_prompts)
    started = time.perf_counter()

    # BEFORE either load. A few hundred bytes of config against several
    # gigabytes of weights: refusing a mismatched pair after loading both of
    # them is the wrong way round when the models worth comparing are the ones
    # near the limit of the machine. Skipped silently when a config cannot be
    # read — the post-load check still runs.
    cheap_a = shape_without_loading(model_a)
    cheap_b = shape_without_loading(model_b)
    if cheap_a and cheap_b:
        check_pair(cheap_a, cheap_b, model_a, model_b)

    captures: dict[str, list] = {}
    head_rankings: dict[str, list] = {}
    token_scores: dict[str, list] = {}
    shapes: dict[str, dict] = {}
    for spec in (model_a, model_b):
        if on_stage:
            on_stage("load", spec)
        model, tokenizer, release = load_side(spec)
        try:
            shapes[spec] = _shape_of(model)
            if len(shapes) == 2:
                check_pair(shapes[model_a], shapes[model_b], model_a, model_b)
            rows = []
            rankings = []
            attributions = []
            for index, prompt in enumerate(wanted):
                if on_stage:
                    on_stage("capture", f"{spec} · prompt {index + 1}/{len(wanted)}")
                encoded = tokenizer(prompt, return_tensors="pt")
                ids = encoded.input_ids.to(next(model.parameters()).device)
                with torch.no_grad():
                    out = model(ids, output_hidden_states=True)
                rows.append(
                    {
                        "ids": [int(i) for i in encoded.input_ids[0]],
                        "probs": ablate.distribution(out.logits[0]).to("cpu"),
                        "top": [
                            int(i) for i in out.logits[0].argmax(dim=-1).to("cpu")
                        ],
                        # Every layer's residual stream, on the CPU, so the
                        # model can go before the second side is loaded.
                        "hidden": [h[0].to("cpu") for h in out.hidden_states],
                    }
                )
                if include_heads:
                    if on_stage:
                        on_stage(
                            "rank",
                            f"{spec} · prompt {index + 1}/{len(wanted)}",
                        )
                    rankings.append(_rank_one(model, tokenizer, prompt))
                if include_tokens:
                    if on_stage:
                        on_stage(
                            "attribute",
                            f"{spec} · prompt {index + 1}/{len(wanted)}",
                        )
                    attributions.append(_attribute_one(model, tokenizer, prompt))
            captures[spec] = rows
            head_rankings[spec] = rankings
            token_scores[spec] = attributions
        finally:
            # In a `finally`: a capture that raises must still give the memory
            # back, or the second side has nowhere to load into and the real
            # error is buried under an out-of-memory.
            release()

    rows_a, rows_b = captures[model_a], captures[model_b]
    results: list[PromptResult] = []
    for prompt, a, b in zip(wanted, rows_a, rows_b):
        check_tokens(a["ids"], b["ids"], prompt)
        kls = [
            ablate.kl_nats(a["probs"][i], b["probs"][i])
            for i in range(a["probs"].shape[0])
        ]
        cosines = cosine_per_layer(a["hidden"], b["hidden"])
        turn, drop = steepest_drop(cosines)
        results.append(
            PromptResult(
                prompt=prompt,
                n_tokens=len(a["ids"]),
                mean_kl=round(sum(kls) / len(kls), 6) if kls else 0.0,
                max_kl=round(max(kls), 6) if kls else 0.0,
                flips=sum(1 for x, y in zip(a["top"], b["top"]) if x != y),
                first_divergent_layer=turn,
                drop=round(drop, 9),
                cosine=cosines,
            )
        )

    diff = summarise(
        model_a,
        model_b,
        results,
        n_layers=len(results[0].cosine) if results else 0,
        seconds=time.perf_counter() - started,
    )
    if include_heads:
        diff.heads = summarise_heads(
            head_rankings[model_a], head_rankings[model_b]
        )
        shape = shapes[model_a]
        diff.head_passes = head_pass_estimate(
            # Not `.get(..., 0)`: `_shape_from_config` is the one builder for
            # both callers and always sets this key, and a 0 slipped in here
            # is what `head_pass_estimate` now refuses outright.
            shape["n_layers"], shape["n_heads"], len(wanted)
        )
    if include_tokens:
        diff.tokens = summarise_tokens(
            token_scores[model_a], token_scores[model_b]
        )
    return diff


def loader(*, dtype: str = "bfloat16", device: str = "cpu", device_kind: str = "cpu"):
    """A `load_side` backed by `behavdiff`'s existing load policy.

    Separate from `compare` so the sequencing above is testable without a
    model, and so the policy — eager attention, GGUF handling, and the
    `nullmodel.teardown` release that actually gives accelerator memory back —
    lives in one place rather than being re-derived here.
    """
    from . import behavdiff

    def load_side(spec: str):
        side = behavdiff.side(spec)
        model, tokenizer = behavdiff._build(
            side, dtype=dtype, device=device, device_kind=device_kind
        )
        return model, tokenizer, lambda: behavdiff._release(model)

    return load_side
