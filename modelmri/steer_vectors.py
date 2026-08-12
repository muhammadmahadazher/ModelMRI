"""Steer a model that has no sparse autoencoder — which is almost all of them.

The features panel can steer, and it needs an SAE. SAEs are published for a
handful of models and nobody is going to train one for the checkpoint you
finetuned last week, so for most models this tool could show you attention and
ablation and then had nothing to offer when you asked "can I push it".

A contrastive direction needs no SAE and no training run: take pairs of prompts
that differ in the property you care about, read the residual stream at the
last token of each, and the difference of the means is a direction. Add it back
during generation and the model moves.

That is also the problem. **Difference-of-means always returns a direction.**
Feed it two arbitrary sets of sentences and it produces a vector with a norm
and a layer and a confident-looking sweep, and adding any large vector to a
residual stream changes the output. The number that comes back looks identical
whether you found something or nothing.

So every direction here is scored against **its own shuffled null**. Refit the
same pairs with the positive/negative labels randomly reassigned, K times, and
measure the same effect. A real direction beats its shuffles; a direction that
does not is reported as *not measured* — not as a small discovery.

Measured on gpt2, bf16 on an RTX 4060, twelve sentiment pairs ("I loved it" /
"I hated it" and similar), fitting on six and scoring on six:

    caa    11 of 12 layers beat their null, best at layer 9
           L0 +1.246 against a null max of 1.246 (does not survive)
           L10 +3.326 against 2.185
    repe   11 of 12, best at layer 10

and then the control that matters — the same 24 sentences split at random
instead of by sentiment: **0 of 12 layers beat their null**, under the
identical estimator. Signal passes, noise does not, and layer 0 failing is
right: that is the embedding, before anything has been computed.

**RepE is not centred, and `pca_lowrank` centres by default.** The first
version of this file subtracted the mean of the paired differences before the
PCA, reasoning that otherwise PC1 is dominated by the centroid shift. That is
backwards for contrastive pairs — the centroid shift IS the signal — and on
two clouds separated by 6.0 along one axis it scored 1.04 against a null whose
worst shuffle reached 1.30. A real direction failing its own control, because
the estimator had thrown the direction away before scoring it.

**Fit on half, score on the other half.** A direction scored on the pairs it
was fitted from will separate them, because it was built to. The split is the
difference between "this direction encodes the property" and "this direction
memorised these sentences".

**Coefficients are not portable.** A scale of 5 means nothing across models or
even across layers: residual norms differ by an order of magnitude between
early and late layers of the same network. Every applied strength here is
reported relative to the measured residual norm at that layer, so "0.5x the
stream's own norm" travels and "5.0" does not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .errors import BadRequest, Refusal

# Below this, a difference of means is mostly the noise of whichever sentences
# happened to be written. Stated rather than tuned: with four pairs the fit
# half is two, and two sentences do not define a direction in 768 dimensions.
MIN_PAIRS = 8

# Shuffled refits per direction. Same count and same reasoning as
# `patch.CONTROL_DRAWS` and `ablate.RESAMPLE_DRAWS` — one shuffle is a coin
# flip, and this package has now been bitten by that three times.
NULL_REFITS = 8

# Reused rather than redefined so every control in the package draws from one
# documented stream. See patch.py.
from .patch import CONTROL_SEED  # noqa: E402

METHODS = ("caa", "repe")


@dataclass
class Direction:
    """One fitted direction, and everything needed to judge it."""

    layer: int
    method: str
    # Cosine separation on the HELD-OUT half, and the same statistic on
    # `NULL_REFITS` label-shuffled refits.
    effect: float
    null_mean: float
    null_max: float
    beats_null: bool
    n_pairs: int
    n_fit: int
    n_score: int
    # The residual norm measured at this layer, so an applied coefficient can
    # be reported relative to something instead of as a bare number.
    residual_norm: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _last_token_states(model, blocks, ids_list, layers: list[int]):
    """Residual stream entering each layer, at the final token, per prompt.

    Uses the same pre-hook point as `patch._capture`, so a direction is fitted
    in the space the patching grid already measures rather than in a second,
    subtly different one.
    """
    import torch

    out: dict[int, list] = {layer: [] for layer in layers}
    for ids in ids_list:
        sink: dict[int, Any] = {}
        handles = [_capture_into(blocks(layer), layer, sink) for layer in layers]
        try:
            with torch.no_grad():
                model(ids)
        finally:
            for handle in handles:
                handle.remove()
        for layer in layers:
            if layer not in sink:
                raise Refusal(
                    f"layer {layer} produced no residual stream on this model, "
                    "so no direction can be fitted there."
                )
            out[layer].append(sink[layer][0, -1, :].float())
    return {layer: torch.stack(vecs) for layer, vecs in out.items()}


def _capture_into(block, layer: int, sink: dict):
    def pre(module, args):
        sink[layer] = args[0].detach()

    return block.register_forward_pre_hook(pre)


def _fit(pos, neg, method: str):
    """A direction from paired activations. Unit norm, so scale is separate."""
    import torch

    if method == "caa":
        # Contrastive activation addition: the mean difference. Simple, and
        # the thing repeng and steering-vectors both ship.
        direction = pos.mean(0) - neg.mean(0)
    elif method == "repe":
        # First principal component of the paired differences.
        #
        # NOT centred, and `pca_lowrank` centres by default — which is why this
        # is spelled out. The first version subtracted the mean of the
        # differences first, reasoning that otherwise PC1 is dominated by the
        # centroid shift and the method is CAA with extra steps. That reasoning
        # is backwards: with contrastive pairs the centroid shift IS the
        # signal, so centring deletes precisely what is being looked for and
        # leaves PC1 fitting noise. Measured on two clouds separated by 6.0
        # along one axis, the centred version scored 1.04 against a null whose
        # worst shuffle reached 1.30 — a real direction that failed its own
        # control, because the estimator had thrown the direction away.
        diffs = pos - neg
        _, _, v = torch.pca_lowrank(
            diffs, q=min(4, *diffs.shape), center=False
        )
        direction = v[:, 0]
        # PCA has no sign convention; align it with the mean difference so
        # "positive" means the same thing it means for CAA.
        if float(direction @ (pos.mean(0) - neg.mean(0))) < 0:
            direction = -direction
    else:
        raise BadRequest(f"unknown method {method!r} — use one of {', '.join(METHODS)}")

    norm = float(direction.norm())
    if norm == 0.0:
        raise Refusal(
            "the positive and negative sets have identical mean activations at "
            "this layer, so there is no direction between them to fit."
        )
    return direction / norm


def _separation(direction, pos, neg) -> float:
    """How far the two sets sit apart along this direction, in units of spread.

    A projection difference in raw units is unreadable and not comparable
    across layers; dividing by the pooled standard deviation of the projections
    makes it a standardised effect size, which is what the null is compared
    against.
    """
    import torch

    p = pos @ direction
    n = neg @ direction
    spread = torch.cat([p - p.mean(), n - n.mean()]).std()
    if float(spread) == 0.0:
        return 0.0
    return float((p.mean() - n.mean()) / spread)


def fit_direction(
    states,
    layer: int,
    *,
    method: str = "caa",
    refits: int = NULL_REFITS,
    seed: int = CONTROL_SEED,
) -> tuple[Direction, Any]:
    """(judgement, unit direction). Fit on half the pairs, score on the other.

    `states` is `(positive, negative)` stacked activations for one layer, both
    `[n_pairs, d_model]` and row-aligned: row i of each is one pair.
    """
    import torch

    pos, neg = states
    n = int(pos.shape[0])
    if n != int(neg.shape[0]):
        raise BadRequest(
            f"{n} positive prompts against {int(neg.shape[0])} negative ones — "
            "contrastive pairs must be matched, because the direction is "
            "fitted from their differences."
        )
    if n < MIN_PAIRS:
        raise Refusal(
            f"{n} pairs is not enough to fit a direction that can be checked. "
            f"This needs at least {MIN_PAIRS}, because half are held out for "
            "scoring and a direction scored on its own fitting set separates "
            "it by construction."
        )

    half = n // 2
    # The generator stays on CPU so the split and the shuffles are identical
    # whatever device the activations live on — a control that depends on the
    # accelerator is not reproducible. The masks move to the data.
    gen = torch.Generator().manual_seed(seed)
    device = pos.device
    order = torch.randperm(n, generator=gen).to(device)
    fit_idx, score_idx = order[:half], order[half:]

    direction = _fit(pos[fit_idx], neg[fit_idx], method)
    effect = _separation(direction, pos[score_idx], neg[score_idx])

    # The null: same pairs, same split, same estimator — labels shuffled.
    # Anything the pipeline produces from structureless data shows up here.
    nulls = []
    for k in range(refits):
        swap = torch.randint(0, 2, (n,), generator=gen).bool().to(device)
        shuffled_pos = torch.where(swap.unsqueeze(1), neg, pos)
        shuffled_neg = torch.where(swap.unsqueeze(1), pos, neg)
        try:
            null_dir = _fit(shuffled_pos[fit_idx], shuffled_neg[fit_idx], method)
        except Refusal:
            continue
        nulls.append(
            abs(_separation(null_dir, shuffled_pos[score_idx], shuffled_neg[score_idx]))
        )

    null_mean = sum(nulls) / len(nulls) if nulls else 0.0
    null_max = max(nulls) if nulls else 0.0
    beats = bool(nulls) and abs(effect) > null_max

    notes = []
    if not nulls:
        notes.append(
            "every shuffled refit collapsed, so there is no null to compare "
            "against and this direction is unverified"
        )
    if not beats and nulls:
        notes.append(
            "this direction does not beat its own label-shuffled refits, so "
            "the separation is not evidence of anything — it is what this "
            "estimator produces from these activations regardless of labels"
        )

    return Direction(
        layer=layer,
        method=method,
        effect=round(effect, 4),
        null_mean=round(null_mean, 4),
        null_max=round(null_max, 4),
        beats_null=beats,
        n_pairs=n,
        n_fit=int(half),
        n_score=int(n - half),
        residual_norm=round(float(torch.cat([pos, neg]).norm(dim=-1).mean()), 3),
        notes=notes,
    ), direction


def sweep(model, blocks, positive_ids, negative_ids, layers, *, method: str = "caa"):
    """Fit a direction at every layer and report which ones survive their null.

    Returns `(rows, vectors)` — the judgements and the directions themselves,
    so a caller can steer with one without refitting.
    """
    if method not in METHODS:
        raise BadRequest(f"unknown method {method!r} — use one of {', '.join(METHODS)}")

    pos_states = _last_token_states(model, blocks, positive_ids, layers)
    neg_states = _last_token_states(model, blocks, negative_ids, layers)

    rows, vectors = [], {}
    for layer in layers:
        judgement, direction = fit_direction(
            (pos_states[layer], neg_states[layer]), layer, method=method
        )
        rows.append(judgement.to_dict())
        vectors[layer] = direction

    survivors = [r for r in rows if r["beats_null"]]
    return {
        "method": method,
        "layers": rows,
        "best_layer": (
            max(survivors, key=lambda r: abs(r["effect"]))["layer"]
            if survivors
            else None
        ),
        "survived": len(survivors),
        "means": (
            "Standardised separation between the two sets along the fitted "
            "direction, on held-out pairs, beside the same statistic from "
            "label-shuffled refits. A layer that does not beat its own null "
            "is not a weak result — it is no result."
        ),
    }, vectors


# ------------------------------------------------------------- the store (#9)
#
# Persistence for directions, not a panel of its own. A direction is only
# meaningful against the model it came from, so what is stored is the vector
# AND the facts needed to refuse it later: model id, revision, layer, dtype,
# hidden size, how it was derived, and whether it beat its null.


def store_dir():
    """Where saved directions live. Same platform discipline as the trace db."""
    from . import paths

    directory = paths.data_dir() / "vectors"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _slug(name: str) -> str:
    """A filename that cannot escape the store.

    Names arrive from a text field. `..` and separators are removed rather than
    escaped — this writes files, and a name is not worth a path traversal.
    """
    cleaned = "".join(c if (c.isalnum() or c in "-_") else "-" for c in name).strip("-")
    if not cleaned:
        raise BadRequest(
            "a vector name needs at least one letter or digit — it becomes a "
            "filename."
        )
    return cleaned[:80]


def save(name: str, direction, meta: dict) -> dict:
    """Write a direction with the provenance needed to judge it later.

    JSON, not safetensors: a `d_model` vector is a few thousand floats, and a
    format `modelmri open` can read without torch is worth more here than the
    bytes saved.
    """
    import json

    required = ("model", "layer", "hidden_size", "method", "dtype")
    missing = [k for k in required if meta.get(k) in (None, "")]
    if missing:
        raise BadRequest(
            f"a saved direction needs {', '.join(missing)} — a vector without "
            "its provenance cannot be checked against a model later, and "
            "steering with the wrong one produces plausible nonsense."
        )

    path = store_dir() / f"{_slug(name)}.json"
    payload = {
        "name": name,
        "values": [float(x) for x in direction.detach().float().cpu().tolist()],
        **meta,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"name": name, "path": str(path), "dims": len(payload["values"])}


def load(name: str, *, hidden_size: int, model: str = ""):
    """Read a direction back, refusing a shape it cannot belong to.

    Dimension mismatch is refused by name — the same rule `saes.py` applies to
    `d_in`. A different model with the SAME hidden size is warned about rather
    than blocked: lifting a direction from a base model onto its finetune is a
    legitimate experiment when the person running it knows that is what they
    are doing, and a wrong-but-plausible steer is exactly what the warning is
    for.
    """
    import json

    import torch

    path = store_dir() / f"{_slug(name)}.json"
    if not path.is_file():
        raise Refusal(f"no saved direction called {name!r} in {store_dir()}.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload["values"]
    except (OSError, UnicodeDecodeError, ValueError, KeyError) as err:
        raise Refusal(
            f"{path.name} is not a direction this version can read."
        ) from err

    if len(values) != hidden_size:
        raise Refusal(
            f"{name!r} is a {len(values)}-dimensional direction and this model's "
            f"residual stream is {hidden_size}. It was fitted on "
            f"{payload.get('model', 'another model')}. Refusing rather than "
            "reshaping it into something that would steer, plausibly, at random."
        )

    warnings = []
    origin = payload.get("model", "")
    if model and origin and origin != model:
        warnings.append(
            f"this direction was fitted on {origin} and you are steering "
            f"{model}. The hidden sizes match, but equal size is not equal "
            "basis — the result may be confident and meaningless."
        )
    if payload.get("beats_null") is False:
        warnings.append(
            "this direction did not beat its own label-shuffled null when it "
            "was fitted, so it was never evidence of anything."
        )
    return torch.tensor(values), payload, warnings


def catalogue() -> list[dict]:
    """Every saved direction, without its values. Newest first."""
    import json

    rows = []
    for path in store_dir().glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            # A file this version cannot read is listed as unreadable rather
            # than dropped: a vector silently missing from its own catalogue
            # is worse than one that says it is damaged.
            rows.append({"name": path.stem, "unreadable": True})
            continue
        payload.pop("values", None)
        payload["dims"] = payload.get("hidden_size")
        rows.append(payload)
    return sorted(rows, key=lambda r: r.get("saved_at", ""), reverse=True)
