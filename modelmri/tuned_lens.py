"""A lens trained on your own text, shown beside the plain one — never instead.

The plain logit lens reads the residual stream through the unembedding it was
never trained for, and #4 made the cost of that visible: a held-out KL per
layer saying how far each row is from the model's real distribution. On some
models that number is large enough that the early rows are not describing the
model at all.

A tuned lens fixes the mismatch by learning a per-layer affine map -- the
Belrose objective, minimising KL(final ‖ head(norm(A_L·h_L + b_L))) -- so each
layer is read through a transform fitted to that layer instead of one fitted to
the last. It is trained here, on this machine, on text you provide.

WHY BOTH COLUMNS STAY ON SCREEN

A tuned lens can be trained to look confident everywhere. That is not a risk
of the method, it is the method: a translator with 590K parameters per layer,
fitted to minimise disagreement with the final distribution, will reduce
disagreement with the final distribution. If it silently replaced the plain
lens, every early layer would suddenly look like it already knew the answer,
and there would be nothing on screen to say whether that is the model or the
translator.

So both rows are always rendered, and the number that decides which to believe
is **held-out KL** -- measured on sequences the translator never saw. Training
KL is not reported anywhere in this module, because a translator's training KL
is a statement about the translator.

WHAT A TUNED LENS IS FOR

A lens trained on 200 sequences of your own text is a lens FOR THAT TEXT. The
corpus is part of the measurement: its hash and token count are in the cache
key and in every response, so two tuned readings taken against different
corpora can never be silently compared.

Nothing is fetched. Pretrained lenses exist on the Hub and downloading one
would break the offline-first promise the rest of this package keeps, so
training is a local, explicit action and there is no code here that reaches the
network.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import paths
from .errors import BadRequest, Refusal

# The share of sequences held back from training. The reported improvement is
# measured only on these, so it has to be big enough to mean something and
# small enough to leave a corpus worth training on.
HELD_OUT_SHARE = 0.25

# Below this many sequences there is no honest split: one held-out sequence
# gives a held-out KL with no spread, which is the number this module exists
# to make trustworthy.
MIN_SEQUENCES = 8

# A translator is d_model x d_model per layer. At 768 that is 590K parameters;
# at 4096 it is 16.8M per layer and a 32-layer model would be 537M parameters
# of translator in fp32 -- 2.1 GB, on a machine chosen for having 8 GB. The
# guard is on the projected total and it prints the number.
MAX_TRANSLATOR_PARAMS = 200_000_000

FORMAT = "modelmri-tuned-lens"
FORMAT_VERSION = 1


@dataclass
class LayerFit:
    """What one layer's translator cost and bought, on held-out text."""

    layer: int
    plain_kl: float
    tuned_kl: float

    @property
    def gain(self) -> float:
        """Nats the translator recovered. Negative means it made this worse."""
        return self.plain_kl - self.tuned_kl

    def to_dict(self) -> dict:
        return {**asdict(self), "gain": self.gain}


@dataclass
class TunedLensInfo:
    """Everything about a trained lens except its weights."""

    model_id: str
    dtype: str
    n_layers: int
    d_model: int
    corpus_label: str
    corpus_sha256: str
    n_sequences: int
    n_tokens: int
    n_held_out: int
    steps: int
    lr: float
    seconds: float
    layers: list[LayerFit] = field(default_factory=list)

    @property
    def helped(self) -> list[LayerFit]:
        return [row for row in self.layers if row.gain > 0]

    @property
    def tokens_per_parameter(self) -> float:
        """How much text each translator parameter was fitted from.

        The number that says how seriously to take this lens. A translator is
        d_model x d_model per layer -- 590K parameters at d_model 768 -- and a
        corpus of a few thousand tokens leaves it orders of magnitude
        under-determined. The held-out KL is still a real measurement of what
        the lens does on unseen text; this says how narrow the text it learnt
        from was.
        """
        per_layer = self.d_model * self.d_model + self.d_model
        return (self.n_tokens / per_layer) if per_layer else 0.0

    @property
    def caution(self) -> str:
        """Empty when the corpus is not conspicuously small for the fit."""
        ratio = self.tokens_per_parameter
        if ratio >= 1.0:
            return ""
        return (
            f"This lens was fitted from {self.n_tokens:,} tokens onto "
            f"{self.d_model * self.d_model + self.d_model:,} parameters per "
            f"layer — {ratio:.4f} tokens per parameter. The held-out numbers "
            f"below are real, and they are about text like the training text. "
            f"Read this as a lens for that corpus, not as a lens for the "
            f"model."
        )

    def to_dict(self) -> dict:
        return {
            **{k: v for k, v in asdict(self).items() if k != "layers"},
            "layers": [row.to_dict() for row in self.layers],
            "n_layers_improved": len(self.helped),
            "tokens_per_parameter": round(self.tokens_per_parameter, 6),
            "caution": self.caution,
            "means": self.means(),
        }

    def means(self) -> str:
        """The sentence that has to travel with these numbers."""
        improved = len(self.helped)
        return (
            f"Held-out KL on {self.n_held_out} sequences the translator never "
            f"saw, in nats. The tuned lens is closer to the model on "
            f"{improved} of {len(self.layers)} layers. It was trained on "
            f"{self.corpus_label} ({self.n_tokens:,} tokens), so it is a lens "
            f"for that text — a translator fitted to minimise disagreement "
            f"with the final distribution will reduce disagreement with the "
            f"final distribution, and the plain lens stays on screen so you "
            f"can see which rows the training actually moved."
        )


def corpus_hash(texts: list[str]) -> str:
    """A stable id for the corpus a lens was fitted to.

    Order-independent and content-only, so the same set of sequences read from
    a `.txt` and from the trace store produces the same lens rather than two
    caches of the same thing.
    """
    digest = hashlib.sha256()
    for text in sorted(texts):
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def cache_path(model_id: str, dtype: str, sha: str, n_tokens: int) -> Path:
    """Where a lens for this exact (model, dtype, corpus, size) lives.

    Every one of the four is in the name because every one of them changes the
    lens. A translator fitted in bf16 is not the one fitted in fp32, and a lens
    for 40,000 tokens of your logs is not the lens for 400,000 of them.
    """
    safe = model_id.replace("/", "--").replace("\\", "--")
    return (
        paths.cache_dir()
        / "tuned-lenses"
        / f"{safe}.{dtype}.{sha}.{n_tokens}.safetensors"
    )


def _split(texts: list[str]) -> tuple[list[str], list[str]]:
    """(train, held out), deterministically.

    Sorted-then-strided rather than shuffled: the same corpus must produce the
    same split on every machine, or a cached lens and a freshly trained one
    would report different held-out numbers for the same inputs and there
    would be no way to tell which.
    """
    ordered = sorted(texts)
    every = max(2, round(1 / HELD_OUT_SHARE))
    held = ordered[::every]
    train = [t for t in ordered if t not in set(held)]
    return train, held


def _projection(model):
    """(final norm, unembedding) or a refusal naming what is missing."""
    from .lens import _final_norm

    head = model.get_output_embeddings()
    if head is None:
        raise Refusal(
            "this model has no output embedding to project through, so "
            "neither lens can be read on it."
        )
    return _final_norm(model), head


def plan(model, texts: list[str]) -> dict:
    """What training will cost, before any of it is paid."""
    config = getattr(model, "config", None)
    d_model = int(getattr(config, "hidden_size", 0) or 0)
    n_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    if not d_model or not n_layers:
        raise Refusal(
            "this model does not report a hidden size and layer count, so the "
            "size of a translator for it cannot be projected."
        )
    per_layer = d_model * d_model + d_model
    total = per_layer * n_layers
    train, held = _split(texts)
    return {
        "d_model": d_model,
        "n_layers": n_layers,
        "params_per_layer": per_layer,
        "params_total": total,
        "bytes_fp32": total * 4,
        "n_sequences": len(texts),
        "n_train": len(train),
        "n_held_out": len(held),
        "affordable": total <= MAX_TRANSLATOR_PARAMS,
    }


def train(
    model,
    tokenizer,
    texts: list[str],
    *,
    corpus_label: str,
    steps: int = 250,
    lr: float = 1e-3,
    max_length: int = 128,
    on_progress=None,
):
    """Fit one affine translator per layer, and measure it on held-out text.

    Returns `(info, state)` — the report and the tensors to save. Blocking;
    call from a worker thread.
    """
    import torch

    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(texts) < MIN_SEQUENCES:
        raise BadRequest(
            f"a tuned lens needs at least {MIN_SEQUENCES} sequences to hold "
            f"any of them back for an honest measurement, and this corpus has "
            f"{len(texts)}. The number that decides whether a tuned lens is "
            f"worth believing is its KL on text it never saw."
        )

    projected = plan(model, texts)
    if not projected["affordable"]:
        raise Refusal(
            f"a translator for this model is "
            f"{projected['params_total'] / 1e6:,.0f}M parameters "
            f"({projected['bytes_fp32'] / 1e9:.1f} GB in fp32), above the "
            f"{MAX_TRANSLATOR_PARAMS / 1e6:,.0f}M this trains. A tuned lens "
            f"for a model this wide is a training run, not a panel."
        )

    norm, head = _projection(model)
    head_dtype = next(head.parameters()).dtype
    device = next(model.parameters()).device
    d_model = projected["d_model"]
    n_layers = projected["n_layers"]

    train_texts, held_texts = _split(texts)
    started = time.perf_counter()

    def hidden_and_target(batch: list[str]):
        """Per-layer hidden states and the model's own final distribution."""
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            out = model(**encoded, output_hidden_states=True)
            # float32 throughout the objective. The translator is fitted in
            # fp32 even when the model runs in bf16: a KL is a difference of
            # logs, and bf16 has ~3 decimal digits — the gradient of a
            # difference that small is noise in that format.
            target = torch.log_softmax(out.logits.float(), dim=-1)
            states = [h.float() for h in out.hidden_states]
        mask = encoded["attention_mask"].bool()
        return states, target, mask

    translators = []
    for _ in range(n_layers):
        # Identity at initialisation, so an untrained translator IS the plain
        # lens. Starting from random would make step 0 worse than doing
        # nothing, and the first thing a reader would see is a lens that had
        # made everything worse.
        weight = torch.eye(d_model, device=device, dtype=torch.float32)
        bias = torch.zeros(d_model, device=device, dtype=torch.float32)
        weight.requires_grad_(True)
        bias.requires_grad_(True)
        translators.append((weight, bias))

    params = [p for pair in translators for p in pair]
    optimiser = torch.optim.Adam(params, lr=lr)

    # THE MODEL'S OWN PARAMETERS ARE FROZEN FOR THE DURATION. Gradient still
    # flows THROUGH the norm and the unembedding -- it has to, that is the
    # path to the translator -- but with requires_grad left on, every backward
    # pass also accumulated a full gradient into the model's weights: a second
    # copy of the model in memory, on a machine chosen for having 8 GB, for
    # numbers no optimiser here would ever read. Restored in the `finally`
    # below, because this model is the runtime's and is still loaded after.
    was_training = {}
    for name, parameter in model.named_parameters():
        was_training[name] = parameter.requires_grad
        parameter.requires_grad_(False)
    try:
        batch_size = max(1, min(4, len(train_texts)))
        for step in range(steps):
            start = (step * batch_size) % max(1, len(train_texts))
            batch = train_texts[start : start + batch_size] or train_texts[:batch_size]
            states, target, mask = hidden_and_target(batch)

            optimiser.zero_grad(set_to_none=True)
            loss_total = 0.0
            for layer in range(n_layers):
                weight, bias = translators[layer]
                translated = states[layer] @ weight.T + bias
                logits = head(norm(translated.to(head_dtype)))
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                # KL(final ‖ tuned), summed over the vocabulary and averaged
                # over real tokens only. Padding positions carry no
                # information and would otherwise pull every translator
                # towards the pad embedding.
                per_token = (target.exp() * (target - log_probs)).sum(-1)
                loss = per_token[mask].mean()
                # No retain_graph: `states` come out of a no_grad forward, so
                # each layer's graph is its own and freeing it is the point.
                loss.backward()
                # .detach() before float(): the bare conversion warns, and the
                # warning is right -- keeping a reference to a grad-tracking
                # tensor in a running total is how a training loop holds the
                # whole graph alive.
                loss_total += float(loss.detach())
            optimiser.step()
            if on_progress is not None and step % 10 == 0:
                on_progress(step, steps, loss_total / n_layers)
    finally:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(was_training.get(name, False))

    # ---- the only number that counts: held-out ----
    import torch as _torch

    fits: list[LayerFit] = []
    with _torch.no_grad():
        states, target, mask = hidden_and_target(held_texts[:8])
        for layer in range(n_layers):
            weight, bias = translators[layer]
            hidden = states[layer]

            def kl_of(stream):
                logits = head(norm(stream.to(head_dtype)))
                log_probs = _torch.log_softmax(logits.float(), dim=-1)
                per_token = (target.exp() * (target - log_probs)).sum(-1)
                return float(per_token[mask].mean())

            fits.append(
                LayerFit(
                    layer=layer,
                    plain_kl=round(kl_of(hidden), 5),
                    tuned_kl=round(kl_of(hidden @ weight.T + bias), 5),
                )
            )

    n_tokens = sum(len(tokenizer(t)["input_ids"]) for t in texts)
    info = TunedLensInfo(
        model_id="",  # filled by the caller, which knows the id
        dtype=str(next(model.parameters()).dtype).removeprefix("torch."),
        n_layers=n_layers,
        d_model=d_model,
        corpus_label=corpus_label,
        corpus_sha256=corpus_hash(texts),
        n_sequences=len(texts),
        n_tokens=n_tokens,
        n_held_out=len(held_texts),
        steps=steps,
        lr=lr,
        seconds=round(time.perf_counter() - started, 1),
        layers=fits,
    )
    state = {}
    for layer, (weight, bias) in enumerate(translators):
        state[f"A.{layer}"] = weight.detach().cpu().contiguous()
        state[f"b.{layer}"] = bias.detach().cpu().contiguous()
    return info, state


def save(info: TunedLensInfo, state: dict, path: str | Path) -> Path:
    """Weights as safetensors, the report as JSON beside them."""
    from safetensors.torch import save_file

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # The report rides in the safetensors metadata AND as a sidecar. The
    # metadata is what makes a lens file self-describing when it is moved; the
    # sidecar is what makes it readable without torch.
    save_file(state, str(target), metadata={"modelmri": json.dumps(info.to_dict())})
    target.with_suffix(".json").write_text(
        json.dumps({"format": FORMAT, "version": FORMAT_VERSION, **info.to_dict()}),
        encoding="utf-8",
    )
    return target


def load(path: str | Path, *, model_id: str, dtype: str):
    """A saved lens, refusing one that was fitted to something else.

    A translator is only meaningful for the model and dtype it was fitted to.
    Loading one across either would produce a confident, plausible, entirely
    wrong reading -- the exact failure the plain lens's double-norm bug was.
    """
    from safetensors import safe_open

    target = Path(path)
    if not target.is_file():
        raise BadRequest(f"no tuned lens at {target.name}")

    with safe_open(str(target), framework="pt") as handle:
        meta = handle.metadata() or {}
        state = {key: handle.get_tensor(key) for key in handle.keys()}
    try:
        info = json.loads(meta.get("modelmri") or "{}")
    except json.JSONDecodeError:
        info = {}

    stored_model = info.get("model_id")
    stored_dtype = info.get("dtype")
    if stored_model and stored_model != model_id:
        raise Refusal(
            f"this lens was fitted to {stored_model} and the loaded model is "
            f"{model_id}. A translator is only meaningful for the model it "
            f"was trained on."
        )
    if stored_dtype and stored_dtype != dtype:
        raise Refusal(
            f"this lens was fitted in {stored_dtype} and this model is running "
            f"in {dtype}. The translator was fitted to that arithmetic."
        )
    return info, state


def read(model, tokenizer, ids, state: dict, top_k: int = 5) -> dict:
    """The tuned reading of one run, layer by layer.

    Deliberately shaped like `lens.logit_lens`'s output so the panel renders
    the two columns with one component, and a reader comparing them is
    comparing like with like.
    """
    import torch

    norm, head = _projection(model)
    device = next(model.parameters()).device
    ids = ids.to(device)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)

    rows = []
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        final = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        for layer in range(len(out.hidden_states) - 1):
            weight = state.get(f"A.{layer}")
            bias = state.get(f"b.{layer}")
            if weight is None or bias is None:
                # A lens missing a layer is reported as missing rather than
                # filled with the plain reading, which would look like a tuned
                # row and be one.
                continue
            hidden = out.hidden_states[layer][0, -1].float().to(weight.device)
            translated = hidden @ weight.T + bias
            logits = head(norm(translated.to(device).to(next(head.parameters()).dtype)))
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            probs = log_probs.exp()
            top = torch.topk(probs, min(top_k, probs.shape[-1]))
            rows.append(
                {
                    "layer": layer,
                    "tokens": [tokenizer.decode([int(i)]) for i in top.indices],
                    "probs": [round(float(p), 5) for p in top.values],
                    "entropy": round(float(-(probs * log_probs).sum()), 4),
                    "kl_to_final": round(
                        float((final.exp() * (final - log_probs)).sum()), 5
                    ),
                }
            )
    # ONE FEWER ROW THAN THE PLAIN LENS, on purpose. `hidden_states` is
    # n_layers + 1 entries -- the embedding output plus every block -- and the
    # last of them is the model's own final state, which needs no translator
    # because it IS the answer. A caller aligns the two columns by `layer`,
    # never by index, and the final plain row simply has no tuned counterpart.
    return {
        "layers": rows,
        "kind": "tuned",
        "n_layers": len(rows),
        "align": (
            "align these rows to the plain lens by `layer`, not by position: "
            "the plain lens has one more row, the model's own final state, "
            "which has no translator because it is not a prediction about the "
            "answer — it is the answer."
        ),
    }
