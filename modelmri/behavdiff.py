# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""What quantisation cost the model's *behaviour*, not just its weights.

`quantdiff.py` answers how far the numbers moved. This answers the question
people actually have, which is whether the model still says the same thing.
They are not the same question and the join between them is the point: a
tensor can move a long way in RMS and change no answer, and a tensor can barely
move and flip the argmax at the one position that mattered.

Three measurements over one identical token sequence:

  * per-position KL between the two next-token distributions, in nats, using
    `ablate.kl_nats` so it is the same quantity the head rankings report;
  * every position where the argmax token flipped, with BOTH candidates and
    both probabilities, because "3 tokens changed" without saying which is a
    statistic about nothing;
  * per-layer mean attention divergence, so damage can be located in depth
    rather than only totalled.

Two models never sit in memory at once. The first is loaded, its outputs are
captured to CPU, and it is torn down before the second is built — on the 8 GB
card this was written on, the alternative is that the comparison only runs for
models small enough to fit twice, which excludes most of the ones worth
comparing.

WHAT THIS DOES NOT MEASURE. It compares two models through HuggingFace's
kernels. llama.cpp will not produce these logits — it has its own kernels for
the quantised types, and this dequantises instead. So this is the quantiser's
damage, not the end-to-end damage of the runtime you would deploy on.

And one prompt is one sample. Every number here describes the prompt it was
taken on, which is why the prompt is in every response and why `positions`
returns the whole per-position series rather than an average that would hide
the one position where the answer changed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from . import fmt
from .errors import BadRequest, Refusal

# Above this many tokens the captured distributions get large: the tensor is
# positions x vocab x 4 bytes, and a 262k vocabulary is 1 MB per position. The
# cap is on the PROMPT rather than on what is reported, and when it bites it is
# recorded in `notes` rather than silently truncating the series.
MAX_POSITIONS = 256

# Attention is layers x heads x seq x seq per model and both sides are held at
# once to difference them. At 64 layers, 64 heads and 256 tokens that is 2.1 GB
# per side, so it is reduced to a per-layer scalar during capture and the full
# grids are never retained.
ATTENTION_NOTE = (
    "Per layer, the mean absolute difference between the two attention "
    "matrices over the identical token sequence. Reduced during capture: the "
    "full grids are never both resident."
)


@dataclass
class PositionDiff:
    index: int
    token: str
    kl: float
    # The argmax on each side, and what each side gave it. Both shown whether
    # or not they differ — "the answer changed" is only readable next to what
    # it changed from.
    top_a: str
    top_b: str
    p_a: float
    p_b: float
    flipped: bool
    # How far ahead the winner was on each side: top-1 minus top-2. A flip
    # where both margins are tiny is a near-tie being broken, which is a very
    # different event from a confident answer changing, and reporting them as
    # the same number of "flips" would overstate the damage.
    #
    # Measured on SmolLM2-135M Q4_K_M against its original, "The capital of
    # France is": the one flip is at ' France', where the original ranked ','
    # at 0.322 against ' is' at 0.319 — a 0.003 margin. Counting that beside a
    # flip at a 0.9 margin would be arithmetic on two different things.
    margin_a: float = 0.0
    margin_b: float = 0.0

    @property
    def contested(self) -> bool:
        """A flip at a near-tie on the reference side. Threshold stated rather
        than tuned: 0.05 of probability mass between first and second."""
        return self.flipped and self.margin_b < 0.05

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "token": self.token,
            "kl": round(self.kl, 6),
            "top_a": self.top_a,
            "top_b": self.top_b,
            "p_a": round(self.p_a, 6),
            "p_b": round(self.p_b, 6),
            "flipped": self.flipped,
            "margin_a": round(self.margin_a, 6),
            "margin_b": round(self.margin_b, 6),
            "contested": self.contested,
        }


@dataclass
class Divergence:
    """The behaviour half of a quantisation damage report."""

    model_a: str
    model_b: str
    prompt: str
    tokens: list[str]
    positions: list[PositionDiff] = field(default_factory=list)
    # layer index -> mean |attention_a - attention_b|. None when either model
    # would not return attention, which is a real state and not a zero.
    attention: list[dict] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def flips(self) -> list[PositionDiff]:
        return [p for p in self.positions if p.flipped]

    def summary(self) -> dict:
        if not self.positions:
            return {
                "positions": 0,
                "means": "nothing was compared; see notes.",
            }
        kls = [p.kl for p in self.positions]
        worst = max(self.positions, key=lambda p: p.kl)
        by_layer = self.attention or []
        worst_layer = (
            max(by_layer, key=lambda r: r["mean_abs_diff"]) if by_layer else None
        )
        contested = [p for p in self.flips if p.contested]
        return {
            "positions": len(self.positions),
            "flips": len(self.flips),
            # Split out, not netted off. A flip is a flip; whether it was a
            # near-tie changes what it MEANS, and the reader gets both numbers
            # rather than one that has quietly decided for them.
            "contested_flips": len(contested),
            "decisive_flips": len(self.flips) - len(contested),
            # Median as well as mean: one position with a large KL drags a mean
            # a long way, and which of those two a reader wants depends on
            # whether they care about the typical token or the worst one.
            "mean_kl": round(sum(kls) / len(kls), 6),
            "median_kl": round(sorted(kls)[len(kls) // 2], 6),
            "max_kl": round(worst.kl, 6),
            "max_kl_at": {"index": worst.index, "token": worst.token},
            "worst_layer": worst_layer,
            "means": (
                "KL is in nats, per position, between the two next-token "
                "distributions over the SAME token ids — the same quantity "
                "`ablate` reports for a head. A flip is a position where the "
                "argmax token changed; both candidates are listed because a "
                "count of flips without them is a statistic about nothing. A "
                "flip is CONTESTED when the reference model's own top-1 beat "
                "its top-2 by under 0.05 — a tie being broken rather than an "
                "answer being changed. "
                "This is one prompt, which is one sample: it describes this "
                "prompt and does not generalise to the model. And it measures "
                "the quantiser through HuggingFace's kernels, NOT llama.cpp's "
                "end-to-end behaviour, which uses its own."
            ),
        }

    def to_dict(self) -> dict:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "prompt": self.prompt,
            "tokens": self.tokens,
            "positions": [p.to_dict() for p in self.positions],
            "flips": [p.to_dict() for p in self.flips],
            "attention": self.attention,
            "attention_means": ATTENTION_NOTE if self.attention else None,
            "notes": self.notes,
            "summary": self.summary(),
        }


# --------------------------------------------------------------- capture


@dataclass
class Capture:
    """One model's outputs, on the CPU, with the model already gone."""

    label: str
    ids: list[int]
    tokens: list[str]
    # positions x vocab, float32, CPU. The distributions, not the logits: the
    # comparison is between distributions and storing logits would mean
    # softmaxing twice with two chances to disagree about temperature.
    probs: Any
    # layer -> per-layer attention averaged over heads, CPU float32. None when
    # the model returned none, which `eager` prevents but a custom
    # architecture might not.
    attention: list[Any] | None
    vocab_size: int
    # The argmax at each position, decoded HERE rather than at comparison time.
    # By the time two captures are differenced both models and both tokenisers
    # are gone -- that is the entire point of capturing -- so an id that was
    # not decoded while the tokeniser existed can never be decoded at all.
    top_ids: list[int] = field(default_factory=list)
    top_texts: list[str] = field(default_factory=list)
    # top-1 minus top-2 at each position. See PositionDiff.margin_a.
    margins: list[float] = field(default_factory=list)


def capture(model, tokenizer, prompt: str, *, label: str, want_attention: bool = True):
    """Run one prompt and keep only what the comparison needs.

    Everything is moved to the CPU before returning so the caller can tear the
    model down and still hold the result. That ordering is the whole reason
    this is a separate function.
    """
    import torch

    from . import ablate

    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    ids = encoded.input_ids.to(device)
    if ids.shape[1] == 0:
        raise BadRequest("that prompt tokenised to nothing")

    with torch.no_grad():
        out = model(ids, output_attentions=want_attention)

    probs = ablate.distribution(out.logits[0]).to("cpu")
    # Decoded now, while there is still a tokeniser to decode with.
    top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1)
    top_ids = [int(i) for i in top2.indices[:, 0]]
    top_texts = [tokenizer.decode([i]) for i in top_ids]
    margins = [
        float(row[0] - row[1]) if row.numel() > 1 else 1.0 for row in top2.values
    ]
    attention = None
    if want_attention and getattr(out, "attentions", None):
        # Kept as full per-layer matrices here and reduced by the caller once
        # BOTH sides exist. Averaged over heads on the way out: the divergence
        # this reports is per layer, and holding heads as well would multiply
        # the retained size by the head count for a number nothing reads.
        attention = [a[0].float().mean(dim=0).to("cpu") for a in out.attentions]

    id_list = [int(i) for i in encoded.input_ids[0]]
    return Capture(
        label=label,
        ids=id_list,
        tokens=[tokenizer.decode([i]) for i in id_list],
        probs=probs,
        attention=attention,
        vocab_size=int(out.logits.shape[-1]),
        top_ids=top_ids,
        top_texts=top_texts,
        margins=margins,
    )


# ------------------------------------------------------------- comparison


def compare_captures(a: Capture, b: Capture, *, prompt: str) -> Divergence:
    """Difference two captures that were taken on the same token sequence.

    Refuses rather than aligns when the two sides disagree about the
    tokenisation or the vocabulary. A KL between distributions over different
    supports, or between positions that hold different tokens, is a number
    with no meaning that looks exactly like one that has.
    """
    import torch

    from . import ablate

    notes: list[str] = []

    if a.ids != b.ids:
        n = min(len(a.ids), len(b.ids))
        first = next(
            (i for i in range(n) if a.ids[i] != b.ids[i]),
            n,
        )
        raise Refusal(
            f"the two models tokenise this prompt differently, so there is no "
            f"shared sequence to compare on. {a.label} produced "
            f"{len(a.ids)} tokens and {b.label} produced {len(b.ids)}; they "
            f"first differ at position {first}"
            + (
                f" ({a.tokens[first]!r} against {b.tokens[first]!r})."
                if first < n
                else "."
            )
            + " A GGUF carries its own tokeniser, and a requantised or "
            "re-converted file can carry a different one from the original."
        )

    if a.vocab_size != b.vocab_size:
        raise Refusal(
            f"vocabulary sizes differ ({a.vocab_size:,} against "
            f"{b.vocab_size:,}), so a KL between these distributions is "
            "undefined — they are not over the same set of outcomes."
        )

    positions: list[PositionDiff] = []
    for i in range(len(a.ids)):
        pa, pb = a.probs[i], b.probs[i]
        # Directional on purpose, and stated: KL(a || b) reads as "how
        # surprised the ORIGINAL is by the quantised model's answer", which is
        # the direction that matches "what did quantising cost".
        kl = ablate.kl_nats(pa, pb)
        ia, ib = a.top_ids[i], b.top_ids[i]
        positions.append(
            PositionDiff(
                index=i,
                token=a.tokens[i],
                kl=float(kl),
                top_a=a.top_texts[i],
                top_b=b.top_texts[i],
                p_a=float(pa[ia]),
                p_b=float(pb[ib]),
                flipped=ia != ib,
                margin_a=a.margins[i] if a.margins else 0.0,
                margin_b=b.margins[i] if b.margins else 0.0,
            )
        )

    attention = None
    if a.attention and b.attention:
        if len(a.attention) != len(b.attention):
            notes.append(
                f"layer counts differ ({len(a.attention)} against "
                f"{len(b.attention)}), so attention was not compared"
            )
        else:
            attention = [
                {
                    "layer": layer,
                    "mean_abs_diff": round(
                        float(torch.abs(x - y).mean()),
                        8,
                    ),
                }
                for layer, (x, y) in enumerate(
                    zip(a.attention, b.attention, strict=True)
                )
            ]
    else:
        notes.append(
            "attention was not returned by both models, so per-layer "
            "divergence is unavailable — not zero"
        )

    return Divergence(
        model_a=a.label,
        model_b=b.label,
        prompt=prompt,
        tokens=a.tokens,
        positions=positions,
        attention=attention,
        notes=notes,
    )


def _cap_prompt(prompt: str, tokenizer, notes: list[str]) -> str:
    """Refuse to capture an unbounded number of positions.

    The retained tensor is positions x vocab x 4 bytes, and a 262k vocabulary
    costs about 1 MB per position, so a long prompt quietly turns into a large
    allocation on both sides at once. Truncation is recorded rather than done
    silently: a divergence report over the first 256 tokens of a 4,000-token
    prompt is not a report about that prompt.
    """
    ids = tokenizer(prompt, return_tensors=None)["input_ids"]
    if len(ids) <= MAX_POSITIONS:
        return prompt
    notes.append(
        f"prompt is {len(ids)} tokens; compared the first {MAX_POSITIONS}. "
        f"The retained distributions are positions x vocabulary, so the whole "
        f"prompt would hold roughly "
        f"{fmt.bytes_si(len(ids) * getattr(tokenizer, 'vocab_size', 32000) * 4)} "
        f"per side."
    )
    return tokenizer.decode(ids[:MAX_POSITIONS])


# ----------------------------------------------------------- orchestration


@dataclass
class Side:
    """One thing to compare: a GGUF file, or a HuggingFace id or directory."""

    spec: str
    kind: str  # "gguf" | "hf"

    @property
    def label(self) -> str:
        from pathlib import Path

        return Path(self.spec).name if self.kind == "gguf" else self.spec


# A HuggingFace repo id: `name` or `namespace/name`, from a small alphabet
# and at most one slash. Anchored, so anything carrying a drive letter, a
# backslash, a leading slash or a `..` fails it and is treated as a path.
HUB_ID = re.compile(r"^[A-Za-z0-9][\w.-]*(?:/[A-Za-z0-9][\w.-]*)?$")


def is_hub_id(spec: str) -> bool:
    """Whether this names a hub repo, decided by SHAPE and nothing else.

    Deliberately not `Path(spec).exists()`. That question cannot be asked
    about caller-supplied text without answering it: a path that exists takes
    one branch and a path that does not takes another, and the two produce
    different errors, so anyone who can call the route can test for the
    existence of any file on the machine. Small, but it is a primitive, and
    the server has no reason to hand one out.

    Shape also fails in the safe direction. A local directory that happens to
    match this pattern goes down the hub branch and is refused by the hub for
    not existing; a hub id can never be mistaken for a path and skip the roots
    gate, which is the failure that would matter.
    """
    spec = (spec or "").strip()
    return bool(spec) and ".." not in spec and bool(HUB_ID.match(spec))


def side(spec: str) -> Side:
    """Classify a side by what it is, not by what the caller called it."""
    from pathlib import Path

    p = Path(spec)
    if p.suffix.lower() == ".gguf" or (p.is_dir() and any(p.glob("*.gguf"))):
        return Side(spec=spec, kind="gguf")

    # A FILE that is not a GGUF is not a HuggingFace side either, and calling
    # it one is this function failing at the job its own docstring names. A
    # Hub id is a NAME; a local model is a DIRECTORY of config and weights.
    # MEASURED: pointing `quantised` at README.md classified it "hf", handed
    # it to `AutoTokenizer.from_pretrained`, and answered 500 carrying
    # transformers' own "It looks like the config file at '…README.md' is not
    # a valid JSON file" — an error about JSON, at somebody who picked the
    # wrong file.
    if p.is_file():
        raise BadRequest(
            f"`{p.name}` is a single file and not a GGUF, so there is no model "
            f"here to run. This side takes a `.gguf`, a directory holding a "
            f"model's config and weights, or a Hub id — {p.suffix or 'that name'} "
            f"is none of them."
        )
    return Side(spec=spec, kind="hf")


def _build(s: Side, *, dtype: str, device: str, device_kind: str):
    """Load one side. Returns (model, tokenizer)."""
    import torch

    if s.kind == "gguf":
        from . import gguf_load

        loaded = gguf_load.load(
            s.spec, dtype=dtype, device=device, device_kind=device_kind, confirm=True
        )
        return loaded.model, loaded.tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(s.spec)
    model = AutoModelForCausalLM.from_pretrained(
        s.spec,
        dtype=getattr(torch, dtype),
        # Same reason as everywhere else in this package: sdpa returns
        # attentions=None without saying so, and the per-layer divergence is a
        # third of what this feature reports.
        attn_implementation="eager",
    ).to(device)
    model.eval()
    return model, tok


def _release(model) -> None:
    """Give the accelerator memory back before the other side is built.

    `del` here would only unbind this function's parameter — the caller's name
    still references the model, so the collector frees nothing and
    `empty_cache()` has nothing to release. `nullmodel.teardown` learned that
    the hard way (255.3 MB still allocated after it claimed to have freed it),
    so this reuses it rather than re-deriving it.
    """
    from . import nullmodel

    nullmodel.teardown(model)


def compare_behaviour(
    a: str,
    b: str,
    prompt: str,
    *,
    dtype: str = "bfloat16",
    device: str = "cpu",
    device_kind: str = "cpu",
    want_attention: bool = True,
    on_stage=None,
) -> Divergence:
    """Load, capture, release, load, capture, compare.

    The order is the feature. Both models resident at once would double the
    requirement, and the models worth comparing are exactly the ones already
    near the limit of the machine — a comparison that only runs when the model
    fits twice would refuse every case it was built for.
    """
    sa, sb = side(a), side(b)
    if sa.spec == sb.spec:
        raise BadRequest(
            "both sides are the same file, so every difference would be zero "
            "by construction. Point this at a quantised model and its "
            "full-precision original, or at two different quantisations."
        )

    notes: list[str] = []
    captures = []
    for s in (sa, sb):
        if on_stage:
            on_stage("load", f"{s.label} ({s.kind})")
        model, tok = _build(s, dtype=dtype, device=device, device_kind=device_kind)
        try:
            # Capped once, on the first side, and the SAME text is then used
            # for the second — re-capping per side could hand the two models
            # different prompts, which is the one thing this must not do.
            if not captures:
                prompt = _cap_prompt(prompt, tok, notes)
            if on_stage:
                on_stage("capture", s.label)
            captures.append(
                capture(
                    model, tok, prompt, label=s.label, want_attention=want_attention
                )
            )
        finally:
            # In a `finally`: a capture that raises must still give the memory
            # back, or the second side has nowhere to load into and the real
            # error is buried under an out-of-memory.
            _release(model)
            del model, tok

    result = compare_captures(captures[0], captures[1], prompt=prompt)
    result.notes = notes + result.notes
    return result
