"""Did the answer come from the document, or from the weights?

Every local-LLM app with RAG shows you WHICH CHUNKS WERE RETRIEVED. None of
them tells you whether the answer used them. Those are different questions and
only the second one is about the model: a retriever that pulled the right
paragraph and a model that ignored it and answered from memory look identical
in every one of those interfaces, and that combination is exactly what a
confident hallucination looks like from the outside.

So this measures two things per chunk and reports them SIDE BY SIDE, because
the interesting cases are the ones where they disagree:

    dependence   how far the answer's next-token distribution moves when that
                 chunk is masked out of attention, in nats
    attention    how much of the answer position's attention actually landed
                 on that chunk's tokens

A chunk the model LOOKED AT AND DID NOT DEPEND ON is called out by name. That
is the signature the roadmap is after — attention on the passage, no causal
dependence on it, answer therefore coming from somewhere else.

WHAT THIS IS NOT
----------------
Not a retrieval engine. There are no embeddings, no index and no similarity
score anywhere in this file. Chunking is by blank line and heading, spans come
from the tokenizer's own offset mapping, and every chunk you hand it is in the
prompt. Retrieval is somebody else's job; this is about what the model did
with what it was given.

THE UNITS ARE NATS AND THEY DO NOT ADD UP
-----------------------------------------
Masking a whole chunk is a much bigger intervention than masking one token,
and the effects are not additive: two chunks that each move the answer by 0.4
nats do not jointly move it by 0.8, and the joint pass here usually shows
that. So nothing in this module is ever presented as a percentage share of the
answer. "Removing this chunk moved the answer by X nats" is the only claim the
measurement supports.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from . import fmt
from .ablate import distribution, kl_nats
from .errors import BadRequest

# One forward pass per chunk plus the fixed passes below, so this is the knob
# that decides whether a request is a click or a job. REFUSED past it rather
# than truncated: an answer scored against the first twelve paragraphs of a
# forty-paragraph document, presented as "grounding", is worse than no answer
# — the eight chunks it actually used might all be in the tail.
MAX_CHUNKS = 24

# Below this many characters a "chunk" is a heading or a stray line, and
# masking it measures the formatting rather than the content. Merged forward
# into the next chunk instead of scored on its own.
MIN_CHUNK_CHARS = 40

# Headings start a chunk even without a blank line before them, because a
# section title belongs with its section and not with the paragraph above it.
_HEADING = re.compile(r"^\s{0,3}(#{1,6}\s+\S|\d+[.)]\s+\S|[A-Z][^\n]{0,60}:\s*$)")


# A `Chunk` dataclass carrying (index, text, start, end, n_tokens) lived here
# and was never constructed: `measure` works off the parallel lists `split`
# and `locate` return, and the only span facts a caller needs come back inside
# `Score`. Deleted rather than left as a shape somebody would later assume was
# the real one.


@dataclass
class Score:
    """What one chunk did to the answer, and what the answer looked at."""

    index: int
    preview: str
    n_tokens: int
    # Nats. How far the answer moved when this chunk was masked out.
    dependence: float
    # Share of the answer position's attention mass that landed here, meaned
    # over every layer and head. A DESCRIPTION of where the model looked, not
    # a claim that looking caused anything -- which is the entire reason it is
    # reported beside `dependence` rather than instead of it.
    #
    # None when this model's attention implementation does not expose it.
    # NOT 0.0: a model whose scores were never returned and a passage nothing
    # looked at are different facts, and printing them the same way is how
    # "unknown" becomes "measured zero" on a page that claims to measure.
    attention: float | None
    # Did the dependence clear the floor the same model produced on a pass
    # that changed nothing?
    depended_on: bool
    # Looked at, and not depended on. The interesting one -- and the one
    # easiest to claim by accident.
    #
    # None means NOT DETERMINABLE on this run, and there are two ways to get
    # there: no attention half, so there is no "looked at"; or a noise floor
    # of exactly 0.0, so there is no "not depended on". False would read as
    # "this passage does not have that problem", which is a claim, and on
    # either of those runs nothing measured it.
    looked_not_used: bool | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Grounding:
    question: str
    answer: str
    answer_p: float
    position: int
    chunks: list[Score] = field(default_factory=list)
    n_chunks: int = 0
    n_prompt_tokens: int = 0
    noise_floor: float = 0.0
    joint: float = 0.0
    # None, not 0.0, when the attention half could not be measured at all.
    attention_share: float | None = None
    # Which half of the feature actually ran. SDPA and FlashAttention never
    # materialise the score matrix, and transformers returns an EMPTY TUPLE
    # rather than None for `output_attentions=True` under them -- so the
    # naive loop over it completes, sums nothing, and reports 0.0 for every
    # passage. Measured exactly that way on a plain from_pretrained load.
    attention_available: bool = True
    attention_note: str = ""
    # The repeat pass reproduced the answer BIT FOR BIT, so the floor is
    # exactly 0.0 and "cleared the floor" degrades to "moved the answer at
    # all" -- a much weaker claim wearing the same words.
    #
    # MEASURED in float32 on CPU: floor 0.0, and all five passages
    # cleared it, including one two orders of magnitude below the top score. On
    # cuda/bf16 the same pass does not reproduce and the floor is real. The
    # degenerate case is named rather than papered over with an invented
    # threshold, for the same reason the probe names a saturated null.
    floor_degenerate: bool = False
    passes: int = 0
    seconds: float = 0.0
    # True when NOTHING cleared the floor. Kept as its own field rather than
    # left to the caller to derive from an empty list, because "no chunk
    # mattered" is a finding and the shape of the response should not make it
    # look like a missing result.
    ungrounded: bool = False

    def to_dict(self) -> dict:
        out = asdict(self)
        out["means"] = self.means()
        return out

    def means(self) -> str:
        looked = [c for c in self.chunks if c.looked_not_used]
        used = [c for c in self.chunks if c.depended_on]
        parts = [
            f"Each of your {self.n_chunks} passages was masked out of the "
            f"model's attention and the answer re-read at the same position. "
            f"The numbers are NATS, and they do not add up: masking two "
            f"passages together moved the answer by {fmt.measured(self.joint)}, not by "
            f"the sum of their separate scores, so none of this is a "
            f"percentage share of the answer.",
        ]
        if self.ungrounded:
            parts.append(
                "NO PASSAGE CLEARED THE NOISE FLOOR. Removing any one of them "
                "moved the answer no further than a pass that changed nothing "
                f"({fmt.measured(self.noise_floor)} nats), so on this evidence the "
                "answer did not depend on the document you attached. That is "
                "a measurement, not a verdict on whether it is correct."
            )
        elif self.floor_degenerate:
            top = max((c.dependence for c in self.chunks), default=0.0)
            parts.append(
                "THE NOISE FLOOR IS EXACTLY ZERO: this model reproduced its "
                "own answer bit for bit, so every passage that moved it at "
                f"all counts as clearing, and {len(used)} of {self.n_chunks} "
                # THROUGH `fmt.measured`, like `joint` and `noise_floor`
                # fourteen lines above. This branch is the one that says "Read
                # the nats, not the verdict" — and `:.4f` floored those very
                # nats to "0.0000" for anything under 5e-5, which is the
                # ordinary size here: the branch is entered precisely when the
                # model reproduced its answer bit for bit, so the movements
                # that "cleared" a floor of zero are the tiny ones.
                "did. Read the nats, not the verdict — the largest here is "
                f"{fmt.measured(top, 4)} and the smallest that 'cleared' is "
                f"{fmt.measured(min((c.dependence for c in used), default=0.0), 4)}"
                ". There is no significance test on this run."
            )
        else:
            names = ", ".join(f"#{c.index}" for c in used[:4])
            parts.append(
                f"{len(used)} of {self.n_chunks} passages cleared the floor "
                f"({names}) — removing those moved the answer further than "
                f"the model's own run-to-run spread "
                f"({fmt.measured(self.noise_floor, 4)})."
            )
        if any(c.looked_not_used is None for c in self.chunks):
            why = (
                "the attention half did not run"
                if not self.attention_available
                else "the noise floor is exactly zero, so every passage that "
                "moved the answer at all counts as depended-on and the flag "
                "could never fire"
            )
            parts.append(
                "LOOKED AT BUT NOT DEPENDED ON COULD NOT BE DECIDED HERE — "
                f"{why}. No passage is flagged, and that is the absence of a "
                "test rather than a clean result."
            )
        elif looked:
            names = ", ".join(f"#{c.index}" for c in looked[:4])
            parts.append(
                f"LOOKED AT, NOT DEPENDED ON: {names}. The answer position "
                "put attention on those passages and removing them changed "
                "nothing measurable — attention is where the model looked, "
                "not what it used, and this pair disagreeing is what an "
                "answer coming from the weights looks like from outside."
            )
        if not self.attention_available:
            # Says what is MISSING and why, and stops there. The line above
            # already reported that the looked-at-not-depended-on reading
            # could not be taken; repeating it here made the same sentence
            # appear twice in one paragraph.
            parts.append(
                f"THE ATTENTION HALF DID NOT RUN — {self.attention_note} Every "
                "passage above carries a dependence score and no attention "
                "share. Blank is what was measured; a zero would have been a "
                "claim."
            )
        else:
            parts.append(
                f"Attention shares are meaned over every layer and head and "
                f"cover {(self.attention_share or 0.0) * 100:.1f}% of the "
                "answer position's mass; the rest went to the question, the "
                "template and any position not inside a passage."
            )
        return " ".join(parts)


def split(text: str) -> list[str]:
    """Blank lines and headings, and nothing cleverer.

    A semantic splitter would be a model, and then the grounding report would
    be partly about THAT model's opinion of where a passage ends. The rule
    here is one a reader can check by eye against their own file.
    """
    if not text.strip():
        raise BadRequest("there is no text to ground the answer in.")

    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        if current and _HEADING.match(line):
            blocks.append("\n".join(current))
            current = [line]
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))

    # A heading on its own is not a passage. Merged FORWARD, into the section
    # it introduces, rather than dropped -- dropping it would put the section
    # title outside every span and make it unmaskable.
    merged: list[str] = []
    held = ""
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        candidate = f"{held}\n{block}".strip() if held else block
        if len(candidate) < MIN_CHUNK_CHARS:
            held = candidate
            continue
        merged.append(candidate)
        held = ""
    if held:
        # Nothing left to merge into, so it stands alone rather than vanishing.
        if merged:
            merged[-1] = f"{merged[-1]}\n{held}"
        else:
            merged.append(held)
    return merged


def build(chunks: list[str], question: str) -> tuple[str, list[tuple[int, int]]]:
    """The prompt, and each chunk's half-open CHARACTER span inside it.

    Character spans, not token spans: token spans come later from the
    tokenizer's offset mapping over this exact string, which is the only way
    to be sure a span covers the tokens the model actually saw.
    """
    if not question.strip():
        raise BadRequest("ask a question — grounding is about one answer.")
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    at = 0
    for chunk in chunks:
        spans.append((at, at + len(chunk)))
        parts.append(chunk)
        at += len(chunk) + 2  # the "\n\n" joined below
    prompt = "\n\n".join(parts) + "\n\n" + question.strip()
    return prompt, spans


def locate(
    tokenizer: Any, prompt: str, spans: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Character spans to token spans, via the tokenizer's own offsets.

    Refuses rather than guesses. A slow tokenizer has no offset mapping, and
    the fallback everybody writes — re-tokenise the chunk and search for the
    id sequence — is wrong in a way that does not announce itself: the same
    words tokenise differently after a preceding space, so the search misses
    and the chunk comes back unmaskable, or worse, matches at the wrong place.
    """
    encoding = tokenizer(prompt, return_offsets_mapping=True, return_tensors=None)
    offsets = encoding.get("offset_mapping")
    if not offsets:
        raise BadRequest(
            "this tokenizer does not report character offsets, so ModelMRI "
            "cannot say which tokens belong to which passage. Grounding needs "
            "a fast tokenizer."
        )

    out: list[tuple[int, int]] = []
    for lo, hi in spans:
        start = end = None
        for index, (a, b) in enumerate(offsets):
            if a == b:  # a special token occupies no characters
                continue
            if b > lo and start is None:
                start = index
            if a < hi:
                end = index + 1
        if start is None or end is None or end <= start:
            raise BadRequest(
                "a passage did not map onto any tokens, which means the "
                "prompt this was measured against is not the prompt that was "
                "built. Nothing is reported rather than something wrong."
            )
        out.append((start, end))
    return out


def measure(
    model: Any,
    tokenizer: Any,
    chunks: list[str],
    question: str,
    *,
    device: Any,
    max_chunks: int = MAX_CHUNKS,
) -> Grounding:
    """Score every passage by dependence and by attention, on one answer.

    Cost is `n_chunks + 4` forward passes: one to read the answer, one repeat
    of it for the noise floor, one with attentions retained, one joint mask,
    and one per chunk.
    """
    if not chunks:
        raise BadRequest("there is no text to ground the answer in.")
    if len(chunks) > max_chunks:
        raise BadRequest(
            f"that is {len(chunks)} passages and this runs one forward pass "
            f"each, so the cap is {max_chunks}. Cutting it down here would "
            f"score the answer against part of your document and call the "
            f"result grounding — shorten the attachment, or raise the cap "
            f"knowing what it costs."
        )

    started = time.perf_counter()
    prompt, char_spans = build(chunks, question)
    token_spans = locate(tokenizer, prompt, char_spans)

    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    seq = int(ids.shape[-1])
    position = seq - 1

    # ONE position_ids tensor, built once and handed to every pass. Rebuilding
    # it per pass is how a masked position silently renumbers everything after
    # it and the score becomes partly about the renumbering — the same rule
    # `attribute.py` states at length, and for the same reason.
    position_ids = torch.arange(seq, device=ids.device).unsqueeze(0)
    ones = torch.ones((1, seq), dtype=torch.long, device=ids.device)
    passes = 0

    def forward(mask: torch.Tensor, attentions: bool = False):
        nonlocal passes
        passes += 1
        with torch.no_grad():
            return model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=position_ids,
                output_attentions=attentions,
            )

    base_out = forward(ones)
    base = distribution(base_out.logits[0, position])
    top = int(base.argmax())

    # The floor: the same pass again, changing nothing. Anything below this is
    # the model's own arithmetic noise, and on bf16 that is not small.
    floor = kl_nats(base, distribution(forward(ones).logits[0, position]))

    # Where the answer position looked. Meaned over layers and heads — a
    # single layer's attention is a claim about that layer, and picking one
    # would be picking the one that agreed.
    attn_out = forward(ones, attentions=True)
    # AN EMPTY TUPLE, not None. SDPA and FlashAttention never materialise the
    # score matrix, and `output_attentions=True` under them returns `()` --
    # which loops zero times, sums nothing, and hands back 0.0 for every
    # passage. Measured on a plain `from_pretrained(...)` load, which picks sdpa
    # by default; ModelRuntime loads eager and does not hit this, but a caller
    # holding its own model does.
    layers = tuple(attn_out.attentions or ())
    attention_available = bool(layers)
    attention_note = ""
    if not attention_available:
        attention_note = (
            f"this model is running the "
            f"{getattr(model.config, '_attn_implementation', 'fused')!r} "
            "attention implementation, which never builds the score matrix "
            "there would be a share of. Load it with "
            'attn_implementation="eager" to measure it.'
        )
    share = torch.zeros(seq, device=ids.device, dtype=torch.float32)
    for layer in layers:
        share += layer[0, :, position, :].float().mean(dim=0)
    if layers:
        share /= len(layers)

    scores: list[Score] = []
    for index, (start, end) in enumerate(token_spans):
        mask = ones.clone()
        mask[0, start:end] = 0
        moved = kl_nats(base, distribution(forward(mask).logits[0, position]))
        looked = float(share[start:end].sum()) if attention_available else None
        scores.append(
            Score(
                index=index,
                preview=_preview(chunks[index]),
                n_tokens=end - start,
                dependence=round(moved, 6),
                attention=None if looked is None else round(looked, 6),
                depended_on=moved > floor,
                # Decided below, against the other passages -- and only when
                # both halves it is made of were actually measured.
                looked_not_used=None,
            )
        )

    # "Looked at" is relative to the other passages in THIS document, not to a
    # constant: a five-chunk prompt and a twenty-chunk prompt spread the same
    # attention mass over different numbers of places, so a fixed threshold
    # would call every chunk of a long document unattended.
    #
    # BOTH HALVES HAVE TO BE REAL. Without attention there is no "looked at" —
    # and 0.0 >= 0.0/2 is True, so the naive rule would flag EVERY passage on
    # a model that reported no attention at all. With a floor of exactly 0.0
    # there is no "not depended on" either: every passage that moved the
    # answer clears the gate, the flag can never fire, and leaving it False
    # reports a clean bill of health from a test that never ran. MEASURED —
    # a model on cuda/bf16 can reproduce its own answer bit for bit, so a zero
    # floor is the ordinary case here and not an edge one.
    decidable = attention_available and floor > 0.0
    if not decidable:
        for s in scores:
            s.looked_not_used = None
    elif scores:
        loudest = max(s.attention or 0.0 for s in scores)
        for s in scores:
            s.looked_not_used = (
                not s.depended_on
                and loudest > 0
                and (s.attention or 0.0) >= loudest / 2
            )

    # The joint mask. Its whole job is to show that the parts do not sum: it
    # is printed in `means()` next to the individual scores so a reader can
    # see for themselves that adding them would be wrong.
    joint = ones.clone()
    for start, end in token_spans:
        joint[0, start:end] = 0
    joint_moved = kl_nats(base, distribution(forward(joint).logits[0, position]))

    scores.sort(key=lambda s: -s.dependence)
    covered = (
        float(sum(s.attention or 0.0 for s in scores)) if attention_available else None
    )
    return Grounding(
        question=question.strip(),
        answer=tokenizer.decode([top]),
        answer_p=round(float(base[top]), 6),
        position=position,
        chunks=scores,
        n_chunks=len(scores),
        n_prompt_tokens=seq,
        noise_floor=round(floor, 6),
        joint=round(joint_moved, 6),
        attention_share=None if covered is None else round(covered, 6),
        attention_available=attention_available,
        attention_note=attention_note,
        passes=passes,
        seconds=round(time.perf_counter() - started, 2),
        ungrounded=not any(s.depended_on for s in scores),
        floor_degenerate=floor <= 0.0,
    )


def _preview(text: str, limit: int = 120) -> str:
    """Enough of a passage to recognise it, on one line."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
