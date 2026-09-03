# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""What does head 14/3 do on YOUR text, and what does it push toward?

"Head 14/3 ranked first" is not a finding. `ablate.rank_heads` says a head
mattered on ONE prompt, and the number moves with the prompt; `head_types`
labels a positional habit on RANDOM REPEATED TOKENS and says so in its own
docstring. Neither is evidence about real text, and this project has had the
corpus version of exactly this for an SAE feature — `feature_corpus.py` —
since long before it had one for a head. That gap is what this closes.

Three readouts, independent on purpose, because a claim that survives all
three is worth something a claim resting on one is not:

  what it writes on    the positions where this head's contribution to the
                       residual stream is largest, in YOUR corpus, with the
                       token it was attending to at each — measured here
  what it pushes at    the tokens it promotes when it reads a given token,
                       read straight off its OV circuit — `ov_circuits`,
                       exact, no corpus
  what removing it does the causal ranking `sweep --metric heads` already
                       produces over a prompt set — NOT re-implemented here,
                       and the sentence says where to get it

## A head is not a feature, and the difference costs a design decision

An SAE feature has ONE scalar per token: it fired this much here. A head has
three candidate quantities and they answer different questions —

  * the size of what it wrote into the stream at this position,
  * how much attention mass it put on its peak target,
  * WHICH token it was reading when it wrote.

The first is the closest analogue to a feature's activation, and it is what
`ablate._cut` actually removes, so it is the quantity the causal ranking is
about. But reporting it alone would repeat the collapse this module exists to
undo: "position 41 wrote hard" without "and it was reading `Paris`" is half a
sentence. So a span here carries a SOURCE and a TARGET, which is why it is not
`feature_corpus.Span` — that dataclass has one `offset` because a feature fires
at one place, and a head's answer is a pair.

## The corpus is part of the result

Every number below is about the text you handed it. A head that never writes
hard in this corpus is NOT SEEN IN THIS CORPUS — not idle, not unimportant.
The label, the token count and the sha256 travel with the answer for the same
reason `feature_corpus.py` carries them, and every cap is named.

## No labels

This says where a head wrote and what it was reading. Calling it an induction
head, a name-mover, or anything else is the reader's job. `head_types.py` gates
every label it attaches on a null it measured on this model, and even then it
says the label is behaviour on random tokens rather than a claim about text.
Nothing here has a null, so nothing here has a label.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .errors import BadRequest, Refusal

#: Tokens one sweep will read before it stops. The cap is REPORTED in the
#: result rather than applied quietly, because a head that "never wrote hard"
#: in the first 200k tokens of a 2M-token log is a different claim from one
#: that never wrote hard in the log. The same number `feature_corpus` uses,
#: deliberately: two corpus sweeps in one tool that silently read different
#: amounts would make their outputs incomparable.
MAX_TOKENS = 200_000

#: Positions kept per head. The whole point is the extreme tail — a head's
#: hundredth-largest write is not what anybody opened this for — and every
#: position kept costs a decode and a source lookup.
TOP_SPANS = 20

#: Tokens either side of the writing position in a reported span. Enough to
#: see what the head was doing, short enough that twenty fit on a screen.
SPAN_CONTEXT = 6


@dataclass
class HeadSpan:
    """One place this head wrote hard, and what it was reading when it did.

    A PAIR, unlike `feature_corpus.Span`. A feature fires at one position and
    one `offset` locates it; a head writes at one position while attending to
    another, and reporting only the first is the half-sentence this module was
    built to stop.
    """

    #: Where the head WROTE — the position whose residual-stream contribution
    #: is large.
    position: int
    token: str
    #: Character offset of `token` inside `text`. Not derivable by searching:
    #: a sentence can contain the same token twice and only one of those
    #: positions wrote. `feature_corpus.Span` carries this for the same reason.
    offset: int
    #: What it was READING — the position it put most attention on, and the
    #: token there. `None` when attention was not captured for this sweep,
    #: which is a real state: the write norms cost one forward pass and the
    #: attention costs another, so a caller may ask for the cheap half alone.
    source_position: int | None
    source_token: str | None
    #: Share of this head's attention mass on `source_position`. `None` when
    #: attention was not captured — never 0.0, which would say the head looked
    #: nowhere.
    source_share: float | None
    #: L2 norm of this head's slice of the attention output projection's input
    #: at `position`. This is exactly the vector `ablate._cut` zeroes, so it is
    #: the quantity the causal ranking is about rather than a proxy for it.
    write_norm: float
    sequence: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HeadCorpus:
    """One head, over one corpus, with everything needed to disbelieve it."""

    layer: int
    head: int
    kv_head: int
    corpus_label: str
    corpus_sha256: str
    n_sequences: int
    n_tokens: int
    #: True when the sweep stopped at `MAX_TOKENS` before the corpus ended.
    truncated: bool
    #: Whether attention was captured. False means every `source_*` on every
    #: span is `None`, and the sentence says so rather than leaving a column
    #: of nulls to be read as "attended nowhere".
    attention_read: bool

    #: The extreme tail, largest first.
    spans: list[HeadSpan]
    #: Distribution of the write norm over EVERY position read, so a reader can
    #: see whether the top spans are a tail or the whole thing. A head whose
    #: largest write is 1.1x its median did not "fire on" anything.
    write_norm_mean: float
    write_norm_median: float
    write_norm_max: float
    #: How many positions were read. `len(spans)` is how many were KEPT.
    n_positions: int
    #: How many of them carried a NON-ZERO write. The difference between this
    #: and `n_positions` is what makes a head sparse rather than absent, and
    #: the two states had one sentence between them until a test on a
    #: one-write-in-ten corpus printed "NOT SEEN IN THIS CORPUS" for a head
    #: whose largest write was 8.49.
    n_wrote: int

    passes: int
    device: str

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "spans": [s.to_dict() for s in self.spans],
            "means": self.means(),
        }

    def means(self) -> str:
        parts = [
            f"Head {self.layer}/{self.head} over {self.corpus_label}: "
            f"{self.n_tokens:,} tokens in {self.n_sequences} sequence(s). The "
            f"number ranked is the L2 norm of what this head wrote into the "
            f"residual stream at each position — the exact vector head "
            f"ablation removes, so it is the quantity the causal ranking is "
            f"about rather than a stand-in for it."
        ]
        # The shape of the distribution, because "the top 20" means nothing
        # without it. A head whose maximum is barely above its median has no
        # tail, and twenty spans off the top of a flat distribution read as
        # twenty findings.
        # BRANCHED ON THE MAX, not the median, and the difference is a head
        # that writes once in a thousand positions. Its median is zero and its
        # largest write can be enormous, and the first version of this printed
        # "NOT SEEN IN THIS CORPUS" for exactly that — a sparse head reported
        # as an absent one, which is the collapse this module is about.
        if self.write_norm_max <= 0:
            parts.append(
                f"This head wrote NOTHING at any of {self.n_positions:,} "
                f"positions. NOT SEEN IN THIS CORPUS is what that means — this "
                f"text never gave it anything to write about — and it is a "
                f"different claim from the head doing nothing in general."
            )
        elif self.write_norm_median > 0:
            ratio = self.write_norm_max / self.write_norm_median
            parts.append(
                f"Across {self.n_positions:,} positions the median write was "
                f"{self.write_norm_median:.4g} and the largest "
                f"{self.write_norm_max:.4g} ({ratio:.1f}x the median). A head "
                f"whose largest write is close to its median did not 'fire on' "
                f"anything here — the spans below are then the top of a flat "
                f"distribution rather than a tail."
            )
        else:
            # Sparse: it wrote somewhere, and at more than half the positions
            # it wrote nothing at all. No ratio, because the median is zero and
            # a ratio against it is undefined rather than large.
            parts.append(
                f"SPARSE HERE: this head wrote at {self.n_wrote:,} of "
                f"{self.n_positions:,} positions and nothing at the other "
                f"{self.n_positions - self.n_wrote:,}, so its median write is "
                f"zero and there is no median to compare the largest "
                f"({self.write_norm_max:.4g}) against. That is a real shape and "
                f"not a failed measurement — a head that acts on a handful of "
                f"tokens looks exactly like this."
            )
        if not self.spans:
            parts.append(
                "No position was kept, so there is nothing below. That is a "
                "fact about this text and not about the head."
            )
        if not self.attention_read:
            parts.append(
                "ATTENTION WAS NOT READ on this sweep, so every span says "
                "where the head wrote and not what it was reading. That is the "
                "cheap half: ask for the attention pass to fill it in."
            )
        if self.truncated:
            parts.append(
                f"The corpus was cut at {MAX_TOKENS:,} tokens; what came after "
                f"it was not read, so a larger write further in is not ruled "
                f"out."
            )
        parts.append(
            "This says WHERE it wrote and WHAT it was reading. It attaches no "
            "label: naming what a head does is the reader's job, and nothing "
            "here measured a null to gate one on."
        )
        return " ".join(parts)


def _whole(name: str, value, *, low: int) -> int:
    """An index or a count, with the bool guard first."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise BadRequest(f"`{name}` is a whole number and this was {value!r}.")
    if value < low:
        raise BadRequest(f"`{name}` is at least {low} and this was {value}.")
    return value


def sweep_cost(n_sequences: int, *, read_attention: bool) -> dict:
    """What this will cost, in forward passes, before any is spent.

    Two passes per sequence when attention is read and one when it is not, and
    the difference is not academic: `output_attentions=True` materialises
    `[1, n_heads, S, S]` per layer, which is where this measurement's memory
    goes. Priced the way `budget.py` prices everything else here — in passes,
    never in seconds, because a duration guessed from another machine is the
    kind of number people plan around.
    """
    n = _whole("n_sequences", n_sequences, low=0)
    per = 2 if read_attention else 1
    return {
        "n_sequences": n,
        "passes": n * per,
        "reads_attention": bool(read_attention),
        "means": (
            f"{n * per} forward pass(es): {n} sequence(s) x {per}. "
            + (
                "The second pass per sequence asks for attention weights, "
                "which the first does not — that is what fills in the token "
                "each write was reading, and it is also where the memory goes: "
                "attention is [heads x S x S] per layer where the write norms "
                "are [S] per head."
                if read_attention
                else "Attention is NOT read, so every span will say where the "
                "head wrote and not what it was reading."
            )
        ),
    }


def _head_writes(model, blocks, ids, *, layer: int, head: int, n_heads: int):
    """`[S]` L2 norms of one head's slice of the projection input.

    Captured through `ablate.capture_projection_inputs`, which is the hook
    `ablate._cut` slices — so this is the same tensor the causal ranking
    removes, at the same point, rather than a second capture free to drift
    from it. `ablate.head_geometry` decides where the slice is, for the reason
    it states: `hidden_size // n_heads` is wrong by 2x on Qwen3-0.6B.
    """
    import torch

    from . import ablate

    head_dim = ablate.head_geometry(blocks(layer), n_heads)
    captured = ablate.capture_projection_inputs(model, blocks, ids, [layer])
    at = captured.get(layer)
    if at is None:
        raise Refusal(
            f"nothing was captured at layer {layer}'s attention output "
            f"projection, so what this head wrote cannot be measured. The "
            f"model ran without reaching that hook — check that layer {layer} "
            f"is inside this model's depth."
        )
    lo, hi = head * head_dim, (head + 1) * head_dim
    with torch.no_grad():
        return at[:, lo:hi].float().norm(dim=-1).cpu()


def _attended(model, ids, *, layer: int, head: int, positions):
    """`{position: (source_position, share)}` for the positions asked about.

    One pass with `output_attentions=True`, and only the rows wanted are kept:
    the full cube is `[1, n_heads, S, S]` per layer and holding it for a
    200k-token corpus is not a thing that fits anywhere. `vla.py` reduces
    attention inside the forward pass for the same reason and says so.
    """
    import torch

    with torch.no_grad():
        out = model(ids, output_attentions=True)
    maps = getattr(out, "attentions", None)
    if not maps or layer >= len(maps):
        raise Refusal(
            "this model did not return attention weights, so what each write "
            "was reading cannot be filled in. Some architectures never expose "
            "them; the write norms above are still real, and asking for the "
            "cheap half alone is a supported answer."
        )
    rows = maps[layer][0, head].float()
    found = {}
    for p in positions:
        if p >= rows.shape[0]:
            continue
        row = rows[p]
        best = int(torch.argmax(row))
        found[p] = (best, float(row[best]))
    del maps, out
    return found


def sweep(
    model,
    tokenizer,
    blocks,
    texts: list[str],
    *,
    layer: int,
    head: int,
    n_heads: int,
    corpus_label: str,
    read_attention: bool = True,
    top_spans: int = TOP_SPANS,
    max_tokens: int = MAX_TOKENS,
) -> HeadCorpus:
    """One head, over one corpus. Memory-bounded and honest about the tail.

    ONE SEQUENCE AT A TIME and nothing accumulated but the running top-k and
    four scalars. The alternative — every position's write norm in a list — is
    `[n_tokens]` floats, which is fine, but the tempting version that keeps the
    ACTIVATIONS to re-rank later is `[n_tokens, head_dim]` and that is 100 MB
    per head at 200k tokens before anything useful happens.
    """
    import heapq
    import math

    from . import ov_circuits
    from .tuned_lens import corpus_hash

    layer = _whole("layer", layer, low=0)
    head = _whole("head", head, low=0)
    top_spans = _whole("top_spans", top_spans, low=1)
    max_tokens = _whole("max_tokens", max_tokens, low=1)
    if (
        not isinstance(texts, list)
        or not texts
        or not all(isinstance(t, str) for t in texts)
    ):
        raise BadRequest(
            "a corpus sweep needs a list of strings to read. Pass `texts`, or "
            "`file` naming a local .txt or .jsonl."
        )
    if not str(corpus_label).strip():
        raise BadRequest(
            "name the corpus. Every number here is about the text you handed "
            "it, and a result that cannot say which text is not one somebody "
            "can check or compare."
        )

    # The geometry, and its refusals, BEFORE any pass is spent: a head index
    # outside the layer is the caller's mistake and should cost nothing.
    d_model = int(model.config.hidden_size)
    geo = ov_circuits.geometry(blocks(layer), n_heads=n_heads, d_model=d_model)
    if not 0 <= head < geo.n_heads:
        raise BadRequest(
            f"head {head} is outside this layer's {geo.n_heads} heads "
            f"(0 to {geo.n_heads - 1})."
        )

    where = next(model.parameters()).device
    keep: list[tuple[float, int, int]] = []  # (norm, sequence, position)
    per_sequence: dict[int, tuple] = {}
    total = 0.0
    total_sq = 0.0
    n_positions = 0
    n_wrote = 0
    biggest = 0.0
    # Exact median needs every value; a running one does not exist. The values
    # are one float per token, so 200k of them is 1.6 MB — affordable, and the
    # only structure here that grows with the corpus. Said out loud because
    # "memory bounded" has to mean something specific.
    all_norms: list[float] = []
    n_sequences = 0
    truncated = False
    passes = 0

    for index, text in enumerate(texts):
        if total_sq >= 0 and n_positions >= max_tokens:
            truncated = True
            break
        ids = tokenizer(text, return_tensors="pt").input_ids.to(where)
        if ids.shape[-1] == 0:
            continue
        n_sequences += 1
        norms = _head_writes(
            model, blocks, ids, layer=layer, head=head, n_heads=n_heads
        )
        passes += 1
        per_sequence[index] = (ids[0].cpu(), text)
        for position in range(int(norms.shape[0])):
            value = float(norms[position])
            if not math.isfinite(value):
                # A non-finite norm is not a large write — it is a broken one,
                # and ranking it first would put the worst-measured position at
                # the top of a list people read as the most important.
                continue
            n_positions += 1
            if value > 0:
                n_wrote += 1
            total += value
            biggest = max(biggest, value)
            all_norms.append(value)
            # A ZERO WRITE IS NOT A SPAN. Caught by the test that asserted
            # `n_positions > len(spans)`: on a six-token corpus where the head
            # wrote at exactly one position, all six were kept and five of them
            # had `write_norm=0.0` — a head with one real write reporting six
            # findings, which is the reading this module exists to prevent.
            #
            # They still count toward `n_positions`, the mean and the median.
            # Those describe the DISTRIBUTION, and dropping the zeros there
            # would inflate every one of them: a head that writes once in
            # 200,000 tokens would report the mean of its single write.
            if value > 0:
                item = (value, index, position)
                if len(keep) < top_spans:
                    heapq.heappush(keep, item)
                elif value > keep[0][0]:
                    heapq.heapreplace(keep, item)
            if n_positions >= max_tokens:
                truncated = True
                break
        del norms

    if not n_positions:
        raise Refusal(
            "none of the text you passed produced a single token, so there is "
            "nothing to measure. Check the corpus file is text rather than a "
            "checkpoint, and that it is not empty."
        )

    top = sorted(keep, reverse=True)
    wanted: dict[int, list[int]] = {}
    for _value, sequence, position in top:
        wanted.setdefault(sequence, []).append(position)

    attended: dict[int, dict[int, tuple[int, float]]] = {}
    attention_read = False
    if read_attention:
        try:
            for sequence, positions in wanted.items():
                ids, _text = per_sequence[sequence]
                attended[sequence] = _attended(
                    model,
                    ids.unsqueeze(0).to(where),
                    layer=layer,
                    head=head,
                    positions=positions,
                )
                passes += 1
            attention_read = True
        except Refusal:
            # A model that will not hand back attention is a real and reported
            # state, not a failure of the sweep: the write norms above cost a
            # pass each and are already measured. Degrading here rather than
            # raising is the difference between half an answer and none.
            attended = {}
            attention_read = False

    spans = []
    for value, sequence, position in top:
        ids, text = per_sequence[sequence]
        token = tokenizer.decode([int(ids[position])])
        lo = max(0, position - SPAN_CONTEXT)
        hi = min(int(ids.shape[0]), position + SPAN_CONTEXT + 1)
        window = tokenizer.decode([int(t) for t in ids[lo:hi]])
        before = tokenizer.decode([int(t) for t in ids[lo:position]])
        source = (attended.get(sequence) or {}).get(position)
        spans.append(
            HeadSpan(
                position=position,
                token=token,
                offset=len(before),
                source_position=None if source is None else int(source[0]),
                source_token=(
                    None
                    if source is None
                    else tokenizer.decode([int(ids[int(source[0])])])
                ),
                source_share=None if source is None else round(float(source[1]), 6),
                write_norm=round(value, 6),
                sequence=sequence,
                text=window,
            )
        )

    ordered = sorted(all_norms)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )

    return HeadCorpus(
        layer=layer,
        head=head,
        kv_head=geo.kv_head(head),
        corpus_label=str(corpus_label),
        corpus_sha256=corpus_hash(texts),
        n_sequences=n_sequences,
        n_tokens=n_positions,
        truncated=truncated,
        attention_read=attention_read,
        spans=spans,
        write_norm_mean=round(total / n_positions, 6),
        write_norm_median=round(float(median), 6),
        write_norm_max=round(biggest, 6),
        n_positions=n_positions,
        n_wrote=n_wrote,
        passes=passes,
        device=str(where),
    )


def evidence(
    model,
    tokenizer,
    blocks,
    texts: list[str],
    *,
    layer: int,
    head: int,
    n_heads: int,
    corpus_label: str,
    read_attention: bool = True,
    top_k: int = 10,
) -> dict:
    """All three readouts for one head, with the third pointed at rather than run.

    `feature_corpus.evidence` runs its three because all three are cheap for a
    feature. The causal leg for a HEAD is not: `sweep --metric heads` is one
    ablation per head per prompt, and running it inside this call would turn a
    two-pass measurement into a several-hundred-pass one without the caller
    asking. So this returns the two it measured and NAMES the command for the
    third, rather than quietly omitting it — a two-legged answer presented as
    the whole thing is the failure `feature_corpus.py:3-5` argues against.
    """
    from . import ov_circuits

    corpus = sweep(
        model,
        tokenizer,
        blocks,
        texts,
        layer=layer,
        head=head,
        n_heads=n_heads,
        corpus_label=corpus_label,
        read_attention=read_attention,
    )

    # WHAT IT PUSHES AT, for the token it most often read. Exact weight
    # arithmetic and no corpus, so it is a different kind of evidence from
    # everything above — which is the entire reason it is worth having beside
    # it rather than instead of it.
    pushes = None
    sources = [s.source_token for s in corpus.spans if s.source_token is not None]
    if sources:
        common = max(set(sources), key=sources.count)
        ids = tokenizer.encode(common, add_special_tokens=False)
        if ids:
            pushes = ov_circuits.ov_vocabulary(
                model,
                tokenizer,
                blocks(layer),
                head,
                n_heads=n_heads,
                source_token_id=ids[0],
                top_k=top_k,
            )

    return {
        "corpus": corpus.to_dict(),
        "pushes_at": pushes,
        # NOT run, and named rather than omitted.
        "causal": {
            "available": False,
            "how": (
                f"modelmri sweep <your-prompts.jsonl> --metric heads --layer {layer}"
            ),
            "why": (
                "The causal leg — what breaks when this head is removed — is "
                "one ablation per head per prompt, hundreds of forward passes "
                "against the two this took. It is not run here without being "
                "asked for; the command above produces it over a prompt set "
                "and stores the result."
            ),
        },
        "means": (
            "Two independent readouts of the same head: where it wrote in your "
            "text and what it was reading, MEASURED here; and what it pushes "
            "the vocabulary toward, read off its weights with no corpus at "
            "all. They can disagree, and that is informative rather than a "
            "fault — a head wired to promote a token it never got to read in "
            "this text is exactly the case the two halves exist to separate. "
            "The third leg is named above and was not run."
        ),
    }
