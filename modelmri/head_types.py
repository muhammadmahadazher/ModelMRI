"""What kind of head is this — measured against a null, or not claimed at all.

A head list is 144 anonymous numbers. Some of those heads are doing something
with a name: attending to the token after the last time this token appeared
(induction), to the token before (previous-token), to the earlier copy itself
(duplicate-token), or to position 0 regardless of anything (a sink).

Detectors for these exist. What they usually do not have is a null. A detector
that hands back 0.31 tells you nothing, because 0.31 reads identically whether
it is remarkable for this model or completely ordinary — and that is how a
head list ends up confidently mislabelled.

SO EVERY LABEL IS GATED ON A NULL, AND THE NULL IS MEASURED

Nothing here is labelled by clearing a threshold somebody chose. Each head gets
its own null, measured on this model, and a label is only attached when the
head clears its own null by a stated number of standard deviations. Everything
else reads "no type detected", which is a real and common answer.

THREE GATES, AND A LABEL NEEDS ALL OF THEM

  significance   3σ above this head's own null, so the score is not the null
  effect size    above chance under the causal mask, so it beats an
                 indifferent head rather than merely beating a null that
                 never moves
  the peak       that offset is the single target this head attends to most,
                 so the label names what the head DOES rather than something
                 it happens to do a little of

Each was added because the ones before it were not enough, and each failure
was measured rather than argued. The σ gate alone labelled 138 of gpt2's 144
heads: a null of 0.0008 with almost no spread means any score clears it, and a
head putting 0.15× chance on the induction offset was labelled an induction
head at 201σ.

TWO NULLS, BECAUSE THE PATTERNS FAIL DIFFERENTLY

The obvious null is the same measurement on NON-repeating sequences, and for
induction and duplicate-token that is exactly right: those offsets are only
special because the sequence repeats, so a head that scores high on repeated
text and ordinary on random text is doing the thing.

It is the wrong null for the other two. A previous-token head attends to i-1
whether or not anything repeats, and a sink attends to position 0 always — so
their non-repeating "null" is just the same number again, the margin is
nothing, and a real previous-token head would never be labelled. Measured on
gpt2 rather than assumed. For those, the null is CHANCE under causal masking:
at position i there are i+1 positions to look at, so an unremarkable head puts
1/(i+1) of its mass on any one of them.

Which null a label cleared travels with the label. They are not interchangeable
and a reader comparing two labels needs to know they were not gated the same
way.

WHAT THESE LABELS ARE NOT

They are behaviour on random repeated tokens, not a claim about real text. A
head that behaves like an induction head on a repeated random sequence may do
something else entirely on English.

**A label must never be carried into the ablation ranking as if it explained
the KL.** The ranking measures what breaks when a head is removed; this
measures a positional habit. A head can be labelled and irrelevant, or
unlabelled and load-bearing, and joining the two would invent a causal story
neither measured.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .errors import Refusal

# How many standard deviations above its own null a head has to score before a
# label is attached. Three because the null is measured from a modest number of
# sequences and this is a labelling decision a reader will act on -- the cost
# of a wrong label is somebody chasing the wrong head.
MARGIN_SIGMA = 3.0

# AND it has to attend to the offset more than an indifferent head would.
#
# The sigma gate alone labelled 138 of gpt2's 144 heads. The null for a
# repetition-dependent pattern is ~0.0008 with a spread near zero, so ANY score
# clears three of them: L7H9 put 0.0043 on the induction offset -- 0.15x what
# chance would give it -- and was labelled an induction head at 201 sigma. That
# is significance without effect size, and it is the same failure as a detector
# with no null at all, arrived at from the other side.
#
# Chance is COMPUTED, not chosen: under the causal mask there are i+1 positions
# available at position i, so an indifferent head puts 1/(i+1) on any one of
# them. A head below that is not doing the thing the label names, however many
# standard deviations separate it from a null that never moves.
MIN_TIMES_CHANCE = 1.0

# AND the offset has to be the single target this head attends to most.
#
# There is no constant for this one, which is the point. A type label claims
# "this is what this head looks at", so the test is whether the pattern's
# offset IS the head's peak — parameter-free, and exclusive by construction
# since only one target can be the peak.
#
# The gate this replaced was "an outlier among this model's own heads", by
# median and MAD. It fails structurally in BOTH directions and both were
# measured on gpt2. When a behaviour is the NORM it cannot be an outlier: an
# outlier test reported ZERO sink heads in a model with heads putting 95% of
# their mass on position 0. And when it merely excluded one pattern it handed
# the label to a weaker one — L5H8 read "induction" at 0.089 while 70% of its
# attention sat on position 0.

# A vocabulary smaller than this cannot be sampled into sequences that mean
# anything: a byte-level tokenizer has ~256 base tokens, and "random tokens"
# drawn from it is random bytes, where a repeat is not the linguistic event
# these detectors are named for. Refused rather than measured badly.
MIN_VOCAB = 1_000

PATTERNS = ("induction", "previous-token", "duplicate-token", "sink")

# Which null gates which pattern, and why. Repetition-dependent patterns are
# gated on non-repeating text; position-dependent ones on chance under the
# causal mask, because non-repeating text does not make them go away.
NULL_KIND = {
    "induction": "repeat",
    "duplicate-token": "repeat",
    "previous-token": "chance",
    "sink": "chance",
}


@dataclass
class HeadLabel:
    layer: int
    head: int
    label: str | None
    # Attention mass this head put on each pattern's offset, on repeated text.
    scores: dict = field(default_factory=dict)
    # The winning pattern's score as a multiple of chance under the causal
    # mask. The effect size, beside the significance -- a head can be many
    # standard deviations from a null that never moves and still be below what
    # an indifferent head would do.
    times_chance: float | None = None
    # The most attention this head puts on any single target, on average. A
    # label is only attached when the pattern's offset IS this peak.
    peak: float = 0.0
    # What that pattern scores when it should mean nothing, per head.
    nulls: dict = field(default_factory=dict)
    # Standard deviations above the null, for the pattern that won. None when
    # nothing cleared -- NOT 0.0, which would read as "exactly at the null".
    margin: float | None = None
    null_kind: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TypeReport:
    n_layers: int
    n_heads: int
    seq_len: int
    n_sequences: int
    margin_sigma: float
    seed: int
    labels: list[HeadLabel] = field(default_factory=list)

    @property
    def named(self) -> list[HeadLabel]:
        return [row for row in self.labels if row.label]

    def counts(self) -> dict:
        out = {p: 0 for p in PATTERNS}
        for row in self.named:
            out[row.label] = out.get(row.label, 0) + 1
        out["no type detected"] = len(self.labels) - len(self.named)
        return out

    def to_dict(self) -> dict:
        return {
            **{k: v for k, v in asdict(self).items() if k != "labels"},
            "labels": [row.to_dict() for row in self.labels],
            "counts": self.counts(),
            "means": self.means(),
        }

    def means(self) -> str:
        counts = self.counts()
        # A label held by most of the heads is not distinguishing them. gpt2
        # attends to the first token throughout, so 90 of its 144 heads have
        # position 0 as their peak -- true, and useless read as "these 90 are
        # special".
        dominant = next(
            (
                (p, n)
                for p, n in counts.items()
                if p in PATTERNS and n > len(self.labels) / 2
            ),
            None,
        )
        named = (
            ", ".join(f"{n} {p}" for p, n in counts.items() if p in PATTERNS and n)
            or "none"
        )
        return (
            f"{len(self.named)} of {len(self.labels)} heads cleared all three "
            f"gates — {self.margin_sigma}σ above their own null, more "
            f"attention on the offset than chance under the causal mask would "
            f"give it, AND that offset being the single target the head "
            f"attends to most — and are labelled ({named}); the "
            f"rest read 'no type detected', which is a result rather than a "
            f"gap. Measured on {self.n_sequences} random token sequences of "
            f"length {self.seq_len}, repeated once — this is BEHAVIOUR ON "
            f"REPEATED RANDOM TOKENS, not a claim about real text, and it must "
            f"not be read as explaining the ablation ranking: a head can be "
            f"labelled and irrelevant, or unlabelled and load-bearing."
            + (
                f" NOTE: {dominant[0]} is the label on {dominant[1]} of "
                f"{len(self.labels)} heads. When most of a model's heads share "
                f"a habit, that is a fact about the model rather than a "
                f"distinction between its heads — read the share beside each "
                f"one, not the label alone."
                if dominant
                else ""
            )
        )


def _sampleable(tokenizer) -> list[int]:
    """Token ids that can be drawn at random, or a refusal saying why not."""
    size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if size < MIN_VOCAB:
        raise Refusal(
            f"this tokenizer has {size:,} tokens, which is a byte-level or "
            f"character-level vocabulary. 'A random token repeated' is random "
            f"bytes there, and a repeat of one is not the event these "
            f"detectors are named for — so they are refused here rather than "
            f"run on something they do not describe."
        )
    special = set(getattr(tokenizer, "all_special_ids", None) or [])
    usable = [i for i in range(size) if i not in special]
    if len(usable) < MIN_VOCAB:
        raise Refusal(
            "almost every token in this vocabulary is a special token, so a "
            "random sequence cannot be built without them."
        )
    return usable


def _sequences(usable: list[int], *, seq_len: int, count: int, repeat: bool, seed: int):
    """[count, 2*seq_len] ids — the second half a copy, or fresh.

    Both halves are drawn WITHOUT replacement within a sequence. A random draw
    that happened to repeat a token inside the "non-repeating" half would put
    an induction target into the null and quietly raise it, which is the one
    thing the null must not contain.
    """
    import torch

    generator = torch.Generator().manual_seed(seed)
    rows = []
    pool = torch.tensor(usable)
    for _ in range(count):
        first = pool[torch.randperm(len(pool), generator=generator)[:seq_len]]
        second = (
            first.clone()
            if repeat
            else pool[torch.randperm(len(pool), generator=generator)[:seq_len]]
        )
        rows.append(torch.cat([first, second]))
    return torch.stack(rows)


def _offsets(index: int, seq_len: int) -> dict:
    """Which position each pattern predicts this one attends to."""
    return {
        # The token AFTER the earlier copy of the current token: what an
        # induction head reaches for.
        "induction": index - seq_len + 1,
        "previous-token": index - 1,
        # The earlier copy itself.
        "duplicate-token": index - seq_len,
        "sink": 0,
    }


def _measure(model, ids, seq_len: int, n_layers: int, n_heads: int):
    """Mean attention mass at each pattern's offset, per head.

    Scored over the SECOND half only. In the first half there is no earlier
    copy to attend to, so an induction score there would be measuring nothing
    and averaging it in would halve every real signal.
    """
    import torch

    device = next(model.parameters()).device
    totals = {p: torch.zeros(n_layers, n_heads, dtype=torch.float64) for p in PATTERNS}
    # Mean attention at every RELATIVE offset, so "is this the single thing
    # this head looks at most" can be asked without inventing a threshold for
    # what counts as a lot.
    by_offset = torch.zeros(n_layers, n_heads, ids.shape[1], dtype=torch.float64)
    counted = 0
    with torch.no_grad():
        # One sequence at a time. The attention cube is layers x heads x S x S
        # and holding a batch of them is the largest allocation this module
        # would ever make -- on a 32-layer model at S=64 that is 134 MB per
        # eight sequences, for numbers that are summed and thrown away.
        for row in range(ids.shape[0]):
            out = model(ids[row : row + 1].to(device), output_attentions=True)
            attentions = out.attentions
            for index in range(seq_len, ids.shape[1]):
                targets = _offsets(index, seq_len)
                for pattern, target in targets.items():
                    if target < 0 or target > index:
                        continue
                    for layer in range(n_layers):
                        block = attentions[layer][0, :, index, target]
                        totals[pattern][layer] += block.detach().float().cpu().double()
                for layer in range(n_layers):
                    row = attentions[layer][0, :, index, : index + 1]
                    row = row.detach().float().cpu().double()
                    # Reversed, so column o is "o positions back from here" and
                    # the same column means the same thing at every position.
                    by_offset[layer, :, : index + 1] += row.flip(-1)
                counted += 1
            del out, attentions
    if not counted:
        raise Refusal(
            "there were no positions to score — the sequence length asked for "
            "leaves no second half to measure."
        )
    out = {p: (t / counted) for p, t in totals.items()}
    out["_by_offset"] = by_offset / counted
    return out


def label_heads(
    model,
    tokenizer,
    *,
    seq_len: int = 24,
    n_sequences: int = 6,
    seed: int = 0,
) -> TypeReport:
    """Label heads by behaviour, gated on a null measured for each one.

    Blocking; call from a worker thread. Cost is `2 * n_sequences` forward
    passes over `2 * seq_len` tokens.
    """
    import torch

    config = getattr(model, "config", None)
    n_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    n_heads = int(getattr(config, "num_attention_heads", 0) or 0)
    if not n_layers or not n_heads:
        raise Refusal(
            "this model does not report a layer and head count, so its heads "
            "cannot be enumerated to label."
        )

    usable = _sampleable(tokenizer)
    repeated = _sequences(
        usable, seq_len=seq_len, count=n_sequences, repeat=True, seed=seed
    )
    fresh = _sequences(
        usable, seq_len=seq_len, count=n_sequences, repeat=False, seed=seed + 1
    )

    scores = _measure(model, repeated, seq_len, n_layers, n_heads)
    repeat_null = _measure(model, fresh, seq_len, n_layers, n_heads)

    # Chance under the causal mask, for the patterns that non-repeating text
    # does not make go away. At position i there are i+1 positions available,
    # so an unremarkable head puts 1/(i+1) on any one of them. Averaged over
    # exactly the positions that were scored.
    positions = range(seq_len, 2 * seq_len)
    chance = sum(1.0 / (i + 1) for i in positions) / len(list(positions))

    # The null's own spread, measured across sequences rather than assumed.
    # Without it there is no σ to state a margin in.
    spreads = {}
    for pattern in PATTERNS:
        per_sequence = []
        for row in range(fresh.shape[0]):
            per_sequence.append(
                _measure(model, fresh[row : row + 1], seq_len, n_layers, n_heads)[
                    pattern
                ]
            )
        stacked = torch.stack(per_sequence)
        spreads[pattern] = stacked.std(dim=0, unbiased=len(per_sequence) > 1)

    # THE HEAD'S FAVOURITE TARGET, which is what a type label claims.
    #
    # The gate this replaces was "an outlier among this model's own heads",
    # and it fails structurally in both directions. When a behaviour is the
    # NORM it cannot be detected as an outlier: gpt2 attends heavily to the
    # first token throughout, so an outlier test reported ZERO sink heads in a
    # model with heads putting 95% of their mass there. And when it merely
    # excludes one pattern it hands the label to a weaker one: L5H8 read
    # "induction" at 0.089 while 70% of its attention sat on position 0.
    #
    # A type label says "this is what this head looks at". So the test is
    # whether the pattern's offset is the single target the head attends to
    # most on average — parameter-free, and exclusive by construction, since
    # only one target can be the peak.
    # The sink is an ABSOLUTE position, not a relative offset, so it does not
    # appear in the relative profile — position 0 is a different offset at
    # every index and its mass is smeared across all of them. Taking the peak
    # over relative offsets alone therefore never tested the sink pattern
    # against anything: a head with 0.95 on position 0 was compared against a
    # relative peak of 0.04 and passed trivially. Both axes, so every pattern
    # faces the same question.
    peak = torch.maximum(scores["_by_offset"].max(dim=-1).values, scores["sink"])
    labels: list[HeadLabel] = []
    for layer in range(n_layers):
        for head in range(n_heads):
            row_scores = {p: float(scores[p][layer, head]) for p in PATTERNS}
            row_nulls = {}
            best, best_margin, best_ratio = None, None, None
            for pattern in PATTERNS:
                kind = NULL_KIND[pattern]
                null = (
                    float(repeat_null[pattern][layer, head])
                    if kind == "repeat"
                    else chance
                )
                row_nulls[pattern] = round(null, 6)
                sigma = float(spreads[pattern][layer, head])
                if sigma <= 0:
                    # No spread means the null never moved across sequences.
                    # A margin in σ is undefined then, so the pattern is not
                    # labelled rather than being labelled with an infinite one.
                    continue
                margin = (row_scores[pattern] - null) / sigma
                ratio = row_scores[pattern] / chance if chance else 0.0
                # THREE gates, and the winner is chosen BY SCORE rather than by
                # margin. Choosing by margin labelled L5H8 "induction" at 0.089
                # while it put 0.703 on the sink -- the head overwhelmingly
                # attends to position 0, and induction won only because its
                # null had less spread. A label names what a head mostly does.
                # Three gates. Significance says the score is not this head's
                # own null; effect size says it beats chance under the causal
                # mask; the peak test says it is the thing this head actually
                # looks at rather than merely a thing it looks at.
                is_peak = row_scores[pattern] >= float(peak[layer, head]) - 1e-9
                if (
                    margin >= MARGIN_SIGMA
                    and ratio >= MIN_TIMES_CHANCE
                    and is_peak
                    and (best is None or row_scores[pattern] > row_scores[best])
                ):
                    best, best_margin, best_ratio = pattern, margin, ratio
            labels.append(
                HeadLabel(
                    layer=layer,
                    head=head,
                    label=best,
                    scores={k: round(v, 5) for k, v in row_scores.items()},
                    peak=round(float(peak[layer, head]), 5),
                    nulls=row_nulls,
                    margin=round(best_margin, 2) if best_margin is not None else None,
                    times_chance=round(best_ratio, 2)
                    if best_ratio is not None
                    else None,
                    null_kind=NULL_KIND[best] if best else "",
                )
            )

    return TypeReport(
        n_layers=n_layers,
        n_heads=n_heads,
        seq_len=seq_len,
        n_sequences=n_sequences,
        margin_sigma=MARGIN_SIGMA,
        seed=seed,
        labels=labels,
    )
