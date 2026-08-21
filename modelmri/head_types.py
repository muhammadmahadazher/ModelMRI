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
was measured rather than argued. The σ gate alone labelled all but a handful
of a model's heads: a null near zero with almost no spread means any score
clears it, and a head putting a fraction of chance on the induction offset was
labelled an induction head at hundreds of σ.

TWO NULLS, BECAUSE THE PATTERNS FAIL DIFFERENTLY

The obvious null is the same measurement on NON-repeating sequences, and for
induction and duplicate-token that is exactly right: those offsets are only
special because the sequence repeats, so a head that scores high on repeated
text and ordinary on random text is doing the thing.

It is the wrong null for the other two. A previous-token head attends to i-1
whether or not anything repeats, and a sink attends to position 0 always — so
their non-repeating "null" is just the same number again, the margin is
nothing, and a real previous-token head would never be labelled. Measured
rather than assumed. For those, the null is CHANCE under causal masking:
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

from . import fmt
from .errors import BadRequest, Refusal

# How many standard deviations above its own null a head has to score before a
# label is attached. Three because the null is measured from a modest number of
# sequences and this is a labelling decision a reader will act on -- the cost
# of a wrong label is somebody chasing the wrong head.
MARGIN_SIGMA = 3.0

# AND it has to attend to the offset more than an indifferent head would.
#
# The sigma gate alone labelled all but a handful of a model's heads. The null
# for a repetition-dependent pattern can sit near zero with a spread near zero,
# so ANY score clears three of them: a head putting a fraction of what chance
# would give it on the induction offset was labelled an induction head. That
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
# measured rather than argued. When a behaviour is the NORM it cannot be an
# outlier: an outlier test reported ZERO sink heads in a model whose heads put
# almost all their mass on position 0. And when it merely excluded one pattern
# it handed the label to a weaker one — a head read "induction" while most of
# its attention sat on position 0.

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
        # A label held by most of the heads is not distinguishing them. A model
        # that attends to the first token throughout can have most of its heads
        # peak at position 0 -- true, and useless read as "these ones are
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
    """Per-sequence attention profiles, gathered rather than looped.

    Returns `[n_sequences, n_layers, n_heads, S]` of mean attention at each
    RELATIVE offset, plus the same shape minus the offset axis for the sink,
    which is an absolute position and has no fixed offset.

    PER SEQUENCE, not averaged here, because the caller needs both the mean
    AND the spread across sequences and computing them from one pass is the
    difference between two forward passes and eight. The first version
    re-measured every null sequence individually to get its standard
    deviation, on top of the pass that had already measured them together:
    68 seconds for a button, and the browser gave up at 30.

    Scored over the SECOND half only. In the first half there is no earlier
    copy to attend to, so an induction score there measures nothing and
    averaging it in would halve every real signal.
    """
    import torch

    device = next(model.parameters()).device
    width = ids.shape[1]
    if width <= seq_len:
        raise Refusal(
            "there were no positions to score — the sequence length asked for "
            "leaves no second half to measure."
        )

    index = torch.arange(seq_len, width)
    # target[k, o] is "o positions back from position index[k]". Negative
    # where that runs off the front of the sequence.
    target = index.unsqueeze(1) - torch.arange(width).unsqueeze(0)
    valid = target >= 0
    safe = target.clamp(min=0)
    # How many of the scored positions can see each offset at all. Dividing by
    # the total would report a smaller mean for far offsets simply because
    # fewer positions could reach them.
    reach = valid.sum(dim=0).clamp(min=1)

    profiles = torch.zeros(ids.shape[0], n_layers, n_heads, width, dtype=torch.float64)
    sinks = torch.zeros(ids.shape[0], n_layers, n_heads, dtype=torch.float64)

    with torch.no_grad():
        # One sequence at a time. The attention cube is layers x heads x S x S
        # and holding a batch of them is the largest allocation this module
        # would ever make -- on a 32-layer model at S=64 that is 134 MB per
        # eight sequences, for numbers that are summed and thrown away.
        for row in range(ids.shape[0]):
            out = model(ids[row : row + 1].to(device), output_attentions=True)
            for layer in range(n_layers):
                # [H, P, S] -- every scored position's whole attention row.
                rows = out.attentions[layer][0, :, index, :].detach().float().cpu()
                sinks[row, layer] = rows[:, :, 0].mean(dim=1).double()
                # One gather instead of a Python loop over positions and
                # heads. The old inner loop did a GPU->CPU transfer per
                # (position, layer) -- hundreds of them per sequence.
                picked = torch.gather(
                    rows, 2, safe.unsqueeze(0).expand(n_heads, -1, -1)
                )
                picked = picked * valid.unsqueeze(0)
                profiles[row, layer] = (picked.sum(dim=1) / reach).double()
            del out
    return profiles, sinks


def _patterns_from(profiles, sinks, seq_len: int) -> dict:
    """Pattern scores read straight off the offset profile.

    Three of the four patterns ARE fixed offsets, so they need no separate
    measurement: induction is the token after the earlier copy, which is
    `seq_len - 1` back; previous-token is 1 back; duplicate-token is `seq_len`
    back. Only the sink is an absolute position and it arrives separately.
    """
    return {
        "induction": profiles[..., seq_len - 1],
        "previous-token": profiles[..., 1],
        "duplicate-token": profiles[..., seq_len],
        "sink": sinks,
    }


#: Shortest probe sequence that can carry a repeat. The labels are measured
#: on the second copy of a repeated sequence, so anything shorter has no
#: second half for a head to attend back into and the measurement is not
#: merely noisy — it does not exist.
MIN_SEQ_LEN = 4

#: Ceilings, because the floors were only half the guard. `seq_len=-4` was
#: caught and `seq_len=100000` was not: the probe runs with
#: `output_attentions=True`, so the tensors handed back are
#: `n_sequences x layers x heads x (2*seq_len)^2 x 4` bytes and that term is
#: QUADRATIC in a number taken straight off a URL. `?seq_len=100000` asks the
#: allocator for hundreds of gigabytes before a single forward pass, and
#: `?n_sequences=1000000` does the same through the profile list.
#:
#: Two gates rather than one. These constants are the cheap structural bound,
#: refused before anything is built. `ATTENTION_BUDGET_BYTES` below is the
#: honest one — it is checked once the model's own layer and head counts are
#: known, so the refusal quotes the arithmetic for THIS model rather than a
#: number chosen for some other one. A 32x16 model and a 4x4 model do not have
#: the same safe sequence length, and a single constant would be wrong for
#: both.
MAX_SEQ_LEN = 512
MAX_SEQUENCES = 64

#: What the attention tensors may occupy before this refuses. Deliberately
#: generous — the point is to stop an allocation nobody could have wanted, not
#: to second-guess a reasonable probe.
ATTENTION_BUDGET_BYTES = 2_000_000_000


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

    # The arguments arrive from a query string, and both of them index into
    # tensors far below here. Unchecked, they came back as errors about
    # torch's internals: `n_sequences=0` produced "stack expects a non-empty
    # TensorList", and `seq_len=-4` asked the allocator for 735,830,067,168
    # bytes — 735 GB, from a number in a URL.
    #
    # The zero case for `seq_len` was ALREADY answered properly, one branch
    # further down, with "there were no positions to score". So this is one
    # question that had two answers depending on whether the bad number
    # happened to be zero or negative.
    if n_sequences < 1:
        raise BadRequest(
            f"labelling heads needs at least one probe sequence to measure "
            f"them on, and this asked for {n_sequences}. Each one is two "
            f"forward passes — the cost is stated before you spend it."
        )
    if seq_len < MIN_SEQ_LEN:
        raise BadRequest(
            f"a probe sequence has to be at least {MIN_SEQ_LEN} tokens for "
            f"the repeat half to exist at all, and this asked for {seq_len}. "
            f"The labels are read from what a head does on the SECOND copy of "
            f"a repeated sequence; there is no second copy below that length."
        )
    if seq_len > MAX_SEQ_LEN:
        raise BadRequest(
            f"a probe sequence of {seq_len:,} tokens is past the "
            f"{MAX_SEQ_LEN:,} this will measure. The attention tensors are "
            f"QUADRATIC in this number — every head hands back a "
            f"{2 * seq_len:,}x{2 * seq_len:,} map — so the cost stops being "
            f"about the model and starts being about the length. The labels "
            f"read a positional habit, and that habit is visible at "
            f"{MIN_SEQ_LEN}-64 tokens."
        )
    if n_sequences > MAX_SEQUENCES:
        raise BadRequest(
            f"{n_sequences:,} probe sequences is past the {MAX_SEQUENCES} "
            f"this will run. Each one is two forward passes with attention "
            f"kept, and the labels are gated on a measured null that settles "
            f"well before this — more sequences past that point buy precision "
            f"nobody reads."
        )

    config = getattr(model, "config", None)
    n_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    n_heads = int(getattr(config, "num_attention_heads", 0) or 0)
    if n_layers and n_heads:
        # THE HONEST BOUND, quoted for this model. `output_attentions=True`
        # returns one [1, heads, S, S] float32 tensor per layer per sequence.
        span = 2 * seq_len
        want = n_sequences * n_layers * n_heads * span * span * 4
        if want > ATTENTION_BUDGET_BYTES:
            raise BadRequest(
                f"measuring {n_sequences} sequence(s) of {seq_len} tokens on "
                f"this model means keeping attention for {n_layers} layers x "
                f"{n_heads} heads at {span}x{span} — "
                f"{fmt.bytes_si(want)}, past the "
                f"{fmt.bytes_si(ATTENTION_BUDGET_BYTES)} this will hold at "
                f"once. The maps are quadratic in the sequence length, so "
                f"halving it quarters this."
            )
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

    # TWO forward-pass sweeps, not two plus one per null sequence. The mean
    # and the spread both come out of the same per-sequence profiles.
    repeat_profiles, repeat_sinks = _measure(
        model, repeated, seq_len, n_layers, n_heads
    )
    null_profiles, null_sinks = _measure(model, fresh, seq_len, n_layers, n_heads)

    per_sequence = _patterns_from(repeat_profiles, repeat_sinks, seq_len)
    null_per_sequence = _patterns_from(null_profiles, null_sinks, seq_len)

    scores = {p: per_sequence[p].mean(dim=0) for p in PATTERNS}
    repeat_null = {p: null_per_sequence[p].mean(dim=0) for p in PATTERNS}
    spreads = {
        p: null_per_sequence[p].std(dim=0, unbiased=fresh.shape[0] > 1)
        for p in PATTERNS
    }

    # Chance under the causal mask, for the patterns that non-repeating text
    # does not make go away. At position i there are i+1 positions available,
    # so an unremarkable head puts 1/(i+1) on any one of them. Averaged over
    # exactly the positions that were scored.
    positions = range(seq_len, 2 * seq_len)
    chance = sum(1.0 / (i + 1) for i in positions) / len(list(positions))

    # THE HEAD'S FAVOURITE TARGET, which is what a type label claims.
    #
    # The gate this replaces was "an outlier among this model's own heads",
    # and it fails structurally in both directions. When a behaviour is the
    # NORM it cannot be detected as an outlier: a model that attends heavily to
    # the first token throughout got ZERO sink heads out of an outlier test,
    # while its heads put almost all their mass there. And when it merely
    # excludes one pattern it hands the label to a weaker one: a head read
    # "induction" while most of its attention sat on position 0.
    #
    # The sink is an ABSOLUTE position and so does not appear in the relative
    # profile -- position 0 is a different offset at every index. Both axes,
    # or the sink pattern never faces the test at all.
    peak = torch.maximum(repeat_profiles.mean(dim=0).max(dim=-1).values, scores["sink"])

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
