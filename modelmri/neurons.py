"""One column of an MLP, for the models nobody ever published an SAE for.

`saes.py` and `feature_corpus.py` are the good version of this panel and they
are the bar. They are also unavailable to almost every model anybody loads:
`sae_registry.for_model` answers "none" for all but a handful of releases, and
when it does, the feature panel offers nothing at all. Not a degraded view — a
blank one. This module is what that panel shows instead.

## A NEURON BROWSER IS NOT A WORSE SAE. IT IS A BLUNTER INSTRUMENT.

This has to be said first, and it has to be said where a reader will see it
rather than in a footnote, because the panel LOOKS like the SAE panel: same
three readouts, same span cards, same firing-rate table. It is not measuring
the same kind of thing.

**Neurons are polysemantic.** A single MLP neuron routinely responds to several
unrelated things, and that is not a defect of this code — it is the entire
reason sparse autoencoders were invented. A model has more things to represent
than it has neurons, so it stores them in superposition: as directions that are
not axis-aligned, spread across many neurons, with each neuron carrying pieces
of several. An SAE's job is to rotate out of the neuron basis into one where
the directions separate. Reading neurons directly is refusing that rotation.

So a neuron's top-activating spans can be a coherent set and the neuron STILL
not be "a detector for" that thing: you are looking at the intersection of the
corpus with one axis of a basis the model did not choose to be interpretable
in. Every readout below is honest about what it measured — this many tokens,
this activation, this span — and none of them is evidence that the neuron has
one meaning. `POLYSEMANTIC` is the sentence, and it travels with every result
this module returns rather than living up here where a caller can forget it.

## THE THREE READOUTS, WHICH ARE THE SAME THREE

`feature_corpus.py` gives an SAE feature three independent readouts and argues
that a claim surviving all three is worth something a claim resting on one is
not. That argument does not depend on the basis, so the discipline carries over
unchanged:

  what it fires on      `sweep` — firing rate, peak and floor for every neuron
                        over YOUR corpus; `evidence` — the top-activating
                        spans and the activation histogram for one of them
  what it pushes at     `logit_weights` — the tokens the neuron's write
                        direction promotes and suppresses. EXACT and needing
                        no corpus, because a neuron's write direction is
                        literally one column of the down-projection
  what removing it does not here. `ablate.py` cuts attention heads and
                        `feature_ablate.py` cuts SAE features; neither cuts a
                        neuron, and this module does not claim a causal number
                        it did not take

## A FIRING RATE HERE DOES NOT MEAN WHAT IT MEANS THERE

`feature_corpus.evidence` flags an SAE feature active on 20% of tokens as "not
selecting anything", and it is right to. Applying that threshold to neurons
would flag most of the layer: a post-GELU neuron sits near a coin flip by
construction, because GELU passes roughly half of a zero-centred pre-activation
and the pre-activations are roughly zero-centred. A rule tuned for a sparse
basis reads a dense one as universally suspicious, which is the same as reading
it not at all.

So the reference is measured instead of chosen: every firing rate is reported
beside `layer_median_firing_rate`, the median rate over every neuron in this
same layer on these same tokens. "Neuron 3011 fires on 63% of tokens" is not a
finding. "63% against a layer median of 49%" is at least a comparison, and it
is a comparison to a number this run measured rather than one this file
decided.

## WHAT NMF DOES AND WHAT IT REFUSES TO SAY

Half of superposition's damage is that one neuron carries several things. The
other half is that one thing is carried by several neurons — and that half is
addressable without an SAE. `decompose` factorises the sampled
[tokens x neurons] activation matrix into `k` non-negative components, so a
group of neurons that rise and fall together becomes one row to look at instead
of forty.

**A component is "these neurons co-fire on these spans". That is the whole
claim.** It is not a concept, it is not a feature, and naming it is the
reader's job — `feature_corpus.py:32-35` states this rule for SAE features and
nothing about NMF weakens it. `Component.label` is `None` and is always `None`.
A generated label would be the one thing on the page that nothing measured.

Three numbers state how far to trust the components, because a factorisation
with no stated resolution is a claim to precision the method does not have:

  `residual`          relative Frobenius error of `W @ H` against the matrix.
                      How much of the data the k components do NOT account
                      for. The same discipline `dla.py` applies when it reports
                      a reconstruction residual as its readability floor.
  `control_residual`  the identical fit on the same matrix with every column
                      independently shuffled. That destroys co-firing while
                      preserving each neuron's own distribution, so it is what
                      a k-component fit achieves on data with no co-firing
                      structure at all. If the real fit does not beat it, the
                      components are not evidence of anything and the response
                      says so. This is `nullmodel.py`'s question asked of a
                      factorisation instead of a ranking.

                      RUN MORE THAN ONCE, because one draw from the null is a
                      number with no error bar and `residual < control` is a
                      coin flip when the two are close. On a 200 x 32 matrix
                      whose columns are drawn INDEPENDENTLY — provably zero
                      co-firing — the shipped margin at seed 0 was +0.000975
                      against a measured control spread of 0.001005, and the
                      bare `<` published "it beats it, so there is co-firing
                      structure here". `control_repeats` shuffles are fitted,
                      their mean and sample standard deviation are both
                      reported, and the verdict is taken against the spread
                      rather than against the mean alone. A margin inside the
                      spread reads as UNDECIDED, which is a third answer and
                      not a quiet False.
  `stability`         mean best-match cosine between this fit's components and
                      a second fit from a different seed. NMF has no unique
                      solution; two seeds can land on two different bases for
                      the same data. A low number means the components you are
                      reading are an artefact of an initialisation.

                      LOW AGAINST WHAT. This statistic does not read 0 when
                      two loading sets are unrelated: it is a mean over
                      best-match cosines of non-negative vectors, which are
                      all in one orthant. Two INDEPENDENT uniform non-negative
                      `[k, m]` matrices score 0.8429 at k=6, m=24 and 0.7793
                      at k=12, m=256 — measured, not argued. So the floor is
                      measured on this fit's own shape and published beside
                      the number as `stability_floor`, and a stability of 0.74
                      is BELOW chance rather than "fairly stable".

## NEGATIVES, WHICH ARE A DECISION AND NOT A DETAIL

NMF needs a non-negative matrix and MLP activations are not one. How far from
one depends on the architecture and is measured rather than assumed:

  * GPT-2-style `gelu(x)` has a bounded negative lobe — no output below about
    -0.17 — so the negative mass is real but small.
  * Llama/Qwen/Gemma-style gated MLPs feed `silu(gate) * up` into the
    down-projection, and `up` is an unbounded linear map. The negative lobe is
    NOT bounded there, and clamping it is not a rounding decision.

So the policy is named in the request and its cost is measured in the response,
never silently applied:

  `clip`   negatives become zero. Simple, and it THROWS INFORMATION AWAY —
           `discarded_mass_share` is the fraction of the matrix's absolute mass
           that went, measured on this corpus, and it is reported whether it is
           0.4% or 30%.
  `split`  every neuron becomes two non-negative channels, `relu(a)` and
           `relu(-a)`. Nothing is discarded, and the thing being factorised
           changes: a component may now group "neuron 5 firing" with "neuron 9
           going negative", which is a different object from a co-firing group
           and is labelled `lobe` so the reader can see which it is. Costs
           twice the columns.

## WHAT IS NOT A NUMBER

An fp16 MLP that overflows produces `inf`; a division that underflows inside
one produces `nan`. Neither is a small activation and neither is a large one —
they are the absence of a measurement, and the whole file is built on the rule
that an absence never gets to look like a value.

They are also invisible to the obvious guard. `nan < x`, `nan > x` and
`nan == x` are all False, so `if activation > 0` counts a NaN token as "did
not fire" and `if residual < control_residual` publishes "the control was not
beaten" for a fit whose arithmetic never happened. Every one of those reads as
a confident negative sentence about the model.

So finiteness is checked explicitly and never inferred from a comparison:

  * `sweep` and `evidence` exclude non-finite entries from every accumulator
    and report `n_nonfinite_entries` beside the totals. A firing rate is over
    the tokens where that neuron produced a NUMBER, which is its own
    denominator and is reported as `n_finite`.
  * A neuron with no finite activation at all is `firing_rate=None`,
    `max_activation=None`, `min_activation=None` and is counted in
    `n_neurons_unmeasured` — NOT in `n_never_fired`, which is a claim about
    the corpus and not about the arithmetic.
  * `nmf` refuses a matrix with a non-finite entry rather than returning a
    NaN residual, and `beats_control` is `None` rather than False when either
    residual is not a number.

## MEMORY, AND WHERE THE CAPS ARE

`[n_tokens, d_mlp]` for an 8192-wide MLP over a 200,000-token corpus is 6.5 GB
in float32. Nothing here ever holds that. The sweep streams one sequence at a
time and keeps per-neuron accumulators; the matrix NMF actually sees is a
bounded reservoir sample of rows over a bounded selection of columns. Three
caps, and every one of them is reported next to the true count it cut:

  `MAX_TOKENS`           how much of the corpus is read at all
  `MAX_SEQUENCE_TOKENS`  the longest single sequence held in memory, which is
                         what actually bounds the peak
  `SAMPLE_ROWS` and `MAX_NMF_NEURONS`  the shape of the matrix that is fitted

Peak for an 8192-wide MLP at the defaults: 4096 x 8192 x 4 = 134 MB for the
live sequence, plus 2048 x 8192 x 4 = 67 MB for the reservoir. `cost` prices
all of it, in forward passes and in bytes, before a pass is spent.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass

from .errors import BadRequest, Refusal

# Shared with the SAE panel ON PURPOSE rather than redeclared. A reader
# comparing a neuron's firing rate against an SAE feature's is comparing two
# numbers whose denominators must be the same corpus prefix, and two constants
# that happen to be equal today are two constants that can stop being equal.
from .feature_corpus import HISTOGRAM_BINS, MAX_TOKENS, SPAN_CONTEXT, Span

# The longest single sequence held in activation form. This, not MAX_TOKENS,
# is what bounds the peak: the sweep streams sequence by sequence, so what
# lives at once is `[S, d_mlp]` for the current S. A 200,000-token corpus split
# into 4,000-token documents costs 134 MB at d_mlp=8192; the same corpus as ONE
# document would cost 6.5 GB, and nothing in a `.txt` file stops it being one
# document. Sequences longer than this are cut, and the cut is counted and
# reported -- see `NeuronStats.n_sequences_cut`.
MAX_SEQUENCE_TOKENS = 4096

# Rows in the matrix NMF sees. A uniform reservoir sample over every token
# read, not the first N: the first N tokens are the first few sequences, and a
# factorisation of the first few sequences is a factorisation of whatever those
# happened to be about.
SAMPLE_ROWS = 2048

# Columns in that matrix. The most-firing neurons in the layer, which is a
# choice with a bias in it -- a neuron that fires rarely and sharply is exactly
# the kind this cap drops. The true count is reported as `n_neurons_offered`
# beside it, and a caller who wants the tail can raise the cap and pay for it.
MAX_NMF_NEURONS = 512

# Multiplicative updates converge slowly and monotonically. 200 is where the
# relative residual stopped moving by more than NMF_TOLERANCE on the fixtures
# in tests/test_neurons.py; the count actually run is reported, and so is
# whether it stopped early or ran out.
NMF_ITERATIONS = 200
NMF_TOLERANCE = 1e-5

# Components. Small on purpose: the point is to turn forty co-firing neurons
# into one row a person can read, and thirty components is another list nobody
# reads.
DEFAULT_COMPONENTS = 12

NEGATIVE_POLICIES = ("clip", "split")

# Draws from the null per `decompose`, and how far outside their own spread a
# margin has to fall before the comparison is called either way.
#
# One draw is what shipped, and one draw has no spread at all: `residual <
# control_residual` then decides on whatever the single shuffle happened to
# land on. Measured on a 200 x 32 matrix whose columns are drawn
# INDEPENDENTLY — provably zero co-firing — the single-draw margin at seed 0
# was +0.000975 against a spread of 0.001005 across draws, and the response
# published "It beats it, so there is co-firing structure here beyond what the
# method finds in anything."
#
# TWO spreads and not one, because the spread is itself estimated from
# CONTROL_REPEATS numbers and a one-sigma rule on a three-sample sigma decides
# almost as freely as no rule. At two, the same structureless matrix returns
# "inside the noise" at every seed tried, and the module's own fixture — where
# the margin is 0.28 against a spread of 0.038, about seven spreads — still
# reads "beaten".
CONTROL_REPEATS = 3
CONTROL_MARGIN_SPREADS = 2.0

# Keeps the multiplicative updates from dividing by zero. Lee & Seung's rules
# have a denominator that goes to zero exactly when a component dies, which is
# a normal thing for a component to do.
_EPS = 1e-10

# The caveat, as a value rather than as prose in a docstring, because a caller
# that returns a dict of numbers to a browser cannot return a docstring. It is
# attached to every result this module produces.
POLYSEMANTIC = (
    "NEURONS ARE POLYSEMANTIC. A single MLP neuron routinely responds to "
    "several unrelated things, and that is the reason sparse autoencoders "
    "exist rather than a defect of this measurement. A coherent-looking set "
    "of top spans is not evidence that this neuron has one meaning — it is "
    "the intersection of your corpus with one axis of a basis the model never "
    "chose to be interpretable in. This is a blunter instrument than an SAE, "
    "not a worse one, and it is the only one available for a model no SAE was "
    "published for."
)


# --------------------------------------------------------- where a neuron is

# Every spelling this module accepts, in one place, so the lookup below and
# the refusal that names them cannot drift apart. They did: `w2` was accepted
# and named nowhere, and a caller holding a Mixtral expert read a refusal that
# listed four spellings and stopped.
_PROJECTION_SPELLINGS = ("down_proj", "c_proj", "dense_4h_to_h", "fc2", "w2")


def mlp_projection(block):
    """The MLP's output projection — the module whose INPUT is the neurons.

    A neuron is one column of this projection's weight, and its activation is
    the corresponding element of this projection's input. That is the only
    place the neurons exist separably: after the projection they are summed
    into the residual stream and cannot be pulled apart, which is the same
    geometry `ablate.out_projection` documents for attention heads one
    sublayer over.

    FIVE spellings, because the families disagree and none of them is
    guessable: Llama/Qwen/Gemma `mlp.down_proj`, GPT-2 `mlp.c_proj` (a Conv1D,
    so its weight is transposed relative to the others), GPT-NeoX and Falcon
    `mlp.dense_4h_to_h`, OPT `fc2` hung directly off the block with no `mlp`
    at all, and Mixtral/Qwen-MoE `w2` on an expert module.

    The refusal below names all five. A refusal that lists four of the five
    spellings it accepts is a dead end for the caller holding the fifth: they
    read "not supported", and it was.
    """
    holder = getattr(block, "mlp", None)
    holder = holder if holder is not None else block
    for name in _PROJECTION_SPELLINGS:
        found = getattr(holder, name, None)
        if found is not None and hasattr(found, "weight"):
            return found
    raise Refusal(
        f"cannot find the MLP output projection on "
        f"{type(holder).__name__}, so there is no place where this model's "
        f"neurons are still separable. Known spellings: mlp.down_proj "
        f"(Llama, Qwen, Gemma), mlp.c_proj (GPT-2), mlp.dense_4h_to_h "
        f"(GPT-NeoX, Falcon), fc2 (OPT), w2 (Mixtral and Qwen-MoE experts)."
    )


def neuron_count(block) -> int:
    """d_mlp, read off the projection's own input width rather than a config.

    The same rule and the same reason as `ablate.head_geometry`: a width taken
    from anywhere but the tensor being sliced is free to disagree with it, and
    the disagreement is silent. `nn.Linear` states `in_features`; GPT-2's
    Conv1D keeps its weight as `[in, out]` and states nothing, so the width is
    the first dimension there.
    """
    proj = mlp_projection(block)
    width = getattr(proj, "in_features", None)
    if width is None:
        width = int(proj.weight.shape[0])
    width = int(width)
    if width < 1:
        raise Refusal(
            f"this model's MLP output projection reports a width of {width}, "
            f"so there are no neurons to browse."
        )
    return width


def write_direction(proj, neuron: int):
    """The `[d_model]` vector this neuron adds to the residual stream.

    EXACT weight math, no corpus involved. `nn.Linear` holds `[out, in]`, so
    neuron j is column j; GPT-2's Conv1D holds `[in, out]`, so neuron j is row
    j. Getting that backwards does not error — it returns a vector of the right
    length belonging to a different neuron, and every logit below it would be
    confidently about the wrong one.
    """
    weight = proj.weight
    if getattr(proj, "in_features", None) is not None:
        width = int(proj.in_features)
    else:
        width = int(weight.shape[0])
    if not isinstance(neuron, int) or isinstance(neuron, bool):
        raise BadRequest(f"a neuron index has to be a whole number, got {neuron!r}.")
    if not 0 <= neuron < width:
        raise BadRequest(
            f"neuron {neuron} is outside this layer, which has {width:,} of "
            f"them — 0 to {width - 1}."
        )
    if getattr(proj, "in_features", None) is not None:
        return weight[:, neuron].detach().float()
    return weight[neuron, :].detach().float()


def capture_neuron_activations(model, block, ids):
    """One pass, returning this block's MLP-projection input: `[S, d_mlp]`.

    Written the way `ablate.capture_projection_inputs` (ablate.py:191) is
    written, and deliberately not a second mechanism: a forward PRE hook on the
    projection, reading `args[0]`, so what comes back is exactly the tensor
    `write_direction` indexes into. Capturing the MLP's OUTPUT and trying to
    invert the projection would be a second implementation of this module's
    geometry, free to drift from it — and not invertible anyway, since
    `down_proj` maps d_mlp down to d_model.

    ONE ROW PER TOKEN IS AN INVARIANT AND IT IS CHECKED HERE. Everything
    downstream indexes `tokens[row]` — the span this activation happened on,
    the sequence it came from, the reservoir's provenance — so a capture whose
    row count disagrees with the tokenisation is not a slightly-off answer, it
    is a span attributed to the wrong token. A routed or expert MLP whose
    shared output projection sees each token more than once produces exactly
    that, and the `w2` spelling this file accepts is the spelling those models
    use. Without this check the failure surfaces one frame later as a raw
    `IndexError` out of a list lookup, which is the error class this module's
    contract promises never to leak.
    """
    import torch

    sink: list = []

    def hook(module, args):  # torch's signature
        sink.append(args[0].detach())

    handle = mlp_projection(block).register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(ids)
    finally:
        handle.remove()
    if not sink:
        raise Refusal(
            "the capture hook on this block's MLP never fired, so no neuron "
            "activations were produced. The block that was hooked is not on "
            "this model's forward path."
        )
    captured = sink[0]
    # `[B, S, d_mlp]` from a real model, `[S, d_mlp]` from anything that skips
    # the batch axis. Both are accepted; what leaves here is always 2-D.
    if captured.dim() == 3:
        captured = captured[0]
    if captured.dim() != 2:
        raise Refusal(
            f"the capture on this block's MLP returned a "
            f"{captured.dim()}-dimensional tensor, and a neuron browser needs "
            f"a [tokens, neurons] matrix. Whatever this projection is fed, it "
            f"is not one activation row per token."
        )
    expected = int(ids.shape[-1])
    if int(captured.shape[0]) != expected:
        raise Refusal(
            f"the capture on this block's MLP returned "
            f"{int(captured.shape[0]):,} activation rows for {expected:,} "
            f"tokens, so there is no one-to-one map from a row to a token and "
            f"every span shown below would be attributed to a token it did "
            f"not happen on. A routed or expert MLP whose shared output "
            f"projection sees each token more than once produces this; a "
            f"per-expert projection has to be browsed one expert at a time."
        )
    return captured.float().cpu()


# ------------------------------------------------------------- what came back


@dataclass
class NeuronRow:
    """Everything the sweep measured about one neuron, over one corpus.

    `mean_positive` and `firing_rate` are `None` rather than `0.0` when there
    is nothing to average or divide by. A neuron that never went positive has
    no mean positive activation; that is not the same statement as a mean of
    zero, and a table that prints 0.0 for both has erased the difference.

    `max_activation` and `min_activation` are real whenever this neuron
    produced a real number anywhere: unlike an SAE feature, a neuron produces
    a value at every single token, so "not measured here" for its range means
    only one thing — every value it produced was NaN or infinite. Then they
    are `None` too, and `n_finite` is 0.

    `n_finite` is the denominator of `firing_rate`, and it is carried rather
    than assumed equal to the corpus size. A neuron whose column overflowed on
    some tokens fired on some fraction of the tokens where it produced a
    NUMBER; dividing by the corpus size instead would silently count every
    unreadable token as a token it did not fire on.
    """

    neuron: int
    n_fired: int
    n_negative: int
    max_activation: float | None
    min_activation: float | None
    mean_positive: float | None
    firing_rate: float | None
    n_finite: int = 0
    n_nonfinite: int = 0

    def to_dict(self) -> dict:
        return {
            "neuron": self.neuron,
            "n_fired": self.n_fired,
            "n_negative": self.n_negative,
            "max_activation": (
                None if self.max_activation is None else round(self.max_activation, 5)
            ),
            "min_activation": (
                None if self.min_activation is None else round(self.min_activation, 5)
            ),
            "mean_positive": (
                None if self.mean_positive is None else round(self.mean_positive, 5)
            ),
            "firing_rate": (
                None if self.firing_rate is None else round(self.firing_rate, 6)
            ),
            "n_finite": self.n_finite,
            "n_nonfinite": self.n_nonfinite,
        }


@dataclass
class NeuronStats:
    """The corpus, its size, and every cap that was applied to it.

    A top activation is a top activation IN THIS CORPUS — the rule
    `feature_corpus.CorpusStats` exists to enforce, unchanged here. Added to
    it: the three caps this module has to apply to stay inside memory, each
    beside the true count it cut, because a truncated corpus that does not say
    it was truncated turns "this neuron never fired" into a claim about text
    nobody read.

    `truncated` MEANS THE TOKEN CAP FIRED, and nothing else. It used to be
    `n_sequences < len(texts)`, which is also true when a sequence tokenised
    to nothing — so a corpus with three blank lines in it published "the
    corpus was cut at 200,000 tokens: 3 of 4 sequences were not read at all"
    after reading four tokens. A cap report naming a cap that never fired is
    worse than no cap report. Blank sequences are their own count,
    `n_sequences_empty`, with their own sentence.
    """

    corpus_label: str
    corpus_sha256: str
    layer: int
    n_sequences: int
    n_sequences_offered: int
    n_sequences_cut: int
    n_tokens: int
    n_tokens_dropped: int
    n_neurons: int
    n_never_fired: int
    negative_mass_share: float | None
    layer_median_firing_rate: float | None
    truncated: bool = False
    n_sequences_empty: int = 0
    n_sequences_unread: int = 0
    n_nonfinite_entries: int = 0
    n_neurons_unmeasured: int = 0

    @property
    def never_fired_share(self) -> float | None:
        return (self.n_never_fired / self.n_neurons) if self.n_neurons else None

    def to_dict(self) -> dict:
        share = self.never_fired_share
        return {
            "corpus_label": self.corpus_label,
            "corpus_sha256": self.corpus_sha256,
            "layer": self.layer,
            "n_sequences": self.n_sequences,
            "n_sequences_offered": self.n_sequences_offered,
            "n_sequences_cut": self.n_sequences_cut,
            "n_sequences_empty": self.n_sequences_empty,
            "n_sequences_unread": self.n_sequences_unread,
            "n_tokens": self.n_tokens,
            "n_tokens_dropped": self.n_tokens_dropped,
            "n_neurons": self.n_neurons,
            "n_never_fired": self.n_never_fired,
            "never_fired_share": None if share is None else round(share, 4),
            # Counted and reported, never folded into `n_never_fired`. A
            # neuron whose every activation was NaN did not "not fire" — it
            # produced nothing that could be compared to zero.
            "n_neurons_unmeasured": self.n_neurons_unmeasured,
            "n_nonfinite_entries": self.n_nonfinite_entries,
            "negative_mass_share": (
                None
                if self.negative_mass_share is None
                else round(self.negative_mass_share, 5)
            ),
            "layer_median_firing_rate": (
                None
                if self.layer_median_firing_rate is None
                else round(self.layer_median_firing_rate, 6)
            ),
            "truncated": self.truncated,
            "max_tokens": MAX_TOKENS,
            "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
            "polysemantic": POLYSEMANTIC,
            "means": self.means(),
        }

    def means(self) -> str:
        share = self.never_fired_share
        median = (
            "not measured — no tokens were read"
            if self.layer_median_firing_rate is None
            else f"{self.layer_median_firing_rate:.1%}"
        )
        text = (
            f"Measured on {self.corpus_label}: {self.n_tokens:,} tokens in "
            f"{self.n_sequences} sequences, read at the MLP of layer "
            f"{self.layer}. {self.n_neurons:,} neurons, of which "
            f"{self.n_never_fired:,}"
            + (f" ({share:.1%})" if share is not None else "")
            + f" never went positive here — that is NOT SEEN IN THIS CORPUS, "
            f"not dead. The median neuron in this layer fired on {median} of "
            f"these tokens, and every firing rate below should be read "
            f"against that rather than against an SAE's, where a rate this "
            f"high would mean something entirely different."
        )
        if self.n_neurons_unmeasured or self.n_nonfinite_entries:
            text += (
                f" {self.n_nonfinite_entries:,} activation"
                f"{'s were' if self.n_nonfinite_entries != 1 else ' was'} NaN "
                f"or infinite and excluded from every count above, and "
                f"{self.n_neurons_unmeasured:,} neuron"
                f"{'s' if self.n_neurons_unmeasured != 1 else ''} produced no "
                f"finite activation at all. Those are UNMEASURED, not silent: "
                f"they are excluded from the never-fired count and from the "
                f"median rather than counted as neurons that fired on nothing."
            )
        if self.negative_mass_share is not None:
            text += (
                f" {self.negative_mass_share:.1%} of the absolute activation "
                f"mass in this layer was NEGATIVE — measured over every neuron "
                f"and every finite token that was read. That is the half "
                f"`decompose` has to make a decision about; it is NOT what the "
                f"`clip` policy actually throws away, which is smaller and is "
                f"a different measurement: clip runs on the selected columns "
                f"of the sampled rows only, and reports its own share as "
                f"`discarded_mass_share` beside the factorisation."
            )
        if self.truncated:
            text += (
                f" The corpus was cut at {MAX_TOKENS:,} tokens: "
                f"{self.n_sequences_unread} of "
                f"{self.n_sequences_offered} sequences were not read at all."
            )
        if self.n_sequences_empty:
            text += (
                f" {self.n_sequences_empty} sequence"
                f"{'s' if self.n_sequences_empty != 1 else ''} tokenised to "
                f"nothing and produced no activations. That is a fact about "
                f"the file — blank lines — and not a cap this module applied."
            )
        if self.n_sequences_cut:
            text += (
                f" {self.n_sequences_cut} sequence"
                f"{'s were' if self.n_sequences_cut != 1 else ' was'} longer "
                f"than {MAX_SEQUENCE_TOKENS:,} tokens and "
                f"{self.n_tokens_dropped:,} tokens were dropped off the end of "
                f"them, because what bounds this sweep's memory is the longest "
                f"single sequence held at once."
            )
        return text


@dataclass
class TokenSample:
    """The bounded matrix NMF is actually fitted to, and where its rows came
    from.

    `values` is `[n_rows, d_mlp]` and SIGNED — the negative policy is applied
    in `decompose`, not here, so that the same sample can be factorised both
    ways and the two answers compared. The row provenance travels alongside it
    so a component can point at real spans without a second forward pass.

    `n_tokens_seen` is the TRUE number of tokens the reservoir sampled from,
    which is the denominator that makes `n_rows` mean something. A sample of
    2,048 rows out of 2,048 tokens is the whole corpus; out of 200,000 it is
    one percent of it, and the two are not the same evidence.
    """

    values: object
    tokens: list[str]
    contexts: list[str]
    offsets: list[int]
    sequences: list[int]
    positions: list[int]
    n_tokens_seen: int
    capacity: int
    seed: int

    @property
    def n_rows(self) -> int:
        return len(self.tokens)

    def to_dict(self) -> dict:
        """Everything except the matrix. A `[2048, 8192]` tensor is not a JSON
        response, and a caller that wants it has the object."""
        return {
            "n_rows": self.n_rows,
            "n_tokens_seen": self.n_tokens_seen,
            "capacity": self.capacity,
            "seed": self.seed,
            "sampled_share": (
                round(self.n_rows / self.n_tokens_seen, 5)
                if self.n_tokens_seen
                else None
            ),
            "means": (
                f"{self.n_rows:,} token rows, sampled uniformly at random from "
                f"the {self.n_tokens_seen:,} tokens that were read, with seed "
                f"{self.seed} so the same corpus resamples the same way. "
                f"Uniform over the whole corpus rather than the first "
                f"{self.capacity:,} tokens — those are the first few sequences, "
                f"and a factorisation of the first few sequences is a "
                f"factorisation of whatever they happened to be about."
            ),
        }


# ------------------------------------------------------------- the sweep


def _reservoir_slot(rng: random.Random, seen: int, capacity: int) -> int | None:
    """Where token number `seen` (0-based) goes in the reservoir, or nowhere.

    Algorithm R. Uniform over a stream whose length is not known in advance,
    which is the situation here — the sweep does not know how many tokens the
    corpus has until it has tokenised all of it, and tokenising all of it just
    to compute a stride is the pass this is trying not to take twice.
    """
    if capacity <= 0:
        return None
    if seen < capacity:
        return seen
    slot = rng.randrange(seen + 1)
    return slot if slot < capacity else None


def _span_text(tokens: list[str], position: int) -> tuple[str, int]:
    """The context window around one token, and where the token starts in it.

    The offset is not derivable by searching the window for the token: a
    window containing the same token twice would highlight both, which tells
    a reader the neuron fired twice there. `feature_corpus.Span` documents
    this at length and carries the field for exactly this reason.
    """
    lo = max(0, position - SPAN_CONTEXT)
    hi = min(len(tokens), position + SPAN_CONTEXT + 1)
    return "".join(tokens[lo:hi]), len("".join(tokens[lo:position]))


def _stream(model, block, tokenizer, texts: list[str], device, state: dict):
    """Yield `(index, token strings, [S, d_mlp] activations)`, one sequence at
    a time, inside both caps.

    `state` IS WHY IT STOPPED, and it exists because a caller cannot tell the
    two reasons apart from the outside. A generator that ran out and a
    generator that hit `MAX_TOKENS` both just stop, and both leave
    `n_sequences < len(texts)` when some sequence tokenised to nothing. The
    consumer used to infer the cap from that inequality and published a cap
    report after reading four tokens of a 200,000-token budget. So the reason
    is recorded here, where it is known:

      `cap_hit`             the MAX_TOKENS cap actually fired
      `n_sequences_empty`   sequences that tokenised to nothing
      `n_sequences_unread`  sequences after the one the cap stopped at

    THE CAP IS APPLIED BEFORE THE FORWARD PASS, not after. `feature_corpus`
    checks its cap at the top of the consuming loop, which means the pass for
    the sequence that trips it has already been paid for. More importantly, it
    means the two passes in a module have to be careful to agree on where they
    stopped — the comment at feature_corpus.py:359 records that they once did
    not, and the firing rate shown for a feature was computed over a different
    denominator from the rates beside it. Deciding here, once, is what makes
    `sweep` and `evidence` read the same corpus prefix by construction.

    The first sequence always goes through even if it alone exceeds the cap.
    Otherwise a corpus that is one long document measures nothing at all and
    reports it as "this neuron never fired".
    """
    state.setdefault("cap_hit", False)
    state.setdefault("n_sequences_empty", 0)
    state.setdefault("n_sequences_unread", 0)
    n_tokens = 0
    for index, text in enumerate(texts):
        ids = tokenizer(text, return_tensors="pt")["input_ids"]
        if ids.shape[-1] == 0:
            state["n_sequences_empty"] += 1
            continue
        dropped = max(0, int(ids.shape[-1]) - MAX_SEQUENCE_TOKENS)
        if dropped:
            ids = ids[:, :MAX_SEQUENCE_TOKENS]
        size = int(ids.shape[-1])
        if n_tokens and n_tokens + size > MAX_TOKENS:
            state["cap_hit"] = True
            state["n_sequences_unread"] = len(texts) - index
            return
        ids = ids.to(device)
        acts = capture_neuron_activations(model, block, ids)
        n_tokens += size
        yield (
            index,
            [tokenizer.decode([int(t)]) for t in ids[0]],
            acts,
            dropped,
        )
        # Dropped as soon as the consumer is done with it. `[S, d_mlp]` is the
        # largest thing this module holds and the next iteration is about to
        # allocate another one.
        del acts


def sweep(
    model,
    block,
    tokenizer,
    texts: list[str],
    *,
    device,
    layer: int,
    corpus_label: str = "",
    corpus_sha: str = "",
    sample_rows: int = SAMPLE_ROWS,
    seed: int = 0,
) -> tuple[NeuronStats, dict, TokenSample]:
    """Firing rate, peak, floor and negative mass for every neuron in a layer.

    Returns `(stats, per_neuron, sample)`. `per_neuron` maps neuron id to a
    `NeuronRow` for EVERY neuron, not only the ones that fired: unlike an SAE
    feature, a neuron produces a value at every token, so "did not appear in
    the table" would be a fact about the table rather than about the model.

    `sample` is the bounded reservoir `decompose` factorises. It is collected
    on this same pass rather than on a second one — the alternative is running
    the whole corpus through the model twice to produce two views of the same
    activations, and the reservoir costs `sample_rows x d_mlp` floats, which
    is 67 MB at the defaults for an 8192-wide MLP.

    Blocking; call it from a worker thread.
    """
    import torch

    if not isinstance(texts, list) or not texts:
        raise BadRequest(
            "a neuron sweep needs text. Pass a list of strings, or a local "
            ".txt/.jsonl through `feature_corpus.load_corpus`. Nothing is "
            "downloaded."
        )
    if isinstance(sample_rows, bool) or not isinstance(sample_rows, int):
        raise BadRequest(f"sample_rows has to be a whole number, got {sample_rows!r}.")
    if sample_rows < 1:
        raise BadRequest(
            f"sample_rows has to be at least 1, got {sample_rows}. Without "
            f"rows there is no matrix to factorise."
        )

    width = neuron_count(block)
    fired = torch.zeros(width, dtype=torch.int64)
    negative = torch.zeros(width, dtype=torch.int64)
    # Per-neuron, not one corpus-wide number: an overflow is a property of one
    # neuron's arithmetic on one token, and this is the denominator every rate
    # below is taken over.
    finite_counts = torch.zeros(width, dtype=torch.int64)
    peak = torch.full((width,), float("-inf"), dtype=torch.float32)
    floor = torch.full((width,), float("inf"), dtype=torch.float32)
    positive_sum = torch.zeros(width, dtype=torch.float64)
    absolute_mass = 0.0
    negative_mass = 0.0

    capacity = int(sample_rows)
    values = torch.zeros(capacity, width, dtype=torch.float32)
    rows: list[tuple[str, str, int, int, int]] = [("", "", 0, 0, 0)] * capacity
    n_rows = 0
    rng = random.Random(seed)

    n_tokens = 0
    n_sequences = 0
    n_sequences_cut = 0
    n_tokens_dropped = 0
    n_nonfinite_entries = 0

    stream_state: dict = {}
    for index, tokens, acts, dropped in _stream(
        model, block, tokenizer, texts, device, stream_state
    ):
        # EXPLICIT, because every comparison below silently answers False for
        # a NaN and would count it as "did not fire" and as "not negative" at
        # the same time. `finite` is the mask; `clean` is the tensor with the
        # unreadable entries replaced by an exact zero, which is neutral for
        # every accumulator here and is never itself counted as a firing.
        finite = torch.isfinite(acts)
        n_finite_here = int(finite.sum())
        n_bad_here = int(acts.numel()) - n_finite_here
        finite_counts += finite.sum(dim=0).to(torch.int64)
        n_nonfinite_entries += n_bad_here
        # `clean` is only built when there is something to clean: on a healthy
        # model this branch never runs and the sweep allocates nothing extra,
        # which matters because `[S, d_mlp]` is the largest thing here.
        clean = acts
        high = acts
        low = acts
        if n_bad_here:
            clean = torch.where(finite, acts, torch.zeros((), dtype=acts.dtype))
            # -inf / +inf where this neuron produced nothing readable, so the
            # running peak and floor are untouched by the bad token rather
            # than poisoned to NaN by it.
            high = torch.where(
                finite, acts, torch.tensor(float("-inf"), dtype=acts.dtype)
            )
            low = torch.where(
                finite, acts, torch.tensor(float("inf"), dtype=acts.dtype)
            )

        positive = clean.clamp(min=0.0)
        fired += (clean > 0).sum(dim=0).to(torch.int64)
        negative += (clean < 0).sum(dim=0).to(torch.int64)
        peak = torch.maximum(peak, high.max(dim=0).values)
        floor = torch.minimum(floor, low.min(dim=0).values)
        positive_sum += positive.sum(dim=0).to(torch.float64)
        absolute_mass += float(clean.abs().sum())
        negative_mass += float((-clean).clamp(min=0.0).sum())

        for offset_in_sequence in range(acts.shape[0]):
            slot = _reservoir_slot(rng, n_tokens + offset_in_sequence, capacity)
            if slot is None:
                continue
            values[slot] = acts[offset_in_sequence]
            text, start = _span_text(tokens, offset_in_sequence)
            rows[slot] = (
                text,
                tokens[offset_in_sequence],
                start,
                index,
                offset_in_sequence,
            )
            n_rows = max(n_rows, slot + 1)

        n_tokens += int(acts.shape[0])
        n_sequences += 1
        if dropped:
            n_sequences_cut += 1
            n_tokens_dropped += dropped

    if n_tokens == 0:
        raise Refusal(
            "every sequence in this corpus tokenised to nothing, so no neuron "
            "activation was produced. Check that the file is text rather than "
            "empty lines."
        )

    # Divided by each neuron's OWN finite count, not by the corpus size. They
    # are equal on every healthy model and they are not equal on one that
    # overflowed, and in that case dividing by the corpus size would report a
    # rate over tokens where the neuron produced nothing to compare to zero.
    finite_list = finite_counts.tolist()
    fired_list = fired.tolist()
    per_neuron = {
        i: NeuronRow(
            neuron=i,
            n_fired=fired_list[i],
            n_negative=int(negative[i]),
            # None, not the sentinel -inf/inf the accumulators started at: a
            # neuron that produced no finite value has no range, and a table
            # printing -inf has published a number nobody measured.
            max_activation=float(peak[i]) if finite_list[i] else None,
            min_activation=float(floor[i]) if finite_list[i] else None,
            # None, not 0.0. A neuron with no positive activations has no mean
            # positive activation, and printing 0.0 makes it indistinguishable
            # from one that fired constantly at exactly zero.
            mean_positive=(
                float(positive_sum[i]) / fired_list[i] if fired_list[i] else None
            ),
            # None, not 0.0, when nothing about this neuron was readable.
            firing_rate=(fired_list[i] / finite_list[i] if finite_list[i] else None),
            n_finite=finite_list[i],
            n_nonfinite=n_tokens - finite_list[i],
        )
        for i in range(width)
    }

    # The median is over the neurons that HAVE a rate. Including the
    # unmeasured ones as rate-0 would drag the reference number every other
    # neuron is reported against toward zero — measured on an 8-neuron fixture
    # with two NaN columns, the median moved from 0.458333 to 0.270833.
    ordered = sorted(r.firing_rate for r in per_neuron.values() if r.n_finite)
    if not ordered:
        median = None
    else:
        middle = len(ordered) // 2
        median = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2.0
        )

    stats = NeuronStats(
        corpus_label=corpus_label or f"{len(texts)} sequences",
        corpus_sha256=corpus_sha or _corpus_hash(texts),
        layer=layer,
        n_sequences=n_sequences,
        n_sequences_offered=len(texts),
        n_sequences_cut=n_sequences_cut,
        n_tokens=n_tokens,
        n_tokens_dropped=n_tokens_dropped,
        n_neurons=width,
        # A neuron with no finite activation is NOT one that never fired: it
        # is one nothing was measured about. Counting it here would publish
        # "never went positive here — that is NOT SEEN IN THIS CORPUS" about
        # arithmetic that never produced a comparable number.
        n_never_fired=int(((fired == 0) & (finite_counts > 0)).sum()),
        n_neurons_unmeasured=int((finite_counts == 0).sum()),
        n_nonfinite_entries=n_nonfinite_entries,
        negative_mass_share=(
            negative_mass / absolute_mass if absolute_mass > 0 else None
        ),
        layer_median_firing_rate=median,
        # The cap fired, or it did not. Not inferred from a sequence count
        # that blank lines also move.
        truncated=bool(stream_state.get("cap_hit")),
        n_sequences_empty=int(stream_state.get("n_sequences_empty", 0)),
        n_sequences_unread=int(stream_state.get("n_sequences_unread", 0)),
    )
    sample = TokenSample(
        values=values[:n_rows],
        tokens=[r[1] for r in rows[:n_rows]],
        contexts=[r[0] for r in rows[:n_rows]],
        offsets=[r[2] for r in rows[:n_rows]],
        sequences=[r[3] for r in rows[:n_rows]],
        positions=[r[4] for r in rows[:n_rows]],
        n_tokens_seen=n_tokens,
        capacity=capacity,
        seed=int(seed),
    )
    return stats, per_neuron, sample


def _corpus_hash(texts: list[str]) -> str:
    from .feature_corpus import corpus_hash

    return corpus_hash(texts)


# ------------------------------------------------------- one neuron, up close


def _histogram(column, bins: int) -> tuple[list[int], list[float]]:
    """Fixed-width bins over [min, max], and the edges that go with them.

    Takes the `[n_tokens]` tensor rather than a Python list of floats: at the
    corpus cap that is 800 KB against about 6.4 MB of boxed floats, for a
    value that only ever feeds this function.

    Over the SIGNED range, not `[0, max]` the way `feature_corpus` does it. An
    SAE feature cannot be negative so the origin is its natural left edge; a
    neuron can, and a histogram that starts at zero silently drops the entire
    negative lobe — which is the half of the distribution the NMF section then
    has to make a decision about.

    NON-FINITE VALUES ARE NOT BINNABLE and are dropped here as a last line of
    defence. `torch.histc` raises `RuntimeError: range of [-nan, -nan] is not
    finite` on them, which is a raw torch error leaking out of a module whose
    contract is Refusal or BadRequest. The caller is expected to have counted
    and reported them already; this only makes the leak impossible.
    """
    import torch

    if int(column.numel()) == 0:
        return [], []
    values = column.float()
    values = values[torch.isfinite(values)]
    if int(values.numel()) == 0:
        return [], []
    lo = float(values.min())
    hi = float(values.max())
    if hi == lo:
        # Every value identical: `histc` over a zero-width range returns
        # nothing useful, and a single bin is the honest picture.
        return [int(values.numel())], [round(lo, 5), round(hi, 5)]
    counts = torch.histc(values, bins=bins, min=lo, max=hi)
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    return [int(v) for v in counts], [round(e, 5) for e in edges]


def evidence(
    model,
    block,
    tokenizer,
    texts: list[str],
    neuron: int,
    *,
    device,
    top_k: int = 10,
) -> dict:
    """Top-activating spans and the activation histogram for ONE neuron.

    A second pass over the same corpus prefix `sweep` read, for the reason
    `feature_corpus.evidence` takes one: keeping the top spans of every neuron
    at once is what would make this the memory-heaviest thing the panel can do.
    The sweep answers "which neurons fire and how often"; this answers "show me
    that one".

    The layer's median firing rate is recomputed here rather than passed in.
    That costs one `[d_mlp]` accumulator and nothing else, and it guarantees
    the comparison is against a median measured over the SAME tokens as the
    rate it is compared to — which is the invariant a caller threading a number
    through from an earlier sweep would have to be trusted to preserve.
    """
    import torch

    width = neuron_count(block)
    if isinstance(neuron, bool) or not isinstance(neuron, int):
        raise BadRequest(f"a neuron index has to be a whole number, got {neuron!r}.")
    if not 0 <= neuron < width:
        raise BadRequest(
            f"neuron {neuron} is outside this layer, which has {width:,} of "
            f"them — 0 to {width - 1}."
        )
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise BadRequest(f"top_k must be a whole number of at least 1, got {top_k!r}.")

    # A HEAP OF `top_k`, NOT A LIST OF EVERY FIRING. The first version kept one
    # Span per firing token and sorted at the end, which is fine for a sparse
    # SAE feature and wrong here: a neuron near the layer median fires on about
    # half of everything, so a full 200,000-token corpus would build 100,000
    # span objects — each carrying a context string — to show ten of them. The
    # heap holds `top_k`, and `n_spans_available` below carries the true count
    # so the cap is reported rather than merely applied.
    #
    # `order` breaks ties: Span is not comparable, and two tokens at exactly
    # the same activation are common in any model with repeated tokens.
    import heapq

    kept: list[tuple[float, int, Span]] = []
    order = 0
    n_spans_available = 0

    # float32 chunks, concatenated once, rather than a Python list of floats:
    # 200,000 tokens is 800 KB as a tensor against about 6.4 MB as boxed
    # floats, for a value that only ever feeds a histogram.
    chunks: list[object] = []
    fired = torch.zeros(width, dtype=torch.int64)
    finite_counts = torch.zeros(width, dtype=torch.int64)
    n_tokens = 0
    n_sequences = 0

    stream_state: dict = {}
    for index, tokens, acts, _ in _stream(
        model, block, tokenizer, texts, device, stream_state
    ):
        # Same rule as `sweep`, for the same reason: `acts > 0` answers False
        # for a NaN, so without the mask an unreadable token is counted as a
        # token this neuron did not fire on.
        finite = torch.isfinite(acts)
        finite_counts += finite.sum(dim=0).to(torch.int64)
        fired += ((acts > 0) & finite).sum(dim=0).to(torch.int64)
        column = acts[:, neuron]
        n_tokens += int(acts.shape[0])
        n_sequences += 1
        chunks.append(column.clone())
        readable = (column > 0) & torch.isfinite(column)
        for position in readable.nonzero(as_tuple=True)[0].tolist():
            n_spans_available += 1
            activation = round(float(column[position]), 5)
            if len(kept) == top_k and activation <= kept[0][0]:
                continue
            text, start = _span_text(tokens, position)
            order += 1
            entry = (
                activation,
                order,
                Span(
                    text=text,
                    token=tokens[position],
                    activation=activation,
                    position=position,
                    sequence=index,
                    offset=start,
                ),
            )
            if len(kept) < top_k:
                heapq.heappush(kept, entry)
            else:
                heapq.heappushpop(kept, entry)

    if n_tokens == 0:
        raise Refusal(
            "every sequence in this corpus tokenised to nothing, so this "
            "neuron produced no activations to show."
        )

    # Every summary below comes off the tensor rather than off a Python list
    # comprehension over it, for the same reason the chunks were kept as
    # tensors: at the corpus cap that is 800 KB and one pass instead of 6.4 MB
    # and four.
    column_all = torch.cat(chunks)
    # The readable part of this neuron's column, and the count of what was
    # not. Every number below is over the readable part and the excluded
    # count travels with them.
    column_finite = column_all[torch.isfinite(column_all)]
    n_finite_here = int(column_finite.numel())
    n_nonfinite_here = int(column_all.numel()) - n_finite_here
    if n_finite_here == 0:
        raise Refusal(
            f"every one of neuron {neuron}'s {n_tokens:,} activations in this "
            f"corpus was NaN or infinite, so there is nothing to show. That is "
            f"a fact about this model's arithmetic — an fp16 MLP that "
            f"overflows produces it — and not about the neuron or the text. "
            f"Try the same layer in float32, or a different layer."
        )

    spans = [entry[2] for entry in sorted(kept, key=lambda e: (-e[0], e[1]))]
    n_fired_here = int((column_finite > 0).sum())
    n_negative_here = int((column_finite < 0).sum())
    peak = float(column_finite.max())
    floor = float(column_finite.min())
    counts, edges = _histogram(column_finite, HISTOGRAM_BINS)

    # Over the neurons that produced a number, and over each one's own finite
    # count — the same rule `sweep` applies, so the two medians are comparable.
    finite_list = finite_counts.tolist()
    fired_list = fired.tolist()
    # Never empty: the refusal above already established that this neuron's
    # own column has a finite value in it, so at least one neuron has a rate.
    rates = sorted(
        fired_list[i] / finite_list[i] for i in range(width) if finite_list[i]
    )
    middle = len(rates) // 2
    median = (
        rates[middle] if len(rates) % 2 else (rates[middle - 1] + rates[middle]) / 2.0
    )
    rate = n_fired_here / n_finite_here

    return {
        "neuron": neuron,
        "layer_width": width,
        "spans": [s.to_dict() for s in spans],
        # The cap and the true count together, never the capped list alone.
        "n_spans_available": n_spans_available,
        "n_spans_shown": len(spans),
        "n_fired": n_fired_here,
        "n_negative": n_negative_here,
        "n_tokens": n_tokens,
        # The denominator of `firing_rate`, and the count that was excluded
        # from it. Equal to `n_tokens` on any model whose arithmetic stayed
        # finite, and reported rather than assumed equal.
        "n_finite": n_finite_here,
        "n_nonfinite": n_nonfinite_here,
        "n_sequences": n_sequences,
        "firing_rate": round(rate, 6),
        "layer_median_firing_rate": round(median, 6),
        "max_activation": round(peak, 5),
        "min_activation": round(floor, 5),
        # None, not 0.0, when it never fired: there is no positive activation
        # to average, which is not the same statement as an average of zero.
        "mean_positive": (
            round(float(column_finite.clamp(min=0.0).sum()) / n_fired_here, 5)
            if n_fired_here
            else None
        ),
        "histogram": counts,
        "bin_edges": edges,
        "polysemantic": POLYSEMANTIC,
        "means": (
            (
                f"Neuron {neuron} went positive on {n_fired_here:,} of "
                f"{n_finite_here:,} readable tokens ({rate:.2%}) in this "
                f"corpus, against a median of {median:.2%} across the "
                f"{len(rates):,} of {width:,} neurons in this layer that "
                f"produced a readable value on these same tokens. Read the "
                f"rate against that median and not against an SAE feature's: a "
                f"post-GELU neuron near a coin flip is the ordinary case, not "
                f"a warning sign. It went NEGATIVE on {n_negative_here:,} "
                f"tokens, down to {floor:.5f} — the part `decompose` has to "
                f"make a decision about. These are its highest activations "
                f"HERE, which is a different claim from what it responds to "
                f"generally."
                if n_fired_here
                else (
                    f"Neuron {neuron} never went positive in "
                    f"{n_finite_here:,} readable tokens. That is NOT SEEN IN "
                    f"THIS CORPUS, not dead — this text never showed it "
                    f"anything it responds to, which is a fact about the text "
                    f"as much as about the neuron."
                )
            )
            + (
                f" {n_nonfinite_here:,} of this neuron's {n_tokens:,} "
                f"activations were NaN or infinite and are excluded from every "
                f"count above, including the denominator of that rate. They "
                f"are not zeros and they are not small — they are tokens where "
                f"this model's arithmetic produced nothing comparable."
                if n_nonfinite_here
                else ""
            )
        ),
        # NO LABEL, and not because one was hard to generate. A neuron is
        # polysemantic; a name would assert the one thing this measurement
        # cannot support.
        "label": None,
    }


def logit_weights(model, tokenizer, block, neuron: int, *, top_k: int = 10) -> dict:
    """Which tokens this neuron's write direction promotes and suppresses.

    EXACT, and needs no corpus, for the reason `feature_corpus.logit_weights`
    gives: a neuron's contribution to the residual stream is a fixed direction
    — one column of `down_proj` — and pushing it through the final norm and the
    unembedding says what it does to every logit. Pure weight math, no
    sampling, nothing that is a sample OF anything.

    The one approximation is the same one, and it is stated in the response
    rather than assumed away: the norm's real scale depends on the stream this
    direction would be added to, so these RANK tokens rather than predict logit
    amounts.

    This is the readout that is arguably BETTER here than for an SAE feature.
    An SAE's decoder row is a learned approximation of a direction the model
    uses; a neuron's write column is a direction the model actually has.
    """
    import torch

    from .lens import _final_norm

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise BadRequest(f"top_k must be a whole number of at least 1, got {top_k!r}.")

    head = model.get_output_embeddings()
    if head is None:
        raise Refusal(
            "this model has no output embedding, so a neuron's effect on the "
            "vocabulary cannot be read."
        )
    norm = _final_norm(model)
    direction = write_direction(mlp_projection(block), neuron)
    if not bool(torch.isfinite(direction).all()):
        n_bad = int((~torch.isfinite(direction)).sum())
        raise Refusal(
            f"{n_bad:,} of the {int(direction.numel()):,} weights in neuron "
            f"{neuron}'s write direction are NaN or infinite, so every logit "
            f"below them would be one too — and a ranking of NaNs is an "
            f"arbitrary order presented as a finding. This is a fact about the "
            f"checkpoint rather than about the neuron; check the layer's "
            f"weights before reading anything off it."
        )

    with torch.no_grad():
        target = direction.to(next(model.parameters()).device)
        parameters = list(norm.parameters())
        if parameters:
            target = target.to(parameters[0].dtype)
        projected = norm(target)
        logits = head(projected).float()
        # Against the vocabulary mean, for the same reason `ablate.py` uses KL
        # and `dla.py` subtracts it: softmax ignores a constant, so a direction
        # that lifts every logit equally has changed nothing.
        centred = logits - logits.mean()
        vocabulary = int(centred.shape[-1])
        if not bool(torch.isfinite(centred).all()):
            n_bad = int((~torch.isfinite(centred)).sum())
            raise Refusal(
                f"{n_bad:,} of this model's {vocabulary:,} logits came back "
                f"NaN or infinite when neuron {neuron}'s write direction was "
                f"pushed through the final norm and the unembedding. Sorting "
                f"them would produce a ranking of nothing; the head or the "
                f"norm is what to look at, not the neuron."
            )
        # HALF THE VOCABULARY, not all of it, AND CUT FROM ONE RANKING. The
        # two lists are opposite ends of the same order, so `top_k` above
        # `vocabulary // 2` makes them OVERLAP and a token is published as
        # both promoted and suppressed — with its own logit printed twice,
        # once negated. Measured on the tiny fixture in tests/test_neurons.py:
        # vocabulary 5, top_k 3, and `<0>` appeared in both lists at -7.33584.
        # Real vocabularies are 32k-150k and never reach it, which is exactly
        # why it would have shipped. The cut is REPORTED as `top_k_applied`
        # rather than made quietly.
        #
        # NO `max(1, ...)` FLOOR. A floor of one reintroduces the defect at
        # vocabulary 1, where `1 // 2` is 0 and a single token was published as
        # both promoted and suppressed. There is no honest one-token answer
        # there, so the honest answer is two empty lists and a sentence saying
        # why — an empty list nobody can misread beats a list of one token
        # that means the opposite of itself.
        #
        # And ONE argsort rather than two independent `topk` calls, because
        # two calls overlap on ties as well: a direction that leaves every
        # logit equal makes `topk(centred, k)` and `topk(-centred, k)` return
        # the SAME first k indices. Slicing one order cannot do that.
        k = min(top_k, vocabulary // 2)
        order = torch.argsort(centred, descending=True)
        promoted = order[:k].tolist()
        suppressed = order[vocabulary - k :].flip(0).tolist() if k else []

    return {
        "neuron": neuron,
        "promotes": [
            {"token": tokenizer.decode([i]), "logit": round(float(centred[i]), 5)}
            for i in promoted
        ],
        "suppresses": [
            {"token": tokenizer.decode([i]), "logit": round(float(centred[i]), 5)}
            for i in suppressed
        ],
        "vocabulary_size": vocabulary,
        "top_k_requested": int(top_k),
        "top_k_applied": k,
        "exact": True,
        "polysemantic": POLYSEMANTIC,
        "means": (
            "What this neuron's write direction — one column of the MLP output "
            "projection — does to the vocabulary, read straight through the "
            "final norm and the unembedding. NO CORPUS AND NO SAMPLING: this "
            "is weight arithmetic and it is the same every time. Values are "
            "relative to the vocabulary mean and at unit scale, so they rank "
            "tokens rather than predict logit amounts. A polysemantic neuron "
            "can promote several unrelated groups of tokens at once, and that "
            "is the expected picture rather than a failed measurement."
            + (
                (
                    f" NEITHER LIST COULD BE FILLED: this vocabulary is "
                    f"{vocabulary:,} token{'s' if vocabulary != 1 else ''}, and "
                    f"the two lists are opposite ends of one ranking, so there "
                    f"is no token that could be shown as promoted without "
                    f"being the same token shown as suppressed. The {top_k} "
                    f"asked for is not available at any size here."
                )
                if k == 0
                else (
                    f" Only {k} token{'s' if k != 1 else ''} per side rather "
                    f"than the {top_k} asked for: this vocabulary is "
                    f"{vocabulary:,} tokens and the two lists come from "
                    f"opposite ends of one ranking, so any more would put the "
                    f"same token in both."
                )
                if k < top_k
                else ""
            )
        ),
    }


# ------------------------------------------------- the non-negative decision


def non_negative(values, policy: str) -> tuple[object, dict]:
    """Make an activation matrix non-negative, and report what that cost.

    Never silent. The two policies answer different questions and one of them
    throws information away; which one ran, and how much it discarded on THIS
    corpus, travels with the factorisation.
    """
    import torch

    if policy not in NEGATIVE_POLICIES:
        raise BadRequest(
            f"'{policy}' is not a negative-handling policy. Known: "
            f"{', '.join(NEGATIVE_POLICIES)}. `clip` zeroes negatives and "
            f"reports the mass it discarded; `split` keeps them as a second "
            f"non-negative channel per neuron and doubles the columns."
        )

    # Before any share is taken. With one `inf` in the matrix the denominator
    # `values.abs().sum()` is `inf`, every share below it rounds to 0.0, and
    # the response reports that clipping discarded nothing — a confident
    # sentence about a matrix nobody could measure. With one NaN every share
    # is NaN and prints as `nan%`.
    n_bad = int((~torch.isfinite(values)).sum())
    if n_bad:
        raise Refusal(
            f"{n_bad:,} of this matrix's {int(values.numel()):,} entries are "
            f"NaN or infinite, so no share of its mass can be measured and no "
            f"policy can be honestly priced against it. Those entries are not "
            f"large activations and they are not small ones — they are tokens "
            f"where the arithmetic produced nothing. Drop them before choosing "
            f"a policy; `decompose` drops the rows that hold them and reports "
            f"how many."
        )

    total = float(values.abs().sum())
    negative_mass = float((-values).clamp(min=0.0).sum())
    n_entries = int(values.numel())
    n_negative = int((values < 0).sum())

    if policy == "split":
        matrix = torch.cat([values.clamp(min=0.0), (-values).clamp(min=0.0)], dim=1)
        # 0.0 and not None, and the difference is real: `split` discards
        # nothing BY CONSTRUCTION, whatever the matrix holds. That is a
        # structural fact about the policy, not a share that was measured.
        discarded = 0.0
    else:
        matrix = values.clamp(min=0.0)
        # A measured share, not an assumed-small one. On a gated MLP the `up`
        # branch is an unbounded linear map, so there is no architectural bound
        # on how large this can be.
        #
        # None when there is no mass to take a share OF. An all-zero matrix
        # has no absolute mass, so "0.00% was discarded" is a denominator of
        # zero printed as a confident percentage — the same collapse the
        # sibling field `negative_mass_share` already refuses two lines below,
        # and the same one `mean_positive` and `firing_rate` refuse upstream.
        discarded = (negative_mass / total) if total > 0 else None

    report = {
        "policy": policy,
        "n_entries": n_entries,
        "n_negative_entries": n_negative,
        "negative_entry_share": (
            round(n_negative / n_entries, 5) if n_entries else None
        ),
        "negative_mass_share": (round(negative_mass / total, 5) if total > 0 else None),
        "discarded_mass_share": (None if discarded is None else round(discarded, 5)),
        "columns_in": int(values.shape[1]),
        "columns_out": int(matrix.shape[1]),
        "means": (
            (
                (
                    f"NEGATIVES WERE CLIPPED TO ZERO, discarding "
                    f"{discarded:.2%} of this matrix's absolute mass."
                    if discarded is not None
                    else (
                        "NEGATIVES WERE CLIPPED TO ZERO. What share of this "
                        "matrix that discarded is NOT MEASURABLE: every entry "
                        "in it is exactly zero, so there is no absolute mass "
                        "to take a share of."
                    )
                )
                + " NMF requires a non-negative "
                "input and MLP activations are not one — a GELU has a bounded "
                "negative lobe, but a gated MLP feeds silu(gate) * up into the "
                "projection and `up` is unbounded, so this share is measured "
                "rather than assumed small. Everything below is a "
                "factorisation of the positive part only."
            )
            if policy == "clip"
            else (
                f"NEGATIVES WERE SPLIT OFF rather than discarded: each of the "
                f"{int(values.shape[1]):,} neurons became two non-negative "
                f"channels, relu(a) and relu(-a), for "
                f"{int(matrix.shape[1]):,} columns. Nothing was thrown away, "
                f"and the thing being factorised changed — a component here can "
                f"group 'this neuron firing' with 'that neuron going "
                f"negative', which is a different object from a co-firing "
                f"group. Each entry below is labelled with its lobe."
            )
        ),
    }
    return matrix, report


# ----------------------------------------------------------------- the NMF


def nmf(
    v,
    n_components: int,
    *,
    iterations: int = NMF_ITERATIONS,
    seed: int = 0,
    tolerance: float = NMF_TOLERANCE,
) -> tuple[object, object, dict]:
    """Lee & Seung multiplicative updates for `V ≈ W H`, all three non-negative.

    Implemented here rather than pulled from scikit-learn, which is not a
    dependency of this package and would be a large one to add for two update
    rules. Written in float64: the updates are a long product chain, and a
    residual reported to five places from a float32 chain is a residual
    reported to more places than it has.

    Returns `(W, H, info)`. `info` carries the iterations actually run, whether
    the fit converged or ran out, and the relative Frobenius residual — which
    is this factorisation's resolution, in the sense `dla.py` means when it
    reports a reconstruction residual as a readability floor. A component whose
    mass share is below the residual is not readable against it.
    """
    import torch

    if v.dim() != 2:
        raise BadRequest(
            f"NMF factorises a matrix, and this input has {v.dim()} dimensions."
        )
    n, m = int(v.shape[0]), int(v.shape[1])
    if n == 0 or m == 0:
        raise Refusal(
            f"this matrix is {n} x {m}, so there is nothing to factorise. A "
            f"sweep that read no tokens, or a layer where no neuron fired, "
            f"produces this."
        )
    if isinstance(n_components, bool) or not isinstance(n_components, int):
        raise BadRequest(
            f"n_components has to be a whole number, got {n_components!r}."
        )
    if n_components < 1:
        raise BadRequest(f"n_components has to be at least 1, got {n_components}.")
    if n_components > min(n, m):
        raise BadRequest(
            f"{n_components} components cannot be fitted to a {n} x {m} "
            f"matrix — the rank is at most {min(n, m)}. Sample more tokens, "
            f"raise the neuron cap, or ask for fewer components."
        )
    # BEFORE the sign check, and that order is the whole point: `nan < 0` is
    # False, so a matrix of NaNs walks straight through a guard written as
    # `if v.min() < 0` and the fit returns a NaN residual. A NaN residual then
    # loses every downstream comparison — `nan < control` is False — and the
    # response publishes "the control was not beaten" about arithmetic that
    # never happened.
    n_bad = int((~torch.isfinite(v)).sum())
    if n_bad:
        raise Refusal(
            f"{n_bad:,} of this {n:,} x {m:,} matrix's {n * m:,} entries are "
            f"NaN or infinite. A fit to them returns a NaN residual, and a NaN "
            f"residual compares False against every bound it is later checked "
            f"against, so it would be published as a confident negative result "
            f"rather than as the absence of one."
        )
    if float(v.min()) < 0:
        raise BadRequest(
            "NMF needs a non-negative matrix and this one has negative "
            "entries. Run it through `non_negative` first, which applies a "
            "named policy and reports what that policy cost."
        )
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise BadRequest(f"iterations has to be a whole number, got {iterations!r}.")
    if iterations < 1:
        raise BadRequest(f"iterations has to be at least 1, got {iterations}.")

    matrix = v.to(torch.float64)
    scale = float(matrix.sum())
    if scale <= 0:
        raise Refusal(
            "every entry in this matrix is zero, so there is nothing to "
            "factorise. Either no neuron fired on the sampled tokens, or the "
            "clip policy removed everything that was there."
        )

    generator = torch.Generator().manual_seed(int(seed))
    start = math.sqrt(float(matrix.mean()) / n_components)
    w = (
        torch.rand(n, n_components, generator=generator, dtype=torch.float64) * start
        + _EPS
    )
    h = (
        torch.rand(n_components, m, generator=generator, dtype=torch.float64) * start
        + _EPS
    )

    norm = float(torch.linalg.norm(matrix))
    previous: float | None = None
    residual = 1.0
    run = 0
    converged = False
    for step in range(1, iterations + 1):
        h = h * ((w.T @ matrix) / ((w.T @ w) @ h + _EPS))
        w = w * ((matrix @ h.T) / (w @ (h @ h.T) + _EPS))
        run = step
        residual = float(torch.linalg.norm(matrix - w @ h)) / norm
        if previous is not None and abs(previous - residual) < tolerance:
            converged = True
            break
        previous = residual

    # The input was finite, but the update chain is a long product and a
    # float64 overflow inside it is still possible. A residual that is not a
    # number is not a resolution, and shipping it would put a NaN into every
    # comparison downstream.
    if not math.isfinite(residual):
        raise Refusal(
            f"the multiplicative updates produced a residual of {residual} "
            f"after {run} iterations, so this factorisation has no resolution "
            f"to report. The input was finite, so the fit itself overflowed — "
            f"fewer components or a rescaled matrix is the next thing to try."
        )

    return (
        w,
        h,
        {
            "iterations": run,
            "iterations_offered": int(iterations),
            "converged": converged,
            "residual": round(residual, 6),
            "seed": int(seed),
            "tolerance": tolerance,
        },
    )


def _shuffle_columns(v, seed: int):
    """Every column independently permuted down the rows.

    THE CONTROL. This keeps each neuron's own activation distribution exactly —
    same values, same firing rate, same peak — and destroys which tokens they
    happened on, which is the only thing "co-firing" can mean. So a k-component
    fit to this is what the method achieves on data with no co-firing structure
    in it at all, and the real fit has to beat it to be evidence of anything.

    `nullmodel.py` makes the same argument for rankings and quotes the 2025
    result behind it: interpretability pipelines produce confident, ordered
    output on random inputs too. A factorisation is not exempt.
    """
    import torch

    generator = torch.Generator().manual_seed(int(seed))
    order = torch.argsort(
        torch.rand(v.shape, generator=generator, dtype=torch.float64), dim=0
    )
    return torch.gather(v, 0, order)


def _agreement(a, b) -> float:
    """Mean best-match cosine between two sets of component loadings.

    NMF's objective has no unique minimiser: two seeds can land on two
    different bases that fit equally well. This asks how much of the first
    fit's structure is present in the second, which is the honest way to say
    whether the components a reader is looking at are a property of the data
    or of an initialisation.
    """
    import torch

    left = torch.nn.functional.normalize(a, dim=1, eps=_EPS)
    right = torch.nn.functional.normalize(b, dim=1, eps=_EPS)
    return float((left @ right.T).max(dim=1).values.mean())


def _agreement_floor(n_components: int, n_cols: int, seed: int) -> float:
    """What `_agreement` reads for two loading sets with NO relationship.

    NOT ZERO, and that is the reason this function exists. `_agreement` is a
    mean over best-match cosines of NON-NEGATIVE vectors, and every
    non-negative vector lives in one orthant: two of them cannot be more than
    90 degrees apart, so their cosine cannot be negative and the best match
    over k candidates is biased upward. Measured on independent uniform
    non-negative matrices: 0.8429 at k=6, m=24 and 0.7793 at k=12, m=256.

    So "a low number means an artefact of an initialisation" needs a floor
    beside it or it is unreadable — a stability of 0.74 sounds like fair
    agreement and is in fact BELOW what two unrelated matrices of that shape
    score. This is the floor, measured on the fit's own shape, for the price
    of two `rand` calls and one matmul.
    """
    import torch

    generator = torch.Generator().manual_seed(int(seed))
    a = torch.rand(n_components, n_cols, generator=generator, dtype=torch.float64)
    b = torch.rand(n_components, n_cols, generator=generator, dtype=torch.float64)
    return _agreement(a, b)


# ------------------------------------------------------------- components


@dataclass
class Component:
    """One NMF component: these neurons, co-firing on these spans.

    `label` is `None` and stays `None`. A component is a co-firing group and
    nothing measured here says what it is about — the rule
    `feature_corpus.py:32-35` states for SAE features, which polysemanticity
    makes stricter rather than looser.
    """

    component: int
    neurons: list[dict]
    n_neurons_loaded: int
    spans: list[dict]
    n_rows_active: int
    n_rows: int
    mass_share: float
    label: None = None

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "neurons": self.neurons,
            "n_neurons_loaded": self.n_neurons_loaded,
            "n_neurons_shown": len(self.neurons),
            "spans": self.spans,
            "n_rows_active": self.n_rows_active,
            "n_rows": self.n_rows,
            "n_spans_shown": len(self.spans),
            "mass_share": round(self.mass_share, 5),
            "label": None,
        }


@dataclass
class Decomposition:
    """The factorisation, everything that bounds it, and its three resolutions."""

    components: list[Component]
    negatives: dict
    residual: float
    control_residual: float | None
    control_seed: int | None
    stability: float | None
    stability_seed: int | None
    iterations: int
    iterations_offered: int
    converged: bool
    seed: int
    n_rows: int
    n_cols: int
    n_tokens_seen: int
    n_neurons_selected: int
    n_neurons_offered: int
    max_neurons: int
    bytes_bound: int
    control_residuals: list[float] | None = None
    control_spread: float | None = None
    control_repeats: int = 0
    stability_floor: float | None = None
    n_rows_nonfinite: int = 0

    @property
    def control_margin(self) -> float | None:
        """How much lower the real fit's residual is than the control's."""
        if self.control_residual is None:
            return None
        if not (math.isfinite(self.residual) and math.isfinite(self.control_residual)):
            return None
        return self.control_residual - self.residual

    @property
    def control_verdict(self) -> str:
        """Which of the FOUR answers this is, named rather than encoded.

        `beats_control` is a tri-state and a boolean cannot carry four states,
        so the reason lives here and the boolean stays readable.
        """
        if self.control_residual is None:
            return "not measured"
        margin = self.control_margin
        if margin is None:
            return "not a number"
        floor = self.margin_floor
        if floor is None:
            return "no noise floor"
        if margin > floor:
            return "beaten"
        if margin < -floor:
            return "not beaten"
        return "inside the noise"

    @property
    def margin_floor(self) -> float | None:
        """How large a margin has to be before it is called either way."""
        if self.control_spread is None:
            return None
        return CONTROL_MARGIN_SPREADS * self.control_spread

    @property
    def control_margin_in_spreads(self) -> float | None:
        """The margin as a multiple of the control's own spread.

        The effect size, which is the number a reader actually needs: a margin
        of 0.001 means nothing without knowing the draws move by 0.001, and it
        means a great deal if they move by 0.0001.
        """
        margin = self.control_margin
        if margin is None or not self.control_spread:
            return None
        return margin / self.control_spread

    @property
    def beats_control(self) -> bool | None:
        """None when the answer is not known. Not False.

        Three ways it is not known, and none of them is "the control won":

          * no control was run — "we did not check" and "we checked and it
            failed" are different answers and only one is about the data;
          * a residual is NaN or infinite — `nan < x` is False, so a bare `<`
            turns arithmetic that never happened into a confident negative
            verdict, which is the single worst failure this file can have;
          * the margin is inside the measured control spread — on a 200 x 32
            matrix of INDEPENDENT columns the margin at seed 0 was +0.000975
            against a control spread of 0.001005, and `residual < control`
            published "it beats it, so there is co-firing structure here".

        `control_verdict` names which one.
        """
        verdict = self.control_verdict
        if verdict == "beaten":
            return True
        if verdict == "not beaten":
            return False
        return None

    def to_dict(self) -> dict:
        return {
            "components": [c.to_dict() for c in self.components],
            "n_components": len(self.components),
            "negatives": self.negatives,
            "residual": round(self.residual, 6),
            "control_residual": (
                None
                if self.control_residual is None
                else round(self.control_residual, 6)
            ),
            "control_seed": self.control_seed,
            "control_margin": (
                None if self.control_margin is None else round(self.control_margin, 6)
            ),
            # The whole null sample, its spread, and how many draws it is —
            # not just the mean. A reader cannot judge a margin without them.
            "control_residuals": (
                None
                if self.control_residuals is None
                else [round(r, 6) for r in self.control_residuals]
            ),
            "control_spread": (
                None if self.control_spread is None else round(self.control_spread, 6)
            ),
            "control_repeats": self.control_repeats,
            "control_margin_in_spreads": (
                None
                if self.control_margin_in_spreads is None
                else round(self.control_margin_in_spreads, 3)
            ),
            "control_margin_spreads_required": CONTROL_MARGIN_SPREADS,
            "beats_control": self.beats_control,
            "control_verdict": self.control_verdict,
            "stability": (None if self.stability is None else round(self.stability, 4)),
            "stability_seed": self.stability_seed,
            # What this statistic reads for loadings with no relationship at
            # all, on this fit's own shape. Without it `stability` has no
            # scale: the floor is around 0.78-0.86, not 0.
            "stability_floor": (
                None if self.stability_floor is None else round(self.stability_floor, 4)
            ),
            "stability_floor_basis": (
                "the same statistic between two INDEPENDENT uniform "
                "non-negative [k, n_cols] matrices — non-negative vectors "
                "share one orthant, so unrelated loadings still agree well "
                "above zero"
            ),
            # Rows the sweep sampled that held a NaN or an infinity and were
            # therefore not fitted. Reported, never dropped silently.
            "n_rows_nonfinite": self.n_rows_nonfinite,
            "iterations": self.iterations,
            "iterations_offered": self.iterations_offered,
            "converged": self.converged,
            "seed": self.seed,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "n_tokens_seen": self.n_tokens_seen,
            "n_neurons_selected": self.n_neurons_selected,
            "n_neurons_offered": self.n_neurons_offered,
            "max_neurons": self.max_neurons,
            "bytes_bound": self.bytes_bound,
            "bytes_basis": (
                "arithmetic bound on the arrays this fit allocates, not a "
                "measured peak — see budget.py for the difference"
            ),
            "polysemantic": POLYSEMANTIC,
            "means": self.means(),
        }

    def means(self) -> str:
        # `n_neurons_selected`, not `n_cols`: under the split policy the matrix
        # has two columns per neuron, and a sentence that read the column count
        # as a neuron count would report twelve neurons chosen out of six.
        text = (
            f"{len(self.components)} components over a {self.n_rows:,} x "
            f"{self.n_cols:,} matrix — {self.n_rows:,} tokens sampled from "
            f"{self.n_tokens_seen:,} read, and the {self.n_neurons_selected:,} "
            f"most active of {self.n_neurons_offered:,} neurons that were "
            f"non-zero at all. A COMPONENT IS 'THESE NEURONS CO-FIRE ON "
            f"THESE SPANS' AND NOTHING MORE. It is not a concept and it has no "
            f"name, because nothing here measured one. "
            f"The fit misses {self.residual:.1%} of the matrix's Frobenius "
            f"norm after {self.iterations} iterations"
            + (
                ", having converged."
                if self.converged
                else f", having run out at {self.iterations_offered}. It had "
                f"not stopped moving, so these components are where the "
                f"optimiser happened to be rather than where it settles."
            )
        )
        if self.n_rows_nonfinite:
            text += (
                f" {self.n_rows_nonfinite:,} sampled row"
                f"{'s' if self.n_rows_nonfinite != 1 else ''} held a NaN or an "
                f"infinity and {'were' if self.n_rows_nonfinite != 1 else 'was'}"
                f" excluded before the fit — not zeroed, excluded, and counted "
                f"here rather than absorbed into the row count above."
            )
        verdict = self.control_verdict
        control = (
            "" if self.control_residual is None else f"{self.control_residual:.1%}"
        )
        shuffle = (
            "the same fit on the same matrix with every column independently "
            "shuffled, which preserves each neuron's own distribution and "
            "destroys only which tokens it fired on"
        )
        if verdict == "not measured":
            text += (
                " NO CONTROL WAS RUN, so nothing here says whether a fit this "
                "good is remarkable. A k-component NMF reduces the residual of "
                "any matrix, including one with no co-firing structure in it."
            )
        elif verdict == "not a number":
            text += (
                " THE CONTROL COMPARISON DID NOT HAPPEN: one of the two "
                "residuals is NaN or infinite, so there is no comparison to "
                "report. That is the absence of a result and not a negative "
                "one — a NaN loses every comparison it is put into, and "
                "publishing the loss as a verdict would be the arithmetic "
                "speaking rather than the data."
            )
        elif verdict == "no noise floor":
            text += (
                f" The control — {shuffle} — was run ONCE and misses {control} "
                f"against the real {self.residual:.1%}, a margin of "
                f"{self.control_margin:.4f}. NO VERDICT IS TAKEN ON IT: one "
                f"draw from the null has no spread, so there is nothing to say "
                f"whether a margin that size is structure or shuffle luck. "
                f"Raise `control_repeats` above 1 to get an answer."
            )
        elif verdict == "beaten":
            text += (
                f" Against the control — {shuffle}, fitted "
                f"{self.control_repeats} times — the real fit misses "
                f"{self.residual:.1%} against the control's mean {control}: a "
                f"margin of {self.control_margin:.4f}, which is "
                f"{self.control_margin_in_spreads:.1f} times the "
                f"{self.control_spread:.4f} spread of the control's own draws. "
                f"That clears the {CONTROL_MARGIN_SPREADS:g} spreads this "
                f"module requires before calling it, so there is co-firing "
                f"structure here beyond what the method finds in anything."
            )
        elif verdict == "not beaten":
            text += (
                f" THE CONTROL WAS NOT BEATEN: {shuffle} misses {control} on "
                f"average over {self.control_repeats} draws against the real "
                f"{self.residual:.1%}, and the real fit is worse by "
                f"{abs(self.control_margin_in_spreads):.1f} times the "
                f"{self.control_spread:.4f} spread of those draws. Shuffling "
                f"destroys co-firing and preserves nothing else, so these "
                f"components are not evidence of neurons firing together. Read "
                f"them as an arbitrary grouping until a larger sample or a "
                f"different layer says otherwise."
            )
        else:
            text += (
                f" THE CONTROL IS NOT SETTLED EITHER WAY. {shuffle[0].upper()}"
                f"{shuffle[1:]}, fitted {self.control_repeats} times, misses "
                f"{control} on average against the real {self.residual:.1%} — a "
                f"margin of {self.control_margin:.4f}, which is only "
                f"{self.control_margin_in_spreads:.1f} times the "
                f"{self.control_spread:.4f} spread of the control's own draws "
                f"and does not reach the {CONTROL_MARGIN_SPREADS:g} spreads "
                f"required. The margin is inside the noise of the control "
                f"itself, so this run does not say whether there is co-firing "
                f"structure here. It is not a pass and it is not a failure; "
                f"more sampled rows or fewer components is what would settle "
                f"it."
            )
        if self.stability is None:
            text += (
                " Stability was not measured, so it is not known whether a "
                "different seed would produce these same components — NMF has "
                "no unique solution."
            )
        else:
            text += (
                f" A second fit from seed {self.stability_seed} agrees with "
                f"this one at a mean best-match cosine of {self.stability:.3f}."
            )
            if self.stability_floor is None:
                text += (
                    " NMF has no unique minimiser, so a low number there means "
                    "the components you are reading are an artefact of an "
                    "initialisation rather than a property of the activations."
                )
            else:
                text += (
                    f" READ THAT AGAINST {self.stability_floor:.3f}, NOT "
                    f"AGAINST ZERO: that is what this same statistic scores "
                    f"for two INDEPENDENT random non-negative loading sets of "
                    f"this fit's shape, because non-negative vectors all share "
                    f"one orthant and cannot disagree by more than 90 degrees. "
                    + (
                        "This fit is at or below that floor, so these "
                        "components are an artefact of an initialisation "
                        "rather than a property of the activations."
                        if self.stability <= self.stability_floor
                        else "This fit is above that floor, so the two seeds "
                        "found more in common than two unrelated fits of this "
                        "shape would."
                    )
                )
        return text + " " + POLYSEMANTIC


def decompose(
    sample: TokenSample,
    per_neuron: dict,
    *,
    n_components: int = DEFAULT_COMPONENTS,
    negatives: str = "clip",
    max_neurons: int = MAX_NMF_NEURONS,
    iterations: int = NMF_ITERATIONS,
    seed: int = 0,
    top_neurons: int = 8,
    top_spans: int = 5,
    control: bool = True,
    control_repeats: int = CONTROL_REPEATS,
    stability: bool = True,
) -> Decomposition:
    """Group neurons that fire together, over the sample `sweep` collected.

    Takes no model and runs no forward pass: everything it needs was captured
    on the sweep. That is deliberate — a caller can refit with a different
    component count, a different negative policy or a different seed for the
    cost of a small matrix multiply, and comparing two policies on the SAME
    activations is the only way the comparison means anything.

    `control_repeats` IS THE PRICE OF A VERDICT. One control fit gives one
    draw from the null and no error bar, and `residual < control_residual` on
    a single draw is a coin flip whenever the two are close — measured on a
    200 x 32 matrix of independent columns, a +0.000975 margin against a
    0.001005 spread published as "it beats it". Three draws cost two extra
    fits of an already-small matrix and buy a spread the verdict can be taken
    against. Set it to 1 to get the old single draw, and the response will say
    it has no noise floor.
    """
    import torch

    if not isinstance(sample, TokenSample):
        raise BadRequest(
            "decompose needs the TokenSample that `sweep` returned, which "
            "carries both the activations and where each row came from."
        )
    if isinstance(max_neurons, bool) or not isinstance(max_neurons, int):
        raise BadRequest(f"max_neurons has to be a whole number, got {max_neurons!r}.")
    if max_neurons < 1:
        raise BadRequest(f"max_neurons has to be at least 1, got {max_neurons}.")
    for name, value in (
        ("top_neurons", top_neurons),
        ("top_spans", top_spans),
        ("control_repeats", control_repeats),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise BadRequest(
                f"{name} must be a whole number of at least 1, got {value!r}."
            )
    if sample.n_rows == 0:
        raise Refusal(
            "the sweep sampled no tokens, so there is no matrix to factorise."
        )

    # Validated HERE and not only inside `non_negative`, because the selection
    # below branches on it. An unknown policy would otherwise fall through the
    # `clip` arm, choose columns on the clip rule, and only then be refused —
    # or worse, be refused after a fit had already been paid for.
    if negatives not in NEGATIVE_POLICIES:
        raise BadRequest(
            f"'{negatives}' is not a negative-handling policy. Known: "
            f"{', '.join(NEGATIVE_POLICIES)}."
        )

    # WHICH NEURONS ARE ELIGIBLE DEPENDS ON THE POLICY, which is not obvious
    # and was wrong the first time. Under `clip` a neuron that never went
    # positive contributes an all-zero column: it cannot co-fire with
    # anything, and including it only spends a column of the cap. Under
    # `split` that same neuron's negative lobe is a real, non-zero column with
    # real structure in it — selecting on `n_fired` alone would silently drop
    # exactly the neurons `split` exists to keep, and the response would still
    # claim nothing had been discarded.
    def activity(row: NeuronRow) -> int:
        return row.n_fired if negatives == "clip" else row.n_fired + row.n_negative

    # Ties broken by peak and then by index so the same sweep selects the same
    # columns every time. A ranking stable only up to Python's dict order is
    # not reproducible, and the whole point of reporting a seed is
    # reproducibility.
    # A neuron with no finite activation has no peak to break ties on, and
    # `activity` is 0 for it anyway — it cannot be selected, so it does not
    # reach the sort. Guarded explicitly rather than relied upon, because the
    # sort key would raise a TypeError on a `None` the day that changes.
    active = [
        row
        for row in per_neuron.values()
        if activity(row) > 0 and row.max_activation is not None
    ]
    if not active:
        raise Refusal(
            "no neuron in this layer went positive on the corpus that was "
            "read, so there is nothing that could co-fire. That is a fact "
            "about this text as much as about the layer."
            if negatives == "clip"
            else "every neuron in this layer was exactly zero on every token "
            "that was read, so there is nothing to factorise. That is a fact "
            "about this text as much as about the layer."
        )
    active.sort(key=lambda r: (-activity(r), -r.max_activation, r.neuron))
    chosen = [row.neuron for row in active[:max_neurons]]

    columns = torch.tensor(chosen, dtype=torch.long)
    raw = sample.values.index_select(1, columns)

    # ROWS WITH A NaN OR AN INFINITY IN THEM ARE EXCLUDED, NOT ZEROED. Zeroing
    # would put a token this model produced nothing readable on into the fit
    # as a token where every selected neuron was silent, which is a
    # measurement nobody took. The count leaves here in `n_rows_nonfinite` and
    # is published beside the row count.
    readable = torch.isfinite(raw).all(dim=1)
    n_rows_nonfinite = int(raw.shape[0]) - int(readable.sum())
    if n_rows_nonfinite:
        raw = raw[readable]
        if int(raw.shape[0]) == 0:
            raise Refusal(
                f"every one of the {sample.n_rows:,} sampled rows held a NaN "
                f"or an infinity in at least one of the {len(chosen):,} "
                f"selected neurons, so there is no matrix left to factorise. "
                f"That is this model's arithmetic overflowing rather than a "
                f"fact about the corpus — an fp16 MLP produces it — and the "
                f"next step is the same layer in float32."
            )
    fitted_rows = int(raw.shape[0])
    row_index = (
        list(range(sample.n_rows))
        if not n_rows_nonfinite
        else readable.nonzero(as_tuple=True)[0].tolist()
    )

    matrix, negative_report = non_negative(raw, negatives)

    w, h, info = nmf(
        matrix, n_components, iterations=iterations, seed=seed, tolerance=NMF_TOLERANCE
    )

    control_residual = None
    control_residuals = None
    control_spread = None
    control_seed = None
    n_control = 0
    if control:
        control_seed = int(seed) + 977  # a fixed offset, so the control is re-runnable
        n_control = int(control_repeats)
        # A DIFFERENT SHUFFLE EACH TIME, the same NMF initialisation every
        # time. What is being estimated is the spread of the NULL, so the
        # thing that varies has to be the draw from it and not the optimiser's
        # starting point.
        control_residuals = [
            float(
                nmf(
                    _shuffle_columns(matrix, control_seed + offset),
                    n_components,
                    iterations=iterations,
                    seed=seed,
                    tolerance=NMF_TOLERANCE,
                )[2]["residual"]
            )
            for offset in range(n_control)
        ]
        control_residual = sum(control_residuals) / len(control_residuals)
        # Sample standard deviation, and None at one draw rather than 0.0 — a
        # single number has no spread, and reporting 0.0 would turn "we have
        # no noise floor" into "the noise floor is zero", which is the
        # unfloored `<` this replaced.
        control_spread = (
            statistics.stdev(control_residuals) if len(control_residuals) > 1 else None
        )

    agreement = None
    stability_seed = None
    stability_floor = None
    if stability:
        stability_seed = int(seed) + 1
        _, other, _ = nmf(
            matrix,
            n_components,
            iterations=iterations,
            seed=stability_seed,
            tolerance=NMF_TOLERANCE,
        )
        agreement = _agreement(h, other)
        stability_floor = _agreement_floor(
            int(h.shape[0]), int(h.shape[1]), int(seed) + 7919
        )

    n_cols = int(matrix.shape[1])
    lobes = ["positive"] * n_cols
    owner = list(chosen)
    if negatives == "split":
        lobes = ["positive"] * len(chosen) + ["negative"] * len(chosen)
        owner = list(chosen) + list(chosen)

    contributions = [
        float(torch.linalg.norm(w[:, c]) * torch.linalg.norm(h[c]))
        for c in range(int(h.shape[0]))
    ]
    total = sum(contributions) or 1.0

    components: list[Component] = []
    for index in range(int(h.shape[0])):
        loadings = h[index]
        order = torch.argsort(loadings, descending=True).tolist()
        shown = [i for i in order if float(loadings[i]) > 0][:top_neurons]
        weights = w[:, index]
        rows = torch.argsort(weights, descending=True).tolist()
        span_rows = [r for r in rows if float(weights[r]) > 0][:top_spans]
        components.append(
            Component(
                component=index,
                neurons=[
                    {
                        "neuron": owner[i],
                        "loading": round(float(loadings[i]), 5),
                        "lobe": lobes[i],
                    }
                    for i in shown
                ],
                # The true count beside the capped list, always.
                n_neurons_loaded=int((loadings > 0).sum()),
                spans=[
                    {
                        # `row_index` maps a row of the FITTED matrix back to
                        # the row of the sample it came from. They are the same
                        # list until a non-finite row is excluded, and after
                        # that indexing the sample directly would point every
                        # span at the wrong token.
                        "text": sample.contexts[row_index[r]],
                        "token": sample.tokens[row_index[r]],
                        "offset": sample.offsets[row_index[r]],
                        "sequence": sample.sequences[row_index[r]],
                        "position": sample.positions[row_index[r]],
                        # NOT called `activation`. This is the component's
                        # weight on a sampled row, which is a different
                        # quantity from any one neuron's activation there, and
                        # a shared key name would invite a reader to compare
                        # them.
                        "weight": round(float(weights[r]), 5),
                    }
                    for r in span_rows
                ],
                n_rows_active=int((weights > 0).sum()),
                n_rows=fitted_rows,
                mass_share=contributions[index] / total,
            )
        )

    return Decomposition(
        components=components,
        negatives=negative_report,
        residual=float(info["residual"]),
        control_residual=control_residual,
        control_residuals=control_residuals,
        control_spread=control_spread,
        control_repeats=n_control,
        control_seed=control_seed,
        stability=agreement,
        stability_seed=stability_seed,
        stability_floor=stability_floor,
        iterations=int(info["iterations"]),
        iterations_offered=int(info["iterations_offered"]),
        converged=bool(info["converged"]),
        seed=int(seed),
        n_rows=fitted_rows,
        n_rows_nonfinite=n_rows_nonfinite,
        n_cols=n_cols,
        n_tokens_seen=sample.n_tokens_seen,
        n_neurons_selected=len(chosen),
        n_neurons_offered=len(active),
        max_neurons=int(max_neurons),
        bytes_bound=_fit_bytes(
            fitted_rows, n_cols, n_components, control=control, stability=stability
        ),
    )


# ------------------------------------------------------- cost before spending


def _fit_byte_parts(
    n_rows: int, n_cols: int, n_components: int, *, control: bool, stability: bool
) -> dict:
    """The bound above, itemised, so each term can be checked on its own.

    Itemised because a total is not checkable. The control term was wrong by
    3x — it budgeted ONE `[n, m]` array for a helper that allocates three —
    and no test could see it, because the total was still large enough to
    swallow the error next to the fit terms. A named term can be measured
    against the thing it claims to cover, and
    `tests/test_neurons.py::test_the_byte_bound_covers_what_the_control_shuffle_allocates`
    does exactly that.
    """
    cell = 8
    v = n_rows * n_cols
    w = n_rows * n_components
    h = n_components * n_cols
    # V, W, H, the two [k, m] and two [n, k] update temporaries, and the
    # [n, m] difference the residual is taken over.
    one_fit = (2 * v) + (2 * w) + (2 * h)
    parts = {"fit": one_fit * cell}
    if control:
        # THREE [n, m] ARRAYS, not one. `_shuffle_columns` allocates the
        # random sort keys (float64), their argsort index (int64) and the
        # gathered copy, and all three are live at the moment `gather`
        # returns. Measured on a 4096 x 1024 float64 matrix: the helper's peak
        # RSS growth was 108.9 MB against the 33.6 MB one array would be.
        parts["control_shuffle"] = 3 * v * cell
        parts["control_fit"] = one_fit * cell
    if stability:
        parts["stability"] = (w + h) * cell
    return parts


def _fit_bytes(
    n_rows: int, n_cols: int, n_components: int, *, control: bool, stability: bool
) -> int:
    """An arithmetic bound on the float64 arrays one `decompose` allocates.

    A BOUND, NOT A PEAK. `budget.py` insists on the distinction and it holds
    here: this counts the arrays this file asks for, and says nothing about
    allocator churn or the temporaries torch builds inside a matmul. Compare it
    against a budget; do not promise it to anybody.

    IT IS ALSO A BOUND THAT WAS MEASURED, because a bound nobody sampled is a
    sentence and not a bound. Cold peak-RSS growth over a full `decompose`,
    sampled every 2 ms:

      2048 x 512, k=12   bound 59.9 MB   measured peak 44.7 MB
      4096 x 512, k=12   bound 119.7 MB  measured peak 75.7 MB

    Only cold runs count: torch's allocator keeps freed blocks, so a second
    run of the same shape peaks far lower and would flatter the bound.

    `control_repeats` does not enter it. The control fits run one after
    another and each one's arrays are freed before the next is asked for, so
    what they cost is one control fit however many times it is repeated —
    which is why the repeats buy a noise floor in time rather than in memory.
    """
    return int(
        sum(
            _fit_byte_parts(
                n_rows, n_cols, n_components, control=control, stability=stability
            ).values()
        )
    )


def cost(
    texts: list[str],
    d_mlp: int,
    *,
    sample_rows: int = SAMPLE_ROWS,
    max_neurons: int = MAX_NMF_NEURONS,
    n_components: int = DEFAULT_COMPONENTS,
    negatives: str = "clip",
    control: bool = True,
    stability: bool = True,
    with_evidence: bool = False,
) -> dict:
    """What this will cost, in forward passes and in bytes, before it is spent.

    The companion `budget.py` asks every expensive thing in this package to
    have. It takes no model and runs nothing: the pass count is exact
    arithmetic over the corpus, and the byte figures are arithmetic bounds on
    the arrays this module allocates.

    `passes` is an UPPER BOUND rather than the exact figure, and the response
    says which it is. The sweep stops at `MAX_TOKENS`, and how many sequences
    that is depends on how many tokens each one has — which is not knowable
    without tokenising the corpus, i.e. without doing most of the work this is
    supposed to price first.
    """
    if not isinstance(texts, list):
        raise BadRequest("cost prices a corpus, so `texts` has to be a list.")
    if isinstance(d_mlp, bool) or not isinstance(d_mlp, int) or d_mlp < 1:
        raise BadRequest(
            f"d_mlp has to be a whole number of at least 1, got {d_mlp!r}. "
            f"`neuron_count(block)` reads it off the projection."
        )
    if negatives not in NEGATIVE_POLICIES:
        raise BadRequest(
            f"'{negatives}' is not a negative-handling policy. Known: "
            f"{', '.join(NEGATIVE_POLICIES)}."
        )

    passes = len(texts) * (2 if with_evidence else 1)
    live_bytes = MAX_SEQUENCE_TOKENS * d_mlp * 4
    reservoir_bytes = sample_rows * d_mlp * 4
    n_cols = min(max_neurons, d_mlp) * (2 if negatives == "split" else 1)
    fit_bytes = _fit_bytes(
        sample_rows, n_cols, n_components, control=control, stability=stability
    )
    # COMPUTED FROM THE ARGUMENTS, like every other figure in this response.
    # It used to be the string "6.5 GB", copied out of the module docstring's
    # 8192-wide example, and it was printed as this corpus's number whatever
    # `d_mlp` was — `cost(['a'], d_mlp=1)` reported 6.5 GB for a single
    # sequence over a single neuron. An upper bound for the same reason
    # `passes` is: the true token count is not knowable without tokenising,
    # and the sweep will not read past MAX_TOKENS of it either way.
    whole_corpus_bytes = MAX_TOKENS * d_mlp * 4

    return {
        "passes": passes,
        "passes_are_an_upper_bound": True,
        "sweep_passes": len(texts),
        "evidence_passes": len(texts) if with_evidence else 0,
        "live_sequence_bytes": live_bytes,
        "reservoir_bytes": reservoir_bytes,
        "fit_bytes": fit_bytes,
        "fit_bytes_parts": _fit_byte_parts(
            sample_rows, n_cols, n_components, control=control, stability=stability
        ),
        "bytes_bound": live_bytes + reservoir_bytes + fit_bytes,
        "whole_corpus_bytes": whole_corpus_bytes,
        "whole_corpus_bytes_is_an_upper_bound": True,
        "bytes_basis": (
            "arithmetic bound on the arrays this module allocates, not a "
            "measured peak — see budget.py, whose `probe_pass` is the thing "
            "that measures one"
        ),
        "matrix_shape": [sample_rows, n_cols],
        "means": (
            f"At most {passes:,} forward passes — one per sequence for the "
            f"sweep"
            + (
                ", and one more each if per-neuron spans are asked for"
                if with_evidence
                else ""
            )
            + f". An upper bound rather than the figure: the sweep stops at "
            f"{MAX_TOKENS:,} tokens, and which sequence that lands in depends "
            f"on token counts nobody has yet. Memory is bounded by the longest "
            f"single sequence held at once — {MAX_SEQUENCE_TOKENS:,} tokens x "
            f"{d_mlp:,} neurons x 4 bytes = {live_bytes / 1e6:.0f} MB — plus "
            f"{reservoir_bytes / 1e6:.0f} MB for the {sample_rows:,}-row "
            f"reservoir and {fit_bytes / 1e6:.0f} MB for the fit. Holding the "
            f"corpus in activation form instead of streaming it would be up to "
            f"{MAX_TOKENS:,} tokens x {d_mlp:,} neurons x 4 bytes = "
            f"{whole_corpus_bytes / 1e9:.2f} GB for these {len(texts):,} "
            f"sequences — computed from this call's width and this module's "
            f"token cap, and an upper bound for the same reason the pass count "
            f"is. That is the number this module streams to avoid."
        ),
    }
