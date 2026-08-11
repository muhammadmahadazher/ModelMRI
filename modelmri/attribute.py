"""Which of the words you typed actually moved the answer?

The attention panel shows what the model looked *at*. That is a different
question from what the answer depended *on* — a token can be attended to
heavily and matter not at all. This one masks a single token out of every
later position's attention, runs the model again, and measures how far the
next-token distribution at `position` moved. It imports `kl_nats` from
ablate.py rather than defining a second one, so a head score and a token
score on the same screen mean the same thing.

Every number below was measured on this machine before this file existed:
prompt "The capital of France is", bfloat16 on cuda, `attn_implementation=
"eager"`, one unbatched sequence, attributing at the last prompt token. A KL
without those conditions attached cannot be checked by anyone.

**position_ids go in explicitly, on every pass.** Several HF decoders derive
them from `attention_mask.cumsum(-1) - 1`. Zeroing an interior mask entry
then shifts the RoPE phase of every token after it, and that shift — the
whole suffix arriving at different angles — gets billed to the one token we
masked. The score would be real and would not be attribution. So every pass
here, base and floor and each masked one, is handed the same
`arange(S)` tensor, and the run refuses if it is ever rebuilt.

Handing them in is not the same as being *read*, and one pass exists to tell
the two apart. An all-ones mask cannot: under it `cumsum(-1) - 1` equals
`arange(S)` by construction, so a model that ignores position_ids entirely
and derives them from the mask agrees with a plain forward pass exactly, and
the agreement check waves it through. Written as a toy model — logits a pure
function of `attention_mask.cumsum(-1) - 1`, input_ids never read — it used
to come back with floor 0.0, a verified mask and a full ranking in which
every score was the suffix's position shift. So the run also feeds one pass a
deliberately WRONG ordering and requires the answer to move. Measured with
`position_ids.flip(-1)` at the last prompt token: gpt2 2.166768, Qwen3-0.6B
0.011300, gemma-3-270m-it 4.616208 nats — the smallest of those is 11,300x
the tolerance. A uniform shift would not do: `position_ids + 1` moves gpt2
3.396605 (learned absolute embeddings) but Qwen3 only 5e-06 and gemma
0.000107, because RoPE is invariant to shifting every position together, and
5e-06 is inside the range Qwen3's own content tokens score in.

**Masking is the only baseline.** The obvious alternative, substituting a
neutral token, is three different experiments wearing one name: gpt2 has no
pad and unk == bos == eos, Qwen3 has a pad and no unk or bos, gemma has a
real <pad>. On gpt2 the substitute is <|endoftext|>, and inserting a
document boundary reverses the ranking end to end — the model is reacting to
being told the document ended, not to losing a word. Masking preserves the
sequence length and every index, so ids.shape is asserted unchanged.

**Index 0 is a sink and never enters the ranked list.** On gpt2 it scores
4.86309, 2.79x the next candidate in bf16 — 4.92884 and 3.23x in fp32, which
is the pair tests/test_attribute.py re-runs, and the two must not be quoted
across each other. Prepending <|endoftext|> moves the
word "The" to index 1 and it falls to 0.46107 — 10.5x — while the score at
index 0 stays at 4.76083, 2.1% from where it was with a completely different
token sitting there. The score follows the position, not the token. (Checked
against the obvious artifact: with a 2D mask, masking key 0 leaves query 0
with no keys at all. Re-run with a 4D mask that spares the diagonal, every
off-diagonal score reproduces bit-for-bit.) It is reported on its own row,
labelled, outside the order.

**The attribution position itself is excluded, on geometry not on size.** It
is the only candidate whose own key is being taken from its own query, and
sparing the diagonal drops its score to exactly 0.0 while leaving every other
score untouched — so what it measures is the mask's shape, not the token. The
"it is tiny anyway" version of this argument is false and must not be
substituted: it is 0.06375 against a max of 4.86309 on gpt2, but on
Qwen3-0.6B the same position scores 6.24429 and is the LARGEST of all 13
candidates, and on gemma-3-270m-it 1.92183 against a max of 9.33529. Without
the rule it tops the Qwen3 ranking on an artifact.

**These do not add up, and the direction is not even fixed.** Over the
tokens this list actually shows — everything before the position except
index 0 — the singles OVER-count: gpt2 3.51088 summed against 1.93419 for
one joint mask of those same three tokens (1.82x), gemma-3-270m-it 34.70212
against 22.02131 (1.58x). Sum a different set and the same passes go the
other way: over the typed span, gemma is 4.62745 against 13.29702 (0.35x),
and gpt2 — whose typed span is the whole prompt, index 0 included — is
8.37397 against 8.53054 (0.98x). Same model, same prompt, same forward
passes, opposite sign. So `sum_of_singles` and `joint_kl` are both returned,
the panel may say the two disagree, and nothing here is a correction factor.
(Head ablation in ablate.py over-counts 8x on gpt2 and under-counts on
gemma. Masking a token and zeroing a head are not the same phenomenon; that
docstring's framing must not be copied onto these numbers.)

**Two things the caller has to get right, because this file cannot.** Pass
`typed_span` whenever a chat template was applied. On Qwen3 the top three
scores are the template's own '\\n' (6.24429, excluded as the self position),
'assistant' (2.02161) and '<|im_start|>' (0.32266), while every word the user
typed sits between 3.1e-05 and 7.9e-05 — four to five orders down. A list
that does not separate them answers "the chat template" every time. And on
that model the entire content signal is 2.50e-04 nats, while merely moving
gpt2 from fp32 to bf16 shifts a distribution by 1.88e-02: below the model's
own arithmetic, the ranking is real in the sense that it reproduces, and is
not evidence that one word mattered more than another.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from typing import Any

import torch

from modelmri.ablate import distribution, kl_nats
from modelmri.errors import BadRequest

# Masking, and nothing else. Named anyway, because ablate.py's lesson is that
# an unlabelled importance number is the lie, and the caller has to be able to
# put the word on screen.
BASELINES = ("mask",)

# Nearest `position`, on the argument that a 400-token prompt does not deserve
# 400 forward passes and recency is the only ordering available before the
# measurement is taken. Truncation is reported, never silent.
MAX_CANDIDATES = 64

# How far the explicit-mask base may sit from a plain model(ids) before we stop
# and say the model's position semantics are not what this file assumes.
# Measured: exactly 0.0 on gpt2, Qwen3-0.6B and gemma-3-270m-it — the logits
# are bit-identical (torch.equal), the explicit arguments do not select a
# different kernel. The tolerance is slack for accelerators that are not
# bit-reproducible, and sits 31x below the smallest per-token score seen on any
# of the three (Qwen3's 3.1e-05).
AGREEMENT_TOLERANCE = 1e-6

# Only the pipe form. The wider `^<\|?.+\|?>$` looks equivalent and is not: on
# gemma-3-270m-it it claims 6573 ids beyond the 8 declared specials, including
# <div>, <b>, <html>, <table> and <li>, which are ordinary content in that
# vocabulary. Greying those out would hide the words an answer depends on.
_PIPE_CONTROL = re.compile(r"^<\|.+\|>$")


class AttributionError(RuntimeError):
    """We cannot take this measurement, and we say why rather than guess."""


def control_token_ids(tokenizer: Any) -> set[int]:
    """Ids of the tokens that are scaffolding rather than language.

    Four arms, each earning its place on a measurement: the declared specials;
    added tokens flagged special (this is the only arm that finds gemma's
    <start_of_turn> 105 and <end_of_turn> 106); the `<|...|>` form; and added
    tokens that the tokenizer's own chat template mentions by name, which is
    the only arm that finds Qwen3's <think> 151667 — the token that model
    emits with p=0.999531 at the end of a templated prompt, and which is
    declared special nowhere. Measured totals: gpt2 1, Qwen3-0.6B 26,
    gemma-3-270m-it 10.

    Compare ids taken from the sequence, never strings you hope tokenize to
    them. `convert_tokens_to_ids(" the")` returns 50256 on gpt2 — the unk
    fallback, which is <|endoftext|>, which is in this set — because GPT-2
    spells that token with U+0120 and not a space. That false positive greys
    out the model's own answer.
    """
    ids: set[int] = {
        int(i) for i in (getattr(tokenizer, "all_special_ids", None) or [])
    }
    try:
        extra = tokenizer.additional_special_tokens or []
    except AttributeError:
        # The slow GPT2Tokenizer overrides __getattr__ and raises for this
        # name; getattr-with-a-default swallows it, a bare access does not.
        extra = []
    for token in extra:
        got = tokenizer.convert_tokens_to_ids(token)
        if got is not None and int(got) >= 0:
            ids.add(int(got))

    added = getattr(tokenizer, "added_tokens_decoder", None) or {}
    template = getattr(tokenizer, "chat_template", None) or ""
    if not isinstance(template, str):  # some tokenizers keep a dict of templates
        template = str(template)
    for i, entry in added.items():
        text = str(entry)
        if getattr(entry, "special", False):
            ids.add(int(i))
        # Delimiter-shaped AND named by the template. Both halves are load
        # bearing: gemma's added vocabulary is 6415 entries deep and most of
        # it is HTML, so "looks like a tag" alone is the wide regex again.
        elif text.startswith("<") and text.endswith(">") and text in template:
            ids.add(int(i))

    for token, i in (getattr(tokenizer, "get_vocab", dict)() or {}).items():
        if _PIPE_CONTROL.match(token):
            ids.add(int(i))
    return ids


def rank_tokens(
    model: Any,
    ids: torch.Tensor,
    *,
    position: int,
    typed_span: tuple[int, int] | None = None,
    n_prompt: int | None = None,
    control_ids: Iterable[int] = (),
    max_candidates: int = MAX_CANDIDATES,
    baseline: str = "mask",
    decode=None,
) -> dict:
    """Rank the tokens before `position` by how far masking one moves the answer.

    `ids` is one unbatched sequence, `[1, S]`. `control_ids` comes from
    `control_token_ids(tokenizer)`. `decode` turns an id into a string for the
    readout.

    `typed_span` is the half-open token span of what the user actually typed;
    pass it whenever a chat template was applied, or the template's own '\\n'
    and 'assistant' will be labelled as the user's words — on Qwen3 those
    outscore every typed token by four orders of magnitude. `None` is NOT
    "all of it is yours": it is the state "the caller could not locate your
    words", which is what `runtime._user_span` returns when the tokenizer is
    slow or the prompt occurs twice in the templated text. Those rows come
    back as `group: "unknown"`, and a caller that renders them under a
    heading reading "what you typed" is making a claim nobody measured.

    `n_prompt` is where the prompt ends and the model's own output begins.
    Without it every token past the prompt falls outside `typed_span` and is
    labelled `"template"` — so attributing at a generated token files the
    model's own words under the chat template, on gpt2, which has no chat
    template at all. Measured there (bf16/cuda, "The capital of France is",
    12 greedy tokens, attributing at index 16): 11 of the 15 ranked rows sit
    past the prompt, and the largest score in the whole run is the model's own
    ' Republic'. Groups are therefore four-valued — `generated` wins over the
    rest, because past the prompt nothing is template and nothing is typed.
    """
    # THE FOUR ARGUMENT CHECKS, AND WHY TWO OF THEM ARE NOT AttributionError.
    #
    # AttributionError means "this measurement cannot be taken honestly" and
    # runtime.py converts it to a Refusal — 409, "ModelMRI decided not to
    # answer". That is right for the four checks further down, which are about
    # what the model did. It is wrong for a caller who passed a bad argument,
    # and `baseline` and `position` are both query parameters on
    # /api/attention/attribute. errors.py names an unknown baseline as the type
    # example of a BadRequest; it was answering 409 while the layer check on
    # the sibling endpoint answered 422.
    #
    # The other two stay AttributionError-shaped but are NOT reachable from a
    # request: runtime.py builds `ids` itself out of `last_ids`, and
    # `max_candidates` comes from a module constant. A violation there is this
    # package contradicting itself, so it is a plain RuntimeError and belongs
    # on the 500 path with a traceback — see the note at `used_position_ids`.
    if baseline not in BASELINES:
        raise BadRequest(
            f"unknown baseline {baseline!r} — this measurement offers only "
            f"{', '.join(BASELINES)}. Substituting a neutral token instead of "
            "masking is three different experiments across gpt2, Qwen3 and "
            "gemma, and on gpt2 it reverses the ranking."
        )
    if ids.dim() != 2 or int(ids.shape[0]) != 1:
        raise RuntimeError(
            f"attribution needs one unbatched sequence shaped [1, S], got "
            f"{tuple(ids.shape)}. Batching changes the kernel and the noise "
            "floor measured for this path was measured unbatched."
        )
    seq = int(ids.shape[1])
    if not 0 <= position < seq:
        raise BadRequest(f"position {position} is outside a sequence of {seq} tokens.")
    if max_candidates < 1:
        raise RuntimeError("max_candidates must be at least 1.")

    control = {int(c) for c in control_ids}
    shape = tuple(ids.shape)
    started = time.perf_counter()

    # One tensor, built once, handed to every pass. See the module docstring:
    # rebuilding it per pass is how the suffix's RoPE phase silently becomes
    # part of the score.
    position_ids = torch.arange(seq, device=ids.device).unsqueeze(0)
    ones = torch.ones((1, seq), dtype=torch.long, device=ids.device)
    used_position_ids: list[torch.Tensor] = []
    passes = 0

    def masked(*off: int) -> torch.Tensor:
        """attention_mask with those key positions switched off."""
        mask = ones.clone()
        for j in off:
            mask[0, j] = 0
        return mask

    def forward(mask: torch.Tensor, **extra: Any):
        nonlocal passes
        kwargs = {
            "input_ids": ids,
            "attention_mask": mask,
            "position_ids": position_ids,
            **extra,
        }
        used_position_ids.append(kwargs["position_ids"])
        passes += 1
        out = model(**kwargs)
        # Masking must not resize anything. A baseline that dropped the token
        # instead would renumber every position after it, and the ranking
        # would be about the renumbering.
        if tuple(ids.shape) != shape:
            # A plain RuntimeError, deliberately, and one of two in this file.
            # This is not a statement about the model or the request — it is
            # this function checking its own bookkeeping, the "this should not
            # happen" class, and a violation is a bug here. As an
            # AttributionError it went through runtime.py's blanket wrap and
            # reached the browser as 409 "ModelMRI decided not to answer",
            # with no traceback logged anywhere. It is a 500 now, with one.
            raise RuntimeError(
                f"the input ids changed shape mid-run, {shape} -> "
                f"{tuple(ids.shape)}; masking must preserve every index."
            )
        return out

    def logits_at(mask: torch.Tensor, **extra: Any) -> torch.Tensor:
        return forward(mask, **extra).logits[0, position]

    # Deliberately wrong, and reversed rather than shifted. RoPE is invariant
    # to moving every position by the same amount, so `arange + 1` is not a
    # probe at all on a RoPE model: it moves Qwen3-0.6B by 5e-06 nats and
    # gemma-3-270m-it by 0.000107, against gpt2's 3.396605, which has learned
    # absolute embeddings. Reversal changes the relative offsets and moves all
    # three. Values stay inside [0, S), so nothing here can index out of a
    # learned position table.
    wrong_position_ids = position_ids.flip(-1)

    with torch.no_grad():
        # Base and floor both run WITH the all-ones mask, so base and ablated
        # share a kernel path and the difference between them cannot be the
        # arguments rather than the model.
        base = distribution(logits_at(ones))
        floor = kl_nats(base, distribution(logits_at(ones)))

        # The one pass that is not part of the ranking: plain model(ids), the
        # call every other reader of this model makes. If supplying explicit
        # position_ids and an all-ones mask moves the answer at all, this
        # model does not derive positions the way this file assumes and every
        # score below would be measuring the arguments.
        passes += 1
        plain = distribution(model(ids).logits[0, position])
        agreement = kl_nats(plain, base)
        if agreement > max(floor, AGREEMENT_TOLERANCE):
            raise AttributionError(
                f"an all-ones attention mask with explicit position_ids moves "
                f"this model's answer by {agreement:.3e} nats against a plain "
                f"forward pass (floor {floor:.3e}). Its position semantics are "
                "not the ones token attribution assumes, so every score would "
                "include that shift. It would work on a model that takes "
                "position_ids at face value."
            )

        # The check above cannot see the failure this file exists to prevent.
        # Under an all-ones mask `attention_mask.cumsum(-1) - 1` IS arange(S),
        # so a model that throws our position_ids away and derives its own
        # from the mask passes it perfectly — and then every masked pass below
        # re-phases the whole suffix and bills it to the one masked token. So
        # ask the only question that separates them: hand the model an
        # ordering that is wrong on purpose and require the answer to move.
        # Measured with flip(-1) at the last prompt token: gpt2 2.166768,
        # Qwen3-0.6B 0.011300, gemma-3-270m-it 4.616208 nats.
        moved = kl_nats(
            base, distribution(logits_at(ones, position_ids=wrong_position_ids))
        )
        if moved <= max(floor, AGREEMENT_TOLERANCE):
            raise AttributionError(
                f"reversing this model's position_ids moves its answer by "
                f"{moved:.3e} nats (floor {floor:.3e}), so it is not reading "
                "them. Every score here would then be measuring what masking "
                "a token did to the POSITIONS of everything after it — HF "
                "decoders fall back to attention_mask.cumsum(-1) - 1, which "
                "renumbers the whole suffix — rather than what losing that "
                "token did to the answer. It would work on a model that takes "
                "the position_ids it is handed at face value."
            )

        top_id = int(base.argmax())
        if top_id in control:
            raise AttributionError(
                f"the token being attributed is a control token "
                f"({decode(top_id) if decode else top_id!r}), which the chat "
                "template all but guarantees — Qwen3 emits <think> here with "
                "p=0.999531. Asking which of your words caused it would rank "
                "noise. Generate at least one token and attribute at a "
                "position where the model is answering."
            )

        # i > position contributes exactly zero under the causal mask and is
        # not a result. i == position and i == 0 are excluded by rule; both get
        # their reasons in the module docstring, and index 0 is still measured
        # because "the sink dominates" is a claim we should keep re-checking.
        candidates = list(range(1, position))
        if not candidates:
            nothing = (
                "is the first token and has read nothing at all"
                if position == 0
                else "has read nothing but index 0, which is an attention sink "
                "rather than content"
            )
            raise AttributionError(
                f"position {position} {nothing}. Attribute at a position that "
                "has read at least one token that is neither the sink nor "
                "itself."
            )
        tested = candidates[-max_candidates:]

        def group_of(index: int) -> str:
            # Order matters. Past the prompt there is no template and no user
            # text left to be inside of, so `generated` is checked first; and
            # an absent span is its own answer rather than a permissive
            # default, because "we could not find your words" and "all of them
            # are yours" are opposite claims and used to share one value.
            if n_prompt is not None and index >= n_prompt:
                return "generated"
            if typed_span is None:
                return "unknown"
            if typed_span[0] <= index < typed_span[1]:
                return "typed"
            return "template"

        def row(index: int, after: torch.Tensor) -> dict:
            return {
                "index": index,
                "token": decode(int(ids[0, index]))
                if decode
                else str(int(ids[0, index])),
                "kl": round(kl_nats(base, after), 5),
                "p_top_before": round(float(base[top_id]), 5),
                "p_top_after": round(float(after[top_id]), 5),
                "flips_top": int(after.argmax()) != top_id,
                "group": group_of(index),
            }

        ranked = [row(i, distribution(logits_at(masked(i)))) for i in tested]
        index0 = row(0, distribution(logits_at(masked(0))))
        index0["note"] = (
            "An attention sink, not content — kept out of the order because "
            "the score belongs to the position rather than to the token "
            "sitting in it. Measured on gpt2 (bf16/cuda, 'The capital of "
            "France is', last prompt token): 4.86309 here, and prepending "
            "<|endoftext|> holds index 0 at 4.76083 while the word that moved "
            "off it to index 1 falls to 0.46107."
        )

        # Sum against joint over the SAME set: the tested rows only. Index 0
        # stays out of both, so the comparison is between numbers the panel
        # actually shows.
        joint = distribution(logits_at(masked(*tested)))

        # ONE verification of the MECHANISM, not of each row: does
        # attention_mask[0, j] = 0 actually empty column j at every layer?
        # This pass runs eager attention to materialise the weights and is
        # therefore NOT the pass that produced any KL above.
        #
        # It does NOT answer for every index, and the claim that it did used
        # to sit here. Index 0 is the exception and it is the row this panel
        # highlights: masking key 0 leaves query 0 with no keys at all, and
        # gpt2's eager path then hands that row a uniform distribution over
        # every position, so column 0 still carries 0.200195 of the weight
        # (Qwen3-0.6B through its chat template, S=13: 0.077148). Every other
        # index measured on either model came back exactly 0. The
        # score survives it — index 0 comes back 4.863085746765137 under this
        # 2D mask and 4.863085746765137 under a diagonal-sparing 4D one that
        # leaves query 0 its own key — so this is an over-broad verification
        # claim rather than a wrong number, which is why the probe is the
        # index nearest `position` and why `mask_check` says at which index it
        # ran.
        probe = tested[-1]
        why = "this model returned no attention weights"
        try:
            attentions = getattr(
                forward(masked(probe), output_attentions=True), "attentions", None
            )
        except Exception as err:  # a failed check is a reported check
            attentions, why = None, f"{type(err).__name__}: {err}"
        if attentions:
            residual: float | None = max(
                float(a[0, :, :, probe].max()) for a in attentions
            )
            verified = residual == 0.0
            note = (
                f"masked column {probe} carries {residual:g} weight at its "
                f"heaviest across {len(attentions)} layers"
            )
        else:
            residual, verified = None, False
            note = f"could not be checked: {why}"

    # Not a formality. Every SCORING pass has to have been handed the same
    # arange(S) object, because a position_ids rebuilt per pass is exactly how
    # the RoPE re-phasing this file exists to avoid would get back in, and it
    # would get back in silently — the scores would still look like scores.
    # Exactly one pass is allowed something else, the reversed probe above,
    # and it is counted rather than merely permitted: an identity test that
    # can only ever pass is not a check, and this one could not fire at all
    # until `forward` had a caller that overrode the tensor.
    #
    # A plain RuntimeError, like the shape check in `forward` and for the same
    # reason: it is an assertion about this file's own instrumentation, not
    # about the model. Nothing the reader can do with it, and everything the
    # person fixing it needs is in the traceback — so it is a 500 with a log
    # line rather than a 409 claiming a decision was made.
    strays = [p for p in used_position_ids if p is not position_ids]
    if len(strays) != 1 or strays[0] is not wrong_position_ids:
        raise RuntimeError(
            f"{len(strays)} passes were handed a position_ids other than the "
            "one arange(S) tensor, and exactly one may be — the reversed "
            "probe. Rebuilding it anywhere else puts the phase shift a masked "
            "token causes in everything after it back into that token's score."
        )

    n_tested, n_candidates = len(tested), len(candidates)
    ranked.sort(key=lambda r: -r["kl"])
    return {
        "baseline": baseline,
        "position": position,
        "target_token": decode(top_id) if decode else str(top_id),
        # Echoed back rather than left for the caller to remember, so a row's
        # `group` can always be checked against the thing that produced it.
        "typed_span": list(typed_span) if typed_span is not None else None,
        "n_prompt": n_prompt,
        # Named so the caller can grey out what is indistinguishable from
        # arithmetic rather than presenting it as a weak result.
        "noise_floor_kl": round(floor, 6),
        "passes": passes,
        "elapsed_s": round(time.perf_counter() - started, 2),
        "ranked": ranked,
        "index0": index0,
        "n_tested": n_tested,
        "n_candidates": n_candidates,
        "truncated": n_tested < n_candidates,
        # Half-open, and the whole point of it is the gap between this and
        # `range(1, position)`. When the cap bites, the indices in that gap
        # were candidates and were not tested, and a reader has no way to tell
        # them from tokens that were tested and scored nothing: measured on
        # gpt2 with a 73-token prompt, 64 of 71 candidates were tested and
        # indices 1..7 came back with no mark of any kind. "Not asked" and
        # "asked, and the answer was nothing" are different findings.
        "tested_span": [tested[0], tested[-1] + 1],
        "coverage": (
            f"{n_tested} of {n_candidates} were tested; one not listed was not "
            "tested, not found unimportant."
        ),
        "sum_of_singles": round(sum(r["kl"] for r in ranked), 5),
        "joint_kl": round(kl_nats(base, joint), 5),
        "mask_verified": verified,
        "max_residual_weight": residual,
        "mask_check": (
            "A check of the MECHANISM, run once at one index, not a check of "
            f"each row: {note}. It uses output_attentions and so runs eager "
            "attention; it is not the pass that produced any KL here."
        ),
        # Said here so it travels with the numbers, not only in the UI.
        "means": (
            "KL divergence in nats of the next-token distribution at this "
            "position when that one token is masked out of every later "
            "position's attention. Larger = removing it alone moves the answer "
            "more. These are not shares of the answer. They do not add up, and "
            "which way they miss depends on which tokens you sum: with the "
            "prompt 'The capital of France is' in bf16 at the last prompt "
            "token, summing exactly the rows in this list over-states one "
            "joint mask of those same rows by 1.82x on gpt2 and 1.58x on "
            "gemma-3-270m-it, while summing only the typed span under-states "
            "it by 0.35x on gemma. sum_of_singles and joint_kl are both here "
            "so the gap is visible; neither is a correction factor."
        ),
    }
