"""What constrained decoding cost you, per step.

Ollama, llama.cpp (GBNF), vLLM and LM Studio all ship constrained decoding as a
black box: you get valid JSON and no idea what it cost. Nobody shows the mask's
effect on the distribution, and "structured output mode makes the model dumber"
is a widespread complaint with no instrument pointed at it.

This is that instrument. Per step it records the model's UNCONSTRAINED top-k
beside the grammar's allowed set, how much probability mass the mask deleted,
and whether the token the model most wanted was forbidden. A step where the
mask removed most of the distribution is where your structured output stopped
being the model's answer and started being the schema's.

MEASURED with a two-field object schema: a single-figure number of tokens is
legal before anything is written (whitespace and the brace variants), and
about as few at the key position — over 99.9% of the vocabulary masked — while
inside a free string value almost the whole vocabulary is legal. The contrast
IS the finding, and neither number is visible from the completion alone.

(An earlier draft of this docstring quoted 2 and 50,174. Those came from a run
with a broken token list — see `tokenizer_data` — and are exactly the kind of
number this project refuses to leave standing: measured once, from a bug, and
written down as a property.)

## The grammar is not ours, on purpose

`lm-format-enforcer` compiles the schema. This module does not write a
token-level grammar compiler and never will: a wrong mask is an invisible
failure in a tool whose whole premise is not shipping invisible wrong answers.
You would get valid-looking JSON built from a mask nobody checked.

What IS written here is the ~40 lines of glue that turn its enforcer into a
`LogitsProcessor`. Their own `integrations.transformers` module cannot be used:
it imports `transformers.tokenization_utils.PreTrainedTokenizerBase`, which
does not exist in transformers 5.x, and raises "transformers is not installed"
on a machine where transformers is installed and working. That is the same
defect class `modelmri_record.verify` exists to catch — an integration layer
reading an attribute that moved.

## The enforcer must see every step

`TokenEnforcer` memoises per prefix and expects to be asked about each position
in order. Handing it an arbitrary prefix it never walked returns the ROOT
state's answer, silently — measured, asking about `{"name": "` cold gave the
same two tokens as asking about the empty string. A `LogitsProcessor` is called
once per step, which is exactly the access pattern it wants, and
`_STEP_ORDER_NOTE` below is why nothing here offers a "just check this prefix"
helper.

## The deleted mass is measured under the sampler that was running

Probability is read from the pre-mask distribution at that step, so
`deleted_mass` is what the mask actually removed from what the model was going
to do — not from a temperature-1 distribution it never had. Temperature and
top-k travel with the record for that reason.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .errors import BadRequest, Refusal

# How many unconstrained candidates are kept per step. Enough to see what the
# model wanted and what it settled for; not so many that a 200-token
# completion becomes megabytes.
TOP_K = 8

# Above this the recorder stops rather than growing without bound. Reported,
# never silent -- a truncated trace read as a complete one is the failure this
# project keeps refusing.
MAX_STEPS = 2_000

_STEP_ORDER_NOTE = (
    "the enforcer is memoised per prefix and must be asked about every step in "
    "order; asking about an arbitrary prefix returns the root state's answer "
    "with nothing saying so"
)


class GrammarError(BadRequest):
    """This schema cannot be enforced honestly, and the message says why."""


def _enforcer_module():
    """`lm-format-enforcer`, or a refusal naming the extra that provides it."""
    try:
        import lmformatenforcer  # noqa: F401
        from lmformatenforcer import (
            JsonSchemaParser,
            TokenEnforcer,
            TokenEnforcerTokenizerData,
        )
    except ImportError as err:
        raise Refusal(
            "Constrained decoding needs `lm-format-enforcer`, which is an "
            "optional extra: `pip install modelmri[grammar]`. It is optional "
            "because it is the only way to get a correct token-level mask, and "
            "ModelMRI will not hand-write one -- a wrong mask is an invisible "
            "failure, and you would get valid-looking output built from it."
        ) from err
    return JsonSchemaParser, TokenEnforcer, TokenEnforcerTokenizerData


@dataclass
class Step:
    """One decoding step, with and without the mask."""

    step: int
    # What the model wanted, before the grammar touched anything.
    wanted: str = ""
    wanted_id: int = -1
    wanted_p: float = 0.0
    # Whether the grammar permitted it.
    wanted_was_allowed: bool = True
    # What was actually emitted.
    chosen: str = ""
    chosen_id: int = -1
    # The emitted token's pre-mask probability, or None when it was not among
    # the `top` rows recorded here. It defaulted to 0.0, which reads as "the
    # model gave this token no probability" -- the opposite of the truth for a
    # token the grammar forced through from outside the top few. `top` holds
    # the strongest handful; anything past that is unrecorded, not zero.
    chosen_p: float | None = None
    allowed_count: int = 0
    vocab_size: int = 0
    # Probability the mask removed, from the PRE-MASK distribution at this
    # step under the sampler that was running.
    deleted_mass: float = 0.0
    top: list = field(default_factory=list)

    @property
    def masked_fraction(self) -> float:
        return 1.0 - (self.allowed_count / self.vocab_size) if self.vocab_size else 0.0

    def to_dict(self) -> dict:
        out = asdict(self)
        out["masked_fraction"] = round(self.masked_fraction, 6)
        return out


@dataclass
class Trace:
    """A whole constrained generation, and what the schema cost it."""

    steps: list = field(default_factory=list)
    schema: str = ""
    temperature: float = 0.0
    truncated: int = 0
    vocab_size: int = 0

    @property
    def overridden(self) -> list:
        """Steps where the token the model most wanted was forbidden."""
        return [s for s in self.steps if not s.wanted_was_allowed]

    def to_dict(self) -> dict:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "schema": self.schema,
            "temperature": self.temperature,
            "truncated": self.truncated,
            "vocab_size": self.vocab_size,
            "n_overridden": len(self.overridden),
            "means": self.means(),
        }

    def means(self) -> str:
        if not self.steps:
            return "Nothing was generated under this schema."
        overridden = self.overridden
        worst = max(self.steps, key=lambda s: s.deleted_mass)
        parts = [
            f"{len(self.steps)} step(s) under this schema. The mask deleted a "
            f"median of {self._median_deleted():.1%} of the probability at each "
            f"step, measured on the distribution the model actually had at that "
            f"step rather than on a temperature-1 one it never had."
        ]
        if overridden:
            names = ", ".join(repr(s.wanted) for s in overridden[:4])
            parts.append(
                f"AT {len(overridden)} STEP(S) THE TOKEN THE MODEL MOST WANTED "
                f"WAS FORBIDDEN ({names}). That is where the output stopped "
                f"being the model's answer and started being the schema's."
            )
        else:
            parts.append(
                "At no step was the model's first choice forbidden, so the "
                "schema shaped the output without overriding it."
            )
        parts.append(
            f"The heaviest single step was {worst.step}, where "
            f"{worst.deleted_mass:.1%} was removed and "
            f"{worst.allowed_count} of {worst.vocab_size} tokens remained."
        )
        if self.truncated:
            parts.append(
                f"{self.truncated} further step(s) were NOT recorded: the "
                f"recorder holds {MAX_STEPS}."
            )
        return " ".join(parts)

    def _median_deleted(self) -> float:
        import statistics

        return statistics.median([s.deleted_mass for s in self.steps])


def tokenizer_data(tokenizer):
    """The vocabulary, in the shape `TokenEnforcer` wants.

    Built once per tokenizer and reused: it walks the whole vocabulary, which
    is 50k+ `convert_ids_to_tokens` calls, and doing that per generation would
    dominate the cost of the thing being measured.
    """
    _, _, TokenEnforcerTokenizerData = _enforcer_module()

    eos = tokenizer.eos_token_id
    if eos is None:
        raise GrammarError(
            "this tokenizer states no end-of-sequence token, so the grammar has "
            "no way to know a value is finished."
        )

    vocab_size = len(tokenizer)
    special = set(getattr(tokenizer, "all_special_ids", ()) or ())

    # This mirrors `lmformatenforcer.integrations.transformers`, which cannot
    # be imported here (see the module docstring). Getting it wrong is not
    # subtle in its effects but IS subtle to spot:
    #
    # A first version used `convert_ids_to_tokens`, which on a byte-level BPE
    # vocabulary returns the ENCODED form — " Doe" comes back as "ĠDoe",
    # where Ġ stands in for the space. The grammar then saw Ġ as a
    # literal character inside a JSON string, its parser dead-ended, and the
    # allowed set collapsed to {EOS} mid-value. Measured: `{"name":"John` had
    # 50,174 tokens allowed and `{"name":"John Doe` had exactly 1.
    #
    # It survived an earlier check because that check used `{"name":"bob"}` —
    # no spaces inside any value, so no word-start token was ever exercised.
    token_zero = tokenizer.encode("0")[-1]
    regular = []
    for token_id in range(vocab_size):
        if token_id in special:
            continue
        # Prepend a known token and drop its one character, so a word-start
        # token yields the leading space it actually contributes.
        after_zero = tokenizer.decode([token_zero, token_id])[1:]
        plain = tokenizer.decode([token_id])
        regular.append((token_id, after_zero, len(after_zero) > len(plain)))

    def decode(ids) -> str:
        # A byte-BPE token can decode to a PARTIAL UTF-8 sequence, which comes
        # back as U+FFFD. Left in place it would be a character the grammar has
        # to account for and cannot.
        return tokenizer.decode(ids).rstrip("�")

    return TokenEnforcerTokenizerData(regular, decode, eos, False, vocab_size)


class MaskRecorder:
    """A `LogitsProcessor` that enforces a schema AND records what it cost.

    One object does both because they must see the same distribution: a
    recorder that ran beside the enforcer would read logits from a different
    step and report a mask that was never applied.
    """

    def __init__(self, tokenizer, schema: dict, data=None, temperature: float = 0.0):
        JsonSchemaParser, TokenEnforcer, _ = _enforcer_module()
        if not isinstance(schema, dict) or not schema:
            raise GrammarError("a JSON schema is an object, and this one is not.")
        try:
            parser = JsonSchemaParser(schema)
        except Exception as err:
            # The library's own words are not published: this authors its own
            # sentence and keeps theirs on the traceback.
            raise GrammarError(
                "that JSON schema could not be compiled into a grammar. Check "
                "that every 'type' is one this supports and that '$ref' targets "
                "resolve."
            ) from err

        import json

        self.tokenizer = tokenizer
        self.data = data if data is not None else tokenizer_data(tokenizer)
        self.enforcer = TokenEnforcer(self.data, parser)
        self.trace = Trace(
            schema=json.dumps(schema, sort_keys=True)[:2000],
            temperature=float(temperature),
            vocab_size=self.data.vocab_size
            if hasattr(self.data, "vocab_size")
            else len(tokenizer.get_vocab()),
        )
        self._n_prompt = None
        # Steps SEEN, which keeps counting after `steps` stops growing at
        # MAX_STEPS. `record_choice` needs to know the difference: past the
        # cap there is no row for the token being reported, and writing it to
        # the last one there is corrupts a row that already described a
        # different step.
        self._steps_seen = 0

    def __call__(self, input_ids, scores):
        import torch

        # The prompt length, fixed at the first call. Everything after it is
        # the generated prefix the enforcer needs -- and it needs EXACTLY the
        # generated part, because the grammar starts at the first new token.
        row = input_ids[0]
        if self._n_prompt is None:
            self._n_prompt = int(row.shape[0])
        generated = [int(t) for t in row[self._n_prompt :]]

        allowed = self.enforcer.get_allowed_tokens(generated).allowed_tokens
        allowed_ids = list(allowed)

        probs = torch.softmax(scores[0].float(), dim=-1)
        top = torch.topk(probs, min(TOP_K, probs.shape[-1]))
        wanted_id = int(top.indices[0])
        allowed_set = set(allowed_ids)

        keep = torch.zeros_like(scores[0], dtype=torch.bool)
        if allowed_ids:
            keep[torch.tensor(allowed_ids, device=scores.device)] = True
        # The mass the mask removes, from the distribution the model actually
        # had at this step.
        #
        # SUMMED IN FLOAT64, then clamped. Adding ~50,000 float32 terms
        # accumulates enough error to exceed 1.0 -- measured, a step where the
        # mask removed essentially everything reported 1.000023, and a
        # probability above 1 is not a thing to print at a reader. float64
        # removes the accumulation; the clamp is the belt for the remaining
        # ulp, not a substitute for it.
        deleted = float(probs.double()[~keep].sum())
        deleted = min(1.0, max(0.0, deleted))

        self._steps_seen += 1
        if len(self.trace.steps) < MAX_STEPS:
            self.trace.steps.append(
                Step(
                    step=len(self.trace.steps),
                    wanted=self.tokenizer.decode([wanted_id]),
                    wanted_id=wanted_id,
                    wanted_p=round(float(top.values[0]), 6),
                    wanted_was_allowed=wanted_id in allowed_set,
                    allowed_count=len(allowed_ids),
                    vocab_size=int(scores.shape[-1]),
                    deleted_mass=round(deleted, 6),
                    top=[
                        {
                            "token": self.tokenizer.decode([int(i)]),
                            "p": round(float(p), 6),
                            "allowed": int(i) in allowed_set,
                        }
                        for p, i in zip(top.values, top.indices, strict=True)
                    ],
                )
            )
        else:
            self.trace.truncated += 1

        # -inf, not a large negative: a finite penalty leaves a forbidden token
        # reachable under sampling, and the whole promise of constrained
        # decoding is that it is not.
        masked = scores.clone()
        masked[0][~keep] = float("-inf")
        return masked

    def record_choice(self, token_id: int) -> None:
        """What was actually emitted at the last recorded step.

        Separate from `__call__` because the processor runs BEFORE the token is
        chosen; the choice is only knowable afterwards. A step whose `chosen`
        is never set keeps `chosen_id = -1`, which reads as "not recorded"
        rather than as token 0.
        """
        # Only while steps are still being RECORDED. `__call__` stops
        # appending at MAX_STEPS and this went on writing to `steps[-1]`, so
        # every token past the cap overwrote the last recorded step's choice
        # -- step 1,999 ended up showing the token emitted at step 5,000,
        # beside its own `wanted` and its own `top`. A row describing two
        # different moments is worse than a missing row.
        if not self.trace.steps or self._steps_seen > len(self.trace.steps):
            return
        step = self.trace.steps[-1]
        step.chosen_id = int(token_id)
        step.chosen = self.tokenizer.decode([int(token_id)])
        for entry in step.top:
            if entry["token"] == step.chosen:
                step.chosen_p = entry["p"]
                break


def plan(schema: dict, tokenizer) -> dict:
    """What the grammar permits at step 0, before anything is generated.

    Cheap, and it answers the question people actually have first: how much of
    the vocabulary does this schema rule out before the model says anything?
    """
    recorder = MaskRecorder(tokenizer, schema)
    allowed = recorder.enforcer.get_allowed_tokens([]).allowed_tokens
    ids = list(allowed)
    total = len(tokenizer.get_vocab())
    return {
        "allowed_at_start": len(ids),
        "vocab_size": total,
        "masked_fraction": round(1 - len(ids) / total, 6) if total else 0.0,
        "examples": [tokenizer.decode([i]) for i in ids[:8]],
        "note": _STEP_ORDER_NOTE,
    }
