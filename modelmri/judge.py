# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Score a rubric by reading the model's probability mass, not by sampling it.

Langfuse, LangSmith, Opik and Weave all run LLM-as-judge, and all return a
single number from a single sample of a hosted model — a sample presented as a
property. Reading the probability instead of sampling the text is only
possible if you hold the weights, which is the whole point of this tool.

So: one forward pass, softmax over the verdict token ids at the final
position. No generation, no temperature, no sampling. The same prompt gives
the same number every time, and the number is the model's actual mass on
"yes" rather than one draw from it.

## Run it k times, on k phrasings

A judge's answer to one phrasing of a rubric is a sample of that phrasing, not
of the rubric. So every score is measured over several paraphrases and
reported as min/median/max. A rubric where the paraphrases disagree is a
rubric this model does not answer stably, and the spread is the finding.

## It refuses in two places, and both matter

**No unambiguous verdict token.** Whether "yes" is one token depends on the
tokenizer and on the leading space — `" yes"` and `"yes"` are different ids in
a BPE vocabulary, and on some tokenizers neither is a single token at all.
Picking "close enough" here would silently read mass off the wrong token and
report a confident number about nothing. So this searches for a pair that both
encode to exactly one id, and REFUSES when there is none.

**The model did not answer.** If p(yes) + p(no) is a rounding error, the model
was going to say something else entirely and the ratio between two tiny
numbers is noise. Below a stated floor this refuses rather than normalising
two negligible masses into a confident-looking percentage.

## A weak judge is a weak judge

A small local model is not a good evaluator, and a well-calibrated report of a
weak judge's opinion is still a weak judge's opinion. So the judge model's
name is attached to every score, and NOTHING here aggregates across rubrics or
across runs into a project-level number. That aggregate is exactly where a
sample starts being treated as a property, which is the error this whole
project exists to avoid.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field

from .errors import BadRequest, Refusal

# EVERY surface form of the verdict, and the mass of all of them is summed.
#
# Not one chosen pair. MEASURED on Qwen2.5-0.5B-Instruct: a prompt ending
# "yes or no:" is answered with lowercase " yes" (0.637 mass on that form),
# and a prompt ending "Answer:" is answered with capitalised " Yes" — where
# reading only the lowercase form saw 0.0215 and concluded the model had not
# answered. It had; the casing is prompt-dependent, and picking one form
# globally means reading the wrong token on some prompts.
#
# Summing is not a heuristic: each form still has to be a single token, and
# every one of them is literally the word the model would be saying.
YES_FORMS = (" yes", " Yes", " YES", "yes", "Yes", " true", " True", "true")
NO_FORMS = (" no", " No", " NO", "no", "No", " false", " False", "false")

# Below this combined mass the model was answering a different question and
# the yes/no ratio is noise between two rounding errors.
#
# MEASURED, not chosen, summing every single-token casing:
#
#   Qwen2.5-0.5B-Instruct   mass 0.021 - 0.974 depending on the phrasing, and
#                           p(yes) 0.990 on the rubric "does it mention a
#                           cat?".
#
# A weak judge sits at the other end: a little mass on the verdict token, and
# a p(yes) near a coin flip REGARDLESS OF THE TEXT.
#
# The floor is for "did not answer", NOT for "is a weak judge". A model
# committing a tenth of its mass HAS answered — badly — and refusing it here
# would be tuning a threshold to exclude one model. What exposes a weak judge
# is that its answer does not move with
# the text, and that the mass is reported beside the ratio rather than hidden
# behind it. An earlier draft used 0.01, which let through a case with 2%.
MIN_VERDICT_MASS = 0.10

# How the same rubric is asked. Each is one forward pass; k of them is k
# passes, which on a 1.7B model is seconds and on 8 GB is affordable.
PARAPHRASES = (
    "{context}\n\nQuestion: {rubric}\nAnswer with one word, yes or no:",
    "{context}\n\nJudge the text above. {rubric}\nyes or no:",
    "Read this:\n{context}\n\n{rubric} Reply yes or no.\nAnswer:",
    "{context}\n\nDoes the following hold? {rubric}\nOne word, yes or no:",
)

MAX_CONTEXT_CHARS = 8_000
MAX_RUBRIC_CHARS = 500


class JudgeError(BadRequest):
    """This rubric cannot be scored honestly, and the message says why."""


@dataclass
class Tokens:
    """Every verdict form this tokenizer can express as a single token."""

    yes_ids: tuple = ()
    no_ids: tuple = ()
    yes_forms: tuple = ()
    no_forms: tuple = ()

    def to_dict(self) -> dict:
        return {
            "yes_ids": list(self.yes_ids),
            "no_ids": list(self.no_ids),
            "yes_forms": list(self.yes_forms),
            "no_forms": list(self.no_forms),
        }


def _single_token_forms(tokenizer, forms) -> tuple:
    """The forms this tokenizer encodes to exactly one id, and those DISTINCT ids.

    DEDUPLICATED, and that is the whole point of this function's shape. Several
    casings can share one id — an uncased vocabulary, or a sentencepiece
    tokenizer with `remove_extra_whitespaces=True`, maps ' yes', ' Yes', 'yes'
    and 'Yes' to the same token. `score` reads the mass with
    `probs[list(ids)].sum()`, so a repeated id is ADDED ONCE PER CASING.

    Measured on `openai-community/openai-gpt`, which lowercases and strips
    whitespace: without this, yes_ids came back as
    (685, 685, 685, 685, 685, 1849, 1849, 1849) and the reported verdict mass
    was 4.136 where the true mass was 0.827. A probability above 1 is not a
    thing, the `mass <= 1` invariant this module's own tests assert was
    violated, and the MIN_VERDICT_MASS floor stopped firing on paraphrases the
    model genuinely had not answered.

    The FORMS list keeps one entry per id rather than being deduplicated
    independently, so what is reported still describes what was found.
    """
    by_id: dict = {}
    for form in forms:
        try:
            encoded = tokenizer.encode(form, add_special_tokens=False)
        except Exception:
            continue
        if len(encoded) == 1:
            # First form wins, so the reported surface form is the one this
            # module prefers rather than whichever casing happened to be last.
            by_id.setdefault(int(encoded[0]), form)
    return tuple(by_id.values()), tuple(by_id)


def verdict_tokens(tokenizer) -> Tokens:
    """Every single-token yes and no form, or a refusal.

    NOT a heuristic, and not one chosen pair. `" yes"` and `"yes"` are
    different ids, and which one a model uses depends on how the prompt ends —
    measured, a prompt ending "Answer:" is answered with " Yes" while one
    ending "yes or no:" is answered with " yes". Reading a single chosen form
    saw 2% mass on a model that had answered with 89% confidence in the other
    casing, and reported that it had not answered.

    Refuses when either side has no single-token form at all: reading mass off
    a multi-token verdict means reading the first piece of a word — `"ye"` —
    and reporting it as the model's answer.
    """
    yes_forms, yes_ids = _single_token_forms(tokenizer, YES_FORMS)
    no_forms, no_ids = _single_token_forms(tokenizer, NO_FORMS)

    # An id on both sides would be counted twice and would mean the tokenizer
    # cannot distinguish the verdicts at all.
    overlap = set(yes_ids) & set(no_ids)
    if overlap:
        # Explicit comprehensions, not `zip(*pairs) or ((), ())`: a `zip`
        # object is ALWAYS truthy, so that fallback never fires and the unpack
        # raises ValueError on an empty list — which is exactly the case this
        # branch exists to handle.
        keep_yes = [
            (f, i) for f, i in zip(yes_forms, yes_ids, strict=True) if i not in overlap
        ]
        keep_no = [
            (f, i) for f, i in zip(no_forms, no_ids, strict=True) if i not in overlap
        ]
        yes_forms = tuple(f for f, _ in keep_yes)
        yes_ids = tuple(i for _, i in keep_yes)
        no_forms = tuple(f for f, _ in keep_no)
        no_ids = tuple(i for _, i in keep_no)

    if not yes_ids or not no_ids:
        # TWO DIFFERENT FAULTS, two different sentences. They used to share
        # one, which sent a reader whose tokenizer encodes every form as a
        # single token off to check tokenisation granularity — a true
        # refusal carrying a false diagnosis.
        if overlap:
            raise Refusal(
                "This tokenizer cannot tell yes from no: every form of both "
                "encodes to the same token id, so there is nothing to compare. "
                "That usually means the tokenizer does not match the model, or "
                "is a domain vocabulary that maps unknown words to one id. "
                "Refusing rather than reading one id as both answers."
            )
        raise Refusal(
            "This tokenizer has no single-token yes/no form, so there is no id "
            f"to read a verdict off. Tried {', '.join(repr(f) for f in YES_FORMS)} "
            f"and {', '.join(repr(f) for f in NO_FORMS)}. Refusing rather than "
            "reading the first piece of a multi-token word and calling it the "
            "model's answer."
        )
    return Tokens(
        yes_ids=tuple(yes_ids),
        no_ids=tuple(no_ids),
        yes_forms=tuple(yes_forms),
        no_forms=tuple(no_forms),
    )


@dataclass
class Pass:
    """One paraphrase, one forward pass."""

    paraphrase: int
    p_yes: float
    p_no: float
    mass: float
    # Did the model put enough on a verdict token for the ratio to mean
    # anything? A paraphrase it did not answer is carried, not dropped — that
    # it does not answer this phrasing is a fact about the rubric.
    answered: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Score:
    """One rubric against one text, over k paraphrases."""

    rubric: str
    passes: list = field(default_factory=list)
    tokens: Tokens | None = None
    # Named on every score. A weak judge's calibrated opinion is still a weak
    # judge's opinion, and the name is what lets a reader weigh it.
    judge_model: str = ""
    dtype: str = ""
    device: str = ""
    seed: int | None = None

    @property
    def scores(self) -> list:
        """p(yes) over the paraphrases the model ACTUALLY ANSWERED.

        A paraphrase it did not answer contributes a ratio between two
        rounding errors, and averaging that in would be exactly the noise the
        floor exists to keep out.
        """
        answered = [p.p_yes for p in self.passes if p.answered]
        return answered or [p.p_yes for p in self.passes]

    @property
    def n_unanswered(self) -> int:
        return sum(1 for p in self.passes if not p.answered)

    def to_dict(self) -> dict:
        out = {
            "rubric": self.rubric,
            "passes": [p.to_dict() for p in self.passes],
            "tokens": self.tokens.to_dict() if self.tokens else None,
            "judge_model": self.judge_model,
            "dtype": self.dtype,
            "device": self.device,
            "seed": self.seed,
            "means": self.means(),
        }
        if self.passes:
            values = self.scores
            out |= {
                "low": round(min(values), 6),
                "median": round(statistics.median(values), 6),
                "high": round(max(values), 6),
                "spread": round(max(values) - min(values), 6),
                "n_paraphrases": len(values),
            }
        return out

    def means(self) -> str:
        if not self.passes:
            return "Nothing was scored."
        values = self.scores
        low, high = min(values), max(values)
        median = statistics.median(values)
        spread = high - low
        parts = [
            f"p(yes) is {median:.3f} across {len(values)} paraphrase(s) of this "
            f"rubric, ranging {low:.3f} to {high:.3f}. This is the model's "
            f"probability mass on the verdict token, read from one forward "
            f"pass each — not a sampled label."
        ]
        # HOW MUCH it committed, always — not only the ratio. A weak judge puts
        # a few percent on a verdict token and then splits it near 50/50
        # whatever it is shown, saying no — weakly — to a true statement as
        # readily as to a false one. The ratio alone hides that; the mass
        # beside it does not.
        masses = [p.mass for p in self.passes if p.answered]
        if masses:
            committed = statistics.median(masses)
            parts.append(f"It put {committed:.1%} of its mass on a verdict token.")
            if committed < 0.25:
                parts.append(
                    "THAT IS MOST OF THE WAY TO NOT ANSWERING: the majority of "
                    "this model's probability was on something other than yes "
                    "or no, so the ratio above describes a small corner of what "
                    "it was going to say."
                )
        # Named, not silently excluded. A phrasing this model does not answer
        # is a fact about the rubric, and a median over 3 of 4 that reads as a
        # median over 4 is the same omission this project keeps refusing.
        if self.n_unanswered:
            parts.append(
                f"{self.n_unanswered} further phrasing(s) are NOT in that "
                f"median: the model put almost nothing on a verdict token for "
                f"them, so there was no answer to include."
            )
        # The spread is the finding when it is wide. A single median from
        # paraphrases that disagree is the "sample presented as a property"
        # error, one level up.
        if spread > 0.25:
            parts.append(
                f"THE PARAPHRASES DISAGREE by {spread:.3f}. This rubric is not "
                f"one this model answers stably, so the median above describes "
                f"the wording as much as the text."
            )
        parts.append(
            f"Judged by {self.judge_model or 'the loaded model'}"
            + (f" in {self.dtype}" if self.dtype else "")
            + (f" on {self.device}" if self.device else "")
            + ". A small local judge is a weak judge, and this number is that "
            "judge's opinion rather than a property of the text."
        )
        return " ".join(parts)


def build_prompt(context: str, rubric: str, index: int) -> str:
    """One paraphrase of the rubric, filled in."""
    template = PARAPHRASES[index % len(PARAPHRASES)]
    return template.format(context=context.strip(), rubric=rubric.strip())


def plan(context: str, rubric: str, n_paraphrases: int = 0) -> list:
    """The prompts that would be run, before any of them is."""
    context = str(context or "")
    rubric = str(rubric or "").strip()
    if not rubric:
        raise JudgeError(
            "a rubric is a question this model can answer yes or no to, and "
            "none was given."
        )
    if len(rubric) > MAX_RUBRIC_CHARS:
        raise JudgeError(
            f"that rubric is {len(rubric)} characters and the cap is "
            f"{MAX_RUBRIC_CHARS}. A rubric long enough to need more than that "
            f"is asking several questions, and the answer would not say which."
        )
    if not context.strip():
        raise JudgeError("there is no text to judge.")
    if len(context) > MAX_CONTEXT_CHARS:
        raise JudgeError(
            f"that text is {len(context):,} characters and the cap is "
            f"{MAX_CONTEXT_CHARS:,}. Judge a passage rather than a corpus, so "
            f"the answer is about something you can point at."
        )
    k = int(n_paraphrases or len(PARAPHRASES))
    if not 1 <= k <= len(PARAPHRASES):
        raise JudgeError(
            f"this asks the rubric {len(PARAPHRASES)} ways at most, and was "
            f"asked for {k}."
        )
    return [build_prompt(context, rubric, i) for i in range(k)]


def score(
    model,
    tokenizer,
    context: str,
    rubric: str,
    *,
    n_paraphrases: int = 0,
    device: str = "cpu",
    seed: int | None = None,
    min_mass: float = MIN_VERDICT_MASS,
) -> Score:
    """Score one rubric. One forward pass per paraphrase, no generation."""
    import torch

    prompts = plan(context, rubric, n_paraphrases)
    tokens = verdict_tokens(tokenizer)

    out = Score(
        rubric=str(rubric).strip(),
        tokens=tokens,
        judge_model=str(
            getattr(getattr(model, "config", None), "_name_or_path", "") or ""
        ),
        device=device,
        seed=seed,
    )
    try:
        out.dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
    except (StopIteration, AttributeError):
        out.dtype = ""

    for i, prompt in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            logits = model(ids).logits[0, -1]
        # Softmax over the WHOLE vocabulary, then read the verdict ids.
        # Softmax over just those would renormalise them to sum to 1 and
        # destroy the very signal the floor is checking for.
        probs = torch.softmax(logits.float(), dim=-1)
        p_yes = float(probs[list(tokens.yes_ids)].sum())
        p_no = float(probs[list(tokens.no_ids)].sum())
        mass = p_yes + p_no
        out.passes.append(
            Pass(
                paraphrase=i,
                # p(yes) AMONG the verdict tokens — which of the two, given it
                # answered at all. `mass` is carried beside it so "it answered,
                # and said yes" stays distinguishable from "it barely
                # answered", and `answered` is that distinction made explicit.
                p_yes=round(p_yes / mass, 6) if mass else 0.0,
                p_no=round(p_no / mass, 6) if mass else 0.0,
                mass=round(mass, 6),
                answered=mass >= min_mass,
            )
        )

    # ONE low-mass paraphrase is not a failed measurement — it is a phrasing
    # this model does not answer, which is itself worth reporting, and three
    # paraphrases that answered clearly still describe the text. An earlier
    # draft raised on the first pass below the floor and threw all of that
    # away: measured, paraphrase 1 gave mass 0.72 and p(yes) 0.998 while
    # paraphrase 2 gave 0.02, and the whole score was refused.
    #
    # The refusal is for a model that answered NO phrasing at all.
    answered = [p for p in out.passes if p.answered]
    if not answered:
        worst = max((p.mass for p in out.passes), default=0.0)
        raise Refusal(
            f"This model did not answer the rubric in any of "
            f"{len(out.passes)} phrasings: the most it put on a verdict token "
            f"was {worst:.2%}, below the {min_mass:.0%} floor. The ratio "
            f"between two rounding errors is noise, so there is no score to "
            f"report."
        )
    return out
