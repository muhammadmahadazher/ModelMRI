# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""A judge that reads probability mass, against a real model.

Two refusals carry this feature and both are tested against planted
conditions rather than asserted in prose:

  * a tokenizer with no single-token yes/no pair. `" yes"` and `"yes"` are
    different ids and on some tokenizers neither is single — reading mass off
    the first piece of a multi-token word would be a confident number about
    nothing;
  * a model that did not answer. When p(yes) + p(no) is a rounding error, the
    ratio between them is noise, and normalising two negligible masses into a
    percentage is the shape of a lie.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from modelmri import judge  # noqa: E402
from modelmri.errors import BadRequest, Refusal  # noqa: E402

# The judge is tested against a model that can ACTUALLY ANSWER a yes/no
# rubric. gpt2 cannot — measured, it puts 2.4%-5.9% on the verdict tokens and
# its p(yes) sits near 0.6 whatever it is shown — so testing this feature on
# gpt2 alone would only ever exercise the refusal path and would never check
# that a score means anything.
#
# gpt2 keeps exactly one job here: the control for "this model cannot answer",
# which is a real test rather than a default nobody revisited.
JUDGE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _judge_model_is_cached() -> bool:
    """Only run against a model already on this machine.

    Same rule `test_saes` follows. gpt2 is small enough that the suite has
    always fetched it, but a 0.5B instruct model is a ~1 GB download and the
    cross-platform CI jobs cache `uv.lock`, not the HuggingFace hub — so
    requiring it turns every runner into a gigabyte of traffic and one more
    thing that can time out. Cached locally: the capable-model assertions run.
    Not cached: they skip, and the gpt2 control below still runs everywhere.
    """
    import os
    from pathlib import Path

    home = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
    roots = [Path(home)] if home else []
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    name = "models--" + JUDGE_MODEL.replace("/", "--")
    return any(p.exists() for root in roots for p in (root / name, root / "hub" / name))


needs_judge_model = pytest.mark.skipif(
    not _judge_model_is_cached(),
    reason=f"{JUDGE_MODEL} is not in the local HF cache",
)


@pytest.fixture(scope="module")
def judge_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, dtype="auto").eval()
    return model, tok


@pytest.fixture(scope="module")
def gpt2():
    """The control: a model that does not answer yes/no rubrics."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    return model, tok


# ---------------------------------------------------- the verdict tokens


@needs_judge_model
def test_a_real_tokenizer_yields_a_single_token_pair(judge_model):
    _, tok = judge_model
    found = judge.verdict_tokens(tok)
    assert found.yes_ids and found.no_ids
    for form in found.yes_forms + found.no_forms:
        assert len(tok.encode(form, add_special_tokens=False)) == 1


@needs_judge_model
def test_every_casing_the_tokenizer_can_express_is_collected(judge_model):
    """Measured: a prompt ending "yes or no:" is answered with " yes" and one
    ending "Answer:" with " Yes". Reading one chosen form saw 2% mass on a
    model that had answered with 89% confidence in the other casing."""
    _, tok = judge_model
    found = judge.verdict_tokens(tok)
    assert " yes" in found.yes_forms and " Yes" in found.yes_forms
    assert " no" in found.no_forms and " No" in found.no_forms
    assert not set(found.yes_ids) & set(found.no_ids)


def test_a_tokenizer_with_no_single_token_verdict_is_refused():
    """NOT a heuristic. Reading `"ye"` and calling it the model's answer is a
    confident number about nothing."""

    class Splitter:
        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3]  # everything is multi-token

    with pytest.raises(Refusal) as caught:
        judge.verdict_tokens(Splitter())
    message = str(caught.value)
    assert "no single-token yes/no form" in message
    assert "first piece of a multi-token word" in message


def test_a_tokenizer_that_maps_yes_and_no_to_one_id_is_refused():
    """Same id for both means there is nothing to compare — and the refusal
    must say THAT, not that the forms are multi-token. They are single tokens;
    the tokenizer just cannot tell them apart. An earlier version shared one
    message for both faults and sent this reader to check tokenisation
    granularity."""

    class Collapsing:
        def encode(self, text, add_special_tokens=False):
            return [7]

    with pytest.raises(Refusal) as caught:
        judge.verdict_tokens(Collapsing())
    message = str(caught.value)
    assert "cannot tell yes from no" in message
    assert "does not match the model" in message
    assert "multi-token" not in message, "that is the other fault's diagnosis"


def test_casings_that_share_an_id_are_counted_once():
    """Found by an adversarial review and MEASURED on
    openai-community/openai-gpt, which lowercases and strips whitespace: every
    yes casing collapsed to id 685 and every 'true' casing to 1849, so
    `probs[list(yes_ids)].sum()` added the same probability five and three
    times. Reported verdict mass was 4.136 against a true 0.827 — above 1,
    breaking this module's own invariant, and the floor stopped firing on
    paraphrases the model had not answered."""

    class Uncased:
        """Every casing of a word maps to one id."""

        def encode(self, text, add_special_tokens=False):
            word = text.strip().lower()
            return [{"yes": 685, "true": 1849, "no": 664, "false": 6843}[word]]

    found = judge.verdict_tokens(Uncased())
    assert found.yes_ids == (685, 1849), found.yes_ids
    assert found.no_ids == (664, 6843), found.no_ids
    assert len(set(found.yes_ids)) == len(found.yes_ids)
    assert len(set(found.no_ids)) == len(found.no_ids)
    # One form reported per id, so what is shown still describes what was found.
    assert len(found.yes_forms) == len(found.yes_ids)


def test_a_collapsing_tokenizer_cannot_push_mass_above_one(gpt2):
    """The invariant the duplication broke: a probability above 1 is not a
    thing, and `mass` is what the floor is compared against."""
    model, tok = gpt2

    class Uncased:
        eos_token_id = tok.eos_token_id

        def encode(self, text, add_special_tokens=False):
            word = text.strip().lower()
            table = {"yes": 3763, "true": 3763, "no": 645, "false": 645}
            return [table[word]]

        def __call__(self, *a, **k):
            return tok(*a, **k)

        def decode(self, ids):
            return tok.decode(ids)

    out = judge.score(
        model, Uncased(), "The cat sat.", "Is there an animal?", min_mass=0.0
    )
    for p in out.passes:
        assert 0.0 <= p.mass <= 1.0, f"mass {p.mass} is not a probability"


def test_a_tokenizer_that_raises_is_not_fatal():
    """A tokenizer failure on one surface form means 'not this pair', not a
    crash — but with every form failing it still refuses."""

    class Angry:
        def encode(self, text, add_special_tokens=False):
            raise RuntimeError("no")

    with pytest.raises(Refusal):
        judge.verdict_tokens(Angry())


# ------------------------------------------------------------- the plan


def test_the_prompts_are_visible_before_any_pass_is_run():
    prompts = judge.plan("the sky is blue", "does the text mention colour?")
    assert len(prompts) == len(judge.PARAPHRASES)
    assert all("the sky is blue" in p for p in prompts)
    assert all("colour" in p for p in prompts)


def test_the_paraphrases_actually_differ():
    """k identical prompts would be one measurement reported as k."""
    prompts = judge.plan("x", "y")
    assert len(set(prompts)) == len(prompts)


def test_an_empty_rubric_is_refused():
    with pytest.raises(BadRequest, match="none was given"):
        judge.plan("some text", "   ")


def test_an_empty_text_is_refused():
    with pytest.raises(BadRequest, match="no text to judge"):
        judge.plan("", "is it good?")


def test_a_rubric_long_enough_to_be_several_questions_is_refused():
    with pytest.raises(BadRequest, match="asking several questions"):
        judge.plan("x", "a" * (judge.MAX_RUBRIC_CHARS + 1))


def test_a_corpus_sized_text_is_refused():
    with pytest.raises(BadRequest, match="rather than a corpus"):
        judge.plan("a" * (judge.MAX_CONTEXT_CHARS + 1), "is it good?")


def test_more_paraphrases_than_exist_is_refused():
    with pytest.raises(BadRequest, match="ways at most"):
        judge.plan("x", "y", n_paraphrases=99)


# ------------------------------------------------------------ the scoring


def test_gpt2_barely_commits_and_the_report_says_so(gpt2):
    """gpt2 is the control: a model that cannot judge.

    MEASURED, summing every single-token casing: it puts 6.4%-15.7% on a
    verdict token and splits that 0.473-0.555 whatever it is shown. It has
    "answered" — with a coin flip — so the mass floor does not and should not
    refuse it; refusing at 15% would be tuning a threshold to exclude one
    model. What the report must not do is print the ratio without the mass.
    """
    model, tok = gpt2
    out = judge.score(
        model, tok, "The cat sat on the mat.", "Does the text mention a cat?"
    )
    assert "of its mass on a verdict token" in out.means()
    assert "MOST OF THE WAY TO NOT ANSWERING" in out.means()


def test_gpt2_says_the_same_thing_whatever_it_is_shown(gpt2):
    """The evidence behind the refusal above: a judge whose answer does not
    move with the text is not answering."""
    import statistics

    model, tok = gpt2
    a = judge.score(
        model,
        tok,
        "The cat sat on the mat. The cat was black.",
        "Does the text above mention a cat?",
        min_mass=0.0,
    )
    b = judge.score(
        model,
        tok,
        "The stock market closed lower on Tuesday.",
        "Does the text above mention a cat?",
        min_mass=0.0,
    )
    gap = abs(statistics.median(a.scores) - statistics.median(b.scores))
    assert gap < 0.1, (
        f"gpt2 separated these by {gap:.3f}; if that is now a real separation "
        f"the floor and this test's premise both need re-measuring"
    )


@needs_judge_model
def test_the_same_prompt_gives_the_same_number_twice(judge_model):
    """Reading mass rather than sampling is the whole claim."""
    model, tok = judge_model
    a = judge.score(model, tok, "The cat sat.", "Is there an animal?")
    b = judge.score(model, tok, "The cat sat.", "Is there an animal?")
    assert a.scores == b.scores


@needs_judge_model
def test_p_yes_and_p_no_sum_to_one_within_the_verdict(judge_model):
    model, tok = judge_model
    out = judge.score(model, tok, "The cat sat.", "Is there an animal?")
    for p in out.passes:
        assert p.p_yes + p.p_no == pytest.approx(1.0, abs=1e-5)


@needs_judge_model
def test_the_raw_mass_is_carried_beside_the_ratio(judge_model):
    """ "It answered, and said yes" has to stay distinguishable from "it barely
    answered"."""
    model, tok = judge_model
    out = judge.score(model, tok, "The cat sat.", "Is there an animal?")
    assert all(0 < p.mass <= 1 for p in out.passes)
    assert any(p.mass < 1 for p in out.passes), "mass is not the ratio"


def test_the_refusal_names_the_mass_and_the_floor(gpt2):
    """Below the floor, the ratio between two rounding errors is noise — and
    the reader needs both numbers to judge the judgement."""
    model, tok = gpt2
    with pytest.raises(Refusal) as caught:
        judge.score(
            model,
            tok,
            "The cat sat.",
            "Is there an animal?",
            min_mass=0.999,
        )
    message = str(caught.value)
    assert "did not answer the rubric" in message
    assert "rounding errors is noise" in message


def test_the_floor_is_stated_in_the_refusal(gpt2):
    model, tok = gpt2
    with pytest.raises(Refusal, match="100% floor"):
        judge.score(model, tok, "x y z", "Is it?", min_mass=1.0)


# ----------------------------------- a weak judge is named, never aggregated


@needs_judge_model
def test_the_judge_model_is_named_on_every_score(judge_model):
    """A well-calibrated report of a weak judge's opinion is still a weak
    judge's opinion, and the name is what lets a reader weigh it."""
    model, tok = judge_model
    out = judge.score(model, tok, "The cat sat.", "Is there an animal?")
    assert out.judge_model == JUDGE_MODEL
    assert JUDGE_MODEL in out.means()
    assert "weak judge" in out.means()


@needs_judge_model
def test_the_dtype_and_device_travel_with_the_score(judge_model):
    model, tok = judge_model
    out = judge.score(model, tok, "The cat sat.", "Is there an animal?")
    assert out.dtype in ("float32", "bfloat16", "float16")
    assert out.device == "cpu"


@needs_judge_model
def test_the_spread_is_reported_not_only_the_median(judge_model):
    model, tok = judge_model
    doc = judge.score(model, tok, "The cat sat.", "Is there an animal?").to_dict()
    assert {"low", "median", "high", "spread", "n_paraphrases"} <= set(doc)
    assert doc["low"] <= doc["median"] <= doc["high"]


@needs_judge_model
def test_disagreeing_paraphrases_are_called_out(judge_model):
    """A single median from paraphrases that disagree is the 'sample presented
    as a property' error one level up."""
    model, tok = judge_model
    out = judge.score(model, tok, "The cat sat.", "Is there an animal?")
    # Force the condition rather than hunting for a rubric that triggers it.
    out.passes[0].p_yes = 0.05
    out.passes[-1].p_yes = 0.95
    assert "THE PARAPHRASES DISAGREE" in out.means()


def test_the_module_exposes_no_aggregate_across_rubrics():
    """That aggregate is exactly where a sample starts being treated as a
    property."""
    banned = [
        name
        for name in dir(judge)
        if any(w in name.lower() for w in ("aggregate", "overall", "project_score"))
    ]
    assert banned == []


@needs_judge_model
def test_it_runs_one_forward_pass_per_paraphrase_and_never_generates(judge_model):
    """A generation would cost k tokens per paraphrase and would be a sample.
    One pass is the feature."""
    model, tok = judge_model
    calls = {"forward": 0, "generate": 0}
    real_forward = model.forward

    def counted(*a, **k):
        calls["forward"] += 1
        return real_forward(*a, **k)

    model.forward = counted
    model.generate = lambda *a, **k: calls.__setitem__(
        "generate", calls["generate"] + 1
    )
    try:
        # min_mass=0.0: this counts passes, and gpt2 legitimately refuses at
        # the real floor — see the headline test above.
        judge.score(
            model,
            tok,
            "The cat sat.",
            "Is there an animal?",
            n_paraphrases=3,
        )
    finally:
        model.forward = real_forward
    assert calls["forward"] == 3
    assert calls["generate"] == 0


@needs_judge_model
def test_a_capable_model_separates_a_true_rubric_from_a_false_one(judge_model):
    """The test gpt2 could never support. MEASURED on Qwen2.5-0.5B-Instruct:
    p(yes) 0.990 for a text that mentions a cat against a much lower number
    for one that does not, on the same rubric."""
    import statistics

    model, tok = judge_model
    yes = judge.score(
        model,
        tok,
        "The cat sat on the mat. The cat was black.",
        "Does the text above mention a cat?",
    )
    no = judge.score(
        model,
        tok,
        "The stock market closed lower on Tuesday.",
        "Does the text above mention a cat?",
    )
    gap = statistics.median(yes.scores) - statistics.median(no.scores)
    assert gap > 0.2, f"separated by only {gap:.3f}: {yes.scores} vs {no.scores}"


@needs_judge_model
def test_one_unanswered_phrasing_does_not_destroy_the_whole_score(judge_model):
    """MEASURED: paraphrase 1 gave mass 0.72 and p(yes) 0.998 while paraphrase
    2 gave 0.02 — because the model answered THAT phrasing with capitalised
    " Yes". An earlier draft raised on the first pass below the floor and threw
    three good measurements away."""
    model, tok = judge_model
    out = judge.score(
        model, tok, "The cat sat on the mat.", "Does the text mention a cat?"
    )
    assert out.passes, "every paraphrase should be recorded"
    assert any(p.answered for p in out.passes)
    # Unanswered phrasings are carried, not dropped, and excluded from the
    # median rather than averaged in as noise.
    assert len(out.scores) == sum(1 for p in out.passes if p.answered)


@needs_judge_model
def test_an_unanswered_phrasing_is_named_in_the_summary(judge_model):
    """A median over 3 of 4 that reads as a median over 4 is the same omission
    this project keeps refusing."""
    model, tok = judge_model
    out = judge.score(
        model, tok, "The cat sat on the mat.", "Does the text mention a cat?"
    )
    out.passes[-1].answered = False
    assert "NOT in that median" in out.means()


@needs_judge_model
def test_the_refusal_is_for_a_model_that_answered_no_phrasing(judge_model):
    model, tok = judge_model
    with pytest.raises(Refusal, match="in any of"):
        judge.score(model, tok, "The cat sat.", "Is there an animal?", min_mass=1.0)
