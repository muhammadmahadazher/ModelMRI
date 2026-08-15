"""What constrained decoding cost, against a real model and a real grammar.

The headline test runs an ACTUAL constrained generation and asserts the output
parses as JSON matching the schema. A test that only checked the recorder's
bookkeeping would pass just as well with a mask that permitted everything,
which is the failure this whole feature exists to make visible.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("lmformatenforcer")

from modelmri import grammar  # noqa: E402
from modelmri.errors import BadRequest  # noqa: E402

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}


@pytest.fixture(scope="module")
def gpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    return model, tok


@pytest.fixture(scope="module")
def data(gpt2):
    _, tok = gpt2
    return grammar.tokenizer_data(tok)


# ------------------------------------------------- the mask is a real mask


def test_the_grammar_actually_constrains_a_generation(gpt2, data):
    """THE test. Everything else checks bookkeeping; this checks that the
    output is valid JSON matching the schema, which a mask that permitted
    everything could not produce."""
    model, tok = gpt2
    recorder = grammar.MaskRecorder(tok, SCHEMA, data=data)
    ids = tok("Describe a person as JSON:", return_tensors="pt").input_ids
    out = model.generate(
        ids,
        max_new_tokens=24,
        do_sample=False,
        logits_processor=[recorder],
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
    parsed = json.loads(text)
    assert set(parsed) == {"name", "age"}
    assert isinstance(parsed["name"], str)
    assert isinstance(parsed["age"], int)


def test_an_unconstrained_generation_does_not_produce_that_json(gpt2):
    """The control. If gpt2 emitted schema-shaped JSON on its own, the test
    above would prove nothing about the mask."""
    model, tok = gpt2
    ids = tok("Describe a person as JSON:", return_tensors="pt").input_ids
    out = model.generate(
        ids, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id
    )
    text = tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)


# ------------------------------------------------------ what it cost


def test_the_structural_positions_mask_nearly_everything(gpt2, data):
    """MEASURED on gpt2: 9 of 50,257 tokens legal before anything is written
    (whitespace plus the brace variants) and 5 at the key position, against
    50,169 inside a free string value. That contrast is the finding, and it is
    invisible from the completion alone."""
    model, tok = gpt2
    recorder = grammar.MaskRecorder(tok, SCHEMA, data=data)
    ids = tok("Describe a person as JSON:", return_tensors="pt").input_ids
    model.generate(
        ids,
        max_new_tokens=24,
        do_sample=False,
        logits_processor=[recorder],
        pad_token_id=tok.eos_token_id,
    )
    steps = recorder.trace.steps
    assert steps, "nothing was recorded"
    assert steps[0].allowed_count <= 12, (
        "the opening position should permit almost nothing"
    )
    assert steps[0].masked_fraction > 0.999
    # Somewhere structural the set narrows further than the opening does.
    assert min(s.allowed_count for s in steps) <= 7
    # Somewhere in the middle a free string value opens the mask right up.
    assert max(s.allowed_count for s in steps) > 1000, (
        "no step permitted a wide set, so this schema never reached a free "
        "string and the contrast is untested"
    )


def test_the_deleted_mass_is_recorded_per_step(gpt2, data):
    model, tok = gpt2
    recorder = grammar.MaskRecorder(tok, SCHEMA, data=data)
    ids = tok("Describe a person as JSON:", return_tensors="pt").input_ids
    model.generate(
        ids,
        max_new_tokens=16,
        do_sample=False,
        logits_processor=[recorder],
        pad_token_id=tok.eos_token_id,
    )
    for step in recorder.trace.steps:
        assert 0.0 <= step.deleted_mass <= 1.0
    assert any(s.deleted_mass > 0.5 for s in recorder.trace.steps), (
        "no step deleted meaningful mass, which would mean the mask did nothing"
    )


def test_the_token_the_model_wanted_is_recorded_even_when_forbidden(gpt2, data):
    """This flag is the diagnostic: it marks where the output stopped being
    the model's answer."""
    model, tok = gpt2
    recorder = grammar.MaskRecorder(tok, SCHEMA, data=data)
    ids = tok("Describe a person as JSON:", return_tensors="pt").input_ids
    model.generate(
        ids,
        max_new_tokens=16,
        do_sample=False,
        logits_processor=[recorder],
        pad_token_id=tok.eos_token_id,
    )
    trace = recorder.trace
    assert trace.overridden, "gpt2 unprompted wants none of this schema's tokens"
    assert "WAS FORBIDDEN" in trace.means()
    for step in trace.overridden:
        assert step.wanted, "the forbidden token must be named"
        assert step.wanted_p > 0


def test_a_forbidden_token_is_masked_to_negative_infinity(gpt2, data):
    """A finite penalty leaves a forbidden token reachable under sampling, and
    the promise of constrained decoding is that it is not."""
    _, tok = gpt2
    recorder = grammar.MaskRecorder(tok, SCHEMA, data=data)
    scores = torch.zeros(1, len(tok.get_vocab()))
    ids = torch.zeros(1, 3, dtype=torch.long)
    out = recorder(ids, scores)
    allowed = recorder.enforcer.get_allowed_tokens([]).allowed_tokens
    blocked = [i for i in range(200) if i not in set(allowed)]
    assert blocked, "nothing to check"
    assert torch.isinf(out[0][blocked]).all()
    assert (out[0][blocked] < 0).all()


# --------------------------------------------------------------- the plan


def test_the_plan_answers_before_anything_is_generated(gpt2):
    _, tok = gpt2
    out = grammar.plan(SCHEMA, tok)
    # 9 on gpt2: the brace, its word-start variants, and the whitespace JSON
    # legally allows before it. Bounded rather than pinned, so a tokenizer with
    # a different whitespace inventory does not fail this.
    assert 1 <= out["allowed_at_start"] <= 12
    assert out["masked_fraction"] > 0.999
    assert out["examples"], "the reader should see what is legal"
    assert "memoised per prefix" in out["note"]


# ------------------------------------------------------------- refusals


def test_a_schema_that_is_not_an_object_is_refused(gpt2):
    _, tok = gpt2
    with pytest.raises(BadRequest, match="a JSON schema is an object"):
        grammar.MaskRecorder(tok, "not a schema")


def test_an_uncompilable_schema_is_refused_without_the_librarys_words(gpt2):
    """This authors its own sentence and keeps theirs on the traceback."""
    _, tok = gpt2
    with pytest.raises(BadRequest) as caught:
        grammar.MaskRecorder(tok, {"type": "not-a-real-type"})
    assert "could not be compiled into a grammar" in str(caught.value)


def test_a_tokenizer_with_no_eos_is_refused(gpt2):
    """Without one the grammar cannot know a value is finished."""
    _, tok = gpt2

    class NoEos:
        eos_token_id = None

        def get_vocab(self):
            return tok.get_vocab()

        def convert_ids_to_tokens(self, i):
            return tok.convert_ids_to_tokens(i)

        def decode(self, ids):
            return tok.decode(ids)

    with pytest.raises(BadRequest, match="no end-of-sequence token"):
        grammar.tokenizer_data(NoEos())


# ------------------------------------------------------------- the record


def test_the_chosen_token_is_absent_rather_than_zero_until_recorded(gpt2, data):
    """A step whose choice was never recorded must not read as token 0."""
    _, tok = gpt2
    recorder = grammar.MaskRecorder(tok, SCHEMA, data=data)
    recorder(torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, len(tok.get_vocab())))
    assert recorder.trace.steps[0].chosen_id == -1
    recorder.record_choice(90)
    assert recorder.trace.steps[0].chosen_id == 90


def test_the_step_cap_is_reported_not_silent(gpt2, data, monkeypatch):
    _, tok = gpt2
    monkeypatch.setattr(grammar, "MAX_STEPS", 2)
    recorder = grammar.MaskRecorder(tok, SCHEMA, data=data)
    for _ in range(5):
        recorder(
            torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, len(tok.get_vocab()))
        )
    assert len(recorder.trace.steps) == 2
    assert recorder.trace.truncated == 3
    assert "were NOT recorded" in recorder.trace.means()


def test_the_trace_serialises_for_the_wire(gpt2, data):
    _, tok = gpt2
    recorder = grammar.MaskRecorder(tok, SCHEMA, data=data, temperature=0.7)
    recorder(torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, len(tok.get_vocab())))
    doc = recorder.trace.to_dict()
    assert doc["temperature"] == 0.7
    assert doc["steps"][0]["masked_fraction"] > 0.99
    assert isinstance(doc["means"], str)
