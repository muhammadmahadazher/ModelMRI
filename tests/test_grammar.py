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
from modelmri.errors import BadRequest, Refusal  # noqa: E402

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
    """This authors its own sentence and keeps theirs on the traceback.

    `errors.py` forbids interpolating a caught exception's text into a
    published sentence — it is machinery talking to itself and carries
    whatever the library felt like carrying. So the offending value is read
    back out of the CALLER'S OWN schema instead, which says strictly more than
    the library's `Unsupported type not-a-real-type` would have.
    """
    _, tok = gpt2
    with pytest.raises(BadRequest) as caught:
        grammar.MaskRecorder(tok, {"type": "not-a-real-type"})
    message = str(caught.value)
    assert "cannot be compiled into a token-level grammar" in message
    assert "'not-a-real-type'" in message
    assert "Unsupported type" not in message, "the library's own words leaked"


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


# ---------------------------- the cap must not corrupt the last row


class _Tok:
    def decode(self, ids):
        return f"T{ids[0]}"


def _recorder_at_cap(cap: int, total: int):
    """A recorder driven past `cap` recorded steps, without a model."""
    rec = grammar.MaskRecorder.__new__(grammar.MaskRecorder)
    rec.tokenizer = _Tok()
    rec.trace = grammar.Trace()
    rec._steps_seen = 0
    for i in range(total):
        rec._steps_seen += 1
        if len(rec.trace.steps) < cap:
            rec.trace.steps.append(
                grammar.Step(step=i, top=[{"token": f"T{100 + i}", "p": 0.5}])
            )
        rec.record_choice(100 + i)
    return rec


def test_a_token_past_the_step_cap_does_not_overwrite_the_last_row():
    """`__call__` stops appending at MAX_STEPS and `record_choice` went on
    writing to `steps[-1]`, so every token past the cap overwrote the last
    recorded step's choice.

    Step 1,999 ended up showing the token emitted at step 5,000, beside its
    own `wanted` and its own `top` — one row describing two different moments,
    which is worse than a row that is simply absent.
    """
    rec = _recorder_at_cap(cap=3, total=8)
    last = rec.trace.steps[-1]

    assert len(rec.trace.steps) == 3
    assert last.step == 2
    assert last.chosen == "T102", "the last row took a later step's token"
    assert last.chosen_id == 102


def test_every_recorded_step_keeps_its_own_choice():
    rec = _recorder_at_cap(cap=4, total=4)
    for i, step in enumerate(rec.trace.steps):
        assert step.chosen == f"T{100 + i}"


def test_a_chosen_token_outside_the_recorded_top_has_no_probability_not_zero():
    """`chosen_p` defaulted to 0.0 and is only set when the emitted token
    appears in `top`, which holds TOP_K rows. A token the grammar forced
    through from outside that handful therefore read as "the model gave this
    no probability at all" — the opposite of the truth, and unrecoverable
    from the record."""
    rec = grammar.MaskRecorder.__new__(grammar.MaskRecorder)
    rec.tokenizer = _Tok()
    rec.trace = grammar.Trace()
    rec._steps_seen = 1
    rec.trace.steps.append(
        grammar.Step(step=0, top=[{"token": "something-else", "p": 0.9}])
    )
    rec.record_choice(42)

    step = rec.trace.steps[0]
    assert step.chosen == "T42"
    assert step.chosen_p is None, "reported an unrecorded probability as zero"
    assert step.to_dict()["chosen_p"] is None


def test_a_chosen_token_inside_the_top_keeps_its_measured_probability():
    rec = grammar.MaskRecorder.__new__(grammar.MaskRecorder)
    rec.tokenizer = _Tok()
    rec.trace = grammar.Trace()
    rec._steps_seen = 1
    rec.trace.steps.append(grammar.Step(step=0, top=[{"token": "T42", "p": 0.37}]))
    rec.record_choice(42)
    assert rec.trace.steps[0].chosen_p == 0.37


# ------------------------------ the schema is compiled before anything runs


def test_a_nested_unsupported_type_is_refused_before_generation():
    """THE pre-flight test, and the failure it prevents is not an exception.

    `JsonSchemaParser` compiles only the ROOT eagerly. MEASURED against the
    installed enforcer: `{"a": {"type": "widget"}}` constructs with no error,
    and then at decode time the allowed set collapses to {EOS} the moment the
    model emits the `:` after the key — the client gets HTTP 200 and the body
    `{"a"`, truncated JSON built from a mask nobody checked.
    """
    schema = {"type": "object", "properties": {"a": {"type": "widget"}}}
    # The root alone compiles, which is the whole trap.
    grammar._enforcer_module()[0](schema)
    with pytest.raises(BadRequest) as caught:
        grammar.validate_schema(schema)
    message = str(caught.value)
    assert "'/properties/a'" in message
    assert "'widget'" in message
    assert "Nothing was generated" in message


def test_a_nested_pattern_with_a_length_bound_is_refused_before_generation():
    """The other half of the same trap: this one raises MID-GENERATION, on
    `generate`'s worker thread, where the exception dies unheard and the
    consumer waits out the streamer's timeout instead."""
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string", "pattern": "[a-z]+", "minLength": 2}},
    }
    with pytest.raises(BadRequest, match="'pattern' and a length bound"):
        grammar.validate_schema(schema)


def test_a_dangling_ref_names_the_target_and_what_is_defined():
    schema = {
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/Nope"}},
        "$defs": {"Thing": {"type": "string"}},
    }
    with pytest.raises(BadRequest) as caught:
        grammar.validate_schema(schema)
    message = str(caught.value)
    assert "#/$defs/Nope" in message
    assert "Thing" in message


def test_a_schema_bad_at_its_root_says_root_rather_than_a_pointer():
    with pytest.raises(BadRequest, match="at its root"):
        grammar.validate_schema({"type": "widget"})


def test_a_bad_member_of_an_anyof_is_located_even_though_the_root_failed():
    """`JsonSchemaParser` refuses the whole schema when an `anyOf` member is
    bad, so the pointer has to come from a second walk against a locator
    root — otherwise the answer is "somewhere in your schema"."""
    with pytest.raises(BadRequest) as caught:
        grammar.validate_schema({"anyOf": [{"type": "string"}, {"type": "widget"}]})
    assert "'/anyOf/1'" in str(caught.value)


def test_a_recursive_ref_does_not_hang_the_walk():
    """A `$defs` entry that references itself is ordinary and must not send
    the walk round forever."""
    grammar.validate_schema(
        {
            "type": "object",
            "properties": {"child": {"$ref": "#/$defs/Node"}},
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {"child": {"$ref": "#/$defs/Node"}},
                }
            },
        }
    )


def test_an_empty_schema_is_refused_for_what_it_actually_means():
    """`{}` is not "an object with no constraints" — to the compiler it is
    "any JSON value at all", so a bare string satisfies it."""
    with pytest.raises(BadRequest, match="any JSON value"):
        grammar.validate_schema({})


def test_a_valid_schema_passes_the_walk_untouched():
    grammar.validate_schema(SCHEMA)
    grammar.validate_schema(grammar.ANY_JSON_OBJECT)


def test_a_broken_preflight_refuses_instead_of_blaming_the_schema(monkeypatch):
    """If the enforcer moves the compiler this borrows, the answer is a
    refusal naming the version — never a valid schema reported as invalid."""
    monkeypatch.setattr(grammar, "_PREFLIGHT", None)
    real = grammar._enforcer_module()

    def blind_parser(schema):
        """A compiler that says yes to everything, which is the dangerous
        direction: a positive-only self-test would wave this through."""

        class _Blind:
            pass

        return _Blind()

    monkeypatch.setattr(
        grammar, "_enforcer_module", lambda: (blind_parser, real[1], real[2])
    )
    monkeypatch.setattr(
        "lmformatenforcer.jsonschemaparser.get_parser", lambda root, node: object()
    )
    with pytest.raises(Refusal, match="cannot check a JSON schema"):
        grammar.validate_schema(SCHEMA)


# --------------------------------- what the mask is measured against


def test_the_deleted_mass_is_measured_at_the_temperature_that_was_running(gpt2, data):
    """`generate` appends its temperature warper AFTER any custom processor,
    so a recorder that left temperature to transformers would report the mass
    removed from a temperature-1 distribution the model never had — while the
    module docstring claimed the opposite. The recorder owns the division."""
    _, tok = gpt2
    scores = torch.randn(1, len(tok.get_vocab()))
    ids = torch.zeros(1, 3, dtype=torch.long)

    cold = grammar.MaskRecorder(tok, SCHEMA, data=data, temperature=0.1)
    hot = grammar.MaskRecorder(tok, SCHEMA, data=data, temperature=2.0)
    cold(ids, scores.clone())
    hot(ids, scores.clone())

    assert cold.trace.steps[0].deleted_mass != hot.trace.steps[0].deleted_mass, (
        "temperature changed nothing, so the recorder is not applying it"
    )
    # And the returned logits are the SCALED ones, or transformers would be
    # sampling from a distribution the receipt does not describe.
    out = hot(ids, scores.clone())
    finite = torch.isfinite(out[0])
    assert torch.allclose(out[0][finite], (scores / 2.0)[0][finite])


def test_a_greedy_recorder_scales_nothing(gpt2, data):
    """temperature 0 is greedy decoding, where there is no warper to stand in
    for and dividing by zero is not a thing to do."""
    _, tok = gpt2
    scores = torch.randn(1, len(tok.get_vocab()))
    rec = grammar.MaskRecorder(tok, SCHEMA, data=data)
    out = rec(torch.zeros(1, 3, dtype=torch.long), scores.clone())
    finite = torch.isfinite(out[0])
    assert torch.allclose(out[0][finite], scores[0][finite])


def test_an_allowed_id_past_the_logits_width_is_dropped_not_raised(gpt2, data):
    """A tokenizer longer than the head that has to produce them is ordinary.
    `keep[ids] = True` with an id past the end raises IndexError inside
    `generate`'s worker thread, where nothing catches it."""
    _, tok = gpt2
    rec = grammar.MaskRecorder(tok, SCHEMA, data=data)
    narrow = torch.zeros(1, 64)
    out = rec(torch.zeros(1, 3, dtype=torch.long), narrow)
    assert out.shape == narrow.shape
    assert rec.trace.steps[0].vocab_size == 64


# ------------------------------------- the choice tap drives the record


def test_the_choice_tap_records_what_generate_actually_emitted(gpt2, data):
    """A logits processor runs BEFORE the token is sampled and never learns
    what was chosen. Without the tap every `chosen` is empty and the trace is
    half a measurement."""
    model, tok = gpt2
    from transformers import LogitsProcessorList, StoppingCriteriaList

    recorder = grammar.MaskRecorder(tok, SCHEMA, data=data)
    ids = tok("Describe a person as JSON:", return_tensors="pt").input_ids
    out = model.generate(
        ids,
        max_new_tokens=16,
        do_sample=False,
        logits_processor=LogitsProcessorList([recorder]),
        stopping_criteria=StoppingCriteriaList([grammar.ChoiceTap(recorder)]),
        pad_token_id=tok.eos_token_id,
    )
    emitted = [int(t) for t in out[0][ids.shape[1] :]]
    steps = recorder.trace.steps
    assert steps, "nothing was recorded"
    assert [s.chosen_id for s in steps] == emitted[: len(steps)]
    for step in steps:
        assert step.chosen == tok.decode([step.chosen_id])


def test_the_choice_tap_never_stops_a_generation(gpt2, data):
    """`StoppingCriteriaList` ORs the results, so one that always says False
    is inert — a tap that ended the generation would be a mask that shortened
    the answer."""
    _, tok = gpt2
    recorder = grammar.MaskRecorder(tok, SCHEMA, data=data)
    tap = grammar.ChoiceTap(recorder)
    recorder(torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, len(tok.get_vocab())))
    verdict = tap(torch.zeros(1, 4, dtype=torch.long), None)
    assert verdict.shape == (1,)
    assert not bool(verdict.any())


def test_a_step_where_only_end_of_sequence_survives_is_flagged(gpt2, data):
    """The runtime signature of a parser collapse. `allowed_count == 1` cannot
    say it on its own: a forced closing brace is also one token, and is
    exactly what should happen."""
    _, tok = gpt2
    recorder = grammar.MaskRecorder(tok, SCHEMA, data=data)

    class _OnlyEos:
        allowed_tokens = [tok.eos_token_id]

    recorder.enforcer = type(
        "E", (), {"get_allowed_tokens": lambda self, ids: _OnlyEos()}
    )()
    recorder(torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, len(tok.get_vocab())))
    assert recorder.trace.steps[0].eos_only is True
    assert recorder.trace.steps[0].to_dict()["eos_only"] is True


# ------------------------ a limit the caller never wrote, said out loud


def test_an_array_with_no_maxitems_is_told_about_the_cap(gpt2, data):
    """The compiler caps an unbounded array, and past the cap the mask stops
    permitting another element — so the array closes and the completion looks
    finished. Nothing in the output separates "the model was done" from
    "somebody else's default ran out"."""
    _, tok = gpt2
    schema = {
        "type": "object",
        "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
    }
    rec = grammar.MaskRecorder(tok, schema, data=data)
    note = " ".join(rec.trace.notes)
    assert "/properties/tags" in note
    assert "maxItems" in note
    # The number is READ from the installed config, not remembered: the
    # enforcer takes `LMFE_MAX_JSON_ARRAY_LENGTH` from the environment, so a
    # machine that moved it must see its own number here.
    assert str(grammar._array_cap()) in note
    assert note in rec.trace.means()
    assert rec.trace.to_dict()["notes"] == rec.trace.notes


def test_an_array_the_schema_bounds_itself_gets_no_note(gpt2, data):
    _, tok = gpt2
    schema = {
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3}
        },
    }
    assert grammar.MaskRecorder(tok, schema, data=data).trace.notes == []


def test_a_schema_with_no_arrays_at_all_gets_no_note(gpt2, data):
    _, tok = gpt2
    assert grammar.MaskRecorder(tok, SCHEMA, data=data).trace.notes == []


def test_the_array_cap_is_read_from_the_installed_config(monkeypatch):
    """Not a constant in this file. The enforcer reads the environment when it
    builds its config, so a machine that raised the cap must be told its own
    number rather than the library's default."""
    monkeypatch.setenv("LMFE_MAX_JSON_ARRAY_LENGTH", "137")
    assert grammar._array_cap() == 137


def test_an_ordinary_narrow_step_is_not_flagged_as_a_collapse(gpt2, data):
    _, tok = gpt2
    recorder = grammar.MaskRecorder(tok, SCHEMA, data=data)
    recorder(torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, len(tok.get_vocab())))
    assert recorder.trace.steps[0].eos_only is False


# ------------------------- a bare boolean where a schema goes

#: One character per token, so a prefix can be spelled out directly.
_ALPHABET = ' {}"[],:0123456789abcdefghijklmnopqrstuvwxyz'
_EOS = "<eos>"


def _allowed_after(schema: dict, text: str) -> list:
    """What the REAL enforcer permits after `text`, over a toy vocabulary.

    A tokenizer is not needed to watch a parser collapse, and building one
    costs the 50k-id walk in `tokenizer_data` for nothing. The enforcer itself
    is the real one, and that half is not optional: the collapse demonstrated
    below happens inside `TokenEnforcer._compute_allowed_tokens`'s own blanket
    `except Exception`, so a stand-in would demonstrate nothing.

    Every position is asked about in order, because the enforcer memoises per
    prefix and answers the ROOT state's question for one it never walked.
    """
    JsonSchemaParser, TokenEnforcer, TokenEnforcerTokenizerData = (
        grammar._enforcer_module()
    )
    eos_id = len(_ALPHABET)
    data = TokenEnforcerTokenizerData(
        [(i, c, False) for i, c in enumerate(_ALPHABET)],
        lambda ids: "".join(_ALPHABET[i] for i in ids if i < len(_ALPHABET)),
        eos_id,
        False,
        eos_id + 1,
    )
    enforcer = TokenEnforcer(data, JsonSchemaParser(schema))
    ids: list = []
    for char in text:
        enforcer.get_allowed_tokens(ids)
        ids.append(_ALPHABET.index(char))
    return [
        _EOS if i == eos_id else _ALPHABET[i]
        for i in enforcer.get_allowed_tokens(ids).allowed_tokens
    ]


#: The four positions where a bare `true` reaches the compiler, and the prefix
#: at which the mask dies. Every one of these passed `validate_schema` clean.
BOOLEAN_SUBSCHEMAS = {
    "/additionalProperties": ({"type": "object", "additionalProperties": True}, '{"a"'),
    "/items": ({"type": "array", "items": True}, ""),
    "/properties/a": ({"type": "object", "properties": {"a": True}}, '{"a"'),
    "/$defs/B": (
        {
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/B"}},
            "$defs": {"B": True},
        },
        '{"a"',
    ),
}


@pytest.mark.parametrize("pointer", sorted(BOOLEAN_SUBSCHEMAS))
def test_a_bare_boolean_collapses_the_real_mask_to_end_of_sequence(pointer):
    """The trap, shown before the refusal that closes it.

    JSON Schema draft-6 and later allow a bare boolean where a schema goes, and
    `lm-format-enforcer`'s object model accepts one -- `items` and
    `additionalProperties` are typed to include `bool` and every `properties`
    value is `Union[JsonSchemaObject, bool]`. Its COMPILER reads `.anyOf` off
    whatever it is handed, so the boolean arrives as an `AttributeError` inside
    a blanket `except Exception` and the allowed set becomes {EOS}.

    The root compiles without complaint, which is the whole trap: nothing
    raises, nothing warns, and the completion comes back as a 200 carrying a
    four-character fragment.
    """
    schema, prefix = BOOLEAN_SUBSCHEMAS[pointer]
    grammar._enforcer_module()[0](schema)  # the root alone is fine
    assert _allowed_after(schema, prefix) == [_EOS]


def test_the_same_schema_without_the_boolean_permits_a_real_set():
    """The control. If the toy vocabulary collapsed everything, the four
    parametrised cases above would prove nothing about booleans."""
    assert len(_allowed_after({"type": "object"}, '{"a"')) > 1


@pytest.mark.parametrize("pointer", sorted(BOOLEAN_SUBSCHEMAS))
def test_a_bare_boolean_sub_schema_is_refused_before_generation(pointer):
    """`_subschemas` yields a child only when it is a dict, so none of these
    was ever handed to `get_parser` and `validate_schema` returned clean on a
    schema that then collapsed mid-object behind a 200 -- the exact failure its
    own docstring says it makes unreachable."""
    schema, _ = BOOLEAN_SUBSCHEMAS[pointer]
    with pytest.raises(BadRequest) as caught:
        grammar.validate_schema(schema)
    message = str(caught.value)
    assert repr(pointer) in message
    assert "`true`" in message
    assert "Nothing was generated" in message


@pytest.mark.parametrize(
    "schema",
    [
        # OpenAI strict mode's own shape. `ObjectParsingState.is_dictionary` is
        # `properties is None`, so with `properties` present the keyword is
        # never read at all and only declared keys are reachable.
        {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": True,
        },
        # Falsy, so it never reaches `get_parser` and nothing collapses. What
        # it does instead is disclosed below, not refused.
        {"type": "object", "additionalProperties": False},
    ],
)
def test_the_shapes_the_boolean_check_must_not_refuse(schema):
    """A blanket "no non-dict child" would turn the commonest strict-mode
    schema in the world into a 400."""
    grammar.validate_schema(schema)


def test_a_property_whose_value_is_a_string_is_located_not_blamed_on_the_root():
    """It reported `at its root` for a fault at `/properties/a`: an authored
    sentence naming the wrong place, which is worse than a vague one."""
    with pytest.raises(BadRequest) as caught:
        grammar.validate_schema({"type": "object", "properties": {"a": "string"}})
    message = str(caught.value)
    assert "'/properties/a'" in message
    assert "at its root" not in message


def test_a_fault_the_diagnosis_cannot_name_still_gets_a_true_sentence():
    """`_fault`'s last branch is the "do not be confidently wrong" one: when no
    specific diagnosis matches, the caller gets a vague-but-true sentence
    naming where the compiler stopped rather than a confident wrong reason."""
    with pytest.raises(BadRequest) as caught:
        grammar.validate_schema(
            {
                "type": "object",
                "properties": {"a": {"type": "string", "minLength": "two"}},
            }
        )
    message = str(caught.value)
    assert "'/properties/a'" in message
    assert "the grammar compiler would not accept it" in message
    for kind in grammar.COMPILABLE_TYPES:
        assert kind in message


# ------------- a constraint the caller wrote and the compiler discards


BOUNDED = {
    "type": "object",
    "properties": {"age": {"type": "integer", "minimum": 0, "maximum": 5}},
    "required": ["age"],
}


def test_a_bound_the_compiler_never_reads_really_is_not_enforced():
    """The measurement the disclosure is written from, so the note cannot
    become a claim nobody checked.

    `minimum` and `maximum` appear nowhere in `jsonschemaparser.py`. After
    `{"age":3` the mask still permits every digit, so `30` satisfies a schema
    that said 5 -- and the completion parses, which is what makes this worse
    than a schema that refuses.
    """
    assert "0" in _allowed_after(BOUNDED, '{"age":3'), (
        "'maximum': 5 was enforced after all, and the note would be a lie"
    )


def test_a_dropped_bound_is_named_in_the_receipt(gpt2, data):
    """The mirror of the array-cap note and the more misleading direction: that
    one discloses a limit the compiler ADDED, this one a limit the caller wrote
    and the compiler threw away, which the caller has every reason to believe
    was applied."""
    _, tok = gpt2
    rec = grammar.MaskRecorder(tok, BOUNDED, data=data)
    note = " ".join(rec.trace.notes)
    assert "/properties/age" in note
    assert "'maximum'" in note and "'minimum'" in note
    assert note in rec.trace.means()


def test_a_schema_the_compiler_reads_whole_gets_no_dropped_note(gpt2, data):
    _, tok = gpt2
    assert grammar.MaskRecorder(tok, SCHEMA, data=data).trace.notes == []


def test_pattern_properties_is_disclosed_rather_than_compiled(gpt2, data):
    """`patternProperties` appears nowhere in the compiler, so a sub-schema
    under it is never built. Walking into it asserted a support that does not
    exist and turned a schema which decodes fine into a 400."""
    _, tok = gpt2
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "patternProperties": {"^x": {"type": "widget"}},
    }
    rec = grammar.MaskRecorder(tok, schema, data=data)
    assert "patternProperties" in " ".join(rec.trace.notes)


def test_additional_properties_false_with_no_properties_is_disclosed(gpt2, data):
    """MEASURED against the installed 0.11.3: the compiler tests the keyword
    for TRUTH, so `false` falls through to its any-JSON default. The caller
    wrote "no properties at all" and the mask permits any key and any value --
    the exact inversion of what was asked for."""
    _, tok = gpt2
    schema = {"type": "object", "additionalProperties": False}
    assert '"' in _allowed_after(schema, "{"), "a free key was not permitted"
    note = " ".join(grammar.MaskRecorder(tok, schema, data=data).trace.notes)
    assert "additionalProperties" in note
    assert "any key, any value" in note


def test_additional_properties_false_beside_properties_gets_no_note(gpt2, data):
    """The strict-mode shape, where the compiler's behaviour and the schema's
    meaning coincide and a note would be noise."""
    _, tok = gpt2
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "additionalProperties": False,
    }
    assert grammar.MaskRecorder(tok, schema, data=data).trace.notes == []


def test_a_root_level_array_with_no_maxitems_is_told_about_the_cap(gpt2, data):
    """`_unbounded_arrays` has a distinct branch for the root, producing a
    distinct sentence, and only the nested branch was exercised."""
    _, tok = gpt2
    rec = grammar.MaskRecorder(
        tok, {"type": "array", "items": {"type": "string"}}, data=data
    )
    note = " ".join(rec.trace.notes)
    assert "the schema's own root" in note
    assert str(grammar._array_cap()) in note


# --------------------------- the receipt describes the run that happened


def test_the_receipt_has_one_step_per_generated_token(gpt2, data):
    """`means()` opens with "N step(s) under this schema", and nothing checked
    that N was the number of tokens generated. A recorder that logged the first
    three of sixteen satisfied every other assertion in this file."""
    model, tok = gpt2
    from transformers import LogitsProcessorList, StoppingCriteriaList

    recorder = grammar.MaskRecorder(tok, SCHEMA, data=data)
    ids = tok("Describe a person as JSON:", return_tensors="pt").input_ids
    out = model.generate(
        ids,
        max_new_tokens=16,
        do_sample=False,
        logits_processor=LogitsProcessorList([recorder]),
        stopping_criteria=StoppingCriteriaList([grammar.ChoiceTap(recorder)]),
        pad_token_id=tok.eos_token_id,
    )
    generated = int(out.shape[1]) - int(ids.shape[1])
    assert generated > 0
    assert len(recorder.trace.steps) == generated
    assert recorder.trace.truncated == 0
    assert f"{generated} step(s) under this schema" in recorder.trace.means()


def test_the_clamp_drops_only_the_ids_the_head_cannot_produce(gpt2, data):
    """`test_an_allowed_id_past_the_logits_width_is_dropped_not_raised` asserts
    a shape, and a clamp that dropped EVERY allowed id passes it -- while
    producing an all-`-inf` row whose softmax is NaN."""
    _, tok = gpt2
    rec = grammar.MaskRecorder(tok, SCHEMA, data=data)

    class _Mixed:
        allowed_tokens = [5, 63, 64, 1000]

    rec.enforcer = type("E", (), {"get_allowed_tokens": lambda self, ids: _Mixed()})()
    narrow = torch.zeros(1, 64)
    out = rec(torch.zeros(1, 3, dtype=torch.long), narrow)

    assert rec.trace.steps[0].allowed_count == 2, "5 and 63 fit in a 64-wide row"
    assert rec.trace.steps[0].vocab_size == 64
    assert bool(torch.isfinite(out[0][[5, 63]]).all())
    assert bool(torch.isinf(out[0][[0, 62]]).all())


def test_a_step_with_no_allowed_token_at_all_is_reported_as_a_collapse(gpt2, data):
    """Strictly worse than the {EOS} collapse `eos_only` exists to flag, and
    `eos_only` cannot say it: an empty set is not the EOS set, so the flag is
    False, the row reads as an ordinary narrow step, and the logits are all
    `-inf` -- a softmax of NaN inside `generate`."""
    _, tok = gpt2
    recorder = grammar.MaskRecorder(tok, SCHEMA, data=data)

    class _Nothing:
        allowed_tokens = []

    recorder.enforcer = type(
        "E", (), {"get_allowed_tokens": lambda self, ids: _Nothing()}
    )()
    recorder(torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, len(tok.get_vocab())))

    step = recorder.trace.steps[0]
    assert step.allowed_count == 0
    assert step.eos_only is False, "an empty set is not the end-of-sequence set"
    assert recorder.trace.collapsed == [step]
    assert "no token at all" in recorder.trace.means()
    assert recorder.trace.to_dict()["n_collapsed"] == 1


def test_an_eos_only_step_is_said_out_loud_in_the_sentence(gpt2, data):
    """`eos_only` was written by the recorder and read by nothing: no sentence,
    no status, no verdict. A completion that ended because the mask ran out
    came back as an ordinary answer that happened not to parse."""
    _, tok = gpt2
    recorder = grammar.MaskRecorder(tok, SCHEMA, data=data)

    class _OnlyEos:
        allowed_tokens = [tok.eos_token_id]

    recorder.enforcer = type(
        "E", (), {"get_allowed_tokens": lambda self, ids: _OnlyEos()}
    )()
    recorder(torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, len(tok.get_vocab())))

    means = recorder.trace.means()
    assert "only end-of-sequence" in means
    assert "nowhere left to go" in means
    assert recorder.trace.to_dict()["n_collapsed"] == 1


def test_an_ordinary_run_says_nothing_about_a_collapse(gpt2, data):
    _, tok = gpt2
    recorder = grammar.MaskRecorder(tok, SCHEMA, data=data)
    recorder(torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, len(tok.get_vocab())))
    assert recorder.trace.collapsed == []
    assert "nowhere left to go" not in recorder.trace.means()
    assert recorder.trace.to_dict()["n_collapsed"] == 0


# ------------------------------ the receipt's own schema field


def _wordy_schema(fields: int = 60) -> dict:
    return {
        "type": "object",
        "properties": {
            f"field_{i}": {
                "type": "string",
                "title": f"a reasonably wordy description of field number {i}",
            }
            for i in range(fields)
        },
    }


def test_a_schema_longer_than_the_receipt_says_the_field_was_cut(gpt2, data):
    """The cut is old; publishing it is not. Nothing put `to_dict()` on the
    wire until `/v1` started returning this receipt, and a field named `schema`
    holding a string that is not the schema and does not parse is an unknown
    rendered as a value."""
    _, tok = gpt2
    doc = grammar.MaskRecorder(tok, _wordy_schema(), data=data).trace.to_dict()

    assert doc["schema_truncated"] is True
    assert doc["schema_bytes"] > grammar.MAX_SCHEMA_CHARS
    assert len(doc["schema"]) == grammar.MAX_SCHEMA_CHARS
    with pytest.raises(ValueError):
        json.loads(doc["schema"])
    assert "IS CUT" in doc["means"]
    assert str(doc["schema_bytes"]) in doc["means"]


def test_a_schema_that_fits_says_nothing_about_being_cut(gpt2, data):
    _, tok = gpt2
    doc = grammar.MaskRecorder(tok, SCHEMA, data=data).trace.to_dict()
    assert doc["schema_truncated"] is False
    assert json.loads(doc["schema"]) == SCHEMA
    assert "IS CUT" not in doc["means"]


# ---------------------------------- the refusal sentences themselves


def test_the_missing_extra_refusal_names_the_extra_and_why_it_is_optional():
    """The route test for this stubs the sentence it then asserts on, so it
    proves the plumbing and nothing about these words. Reword the real one to
    "constrained decoding is unavailable on this build" and the whole suite
    stayed green while "refused by name" stopped being true.

    `sys.modules[name] = None` is what makes `import name` raise ImportError
    without uninstalling anything.
    """
    import sys

    real = sys.modules.get("lmformatenforcer")
    sys.modules["lmformatenforcer"] = None
    try:
        with pytest.raises(Refusal) as caught:
            grammar._enforcer_module()
    finally:
        if real is None:
            sys.modules.pop("lmformatenforcer", None)
        else:
            sys.modules["lmformatenforcer"] = real

    message = str(caught.value)
    assert "lm-format-enforcer" in message
    assert "modelmri[grammar]" in message
    assert "invisible failure" in message
