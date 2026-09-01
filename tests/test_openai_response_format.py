"""`/v1` honours `response_format`, and every way it cannot is a named refusal.

`grammar.py` has been a complete constrained-decoding instrument with nothing
pointed at it, while `/v1` answered `response_format` with "no constrained
decoding on this path". These tests hold the wiring between them, and — more
of them than that — hold the six ways the wiring is allowed to say no:

  * `lm-format-enforcer` is not installed          -> 409 naming the extra
  * the backend is Ollama, which has no forward pass -> 409 naming the source
  * `logprobs` was asked for beside a schema       -> 400 naming both
  * the schema cannot be compiled into a mask      -> 400 naming where
  * `response_format.type` is not one of three     -> 400 naming the three
  * the request went to `/v1/completions`          -> 400 naming the route

Every one of those exists because the alternative is a completion that looks
like the one that was asked for and is not.
"""

from __future__ import annotations

import json

import pytest

# The route under test enforces a real token-level mask through a real model,
# so all three are hard requirements of this file rather than of one test in
# it. Guarded in the shape `tests/test_grammar.py` uses, and every target here
# is a dev dependency — `tests/test_suite_actually_runs.py` is what keeps that
# true, so this cannot become a file that silently stops running.
pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("lmformatenforcer")

from fastapi.testclient import TestClient

from modelmri import openai_api
from modelmri.errors import BadRequest, Refusal
from modelmri.server import create_app

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}

JSON_SCHEMA_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "person", "schema": SCHEMA},
}


# ------------------------------------------------- the contract, on its own


def test_response_format_moved_from_unsupported_to_supported():
    """The module docstring says SUPPORTED and UNSUPPORTED are the whole
    contract, so the move IS the change. `/v1/models` publishes both."""
    assert "response_format" in openai_api.SUPPORTED
    assert "response_format" not in openai_api.UNSUPPORTED


@pytest.mark.parametrize("name", ["stop", "seed", "top_p", "logit_bias", "tools"])
def test_the_other_refusals_are_untouched(name):
    """Only `response_format` moved. A change that quietly widened the
    surface would be the opposite of what this module is for."""
    assert name in openai_api.UNSUPPORTED


@pytest.mark.parametrize(
    "fmt",
    [
        {"type": "text"},
        {"type": "json_object"},
        JSON_SCHEMA_FORMAT,
    ],
)
def test_the_three_shapes_this_understands_are_accepted(fmt):
    openai_api.check_parameters({"response_format": fmt})


def test_json_object_means_an_object_and_not_any_json_value():
    """`None` and `{}` both mean "any JSON VALUE" to the enforcer, so a bare
    string or `null` would satisfy them — which is not what a client asking
    for `json_object` asked for."""
    schema = openai_api.schema_from_response_format(
        {"response_format": {"type": "json_object"}}
    )
    assert schema == {"type": "object"}


def test_a_json_schema_request_carries_the_callers_own_schema():
    schema = openai_api.schema_from_response_format(
        {"response_format": JSON_SCHEMA_FORMAT}
    )
    assert schema == SCHEMA


def test_text_asks_for_no_grammar_at_all():
    assert (
        openai_api.schema_from_response_format({"response_format": {"type": "text"}})
        is None
    )
    assert openai_api.schema_from_response_format({}) is None
    assert openai_api.schema_from_response_format({"response_format": None}) is None


def test_an_unknown_response_format_type_is_refused_by_name():
    with pytest.raises(BadRequest) as caught:
        openai_api.check_parameters({"response_format": {"type": "yaml_object"}})
    message = str(caught.value)
    assert "'yaml_object'" in message
    for known in openai_api.RESPONSE_FORMATS:
        assert known in message


def test_a_json_schema_block_with_no_schema_is_refused_by_name():
    with pytest.raises(BadRequest) as caught:
        openai_api.check_parameters(
            {"response_format": {"type": "json_schema", "json_schema": {"name": "p"}}}
        )
    assert "json_schema.schema" in str(caught.value)


def test_a_response_format_that_is_not_an_object_is_refused():
    with pytest.raises(BadRequest, match="'response_format'"):
        openai_api.check_parameters({"response_format": "json_object"})


def test_logprobs_beside_a_schema_is_refused_by_name():
    """The logprobs this server returns come from a second teacher-forced pass
    over the finished completion — the model's FREE-RUNNING distribution. Under
    a schema every token was drawn from one the grammar had already masked, so
    those numbers would describe a choice the model was not free to make."""
    with pytest.raises(BadRequest) as caught:
        openai_api.check_parameters(
            {"response_format": {"type": "json_object"}, "logprobs": True}
        )
    message = str(caught.value)
    assert "'logprobs'" in message and "'response_format'" in message


def test_logprobs_alone_and_a_schema_alone_both_still_pass():
    openai_api.check_parameters({"logprobs": True})
    openai_api.check_parameters({"response_format": {"type": "json_object"}})
    # False is what an SDK sends for "no", and it changes nothing.
    openai_api.check_parameters(
        {"response_format": {"type": "json_object"}, "logprobs": False}
    )


def test_the_text_completions_route_still_refuses_response_format():
    """OpenAI's own contract puts `response_format` on chat completions only."""
    with pytest.raises(BadRequest) as caught:
        openai_api.check_parameters(
            {"response_format": {"type": "json_object"}}, chat=False
        )
    message = str(caught.value)
    assert "'response_format'" in message
    assert "silently ignore" in message


@pytest.mark.parametrize("kind", sorted(openai_api.RESPONSE_FORMATS))
def test_a_published_format_is_never_answered_as_an_unknown_one(kind):
    """The enumeration and the implementation cannot drift.

    `RESPONSE_FORMATS` is what `/v1/models` advertises and what the
    unknown-type refusal lists. A type added to the dict and not handled below
    it would be waved through the membership check and then fall into
    somebody else's branch — refused, but with a message about a different
    parameter, which is the shape of answer this module exists to not give.
    So a listed type either resolves or refuses IN ITS OWN NAME.
    """
    try:
        openai_api.schema_from_response_format({"response_format": {"type": kind}})
    except BadRequest as err:
        assert kind in str(err), "a published type refused with another one's message"


def test_models_publishes_the_formats_and_the_chat_only_list():
    class _R:
        hf_id = None

    extra = openai_api.models_payload(_R())["modelmri"]
    assert set(extra["response_formats"]) == set(openai_api.RESPONSE_FORMATS)
    assert set(extra["chat_only"]) == set(openai_api.CHAT_ONLY)


# ---------------------------------------------------------- the route wiring


class _Recorder:
    """A recorder that records nothing, for the plumbing tests.

    It carries `temperature` into its trace exactly as the real one does,
    because that number is a claim the receipt makes about the run and the
    route is the only thing that supplies it.
    """

    def __init__(self, steps: int = 2, temperature: float = 0.0):
        from modelmri import grammar

        self.trace = grammar.Trace(
            schema=json.dumps(SCHEMA), vocab_size=100, temperature=float(temperature)
        )
        for i in range(steps):
            self.trace.steps.append(
                grammar.Step(step=i, allowed_count=3, vocab_size=100, deleted_mass=0.5)
            )


def app_with(tmp_path, pieces=('{"name": "a", ', '"age": 1}'), backend="hf"):
    """An app whose model is a fake generator, so nothing is downloaded.

    Every patch is on the runtime INSTANCE, never on the class: `loaded` is a
    read-only property, and assigning one to `type(runtime)` reaches every
    later test in the session.
    """
    app = create_app(trace_db=str(tmp_path / "t.sqlite"))
    seen: dict = {}

    def fake(prompt, max_new_tokens=256, temperature=0.7, commit=True, **kw):
        seen["recorder"] = kw.get("recorder")
        yield from pieces

    # `temperature` is REQUIRED here, with no default to fall back on. The
    # real `mask_recorder` defaults it to 0.0, and a fake that copied that
    # default would silently absorb a route which stopped passing the caller's
    # temperature at all — the receipt would then say 0.0 while the sampler ran
    # at 1.0, and every test in this file would stay green.
    def fake_recorder(schema, temperature):
        seen["schema"] = schema
        seen["temperature"] = temperature
        return _Recorder(temperature=temperature)

    app.state.runtime.generate_stream = fake
    app.state.runtime.mask_recorder = fake_recorder
    app.state.runtime.hf_id = "acme/tiny-1b"
    app.state.runtime.backend = backend
    app.state.runtime.model = object()
    assert app.state.runtime.loaded
    return app, seen


def test_a_schema_request_reaches_the_generation_path_as_a_recorder(tmp_path):
    app, seen = app_with(tmp_path)
    body = (
        TestClient(app)
        .post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": JSON_SCHEMA_FORMAT,
            },
        )
        .json()
    )
    assert seen["recorder"] is not None, "the mask never reached the generation path"
    assert body["choices"][0]["message"]["content"] == '{"name": "a", "age": 1}'


def test_the_mask_receipt_rides_the_response_without_being_asked_for(tmp_path):
    """You did not ask for the `modelmri` block; you asked for a grammar, and
    what the grammar cost is the receipt for the answer you got."""
    app, _ = app_with(tmp_path)
    body = (
        TestClient(app)
        .post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": JSON_SCHEMA_FORMAT,
            },
        )
        .json()
    )
    mask = body["modelmri"]["mask"]
    assert len(mask["steps"]) == 2
    assert mask["output_parses_as_json"] is True
    assert "step(s) under this schema" in mask["means"]


def test_an_unconstrained_completion_carries_no_mask_receipt(tmp_path):
    app, seen = app_with(tmp_path)
    body = (
        TestClient(app)
        .post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        .json()
    )
    assert "modelmri" not in body
    assert seen["recorder"] is None


def test_a_completion_that_does_not_parse_as_json_says_so(tmp_path):
    """A schema-constrained run that hit its token budget mid-object returns a
    fragment. Reported, not presented as an answer."""
    app, _ = app_with(tmp_path, pieces=('{"name": "a", ',))
    body = (
        TestClient(app)
        .post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": JSON_SCHEMA_FORMAT,
            },
        )
        .json()
    )
    mask = body["modelmri"]["mask"]
    assert mask["output_parses_as_json"] is False
    assert "DOES NOT PARSE AS JSON" in mask["means"]
    # Nothing collapsed, so the budget IS the honest diagnosis here — and the
    # test below is what stops it being the diagnosis in the other case too.
    assert mask["n_collapsed"] == 0
    assert "token budget ran out" in mask["means"]


def test_a_fragment_after_a_collapse_names_the_collapse_not_the_budget(tmp_path):
    """The receipt stated two causes and excluded the one the trace itself had
    recorded.

    MEASURED on the real route with `{"additionalProperties": true}`: the
    parser collapsed at step 2, `eos_only` was True in the same dict this
    sentence is written into, and generation ended because the mask permitted
    nothing but end-of-sequence — while the receipt blamed the token budget.
    A receipt that asserts a cause it did not check is the failure this whole
    module exists against, one level down.
    """
    app, _ = app_with(tmp_path, pieces=('{"name": "a", ',))

    def collapsed(schema, temperature):
        rec = _Recorder(temperature=temperature)
        rec.trace.steps[1].eos_only = True
        return rec

    app.state.runtime.mask_recorder = collapsed
    body = (
        TestClient(app)
        .post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": JSON_SCHEMA_FORMAT,
            },
        )
        .json()
    )
    mask = body["modelmri"]["mask"]
    assert mask["output_parses_as_json"] is False
    assert mask["n_collapsed"] == 1
    assert "ran out of anything to permit at step 1" in mask["means"]
    assert "token budget" not in mask["means"], "blamed a cause it did not check"


def test_the_receipt_reports_the_temperature_the_request_actually_ran_at(tmp_path):
    """The number the receipt publishes has to be the number the run used.

    `means()` states in words that the deleted mass was "measured on the
    distribution the model actually had at that step rather than on a
    temperature-1 one it never had", and the recorder is what makes that true —
    it owns the division. Drop `temperature` from the route's
    `mask_recorder(...)` call and three things become false at once: the
    recorder scales nothing, `_install_grammar` still overwrites HF's own
    `temperature` with 1.0, so the model samples at 1.0 while the caller asked
    for 0.9, and the receipt reports 0.0.
    """
    app, seen = app_with(tmp_path)
    body = (
        TestClient(app)
        .post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": JSON_SCHEMA_FORMAT,
                "temperature": 0.9,
            },
        )
        .json()
    )
    assert seen["temperature"] == pytest.approx(0.9)
    assert body["modelmri"]["mask"]["temperature"] == pytest.approx(0.9)


def test_a_request_with_no_temperature_reaches_the_recorder_as_the_default(tmp_path):
    """The route's own default, not the recorder's: they are different numbers
    and only one of them describes the sampler that ran."""
    app, seen = app_with(tmp_path)
    TestClient(app).post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": JSON_SCHEMA_FORMAT,
        },
    )
    assert seen["temperature"] == pytest.approx(0.7)


def test_the_callers_own_schema_is_what_reaches_the_recorder(tmp_path):
    """`is not None` passes for a recorder built from somebody else's schema."""
    app, seen = app_with(tmp_path)
    TestClient(app).post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": JSON_SCHEMA_FORMAT,
        },
    )
    assert seen["schema"] == SCHEMA


def test_a_strict_false_request_is_told_the_schema_was_enforced_anyway(tmp_path):
    """OpenAI's `strict: false` asks for a completion the model may deviate
    from, and there is no such path here — one hard token-level mask. Answering
    it silently with a stricter completion is the same failure as a silently
    ignored `logit_bias`, so it rides in the receipt."""
    app, _ = app_with(tmp_path)
    ask = {
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "person", "schema": SCHEMA, "strict": False},
        },
    }
    notes = (
        TestClient(app)
        .post("/v1/chat/completions", json=ask)
        .json()["modelmri"]["mask"]["notes"]
    )
    assert openai_api.STRICT_IS_UNCONDITIONAL in notes


@pytest.mark.parametrize("strict", [True, None])
def test_strict_true_or_absent_says_nothing_because_nothing_differs(tmp_path, strict):
    app, _ = app_with(tmp_path)
    block = {"name": "person", "schema": SCHEMA}
    if strict is not None:
        block["strict"] = strict
    notes = (
        TestClient(app)
        .post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {"type": "json_schema", "json_schema": block},
            },
        )
        .json()["modelmri"]["mask"]["notes"]
    )
    assert notes == []


def test_the_streaming_arm_carries_the_same_text_and_the_same_receipt(tmp_path):
    app, _ = app_with(tmp_path)
    c = TestClient(app)
    ask = {
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": JSON_SCHEMA_FORMAT,
    }
    plain = c.post("/v1/chat/completions", json=ask).json()
    streamed = c.post("/v1/chat/completions", json={**ask, "stream": True}).text

    frames = [
        f
        for f in streamed.split("\n\n")
        if f.startswith("data: ") and "[DONE]" not in f
    ]
    text = "".join(
        json.loads(f[6:])["choices"][0]["delta"].get("content", "") for f in frames
    )
    assert text == plain["choices"][0]["message"]["content"]
    tail = json.loads(frames[-1][6:])
    assert tail["modelmri"]["mask"]["output_parses_as_json"] is True


def test_the_modelmri_block_and_the_mask_receipt_share_one_response(tmp_path):
    """Asking for internals must not displace the receipt, or the other way."""
    app, _ = app_with(tmp_path)
    app.state.runtime.logit_lens = lambda top_k=5: {"rows": []}
    body = (
        TestClient(app)
        .post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": JSON_SCHEMA_FORMAT,
                "modelmri": {"lens": True},
            },
        )
        .json()
    )
    assert "mask" in body["modelmri"]
    assert "lens" in body["modelmri"]
    assert "extra_ms" in body["modelmri"]


# -------------------------------------------- the mask is the last word


def test_the_constrained_run_neutralises_the_samplers_that_run_after_it():
    """`generate` merges custom processors and then appends its own sampling
    warpers AFTER them, with a `# TODO (joao)` admitting there is no way to
    ask for another position. So the only way for the mask to be last is for
    nothing else to be there.

    `top_k` SILENTLY DEFAULTS TO 50 in transformers, and a checkpoint's own
    `generation_config.json` can set `top_p` or `typical_p`. Left alone they
    would truncate the distribution after the recorder had already measured
    the mask against the untruncated one — a receipt describing a step that
    did not happen. Temperature is worse than that: the recorder applies it
    itself, so leaving HF's in place applies it twice.
    """
    transformers = pytest.importorskip("transformers")
    from modelmri.runtime import ModelRuntime

    class _Model:
        generation_config = transformers.GenerationConfig()

    runtime = ModelRuntime()
    runtime.model = _Model()
    recorder = _Recorder()

    kwargs: dict = {"do_sample": True, "temperature": 0.7}
    runtime._install_grammar(kwargs, recorder)

    assert kwargs["logits_processor"][0] is recorder
    assert kwargs["temperature"] == 1.0, "temperature would be applied twice"
    assert kwargs["top_k"] == 0, "the default top-50 would truncate after the mask"
    assert kwargs["top_p"] == 1.0
    assert kwargs["typical_p"] == 1.0
    # One sequence, masked and recorded. Any other row comes back unmasked.
    assert kwargs["num_beams"] == 1
    assert kwargs["num_return_sequences"] == 1

    # EVERY warper this transformers has, not the four that were listed here.
    # A knob left un-neutralised truncates the distribution AFTER the recorder
    # measured the mask against the untruncated one, so the receipt describes a
    # step that did not happen — and `min_p`, `top_h`, `epsilon_cutoff` and
    # `eta_cutoff` were unasserted. The expected values are the ones that make
    # `generate` skip each warper entirely; which knobs EXIST is asked of the
    # installed config, the same way `_install_grammar` asks.
    neutral = {
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": None,
        "top_h": None,
        "typical_p": 1.0,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
    }
    known = [n for n in neutral if hasattr(_Model.generation_config, n)]
    assert len(known) >= 6, "this transformers has fewer warpers than expected"
    for name in known:
        assert kwargs[name] == neutral[name], f"{name} would run after the mask"


def test_the_tap_installed_is_the_one_bound_to_this_recorder():
    """`assert kwargs["stopping_criteria"]` is truthiness where an identity is
    meant: any non-empty list passes it, including one holding a criterion
    bound to a DIFFERENT recorder, which would leave every `chosen_id` at -1
    on the trace that gets published."""
    transformers = pytest.importorskip("transformers")
    from modelmri import grammar
    from modelmri.runtime import ModelRuntime

    class _Model:
        generation_config = transformers.GenerationConfig()

    runtime = ModelRuntime()
    runtime.model = _Model()
    recorder = _Recorder()

    kwargs: dict = {"do_sample": True, "temperature": 0.7}
    runtime._install_grammar(kwargs, recorder)

    taps = list(kwargs["stopping_criteria"])
    assert len(taps) == 1
    assert isinstance(taps[0], grammar.ChoiceTap)
    assert taps[0].recorder is recorder, "the tap records onto another trace"


def test_a_greedy_constrained_run_leaves_the_sampler_alone():
    """There are no warpers under `do_sample=False`, and setting `temperature`
    beside it earns a warning from transformers for nothing."""
    pytest.importorskip("transformers")
    from modelmri.runtime import ModelRuntime

    runtime = ModelRuntime()
    kwargs: dict = {"do_sample": False}
    runtime._install_grammar(kwargs, _Recorder())
    assert "temperature" not in kwargs
    assert "top_k" not in kwargs
    assert kwargs["logits_processor"], "the mask must be installed either way"


# --------------------------------------------------------------- the refusals


def test_the_missing_extra_is_refused_by_name_with_409(tmp_path, monkeypatch):
    """Never a degraded unconstrained completion. The enforcer is the only
    thing here that builds a correct mask, and a wrong mask is invisible.

    THE PLUMBING, not the sentence: the stub below supplies the substring this
    asserts, so what is proven here is that a `Refusal` from `_enforcer_module`
    reaches the client as a 409 rather than degrading to free text. The words
    themselves are pinned where they are written —
    `tests/test_grammar.py::test_the_missing_extra_refusal_names_the_extra_and_why_it_is_optional`.
    """
    from modelmri import grammar

    app = create_app(trace_db=str(tmp_path / "t.sqlite"))
    app.state.runtime.generate_stream = lambda *a, **k: iter(("x",))
    app.state.runtime.backend = "hf"
    app.state.runtime.model = object()

    def gone():
        raise Refusal(
            "Constrained decoding needs `lm-format-enforcer`, which is an "
            "optional extra: `pip install modelmri[grammar]`."
        )

    monkeypatch.setattr(grammar, "_enforcer_module", gone)
    r = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": JSON_SCHEMA_FORMAT,
        },
    )
    assert r.status_code == 409
    assert "modelmri[grammar]" in r.json()["error"]["message"]


def assert_is_the_grammar_refusal(message: str) -> None:
    """The grammar refusal, told apart from the other refusals it shares a
    status with.

    `"Ollama" in message` cannot do that. `runtime.py` carries a SECOND Ollama
    refusal — "Ollama serves text, not activations, so there is no ..." — that
    also contains the word, and 409 is also what "No model loaded" answers. A
    substring that matches a different refusal in the same module is a test
    that passes while the sentence it was written for has been replaced.

    These three are unique to the grammar refusal and each names a different
    half of it: what was asked for, what to do instead, and why the backend
    cannot do it.
    """
    assert "response_format" in message
    assert "'hf'" in message
    assert "forward pass" in message


def test_the_ollama_backend_refuses_rather_than_returning_free_text(tmp_path):
    """`generate_stream` returns before any forward pass exists for Ollama, so
    a schema there would be a request for structured output answered with
    unconstrained text and nothing saying so."""
    app = create_app(trace_db=str(tmp_path / "t.sqlite"))
    app.state.runtime.backend = "ollama"
    app.state.runtime.hf_id = "llama3"
    app.state.runtime.model = object()
    r = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": JSON_SCHEMA_FORMAT,
        },
    )
    assert r.status_code == 409
    assert_is_the_grammar_refusal(r.json()["error"]["message"])


def test_an_unenforceable_schema_is_a_400_and_not_a_500(tmp_path):
    """A nested unsupported type constructs WITHOUT error and then collapses
    the mask to end-of-sequence mid-object at decode time — a 200 carrying
    `{"a"`. The pre-flight is what makes that unreachable."""
    app, _ = app_with(tmp_path)
    del app.state.runtime.mask_recorder  # the real one, so the pre-flight runs
    r = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "x",
                    "schema": {
                        "type": "object",
                        "properties": {"a": {"type": "widget"}},
                    },
                },
            },
        },
    )
    assert r.status_code == 400
    message = r.json()["error"]["message"]
    assert "/properties/a" in message
    assert "widget" in message
    assert "Nothing was generated" in message


def test_a_schema_bad_at_its_root_is_also_a_400(tmp_path):
    app, _ = app_with(tmp_path)
    del app.state.runtime.mask_recorder
    r = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "x", "schema": {"type": "widget"}},
            },
        },
    )
    assert r.status_code == 400
    assert "widget" in r.json()["error"]["message"]


def test_a_boolean_sub_schema_is_a_400_and_not_a_four_character_200(tmp_path):
    """MEASURED on this route against real gpt2 before the pre-flight learned
    to look: `{"type": "object", "additionalProperties": true}` returned 200
    with a body of ` { "` and `finish_reason: "stop"`, its mask collapsed to
    end-of-sequence at step 2. JSON Schema allows a bare boolean where a schema
    goes; the grammar compiler reads `.anyOf` off it and dies inside a blanket
    `except Exception`."""
    app, _ = app_with(tmp_path)
    del app.state.runtime.mask_recorder  # the real one, so the pre-flight runs
    r = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "x",
                    "schema": {"type": "object", "additionalProperties": True},
                },
            },
        },
    )
    assert r.status_code == 400
    message = r.json()["error"]["message"]
    assert "'/additionalProperties'" in message
    assert "`true`" in message
    assert "Nothing was generated" in message


def test_logprobs_beside_a_schema_is_a_400_at_the_route(tmp_path):
    """Refused in `check_parameters`, which every other refusal in this file
    also has a route-level sibling for — and this is the one the brief calls a
    fabricated measurement."""
    app, seen = app_with(tmp_path)
    r = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": JSON_SCHEMA_FORMAT,
            "logprobs": True,
        },
    )
    assert r.status_code == 400
    message = r.json()["error"]["message"]
    assert "'logprobs'" in message and "'response_format'" in message
    assert "recorder" not in seen, "it was refused after generating"


def test_the_text_route_refuses_the_same_request_with_400(tmp_path):
    app, _ = app_with(tmp_path)
    r = TestClient(app).post(
        "/v1/completions",
        json={"prompt": "hi", "response_format": JSON_SCHEMA_FORMAT},
    )
    assert r.status_code == 400
    assert "'response_format'" in r.json()["error"]["message"]


@pytest.mark.parametrize("body", [{"stop": ["\n"]}, {"seed": 7}])
def test_the_contract_regression_stop_and_seed_are_still_refused(tmp_path, body):
    app, _ = app_with(tmp_path)
    r = TestClient(app).post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], **body},
    )
    assert r.status_code == 400
    assert repr(next(iter(body))) in r.json()["error"]["message"]


def test_a_streaming_refusal_is_a_refusal_and_not_an_empty_200(tmp_path):
    """A Refusal raised inside the SSE generator returns 200 with a zero-byte
    body, which an OpenAI client reads as a successful empty completion. Every
    grammar refusal fires before `StreamingResponse` is constructed.

    The status alone is not the assertion: `Refusal("No model loaded. POST
    /api/model/load first.")` is a 409 on this route too, so a mutation that
    turned the grammar refusal into any other 409 would pass a truthiness
    check on the message.
    """
    app = create_app(trace_db=str(tmp_path / "t.sqlite"))
    app.state.runtime.backend = "ollama"
    app.state.runtime.hf_id = "llama3"
    app.state.runtime.model = object()
    r = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": JSON_SCHEMA_FORMAT,
            "stream": True,
        },
    )
    assert r.status_code == 409
    assert_is_the_grammar_refusal(r.json()["error"]["message"])


# --------------------------------- against a real model, through the real route


@pytest.fixture(scope="module")
def gpt2_app():
    """The whole path, end to end: a real model behind the real route.

    gpt2 is this suite's standard text model and is already the fixture in
    `tests/test_grammar.py`, so nothing new is downloaded.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("lmformatenforcer")
    app = create_app()
    app.state.runtime.load("gpt2", device="cpu")
    return app


def test_the_route_returns_json_that_matches_the_schema(gpt2_app):
    """THE test. Bookkeeping passes just as well with a mask that permitted
    everything; valid JSON out of gpt2 does not."""
    r = TestClient(gpt2_app).post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Describe a person as JSON:"}],
            "response_format": JSON_SCHEMA_FORMAT,
            "temperature": 0,
            "max_tokens": 24,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    parsed = json.loads(body["choices"][0]["message"]["content"])
    assert set(parsed) == {"name", "age"}
    assert isinstance(parsed["name"], str)
    assert isinstance(parsed["age"], int)

    mask = body["modelmri"]["mask"]
    assert mask["output_parses_as_json"] is True
    assert mask["steps"], "the receipt recorded nothing"
    assert max(s["masked_fraction"] for s in mask["steps"]) > 0.999
    assert all(s["chosen_id"] >= 0 for s in mask["steps"]), (
        "no step recorded what was actually emitted"
    )


def _json_object_completion(app, max_tokens: int) -> dict:
    r = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Describe a person as JSON:"}],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": max_tokens,
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_json_object_mode_constrains_the_completion_to_a_json_object(gpt2_app):
    """`json_object` is `{"type": "object"}`, not `{}` and not None — both of
    those mean "any JSON VALUE" to the enforcer, so a bare string would
    satisfy them. What comes back has to open an object from the first token,
    which the same model unprompted does not."""
    body = _json_object_completion(gpt2_app, 24)
    text = body["choices"][0]["message"]["content"]
    assert text.lstrip().startswith("{"), text
    steps = body["modelmri"]["mask"]["steps"]
    assert steps[0]["masked_fraction"] > 0.999, (
        "the opening position permitted a wide set, so nothing was enforced"
    )
    assert not any(s["eos_only"] for s in steps), (
        "the mask collapsed to end-of-sequence, which is the parser-collapse "
        "signature the pre-flight exists to make unreachable"
    )


def test_a_free_object_schema_that_never_closes_is_reported_not_presented(gpt2_app):
    """MEASURED, and it is the reason the receipt carries this at all: gpt2
    under `{"type": "object"}` keeps opening new keys and never emits the
    closing brace — identical output at budgets of 48, 96, 160 and 256
    tokens. A free object has no forcing function, so the completion is a
    fragment.

    The mask did its job; the model never finished. Handed back under
    `finish_reason: "stop"` with no receipt, that fragment reads as structured
    output, which is the whole failure this module exists to refuse.
    """
    mask = _json_object_completion(gpt2_app, 24)["modelmri"]["mask"]
    assert mask["output_parses_as_json"] is False
    assert "DOES NOT PARSE AS JSON" in mask["means"]


def test_a_schema_with_an_end_does_close_and_does_parse(gpt2_app):
    """The control for the test above: the same model, the same budget, a
    schema whose required fields run out — and the completion parses."""
    body = _json_object_completion(gpt2_app, 24)
    assert body["modelmri"]["mask"]["output_parses_as_json"] is False
    r = TestClient(gpt2_app).post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Describe a person as JSON:"}],
            "response_format": JSON_SCHEMA_FORMAT,
            "temperature": 0,
            "max_tokens": 24,
        },
    )
    assert isinstance(json.loads(r.json()["choices"][0]["message"]["content"]), dict)
    assert r.json()["modelmri"]["mask"]["output_parses_as_json"] is True


def test_streaming_and_non_streaming_agree_under_the_same_greedy_settings(gpt2_app):
    """At temperature 0 both arms are deterministic, so the two receipts must
    be IDENTICAL rather than merely both present.

    They agree by construction today — one recorder, one `with_mask`, one
    joined text — and nothing would have noticed if that stopped being true.
    A streaming arm that built a second recorder and published its trace passed
    every assertion in this file.
    """
    c = TestClient(gpt2_app)
    ask = {
        "messages": [{"role": "user", "content": "Describe a person as JSON:"}],
        "response_format": JSON_SCHEMA_FORMAT,
        "temperature": 0,
        "max_tokens": 24,
    }
    plain = c.post("/v1/chat/completions", json=ask).json()
    streamed = c.post("/v1/chat/completions", json={**ask, "stream": True}).text

    frames = [
        f
        for f in streamed.split("\n\n")
        if f.startswith("data: ") and "[DONE]" not in f
    ]
    for frame in frames:
        json.loads(frame[6:])  # every frame is well-formed
    text = "".join(
        json.loads(f[6:])["choices"][0]["delta"].get("content", "") for f in frames
    )
    assert text == plain["choices"][0]["message"]["content"]
    assert json.loads(text)

    tail = json.loads(frames[-1][6:])["modelmri"]["mask"]
    assert tail["steps"] == plain["modelmri"]["mask"]["steps"]
    assert tail["means"] == plain["modelmri"]["mask"]["means"]
    assert (
        tail["output_parses_as_json"]
        == (plain["modelmri"]["mask"]["output_parses_as_json"])
    )


def test_a_sampled_run_reports_the_temperature_it_actually_used(gpt2_app):
    """The only constrained run in this file with `do_sample=True`, and the one
    that exercises `_install_grammar`'s neutralisation for real.

    Nothing is asserted about the TEXT: at 0.9 it is not deterministic, and a
    test tuned until a small model happened to close its object would be
    pinning luck. What is asserted is the number the receipt publishes about
    the run, which is deterministic whatever the sampler did.
    """
    r = TestClient(gpt2_app).post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Describe a person as JSON:"}],
            "response_format": JSON_SCHEMA_FORMAT,
            "temperature": 0.9,
            "max_tokens": 12,
        },
    )
    assert r.status_code == 200, r.text
    mask = r.json()["modelmri"]["mask"]
    assert mask["temperature"] == pytest.approx(0.9)
    assert mask["steps"], "the receipt recorded nothing"
    assert all(s["chosen_id"] >= 0 for s in mask["steps"])
    assert not any(s["eos_only"] for s in mask["steps"])


def test_a_sampled_constrained_run_hands_transformers_a_neutral_sampler(
    gpt2_app, monkeypatch
):
    """`_install_grammar` is otherwise only ever driven by a hand-built kwargs
    dict, so the ORDER it runs in inside `generate_stream` is unpinned.

    `generate_stream` sets `do_sample=True, temperature=<caller's>` and then
    calls `_install_grammar`, which overwrites the temperature with 1.0 because
    the recorder owns that division. Swap those two statements and HF's own
    warper survives: the caller's temperature is applied twice, once by
    transformers and once by the recorder, and the receipt describes neither
    distribution. Spied on the real `generate` of the real model, so this is
    what transformers was actually handed.
    """
    runtime = gpt2_app.state.runtime
    seen: dict = {}
    real = runtime.model.generate

    def spy(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(runtime.model, "generate", spy)
    r = TestClient(gpt2_app).post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Describe a person as JSON:"}],
            "response_format": JSON_SCHEMA_FORMAT,
            "temperature": 0.9,
            "max_tokens": 8,
        },
    )
    assert r.status_code == 200, r.text
    assert seen["do_sample"] is True, (
        "this must be the sampling path or it proves nothing"
    )
    assert seen["temperature"] == 1.0, "the caller's temperature would be applied twice"
    assert seen["top_k"] == 0
    assert seen["num_beams"] == 1
    assert r.json()["modelmri"]["mask"]["temperature"] == pytest.approx(0.9)


def test_a_bound_the_compiler_drops_is_disclosed_in_the_response(gpt2_app):
    """`minimum` and `maximum` are never compiled, so the mask permits values
    the schema forbids — and the completion still parses, still carries
    `output_parses_as_json`, and still comes back beside a sentence about what
    the mask cost. The note is the only thing that says the bound was dropped.
    """
    r = TestClient(gpt2_app).post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Describe a person as JSON:"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "age": {"type": "integer", "minimum": 0, "maximum": 5}
                        },
                        "required": ["age"],
                    },
                },
            },
            "temperature": 0,
            "max_tokens": 12,
        },
    )
    assert r.status_code == 200, r.text
    mask = r.json()["modelmri"]["mask"]
    note = " ".join(mask["notes"])
    assert "/properties/age" in note
    assert "'maximum'" in note
    assert note in mask["means"], "a note only a machine reads is a note nobody reads"
