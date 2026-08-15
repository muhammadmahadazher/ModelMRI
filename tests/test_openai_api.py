"""An OpenAI-compatible `/v1`, and the refusals that make it trustworthy.

The invariant worth the most: a parameter this cannot honour is REFUSED BY
NAME. llama.cpp and Ollama both silently ignore `logit_bias` and return a
completion computed without it, with nothing saying so — the answer looks like
the one that was asked for and is not.
"""

from __future__ import annotations

import json

import pytest

from modelmri import openai_api
from modelmri.errors import BadRequest

# ----------------------------------------- unsupported parameters, by name


@pytest.mark.parametrize(
    "body",
    [
        {"n": 2},
        {"best_of": 3},
        {"logit_bias": {"123": 5}},
        {"presence_penalty": 0.5},
        {"frequency_penalty": 1.0},
        {"seed": 42},
        {"tools": [{"type": "function"}]},
        {"functions": [{"name": "f"}]},
        {"response_format": {"type": "json_object"}},
        {"stop": ["\n"]},
    ],
)
def test_a_parameter_this_cannot_honour_is_refused_by_name(body):
    """The opposite of the category norm, and the point of the feature."""
    name = next(iter(body))
    with pytest.raises(BadRequest) as caught:
        openai_api.check_parameters(body)
    message = str(caught.value)
    assert repr(name) in message
    assert "silently ignore" in message


@pytest.mark.parametrize(
    "body",
    [
        {"n": 1},
        {"best_of": 1},
        {"presence_penalty": 0},
        {"frequency_penalty": 0},
        {"logit_bias": {}},
        {"stop": []},
        {"seed": None},
        {},
    ],
)
def test_a_value_that_changes_nothing_is_allowed(body):
    """`n=1` is what this does anyway; turning that client away would be
    pedantry rather than honesty."""
    openai_api.check_parameters(body)


def test_top_logprobs_is_bounded_like_openai_bounds_it():
    openai_api.check_parameters({"top_logprobs": 20})
    with pytest.raises(BadRequest, match="between 0 and 20"):
        openai_api.check_parameters({"top_logprobs": 21})
    with pytest.raises(BadRequest, match="whole number"):
        openai_api.check_parameters({"top_logprobs": 1.5})


def test_every_unsupported_parameter_says_what_it_would_have_changed():
    for name, why in openai_api.UNSUPPORTED.items():
        assert why.strip(), name
        assert not why.endswith("."), f"{name}: reads as a sentence fragment"


# ------------------------------------------------------------ the prompt


class _Tok:
    chat_template = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        body = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return body + ("<assistant>" if add_generation_prompt else "")


class _Runtime:
    tokenizer = _Tok()
    hf_id = "test/model"


class _BaseRuntime:
    """No chat template — a base model."""

    class _Plain:
        chat_template = None

    tokenizer = _Plain()
    hf_id = "test/base"


def test_messages_go_through_the_tokenizers_own_chat_template():
    """Concatenating roles by hand produces a prompt the model was never
    trained on and quietly changes every number downstream."""
    prompt = openai_api.build_prompt(
        _Runtime(), {"messages": [{"role": "user", "content": "hi"}]}
    )
    assert prompt == "<user>hi</user><assistant>"


def test_a_base_model_with_no_template_still_answers():
    prompt = openai_api.build_prompt(
        _BaseRuntime(), {"messages": [{"role": "user", "content": "hi"}]}
    )
    assert "user: hi" in prompt and prompt.endswith("assistant:")


def test_a_raw_prompt_is_passed_through():
    assert openai_api.build_prompt(_Runtime(), {"prompt": "once upon"}) == "once upon"


def test_several_prompts_in_a_list_are_refused_rather_than_joined():
    """Joining them would be one different prompt, answered once."""
    with pytest.raises(BadRequest, match="separate requests"):
        openai_api.build_prompt(_Runtime(), {"prompt": ["a", "b"]})


def test_neither_messages_nor_prompt_is_refused():
    with pytest.raises(BadRequest, match="either 'messages' or 'prompt'"):
        openai_api.build_prompt(_Runtime(), {})


def test_text_content_parts_are_joined():
    prompt = openai_api.build_prompt(
        _Runtime(),
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
    )
    assert "hi" in prompt


def test_an_image_part_is_refused_rather_than_dropped():
    """Dropping it would answer a different question than the one asked."""
    with pytest.raises(BadRequest, match="text model"):
        openai_api.build_prompt(
            _Runtime(),
            {
                "messages": [
                    {"role": "user", "content": [{"type": "image_url", "url": "x"}]}
                ]
            },
        )


# ------------------------------------------------------------- the shapes


def test_a_chat_completion_has_openais_shape():
    body = openai_api.completion_payload(
        "hello", "m", chat=True, prompt_tokens=3, completion_tokens=2
    )
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "hello"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] == 5


def test_a_text_completion_has_openais_shape():
    body = openai_api.completion_payload("hello", "m", chat=False)
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == "hello"


def test_a_stream_frame_is_valid_sse_and_carries_the_role_once():
    first = openai_api.chunk_payload("he", "m", chat=True, first=True)
    later = openai_api.chunk_payload("llo", "m", chat=True)
    assert first.startswith("data: ") and first.endswith("\n\n")
    assert json.loads(first[6:])["choices"][0]["delta"]["role"] == "assistant"
    assert "role" not in json.loads(later[6:])["choices"][0]["delta"]


def test_the_final_frame_closes_the_stream():
    tail = openai_api.final_chunk("m", chat=True)
    assert tail.endswith("data: [DONE]\n\n")
    body = json.loads(tail.split("\n\n")[0][6:])
    assert body["choices"][0]["finish_reason"] == "stop"


def test_the_extension_block_rides_the_final_frame():
    tail = openai_api.final_chunk("m", chat=True, extension={"extra_ms": 7})
    body = json.loads(tail.split("\n\n")[0][6:])
    assert body["modelmri"]["extra_ms"] == 7


# ------------------------------------------- the claimed surface is the real one


def test_models_carries_what_is_and_is_not_honoured():
    """A client that wants to know what is supported can read it rather than
    discover the gaps in production."""
    payload = openai_api.models_payload(_Runtime())
    assert payload["object"] == "list"
    extra = payload["modelmri"]
    assert set(extra["supported"]) == set(openai_api.SUPPORTED)
    assert set(extra["unsupported"]) == set(openai_api.UNSUPPORTED)
    assert "refused by name" in extra["note"]


def test_the_loaded_model_is_listed_first():
    payload = openai_api.models_payload(_Runtime())
    assert payload["data"][0]["id"] == "test/model"


# -------------------------------------------- the internals cost is measured


class _Measuring:
    """A runtime whose measurements succeed, to check the block's shape."""

    def logit_lens(self, top_k=5):
        return {"rows": [1, 2, 3]}

    def ablate_heads(self, layer=None, baseline="zero"):
        return {"ranked": [{"h": i} for i in range(40)], "noise_floor_kl": 0.01}

    def attribute_tokens(self, position=None):
        return {"ranked": []}


def test_the_extra_time_is_measured_not_estimated():
    out = openai_api.internals(_Measuring(), {"lens": True})
    assert isinstance(out["extra_ms"], int)
    assert "cost" in out["means"] and "ms" in out["means"]


def test_the_head_cap_is_stated():
    """The top 10 of 144 is a different claim from "these are the heads that
    mattered"."""
    out = openai_api.internals(_Measuring(), {"heads": 10})
    assert out["heads"]["shown"] == 10
    assert out["heads"]["of"] == 40
    assert len(out["heads"]["ranked"]) == 10


def test_the_noise_floor_travels_with_the_ranking():
    out = openai_api.internals(_Measuring(), {"heads": 5})
    assert out["heads"]["noise_floor_kl"] == 0.01


def test_a_measurement_that_refuses_is_named_not_omitted():
    """A block that silently lacks `lens` reads as "the lens found nothing"."""
    from modelmri.errors import Refusal

    class Refusing(_Measuring):
        def logit_lens(self, top_k=5):
            raise Refusal("Generate something first.")

    out = openai_api.internals(Refusing(), {"lens": True, "heads": 5})
    assert "lens" not in out
    assert "Generate something first." in out["not_measured"]["lens"]
    assert out["heads"]["shown"] == 5, "one refusal must not lose the others"


def test_nothing_asked_for_costs_almost_nothing():
    out = openai_api.internals(_Measuring(), {})
    assert set(out) == {"extra_ms", "means"}
