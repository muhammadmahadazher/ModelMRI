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
    """`CHAT_ONLY` goes through the same message template, so the same rule
    applies to it: the reason is interpolated mid-sentence, and one ending in
    a full stop reads as a fragment where it lands."""
    for name, why in {**openai_api.UNSUPPORTED, **openai_api.CHAT_ONLY}.items():
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


# ------------------------------------------ a count a client can act on


def test_a_negative_head_count_is_refused_not_sliced():
    """`rows[:-3]` drops rows off the END of a ranking sorted best-first and
    then `"shown": -3` was reported beside them. Neither the list nor the
    count meant anything a client could use."""
    with pytest.raises(BadRequest) as caught:
        openai_api.internals(_Measuring(), {"heads": -3})
    assert "modelmri.heads" in str(caught.value)


@pytest.mark.parametrize("bad", ["5", [1], {"n": 1}, 2.5, 10**9])
def test_a_head_count_that_is_not_a_count_is_a_bad_request(bad):
    """These raised ValueError or TypeError from inside the measurement, which
    the route turns into a 500 — an input mistake reported as a fault of the
    server."""
    with pytest.raises(BadRequest):
        openai_api.internals(_Measuring(), {"heads": bad})


@pytest.mark.parametrize("bad", ["5", -1, 0, [3]])
def test_a_lens_top_k_that_is_not_a_count_is_a_bad_request(bad):
    """A negative k reaches torch.topk, which raises from inside the lens."""
    with pytest.raises(BadRequest):
        openai_api.internals(_Measuring(), {"lens": True, "top_k": bad})


def test_shown_never_exceeds_what_was_ranked():
    """Asking for more heads than exist is not an error — 100 is inside the
    cap, and the answer is however many there were."""
    out = openai_api.internals(_Measuring(), {"heads": 100})
    assert out["heads"]["shown"] == out["heads"]["of"] == 40
    assert len(out["heads"]["ranked"]) == 40


def test_asking_for_no_heads_asks_for_nothing():
    """0 and false both mean "do not measure this", and neither is an error."""
    for nothing in (0, False):
        assert "heads" not in openai_api.internals(_Measuring(), {"heads": nothing})


def test_true_still_means_the_default_slice():
    out = openai_api.internals(_Measuring(), {"heads": True})
    assert out["heads"]["shown"] == openai_api.DEFAULT_HEADS_SHOWN


# ------------------------------------ an unknown extension key is refused too


def test_an_unknown_modelmri_key_is_refused_by_name():
    """The same rule as `logit_bias`, one level down. `{"lense": true}` used to
    return a 200 with an empty block, which reads as "the lens found nothing"
    rather than "you misspelled it"."""
    with pytest.raises(BadRequest) as caught:
        openai_api.check_parameters({"modelmri": {"lense": True}})
    said = str(caught.value)
    assert "'lense'" in said
    assert "lens" in said and "heads" in said, "it must name what IS understood"


def test_every_understood_extension_key_is_accepted():
    """The enumeration and the implementation cannot drift: anything listed
    must pass the check that guards the implementation."""
    openai_api.check_parameters(
        {"modelmri": {k: True for k in openai_api.MODELMRI_KEYS}}
    )


def test_a_modelmri_block_that_is_not_an_object_is_refused():
    with pytest.raises(BadRequest, match="must be an object"):
        openai_api.check_parameters({"modelmri": True})


def test_the_extension_keys_are_published_so_nobody_reads_the_source():
    payload = openai_api.models_payload(_Runtime())
    assert set(payload["modelmri"]["extension_keys"]) == set(openai_api.MODELMRI_KEYS)


# ------------------------------------------------- the `.mri` id, and its bounds


class _Exporting(_Measuring):
    def __init__(self):
        self.exports = 0

    def export_session(self, *a, **kw):
        self.exports += 1
        return b"gzipped-mri-bytes"


def test_asking_for_an_mri_hands_back_a_fetchable_id():
    store = openai_api.MriStore()
    out = openai_api.internals(_Exporting(), {"mri": True}, store)

    assert out["mri"]["id"].startswith("mri")
    assert out["mri"]["url"] == f"/v1/mri/{out['mri']['id']}"
    assert out["mri"]["bytes"] == len(b"gzipped-mri-bytes")
    assert store.get(out["mri"]["id"]) == b"gzipped-mri-bytes"


def test_the_mri_is_opt_in_because_building_one_costs_a_capture():
    runtime = _Exporting()
    openai_api.internals(runtime, {"lens": True}, openai_api.MriStore())
    assert runtime.exports == 0, "a client that did not ask for one paid for one"


def test_the_mri_cost_is_reported_apart_from_the_rest():
    """It is the one part of the block a client can decline to pay for, so it
    needs its own number rather than being folded into the total."""
    out = openai_api.internals(
        _Exporting(), {"lens": True, "mri": True}, openai_api.MriStore()
    )
    assert isinstance(out["mri"]["extra_ms"], int)
    assert out["mri"]["extra_ms"] <= out["extra_ms"]


def test_how_long_it_is_held_is_stated_rather_than_discovered():
    out = openai_api.internals(_Exporting(), {"mri": True}, openai_api.MriStore())
    held = out["mri"]["held"]
    assert "in memory" in held and "restart" in held and "evict" in held


def test_the_store_is_bounded_and_remembers_what_it_evicted():
    """An id that WAS held and an id that never existed are different answers:
    "ask again, sooner" and "you have the wrong id" have different fixes."""
    store = openai_api.MriStore(limit=2)
    first = store.put(b"a")
    store.put(b"b")
    third = store.put(b"c")

    assert store.get(first) is None, "the store grew past its limit"
    assert store.was_evicted(first) is True
    assert store.get(third) == b"c"
    assert store.was_evicted("mri-never-issued") is False


def test_an_export_that_refuses_is_named_not_omitted():
    from modelmri.errors import Refusal

    class Refusing(_Exporting):
        def export_session(self, *a, **kw):
            raise Refusal("Generate something first.")

    out = openai_api.internals(
        Refusing(), {"mri": True, "heads": 5}, openai_api.MriStore()
    )
    assert "mri" not in out
    assert "Generate something first." in out["not_measured"]["mri"]
    assert out["heads"]["shown"] == 5


def test_asking_for_an_mri_with_no_store_says_so_rather_than_dropping_it():
    out = openai_api.internals(_Exporting(), {"mri": True})
    assert "mri" not in out
    assert "not holding" in out["not_measured"]["mri"]


# --------------------------------------------------------------------------
# The parameters the contract was skipping.
#
# `SUPPORTED` and `UNSUPPORTED` are documented as "the whole contract", and
# `check_parameters` validates `top_logprobs` by name, type and range. Four
# parameters were not checked at all, and this is the surface other people's
# clients drive — so a wrong type or a wrong sign is ordinary rather than
# exotic. Each case below was measured against a live model first.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
def test_a_token_budget_of_zero_is_refused_not_replaced(field):
    """MEASURED: `{"max_tokens": 0}` returned 200 after 18.2 s with
    `usage.completion_tokens: 256`. `x or DEFAULT` cannot tell an explicit 0
    from an absent field, so a caller who asked for none got the default and
    nothing in the response said their request had been replaced."""
    with pytest.raises(BadRequest) as err:
        openai_api.check_parameters({field: 0})
    assert field in str(err.value)


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
def test_a_negative_token_budget_is_refused_immediately(field):
    """MEASURED: `{"max_tokens": -5}` answered 500 after 180 SECONDS."""
    with pytest.raises(BadRequest):
        openai_api.check_parameters({field: -5})


@pytest.mark.parametrize("value", ["many", [1], {"a": 1}, True])
def test_a_token_budget_of_the_wrong_type_names_itself(value):
    with pytest.raises(BadRequest):
        openai_api.check_parameters({"max_tokens": value})


def test_a_null_temperature_is_what_an_sdk_sends_and_is_not_an_error():
    """This one is NOT a client mistake: `null` is what an SDK emits for an
    optional field the caller left unset. Rejecting an SDK's normal output
    would make this surface unusable from the clients it exists for."""
    openai_api.check_parameters({"temperature": None})


@pytest.mark.parametrize("value", [-1.0, -99.0, 2.5, 99.0])
def test_a_temperature_outside_openais_range_is_refused(value):
    """MEASURED: -1.0, -99.0 and 0.0 returned byte-identical completions,
    because `generate_stream` branches on `if temperature > 0` and a negative
    value quietly became greedy decoding."""
    with pytest.raises(BadRequest) as err:
        openai_api.check_parameters({"temperature": value})
    assert "temperature" in str(err.value)


@pytest.mark.parametrize("value", [0, 0.0, 0.7, 2.0])
def test_a_temperature_inside_the_range_passes(value):
    openai_api.check_parameters({"temperature": value})


def test_top_p_is_part_of_the_contract_now():
    """It was in NEITHER `SUPPORTED` nor `UNSUPPORTED`, which this module's
    docstring says is impossible — so it was accepted and applied to nothing.
    MEASURED: a completion with top_p 0.0 came back byte-identical to one
    with no top_p at all."""
    assert "top_p" in openai_api.UNSUPPORTED
    with pytest.raises(BadRequest):
        openai_api.check_parameters({"top_p": 0.9})

    # 1.0 is OpenAI's default and disables nucleus sampling, so a client
    # sending it asks for exactly what this does — the same allowance
    # `n == 1` and `presence_penalty == 0` already get.
    openai_api.check_parameters({"top_p": 1.0})
    openai_api.check_parameters({"top_p": 1})


def test_created_is_a_fact_about_the_model_not_about_the_request():
    """MEASURED: 29 entries, every one stamped `int(time.time())`, so the
    payload asserted all of them were created at the instant of the request —
    and asserted something different six seconds later."""

    class _R:
        hf_id = None

    first = openai_api.models_payload(_R())
    second = openai_api.models_payload(_R())
    a = {m["id"]: m["created"] for m in first["data"]}
    b = {m["id"]: m["created"] for m in second["data"]}
    assert a == b, "the same disk gave two different answers"

    # And an unknown one is 1970 rather than "now": obviously not a real
    # creation date, where `now` looks plausible and is wrong.
    assert openai_api._first_seen("definitely/not-on-this-disk-xyz") == 0


# --------------------------------------------------------------------------
# One store, several completions at once.
#
# `app.state.mri_store` is a single object, and BOTH `/v1/chat/completions`
# and `/v1/completions` reach it through `asyncio.to_thread` — so concurrent
# requests asking for `{"modelmri": {"mri": true}}` run `put` at the same
# time, on a body that held no lock at all.
#
# MEASURED on the unlocked version, 16 threads, default switchinterval: 4
# escapes in 960,000 puts — {'KeyError': 2, 'RuntimeError': 2}. With
# `sys.setswitchinterval(1e-6)`, which widens the window without changing the
# code: 61,019 escapes in 320,000 puts and 34,508 observations of the store
# holding more than its stated limit. `next(iter(self._held))` raises
# RuntimeError when an insert resizes the dict mid-iteration, and two threads
# reading the same `oldest` make the loser's `del` raise KeyError.
#
# Neither is in `/v1`'s `except (Refusal, BadRequest)`, so both arrived as a
# 500 on a completion that had already been generated and committed — the
# work paid for, and nothing handed back.
#
# These are not a proof of absence: a race that survives is one that got
# lucky. What they do is fail loudly against the unlocked version, which the
# fix has to beat. The same standard as `test_traces_concurrency.py`, against
# the sibling store on the same `app.state`.


def _hammer(store, *, threads: int, per_thread: int) -> tuple[list, list]:
    """Run `put` from `threads` threads at once; collect escapes and overflows.

    Exceptions are collected rather than left to kill a worker in silence — an
    exception on a worker thread is how this stayed invisible, and it is also
    how it reached a user: as somebody else's 500.
    """
    import threading as _threading

    errors: list = []
    over: list = []
    guard = _threading.Lock()
    start = _threading.Barrier(threads)

    def worker():
        start.wait()
        for _ in range(per_thread):
            try:
                store.put(b"x" * 64)
            # `Exception`, not `BaseException`. The escapes this counts are
            # KeyError and RuntimeError; catching BaseException as well would
            # swallow a KeyboardInterrupt and log it as a store failure.
            except Exception as err:
                with guard:
                    errors.append(err)
            # Observed the way any other caller observes it — under the
            # store's own lock. An unlocked peek lands inside `put`'s own
            # insert-then-evict window and reports an overflow no caller can
            # ever see.
            with store._lock:
                held = len(store._held)
            if held > store.limit:
                with guard:
                    over.append(held)

    workers = [_threading.Thread(target=worker) for _ in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join(timeout=60)
    return errors, over


def test_concurrent_puts_do_not_500_a_completion_that_already_succeeded():
    """The reported shape: an exception out of `put`, on work already done.

    Run under `sys.setswitchinterval(1e-6)`, which widens the window without
    touching the code. At the default 5ms the rate is 4 per million and a
    suite-sized run passes on the unlocked version too — which would make
    this a test that says this machine was fast, not that the store is safe.
    """
    import sys as _sys

    was = _sys.getswitchinterval()
    _sys.setswitchinterval(1e-6)
    try:
        store = openai_api.MriStore(limit=8)
        errors, over = _hammer(store, threads=16, per_thread=3000)
    finally:
        _sys.setswitchinterval(was)

    assert not errors, (
        f"{len(errors)} of 48,000 puts escaped as an exception `/v1`'s "
        f"`except (Refusal, BadRequest)` does not catch, which is a 500 on a "
        f"completion already generated: {[type(e).__name__ for e in errors[:5]]}"
    )
    assert not over, (
        f"the store held more than its stated limit of {store.limit}: "
        f"{sorted(set(over))}"
    )
    assert len(store._held) == store.limit


def test_an_id_is_recorded_as_evicted_before_it_stops_being_held():
    """`GET /v1/mri/{id}` asks TWO questions — `get`, then `was_evicted` — and
    no lock inside this class can span the gap between them. So the ORDER
    inside `put` is what closes it.

    Recording the eviction after the delete left a window where an id this
    server had issued answered None to the first and False to the second, and
    the route turns that pair into the 404 that says "this server has never
    issued that id" — sending a client to check its id when the real answer
    was "ask again, sooner". `MriStore`'s own docstring is explicit that those
    are different answers with different fixes.

    Asserted at the instant of eviction rather than by racing two threads at
    it: the window is a single bytecode wide, so a threaded version passes
    against the wrong order most runs, which is worse than no test.
    """
    store = openai_api.MriStore(limit=1)
    still_held: list[bool] = []

    class _Watch(set):
        def add(self, item):
            still_held.append(item in store._held)
            return super().add(item)

    store._evicted = _Watch()
    first = store.put(b"a")
    store.put(b"b")

    assert still_held == [True], (
        "the id stopped being held before it was recorded as evicted, so a "
        "reader between the route's two questions sees neither"
    )
    # And the settled state is still the one the route documents.
    assert store.get(first) is None
    assert store.was_evicted(first) is True


def test_every_method_that_touches_the_store_serialises_it():
    """Against the source, because the defect was an ABSENCE.

    Nothing in the class said the rule existed, so the three methods simply
    did not follow one. A fourth added tomorrow is the same bug, and the
    sibling store carries the identical test for the identical reason.
    """
    import inspect
    import re

    src = inspect.getsource(openai_api.MriStore)
    offenders = []
    for part in re.split(r"\n    def ", src)[1:]:
        name = part.split("(")[0]
        if name == "__init__":
            continue
        if (
            re.search(r"self\._(held|evicted)\b", part)
            and "with self._lock" not in part
        ):
            offenders.append(name)
    assert not offenders, (
        "these read or write the shared mapping without holding the lock, "
        "which is what 500'd completions that had already been generated: "
        + ", ".join(offenders)
    )
