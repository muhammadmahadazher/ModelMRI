# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""An OpenAI-compatible `/v1` that returns what the model did, not just text.

Every runner exposes `/v1`, and ModelMRI could not be dropped into any client
that speaks it. This closes that — and inverts it, because everyone else's
`/v1` returns text and this one can return the logit-lens trajectory, the
ablation-ranked heads and the per-token attribution for the exact completion
you just received.

## Unsupported parameters are REFUSED, by name

llama.cpp and Ollama both silently ignore parameters they do not honour. A
client sends `logit_bias` and gets a completion computed without it, with
nothing saying so — the answer looks like the one that was asked for and is
not.

So anything here cannot honour is a 400 that NAMES the parameter. That is the
opposite of the category norm and it is the point: a refusal is cheap to
handle and a silently-wrong completion is not.

## The claimed surface is the implemented surface

`SUPPORTED`, `UNSUPPORTED` and `CHAT_ONLY` below are the whole contract, and
`/v1/models` carries all three. Half-supporting a long tail is how
compatibility layers become untrustworthy.

`CHAT_ONLY` exists because `response_format` is honoured on
`/v1/chat/completions` and refused on `/v1/completions`, which is where
OpenAI's own contract puts it. A parameter that is real on one route and not
the other cannot be described by two dicts, and the answer to that is a third
dict rather than a caveat in prose nobody publishes.

## Structured output is enforced, and what it cost is returned

`response_format` runs the completion under a token-level mask built by
`lm-format-enforcer` (see `grammar.py`). Every other runner ships this as a
black box: valid JSON out, no idea what it cost. Here the `modelmri` block
carries the per-step receipt — how much of the vocabulary was legal, how much
probability the mask deleted, and every step where the token the model most
wanted was forbidden — and it rides along without being asked for, because it
is the receipt for the answer that was just handed over.

Two combinations are refused rather than approximated. `logprobs` beside a
schema, because the logprobs here come from a second teacher-forced pass and
are therefore the model's FREE-RUNNING probabilities, which describe a choice
the model was not free to make. And a schema on the Ollama backend, which
returns finished text rather than a forward pass there is anything to mask.

"Enforced token by token" is not the same as "the whole schema was applied",
and the difference is published rather than glossed. The grammar compiler
reads sixteen schema keywords and no others, so `minimum`, `format`,
`patternProperties` and their kin are accepted and dropped — each one named in
the receipt's `notes` at the pointer where it was written. `json_schema.strict`
is there too: `strict: false` asks for a completion the model may deviate from
and there is no such path here, so the receipt says the schema was enforced
anyway. A silently-ignored schema keyword is the same failure as a
silently-ignored `logit_bias`, one level down.

## The internals cost time, and the time is measured

Asking for the `modelmri` block roughly doubles or triples the latency. That
extra is measured on the run that paid it and reported in the block, rather
than estimated or omitted.

## Loopback, always, unless told otherwise out loud

Binding beyond 127.0.0.1 turns this into an unauthenticated remote path onto
somebody's GPU. `serve` defaults to loopback and prints a warning when given
anything else; there is no exception and no "it's fine on a LAN".
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path

from .errors import BadRequest, Refusal
from .grammar import ANY_JSON_OBJECT

# What this actually implements. Enumerated because the alternative — claiming
# "OpenAI-compatible" and discovering the gaps in production — is what makes
# compatibility layers untrustworthy.
SUPPORTED = (
    "model",
    "messages",
    "prompt",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "stream",
    "logprobs",
    "top_logprobs",
    "response_format",
    "modelmri",
)

# The `response_format.type` values this understands, and what each one asks
# for. Enumerated for the same reason `SUPPORTED` is: an unrecognised type
# that fell through to an unconstrained completion would answer a request for
# structured output with free text and say nothing about it.
RESPONSE_FORMATS = {
    "text": "the ordinary completion, with no grammar over it",
    "json_object": "any JSON object, enforced token by token",
    "json_schema": "the schema at 'json_schema.schema', enforced token by "
    "token; any keyword the grammar compiler does not read is named in the "
    "mask receipt rather than silently dropped",
}

#: What `json_schema.strict` can and cannot change here, said in the receipt.
#:
#: There is one constrained-decoding path in this server and it is a hard
#: token-level mask. OpenAI's `strict: false` asks for a completion the model
#: may deviate from, and this cannot produce one — a run that could not deviate
#: is a DIFFERENT completion, not a better one, so it is disclosed rather than
#: substituted. Not a refusal: refusing would turn away a legal request this
#: server can answer, and answering it silently is the failure this module is
#: about. `strict: true` needs no note; it is exactly what happens.
STRICT_IS_UNCONDITIONAL = (
    "THE REQUEST SENT 'json_schema.strict' AS FALSE, AND THIS SERVER ENFORCED "
    "THE SCHEMA ANYWAY: there is one constrained-decoding path here and it is "
    "a hard token-level mask. What came back could not deviate from the "
    "schema, which is a different completion from the one 'strict': false "
    "asks for — said here rather than silently substituted."
)


def strict_was_overruled(body: dict) -> bool:
    """Whether this request asked for enforcement it does not get to skip.

    Shape only, like `schema_from_response_format` beside it, and it answers
    False for every request that did not send the key: an absent `strict` is
    OpenAI's own default and asks for nothing this contradicts.
    """
    fmt = body.get("response_format")
    if not isinstance(fmt, dict) or fmt.get("type") != "json_schema":
        return False
    block = fmt.get("json_schema")
    if not isinstance(block, dict) or "strict" not in block:
        return False
    return not block["strict"]


# Real on `/v1/chat/completions`, refused on `/v1/completions`. Not a gap —
# OpenAI's own contract puts `response_format` on chat completions only, and
# the legacy text endpoint never had it. Published beside SUPPORTED and
# UNSUPPORTED so a client reads the route split rather than discovering it.
CHAT_ONLY = {
    "response_format": "constrained decoding is wired into "
    "/v1/chat/completions, which is the route OpenAI's own contract gives it",
}

# The keys the `modelmri` block itself understands. Enumerated for the same
# reason `SUPPORTED` is, one level down: `{"modelmri": {"lense": true}}` used
# to return a block with nothing in it and a 200, which reads as "the lens
# found nothing" rather than "you misspelled it". A silently-ignored extension
# key is the same failure as a silently-ignored `logit_bias`.
MODELMRI_KEYS = {
    "lens": "the logit-lens trajectory for this completion",
    "heads": "the ablation-ranked attention heads",
    "attribute": "per-token attribution for this completion",
    "mri": "the whole analysis as a portable `.mri`, fetchable by id",
    "top_k": "how many tokens per layer the lens reports",
    "baseline": "the ablation baseline the head ranking uses",
}

# Parameters a client may legitimately send that this cannot honour. Each is
# refused BY NAME with what it would have changed, so the caller can decide
# rather than receiving a completion computed without it.
UNSUPPORTED = {
    "n": "this returns one completion; n>1 would need n independent runs and "
    "the cost is not hidden here",
    "best_of": "same reason as n — there is no sampling pool to pick from",
    "logit_bias": "the generation path applies no bias, so honouring this "
    "would mean returning a completion that ignored it",
    "presence_penalty": "not applied by this generation path",
    "frequency_penalty": "not applied by this generation path",
    "seed": "the backend does not thread a per-request seed, so a completion "
    "returned under one would not be reproducible by it",
    "tools": "no tool-calling surface",
    "functions": "no tool-calling surface",
    "stop": "no stop-sequence handling on this path",
    # In NEITHER dict until now, which this module's own docstring says is
    # impossible: "SUPPORTED and UNSUPPORTED below are the whole contract".
    # So it was accepted and applied to nothing — MEASURED, a completion with
    # top_p 0.0 came back byte-identical to one with no top_p at all.
    "top_p": "the generation path applies no nucleus sampling, so a "
    "completion returned under it would have ignored it",
}

# OpenAI caps this at 20; matching keeps clients that validate happy.
MAX_TOP_LOGPROBS = 20

#: OpenAI's own upper bound for `temperature`. Matched for the same reason
#: as MAX_TOP_LOGPROBS: a client that validates before sending should not
#: find this server stricter than the API it is imitating.
MAX_TEMPERATURE = 2.0

DEFAULT_MAX_TOKENS = 256

# How many ranked heads `{"heads": true}` shows, and the ceiling on asking for
# more. The cap is not about cost — the ranking is already computed — it is
# about `shown` staying a number a client can act on.
DEFAULT_HEADS_SHOWN = 10
MAX_HEADS_SHOWN = 4096
DEFAULT_LENS_TOP_K = 5
MAX_LENS_TOP_K = 100

# `.mri` files held for `{"mri": true}` to hand back by id. A `.mri` of a real
# analysis is on the order of 100 KB, so this is a few megabytes at most.
# BOUNDED on purpose: an eval loop asking for one per completion would
# otherwise grow the server's memory without limit for the length of the run.
MAX_HELD_MRI = 8


class MriStore:
    """The last `MAX_HELD_MRI` exports, by id.

    In memory and bounded, and BOTH of those are reported rather than left for
    a client to discover. An id whose file has been evicted answers differently
    from an id that never existed: "this expired, ask again" and "you have the
    wrong id" lead to different fixes, and collapsing them into one 404 sends
    people to debug the wrong one.

    Not on disk. Writing a file per completion on a client's say-so is the
    thing #40's caveat refuses for `export_mri`, and the same reasoning holds
    here: an eval loop would fill the user's data directory silently.
    """

    def __init__(self, limit: int = MAX_HELD_MRI):
        self.limit = limit
        self._held: dict[str, bytes] = {}
        # Ids evicted rather than never-issued. Just the ids, so this stays
        # small; it is what lets 410 and 404 be different answers.
        self._evicted: set[str] = set()
        # One store, several requests. `/v1/chat/completions` and
        # `/v1/completions` both reach this through `asyncio.to_thread`, so
        # concurrent completions asking for `{"mri": true}` land in `put` at
        # once — and every line of it used to run unserialised.
        #
        # MEASURED on this class, 16 threads, default switchinterval: 4
        # escapes in 960,000 puts — {'KeyError': 2, 'RuntimeError': 2}. With
        # `sys.setswitchinterval(1e-6)` to widen the window rather than change
        # the code: 61,019 escapes in 320,000 puts, and 34,508 observations of
        # the store holding MORE than its stated limit. `next(iter(self._held))`
        # raised RuntimeError when an insert resized the dict mid-iteration,
        # and two threads reading the same `oldest` made the loser's `del`
        # raise KeyError. Neither is in the route's `except (Refusal,
        # BadRequest)`, so both became a 500 on a completion that had already
        # been generated and committed — the worst shape a failure can take
        # here, because the work is done and paid for and the caller gets
        # nothing.
        #
        # `threading.Lock`, not the `RLock` the sibling store on the same
        # `app.state` uses (`traces.py`): that one needs reentrancy because its
        # methods call each other, and these three do not touch each other at
        # all. A non-reentrant lock keeps it that way — a future method that
        # calls another while holding it hangs, loudly, instead of quietly
        # reading a half-evicted mapping, which is the failure class this whole
        # comment exists about.
        self._lock = threading.Lock()

    def put(self, blob: bytes) -> str:
        mri_id = _rid("mri")
        with self._lock:
            self._held[mri_id] = blob
            while len(self._held) > self.limit:
                oldest = next(iter(self._held))
                # Recorded as evicted BEFORE it stops being held, and the order
                # is the point. `/v1/mri/{id}` asks two questions in sequence —
                # `get` then `was_evicted` — and no lock this class can hold
                # spans the gap between them. Deleting first opened a window
                # where an id this server had issued answered None to one and
                # False to the other, and the route turned that pair into the
                # 404 that says "this server has never issued that id" — the
                # answer that sends somebody to check their id, when the real
                # one was "ask again, sooner". Adding first, the worst a
                # caller can see is a blob that is still there.
                self._evicted.add(oldest)
                del self._held[oldest]
        return mri_id

    def get(self, mri_id: str) -> bytes | None:
        with self._lock:
            return self._held.get(mri_id)

    def was_evicted(self, mri_id: str) -> bool:
        with self._lock:
            return mri_id in self._evicted


def _now() -> int:
    return int(time.time())


def _whole(value, field: str, *, default: int, cap: int) -> int:
    """A count from the `modelmri` block, or a refusal that names the field.

    Neither of these was checked. `{"heads": -3}` reached `rows[:-3]`, which
    drops rows off the END of a ranking sorted best-first and then reported
    `"shown": -3` beside them — a count no client can act on, over a list that
    had quietly lost its tail. `{"top_k": "5"}` and `{"heads": [1]}` raised
    ValueError and TypeError from inside the measurement, which the caller
    turns into a 500: an input mistake reported as a server fault.
    """
    if value is True:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BadRequest(
            f"'modelmri.{field}' must be a whole number or true, not "
            f"{type(value).__name__}."
        )
    if isinstance(value, float) and value != int(value):
        raise BadRequest(f"'modelmri.{field}' must be a whole number, not {value}.")
    count = int(value)
    if not 1 <= count <= cap:
        raise BadRequest(
            f"'modelmri.{field}' must be between 1 and {cap}, or omitted. Got {count}."
        )
    return count


def _rid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def schema_from_response_format(body: dict) -> dict | None:
    """The JSON schema this request asks to be decoded under, or None.

    None means "no grammar", and it is a real answer rather than a fallback:
    an absent `response_format` and an explicit `{"type": "text"}` both mean
    the ordinary completion. Everything else either names a schema or is
    refused by name — there is no path from here to "asked for structure,
    received free text".

    The `json_schema` nesting is OpenAI's:
    `{"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}`.
    The schema itself is NOT compiled here. Whether a schema can be turned
    into a token-level mask is a question for `grammar.validate_schema`, which
    needs the enforcer installed; this is shape-checking, and it runs on every
    request including the ones with no model loaded.
    """
    fmt = body.get("response_format")
    if fmt is None:
        return None
    if not isinstance(fmt, dict):
        raise BadRequest(
            f"'response_format' is an object naming a type, and this request "
            f"sent {type(fmt).__name__}. For example "
            f'{{"response_format": {{"type": "json_object"}}}}.'
        )
    kind = fmt.get("type")
    if kind not in RESPONSE_FORMATS:
        known = ", ".join(sorted(RESPONSE_FORMATS))
        raise BadRequest(
            f"this server does not understand a 'response_format' of type "
            f"{kind!r}. It honours: {known}. Refusing rather than falling back "
            f"to an unconstrained completion — which is what a client asking "
            f"for structured output would otherwise receive, with nothing "
            f"saying so."
        )
    if kind == "text":
        return None
    if kind == "json_object":
        # Not `{}` and not None. Both of those mean "any JSON VALUE" to the
        # enforcer, so a bare string would satisfy them — see
        # `grammar.ANY_JSON_OBJECT`, where that is measured.
        return dict(ANY_JSON_OBJECT)

    block = fmt.get("json_schema")
    if not isinstance(block, dict) or not isinstance(block.get("schema"), dict):
        raise BadRequest(
            "a 'response_format' of type 'json_schema' carries the schema at "
            "'json_schema.schema', and this request sent none. For example "
            '{"response_format": {"type": "json_schema", "json_schema": '
            '{"name": "person", "schema": {"type": "object", "properties": '
            '{"name": {"type": "string"}}}}}}.'
        )
    return block["schema"]


def check_parameters(body: dict, *, chat: bool = True) -> None:
    """Refuse, by name, anything this cannot honour.

    Only values that would CHANGE something are refused: `n=1` is what this
    does anyway, and a client that sends it should not be turned away.

    `chat` picks the route's contract. `response_format` is honoured on
    `/v1/chat/completions` and refused on `/v1/completions` — see `CHAT_ONLY`.
    """
    refuse = dict(UNSUPPORTED) if chat else {**UNSUPPORTED, **CHAT_ONLY}
    for name, why in refuse.items():
        if name not in body:
            continue
        value = body[name]
        if value is None:
            continue
        if name in ("n", "best_of") and value == 1:
            continue
        if name in ("presence_penalty", "frequency_penalty") and value == 0:
            continue
        # 1.0 is OpenAI's default and disables nucleus sampling entirely, so
        # a client sending it is asking for exactly what this does. Turning
        # those away would refuse most SDKs for describing their own default,
        # which is the same mistake `n == 1` above exists to avoid.
        if name == "top_p" and value == 1:
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        raise BadRequest(
            f"this server does not honour {name!r}: {why}. Refusing rather "
            f"than returning a completion computed without it — which is what "
            f"the runners that silently ignore it give you."
        )

    ask = body.get("modelmri")
    if ask is not None:
        if not isinstance(ask, dict):
            raise BadRequest(
                f"'modelmri' must be an object naming what to measure, not "
                f"{type(ask).__name__}. For example "
                f'{{"modelmri": {{"lens": true, "heads": 10}}}}.'
            )
        unknown = [k for k in ask if k not in MODELMRI_KEYS]
        if unknown:
            known = ", ".join(sorted(MODELMRI_KEYS))
            raise BadRequest(
                f"this server does not understand "
                f"{', '.join(repr(k) for k in sorted(unknown))} inside "
                f"'modelmri'. It understands: {known}. Refusing rather than "
                f"returning a block without it — an extension key that is "
                f"silently dropped reads as a measurement that found nothing."
            )

    # A TOKEN BUDGET, checked the way `top_logprobs` below already is.
    #
    # `int(body.get("max_completion_tokens") or body.get("max_tokens") or
    # DEFAULT)` cannot tell an explicit 0 from an absent field, so a caller who
    # asked for zero tokens received 256 and waited eighteen seconds for them.
    # A negative one waited three minutes and got a 500.
    for field in ("max_tokens", "max_completion_tokens"):
        budget = body.get(field)
        if budget is None:
            continue
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise BadRequest(
                f"{field!r} must be a whole number of tokens, and this "
                f"request sent {budget!r}."
            )
        if budget < 1:
            raise BadRequest(
                f"{field!r} must be at least 1, and this request sent "
                f"{budget}. A completion of zero tokens is not a completion, "
                f"and this used to answer it with {DEFAULT_MAX_TOKENS} — the "
                f"default, silently, because an explicit 0 is indistinguishable "
                f"from an absent field to `or`."
            )

    # TEMPERATURE, in OpenAI's own range.
    #
    # `null` is deliberately NOT an error: it is what an SDK emits for an
    # optional field the caller left unset, and rejecting an SDK's normal
    # output would make this surface unusable from the clients it exists for.
    # A negative value is a different matter — it was accepted and silently
    # became greedy decoding, because `generate_stream` branches on
    # `if temperature > 0`. Three negative values returned identical text.
    temperature = body.get("temperature")
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise BadRequest(
                f"'temperature' must be a number, and this request sent "
                f"{temperature!r}."
            )
        if not 0.0 <= float(temperature) <= MAX_TEMPERATURE:
            why = (
                "A negative value is not colder than 0 — it was accepted and "
                "quietly became greedy decoding, which is a completion "
                "computed under a setting nobody asked for."
                if float(temperature) < 0
                else "Above 2 the sampler is not meaningfully hotter, and "
                "OpenAI's own range stops there."
            )
            raise BadRequest(
                f"'temperature' must be between 0 and {MAX_TEMPERATURE:g}, and "
                f"this request sent {temperature}. {why}"
            )

    top = body.get("top_logprobs")
    if top is not None:
        if not isinstance(top, int) or isinstance(top, bool):
            raise BadRequest("'top_logprobs' must be a whole number.")
        if not 0 <= top <= MAX_TOP_LOGPROBS:
            raise BadRequest(
                f"'top_logprobs' must be between 0 and {MAX_TOP_LOGPROBS}."
            )

    # RESPONSE FORMAT, and the one combination it cannot be in.
    #
    # Shape only — whether the schema compiles is `grammar.validate_schema`'s
    # question and needs the optional extra. Called for its refusals here; the
    # route calls it again for the schema itself, which is pure.
    schema = schema_from_response_format(body) if chat else None
    if schema is not None and body.get("logprobs"):
        raise BadRequest(
            "'logprobs' and 'response_format' cannot both be honoured. The "
            "logprobs this server returns come from a second teacher-forced "
            "pass over the finished completion, so they are the model's own "
            "free-running probabilities — but under a schema every token was "
            "drawn from a distribution the grammar had already masked, and "
            "nothing in the response would say so. Those numbers would "
            "describe a choice the model was not free to make. Send one or the "
            "other; the per-step mask receipt in the 'modelmri' block reports "
            "what the schema cost instead."
        )


def build_prompt(runtime, body: dict) -> str:
    """The prompt this request actually asks for.

    Chat messages go through the tokenizer's own chat template when it has
    one. Concatenating roles by hand would produce a prompt the model was
    never trained on and quietly change every number downstream.
    """
    if "prompt" in body:
        prompt = body["prompt"]
        if isinstance(prompt, list):
            # OpenAI allows a list; this takes the first and says so rather
            # than silently joining them into one different prompt.
            if len(prompt) > 1:
                raise BadRequest(
                    "'prompt' as a list of several prompts would be several "
                    "completions; send them as separate requests."
                )
            prompt = prompt[0] if prompt else ""
        return str(prompt or "")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise BadRequest("send either 'messages' or 'prompt'.")

    clean = []
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise BadRequest(f"message {i} is not an object.")
        role = str(message.get("role") or "")
        content = message.get("content")
        if isinstance(content, list):
            # Content parts: keep the text ones, name the others rather than
            # dropping them into an empty string.
            parts = []
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block, dict) and block.get("type"):
                    raise BadRequest(
                        f"message {i} carries a {block['type']!r} part, and "
                        f"this serves a text model."
                    )
            content = "\n".join(parts)
        clean.append({"role": role, "content": str(content or "")})

    tokenizer = getattr(runtime, "tokenizer", None)
    template = getattr(tokenizer, "chat_template", None)
    if tokenizer is not None and template:
        return tokenizer.apply_chat_template(
            clean, tokenize=False, add_generation_prompt=True
        )
    # No template: this is a base model, and saying so beats inventing a
    # conversation format it was never trained on.
    return "\n".join(f"{m['role']}: {m['content']}" for m in clean) + "\nassistant:"


def token_logprobs(runtime, top_k: int = 0) -> list:
    """Exact logprobs for the tokens the model just produced.

    ONE teacher-forced pass over the committed sequence, reading each
    generated token's probability off the position before it. This is not an
    approximation of the sampling-time distribution — in eval mode it is the
    same computation, and it is the model's actual probability for the token
    it actually emitted.

    Done after the completion rather than inside `generate` so the shared
    generation path stays exactly as it is; a second implementation of decode
    would be a second thing to keep correct.
    """
    import torch

    from . import ablate

    ids = getattr(runtime, "last_ids", None)
    n_prompt = int(getattr(runtime, "last_n_prompt_tokens", 0) or 0)
    if ids is None or not n_prompt:
        return []
    # `last_ids` is stored 1-D and on CPU (see `runtime.generate_stream`), so
    # it needs a batch axis and the model's device before a forward pass.
    ids = ids.unsqueeze(0) if ids.dim() == 1 else ids
    ids = ids.to(next(runtime.model.parameters()).device)
    total = int(ids.shape[1])
    if total <= n_prompt:
        return []

    with torch.no_grad():
        logits = runtime.model(ids).logits[0]

    tokenizer = runtime.tokenizer
    out = []
    for position in range(n_prompt, total):
        # The distribution that PRODUCED this token is the one at the step
        # before it.
        probs = ablate.distribution(logits[position - 1])
        chosen = int(ids[0, position])
        entry = {
            "token": tokenizer.decode([chosen]),
            "logprob": float(torch.log(probs[chosen].clamp_min(1e-12))),
            "bytes": list(tokenizer.decode([chosen]).encode("utf-8")),
        }
        if top_k:
            top = torch.topk(probs, min(top_k, probs.shape[-1]))
            entry["top_logprobs"] = [
                {
                    "token": tokenizer.decode([int(i)]),
                    "logprob": float(torch.log(p.clamp_min(1e-12))),
                    "bytes": list(tokenizer.decode([int(i)]).encode("utf-8")),
                }
                for p, i in zip(top.values, top.indices, strict=True)
            ]
        out.append(entry)
    return out


def internals(runtime, ask: dict, store: MriStore | None = None) -> dict:
    """The `modelmri` extension block, and what it cost to produce.

    Every measurement here is the SAME call the HTTP routes make. A second
    implementation would drift, and an agent reading this block would have no
    way to know which one was stale.
    """
    started = time.perf_counter()
    out: dict = {}
    failed: dict = {}

    if ask.get("lens"):
        asked_k = ask.get("top_k")
        top_k = (
            DEFAULT_LENS_TOP_K
            if asked_k is None
            else _whole(
                asked_k, "top_k", default=DEFAULT_LENS_TOP_K, cap=MAX_LENS_TOP_K
            )
        )
        try:
            out["lens"] = runtime.logit_lens(top_k)
        except (Refusal, BadRequest) as err:
            # Named, not omitted. A block that silently lacks `lens` reads as
            # "the lens found nothing".
            failed["lens"] = str(err)

    heads = ask.get("heads")
    if heads:
        limit = _whole(heads, "heads", default=DEFAULT_HEADS_SHOWN, cap=MAX_HEADS_SHOWN)
        try:
            ranked = runtime.ablate_heads(None, str(ask.get("baseline") or "zero"))
            rows = ranked.get("ranked") or []
            out["heads"] = {
                **{k: v for k, v in ranked.items() if k != "ranked"},
                "ranked": rows[:limit],
                # The cap, stated. The top 10 of 144 is a different claim from
                # "these are the heads that mattered".
                "shown": min(limit, len(rows)),
                "of": len(rows),
            }
        except (Refusal, BadRequest) as err:
            failed["heads"] = str(err)

    if ask.get("attribute"):
        try:
            out["attribute"] = runtime.attribute_tokens(None)
        except (Refusal, BadRequest) as err:
            failed["attribute"] = str(err)

    # The whole analysis as one portable file, when asked for. Opt-in, unlike
    # the blocks above, because it captures attention to build -- so a client
    # that does not want a file does not pay for one.
    if ask.get("mri"):
        if store is None:
            failed["mri"] = (
                "this server is not holding `.mri` files, so there is no id to "
                "hand back. Use GET /api/session/export to download one."
            )
        else:
            mri_started = time.perf_counter()
            try:
                blob = runtime.export_session(note="via /v1")
                mri_id = store.put(blob)
                out["mri"] = {
                    "id": mri_id,
                    "url": f"/v1/mri/{mri_id}",
                    "bytes": len(blob),
                    # Separate from the block's own `extra_ms`, because this is
                    # the one part a client can decline to pay for.
                    "extra_ms": int((time.perf_counter() - mri_started) * 1000),
                    "held": (
                        f"in memory, and only the last {store.limit}. It does "
                        f"not survive a restart of this server, and asking for "
                        f"{store.limit} more will evict it — fetch it before "
                        f"the run moves on."
                    ),
                }
            except (Refusal, BadRequest) as err:
                failed["mri"] = str(err)

    if failed:
        out["not_measured"] = failed
    # MEASURED, on the run that paid it. Asking for internals roughly doubles
    # or triples the latency and a client deciding whether to keep asking
    # needs the real number rather than an estimate.
    out["extra_ms"] = int((time.perf_counter() - started) * 1000)
    out["means"] = (
        f"These are measurements of THIS completion, taken after it committed, "
        f"and they cost {out['extra_ms']} ms on top of generating it."
    )
    return out


def _first_seen(name: str) -> int:
    """When this checkpoint landed on this disk, or 0 for "not known".

    `created` in `/v1/models` is a fact about the MODEL, and `_now()` made it
    a fact about the request: every entry claimed to have been created at the
    instant it was listed, and claimed something different six seconds later.
    MEASURED — 29 entries, all stamped with the same moving number.

    The cache directory's mtime is the closest thing to that fact this server
    can actually observe: when the weights arrived here. `0` when even that
    cannot be read, which is 1970 and therefore obviously not a real creation
    date — unlike "now", which looks plausible and is wrong. The field stays
    present either way, because clients index it.
    """
    from . import paths

    safe = re.sub(r"[^A-Za-z0-9._-]", "--", name.replace("/", "--"))
    for candidate in (
        paths.hf_hub_cache() / f"models--{safe}",
        Path(name).expanduser(),
    ):
        try:
            if candidate.exists():
                return int(candidate.stat().st_mtime)
        except OSError:
            continue
    return 0


def models_payload(runtime) -> dict:
    """`/v1/models`, plus what this server does and does not honour."""
    from . import discover

    found = discover.discover()
    ids = []
    for entry in found.get("models", []) if isinstance(found, dict) else []:
        name = entry.get("id") or entry.get("name") if isinstance(entry, dict) else None
        if name:
            ids.append(str(name))
    loaded = getattr(runtime, "hf_id", None)
    if loaded and loaded not in ids:
        ids.insert(0, str(loaded))

    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                # A per-MODEL fact, not a per-request one. See `_first_seen`.
                "created": _first_seen(name),
                "owned_by": "local",
            }
            for name in ids
        ],
        # Not part of OpenAI's shape, and deliberately here: a client that
        # wants to know what is honoured can read it rather than discover the
        # gaps in production.
        "modelmri": {
            "supported": list(SUPPORTED),
            "unsupported": dict(UNSUPPORTED),
            # Real on one route and refused on the other, which neither of the
            # two dicts above can say.
            "chat_only": dict(CHAT_ONLY),
            # What `response_format` accepts, so an unrecognised type is a
            # thing a client can look up rather than discover.
            "response_formats": dict(RESPONSE_FORMATS),
            # The extension's own keys, for the same reason: a client should
            # not have to read the source to learn what it may ask for.
            "extension_keys": dict(MODELMRI_KEYS),
            "note": "Unsupported parameters are refused by name with a 400, "
            "not silently ignored. That applies inside 'modelmri' too.",
        },
    }


def mask_block(trace, text: str) -> dict:
    """The receipt for a constrained completion, and whether it finished.

    `trace` carries what the mask cost per step. The second half is not a
    formality: a schema-constrained run that reaches `max_tokens` mid-object
    returns a fragment, and a fragment handed back under `finish_reason:
    "stop"` beside a mask receipt reads as a structured answer. So the
    completion is parsed, once, and the answer is stated — an unparseable one
    says so IN the sentence a reader actually reads, not only in a boolean
    they might not.
    """
    doc = trace.to_dict()
    try:
        json.loads(text)
    except ValueError:
        doc["output_parses_as_json"] = False
        # THE CAUSE IS READ OFF THE TRACE, not guessed at.
        #
        # This used to state two causes — "the token budget ran out mid-object,
        # or generation was cut short" — and exclude the one the trace itself
        # had already recorded. MEASURED, on a schema whose `additionalProperties`
        # was `true`: generation ended at step 2 because the mask permitted
        # nothing but end-of-sequence, `eos_only` was True in the same dict this
        # sentence is being written into, and the receipt blamed the budget. A
        # receipt asserting a cause it did not check is the failure this module
        # exists against, one level down.
        collapsed = trace.collapsed
        if collapsed:
            why = (
                f"the grammar ran out of anything to permit at step "
                f"{collapsed[0].step}, so the enforcement failed rather than "
                f"the budget."
            )
        else:
            why = "the token budget ran out mid-object, or generation was cut short."
        doc["means"] = (
            f"{doc['means']} THE COMPLETION DOES NOT PARSE AS JSON, so the "
            f"grammar never reached the end of a value: {why} What came back "
            f"is a fragment, not an answer in the shape that was asked for."
        )
    else:
        doc["output_parses_as_json"] = True
    return doc


def completion_payload(
    text: str,
    model: str,
    *,
    chat: bool,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    logprobs=None,
    extension: dict | None = None,
) -> dict:
    body: dict = {
        "id": _rid("chatcmpl" if chat else "cmpl"),
        "object": "chat.completion" if chat else "text_completion",
        "created": _now(),
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    if chat:
        choice: dict = {
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }
        if logprobs is not None:
            choice["logprobs"] = {"content": logprobs}
    else:
        choice = {"index": 0, "text": text, "finish_reason": "stop"}
        if logprobs is not None:
            choice["logprobs"] = {"content": logprobs}
    body["choices"] = [choice]
    if extension is not None:
        body["modelmri"] = extension
    return body


def chunk_payload(delta: str, model: str, *, chat: bool, first: bool = False) -> str:
    """One SSE `data:` frame, in OpenAI's shape."""
    if chat:
        piece: dict = {"content": delta}
        if first:
            piece["role"] = "assistant"
        choice = {"index": 0, "delta": piece, "finish_reason": None}
        obj = "chat.completion.chunk"
    else:
        choice = {"index": 0, "text": delta, "finish_reason": None}
        obj = "text_completion"
    body = {
        "id": _rid("chatcmpl" if chat else "cmpl"),
        "object": obj,
        "created": _now(),
        "model": model,
        "choices": [choice],
    }
    return f"data: {json.dumps(body)}\n\n"


def final_chunk(model: str, *, chat: bool, extension: dict | None = None) -> str:
    choice = {"index": 0, "finish_reason": "stop"}
    choice["delta" if chat else "text"] = {} if chat else ""
    body = {
        "id": _rid("chatcmpl" if chat else "cmpl"),
        "object": "chat.completion.chunk" if chat else "text_completion",
        "created": _now(),
        "model": model,
        "choices": [choice],
    }
    if extension is not None:
        body["modelmri"] = extension
    return f"data: {json.dumps(body)}\n\ndata: [DONE]\n\n"
