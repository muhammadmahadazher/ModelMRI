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

`SUPPORTED` and `UNSUPPORTED` below are the whole contract, and `/v1/models`
carries them. Half-supporting a long tail is how compatibility layers become
untrustworthy.

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
import time
import uuid

from .errors import BadRequest, Refusal

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
    "modelmri",
)

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
    "response_format": "no constrained decoding on this path",
    "stop": "no stop-sequence handling on this path",
}

# OpenAI caps this at 20; matching keeps clients that validate happy.
MAX_TOP_LOGPROBS = 20

DEFAULT_MAX_TOKENS = 256


def _now() -> int:
    return int(time.time())


def _rid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def check_parameters(body: dict) -> None:
    """Refuse, by name, anything this cannot honour.

    Only values that would CHANGE something are refused: `n=1` is what this
    does anyway, and a client that sends it should not be turned away.
    """
    for name, why in UNSUPPORTED.items():
        if name not in body:
            continue
        value = body[name]
        if value is None:
            continue
        if name in ("n", "best_of") and value == 1:
            continue
        if name in ("presence_penalty", "frequency_penalty") and value == 0:
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        raise BadRequest(
            f"this server does not honour {name!r}: {why}. Refusing rather "
            f"than returning a completion computed without it — which is what "
            f"the runners that silently ignore it give you."
        )

    top = body.get("top_logprobs")
    if top is not None:
        if not isinstance(top, int) or isinstance(top, bool):
            raise BadRequest("'top_logprobs' must be a whole number.")
        if not 0 <= top <= MAX_TOP_LOGPROBS:
            raise BadRequest(
                f"'top_logprobs' must be between 0 and {MAX_TOP_LOGPROBS}."
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
                for p, i in zip(top.values, top.indices)
            ]
        out.append(entry)
    return out


def internals(runtime, ask: dict) -> dict:
    """The `modelmri` extension block, and what it cost to produce.

    Every measurement here is the SAME call the HTTP routes make. A second
    implementation would drift, and an agent reading this block would have no
    way to know which one was stale.
    """
    started = time.perf_counter()
    out: dict = {}
    failed: dict = {}

    if ask.get("lens"):
        try:
            out["lens"] = runtime.logit_lens(int(ask.get("top_k") or 5))
        except (Refusal, BadRequest) as err:
            # Named, not omitted. A block that silently lacks `lens` reads as
            # "the lens found nothing".
            failed["lens"] = str(err)

    heads = ask.get("heads")
    if heads:
        try:
            ranked = runtime.ablate_heads(None, str(ask.get("baseline") or "zero"))
            limit = int(heads) if not isinstance(heads, bool) else 10
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
            {"id": name, "object": "model", "created": _now(), "owned_by": "local"}
            for name in ids
        ],
        # Not part of OpenAI's shape, and deliberately here: a client that
        # wants to know what is honoured can read it rather than discover the
        # gaps in production.
        "modelmri": {
            "supported": list(SUPPORTED),
            "unsupported": dict(UNSUPPORTED),
            "note": "Unsupported parameters are refused by name with a 400, "
            "not silently ignored.",
        },
    }


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
