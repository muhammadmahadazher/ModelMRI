# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""What the last run cost, and how much of that was this tool watching.

Every local runner shows tokens/sec. What none of them shows is the cost of
being looked at — and ModelMRI is slower than Ollama for a specific, nameable
reason rather than a vague one. It forces `attn_implementation="eager"` and
asks for `output_attentions=True`, which materialises an
`n_layers x n_heads x S x S` score tensor that a runner never allocates. On a
12-layer, 12-head model at 4,096 tokens that is 4.8 GB — larger than the
weights of most models this runs on.

A tool whose whole argument is that it does not hide numbers should not hide
that one. So the introspection cost is its own line, computed from the shape
rather than measured indirectly, and it is warned about **before** a run that
would not fit rather than reported after the allocation fails.

**These are the allocator's numbers, not the driver's.** `max_memory_allocated`
counts what PyTorch handed out; the caching allocator reserves more and gives
it back slowly, and other processes are invisible to it. Every field here is
labelled `allocated by PyTorch` for that reason — calling it "VRAM used" would
be a claim about the card that this cannot make. `budget.py` reads the driver
side through `mem_get_info` and the two are kept separate on purpose.

**One generation is one sample.** Tokens/sec depends on the prompt, the
sequence length, the dtype and what else is on the GPU — measured on one RTX
4060 the same model ranged 12 to 71 ms/pass across sessions. So the prompt
length and the sequence length travel with the rate, and nothing here is
presented as a property of the model.

**A cell that could not be measured says so.** CPU has no allocator to ask and
MPS reports far less than CUDA; those read "could not measure" rather than 0,
because a zero in a memory column is a claim that nothing was used.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

# Above this, a context length is a sentinel rather than a limit. Several
# tokenizers report `model_max_length` as 1000000000000000019884624838656
# (int(1e30) rounded to a float and back), and a "0.0% of context used" derived
# from that is arithmetic performed on a placeholder.
SENTINEL_CONTEXT = 1_000_000


@dataclass
class Telemetry:
    """One generation, measured. Every field may be None and none may be faked."""

    prompt_tokens: int
    generated_tokens: int
    # Time to the FIRST streamed token — the prompt being processed — kept
    # apart from the decode rate it would otherwise distort. A 2,000-token
    # prompt makes a fast model look slow when the two are averaged.
    prompt_ms: float | None = None
    decode_ms: float | None = None
    tokens_per_s: float | None = None
    peak_bytes: int | None = None
    reserved_bytes: int | None = None
    memory_note: str = ""
    context_used: int = 0
    context_limit: int | None = None
    context_fraction: float | None = None
    introspection_bytes: int | None = None
    introspection_note: str = ""
    device: str = ""
    dtype: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("prompt_ms", "decode_ms"):
            if d[key] is not None:
                d[key] = round(d[key], 1)
        if d["tokens_per_s"] is not None:
            d["tokens_per_s"] = round(d["tokens_per_s"], 2)
        if d["context_fraction"] is not None:
            d["context_fraction"] = round(d["context_fraction"], 4)
        d["means"] = (
            "Memory is what PyTorch's allocator handed out, not what the card "
            "reports — other processes are invisible to it. One generation is "
            "one sample: the rate depends on this prompt, this sequence length "
            "and this dtype."
        )
        return d


def context_limit(model, tokenizer) -> tuple[int | None, str]:
    """(usable context length, where it came from) — or None with the reason.

    `tokenizer.model_max_length` is the field people reach for and it is
    frequently a sentinel rather than a limit, so the config's own
    `max_position_embeddings` is preferred and the sentinel is rejected by
    size rather than by matching a magic constant.
    """
    config = getattr(model, "config", None)
    for holder, name, label in (
        (config, "max_position_embeddings", "config.max_position_embeddings"),
        (config, "n_positions", "config.n_positions"),
        (tokenizer, "model_max_length", "tokenizer.model_max_length"),
    ):
        value = getattr(holder, name, None) if holder is not None else None
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue
        if 0 < value < SENTINEL_CONTEXT:
            return value, label
    return None, (
        "this model reports no usable context length — the field is absent or "
        "is a sentinel, and a percentage computed from a placeholder would be "
        "arithmetic about nothing"
    )


def eager_attention_bytes(
    n_layers: int, n_heads: int, seq_len: int, dtype_bytes: int = 2
) -> int:
    """`n_layers x n_heads x S^2 x dtype` — what watching the model costs.

    Quadratic in sequence length, and charged to this tool rather than to the
    model: a runner that never shows you attention never allocates it.
    """
    return int(n_layers) * int(n_heads) * int(seq_len) * int(seq_len) * int(dtype_bytes)


class Run:
    """Times one generation: prompt processing, then decode.

    Used as a context manager around the streaming loop, with `first_token()`
    called when the first token arrives — that boundary is the only place the
    two phases can be separated without a second forward pass.
    """

    def __init__(self, device_kind: str = "cpu") -> None:
        self.device_kind = device_kind
        self.started = 0.0
        self.first = None
        self.ended = None
        self.tokens = 0

    def __enter__(self) -> Run:
        from . import budget

        budget.reset_peak(self.device_kind)
        self.started = time.perf_counter()
        return self

    def first_token(self) -> None:
        if self.first is None:
            self.first = time.perf_counter()

    def token(self) -> None:
        self.first_token()
        self.tokens += 1

    def __exit__(self, *exc) -> None:
        from . import budget

        budget._synchronize(self.device_kind)
        self.ended = time.perf_counter()

    def finish(
        self,
        *,
        prompt_tokens: int,
        generated_tokens: int | None = None,
        n_layers: int = 0,
        n_heads: int = 0,
        dtype_bytes: int = 2,
        device: str = "",
        dtype: str = "",
        context: tuple[int | None, str] = (None, ""),
    ) -> Telemetry:
        """Turn the timings into a report, inventing nothing that was not seen."""
        from . import budget

        ended = self.ended if self.ended is not None else time.perf_counter()
        prompt_ms = (self.first - self.started) * 1000 if self.first else None
        decode_ms = (ended - self.first) * 1000 if self.first else None

        # `self.tokens` counts STREAM CHUNKS, and a TextIteratorStreamer yields
        # one per token PLUS a final flush from `TextStreamer.end()` — so it is
        # always one too many. The caller passes the real count from
        # `generate`'s own output ids; the counter is only a fallback for a
        # caller that has none, and it is stated as approximate when used.
        counted = self.tokens if generated_tokens is None else int(generated_tokens)

        # Divide by the INTERVALS, not the tokens. `self.first` is stamped when
        # the first token arrives, so the decode window spans n-1 gaps; using n
        # inflated the rate by n/(n-1), which is unbounded as n approaches 1.
        # Measured against a synthetic true 20.00 tok/s: 36,630 tok/s at n=1,
        # 39.6 at n=2, 26.4 at n=4, 20.1 at n=64 — the n/(n-1) curve exactly.
        rate = None
        if decode_ms and decode_ms > 0 and counted >= 2:
            rate = (counted - 1) / (decode_ms / 1000)

        mem = budget.free_memory(self.device_kind)
        peak = budget._peak_allocated(self.device_kind)
        reserved = None
        mod = budget._backend(self.device_kind)
        fn = getattr(mod, "memory_reserved", None) if mod else None
        if fn is not None:
            try:
                reserved = int(fn())
            except Exception:
                reserved = None

        total = prompt_tokens + counted
        limit, source = context
        fraction = (total / limit) if limit else None

        introspection = None
        note = ""
        if n_layers and n_heads:
            introspection = eager_attention_bytes(n_layers, n_heads, total, dtype_bytes)
            note = (
                f"{n_layers} layers x {n_heads} heads x {total}^2 x "
                f"{dtype_bytes} bytes — the attention scores ModelMRI asks for "
                "and a plain runner never allocates"
            )

        notes = []
        if peak is None:
            notes.append(
                mem.reason
                or "this backend does not report the allocator's peak, so the "
                "memory figures could not be measured"
            )
        if limit is None and source:
            notes.append(source)
        if counted == 0:
            notes.append("no tokens were generated, so there is no decode rate")
        elif counted == 1:
            # One token gives no interval to measure a rate over, and the
            # near-zero divisor used to produce numbers like 308 tok/s on a
            # machine doing 31.
            notes.append(
                "only one token was generated, so there is no interval to "
                "measure a rate over"
            )
        if generated_tokens is None and counted:
            notes.append(
                "token count is approximate — taken from the stream rather "
                "than from the model's output ids"
            )

        return Telemetry(
            prompt_tokens=prompt_tokens,
            generated_tokens=counted,
            prompt_ms=prompt_ms,
            decode_ms=decode_ms,
            tokens_per_s=rate,
            peak_bytes=peak,
            reserved_bytes=reserved,
            memory_note="allocated by PyTorch, not read from the card",
            context_used=total,
            context_limit=limit,
            context_fraction=fraction,
            introspection_bytes=introspection,
            introspection_note=note,
            device=device,
            dtype=dtype,
            notes=notes,
        )


def warn_before(
    n_layers: int, n_heads: int, seq_len: int, dtype_bytes: int, free_bytes: int | None
) -> str:
    """A sentence to show BEFORE a run whose attention will not fit, or "".

    Before, not after: the allocation this predicts is the one that fails, and
    a report explaining the failure afterwards is a worse product than a
    warning that prevents it.
    """
    need = eager_attention_bytes(n_layers, n_heads, seq_len, dtype_bytes)
    if not free_bytes or need < free_bytes * 0.5:
        return ""
    return (
        f"At {seq_len:,} tokens the attention scores alone need "
        f"{need / 1e9:,.1f} GB ({n_layers} x {n_heads} x {seq_len}^2 x "
        f"{dtype_bytes} bytes) and there is {free_bytes / 1e9:,.1f} GB free. "
        "That is what ModelMRI adds by watching the model; a shorter prompt "
        "or fewer new tokens is the fix."
    )
