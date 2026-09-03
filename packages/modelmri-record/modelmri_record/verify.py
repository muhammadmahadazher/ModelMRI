# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Check the SDK before patching it, and refuse rather than half-record.

This is the systemic failure across agent tracing. AgentOps broke on
`openai.resources.beta.chat` and again on `google.adk.telemetry`; Langfuse
dropped PydanticAI's instructions field; Opik's LangChain tracer leaked
per-trace state. Every one is auto-instrumentation monkey-patching a moving
SDK and then emitting an incomplete span rather than failing loudly.

An incomplete span is the worse outcome, and specifically so here: the trace
store's token columns are nullable precisely so "the provider said nothing"
is recordable. A wrapper reading a moved attribute produces exactly that
shape — `tokens_in=None` — and the ledger would then faithfully report "not
reported by provider" about a provider that reported perfectly well. The
absence would be real and the reason would be wrong.

So: fingerprint first, patch second.

## What is fingerprinted, and what is deliberately not

ONLY the attributes `instrument_anthropic`'s wrapper actually reads. Not the
whole response object, not the client, not anything the recorder never
touches. A fingerprint that is too strict refuses a working SDK after a
harmless minor release, which is worse than the partial span it was meant to
prevent — the user loses tracing entirely over a field nobody uses.

Anthropic only. That is the one instrumentation that exists. Three providers
verified beats seventy claimed; one verified beats three guessed.

## Why the response shape is checked against the CLASS

The obvious check is "call it and look at the result", which needs a network
round trip, an API key and money. Instead this reads the response model's
declared fields — `anthropic.types.Message` and `Usage` are Pydantic models
and carry `model_fields` — so the check is free, offline, and runs at import.

A class that is not introspectable is UNKNOWN, not broken. Refusing on
"I could not tell" would take tracing away from anybody whose SDK is merely
unusual, so unknown patches with a warning and marks its steps `partial`.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

# The attributes the wrapper reads, and nothing else. Each entry is
# (owner, attribute, required).
#
# `required=False` marks a field Anthropic returns only sometimes: the cache
# counts appear only when a cache was in play. Their ABSENCE FROM THE MODEL is
# still worth reporting — it means this SDK version cannot ever supply them —
# but it must not block patching, because token counting works without them.
USAGE_FIELDS = (
    ("Usage", "input_tokens", True),
    ("Usage", "output_tokens", True),
    ("Usage", "cache_read_input_tokens", False),
    ("Usage", "cache_creation_input_tokens", False),
)

MESSAGE_FIELDS = (
    ("Message", "usage", True),
    ("Message", "content", True),
)

# The keyword the wrapper reads off the call to name the step.
CALL_KWARGS = ("model",)

# Set this to skip the gate. Named in every refusal, because a refusal with no
# way past it is a dead end for somebody whose SDK is fine and whose
# fingerprint is merely stale.
FORCE_ENV = "MODELMRI_FORCE_INSTRUMENT"


@dataclass
class Report:
    """Whether this SDK can be instrumented, and what moved if not."""

    package: str = "anthropic"
    version: str = ""
    installed: bool = False
    ok: bool = False
    # "full" — every field the wrapper reads is present.
    # "partial" — the required ones are there, some optional ones are not, or
    #             the shape could not be introspected at all.
    # "none" — a required field moved; the wrapper would produce empty spans.
    capture: str = "none"
    missing: list = field(default_factory=list)
    missing_optional: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    forced: bool = False

    def to_dict(self) -> dict:
        return {
            "package": self.package,
            "version": self.version,
            "installed": self.installed,
            "ok": self.ok,
            "capture": self.capture,
            "missing": list(self.missing),
            "missing_optional": list(self.missing_optional),
            "notes": list(self.notes),
            "forced": self.forced,
            "reason": self.reason(),
        }

    def reason(self) -> str:
        """One line, naming the package, the version and what moved."""
        if not self.installed:
            return "anthropic is not installed, so there is nothing to instrument."
        where = f"anthropic {self.version}" if self.version else "anthropic"
        if self.forced:
            return (
                f"{where}: instrumenting anyway because {FORCE_ENV} is set. "
                f"Spans may have empty fields."
            )
        if self.missing:
            moved = ", ".join(self.missing)
            return (
                f"{where} no longer exposes {moved}, which the recorder reads "
                f"to fill a step's token counts. NOT instrumenting: a span "
                f"with empty token fields is indistinguishable from a provider "
                f"that reported nothing, and would be recorded as one. Set "
                f"{FORCE_ENV}=1 to patch regardless, or upgrade modelmri-record."
            )
        if self.missing_optional:
            absent = ", ".join(self.missing_optional)
            return (
                f"{where}: instrumented. This version does not carry {absent}, "
                f"so those columns will read 'not reported by provider' for "
                f"every call — which is true of this SDK, not of your usage."
            )
        if self.notes:
            return f"{where}: instrumented. " + " ".join(self.notes)
        return f"{where}: instrumented, every field the recorder reads is present."


def _model_field_names(cls) -> set | None:
    """The declared field names of a response model, or None if unreadable.

    Pydantic v2 exposes `model_fields`, v1 `__fields__`. A class carrying
    neither is not necessarily broken — it may be a plain object or a stub —
    so this returns None for "could not tell" rather than an empty set, which
    would read as "has no fields" and fail every check.

    An EMPTY `model_fields` counts as "could not tell" for the same reason.
    A response model with genuinely zero fields is not a thing Anthropic
    ships; an empty dict means something else is going on — a stub, a lazy
    proxy, a mock. An SDK that renamed a field has DIFFERENT fields, and that
    case is caught below. Reading empty as "everything moved" would refuse to
    instrument anyone whose SDK is merely wrapped.
    """
    for attr in ("model_fields", "__fields__"):
        found = getattr(cls, attr, None)
        if isinstance(found, dict) and found:
            return set(found)
    # A dataclass or plain annotated class.
    hints = getattr(cls, "__annotations__", None)
    if isinstance(hints, dict) and hints:
        return set(hints)
    return None


def check(force: bool = False) -> Report:
    """Fingerprint the installed anthropic SDK. Never raises."""
    import os

    out = Report(forced=bool(force) or bool(os.environ.get(FORCE_ENV)))
    try:
        import anthropic
    except Exception:
        return out
    out.installed = True
    out.version = str(getattr(anthropic, "__version__", "") or "")

    # The call site. `Messages.create` is what gets patched, so a signature
    # that no longer accepts the keyword the wrapper reads is a real break.
    try:
        from anthropic.resources.messages import Messages

        sig = inspect.signature(Messages.create)
        params = sig.parameters
        takes_kwargs = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        for name in CALL_KWARGS:
            if name not in params and not takes_kwargs:
                out.missing.append(f"Messages.create(..., {name}=)")
    except Exception as err:
        out.notes.append(
            f"Could not read Messages.create's signature ({type(err).__name__}); "
            f"proceeding, and the step name may fall back to 'anthropic'."
        )

    # The response shape, from the declared models — no network, no key.
    try:
        from anthropic.types import Message, Usage

        message_fields = _model_field_names(Message)
        usage_fields = _model_field_names(Usage)
    except Exception as err:
        out.notes.append(
            f"Could not import anthropic.types.Message/Usage "
            f"({type(err).__name__}), so the response shape was not checked."
        )
        message_fields = usage_fields = None

    unknown = False
    for owner, fields, table in (
        ("Message", message_fields, MESSAGE_FIELDS),
        ("Usage", usage_fields, USAGE_FIELDS),
    ):
        if fields is None:
            unknown = True
            continue
        for cls_name, attr, required in table:
            if cls_name != owner or attr in fields:
                continue
            (out.missing if required else out.missing_optional).append(
                f"{owner}.{attr}"
            )

    if unknown:
        out.notes.append(
            "The response models could not be introspected, so this is "
            "'unknown', not 'broken' — patching anyway and marking captures "
            "partial."
        )

    if out.missing and not out.forced:
        out.ok = False
        out.capture = "none"
    else:
        out.ok = True
        if out.missing or unknown or out.missing_optional:
            out.capture = "partial"
        else:
            out.capture = "full"
    return out
