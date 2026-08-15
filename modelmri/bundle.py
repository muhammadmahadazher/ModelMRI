"""Redact and budget a bundle BEFORE it leaves the machine.

`.mri` export is the one path in this project where data goes from a laptop to
somebody else, and two things about it are load-bearing.

## The recorder's redaction does not cover this

`modelmri_record` redacts at DELIVERY — the moment a finished trace is posted
to the server. A bundle assembled here is built from steps ALREADY IN THE
STORE, so that pass has run and is behind us; anything the user added to the
store by another route (an imported document, an OTLP ingest, a hand-written
JSON) never went through it at all.

So export redacts again, at export, over both halves: the trace's step
payloads AND the `.mri`'s own prompt and generation. Redacting twice is
cheap. Redacting once, in the wrong place, is a credential in a file somebody
posts in a GitHub issue.

The patterns come from `modelmri_record.redact`, not from a copy here. There
was an in-tree copy of that recorder once; it drifted, and what it had lost
was precisely the credential redaction. One implementation.

## What is about to leave is shown before it leaves

`preview()` returns what the bundle will contain — how many redactions fired
and of what kind, how many steps, how many bytes, what was truncated. A
share button that ships a file without showing what is in it is asking the
user to trust a process they cannot see.

## The trace half has its own budget

An `.mri` is ~54 KB. A 400-step run with 20,000-character payloads is two
orders of magnitude bigger and would turn a shareable artefact into
something nobody can open. Steps are capped and payloads clipped — and every
cut is REPORTED, never silent, because a truncated trace that reads as a
complete one is how you debug the wrong thing for an hour.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .errors import BadRequest

# One `.mri` of a normal analysis is around 54 KB. The trace section gets its
# own budget rather than sharing that one, so a bundle stays openable.
MAX_TRACE_STEPS = 500
MAX_STEP_TEXT = 4_000
# Above this the whole trace section is refused rather than trimmed to fit:
# silently shipping 500 of 20,000 steps and calling it "the run" is a
# different artefact from the one somebody asked to share.
HARD_STEP_LIMIT = 20_000


class BundleError(BadRequest):
    """This bundle cannot be built honestly, and the message says why.

    A `BadRequest`, not a new root type. Every message here is authored — the
    `_PUBLISHED` invariant in `test_no_machine_leaks` is what makes it safe for
    a route to stringify one — and inheriting means the server's existing
    `except BadRequest` answers 422 without a special case. A separate root
    type needed its own handler, and a handler placed above `except Refusal`
    would have swallowed every refusal on that route.
    """


def _patterns():
    """The credential shapes, from the recorder. Never a copy."""
    from .record import redact

    return redact


@dataclass
class Redaction:
    """One kind of secret, and how many times it was found."""

    label: str
    count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Preview:
    """Exactly what is about to leave the machine."""

    n_steps: int = 0
    n_steps_dropped: int = 0
    n_payloads_clipped: int = 0
    chars_clipped: int = 0
    redactions: list = field(default_factory=list)
    fields_scanned: int = 0

    @property
    def n_redactions(self) -> int:
        return sum(r.count for r in self.redactions)

    def to_dict(self) -> dict:
        return {
            "n_steps": self.n_steps,
            "n_steps_dropped": self.n_steps_dropped,
            "n_payloads_clipped": self.n_payloads_clipped,
            "chars_clipped": self.chars_clipped,
            "redactions": [r.to_dict() for r in self.redactions],
            "n_redactions": self.n_redactions,
            "fields_scanned": self.fields_scanned,
            "means": self.means(),
        }

    def means(self) -> str:
        parts = [f"{self.n_steps} step(s) will be in this file."]
        if self.redactions:
            kinds = ", ".join(f"{r.count}x {r.label}" for r in self.redactions)
            parts.append(
                f"{self.n_redactions} credential-shaped value(s) were replaced "
                f"before writing: {kinds}."
            )
        else:
            parts.append(
                "No credential-shaped values were found. That is not a "
                "guarantee — the patterns cover known shapes, and your own "
                "tokens may not look like any of them."
            )
        if self.n_steps_dropped:
            parts.append(
                f"{self.n_steps_dropped} step(s) are NOT in this file: the "
                f"trace section holds {MAX_TRACE_STEPS} and the run was longer."
            )
        if self.n_payloads_clipped:
            parts.append(
                f"{self.n_payloads_clipped} payload(s) were clipped to "
                f"{MAX_STEP_TEXT} characters ({self.chars_clipped:,} characters "
                f"not included), and each carries a marker saying so."
            )
        return " ".join(parts)


def _scan_and_redact(text: str, tally: dict) -> str:
    """Redact `text`, counting what fired so the preview can report it.

    Applies the patterns SEQUENTIALLY, exactly as `default_redactor` does,
    rather than counting each against the original. Two patterns can match the
    same value — `bearer` and `api-key` overlap on an `Authorization:` header
    — and counting both against the untouched text would report two secrets
    where there was one. The number in front of the user has to be the number
    of things actually replaced.
    """
    if not isinstance(text, str) or not text:
        return text if isinstance(text, str) else ""
    redact = _patterns()
    for name, pattern in redact._PATTERNS:
        text, n = pattern.subn(f"[redacted:{name}]", text)
        if n:
            tally[name] = tally.get(name, 0) + n
    return text


def _clip(text: str, preview: Preview) -> str:
    """Cut a payload to budget, MARKING the cut rather than hiding it."""
    if not isinstance(text, str) or len(text) <= MAX_STEP_TEXT:
        return text if isinstance(text, str) else ""
    cut = len(text) - MAX_STEP_TEXT
    preview.n_payloads_clipped += 1
    preview.chars_clipped += cut
    return text[:MAX_STEP_TEXT] + f"\n… [{cut:,} characters not included]"


def prepare(
    trace: dict | None,
    *,
    prompt: str = "",
    generation: str = "",
    step_ref: str = "",
) -> tuple:
    """(clean trace, clean prompt, clean generation, Preview).

    Never mutates the caller's document: the store hands out live dicts, and
    redacting one in place would edit the user's own recorded trace.
    """
    preview = Preview()
    tally: dict = {}

    clean_prompt = _scan_and_redact(prompt, tally)
    clean_generation = _scan_and_redact(generation, tally)
    preview.fields_scanned += 2

    clean_trace = None
    if trace is not None:
        if not isinstance(trace, dict):
            raise BundleError("a trace section must be an object with a 'steps' list.")
        steps = trace.get("steps")
        if not isinstance(steps, list):
            raise BundleError("a trace section must be an object with a 'steps' list.")
        if len(steps) > HARD_STEP_LIMIT:
            raise BundleError(
                f"this run has {len(steps):,} steps. Shipping the first "
                f"{MAX_TRACE_STEPS} and calling it the run would be a different "
                f"artefact from the one you asked to share, so this refuses "
                f"rather than trimming. Export the failing subtree instead."
            )

        kept = []
        for step in steps[:MAX_TRACE_STEPS]:
            if not isinstance(step, dict):
                continue
            copy = dict(step)
            for name in ("input", "output"):
                value = copy.get(name)
                if isinstance(value, str):
                    copy[name] = _clip(_scan_and_redact(value, tally), preview)
                    preview.fields_scanned += 1
            # `meta` carries machine facts — model id, token ids, dtype. Never
            # prompt text by contract, but a hand-written or ingested document
            # is not bound by that contract, so its strings are scanned too.
            meta = copy.get("meta")
            if isinstance(meta, dict):
                copy["meta"] = {
                    k: (_scan_and_redact(v, tally) if isinstance(v, str) else v)
                    for k, v in meta.items()
                }
                preview.fields_scanned += sum(
                    1 for v in meta.values() if isinstance(v, str)
                )
            kept.append(copy)

        preview.n_steps = len(kept)
        preview.n_steps_dropped = max(0, len(steps) - len(kept))
        clean_trace = {
            "id": str(trace.get("id") or ""),
            "name": _scan_and_redact(str(trace.get("name") or ""), tally),
            "started_at": str(trace.get("started_at") or ""),
            "steps": kept,
            "n_steps_total": len(steps),
            "truncated": preview.n_steps_dropped,
        }
        if step_ref:
            if not any(str(s.get("id") or "") == step_ref for s in kept):
                raise BundleError(
                    f"step {step_ref!r} is not among the steps being exported, "
                    f"so the viewer would open a bundle whose highlighted step "
                    f"is not in it."
                )
            clean_trace["step_ref"] = step_ref

    preview.redactions = [
        Redaction(label=k, count=v) for k, v in sorted(tally.items())
    ]
    return clean_trace, clean_prompt, clean_generation, preview


def preview(trace: dict | None, *, prompt: str = "", generation: str = "") -> Preview:
    """What a bundle would contain, without building one."""
    return prepare(trace, prompt=prompt, generation=generation)[3]
