# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Strip credentials out of a trace before it leaves the process.

An agent recorder writes prompts to disk and posts them over a socket. Those
prompts routinely contain the key the agent was handed, because people put
credentials in system prompts and tools echo their own config. A tracing tool
that ships those verbatim is a liability dressed as an observability feature,
and "be careful what you log" is not a control.

So redaction is ON by default and has to be switched off deliberately.

The patterns are deliberately narrow: known credential shapes with distinctive
prefixes, not "anything that looks high-entropy". A greedy redactor that eats
hashes, UUIDs and base64 payloads makes traces useless, and a useless trace
gets the whole feature turned off -- which protects nobody.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

Redactor = Callable[[str], str]

# Each entry is (name, compiled pattern). The pattern must match the SECRET,
# not its surroundings, so the replacement can name what it removed.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Anthropic, OpenAI and friends: sk-ant-..., sk-proj-..., sk-...
    ("api-key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")),
    # HuggingFace
    ("hf-token", re.compile(r"\bhf_[A-Za-z0-9]{16,}")),
    # PyPI
    ("pypi-token", re.compile(r"\bpypi-[A-Za-z0-9_-]{16,}")),
    # GitHub: ghp_ (classic), gho_/ghu_/ghs_/ghr_, github_pat_
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}")),
    ("github-pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}")),
    # Slack
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    # Google API keys
    ("google-key", re.compile(r"\bAIza[A-Za-z0-9_-]{30,}")),
    # AWS access key ids (the secret itself is unguessable, but the id is a
    # reliable marker that credentials are in the payload)
    ("aws-key-id", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    # Bearer tokens in a header dump
    ("bearer", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{16,}")),
    # PEM private keys, whole block
    (
        "private-key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.S,
        ),
    ),
    # ...and a block whose tail was cut off. The recorder truncates payloads
    # to 2-4k BEFORE redaction runs, so a real key arrives here headless: the
    # -----END sentinel is gone and the rule above cannot match, leaving the
    # base64 body in the clear. Found in the wild against 0.1.0. Anchored on
    # the BEGIN line so it cannot run away on ordinary prose.
    (
        "private-key",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*", re.S),
    ),
    # The mirror case: truncated from the front, leaving a dangling END.
    (
        "private-key",
        re.compile(r"[A-Za-z0-9+/=\s]{64,}-----END [A-Z ]*PRIVATE KEY-----", re.S),
    ),
]


def default_redactor(text: str) -> str:
    """Replace known credential shapes with a labelled placeholder."""
    if not text:
        return text
    for name, pat in _PATTERNS:
        text = pat.sub(f"[redacted:{name}]", text)
    return text


def make_redactor(
    extra: Iterable[str | re.Pattern[str]] = (),
    *,
    include_defaults: bool = True,
) -> Redactor:
    """Build a redactor with your own patterns on top of the defaults.

    Anything your tools embed -- an internal token prefix, a customer id
    format -- belongs here, because only you know its shape.
    """
    compiled = [p if isinstance(p, re.Pattern) else re.compile(p) for p in extra]

    def redactor(text: str) -> str:
        if not text:
            return text
        if include_defaults:
            text = default_redactor(text)
        for pat in compiled:
            text = pat.sub("[redacted:custom]", text)
        return text

    return redactor


def redact_document(doc: dict, redactor: Redactor) -> dict:
    """Apply a redactor to every free-text field of a trace document.

    Only the fields that carry model or tool payloads: names, ids, timings and
    token counts are structural and stay intact, because redacting them would
    destroy the trace without protecting anything.
    """
    for s in doc.get("steps", []):
        for field in ("input", "output"):
            v = s.get(field)
            if isinstance(v, str):
                s[field] = redactor(v)
    return doc
