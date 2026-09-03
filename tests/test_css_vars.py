# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Custom properties that silently invalidate the declaration they're in.

This project has now shipped this bug three times, and it is invisible every
time — an undefined or cyclic `var()` does not warn, does not fall back, and
does not fail a build. It makes the WHOLE declaration invalid at
computed-value time, so the property falls back to its initial value and the
element simply renders as if you never wrote the line.

  * the progress bar referenced `--color-accent`, which does not exist (the
    token is `--color-cobalt`), so the fill painted nothing. It had been
    "verified" by reading its width, which was correct.
  * the attention arcs referenced `--model`, which does not exist; canvas
    silently ignores an unparseable strokeStyle and drew in default black.
  * `--glass-fill: var(--glass-fill)` is a cycle, so every liquid-glass
    surface computed to `transparent`. The model picker had blur and no
    frost, and the hero headline read straight through the model list.

Both failure modes are mechanical, so a test can find them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CSS = Path(__file__).resolve().parents[1] / "frontend" / "src" / "styles.css"

# A `var(--x)` with a fallback is safe even when --x is never defined in CSS:
# that is the correct way to read a property set from JS at runtime.
_USE_NO_FALLBACK = re.compile(r"var\(\s*(--[A-Za-z0-9_-]+)\s*\)")
_DEFINE = re.compile(r"(--[A-Za-z0-9_-]+)\s*:")
_DECL = re.compile(r"(--[A-Za-z0-9_-]+)\s*:([^;{}]*)[;}]")


def _strip_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", " ", text, flags=re.S)


@pytest.fixture(scope="module")
def css() -> str:
    assert CSS.is_file(), f"{CSS} is missing"
    return _strip_comments(CSS.read_text(encoding="utf-8"))


def test_no_custom_property_references_itself(css: str) -> None:
    """`--x: var(--x)` is a cycle, and a cycle computes to nothing."""
    cycles = []
    for match in _DECL.finditer(css):
        name, value = match.group(1), match.group(2)
        if re.search(rf"var\(\s*{re.escape(name)}\s*[,)]", value):
            cycles.append(f"{name}:{value.strip()[:60]}")
    assert cycles == [], (
        "self-referential custom properties compute to the guaranteed-invalid "
        f"value, silently voiding every declaration that reads them: {cycles}"
    )


def test_every_var_without_a_fallback_is_defined(css: str) -> None:
    """An undefined `var(--x)` with no fallback voids its whole declaration."""
    defined = set(_DEFINE.findall(css))
    used = set(_USE_NO_FALLBACK.findall(css))
    missing = sorted(used - defined)
    assert missing == [], (
        "these are read but never defined, so any declaration using them is "
        f"dropped entirely rather than falling back: {missing}. Either define "
        "them, or give the var() a fallback if it is set from JS."
    )


def test_the_glass_surfaces_still_have_a_fill(css: str) -> None:
    """Guards the specific regression: glass with blur but no frost.

    The token has to resolve to something with alpha, in both themes, or the
    model picker becomes a transparent hole with the page legible through it.
    """
    fills = re.findall(r"--glass-fill:\s*([^;]+);", css)
    assert len(fills) >= 2, f"expected a --glass-fill per theme, found {fills}"
    for value in fills:
        assert "var(--glass-fill" not in value, f"cyclic: {value}"
        alpha = re.search(r"/\s*([0-9.]+)\s*\)", value)
        assert alpha, f"--glass-fill has no alpha channel: {value!r}"
        assert 0.3 <= float(alpha.group(1)) <= 0.95, (
            f"--glass-fill alpha {alpha.group(1)} — too transparent to read "
            f"text over a busy backdrop, or too opaque to be glass: {value!r}"
        )


def test_the_scrim_actually_blurs(css: str) -> None:
    """The scrim carries the whole effect, so its blur is load-bearing.

    An element with backdrop-filter becomes a backdrop root for its
    descendants, so .sheet's own blur(40px) never sampled the page behind it —
    only this scrim's flat tint. Whatever is set here is what the user sees.
    """
    block = re.search(r"\.sheet-scrim\s*\{(.*?)\}", css, re.S)
    assert block, ".sheet-scrim rule not found"
    blur = re.search(r"backdrop-filter:[^;]*?blur\(\s*([0-9.]+)px", block.group(1))
    assert blur, f"the scrim has no backdrop blur: {block.group(1)}"
    assert float(blur.group(1)) >= 12, (
        f"scrim blur is {blur.group(1)}px. It shipped at 3px, which left an "
        "animated hero fully legible through the model picker."
    )
