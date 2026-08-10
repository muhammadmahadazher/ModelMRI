"""The Ollama tab should start somewhere, and its numbers should be real.

The HuggingFace tab opens on curated picks annotated with whether they fit
your GPU. The Ollama tab opened on an empty box — and its "or one of these"
strip carried sizes typed into the source: `{"size": "2.6 GB"}`. That is a
figure nobody rechecks, attached to tags that get republished, which is the
same species as every wrong number this project has had to correct.

Names are curated. Sizes are resolved, and marked against the GPU actually
present — never a constant, never this developer's card.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from modelmri import ollama

SOURCE = Path(ollama.__file__).read_text("utf-8")


def test_the_curated_list_carries_names_and_nothing_else():
    """A size beside a name in source is a size that goes stale silently."""
    assert ollama.SUGGESTED, "the tab needs somewhere to start"
    assert all(isinstance(n, str) for n in ollama.SUGGESTED)

    # And no hand-written size survives as a VALUE anywhere in the module.
    # Walking the AST rather than the text on purpose: the comment explaining
    # why the old "2.6 GB" strings were removed contains "2.6 GB", and a
    # check that cannot tell an explanation from the thing it explains would
    # forbid documenting the fix.
    literals = [
        node.value
        for node in ast.walk(ast.parse(SOURCE))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    hardcoded = [
        s for s in literals if re.fullmatch(r"\s*\d+(?:\.\d+)?\s*(?:GB|MB|TB)\s*", s)
    ]
    assert not hardcoded, f"hand-written sizes are back: {hardcoded}"


def test_sizes_come_from_the_registry(monkeypatch):
    seen: list[str] = []

    def fake_resolve(name, timeout=10.0):
        seen.append(name)
        return {"found": True, "bytes": 2_000_000_000, "name": name, "error": ""}

    monkeypatch.setattr(ollama, "resolve", fake_resolve)
    out = ollama.suggested(vram_gb=8.0)

    assert seen == list(ollama.SUGGESTED), "every entry must be looked up"
    assert all(row["size_gb"] == 2.0 for row in out)


def test_the_fit_verdict_uses_this_machines_gpu(monkeypatch):
    """`fits` must be about the GPU in front of the user, and must be None —
    not False — when there is nothing to judge against. "Unknown" and "too
    big" are different answers."""
    monkeypatch.setattr(
        ollama,
        "resolve",
        lambda name, timeout=10.0: {"found": True, "bytes": 60_000_000_000},
    )
    big = ollama.suggested(vram_gb=8.0)
    assert all(row["fits"] is False for row in big), "60 GB does not fit 8 GB"

    roomy = ollama.suggested(vram_gb=80.0)
    assert all(row["fits"] is True for row in roomy)

    unknown = ollama.suggested(vram_gb=None)
    assert all(row["fits"] is None for row in unknown), "no GPU is not 'too big'"


def test_a_registry_that_cannot_be_reached_still_offers_the_names(monkeypatch):
    """Offline, a picker with names and no metadata beats an empty one."""

    def boom(name, timeout=10.0):
        raise OSError("no network")

    monkeypatch.setattr(ollama, "resolve", boom)
    out = ollama.suggested(vram_gb=8.0)
    assert [row["name"] for row in out] == list(ollama.SUGGESTED)
    assert all(row["size_gb"] == 0.0 and row["fits"] is None for row in out)


def test_status_still_names_somewhere_to_start_when_ollama_is_down():
    down = ollama.status(host="http://127.0.0.1:1", timeout=0.2)
    assert down["up"] is False
    assert down["suggested"] == ollama.SUGGESTED
