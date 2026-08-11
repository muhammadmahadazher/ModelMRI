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
    """Every name is resolved live, and the picker's order is still the
    curated one.

    These are two different claims and only one of them is about ordering.
    `suggested` resolves through a ThreadPoolExecutor, so the order the
    lookups *happen* in is whatever the scheduler chose — comparing `seen`
    against SUGGESTED positionally asserted a property the code never had,
    and duly went red once the machine was busy enough to reorder two of the
    eight. What must hold is that each name is looked up exactly once, in any
    order, and that `pool.map` hands the rows back in the curated order,
    because that order is what the user reads down.
    """
    seen: list[str] = []

    def fake_resolve(name, timeout=10.0):
        seen.append(name)  # list.append is atomic under the GIL; the order is not
        return {"found": True, "bytes": 2_000_000_000, "name": name, "error": ""}

    monkeypatch.setattr(ollama, "resolve", fake_resolve)
    out = ollama.suggested(vram_gb=8.0)

    assert sorted(seen) == sorted(ollama.SUGGESTED), (
        "every entry must be looked up exactly once"
    )
    assert [row["name"] for row in out] == list(ollama.SUGGESTED), (
        "the picker must keep the curated order"
    )
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


def test_the_fit_verdict_is_not_a_constant(monkeypatch):
    """It was one, and that is the whole reason this test exists.

    `fits` used `max(4 * vram_gb, 20.0)` — the ceiling for refusing a
    DOWNLOAD, which is deliberately generous because a model too big for VRAM
    still runs on the CPU. Against a 20 GB floor every curated entry is under
    6 GB, so the answer was True on a 24 GB card and True on a 1 GB card
    alike: a chip that looked like a measurement and was a constant.
    """
    monkeypatch.setattr(
        ollama,
        "resolve",
        lambda name, timeout=10.0: {"found": True, "bytes": 5_000_000_000},
    )
    verdicts = {
        vram: ollama.suggested(vram_gb=vram)[0]["fits"] for vram in (2.0, 8.6, 24.0)
    }
    assert verdicts[2.0] is False, "5 GB cannot run on a 2 GB card"
    assert verdicts[24.0] is True
    assert len(set(verdicts.values())) > 1, f"still constant across GPUs: {verdicts}"


def test_a_base_ollama_tag_is_not_called_instruction_tuned(monkeypatch):
    """Ollama publishes base tags — llama3.2:1b-text-fp16, qwen2.5:0.5b-base.
    Reporting every Ollama model as instruction-tuned silenced the base-model
    caveat for exactly the models that need it."""
    import json as _json
    import urllib.request
    from contextlib import contextmanager

    def answer(template: str):
        @contextmanager
        def _open(_req, timeout=None):
            class R:
                def read(self_inner):
                    return _json.dumps({"template": template}).encode()

            yield R()

        return _open

    monkeypatch.setattr(urllib.request, "urlopen", answer("{{ .Prompt }}"))
    assert ollama.is_instruct("qwen3:0.6b") is True

    monkeypatch.setattr(urllib.request, "urlopen", answer(""))
    assert ollama.is_instruct("llama3.2:1b-text-fp16") is False

    def boom(_req, timeout=None):
        raise OSError("daemon down")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert ollama.is_instruct("anything") is None, "unknown must not read as base"


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
