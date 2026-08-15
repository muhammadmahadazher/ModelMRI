"""The generator must not eat the documentation it is checking.

`scripts/gen_api_docs.py --check` runs in CI, and when it fails it prints one
instruction: run the generator. That instruction used to delete 498 lines --
every hand-written section of the API reference, which is where the worked
examples for patchscopes, path patching, the finetune diff, grounding, probes,
head types, DLA, the two lenses and receipts live. The script rewrote the whole
file from the OpenAPI schema, and only the route tables come from there.

A check whose remedy destroys the thing it is checking is worse than no check,
and it fails in the least visible way: the docs build still succeeds, the
`--check` goes green, and the loss shows up only when somebody goes looking for
a section that is no longer there.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gen_api_docs.py"
PAGE = ROOT / "docs" / "reference" / "api.md"


@pytest.fixture(scope="module")
def gen():
    pytest.importorskip("fastapi")
    # spec/exec rather than SourceFileLoader.load_module, which is deprecated
    # and removed in 3.15 — this suite runs on 3.10 through 3.13 and should
    # not be emitting a warning that outlives the versions it supports.
    spec = importlib.util.spec_from_file_location("gen_api_docs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_shipped_page_carries_the_fence(gen):
    """Without the markers the generator has no way to know what is hand
    written, and falls back to rewriting the whole page."""
    page = PAGE.read_text(encoding="utf-8")
    assert gen.BEGIN in page
    assert gen.END in page
    assert page.index(gen.BEGIN) < page.index(gen.END)


def test_regenerating_preserves_everything_outside_the_fence(gen):
    """The exact failure: hand-written prose after the tables was dropped."""
    page = PAGE.read_text(encoding="utf-8")
    fresh = gen.render(page)

    before = page.split(gen.BEGIN, 1)[0]
    after = page.split(gen.END, 1)[1]
    assert fresh.startswith(before)
    assert fresh.endswith(after)
    # Named, not just counted: these are the sections that were lost.
    for section in (
        "## Patchscopes",
        "## Path patching",
        "## Grounding",
        "## Direct logit attribution",
        "## The two lenses",
        "## Receipts",
    ):
        assert section in fresh, f"{section} did not survive a regeneration"


def test_regenerating_is_idempotent(gen):
    """A second run must be a no-op, or `--check` can never settle."""
    page = PAGE.read_text(encoding="utf-8")
    once = gen.render(page)
    assert gen.render(once) == once


def test_the_page_is_up_to_date_with_the_routes(gen):
    """What CI's `--check` asserts, so a stale page fails here first — with
    the whole suite's context rather than one line of a workflow log."""
    page = PAGE.read_text(encoding="utf-8")
    assert gen.render(page) == page, (
        "docs/reference/api.md no longer matches the app's routes — run "
        "`uv run python scripts/gen_api_docs.py`"
    )


def test_a_page_with_no_fence_is_written_whole(gen):
    """The first-run path. Nothing to preserve, because nothing said what was
    worth preserving — but it must put the fence in so the NEXT run does."""
    fresh = gen.render("")
    assert gen.BEGIN in fresh
    assert gen.END in fresh
    assert "# HTTP API" in fresh
    assert "## Status codes" in fresh


def test_the_openai_surface_is_documented(gen):
    """It shipped implemented, tested and entirely absent from the reference:
    a compatibility layer nobody could discover."""
    page = PAGE.read_text(encoding="utf-8")
    for route in ("/v1/models", "/v1/chat/completions", "/v1/completions"):
        assert route in page
