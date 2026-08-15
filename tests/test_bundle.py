"""Redaction and budget for a file that is about to leave the machine.

The recorder redacts at DELIVERY. A bundle built here comes from steps already
in the store, so that pass is behind us — and a document that arrived by
import or OTLP ingest never went through it at all. Export redacts again, and
these tests are about that second pass actually happening.
"""

from __future__ import annotations

import pytest

from modelmri import bundle

# Shapes the recorder's own patterns cover. Real-looking, none real.
API_KEY = "sk-ant-api03-" + "A" * 88
GH_TOKEN = "ghp_" + "B" * 36
AWS_ID = "AKIA" + "C" * 16


def _trace(n_steps=3, payload="hello"):
    return {
        "id": "t1",
        "name": "a run",
        "started_at": "2026-08-14T00:00:00Z",
        "steps": [
            {
                "id": f"s{i}",
                "kind": "llm_call",
                "name": "call",
                "input": payload,
                "output": "fine",
                "started_ms": i * 10,
                "meta": {"model": "m"},
            }
            for i in range(n_steps)
        ],
    }


# ------------------------------------------------------ it actually redacts


def test_a_credential_in_a_step_does_not_reach_the_file():
    doc = _trace(n_steps=1, payload=f"call it with {API_KEY} please")
    clean, _, _, preview = bundle.prepare(doc)
    text = clean["steps"][0]["input"]
    assert API_KEY not in text
    assert "[redacted:api-key]" in text
    assert preview.n_redactions == 1


def test_a_credential_in_the_prompt_does_not_reach_the_file():
    """The `.mri` half, not only the trace half."""
    _, prompt, gen, preview = bundle.prepare(
        None, prompt=f"my key is {API_KEY}", generation=f"and {GH_TOKEN}"
    )
    assert API_KEY not in prompt and GH_TOKEN not in gen
    assert preview.n_redactions == 2


def test_a_credential_in_step_meta_is_scanned_too():
    """`meta` carries machine facts by contract, but an imported or
    hand-written document is not bound by that contract."""
    doc = _trace()
    doc["steps"][0]["meta"] = {"model": "m", "note": f"token {GH_TOKEN}"}
    clean, _, _, preview = bundle.prepare(doc)
    assert GH_TOKEN not in clean["steps"][0]["meta"]["note"]
    assert preview.n_redactions == 1


def test_the_trace_name_is_scanned():
    doc = _trace()
    doc["name"] = f"run with {AWS_ID}"
    clean, _, _, _ = bundle.prepare(doc)
    assert AWS_ID not in clean["name"]


def test_several_kinds_are_reported_separately():
    doc = _trace(payload=f"{API_KEY} and {GH_TOKEN} and {AWS_ID}")
    _, _, _, preview = bundle.prepare(doc)
    labels = {r.label for r in preview.redactions}
    assert {"api-key", "github-token", "aws-key-id"} <= labels
    assert preview.n_redactions >= 3


def test_the_caller_document_is_never_mutated():
    """The store hands out live dicts; redacting one in place would edit the
    user's own recorded trace."""
    doc = _trace(payload=f"secret {API_KEY}")
    original = doc["steps"][0]["input"]
    bundle.prepare(doc)
    assert doc["steps"][0]["input"] == original
    assert API_KEY in doc["steps"][0]["input"]


def test_the_scan_and_the_recorders_redactor_agree():
    """`_scan_and_redact` reimplements the pattern loop so it can COUNT what
    fired. If that ever diverges from `default_redactor`, the file would be
    redacted differently from what the recorder promises."""
    from modelmri.record import redact

    samples = [
        f"key {API_KEY} here",
        f"Authorization: Bearer {GH_TOKEN}",
        f"aws {AWS_ID} and again {AWS_ID}",
        "nothing sensitive at all",
        "",
    ]
    for text in samples:
        mine = bundle._scan_and_redact(text, {})
        theirs = redact.default_redactor(text)
        assert mine == theirs, f"diverged on {text[:40]!r}"


def test_overlapping_patterns_are_not_double_counted():
    """Two patterns can match one value; reporting two secrets where there
    was one makes the preview a number nobody can trust."""
    doc = _trace(payload=f"Authorization: Bearer {API_KEY}")
    _, _, _, preview = bundle.prepare(doc)
    # However the patterns split it, the count is what was actually replaced.
    clean, _, _, _ = bundle.prepare(doc)
    assert API_KEY not in clean["steps"][0]["input"]
    assert preview.n_redactions == sum(r.count for r in preview.redactions)


def test_a_clean_run_says_no_guarantee_was_made():
    """ "Nothing found" is not "nothing there" — the patterns cover known
    shapes and somebody's internal token may not look like any of them."""
    _, _, _, preview = bundle.prepare(_trace())
    assert preview.n_redactions == 0
    assert "not a guarantee" in preview.means()


# --------------------------------------------------------------- the budget


def test_a_long_run_is_capped_and_says_so():
    doc = _trace(n_steps=bundle.MAX_TRACE_STEPS + 40)
    clean, _, _, preview = bundle.prepare(doc)
    assert len(clean["steps"]) == bundle.MAX_TRACE_STEPS
    assert clean["truncated"] == 40
    assert clean["n_steps_total"] == bundle.MAX_TRACE_STEPS + 40
    assert preview.n_steps_dropped == 40
    assert "are NOT in this file" in preview.means()


def test_a_huge_payload_is_clipped_with_a_marker():
    doc = _trace(n_steps=1, payload="x" * 30_000)
    clean, _, _, preview = bundle.prepare(doc)
    text = clean["steps"][0]["input"]
    assert len(text) < 30_000
    assert "characters not included" in text
    assert preview.n_payloads_clipped == 1
    assert preview.chars_clipped == 30_000 - bundle.MAX_STEP_TEXT


def test_a_run_beyond_the_hard_limit_is_refused_not_trimmed():
    """Shipping the first 500 of 20,000 steps and calling it 'the run' is a
    different artefact from the one somebody asked to share."""
    doc = _trace(n_steps=bundle.HARD_STEP_LIMIT + 1)
    with pytest.raises(bundle.BundleError, match="refuses rather than trimming"):
        bundle.prepare(doc)


def test_the_budget_keeps_a_realistic_bundle_small():
    """An .mri is ~54 KB. The trace half must not turn that into something
    nobody can open."""
    import gzip
    import json

    doc = _trace(n_steps=400, payload="a tool call payload " * 60)
    clean, _, _, _ = bundle.prepare(doc)
    size = len(gzip.compress(json.dumps(clean).encode()))
    assert size < 400_000, f"trace section compressed to {size:,} bytes"


# ------------------------------------------------------------- the step ref


def test_the_highlighted_step_must_be_in_the_bundle():
    """Otherwise the viewer opens a file whose highlighted step is not in
    it."""
    doc = _trace(n_steps=bundle.MAX_TRACE_STEPS + 5)
    with pytest.raises(bundle.BundleError, match="not among the steps"):
        bundle.prepare(doc, step_ref=f"s{bundle.MAX_TRACE_STEPS + 2}")


def test_a_valid_step_ref_is_carried():
    clean, _, _, _ = bundle.prepare(_trace(), step_ref="s1")
    assert clean["step_ref"] == "s1"


def test_no_step_ref_carries_no_key():
    clean, _, _, _ = bundle.prepare(_trace())
    assert "step_ref" not in clean


# ----------------------------------------------------------------- shapes


def test_a_trace_that_is_not_an_object_is_refused():
    with pytest.raises(bundle.BundleError, match="'steps' list"):
        bundle.prepare(["not", "a", "trace"])


def test_a_trace_with_no_steps_list_is_refused():
    with pytest.raises(bundle.BundleError, match="'steps' list"):
        bundle.prepare({"id": "x", "name": "y"})


def test_a_non_dict_step_is_skipped_rather_than_fatal():
    doc = _trace(n_steps=2)
    doc["steps"].insert(1, "nonsense")
    clean, _, _, preview = bundle.prepare(doc)
    assert len(clean["steps"]) == 2
    assert preview.n_steps == 2


def test_no_trace_at_all_is_allowed():
    clean, prompt, gen, preview = bundle.prepare(None, prompt="hi", generation="yo")
    assert clean is None
    assert preview.n_steps == 0


def test_the_preview_serialises_for_the_wire():
    doc = _trace(n_steps=1, payload=f"key {API_KEY}")
    out = bundle.preview(doc).to_dict()
    assert out["n_redactions"] == 1
    assert out["redactions"][0]["label"] == "api-key"
    assert isinstance(out["means"], str)
