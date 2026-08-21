"""A finetune comparison inside a `.mri`, and what it will not let through.

This section is the odd one out in the format: every other one describes the
model the file is about, and this one names TWO OTHER MODELS. A comparison of
two checkpoints can legitimately ride in a `.mri` taken on a third, so the
section is required to say what it compared — otherwise a reader takes it as
being about the file's own model, which is the single confusion it can cause.
"""

from __future__ import annotations

import gzip
import json

import pytest

torch = pytest.importorskip("torch")

from modelmri import session, verify  # noqa: E402
from modelmri.errors import BadRequest  # noqa: E402


def _prompt_row(**over) -> dict:
    row = {
        "prompt": "The capital of France is",
        "n_tokens": 6,
        "mean_kl": 0.22,
        "max_kl": 0.9,
        "flips": 3,
        "first_divergent_layer": 4,
        "drop": 0.000525,
    }
    row.update(over)
    return row


def _diff(**over) -> dict:
    out = {
        "model_a": "base",
        "model_b": "tuned",
        "n_prompts": 4,
        "n_layers": 7,
        "prompts": [_prompt_row() for _ in range(4)],
        "layers": [
            {"layer": i, "median": 1.0, "low": 1.0, "high": 1.0, "n": 4, "n_first": 0}
            for i in range(7)
        ],
        "kl": {"name": "KL", "median": 0.22, "low": 0.21, "high": 0.23, "n": 4},
        "flips": {"name": "flips", "median": 3.0, "low": 3.0, "high": 3.0, "n": 4},
        "consensus_layer": 4,
        "consensus_share": 1.0,
        "seconds": 1.2,
    }
    out.update(over)
    return out


def _build(**over) -> bytes:
    args = dict(
        model_id="Qwen/Qwen3-1.7B",
        device="cpu",
        dtype="float32",
        n_params=1,
        tokens=["a"],
        prompt="a",
        generation="",
        attention={},
        n_layers=1,
        n_heads=1,
    )
    args.update(over)
    return session.build(**args)


# ------------------------------------------------------------- round trip


def test_a_comparison_survives_the_round_trip():
    parsed = session.parse(_build(model_diff=_diff()))
    assert parsed.has_model_diff()
    assert parsed.model_diff["model_a"] == "base"
    assert parsed.model_diff["consensus_layer"] == 4
    assert len(parsed.model_diff["prompts"]) == 4


def test_a_session_without_one_carries_no_empty_section():
    raw = json.loads(gzip.decompress(_build()).decode("utf-8"))
    assert "model_diff" not in raw
    assert session.parse(_build()).has_model_diff() is False


def test_a_null_consensus_layer_survives_as_null():
    """None means the cosine never fell, which is a result and not a gap."""
    parsed = session.parse(
        _build(
            model_diff=_diff(
                consensus_layer=None,
                consensus_share=0.0,
                prompts=[_prompt_row(first_divergent_layer=None) for _ in range(4)],
            )
        )
    )
    assert parsed.model_diff["consensus_layer"] is None
    assert parsed.model_diff["prompts"][0]["first_divergent_layer"] is None


# ------------------------------------------------------- it names its sides


def test_a_comparison_that_does_not_name_its_models_is_refused():
    """A `.mri` is one analysis of one model; a diff is a comparison of two
    others, and it can ride in a file about a third. A section that does not
    say what it compared would be read as being about this file's model."""
    with pytest.raises(BadRequest, match="which model it compared FROM"):
        session._model_diff({"model_diff": _diff(model_a="")})
    with pytest.raises(BadRequest, match="which model it compared TO"):
        session._model_diff({"model_diff": _diff(model_b=None)})


def test_the_writer_refuses_the_same_shape_the_reader_refuses():
    with pytest.raises(BadRequest, match="which model it compared FROM"):
        _build(model_diff=_diff(model_a=""))


# ------------------------------------------- a spread needs its prompt count


def test_a_median_without_its_n_is_refused():
    """The entire content of this section is that its numbers are
    distributions over a prompt set rather than single measurements, and a
    median arriving without the count is exactly the single number the module
    exists to avoid printing."""
    with pytest.raises(BadRequest, match="carries no prompt count"):
        session._model_diff(
            {"model_diff": _diff(kl={"median": 0.22, "low": 0.2, "high": 0.3})}
        )


def test_a_spread_missing_a_quartile_is_refused():
    with pytest.raises(BadRequest, match="has no high"):
        session._model_diff(
            {"model_diff": _diff(kl={"median": 0.22, "low": 0.2, "n": 4})}
        )


def test_a_non_finite_number_in_a_spread_is_refused():
    with pytest.raises(BadRequest, match="has no median"):
        session._model_diff(
            {
                "model_diff": _diff(
                    kl={"median": float("nan"), "low": 0.2, "high": 0.3, "n": 4}
                )
            }
        )


# --------------------------------------------------------------- the bounds


def test_an_absurd_row_count_is_refused_before_a_browser_sees_it():
    with pytest.raises(BadRequest, match="above the"):
        session._model_diff(
            {"model_diff": _diff(heads=[{"layer": 0}] * (session.MAX_DIFF_ROWS + 1))}
        )


def test_a_prompt_is_bounded_but_does_travel():
    """Unlike a grounding document. The two are not the same kind of text: a
    grounded document is source material somebody attached, and a prompt set
    is what they ASKED — the same thing this format already carries whole."""
    out = session._model_diff(
        {"model_diff": _diff(prompts=[_prompt_row(prompt="x" * 99_999)])}
    )
    assert len(out["prompts"][0]["prompt"]) == session.MAX_DIFF_TEXT
    assert out["prompts"][0]["prompt"].startswith("xxx")


def test_the_section_is_not_a_set_of_fields_is_refused():
    with pytest.raises(BadRequest, match="not a set of fields"):
        session._model_diff({"model_diff": [1, 2, 3]})


# ------------------------------------------------------------------ verify


def _check(parsed):
    return verify._check_model_diff(parsed, runtime=None, blocked="")


def test_a_comparison_is_never_reported_as_reproduced():
    """It compares two models, neither of which is the one this file describes
    and neither of which need be on this machine."""
    check = _check(session.parse(_build(model_diff=_diff())))
    assert check.verdict == verify.NOT_VERIFIABLE
    assert "NOT RE-MEASURED" in check.detail
    assert "base against tuned" in check.detail


def test_a_headline_edited_away_from_its_own_rows_is_caught():
    """The one thing this CAN check without the models: a file whose summary
    no longer follows from the per-prompt rows it claims to summarise."""
    parsed = session.parse(_build(model_diff=_diff()))
    parsed.model_diff["consensus_layer"] = 2
    check = _check(parsed)
    assert check.verdict == verify.DIFFERS
    assert "does not agree with itself" in check.detail
    assert "names layer 2" in check.detail and "layer 4 most often" in check.detail


def test_a_share_that_does_not_follow_from_the_rows_is_caught():
    parsed = session.parse(_build(model_diff=_diff()))
    parsed.model_diff["consensus_share"] = 0.25
    check = _check(parsed)
    assert check.verdict == verify.DIFFERS
    assert "25% of prompts and its rows say 100%" in check.detail


def test_a_spread_count_that_disagrees_with_the_prompt_count_is_caught():
    parsed = session.parse(_build(model_diff=_diff()))
    parsed.model_diff["kl"]["n"] = 99
    check = _check(parsed)
    assert check.verdict == verify.DIFFERS
    assert "over 99 prompts and the file carries 4" in check.detail


def test_naming_a_layer_when_nothing_diverged_is_caught():
    parsed = session.parse(
        _build(
            model_diff=_diff(
                prompts=[_prompt_row(first_divergent_layer=None) for _ in range(4)]
            )
        )
    )
    check = _check(parsed)
    assert check.verdict == verify.DIFFERS
    assert "not one of its prompt rows reports a divergence" in check.detail


def test_a_file_with_no_comparison_and_no_receipt_produces_no_check():
    assert _check(session.parse(_build())) is None


# ------------------------------------------------------- the export helper


def test_the_export_survives_a_model_change():
    """NO EPOCH CHECK, alone among the export helpers. Unloading the model does
    not make "these two checkpoints differ at layer 4" untrue."""
    from modelmri.runtime import ModelRuntime

    rt = ModelRuntime()
    rt._last_model_diff = {**_diff(), "means": "a sentence", "receipt": {"op": "x"}}
    rt.epoch = 1
    first = rt._model_diff_for_export()
    rt.epoch = 99
    assert rt._model_diff_for_export() == first
    assert first["model_a"] == "base"


def test_the_export_drops_the_summary_and_the_receipt():
    """Both are carried once elsewhere — the receipts list holds the receipt,
    and the sentence is regenerated from the numbers."""
    from modelmri.runtime import ModelRuntime

    rt = ModelRuntime()
    rt._last_model_diff = {**_diff(), "means": "a sentence", "receipt": {"op": "x"}}
    out = rt._model_diff_for_export()
    assert "means" not in out and "receipt" not in out


def test_nothing_measured_exports_nothing():
    from modelmri.runtime import ModelRuntime

    assert ModelRuntime()._model_diff_for_export() == {}
