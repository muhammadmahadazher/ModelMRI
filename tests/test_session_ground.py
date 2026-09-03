# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Grounding inside a `.mri`: what travels, what does not, and what is refused.

"The model answered from its weights, not from the document I gave it" is a
finding somebody wants to SHOW a colleague, and until this section existed it
was one of the few results in the tool that could not leave the machine that
took it.

It is also the section with the most to get wrong, because two of its fields
are three-valued and the third is a verdict that means nothing without the
number it cleared. Most of these tests are about those three things surviving
a round trip intact, and about the section refusing rather than flattening.
"""

from __future__ import annotations

import gzip
import json

import pytest

from modelmri import session
from modelmri.errors import BadRequest

QUESTION = "Question: In which year was it recovered?\nAnswer:"


def _chunk(**over) -> dict:
    row = {
        "index": 0,
        "preview": "The Antikythera mechanism was recovered from a shipwreck in 1901.",
        "n_tokens": 16,
        "dependence": 3.2526,
        "attention": 0.3775,
        "depended_on": True,
        "looked_not_used": False,
    }
    row.update(over)
    return row


def _ground(**over) -> dict:
    out = {
        "question": QUESTION,
        "answer": " 1901",
        "answer_p": 0.62,
        "position": 131,
        "chunks": [_chunk(), _chunk(index=1, dependence=0.0717, attention=0.0293)],
        "n_chunks": 2,
        "n_prompt_tokens": 132,
        "noise_floor": 0.01,
        "joint": 10.2723,
        "attention_share": 0.5275,
        "attention_available": True,
        "attention_note": "",
        "floor_degenerate": False,
        "ungrounded": False,
        "passes": 6,
        "seconds": 1.29,
    }
    out.update(over)
    return out


def _build(**over) -> bytes:
    args = dict(
        model_id="gpt2",
        device="cuda:0",
        dtype="bfloat16",
        n_params=124_000_000,
        tokens=["The", " cat"],
        prompt="The",
        generation=" cat",
        attention={},
        n_layers=1,
        n_heads=1,
    )
    args.update(over)
    return session.build(**args)


# ------------------------------------------------------------- round trip


def test_a_grounding_survives_the_round_trip():
    parsed = session.parse(_build(ground=_ground()))
    assert parsed.has_ground()
    assert parsed.ground["question"] == QUESTION
    assert parsed.ground["chunks"][0]["dependence"] == pytest.approx(3.2526)
    assert parsed.ground["noise_floor"] == pytest.approx(0.01)


def test_a_session_without_grounding_carries_no_empty_section():
    """An empty key would make every file claim a grounding section and every
    reader render an empty one."""
    raw = json.loads(gzip.decompress(_build()).decode("utf-8"))
    assert "ground" not in raw
    assert session.parse(_build()).has_ground() is False


def test_the_flags_survive_as_flags():
    parsed = session.parse(
        _build(ground=_ground(ungrounded=True, floor_degenerate=True))
    )
    assert parsed.ground["ungrounded"] is True
    assert parsed.ground["floor_degenerate"] is True


# ------------------------------------------- the two three-valued fields


def test_an_unmeasured_attention_share_survives_as_none_not_as_zero():
    """A model whose attention implementation never built the score matrix and
    a passage nothing looked at are different facts. Flattened to 0.0 inside a
    file, the difference is unrecoverable — and the file is the one place it
    matters most, because it has travelled away from the machine that could
    tell them apart."""
    parsed = session.parse(
        _build(
            ground=_ground(
                chunks=[_chunk(attention=None, looked_not_used=None)],
                attention_share=None,
                attention_available=False,
                attention_note="this model runs 'sdpa'.",
            )
        )
    )
    row = parsed.ground["chunks"][0]
    assert row["attention"] is None
    assert parsed.ground["attention_available"] is False
    assert "sdpa" in parsed.ground["attention_note"]


def test_an_undecidable_flag_survives_as_none_not_as_false():
    """False reads as "this passage does not have that problem". On a run
    where the flag could not be decided, nothing measured that."""
    parsed = session.parse(
        _build(ground=_ground(chunks=[_chunk(looked_not_used=None)]))
    )
    assert parsed.ground["chunks"][0]["looked_not_used"] is None


def test_a_decided_flag_survives_as_itself():
    for value in (True, False):
        parsed = session.parse(
            _build(
                ground=_ground(
                    chunks=[_chunk(depended_on=False, looked_not_used=value)]
                )
            )
        )
        assert parsed.ground["chunks"][0]["looked_not_used"] is value


def test_an_old_file_without_the_attention_flag_does_not_claim_it_was_measured():
    """The safe reading of an absent flag is the one that claims least: a
    blank attention column presented as measured would be a claim about a
    field the writer never had."""
    doc = json.loads(gzip.decompress(_build(ground=_ground())).decode("utf-8"))
    del doc["ground"]["attention_available"]
    raw = gzip.compress(json.dumps(doc).encode("utf-8"), 6)
    assert session.parse(raw).ground["attention_available"] is False


# --------------------------------------------------------- what it refuses


def test_a_verdict_without_the_floor_it_cleared_is_refused():
    """The whole content of "this passage mattered" is that removing it moved
    the answer further than a pass that changed nothing. A row carrying the
    verdict and not the reference is the bare claim this section exists to
    replace — the same rule `_head_types` applies to a label with no margin."""
    with pytest.raises(BadRequest, match="does not say what it cleared"):
        session._ground(
            {"ground": {"chunks": [_chunk(depended_on=True)]}}  # no noise_floor
        )


def test_a_verdict_of_false_needs_no_floor():
    """ "This passage did not matter" is not a claim that needs a reference to
    be readable; refusing it would make an honest negative unwriteable."""
    out = session._ground(
        {"ground": {"chunks": [_chunk(depended_on=False, looked_not_used=None)]}}
    )
    assert out["chunks"][0]["depended_on"] is False


def test_a_row_with_no_dependence_score_is_refused():
    with pytest.raises(BadRequest, match="no dependence score"):
        session._ground({"ground": {"chunks": [{"index": 0}]}})


def test_a_row_that_does_not_name_a_passage_is_refused():
    with pytest.raises(BadRequest, match="does not name a passage"):
        session._ground({"ground": {"chunks": [{"dependence": 1.0}]}})


def test_a_non_finite_dependence_is_refused_rather_than_rendered():
    with pytest.raises(BadRequest, match="no dependence score"):
        session._ground(
            {"ground": {"chunks": [{"index": 0, "dependence": float("nan")}]}}
        )


def test_a_grounding_that_is_not_fields_is_refused():
    with pytest.raises(BadRequest, match="not a set of fields"):
        session._ground({"ground": [1, 2, 3]})


def test_a_grounding_with_no_passages_is_refused():
    with pytest.raises(BadRequest, match="carries no passages"):
        session._ground({"ground": {"question": "Q?"}})


def test_an_absurd_passage_count_is_refused_before_a_browser_sees_it():
    rows = [_chunk(index=i, depended_on=False) for i in range(3)]
    doc = {"ground": {"chunks": rows}}
    session._ground(doc)  # three is fine
    with pytest.raises(BadRequest, match="above the"):
        session._ground(
            {
                "ground": {
                    "chunks": [{"index": 0, "dependence": 0.0}]
                    * (session.MAX_GROUND_CHUNKS + 1)
                }
            }
        )


def test_a_preview_is_bounded_before_it_reaches_a_browser():
    out = session._ground(
        {"ground": {"chunks": [_chunk(depended_on=False, preview="x" * 99_999)]}}
    )
    assert len(out["chunks"][0]["preview"]) == session.MAX_GROUND_TEXT


def test_a_preview_that_is_not_text_becomes_empty_rather_than_rendering():
    out = session._ground(
        {"ground": {"chunks": [_chunk(depended_on=False, preview={"x": 1})]}}
    )
    assert out["chunks"][0]["preview"] == ""


# ------------------------------------------- the writer is not laxer than the reader


def test_the_writer_refuses_the_same_shape_the_reader_refuses():
    """A writer laxer than the reader is how you build files nobody can open.
    Every additive section in this format goes out through the READER's
    validator for that reason, and this one is no exception."""
    with pytest.raises(BadRequest, match="does not say what it cleared"):
        _build(
            ground={
                "chunks": [_chunk(depended_on=True)],
                "question": QUESTION,
            }
        )


def test_a_non_finite_number_anywhere_in_the_section_is_refused_at_write_time():
    """Python writes a bare `NaN` token that every browser's JSON.parse
    rejects, so a file carrying one opens here and nowhere else."""
    with pytest.raises(BadRequest):
        _build(ground=_ground(chunks=[_chunk(dependence=float("inf"))]))
