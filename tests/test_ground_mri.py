"""What `verify` and `diff` can say about a grounding, and what they cannot.

Grounding is the odd one out in both tools. Every other section is re-taken by
running the model again; this one cannot be, because the thing that would have
to be re-run is the DOCUMENT, and a `.mri` deliberately does not carry it — it
carries ~120-character previews, because the text somebody grounds an answer
in is usually the half they do not want forwarded.

So the tests here are mostly about the two tools being honest about that:
`verify` never claims to have reproduced it, and `diff` refuses when the
indices no longer name the same passages.
"""

from __future__ import annotations

import pytest

from modelmri import mri_diff, session, verify

QUESTION = "Question: In which year was it recovered?\nAnswer:"
PREVIEW_A = "The Antikythera mechanism was recovered from a shipwreck in 1901."
PREVIEW_B = "Unrelated paragraph about coffee."


def _chunk(index=0, dependence=3.25, attention=0.377, preview=PREVIEW_A, **over):
    row = {
        "index": index,
        "preview": preview,
        "n_tokens": 16,
        "dependence": dependence,
        "attention": attention,
        "depended_on": dependence > 0.01,
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
        "chunks": [
            _chunk(),
            _chunk(index=1, dependence=0.0717, attention=0.029, preview=PREVIEW_B),
        ],
        "n_chunks": 2,
        "n_prompt_tokens": 132,
        "noise_floor": 0.01,
        "joint": 10.27,
        "attention_share": 0.527,
        "attention_available": True,
        "floor_degenerate": False,
        "ungrounded": False,
        "passes": 6,
        "seconds": 1.29,
    }
    out.update(over)
    return out


def _parsed(**over):
    return session.parse(
        session.build(
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
            ground=_ground(**over),
        )
    )


# ------------------------------------------------------------------ verify


def _check(parsed):
    return verify._check_ground(parsed, runtime=None, blocked="")


def test_a_grounding_is_never_reported_as_reproduced():
    """It cannot be. The document is not in the file, so nothing here can be
    masked out and run again — and a verify report that said REPRODUCED would
    be claiming a measurement nobody took."""
    check = _check(_parsed())
    assert check.verdict == verify.NOT_VERIFIABLE
    assert "NOT RE-MEASURED" in check.detail
    assert "private half of the pair" in check.detail


def test_the_check_says_which_passage_carries_the_answer():
    check = _check(_parsed())
    assert "#0" in check.detail
    assert "3.2500" in check.detail


def test_a_verdict_its_own_numbers_contradict_is_caught():
    """The one thing this CAN check without the document: a file edited to
    move the verdict without moving the number it was derived from."""
    doctored = _parsed()
    doctored.ground["chunks"][1]["depended_on"] = False  # 0.0717 > 0.01
    check = _check(doctored)
    assert check.verdict == verify.DIFFERS
    assert "their own numbers do not support" in check.detail
    assert check.measured["inconsistent_passages"] == [1]


def test_a_zero_floor_is_named_in_the_check_too():
    check = _check(_parsed(noise_floor=0.0, floor_degenerate=True))
    assert "exactly zero" in check.detail
    assert "no significance test" in check.detail


def test_a_file_with_no_grounding_and_no_receipt_produces_no_check():
    """Silence is right when nothing claims a grounding ran. A check saying
    "not verifiable" on every file that never did one is noise."""
    plain = session.parse(
        session.build(
            model_id="gpt2",
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
    )
    assert _check(plain) is None


# -------------------------------------------------------------------- diff


def test_two_identical_groundings_are_the_same():
    delta = mri_diff._diff_ground(_parsed(), _parsed())
    assert delta.status == mri_diff.SAME
    assert "#0 carries the answer in both" in delta.detail


def test_the_regression_this_exists_for_is_named_when_it_happens():
    """A finetune that stops reading the document and starts answering from
    its weights. Invisible in every other section — the generation can be word
    for word identical while the thing producing it has moved."""
    before = _parsed()
    after = _parsed(
        ungrounded=True,
        chunks=[
            _chunk(dependence=0.001, depended_on=False),
            _chunk(index=1, dependence=0.0005, preview=PREVIEW_B, depended_on=False),
        ],
    )
    delta = mri_diff._diff_ground(before, after)
    assert delta.status == mri_diff.CHANGED
    assert "stopped depending on the document" in delta.detail
    assert delta.measured["ungrounded_b"] is True


def test_a_different_passage_carrying_the_answer_is_its_own_finding():
    """Which passage the answer rests on and how much it rests on it are
    different claims, and a finetune can hold one while breaking the other."""
    after = _parsed(
        chunks=[
            _chunk(dependence=0.05),
            _chunk(index=1, dependence=3.30, preview=PREVIEW_B),
        ]
    )
    delta = mri_diff._diff_ground(_parsed(), after)
    assert delta.status == mri_diff.CHANGED
    assert "#0 became #1" in delta.detail
    assert delta.measured["top_passage_b"] == 1


def test_a_score_moving_past_the_floor_is_reported_with_the_floor():
    after = _parsed(
        chunks=[
            _chunk(dependence=1.10),
            _chunk(index=1, dependence=0.0717, preview=PREVIEW_B),
        ]
    )
    delta = mri_diff._diff_ground(_parsed(), after)
    assert delta.status == mri_diff.CHANGED
    assert "against a floor of" in delta.detail
    assert "still carries the answer" in delta.detail


def test_a_score_moving_under_the_floor_is_not_a_change():
    after = _parsed(
        chunks=[
            _chunk(dependence=3.255),
            _chunk(index=1, dependence=0.0717, preview=PREVIEW_B),
        ]
    )
    delta = mri_diff._diff_ground(_parsed(), after)
    assert delta.status == mri_diff.SAME


def test_two_different_questions_are_not_comparable():
    """A passage that mattered for one question need not have been asked about
    in the other, so comparing them would measure the question."""
    delta = mri_diff._diff_ground(_parsed(), _parsed(question="Something else?"))
    assert delta.status == mri_diff.NOT_COMPARABLE
    assert "measure the question" in delta.detail


def test_a_reordered_document_is_refused_rather_than_read_as_a_model_change():
    """Indices only mean something if they name the same passages. Without
    this, swapping two paragraphs in the source file reports as the answer
    moving to a different passage — a change in the document, presented as a
    change in the model."""
    swapped = _parsed(
        chunks=[
            _chunk(preview=PREVIEW_B),
            _chunk(index=1, dependence=0.0717, preview=PREVIEW_A),
        ]
    )
    delta = mri_diff._diff_ground(_parsed(), swapped)
    assert delta.status == mri_diff.NOT_COMPARABLE
    assert "document changed between these runs" in delta.detail


def test_a_missing_section_is_unavailable_rather_than_clean():
    plain = session.parse(
        session.build(
            model_id="gpt2",
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
    )
    delta = mri_diff._diff_ground(_parsed(), plain)
    assert delta.status == mri_diff.NOT_COMPARABLE
    assert "A missing section is not a zero" in delta.detail


def test_a_zero_floor_on_both_sides_says_bit_for_bit_not_within_tolerance():
    """`worst_gap <= floor` with floor 0.0 only holds when the numbers are
    identical. Reporting that as "within tolerance" would imply a tolerance
    was applied when none exists."""
    flat = dict(noise_floor=0.0, floor_degenerate=True)
    delta = mri_diff._diff_ground(_parsed(**flat), _parsed(**flat))
    assert delta.status == mri_diff.SAME
    assert "bit-for-bit equality" in delta.detail


# ----------------------------------------------------------------- replay


def test_a_recording_serves_its_grounding_rather_than_refusing():
    """The point of carrying it. "The answer came from the weights, not from
    the document I gave it" is a finding somebody wants to show a colleague,
    and a viewer that could only apologise would make the section pointless."""
    from modelmri.runtime import ModelRuntime

    rt = ModelRuntime()
    rt.replay = _parsed()
    out = rt.ground_answer("ignored", QUESTION)
    assert out["recorded"] is True
    assert out["n_chunks"] == 2
    assert out["chunks"][0]["dependence"] == pytest.approx(3.25)


def test_the_recorded_summary_is_rebuilt_from_the_numbers_not_stored():
    """`means` is not written into the file. Rebuilding it means the prose and
    the fields can never disagree — a file hand-edited to flip `ungrounded`
    gets a summary that says so, instead of a stored sentence describing the
    numbers it used to have."""
    from modelmri.runtime import ModelRuntime

    rt = ModelRuntime()
    rt.replay = _parsed(ungrounded=True)
    out = rt.ground_answer("ignored", QUESTION)
    assert "NO PASSAGE CLEARED THE NOISE FLOOR" in out["means"]


def test_the_summary_is_not_carried_in_the_file():
    raw = _parsed()
    assert "means" not in raw.ground


def test_a_recording_without_a_grounding_refuses_and_says_why():
    from modelmri.errors import Refusal
    from modelmri.runtime import ModelRuntime

    plain = session.parse(
        session.build(
            model_id="gpt2",
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
    )
    rt = ModelRuntime()
    rt.replay = plain
    with pytest.raises(Refusal, match="does not carry a grounding"):
        rt.ground_answer("doc", "Q?")


def test_the_session_info_tells_the_panel_whether_to_offer_anything():
    """Without it the grounding panel on a recording is a form whose only
    button refuses."""
    from modelmri.runtime import ModelRuntime

    rt = ModelRuntime()
    rt.replay = _parsed()
    info = rt.session_info()
    assert info["ground"]["available"] is True
    assert info["ground"]["question"] == QUESTION


def test_an_export_drops_a_grounding_measured_under_a_different_model():
    """The epoch moves on load and unload. A grounding measured on one model,
    written beside another model's attention, would be a claim about weights
    the file does not describe."""
    from modelmri.runtime import ModelRuntime

    rt = ModelRuntime()
    rt._last_ground = {**_ground(), "epoch": 7}
    rt.epoch = 7
    assert rt._ground_for_export()["n_chunks"] == 2
    rt.epoch = 8
    assert rt._ground_for_export() == {}


def test_the_export_drops_the_summary_and_the_receipt():
    """Both are carried once elsewhere — the receipts list holds the receipt,
    and the sentence is regenerated — so a second copy is a second thing to
    keep in step."""
    from modelmri.runtime import ModelRuntime

    rt = ModelRuntime()
    rt.epoch = 3
    rt._last_ground = {
        **_ground(),
        "epoch": 3,
        "means": "a sentence",
        "receipt": {"op": "ground"},
    }
    out = rt._ground_for_export()
    assert "means" not in out and "receipt" not in out and "epoch" not in out


# ------------------------------------------------------- the whole way round


@pytest.fixture(scope="module")
def gpt2_runtime():
    import os

    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from modelmri import receipts as _receipts

    if _receipts.revision_of("gpt2")[0] is None and not os.environ.get(
        "MODELMRI_TEST_DOWNLOAD"
    ):
        pytest.skip("gpt2 is not in the local model cache")

    from modelmri.runtime import ModelRuntime

    rt = ModelRuntime()
    try:
        rt.load("gpt2")
    except Exception as err:
        pytest.skip(f"gpt2 is not available here: {err}")
    yield rt
    rt.unload()


def test_a_grounding_measured_here_survives_export_and_reopening(gpt2_runtime):
    """Measured, exported, reopened, checked and compared — on a real model,
    because every step above this is a unit test against a dict somebody
    typed. MEASURED on gpt2: 3 passages, top 5.96 nats, 9090 bytes."""
    doc = "\n\n".join(
        [
            "The Antikythera mechanism was recovered from a shipwreck in 1901.",
            "It is an ancient Greek geared device for astronomical positions.",
            "Unrelated paragraph about coffee. Beans at altitude ripen slowly.",
        ]
    )
    question = "Question: In which year was the mechanism recovered?\nAnswer:"

    list(gpt2_runtime.generate_stream("The capital of France is", 4, temperature=0.0))
    live = gpt2_runtime.ground_answer(doc, question)
    assert live["n_chunks"] == 3

    parsed = session.parse(gpt2_runtime.export_session())
    assert parsed.has_ground()
    assert len(parsed.ground["chunks"]) == 3
    assert parsed.ground["question"] == question

    # Every dependence survives to the precision the file stores.
    for before, after in zip(live["chunks"], parsed.ground["chunks"], strict=True):
        assert after["dependence"] == pytest.approx(before["dependence"], abs=1e-9)
        assert after["depended_on"] == before["depended_on"]

    check = verify._check_ground(parsed, runtime=None, blocked="")
    assert check.verdict == verify.NOT_VERIFIABLE

    delta = mri_diff._diff_ground(parsed, parsed)
    assert delta.status == mri_diff.SAME
    # The field that was silently a dict until the Delta arguments were
    # keyworded, and which `render --fail-over` compares against a float.
    assert isinstance(delta.magnitude, float)


def test_the_document_itself_never_reaches_the_file(gpt2_runtime):
    """A grounding document is usually the private half of the pair. The file
    carries ~120-character previews so a finding can be forwarded without the
    source material going with it, and this is the test that keeps it true."""
    secret = "SHIBBOLETH-" + "x" * 400
    doc = "\n\n".join(
        [
            f"The recovery happened in 1901. {secret}",
            "Unrelated paragraph about coffee. Beans at altitude ripen slowly.",
        ]
    )
    gpt2_runtime.ground_answer(doc, "Question: which year?\nAnswer:")
    blob = gpt2_runtime.export_session()

    import gzip

    body = gzip.decompress(blob).decode("utf-8")
    assert "SHIBBOLETH" in body, (
        "the preview is meant to be recognisable, so the START of a passage "
        "does travel — this asserts the boundary is where it is claimed"
    )
    assert body.count("x" * 200) == 0, (
        "a whole passage reached the file. The preview bound is what stops a "
        "shared finding from carrying somebody's source document with it."
    )
