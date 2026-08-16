"""Retrieved is not the same as used.

Every local-LLM app with RAG shows which chunks were retrieved. The question
this module exists for is the other one — did the answer depend on them — and
most of these tests are about the two measurements being kept apart, and about
what the module refuses to say when they disagree.
"""

from __future__ import annotations

import json
import os

import pytest

from modelmri import ground
from modelmri.errors import BadRequest

DOC = """The Antikythera mechanism was recovered from a shipwreck in 1901.

It is an ancient Greek geared device used to predict astronomical positions
and eclipses decades in advance.

The device was found off the coast of the island of Antikythera, between
Crete and the Peloponnese.

Unrelated paragraph about coffee. Beans grown at high altitude ripen more
slowly and are usually more acidic than beans grown lower down.

Another unrelated paragraph. The train from the coast leaves every twenty
minutes on weekdays and hourly at the weekend.
"""

QUESTION = "Question: In which year was the mechanism recovered?\nAnswer:"


# ----------------------------------------------------------------- chunking


def test_blank_lines_split_passages():
    chunks = ground.split(DOC)
    assert len(chunks) == 5
    assert chunks[0].startswith("The Antikythera mechanism")
    assert "coffee" in chunks[3]


def test_a_heading_starts_a_passage_without_a_blank_line():
    text = (
        "# Introduction\n"
        "This section is long enough to stand on its own as a passage here.\n"
        "## Method\n"
        "This one is also long enough to stand on its own as a passage here.\n"
    )
    chunks = ground.split(text)
    assert len(chunks) == 2
    assert chunks[0].startswith("# Introduction")
    assert chunks[1].startswith("## Method")


def test_a_bare_heading_merges_forward_rather_than_being_dropped():
    """Dropping it would put the section title outside every span, and a
    token outside every span cannot be masked — so the one line naming what
    the section is about would be the one line grounding never tests."""
    text = "# Eclipses\n\nThe device predicted eclipses decades in advance, which is the point.\n"
    chunks = ground.split(text)
    assert len(chunks) == 1
    assert chunks[0].startswith("# Eclipses")
    assert "eclipses decades" in chunks[0]


def test_a_trailing_scrap_joins_the_passage_before_it():
    text = "A passage long enough to count as one on its own, easily.\n\nshort\n"
    chunks = ground.split(text)
    assert len(chunks) == 1
    assert chunks[0].endswith("short")


def test_a_scrap_with_nothing_to_merge_into_still_appears():
    chunks = ground.split("short\n")
    assert chunks == ["short"]


def test_empty_text_is_refused_rather_than_scored_as_zero_passages():
    with pytest.raises(BadRequest, match="no text"):
        ground.split("   \n\n  ")


# ------------------------------------------------------- building the prompt


def test_every_passage_is_in_the_prompt_at_the_span_reported():
    chunks = ground.split(DOC)
    prompt, spans = ground.build(chunks, QUESTION)
    for chunk, (lo, hi) in zip(chunks, spans, strict=True):
        assert prompt[lo:hi] == chunk


def test_the_question_comes_last_so_the_answer_position_is_the_end():
    chunks = ground.split(DOC)
    prompt, _ = ground.build(chunks, QUESTION)
    assert prompt.rstrip().endswith("Answer:")


def test_a_missing_question_is_refused():
    with pytest.raises(BadRequest, match="ask a question"):
        ground.build(["some passage"], "   ")


# ------------------------------------------------------------- the refusals


def test_too_many_passages_is_refused_not_truncated():
    """Scoring an answer against the first twelve paragraphs of forty and
    calling the result grounding is worse than no answer: the passages it
    actually used might all be in the tail."""

    class _Tok:
        def __call__(self, *a, **k):
            raise AssertionError("must refuse before touching the tokenizer")

    with pytest.raises(BadRequest, match="cap is 3"):
        ground.measure(
            object(),
            _Tok(),
            ["a passage"] * 4,
            "Q?",
            device="cpu",
            max_chunks=3,
        )


def test_the_refusal_says_what_it_would_cost_rather_than_just_saying_no():
    with pytest.raises(BadRequest, match="one forward pass each"):
        ground.measure(object(), object(), ["a"] * 99, "Q?", device="cpu", max_chunks=2)


def test_no_passages_is_refused():
    with pytest.raises(BadRequest, match="no text"):
        ground.measure(object(), object(), [], "Q?", device="cpu")


def test_a_tokenizer_without_offsets_is_refused_rather_than_guessed_at():
    """The fallback everybody writes — re-tokenise the chunk and search for
    the id sequence — is wrong in a way that does not announce itself: the
    same words tokenise differently after a preceding space."""

    class _Slow:
        def __call__(self, *a, **k):
            return {"input_ids": [[1, 2, 3]]}

    with pytest.raises(BadRequest, match="character offsets"):
        ground.locate(_Slow(), "some prompt", [(0, 4)])


# ------------------------------------------------------- what it never says


def _grounding(**over) -> ground.Grounding:
    kw = dict(
        question="Q?",
        answer=" 1901",
        answer_p=0.62,
        position=88,
        chunks=[
            ground.Score(
                index=0,
                preview="The Antikythera mechanism…",
                n_tokens=20,
                dependence=0.9,
                attention=0.31,
                depended_on=True,
                looked_not_used=False,
            ),
            ground.Score(
                index=1,
                preview="Unrelated paragraph about coffee…",
                n_tokens=22,
                dependence=0.001,
                attention=0.28,
                depended_on=False,
                looked_not_used=True,
            ),
        ],
        n_chunks=2,
        n_prompt_tokens=89,
        noise_floor=0.01,
        joint=1.1,
        attention_share=0.59,
        passes=6,
        seconds=0.4,
    )
    kw.update(over)
    return ground.Grounding(**kw)


def test_nothing_is_ever_reported_as_a_percentage_share_of_the_answer():
    """Masking a whole chunk is a big intervention and the effects are not
    additive. A share implies they are."""
    means = _grounding().means()
    assert "nats" in means.lower()
    assert "do not add up" in means


def test_the_joint_mask_is_quoted_beside_the_parts_that_do_not_sum_to_it():
    means = _grounding().means()
    assert "1.1000" in means or "1.1" in means


def test_looked_at_but_not_depended_on_is_named_not_left_to_be_noticed():
    means = _grounding().means()
    assert "LOOKED AT, NOT DEPENDED ON" in means
    assert "#1" in means
    assert "not what it used" in means


def test_nothing_clearing_the_floor_is_a_finding_with_its_own_wording():
    flat = _grounding(
        ungrounded=True,
        chunks=[
            ground.Score(
                index=0,
                preview="p",
                n_tokens=4,
                dependence=0.002,
                attention=0.5,
                depended_on=False,
                looked_not_used=False,
            )
        ],
        n_chunks=1,
    )
    means = flat.means()
    assert "NO PASSAGE CLEARED THE NOISE FLOOR" in means
    assert "did not depend on the document" in means
    # And it must not become a claim about correctness.
    assert "not a verdict on whether it is correct" in means


def test_the_attention_coverage_is_stated_so_the_shares_are_not_read_as_all():
    means = _grounding().means()
    assert "59.0%" in means
    assert "the question, the template" in means


def test_the_report_survives_json():
    out = json.loads(json.dumps(_grounding().to_dict(), allow_nan=False))
    assert out["chunks"][0]["dependence"] == 0.9
    assert "means" in out


# --------------------------------------------------------- against gpt2


@pytest.fixture(scope="module")
def gpt2():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from modelmri import receipts as _receipts

    if _receipts.revision_of("gpt2")[0] is None and not os.environ.get(
        "MODELMRI_TEST_DOWNLOAD"
    ):
        pytest.skip("gpt2 is not in the local model cache")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except Exception as err:
        pytest.skip(f"gpt2 is not available here: {err}")
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="module")
def eager_gpt2(gpt2):
    """The same model with eager attention, which is what ModelRuntime loads.

    Fused kernels never build the score matrix, so the attention half of this
    feature is only measurable under eager -- and the product path already
    uses it. This fixture is the product's configuration, not a workaround.
    """
    from transformers import AutoModelForCausalLM

    _, tokenizer = gpt2
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2", attn_implementation="eager"
        )
    except Exception as err:
        pytest.skip(f"eager attention is not available here: {err}")
    model.eval()
    return model, tokenizer


def test_the_token_spans_cover_the_passage_and_nothing_else(gpt2):
    """A span off by one token silently masks a neighbour's word, and the
    score then belongs partly to a passage nobody asked about."""
    _, tokenizer = gpt2
    chunks = ground.split(DOC)
    prompt, char_spans = ground.build(chunks, QUESTION)
    spans = ground.locate(tokenizer, prompt, char_spans)
    ids = tokenizer(prompt)["input_ids"]
    for chunk, (start, end) in zip(chunks, spans, strict=True):
        covered = tokenizer.decode(ids[start:end])
        # The decode may pick up the leading space of the first token; the
        # test is that the passage's own words are all inside its span.
        assert chunk.split()[0] in covered
        assert chunk.split()[-1].rstrip(".") in covered


def test_every_pass_is_counted_and_the_count_is_the_portable_part(gpt2):
    model, tokenizer = gpt2
    chunks = ground.split(DOC)
    out = ground.measure(model, tokenizer, chunks, QUESTION, device="cpu")
    # one answer read, one repeat for the floor, one with attentions, one
    # joint, and one per chunk
    assert out.passes == len(chunks) + 4


def test_the_parts_do_not_sum_to_the_joint(gpt2):
    """The reason nothing here is a percentage. Measured, not asserted in a
    comment: if these ever did sum, the wording in `means` would be wrong."""
    model, tokenizer = gpt2
    chunks = ground.split(DOC)
    out = ground.measure(model, tokenizer, chunks, QUESTION, device="cpu")
    parts = sum(c.dependence for c in out.chunks)
    assert abs(parts - out.joint) > 1e-6


def test_the_floor_is_measured_on_this_model_not_carried_from_another(gpt2):
    model, tokenizer = gpt2
    out = ground.measure(model, tokenizer, ground.split(DOC), QUESTION, device="cpu")
    assert out.noise_floor >= 0.0
    for chunk in out.chunks:
        assert chunk.depended_on == (chunk.dependence > out.noise_floor)


def test_attention_and_dependence_are_reported_separately_for_every_passage(
    eager_gpt2,
):
    """The whole feature. A single combined score would hide exactly the
    disagreement this exists to surface."""
    model, tokenizer = eager_gpt2
    out = ground.measure(model, tokenizer, ground.split(DOC), QUESTION, device="cpu")
    assert out.attention_available
    for chunk in out.chunks:
        assert chunk.attention is not None and chunk.attention >= 0.0
        assert chunk.dependence >= 0.0
    assert out.attention_share and out.attention_share > 0.0


def test_a_fused_attention_model_reports_unknown_rather_than_zero(gpt2):
    """MEASURED: `from_pretrained("gpt2")` picks sdpa, and sdpa returns an
    EMPTY TUPLE for `output_attentions=True` rather than None. The obvious
    loop over it completes, sums nothing and reports 0.0 for every passage --
    a page claiming to measure attention, printing a measured-looking zero for
    a number that was never returned.

    ModelRuntime loads eager and never hits this. A caller holding its own
    model does, and 0.0 would be indistinguishable from "nothing looked here".
    """
    model, tokenizer = gpt2
    if getattr(model.config, "_attn_implementation", "eager") == "eager":
        pytest.skip("this transformers build defaults gpt2 to eager")
    out = ground.measure(model, tokenizer, ground.split(DOC), QUESTION, device="cpu")
    assert out.attention_available is False
    assert out.attention_share is None
    for chunk in out.chunks:
        assert chunk.attention is None, "unknown must not print as 0.0"
        # The dependence half is unaffected and still worth having.
        assert chunk.dependence >= 0.0
    assert "THE ATTENTION HALF DID NOT RUN" in out.means()
    assert 'attn_implementation="eager"' in out.means()


def test_looked_but_not_used_is_never_claimed_without_the_attention_half(gpt2):
    """0.0 >= 0.0/2 is True, so the naive rule would qualify EVERY passage as
    "looked at but not depended on" on a model that reported no attention at
    all -- the panel's headline finding, from nothing."""
    model, tokenizer = gpt2
    if getattr(model.config, "_attn_implementation", "eager") == "eager":
        pytest.skip("this transformers build defaults gpt2 to eager")
    out = ground.measure(model, tokenizer, ground.split(DOC), QUESTION, device="cpu")
    # None, not False. False reads as "this passage does not have that
    # problem", which is also a claim nothing measured.
    assert all(c.looked_not_used is None for c in out.chunks)
    assert "COULD NOT BE DECIDED HERE" in out.means()
    assert "absence of a test rather than a clean result" in out.means()


def test_a_zero_floor_also_makes_the_flag_undecidable_not_false(eager_gpt2):
    """The case that would otherwise kill the flagship reading silently.

    With a floor of exactly 0.0 every passage that moved the answer at all is
    `depended_on`, so `not depended_on` is never true and the flag can never
    fire. Reporting False for all of them is a clean bill of health from a
    test that never ran -- and gpt2 on cuda/bf16 reproduces its own answer bit
    for bit, so this is the ordinary path and not an edge case.
    """
    model, tokenizer = eager_gpt2
    out = ground.measure(model, tokenizer, ground.split(DOC), QUESTION, device="cpu")
    if out.noise_floor > 0.0:
        pytest.skip("this build's repeat pass does not reproduce exactly")
    assert out.attention_available, "the attention half is the OTHER cause"
    assert all(c.looked_not_used is None for c in out.chunks)
    means = out.means()
    assert "COULD NOT BE DECIDED HERE" in means
    assert "could never fire" in means


def test_the_passages_come_back_ordered_by_dependence_not_by_position(gpt2):
    model, tokenizer = gpt2
    out = ground.measure(model, tokenizer, ground.split(DOC), QUESTION, device="cpu")
    scores = [c.dependence for c in out.chunks]
    assert scores == sorted(scores, reverse=True)


def test_a_floor_of_exactly_zero_is_named_not_reported_as_a_clean_sweep(
    eager_gpt2,
):
    """MEASURED on gpt2 in float32 on CPU: the repeat pass reproduces the
    answer bit for bit, the floor is 0.0, and all five passages "clear" it --
    including one at 0.0107 nats against a top of 2.7627.

    "5 of 5 passages cleared the floor" is true and useless there. The
    degenerate case gets its own wording, the same way the probe names a
    saturated null, rather than an invented threshold that would make the
    verdict look earned.
    """
    model, tokenizer = eager_gpt2
    out = ground.measure(model, tokenizer, ground.split(DOC), QUESTION, device="cpu")
    if out.noise_floor > 0.0:
        pytest.skip("this build's repeat pass does not reproduce exactly")
    assert out.floor_degenerate
    means = out.means()
    assert "THE NOISE FLOOR IS EXACTLY ZERO" in means
    assert "no significance test on this run" in means


def test_the_passage_carrying_the_answer_outscores_the_unrelated_ones(
    eager_gpt2,
):
    """The feature working, on text where the right answer is known: the
    paragraph containing 1901 must move the answer further than the one about
    coffee. Not a tautology -- an implementation that masked the wrong spans
    would pass every structural test above and fail this."""
    model, tokenizer = eager_gpt2
    out = ground.measure(model, tokenizer, ground.split(DOC), QUESTION, device="cpu")
    by_index = {c.index: c for c in out.chunks}
    assert by_index[0].dependence > by_index[3].dependence * 4, (
        "the passage with the date must beat the one about coffee by a margin "
        "no reordering could produce by chance"
    )
    assert out.chunks[0].index == 0


def test_the_degenerate_floor_flag_is_not_set_when_the_floor_is_real():
    assert not _grounding(noise_floor=0.01).floor_degenerate


# ---------------------------------------------------------- through the API


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from modelmri.server import create_app

    # An explicit loopback peer, because one of these tests names a FILE and
    # the server only reads a path for a request it can tell came from this
    # machine. TestClient's default peer is the literal string "testclient",
    # which is not an address at all — so without this the suite would be
    # exercising a path no real local client takes.
    return TestClient(create_app(), client=("127.0.0.1", 51111))


def test_a_request_without_a_document_says_nothing_is_downloaded(client):
    """The sentence matters as much as the refusal. Every tool in this
    category downloads an embedding model on first use, and a user reading a
    422 here should learn that this one does not."""
    r = client.post("/api/ground", json={"question": "Q?"})
    assert r.status_code == 422
    assert "Nothing is downloaded" in r.json()["error"]
    assert "nothing is indexed" in r.json()["error"]


def test_a_body_that_is_not_json_is_named_as_such(client):
    r = client.post(
        "/api/ground",
        content=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422
    assert "not JSON" in r.json()["error"]


def test_grounding_without_a_model_is_a_refusal_not_an_empty_report(client):
    r = client.post("/api/ground", json={"document": DOC, "question": QUESTION})
    # 409 with no model loaded; anything else means it tried to measure.
    assert r.status_code in (409, 200)
    if r.status_code == 409:
        assert "No model loaded" in r.json()["error"]


def test_a_corpus_file_is_read_with_passage_boundaries_between_its_lines(
    client, tmp_path
):
    """A .txt is one sequence per line, and joining them with a single newline
    would arrive at `split` as ONE passage — the whole document scored as a
    single chunk, which is a grounding report with nothing in it."""
    path = tmp_path / "notes.txt"
    path.write_text(
        "The mechanism was recovered from a shipwreck in the year 1901.\n"
        "Beans grown at high altitude ripen more slowly than beans lower down.\n",
        encoding="utf-8",
    )
    r = client.post("/api/ground", json={"file": str(path), "question": "Q?"})
    # Either it measured (two passages) or it refused for want of a model.
    if r.status_code == 200:
        assert r.json()["n_chunks"] == 2
    else:
        assert r.status_code == 409
