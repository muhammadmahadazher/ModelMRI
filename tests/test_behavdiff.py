"""Comparing two models' behaviour, and refusing the comparisons that mean
nothing.

The failure this guards against is not a crash. It is a report that looks
right: a KL between distributions over different vocabularies, a per-position
series where position 3 holds a different token on each side, or a flip count
that treats a broken 0.003-margin tie the same as a confident answer changing.
Each of those produces numbers, and none of them produces a fact.

Nothing here loads a model. Captures are synthesised, because what a unit test
can own is the arithmetic and the refusals; the load-capture-release path is
exercised by `scripts/measure_docs.py --gguf` and by the route tests.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from modelmri import behavdiff  # noqa: E402
from modelmri.errors import BadRequest, Refusal  # noqa: E402


def _capture(label, ids, rows, tokens=None, attention=None):
    """A synthetic capture. `rows` are unnormalised weights per position."""
    probs = torch.tensor(rows, dtype=torch.float32)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1)
    top_ids = [int(i) for i in top2.indices[:, 0]]
    return behavdiff.Capture(
        label=label,
        ids=ids,
        tokens=tokens or [f"t{i}" for i in ids],
        probs=probs,
        attention=attention,
        vocab_size=probs.shape[-1],
        top_ids=top_ids,
        top_texts=[f"v{i}" for i in top_ids],
        margins=[float(r[0] - r[1]) if r.numel() > 1 else 1.0 for r in top2.values],
    )


# ------------------------------------------------------------- refusals


def test_different_tokenisations_are_refused_not_aligned():
    """A GGUF carries its own tokeniser. Comparing position 3 of one
    tokenisation against position 3 of another is a number about nothing."""
    a = _capture("quant", [1, 2, 3], [[0.9, 0.1]] * 3)
    b = _capture("orig", [1, 9, 3], [[0.9, 0.1]] * 3)
    with pytest.raises(Refusal, match="tokenise this prompt differently"):
        behavdiff.compare_captures(a, b, prompt="x")


def test_the_refusal_names_where_they_first_diverge():
    a = _capture("quant", [1, 2, 3], [[0.9, 0.1]] * 3)
    b = _capture("orig", [1, 9, 3], [[0.9, 0.1]] * 3)
    with pytest.raises(Refusal) as err:
        behavdiff.compare_captures(a, b, prompt="x")
    assert "position 1" in str(err.value)


def test_different_lengths_are_refused_with_both_counts():
    a = _capture("quant", [1, 2, 3], [[0.9, 0.1]] * 3)
    b = _capture("orig", [1, 2], [[0.9, 0.1]] * 2)
    with pytest.raises(Refusal) as err:
        behavdiff.compare_captures(a, b, prompt="x")
    assert "3 tokens" in str(err.value) and "2" in str(err.value)


def test_different_vocabularies_are_refused():
    """KL is between distributions over the SAME set of outcomes. Two
    vocabularies are two different sets."""
    a = _capture("quant", [1, 2], [[0.5, 0.5]] * 2)
    b = _capture("orig", [1, 2], [[0.4, 0.3, 0.3]] * 2)
    with pytest.raises(Refusal, match="undefined"):
        behavdiff.compare_captures(a, b, prompt="x")


def test_comparing_a_file_with_itself_is_refused():
    """Every difference would be zero by construction, which is not a result."""
    with pytest.raises(BadRequest, match="same file"):
        behavdiff.compare_behaviour("a.gguf", "a.gguf", "hello")


# ------------------------------------------------------------ the numbers


def test_identical_models_have_zero_divergence_and_no_flips():
    rows = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]
    a = _capture("a", [1, 2], rows)
    b = _capture("b", [1, 2], rows)
    d = behavdiff.compare_captures(a, b, prompt="x")
    assert d.summary()["max_kl"] == 0.0
    assert d.summary()["flips"] == 0


def test_kl_is_the_same_quantity_ablate_reports():
    """Not a second KL with its own conventions. Two KLs in one package drift
    into meaning two different things."""
    from modelmri import ablate

    a = _capture("a", [1], [[0.7, 0.2, 0.1]])
    b = _capture("b", [1], [[0.1, 0.8, 0.1]])
    d = behavdiff.compare_captures(a, b, prompt="x")
    assert d.positions[0].kl == pytest.approx(
        ablate.kl_nats(a.probs[0], b.probs[0]), rel=1e-9
    )


def test_a_flip_is_recorded_with_both_candidates():
    """ "3 tokens changed" without saying which is a statistic about nothing."""
    a = _capture("a", [1], [[0.9, 0.1]])
    b = _capture("b", [1], [[0.1, 0.9]])
    d = behavdiff.compare_captures(a, b, prompt="x")
    p = d.positions[0]
    assert p.flipped
    assert p.top_a != p.top_b
    assert p.p_a > 0 and p.p_b > 0


def test_a_near_tie_flip_is_contested_not_decisive():
    """Measured on SmolLM2-135M Q4_K_M: its one flip sat at a 0.038 margin —
    ' is' at 0.319 against ',' at 0.322. Counting that beside a flip at a 0.9
    margin would be arithmetic over two different events."""
    a = _capture("a", [1], [[0.51, 0.49]])
    b = _capture("b", [1], [[0.49, 0.51]])
    d = behavdiff.compare_captures(a, b, prompt="x")
    assert d.positions[0].flipped
    assert d.positions[0].contested
    assert d.summary()["contested_flips"] == 1
    assert d.summary()["decisive_flips"] == 0


def test_a_confident_flip_is_decisive():
    a = _capture("a", [1], [[0.95, 0.05]])
    b = _capture("b", [1], [[0.05, 0.95]])
    d = behavdiff.compare_captures(a, b, prompt="x")
    assert d.positions[0].flipped and not d.positions[0].contested
    assert d.summary()["decisive_flips"] == 1


def test_both_flip_counts_are_reported_rather_than_netted():
    """The reader gets the split; the tool does not decide for them."""
    s = behavdiff.compare_captures(
        _capture("a", [1, 2], [[0.51, 0.49], [0.95, 0.05]]),
        _capture("b", [1, 2], [[0.49, 0.51], [0.05, 0.95]]),
        prompt="x",
    ).summary()
    assert s["flips"] == 2
    assert s["contested_flips"] + s["decisive_flips"] == s["flips"]


def test_the_whole_per_position_series_survives():
    """An average would hide the one position where the answer changed, which
    is the position the whole feature exists to find."""
    a = _capture("a", [1, 2, 3], [[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
    b = _capture("b", [1, 2, 3], [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])
    d = behavdiff.compare_captures(a, b, prompt="x")
    assert len(d.positions) == 3
    assert [p.flipped for p in d.positions] == [False, True, False]


def test_median_is_reported_beside_the_mean():
    """One large position drags a mean a long way; which one a reader wants
    depends on whether they care about the typical token or the worst."""
    rows_a = [[0.5, 0.5]] * 4 + [[0.99, 0.01]]
    rows_b = [[0.5, 0.5]] * 4 + [[0.01, 0.99]]
    s = behavdiff.compare_captures(
        _capture("a", list(range(5)), rows_a),
        _capture("b", list(range(5)), rows_b),
        prompt="x",
    ).summary()
    assert s["median_kl"] < s["mean_kl"]


def test_the_worst_position_is_named_with_its_token():
    a = _capture("a", [1, 2], [[0.5, 0.5], [0.99, 0.01]], tokens=["cat", "dog"])
    b = _capture("b", [1, 2], [[0.5, 0.5], [0.01, 0.99]], tokens=["cat", "dog"])
    s = behavdiff.compare_captures(a, b, prompt="x").summary()
    assert s["max_kl_at"]["token"] == "dog"


# --------------------------------------------------------------- attention


def test_attention_divergence_is_per_layer():
    at_a = [torch.zeros(2, 2), torch.zeros(2, 2)]
    at_b = [torch.zeros(2, 2), torch.full((2, 2), 0.5)]
    d = behavdiff.compare_captures(
        _capture("a", [1, 2], [[0.5, 0.5]] * 2, attention=at_a),
        _capture("b", [1, 2], [[0.5, 0.5]] * 2, attention=at_b),
        prompt="x",
    )
    assert [r["layer"] for r in d.attention] == [0, 1]
    assert d.attention[0]["mean_abs_diff"] == 0.0
    assert d.attention[1]["mean_abs_diff"] == pytest.approx(0.5)
    assert d.summary()["worst_layer"]["layer"] == 1


def test_missing_attention_is_none_and_noted_never_zero():
    """A zero would read as "the attention was identical", which is a strong
    claim about something that was not measured."""
    d = behavdiff.compare_captures(
        _capture("a", [1], [[0.5, 0.5]]),
        _capture("b", [1], [[0.5, 0.5]]),
        prompt="x",
    )
    assert d.attention is None
    assert any("not zero" in n for n in d.notes)
    assert d.summary()["worst_layer"] is None


def test_a_layer_count_mismatch_is_noted_rather_than_zipped():
    d = behavdiff.compare_captures(
        _capture("a", [1], [[0.5, 0.5]], attention=[torch.zeros(1, 1)]),
        _capture("b", [1], [[0.5, 0.5]], attention=[torch.zeros(1, 1)] * 2),
        prompt="x",
    )
    assert d.attention is None
    assert any("layer counts differ" in n for n in d.notes)


# ------------------------------------------------------------- the report


def test_the_report_says_what_it_does_not_measure():
    """It compares two models through HuggingFace's kernels. llama.cpp has its
    own, so this is the quantiser's damage and not the runtime's."""
    d = behavdiff.compare_captures(
        _capture("a", [1], [[0.5, 0.5]]), _capture("b", [1], [[0.5, 0.5]]), prompt="x"
    )
    means = d.summary()["means"]
    assert "llama.cpp" in means
    assert "one sample" in means


def test_the_report_is_json_safe():
    import json

    d = behavdiff.compare_captures(
        _capture("a", [1, 2], [[0.6, 0.4], [0.3, 0.7]], attention=[torch.zeros(2, 2)]),
        _capture("b", [1, 2], [[0.4, 0.6], [0.7, 0.3]], attention=[torch.ones(2, 2)]),
        prompt="hello",
    )
    json.dumps(d.to_dict())


def test_the_prompt_travels_with_the_result():
    """One prompt is one sample, so the sample has to be named."""
    d = behavdiff.compare_captures(
        _capture("a", [1], [[0.5, 0.5]]),
        _capture("b", [1], [[0.5, 0.5]]),
        prompt="The capital of France is",
    )
    assert d.to_dict()["prompt"] == "The capital of France is"


# ------------------------------------------------------------ classifying


@pytest.mark.parametrize(
    "spec,kind",
    [
        ("model.gguf", "gguf"),
        ("MODEL.GGUF", "gguf"),
        ("Qwen/Qwen3-0.6B", "hf"),
        ("/some/checkpoint/dir", "hf"),
    ],
)
def test_a_side_is_classified_by_what_it_is(spec, kind):
    assert behavdiff.side(spec).kind == kind


def test_a_gguf_side_is_labelled_by_filename_not_by_path():
    s = behavdiff.side("/very/long/path/SmolLM2-135M-Instruct-Q4_K_M.gguf")
    assert s.label == "SmolLM2-135M-Instruct-Q4_K_M.gguf"
