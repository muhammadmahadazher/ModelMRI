"""A chart that adds to 100% when the decomposition does not is a fabrication.

Direct logit attribution is exact only if the final normalisation is linear,
and it is not. This module's whole claim to be honest rests on two things: the
reconstruction residual is measured and shown, and a component smaller than
that residual is labelled unreadable rather than small. Most of these tests
are about those two.
"""

from __future__ import annotations

import os

import pytest

from modelmri.errors import Refusal


@pytest.fixture(scope="module")
def gpt2():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from modelmri import receipts as _receipts

    if _receipts.revision_of("gpt2")[0] is None and not os.environ.get(
        "MODELMRI_TEST_DOWNLOAD"
    ):
        pytest.skip("gpt2 is not in the local model cache")

    from modelmri.runtime import ModelRuntime

    runtime = ModelRuntime()
    try:
        runtime.load("gpt2")
    except Exception as err:
        pytest.skip(f"gpt2 is not available here: {err}")
    yield runtime
    runtime.unload()


@pytest.fixture(scope="module")
def attributed(gpt2):
    from modelmri import dla

    list(
        gpt2.generate_stream(
            "The Eiffel Tower is located in the city of",
            max_new_tokens=1,
            temperature=0.0,
        )
    )
    return dla.attribute(
        gpt2.model,
        gpt2.tokenizer,
        gpt2.last_ids,
        position=gpt2.last_n_prompt_tokens - 1,
    )


# ------------------------------------------------------- it decomposes


def test_it_attributes_the_token_the_model_actually_predicted(attributed):
    assert attributed.token.strip() == "Paris"
    assert attributed.norm_kind == "LayerNorm"


def test_every_component_is_named_by_layer_and_head(attributed):
    heads = [c for c in attributed.components if c.kind == "head"]
    mlps = [c for c in attributed.components if c.kind == "mlp"]
    embeds = [c for c in attributed.components if c.kind == "embed"]

    assert len(heads) == 12 * 12, "gpt2 is 12 layers of 12 heads"
    assert len(mlps) == 12
    assert len(embeds) == 1
    assert all(c.layer is not None and c.head is not None for c in heads)


def test_contributions_can_be_negative(attributed):
    """A component can and does push AGAINST the token the model chose.
    Reporting magnitudes only would hide half the mechanism."""
    assert any(c.logits < 0 for c in attributed.components)


def test_the_pieces_add_up_to_the_real_logit_minus_the_residual(attributed):
    """Which is the arithmetic the residual is defined by, and the reason it
    can be trusted as a floor."""
    total = sum(c.logits for c in attributed.components) + attributed.bias
    assert abs((total + attributed.residual) - attributed.real_logit) < 1e-3


# ------------------------------------------------------- the residual


def test_the_residual_is_measured_and_not_zero_on_gpt2(attributed):
    """The roadmap predicts this and it holds: freezing the LayerNorm scale
    costs something real. Showing it is mandatory — without it the chart is a
    fabricated 100%."""
    assert attributed.residual != 0.0
    assert 0 < attributed.residual_share < 0.5


def test_the_residual_share_is_reported_against_the_real_logit(attributed):
    expected = abs(attributed.residual) / abs(attributed.real_logit)
    # 5 decimal places, because that is what the field is rounded to. A
    # tolerance tighter than the rounding tests the rounding.
    assert abs(attributed.residual_share - expected) < 1e-5


def test_a_component_under_the_residual_is_unreadable_not_zero(attributed):
    """ "Direct-path attribution cannot see indirect effects" is a different
    statement from "this head contributed nothing", and the field that carries
    it is not a magnitude."""
    floor = abs(attributed.residual)
    for component in attributed.components:
        assert component.unreadable == (abs(component.logits) < floor)
    assert any(c.unreadable for c in attributed.components)


def test_the_sentence_says_direct_path_only(attributed):
    """A head that feeds a later head shows near zero here and can still
    decide the answer. That has to travel with the numbers."""
    means = attributed.means()
    assert "DIRECT-PATH ONLY" in means
    assert "not the same as their being unimportant" in means
    assert "reconstruction residual" in means


# --------------------------------------------- the approximation is checked


def test_the_norm_reconstruction_is_verified_before_anything_is_reported(gpt2):
    """The guard caught a real bug in this module twice: decomposing the
    POST-norm stream (hidden_states[-1] is already normalised), and an
    absolute tolerance that was measuring bfloat16 rather than the model."""
    import torch

    from modelmri import dla

    class _Wrong(torch.nn.Module):
        """A norm that is not the affine form this module assumes."""

        def __init__(self, real):
            super().__init__()
            self.weight = real.weight
            self.bias = real.bias
            self.eps = real.eps

        def forward(self, x):
            return torch.tanh(x) * self.weight + self.bias

    base = gpt2.model.transformer
    real = base.ln_f
    base.ln_f = _Wrong(real)
    try:
        with pytest.raises(Refusal, match="does not match the affine form"):
            dla.attribute(gpt2.model, gpt2.tokenizer, gpt2.last_ids)
    finally:
        base.ln_f = real


def test_the_tolerance_is_derived_from_the_dtype_not_chosen():
    """1e-3 absolute refused a healthy model on a bf16 load: at the magnitude
    the stream actually reaches, the reconstruction differed by less than one
    representable bf16 step. The tolerance was measuring the dtype rather than
    the model. `lens.py` records finding the same bug in its own agreement
    check."""
    from modelmri import dla

    assert not hasattr(dla, "NORM_AGREEMENT"), "the absolute epsilon is gone"
    assert dla.NORM_AGREEMENT_STEPS > 0


# --------------------------------------------------------------- trimming


def test_top_k_trims_by_magnitude_and_zero_returns_everything(gpt2, attributed):
    from modelmri import dla

    trimmed = dla.attribute(
        gpt2.model,
        gpt2.tokenizer,
        gpt2.last_ids,
        position=gpt2.last_n_prompt_tokens - 1,
        top_k=5,
    )
    assert len(trimmed.components) == 5
    assert len(attributed.components) > 5
    # Sorted by magnitude, so a trim keeps the strongest rather than the first.
    magnitudes = [abs(c.logits) for c in trimmed.components]
    assert magnitudes == sorted(magnitudes, reverse=True)


# ------------------------------------------------------------ through the API


def test_the_route_answers_and_carries_a_receipt(gpt2):
    from fastapi.testclient import TestClient

    from modelmri.server import create_app

    app = create_app()
    app.state.runtime = gpt2
    client = TestClient(app)
    client.post("/api/model/load", json={"hf_id": "gpt2", "source": "hf"})
    with client.stream(
        "POST",
        "/api/model/prompt",
        json={
            "prompt": "The Eiffel Tower is located in the city of",
            "max_new_tokens": 1,
            "temperature": 0,
        },
    ) as response:
        for _ in response.iter_bytes():
            pass

    body = client.get("/api/attention/direct?top_k=6").json()
    assert body["token"].strip() == "Paris"
    assert len(body["components"]) == 6
    assert body["residual"] != 0
    assert body["receipt"]["op"] == "direct_attribution"
    assert body["n_unreadable"] >= 0


def test_a_recording_refuses_with_a_reason():
    from modelmri.runtime import ModelRuntime

    runtime = ModelRuntime()
    runtime.replay = object()
    with pytest.raises(Refusal, match="This is a recording"):
        runtime.direct_attribution()


# ------------------------------- the count that was dropped, said out loud


def test_the_counts_describe_the_decomposition_not_the_visible_slice():
    """`top_k` sorts by magnitude and cuts, and both counts were read off the
    survivors — the STRONGEST rows, which are the least likely to be below
    the residual floor.

    MEASURED at the default top_k=40: most of the decomposed components fall
    below the floor and never reach the visible slice, and the sentence
    counting them said 0.
    """
    from modelmri.dla import Attribution, Contribution

    shown = [
        Contribution(name=f"L0H{i}", kind="head", layer=0, head=i, logits=1.0 + i)
        for i in range(3)
    ]
    for c in shown:
        c.unreadable = False

    made = Attribution(
        token=" Paris",
        token_id=7,
        position=6,
        real_logit=14.748,
        bias=0.0,
        residual=-0.0737,
        residual_share=0.005,
        norm_kind="layernorm",
        components=shown,
        n_components=157,
        n_unreadable=117,
    )

    body = made.to_dict()
    assert body["n_components"] == 157
    assert body["n_unreadable"] == 117, (
        "counted over every component, not over the rows that survived top_k"
    )
    assert len(body["components"]) == 3

    said = made.means()
    assert "3 of 157 components are listed" in said
    assert "other 154 were decomposed and are not shown" in said
    # The two counts are independent: 154 were cut from the table, 117 sit
    # below the residual floor, and neither implies the other.
    assert "117 of 157 components fall below that residual" in said


def test_nothing_is_claimed_to_be_dropped_when_nothing_was():
    """top_k=0 returns everything, and the sentence must not invent a cut."""
    from modelmri.dla import Attribution, Contribution

    rows = [Contribution(name="L0H0", kind="head", layer=0, head=0, logits=2.0)]
    made = Attribution(
        token=" x",
        token_id=1,
        position=0,
        real_logit=2.0,
        bias=0.0,
        residual=0.0,
        residual_share=0.0,
        norm_kind="layernorm",
        components=rows,
        n_components=1,
        n_unreadable=0,
    )
    said = made.means()
    assert "are not shown" not in said
    assert said.startswith("Direct contribution")


def test_the_real_model_reports_both_counts(gpt2):
    """The end-to-end version of the same claim, on a real decomposition.

    `attributed` uses the default top_k=0 (everything); this goes through the
    runtime, whose default is 40 — which is where the cut actually happens.
    """
    "".join(
        gpt2.generate_stream("The Eiffel Tower is in the city of", 4, 0.0, commit=True)
    )
    out = gpt2.direct_attribution()

    assert out["n_components"] > len(out["components"]), (
        "gpt2 decomposes into 157 components and the runtime default is 40"
    )
    # The surviving rows are the strongest, so counting unreadable over them
    # alone reported zero where the true figure was 117.
    assert out["n_unreadable"] > sum(1 for c in out["components"] if c["unreadable"])
    assert f"of {out['n_components']} components are listed" in out["means"]
