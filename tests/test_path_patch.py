"""An ordered list of senders is easy to produce and easy to over-read.

Path patching answers "what put it there" for a bright cell in the node grid.
The failure that matters is not a wrong number, it is a CONFIDENT ORDERING of
numbers that are tied — which is what happens when a recovery fraction is
quantised by the dtype and nothing on screen says so.
"""

from __future__ import annotations

import os

import pytest

from modelmri.errors import BadRequest

CLEAN = "The Eiffel Tower is located in the city of"
CORRUPT = "The Colosseum is located in the city of"


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
def traced(gpt2):
    return gpt2.path_trace(CLEAN, CORRUPT, layer=10, position=10)


# ------------------------------------------------------------- it measures


def test_every_earlier_component_is_scored_as_a_sender(traced, gpt2):
    n_heads = int(gpt2.model.config.num_attention_heads)
    # Ten layers of twelve heads plus ten MLPs, into layer 10.
    assert traced["n_senders"] == 10 * (n_heads + 1)
    names = {row["name"] for row in traced["senders"]}
    assert "L9 MLP" in names
    assert "L0H0" in names
    assert not any(name.startswith("L10") for name in names), (
        "the receiver's own layer cannot have written into its input"
    )


def test_the_score_is_the_same_fraction_the_node_grid_reports(traced):
    """So an edge number and a node number are on one scale. If they were not,
    reading them together would be the mistake the feature invites."""
    assert "recovery" in traced["senders"][0]
    assert traced["gap"] > 0
    assert "Same fraction the node grid reports" in traced["means"]


def test_senders_are_ranked_by_recovery(traced):
    scores = [row["recovery"] for row in traced["senders"]]
    assert scores == sorted(scores, reverse=True)


def test_both_controls_run_on_the_top_senders(traced):
    controlled = [row for row in traced["senders"] if "clears_control" in row]
    assert controlled, "the strongest senders are controlled"
    for row in controlled:
        assert row["control_draws"] == 8
        assert "shifted_position" in row
        # The flags are computed on FULL precision and reported rounded to 6
        # decimals, so they cannot be re-derived from the reported numbers: on
        # macOS `recovery` and `control_max` both rounded to 0.014085 while
        # the underlying comparison was a genuine win. Assert consistency only
        # where the rounded values actually separate.
        for flag, against in (
            ("clears_control", "control_max"),
            ("clears_position", "shifted_position"),
        ):
            if abs(row["recovery"] - row[against]) > 1e-6:
                assert row[flag] == (row["recovery"] > row[against])


def test_uncontrolled_senders_carry_a_score_and_no_verdict(traced):
    """Absence of a verdict is not a verdict. A sender that was never
    controlled must not look like one that failed."""
    uncontrolled = [row for row in traced["senders"] if "clears_control" not in row]
    assert uncontrolled, "not every sender is controlled — that is the point"
    for row in uncontrolled:
        assert "recovery" in row
        assert "clears_position" not in row


# --------------------------------------------------- what it refuses to hide


def test_the_seeding_rule_is_stated(traced):
    """Edge count is quadratic in general. This is linear only because the
    receiver is fixed, and saying which edges were considered is the
    difference between "the strongest sender" and "the strongest we looked
    at"."""
    seeding = traced["seeding"]
    assert "every attention head and MLP" in seeding
    assert "Controls ran on the top" in seeding
    assert str(traced["n_controlled"]) in seeding


def test_the_scope_names_what_it_did_not_split(traced):
    """v1 does not split q/k/v, and a reader who assumed otherwise would take
    "head 9.6 wrote it" as "head 9.6's query carried it"."""
    scope = traced["scope"]
    assert "residual receivers only" in scope
    assert "query, key or value" in scope


def test_the_resolution_of_the_recovery_fraction_is_reported(traced):
    """MEASURED: on gpt2 in bfloat16 the logits reach 128, where one
    representable step is 1.0, so every sender scored a multiple of 0.125 on
    the reference pair and a dozen tied exactly. Without this number there is
    nothing on screen to say which part of the ordering is real."""
    resolution = traced["recovery_resolution"]
    assert resolution > 0
    assert "RESOLUTION" in traced["means"]
    assert "tied, not ranked" in traced["means"]


def test_the_node_grid_reports_the_same_resolution(gpt2):
    """It uses the identical formula and had the identical blind spot."""
    node = gpt2.patch_trace(CLEAN, CORRUPT)
    assert node["recovery_resolution"] > 0


def test_many_senders_really_are_tied_at_this_precision(traced):
    """The reason the field exists, asserted rather than described."""
    resolution = traced["recovery_resolution"]
    top = traced["senders"][0]["recovery"]
    tied = [row for row in traced["senders"] if abs(row["recovery"] - top) < resolution]
    assert len(tied) > 1, (
        "on this pair the quantisation ties several senders with the top one, "
        "which is exactly what the resolution field warns about"
    )


# ------------------------------------------------------------- it refuses


def test_layer_zero_has_no_earlier_sender(gpt2):
    with pytest.raises(BadRequest, match="no earlier component"):
        gpt2.path_trace(CLEAN, CORRUPT, layer=0, position=5)


def test_a_layer_outside_the_model_is_refused(gpt2):
    with pytest.raises(BadRequest, match="outside this model"):
        gpt2.path_trace(CLEAN, CORRUPT, layer=999, position=5)


def test_a_position_outside_the_prompt_is_refused(gpt2):
    with pytest.raises(BadRequest, match="outside these"):
        gpt2.path_trace(CLEAN, CORRUPT, layer=5, position=999)


def test_prompts_of_different_lengths_are_refused(gpt2):
    """Position N only means the same thing in both runs if they tokenise to
    the same length."""
    with pytest.raises(BadRequest, match="tokenise"):
        gpt2.path_trace(CLEAN, "A much shorter one", layer=5, position=3)


def test_two_prompts_with_the_same_answer_are_refused(gpt2):
    """The denominator is the gap between the two answers."""
    with pytest.raises(BadRequest, match="same token|disagree by only"):
        gpt2.path_trace(CLEAN, CLEAN, layer=5, position=5)


# ------------------------------------------------------------ the harness


def test_patching_a_sender_with_its_own_contribution_is_a_no_op(gpt2):
    """The identity check the node grid documents, at the edge level: adding
    (corrupt - corrupt) must change nothing, or every number here is measuring
    the harness rather than the model."""
    import torch

    from modelmri import patch

    n_layers = int(gpt2.model.config.num_hidden_layers)
    blocks = [gpt2._block(i) for i in range(n_layers)]
    ids = gpt2.tokenizer(CORRUPT, return_tensors="pt")["input_ids"].to(gpt2.device)
    with torch.no_grad():
        before = gpt2.model(ids).logits[0, -1].float().clone()

    zero = torch.zeros(int(gpt2.model.config.hidden_size), device=gpt2.device)
    handle = patch._add_at(blocks[6], 4, zero)
    try:
        with torch.no_grad():
            after = gpt2.model(ids).logits[0, -1].float()
    finally:
        handle.remove()
    assert torch.equal(before, after), "adding zero must be exactly a no-op"


def test_the_receipt_records_both_prompts(traced):
    """A patching result is a statement about a PAIR."""
    request = traced["receipt"]["request"]
    assert request["receiver_layer"] == 10
    assert request["clean_sha256"] != request["corrupt_sha256"]


def test_the_report_survives_json(traced):
    import json

    out = json.loads(json.dumps(traced, allow_nan=False))
    assert out["n_senders"] > 0
    assert out["receiver"] == {"layer": 10, "position": 10}
