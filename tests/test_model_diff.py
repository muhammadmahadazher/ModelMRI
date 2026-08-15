"""A finetune diff over a PROMPT SET, and what it refuses to say about one.

`behavdiff` answers "what did quantising cost me", where both sides are the
same weights and one prompt is a fair sample. This answers "what did my
finetune change", where it is not: the finetune changed the model on purpose,
in some places and not others, and one prompt's diff presented as a property
of the finetune is the error the whole module exists to refuse.

So most of these tests are about the SPREAD surviving into every claim, and
about the pair being refused when a per-layer table would line up the wrong
things.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from modelmri import model_diff as md  # noqa: E402
from modelmri.errors import BadRequest  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The largest planet in the solar system is",
    "Photosynthesis converts sunlight into",
    "The author of Hamlet was",
    "A triangle has this many sides:",
]


class TinyLM(nn.Module):
    """Small enough to build twice, shaped like the thing being diffed.

    `hidden_states[i + 1]` is the output of block `i`, exactly as a
    HuggingFace decoder reports them — which is why a change planted in block
    3 must first show at index 4 and not at index 3.
    """

    def __init__(self, vocab=64, hidden=32, layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.blocks = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.head = nn.Linear(hidden, vocab)
        self.config = type(
            "C",
            (),
            {
                "num_hidden_layers": layers,
                "hidden_size": hidden,
                "vocab_size": vocab,
            },
        )()

    def forward(self, input_ids, output_hidden_states=False, **kw):
        x = self.embed(input_ids)
        hidden = [x]
        for block in self.blocks:
            x = x + torch.tanh(block(x))
            hidden.append(x)
        return type("O", (), {"logits": self.head(x), "hidden_states": hidden})()


class FakeTokenizer:
    def __init__(self, shift: int = 0):
        self.shift = shift

    def __call__(self, text, return_tensors=None):
        ids = torch.tensor([[((ord(c) + self.shift) % 60) + 1 for c in text[:24]]])
        return type("E", (), {"input_ids": ids})()


def _loader(models, tokenizers=None):
    def load_side(spec):
        model = models[spec]
        model.eval()
        tok = (tokenizers or {}).get(spec) or FakeTokenizer()
        return model, tok, lambda: None

    return load_side


@pytest.fixture(scope="module")
def pair():
    """A base, an identical twin, and one with block 3 perturbed."""
    torch.manual_seed(0)
    base = TinyLM()
    twin = TinyLM()
    twin.load_state_dict(base.state_dict())
    tuned = TinyLM()
    tuned.load_state_dict(base.state_dict())
    with torch.no_grad():
        tuned.blocks[3].weight.add_(torch.randn_like(tuned.blocks[3].weight) * 0.6)
    return base, twin, tuned


# ----------------------------------------------------------- what it finds


def test_identical_weights_diverge_nowhere(pair):
    """The floor case, and the one a threshold copied from another pair would
    get wrong: two models with the same weights must produce no divergence at
    all, not a small one."""
    base, twin, _ = pair
    out = md.compare(_loader({"a": base, "b": twin}), "a", "b", PROMPTS)
    assert out.consensus_layer is None
    assert out.kl.median == 0.0
    assert all(p.first_divergent_layer is None for p in out.prompts)
    assert "THE COSINE NEVER FALLS" in out.means()


def test_a_planted_change_is_found_at_the_layer_it_was_planted(pair):
    """MEASURED: block 3 perturbed, and layer 4 comes back as the first
    divergent on 6 of 6 prompts — `hidden_states[i + 1]` is the output of
    block `i`, so 4 is where block 3's change first appears."""
    base, _, tuned = pair
    out = md.compare(_loader({"a": base, "b": tuned}), "a", "b", PROMPTS)
    assert out.consensus_layer == 4
    assert out.consensus_share == 1.0
    assert [p.first_divergent_layer for p in out.prompts] == [4] * len(PROMPTS)


def test_the_layers_before_the_change_are_untouched(pair):
    """A cosine of 1.0 through the early layers is what makes the divergence
    point meaningful. If everything drifted, naming a layer would be naming
    the first one that happened to cross a line."""
    base, _, tuned = pair
    out = md.compare(_loader({"a": base, "b": tuned}), "a", "b", PROMPTS)
    for row in out.layers[:4]:
        assert row.median == pytest.approx(1.0, abs=1e-6)
    assert out.layers[4].median < 0.9


# --------------------------------------------------- the spread is the claim


def test_every_headline_number_is_a_median_with_its_middle_half(pair):
    base, _, tuned = pair
    out = md.compare(_loader({"a": base, "b": tuned}), "a", "b", PROMPTS)
    assert out.kl.n == len(PROMPTS)
    assert out.kl.low <= out.kl.median <= out.kl.high
    assert "MEDIAN over those prompts" in out.means()
    assert "middle half" in out.means()


def test_a_set_that_disagrees_with_itself_says_so_rather_than_averaging():
    """The whole point. A finetune that moved one prompt by 4 nats and three
    prompts by nothing has no single amount it moved the answer by, and a mean
    would print one."""
    wide = md.summarise(
        "a",
        "b",
        [
            md.PromptResult(prompt="p", n_tokens=4, mean_kl=k, max_kl=k, flips=0)
            for k in (0.0, 0.01, 0.02, 4.0)
        ],
        n_layers=0,
    )
    assert not wide.kl.stable()
    assert "NOT typical" in wide.means()
    assert "no single amount" in wide.means()


def test_a_set_that_agrees_says_that_instead():
    tight = md.summarise(
        "a",
        "b",
        [
            md.PromptResult(prompt="p", n_tokens=4, mean_kl=k, max_kl=k, flips=0)
            for k in (0.40, 0.41, 0.42, 0.43)
        ],
        n_layers=0,
    )
    assert tight.kl.stable()
    assert "the prompts agree with each other" in tight.means()


def test_a_plurality_is_not_reported_as_the_place_it_changed():
    """A layer that was first on 2 of 6 prompts is the commonest of several,
    not where the finetune starts to differ."""
    scattered = md.summarise(
        "a",
        "b",
        [
            md.PromptResult(
                prompt="p", n_tokens=4, mean_kl=0.1, max_kl=0.1, flips=0,
                first_divergent_layer=layer, cosine=[1.0] * 6,
            )
            for layer in (1, 1, 2, 3, 4, 5)
        ],
        n_layers=6,
    )
    assert scattered.consensus_layer == 1
    assert scattered.consensus_share == pytest.approx(2 / 6, abs=1e-4)
    assert "a plurality and not a majority" in scattered.means()
    assert "MOVES between your prompts" in scattered.means()


def test_the_consensus_share_is_out_of_all_prompts_not_the_ones_that_diverged():
    """A layer first on both of the two prompts that diverged out of twenty is
    not "100% of prompts", and that is exactly how a rate becomes a claim
    nobody measured."""
    mostly_quiet = md.summarise(
        "a",
        "b",
        [
            md.PromptResult(
                prompt="p", n_tokens=4, mean_kl=0.0, max_kl=0.0, flips=0,
                first_divergent_layer=layer, cosine=[1.0] * 4,
            )
            for layer in (2, 2, None, None, None, None, None, None)
        ],
        n_layers=4,
    )
    assert mostly_quiet.consensus_layer == 2
    assert mostly_quiet.consensus_share == pytest.approx(2 / 8, abs=1e-4)


def test_one_prompt_is_refused_because_the_output_is_a_spread():
    with pytest.raises(BadRequest, match="at least 4 prompts"):
        md.plan(["only one"])


def test_the_refusal_says_why_one_prompt_cannot_answer_this():
    with pytest.raises(BadRequest, match="is a sample"):
        md.plan(["a", "b"])


def test_too_many_prompts_is_refused_with_what_it_would_cost():
    with pytest.raises(BadRequest, match="forward pass on both sides"):
        md.plan([f"prompt {i}" for i in range(md.MAX_PROMPTS + 1)])


# ------------------------------------------------------ the pair has to match


def test_different_layer_counts_are_refused_and_both_are_named():
    with pytest.raises(BadRequest, match="24 layers and .* has 32"):
        md.check_pair(
            {"n_layers": 24, "hidden": 2048, "vocab": 32000},
            {"n_layers": 32, "hidden": 2048, "vocab": 32000},
            "base",
            "finetune",
        )


def test_depths_are_never_normalised_to_a_fraction_to_make_them_line_up():
    """Layer 12 of 24 and layer 12 of 32 are not the same place, and a depth
    fraction would assert that they are."""
    with pytest.raises(BadRequest) as caught:
        md.check_pair(
            {"n_layers": 24, "hidden": 8, "vocab": 8},
            {"n_layers": 32, "hidden": 8, "vocab": 8},
            "base",
            "finetune",
        )
    assert "depth fraction" in str(caught.value)


def test_different_hidden_sizes_are_refused():
    with pytest.raises(BadRequest, match="no cosine between vectors"):
        md.check_pair(
            {"n_layers": 24, "hidden": 2048, "vocab": 32000},
            {"n_layers": 24, "hidden": 4096, "vocab": 32000},
            "base",
            "finetune",
        )


def test_a_finetune_that_added_tokens_is_refused_with_both_counts():
    """A real and common case — which is why it says both numbers rather than
    refusing anonymously."""
    with pytest.raises(BadRequest, match="32,000 and .* has 32,016"):
        md.check_pair(
            {"n_layers": 24, "hidden": 2048, "vocab": 32000},
            {"n_layers": 24, "hidden": 2048, "vocab": 32016},
            "base",
            "finetune",
        )


def test_a_matching_pair_passes():
    shape = {"n_layers": 24, "hidden": 2048, "vocab": 32000}
    md.check_pair(shape, dict(shape), "base", "finetune")


def test_two_unread_shapes_do_not_pass_the_gate_as_a_match():
    """`0 != 0` is false, so two configs that both failed to state their depth
    used to sail through every check below and get compared on shapes nobody
    had read."""
    blank = {"n_layers": 0, "hidden": 0, "vocab": 0}
    with pytest.raises(md.DiffError, match="does not state its"):
        md.check_pair(blank, dict(blank), "base", "finetune")


def test_the_refusal_names_the_side_and_the_field():
    """"Incompatible" sends the reader to check both models."""
    good = {"n_layers": 24, "hidden": 2048, "vocab": 32000}
    with pytest.raises(md.DiffError) as caught:
        md.check_pair(good, {**good, "hidden": 0}, "base", "finetune")
    message = str(caught.value)
    assert "finetune's config" in message and "hidden" in message
    assert "base's config" not in message


def test_an_unstated_head_count_is_not_priced_as_thirty_six_passes():
    """(0 * 0 + 3) * 2 * 6 = 36, quoted for a run of several thousand. A
    preflight that under-quotes is the number somebody plans around."""
    assert md.head_pass_estimate(12, 12, 6) == (12 * 12 + 3) * 2 * 6
    with pytest.raises(md.DiffError, match="cannot price"):
        md.head_pass_estimate(12, 0, 6)
    with pytest.raises(md.DiffError, match="cannot price"):
        md.head_pass_estimate(0, 12, 6)


def test_the_same_model_on_both_sides_is_refused(pair):
    base, _, _ = pair
    with pytest.raises(BadRequest, match="zero by construction"):
        md.compare(_loader({"a": base}), "a", "a", PROMPTS)


# --------------------------------------------- tokenisers, checked per prompt


def test_a_prompt_the_two_tokenisers_split_differently_is_refused():
    """Two models can share a tokeniser config and still disagree on one
    string, and the disagreement is invisible until that string arrives.
    Position 7 of one run being a different word from position 7 of the other
    is silent and produces a table of nonsense."""
    with pytest.raises(BadRequest, match="first differing at position 2"):
        md.check_tokens([1, 2, 3, 4], [1, 2, 9, 4], "some prompt")


def test_the_token_check_runs_per_prompt_and_not_once_for_the_pair(pair):
    base, twin, _ = pair
    shifted = FakeTokenizer(shift=1)
    with pytest.raises(BadRequest, match="tokenisers split this prompt"):
        md.compare(
            _loader({"a": base, "b": twin}, {"b": shifted}), "a", "b", PROMPTS
        )


def test_identical_token_ids_pass():
    md.check_tokens([1, 2, 3], [1, 2, 3], "p")


# ------------------------------------------------------------ what it is not


def test_it_never_calls_itself_model_diffing(pair):
    """crosscode and OpenMOSS train a shared crosscoder and can say a FEATURE
    moved. This cannot, and the summary says so rather than leaving a reader
    to assume the stronger claim."""
    base, _, tuned = pair
    out = md.compare(_loader({"a": base, "b": tuned}), "a", "b", PROMPTS)
    means = out.means()
    assert "cannot say that a shared feature moved" in means
    assert "not model diffing in that sense" in means


def test_cosine_is_used_rather_than_distance(pair):
    """The two streams can differ in SCALE without differing in direction — a
    finetune that changed a norm gain moves every vector's length and none of
    their meanings — and a distance would report that as the model having
    changed everywhere."""
    a = [torch.randn(4, 8)]
    scaled = [a[0] * 7.0]
    assert md.cosine_per_layer(a, scaled)[0] == pytest.approx(1.0, abs=1e-5)


def test_the_report_survives_json(pair):
    base, _, tuned = pair
    out = md.compare(_loader({"a": base, "b": tuned}), "a", "b", PROMPTS)
    doc = json.loads(json.dumps(out.to_dict(), allow_nan=False))
    assert doc["consensus_layer"] == 4
    assert "means" in doc
    assert len(doc["prompts"]) == len(PROMPTS)


def test_each_side_is_loaded_exactly_once(pair):
    """The models worth comparing are near the limit of the machine, so
    reloading per prompt would turn a 20-prompt comparison into 40 loads."""
    base, _, tuned = pair
    loads: list[str] = []
    models = {"a": base, "b": tuned}

    def counting_loader(spec):
        loads.append(spec)
        return models[spec], FakeTokenizer(), lambda: None

    md.compare(counting_loader, "a", "b", PROMPTS)
    assert loads == ["a", "b"]


def test_the_model_is_released_even_when_a_capture_raises(pair):
    """A capture that raises must still give the memory back, or the second
    side has nowhere to load into and the real error is buried under an
    out-of-memory."""
    base, _, _ = pair
    released: list[str] = []

    class Exploding(nn.Module):
        config = type("C", (), {"num_hidden_layers": 6, "hidden_size": 32,
                                "vocab_size": 64})()

        def parameters(self, recurse=True):
            return iter([torch.zeros(1)])

        def forward(self, *a, **k):
            raise RuntimeError("boom")

    def loader(spec):
        model = base if spec == "a" else Exploding()
        return model, FakeTokenizer(), lambda: released.append(spec)

    with pytest.raises(RuntimeError, match="boom"):
        md.compare(loader, "a", "b", PROMPTS)
    assert "b" in released, "the failing side was never released"


# ------------------------------------------------- refusing before the loads


def test_the_gate_can_run_from_configs_alone(monkeypatch):
    """A few hundred bytes of JSON against several gigabytes of weights.

    Without this the compatibility gate fires only once BOTH models have
    loaded and been released — and the models worth comparing are exactly the
    ones near the limit of the machine, so refusing a mismatched pair after
    two full loads is the wrong way round.
    """
    seen: list[str] = []

    class FakeConfig:
        num_hidden_layers = 12
        hidden_size = 768
        vocab_size = 50257

    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        classmethod(lambda cls, spec, *a, **k: seen.append(spec) or FakeConfig()),
    )
    out = md.shape_without_loading("some/model")
    # `n_heads` comes back 0 here because this fake config does not publish
    # `num_attention_heads` — which is the honest reading of a config that
    # does not say, and the head sweep refuses on it rather than guessing.
    assert out == {"n_layers": 12, "hidden": 768, "vocab": 50257, "n_heads": 0}
    assert seen == ["some/model"]


def test_an_unreadable_config_is_not_a_refusal(monkeypatch):
    """A local GGUF, an unresolvable path, no network for a Hub id. None of
    those mean the pair is incompatible — they mean the cheap path did not
    work, and the check that needs the model still runs."""
    import transformers

    def boom(cls, spec, *a, **k):
        raise OSError("no network")

    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", classmethod(boom)
    )
    assert md.shape_without_loading("some/model") is None


def test_a_mismatched_pair_is_refused_before_either_model_is_loaded(monkeypatch):
    loads: list[str] = []

    shapes = {
        "base": {"n_layers": 12, "hidden": 768, "vocab": 50257},
        "small": {"n_layers": 6, "hidden": 768, "vocab": 50257},
    }
    monkeypatch.setattr(md, "shape_without_loading", lambda spec: shapes[spec])

    def loader(spec):
        loads.append(spec)
        raise AssertionError("must refuse before loading anything")

    with pytest.raises(BadRequest, match="12 layers and small has 6"):
        md.compare(loader, "base", "small", PROMPTS)
    assert loads == []


def test_the_example_layer_in_the_refusal_exists_in_both_models():
    """MEASURED on gpt2 against distilgpt2: a hardcoded example produced
    "layer 12 of 12 is the same place as layer 12 of 6" — an illustration
    naming a layer one of the two models does not have."""
    with pytest.raises(BadRequest) as caught:
        md.check_pair(
            {"n_layers": 12, "hidden": 768, "vocab": 50257},
            {"n_layers": 6, "hidden": 768, "vocab": 50257},
            "gpt2",
            "distilgpt2",
        )
    message = str(caught.value)
    assert "layer 3 of 12 is the same place as layer 3 of 6" in message


# ------------------------------------------ where the streams come apart


def test_the_divergence_layer_needs_no_threshold():
    """The first version compared each layer against a floor, and the floor
    was the constant 0.999 wearing a docstring that claimed it was measured on
    the pair.

    MEASURED on gpt2 against a copy with one head zeroed in block 6: the
    cosine reads 1.000000000 through layer 6 and 0.999475 at layer 7 — exactly
    where that block's output first appears — and 0.999475 sits ABOVE 0.999. A
    real divergence, correctly measured, reported as none at all.
    """
    curve = [1.0] * 7 + [0.999475, 0.999532, 0.999616, 0.999708, 0.999845]
    layer, drop = md.steepest_drop(curve)
    assert layer == 7
    assert drop == pytest.approx(5.25e-4, rel=1e-3)


def test_streams_that_never_come_apart_report_no_layer():
    layer, drop = md.steepest_drop([1.0] * 12)
    assert layer is None
    assert drop == 0.0


def test_a_curve_that_only_rises_reports_no_layer():
    """Cosine can increase between layers — two streams that differ early can
    re-converge — and a rise is not a drop."""
    layer, _ = md.steepest_drop([0.90, 0.93, 0.97, 0.99])
    assert layer is None


def test_the_drop_size_travels_with_the_layer(pair):
    """5e-04 and 0.4 are both "the steepest drop" and only one of them is a
    change, so the size is printed beside the layer rather than left to be
    asked for."""
    base, _, tuned = pair
    out = md.compare(_loader({"a": base, "b": tuned}), "a", "b", PROMPTS)
    assert all(p.drop > 0 for p in out.prompts)
    assert "read that size before the layer" in out.means()


# --------------------------------------------------------- the head half


def _ranking(scores: dict) -> dict:
    return {k: float(v) for k, v in scores.items()}


def test_a_head_that_stopped_mattering_is_found():
    """MEASURED on gpt2 against a copy with L0H10's output projection zeroed:
    that head goes from 0.3583 nats to 0.0000 and from the top eight on 4 of 4
    prompts to none of them."""
    a = [_ranking({(0, 10): 0.36, (1, 11): 0.003, (2, 5): 0.02})] * 4
    b = [_ranking({(0, 10): 0.00, (1, 11): 1.25, (2, 5): 0.34})] * 4
    heads = md.summarise_heads(a, b)
    by = {(h.layer, h.head): h for h in heads}
    assert by[(0, 10)].median_a == pytest.approx(0.36)
    assert by[(0, 10)].median_b == 0.0
    assert by[(0, 10)].shift == pytest.approx(-0.36)


def test_compensation_ranks_as_high_as_damage():
    """Removing a head redistributes the work, and the head that PICKED IT UP
    is as much a finding as the one that lost it. MEASURED: zeroing L0H10 sent
    L1H11 from 0.0026 to 1.2494 — a larger move than the damage itself."""
    a = [_ranking({(0, 10): 0.36, (1, 11): 0.003})] * 4
    b = [_ranking({(0, 10): 0.00, (1, 11): 1.25})] * 4
    heads = md.summarise_heads(a, b)
    # Sorted by the SIZE of the move, so the compensating head comes first.
    assert (heads[0].layer, heads[0].head) == (1, 11)
    assert heads[0].shift > 0
    assert heads[1].shift < 0


def test_both_sides_are_carried_not_only_the_difference():
    """A head that went from 0.02 to 0.06 and one that went from 4.00 to 4.04
    moved by the same amount and are not the same finding."""
    a = [_ranking({(0, 0): 0.02, (1, 0): 4.00})] * 4
    b = [_ranking({(0, 0): 0.06, (1, 0): 4.04})] * 4
    heads = md.summarise_heads(a, b)
    for h in heads:
        assert h.shift == pytest.approx(0.04)
    assert {h.median_a for h in heads} == {0.02, 4.0}


def test_the_top_k_membership_is_counted_separately_from_the_score():
    """Which head carries the answer and how much it carries are different
    claims, and a finetune can hold one while breaking the other."""
    a = [_ranking({(0, i): 1.0 - i * 0.1 for i in range(10)})] * 3
    b = [_ranking({(0, i): 0.1 + i * 0.1 for i in range(10)})] * 3
    heads = md.summarise_heads(a, b)
    by = {(h.layer, h.head): h for h in heads}
    # Head 0 was top on every prompt in A and is bottom in B.
    assert by[(0, 0)].top_a == 3
    assert by[(0, 0)].top_b == 0
    assert by[(0, 9)].top_a == 0
    assert by[(0, 9)].top_b == 3


def test_a_head_missing_from_one_side_is_skipped_rather_than_scored():
    a = [_ranking({(0, 0): 1.0, (0, 1): 0.5})] * 4
    b = [_ranking({(0, 0): 1.0})] * 4
    heads = md.summarise_heads(a, b)
    assert {(h.layer, h.head) for h in heads} == {(0, 0)}


def test_the_head_cost_is_projected_before_it_is_run():
    """`rank_heads` is one pass per head plus a base, a repeat for the noise
    floor and a joint check — times two sides, times every prompt. MEASURED on
    gpt2 with four prompts: 1,176 passes."""
    assert md.head_pass_estimate(12, 12, 4) == 1176
    assert md.head_pass_estimate(28, 16, 6) == 5412


def test_the_head_half_is_off_by_default(pair):
    """It costs two orders of magnitude more than everything else here, so it
    is opted into rather than out of."""
    base, _, tuned = pair
    out = md.compare(_loader({"a": base, "b": tuned}), "a", "b", PROMPTS)
    assert out.heads == []
    assert out.head_passes == 0


def test_the_summary_names_both_sides_of_the_top_head():
    diff = md.summarise(
        "base",
        "tuned",
        [
            md.PromptResult(prompt="p", n_tokens=4, mean_kl=0.1, max_kl=0.1, flips=0)
            for _ in range(4)
        ],
        n_layers=0,
    )
    diff.heads = md.summarise_heads(
        [_ranking({(6, 9): 0.02})] * 4, [_ranking({(6, 9): 0.06})] * 4
    )
    means = diff.means()
    assert "L6H9" in means
    assert "0.0200" in means and "0.0600" in means
    assert "BOTH sides are printed" in means


# --------------------------------------------------------- the token half


def _attr(scores: dict, floor: float, tokens: dict | None = None) -> dict:
    return {
        "scores": {k: float(v) for k, v in scores.items()},
        "tokens": tokens or {k: f"tok{k}" for k in scores},
        "floor": floor,
    }


def test_each_side_is_judged_against_its_own_noise_floor():
    """The two models have DIFFERENT floors — a finetune changes the
    arithmetic as well as the weights — so "above 0.05 in both" would compare
    one model's signal against the other model's noise."""
    a = [_attr({1: 0.9, 2: 0.02, 3: 0.5}, floor=0.05)]
    b = [_attr({1: 0.9, 2: 0.40, 3: 0.01}, floor=0.30)]
    by = {t.index: t for t in md.summarise_tokens(a, b)}
    # 0.40 clears B's 0.30; 0.02 does not clear A's 0.05.
    assert by[2].newly_used is True
    assert by[2].newly_ignored is False
    # 0.50 clears A's 0.05; 0.01 does not clear B's 0.30.
    assert by[3].newly_ignored is True
    # Above both floors: no crossing, whatever the scores did.
    assert by[1].newly_used is False and by[1].newly_ignored is False


def test_a_shared_threshold_would_get_this_wrong():
    """The same data under one pooled floor: token 2 would look unchanged and
    token 3 would look newly ignored for the wrong reason."""
    a = [_attr({2: 0.02}, floor=0.05)]
    b = [_attr({2: 0.40}, floor=0.30)]
    assert md.summarise_tokens(a, b)[0].newly_used is True
    # Same scores, same floors on both sides -> no crossing at all.
    flat_a = [_attr({2: 0.02}, floor=0.05)]
    flat_b = [_attr({2: 0.40}, floor=0.05)]
    row = md.summarise_tokens(flat_a, flat_b)[0]
    assert row.newly_used is True  # 0.02 below 0.05, 0.40 above it


def test_crossings_are_ranked_above_mere_movement():
    """A token that changed KIND is the finding; one that changed degree is
    context for it."""
    a = [_attr({1: 5.0, 2: 0.01}, floor=0.05)]
    b = [_attr({1: 9.0, 2: 0.90}, floor=0.05)]
    rows = md.summarise_tokens(a, b)
    # Token 2 moved less in absolute terms but crossed the floor.
    assert rows[0].index == 2 and rows[0].newly_used
    assert rows[1].index == 1 and not rows[1].newly_used


def test_tokens_are_kept_per_prompt_and_never_pooled():
    """Token 4 of one prompt and token 4 of another are different words, and
    averaging them would be arithmetic on a coincidence of position."""
    a = [_attr({1: 0.01}, floor=0.05), _attr({1: 0.90}, floor=0.05)]
    b = [_attr({1: 0.90}, floor=0.05), _attr({1: 0.01}, floor=0.05)]
    rows = md.summarise_tokens(a, b)
    assert {r.prompt_index for r in rows} == {0, 1}
    assert len(rows) == 2
    # One gained, one lost — a pooled mean would have shown neither.
    assert sum(r.newly_used for r in rows) == 1
    assert sum(r.newly_ignored for r in rows) == 1


def test_a_token_missing_from_one_side_is_skipped():
    a = [_attr({1: 0.9, 2: 0.5}, floor=0.05)]
    b = [_attr({1: 0.9}, floor=0.05)]
    assert [r.index for r in md.summarise_tokens(a, b)] == [1]


def test_the_token_cost_is_projected_before_it_is_run():
    assert md.token_pass_estimate(24, 4) == 248
    # Capped at the attribution module's own candidate limit.
    assert md.token_pass_estimate(500, 1) == (64 + 7) * 2


def test_the_token_half_is_off_by_default(pair):
    base, _, tuned = pair
    out = md.compare(_loader({"a": base, "b": tuned}), "a", "b", PROMPTS)
    assert out.tokens == []


def test_the_summary_names_gained_and_lost_tokens_separately():
    diff = md.summarise(
        "base",
        "tuned",
        [
            md.PromptResult(prompt="p", n_tokens=4, mean_kl=0.1, max_kl=0.1, flips=0)
            for _ in range(4)
        ],
        n_layers=0,
    )
    diff.tokens = md.summarise_tokens(
        [_attr({1: 0.01, 2: 0.90}, floor=0.05, tokens={1: " the", 2: " is"})],
        [_attr({1: 0.90, 2: 0.01}, floor=0.05, tokens={1: " the", 2: " is"})],
    )
    means = diff.means()
    assert "NEWLY depends on" in means
    assert "STOPPED depending on" in means
    assert "change in KIND rather than in degree" in means


def test_no_crossing_is_reported_as_a_result():
    diff = md.summarise(
        "base",
        "tuned",
        [
            md.PromptResult(prompt="p", n_tokens=4, mean_kl=0.1, max_kl=0.1, flips=0)
            for _ in range(4)
        ],
        n_layers=0,
    )
    diff.tokens = md.summarise_tokens(
        [_attr({1: 0.90}, floor=0.05)], [_attr({1: 0.95}, floor=0.05)]
    )
    assert "depend on the same words" in diff.means()
