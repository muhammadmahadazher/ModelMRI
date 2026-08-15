"""Causal ablation for a network somebody trained themselves.

The models here have KNOWN answers — a branch wired to zero, a label that
depends on two features and eighteen decoys — so the sweep can be wrong in
ways a smoke test cannot see. Two of these tests exist because it was: both
controls in the first version were measurably the wrong null, in opposite
directions, and both were caught by a model whose answer was known in advance
rather than by reading the code.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from modelmri import custom_ablate as ca  # noqa: E402
from modelmri.errors import BadRequest  # noqa: E402

# --------------------------------------------------------------- the models


class OnlyFirstFive(nn.Module):
    """The output depends on input features 0..4 and on nothing else."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 3)
        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.weight[:, :5] = torch.randn(3, 5) * 2.0
            self.fc.bias.zero_()

    def forward(self, x):
        return self.fc(x)


class TwoBranch(nn.Module):
    """`useful` carries the signal. `useless` is multiplied by zero."""

    def __init__(self):
        super().__init__()
        self.useful = nn.Linear(20, 16)
        self.useless = nn.Linear(20, 16)
        self.head = nn.Linear(16, 3)
        with torch.no_grad():
            self.head.weight.mul_(6.0)

    def forward(self, x):
        return self.head(self.useful(x) + 0.0 * self.useless(x))


@pytest.fixture(scope="module")
def trained():
    """A net that has actually learned something.

    Every earlier model in this file is randomly initialised, and a random net
    is FRAGILE: any perturbation of any size scrambles it, so "a random edit
    did as much" is true and tells you nothing. A control has to be beatable
    by a real effect somewhere, or it is not a test.
    """
    torch.manual_seed(0)
    x = torch.randn(1024, 20)
    y = (x[:, 0] + x[:, 1] > 0).long() + (x[:, 0] - x[:, 1] > 0).long()
    net = nn.Sequential(
        nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3)
    )
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(400):
        opt.zero_grad()
        nn.functional.cross_entropy(net(x), y).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        assert float((net(x).argmax(dim=1) == y).float().mean()) > 0.95
    return net, x[:64]


@pytest.fixture(scope="module")
def samples():
    torch.manual_seed(0)
    return torch.randn(64, 20)


# ------------------------------------------------------- what it finds


def test_a_branch_wired_to_zero_scores_exactly_zero(samples):
    """The floor case. A module whose contribution is multiplied by zero must
    come back at 0.0 and not at "small"."""
    torch.manual_seed(0)
    out = ca.sweep_layers(TwoBranch().eval(), samples, task="classification")
    by_name = {s.name: s for s in out.sites}
    assert by_name["useless"].effect == 0.0
    assert by_name["useful"].effect > 0.1


def test_occlusion_finds_the_features_the_output_actually_depends_on(samples):
    """Ground truth: the output is wired to features 0..4 and to nothing
    else. The five strongest must be exactly those, and every other feature
    must be exactly zero rather than merely small."""
    torch.manual_seed(0)
    out = ca.sweep_inputs(OnlyFirstFive().eval(), samples, task="classification")
    top5 = sorted(int(s.name.split()[-1]) for s in out.sites[:5])
    assert top5 == [0, 1, 2, 3, 4]
    for site in out.sites[5:]:
        assert site.effect == 0.0, f"{site.name} is not wired in and scored above zero"


def test_the_real_features_beat_their_control_on_a_trained_net(trained):
    """The whole feature, on a model where the answer is known.

    MEASURED: features 0 and 1 come back at 4.7596 and 2.4738 nats against a
    control of 0.0081 — margins of 590x and 305x. That is what a finding looks
    like, and the first version of this control reported both as beats=False.
    """
    net, batch = trained
    out = ca.sweep_inputs(net, batch, task="classification", max_controlled=20)
    by_name = {s.name: s for s in out.sites}
    for feature in ("feature 0", "feature 1"):
        site = by_name[feature]
        assert site.beats_control is True, f"{feature} carries the label and lost"
        assert site.effect > 10 * (site.control_max or 0.0)
    # And they are the two strongest, not merely significant.
    assert {s.name for s in out.sites[:2]} == {"feature 0", "feature 1"}


def test_a_layer_can_beat_its_control_on_a_trained_net(trained):
    """A control that nothing can ever beat is not a test. MEASURED on this
    net: the second Linear scores 9.1588 against a control of 0.3928."""
    net, batch = trained
    out = ca.sweep_layers(net, batch, task="classification")
    assert any(s.beats_control for s in out.sites)


# ------------------------------------- the two controls, and why they differ


def test_the_layer_control_matches_the_size_of_the_edit_not_the_replacement():
    """The first version matched the NORM OF THE REPLACEMENT, copied from
    `patch.py`. That is apples to apples there and not here: the mean is the
    centre of the distribution and therefore the gentlest same-norm
    replacement, while a random vector of that norm is nearly orthogonal to
    the data.

    MEASURED on a 20->64->3 net over 64 samples: the mean edit moved the
    activation by 2.6735 and the same-norm random control moved it by 3.7624 —
    a 1.41x larger intervention than the thing it was the null for.
    """
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 3)).eval()
    batch = torch.randn(64, 20)
    mean = ca._mean_outputs(net, net[1], batch)

    captured: list = []
    handle = net[1].register_forward_hook(lambda m, i, o: captured.append(o.detach()))
    try:
        with torch.no_grad():
            net(batch)
    finally:
        handle.remove()
    actual = captured[0]

    treatment = torch.linalg.vector_norm(actual - mean.unsqueeze(0), dim=1)

    gen = torch.Generator().manual_seed(ca.CONTROL_SEED)
    captured.clear()
    handle = ca._jitter_like(net[1], mean, gen)
    probe = net[1].register_forward_hook(lambda m, i, o: captured.append(o.detach()))
    try:
        with torch.no_grad():
            net(batch)
    finally:
        handle.remove()
        probe.remove()
    control = torch.linalg.vector_norm(captured[-1] - actual, dim=1)

    assert float(control.mean()) == pytest.approx(float(treatment.mean()), rel=1e-4), (
        "the control must move the activation exactly as far as the mean does"
    )


def test_the_occlusion_control_is_a_different_region_not_a_random_direction(
    trained,
):
    """In one dimension a "random direction" is +1 or -1, so a jitter control
    performs the same edit as the treatment up to a sign and the comparison is
    a coin flip.

    MEASURED with that control on this net: features 0 and 1 — the two the
    label depends on — came back beats=False at 4.7596 and 2.4738 against
    controls of 4.8044 and 2.5052, while noise features 3, 6, 13 and 16 came
    back beats=True at 0.1278 and below. The null was labelling the finding as
    noise and the noise as findings.
    """
    net, batch = trained
    out = ca.sweep_inputs(net, batch, task="classification", max_controlled=20)
    tested = [s for s in out.sites if s.control_max is not None]
    assert tested
    # The give-away of the broken control: it landed within a hair of the
    # treatment on every row, because it WAS the treatment up to a sign.
    ties = [s for s in tested if abs(s.effect - (s.control_max or 0)) < 0.05 * s.effect]
    assert len(ties) < len(tested) / 2, (
        "most controls sit on top of their treatment, which is what the "
        "1-D jitter control did"
    )


def test_an_untested_site_reports_no_verdict_rather_than_a_failing_one(samples):
    """`beats_control` is None for a site nobody controlled. False would read
    as "a random edit did as much", which nothing measured."""
    torch.manual_seed(0)
    out = ca.sweep_layers(
        TwoBranch().eval(), samples, task="classification", max_controlled=1
    )
    untested = [s for s in out.sites if s.control_max is None]
    assert untested
    for site in untested:
        assert site.beats_control is None
        assert site.control_draws == 0


def test_the_multiple_comparison_is_stated_beside_the_count(trained):
    """Each site is compared against the strongest of `draws` draws, so under
    a null where every site is equivalent the real edit wins 1 time in
    (draws+1). Sweeping twenty sites clears 2.2 of them having done nothing."""
    net, batch = trained
    out = ca.sweep_inputs(net, batch, task="classification", max_controlled=20)
    assert out.expected_false_positives == pytest.approx(20 / 9, abs=0.01)
    assert "by chance" in out.means()


# ------------------------------------------------- what mean ablation cannot do


def test_cutting_a_sequential_chain_anywhere_gives_the_same_answer():
    """Not a bug, and worth knowing before reading a layer sweep: replacing
    any module's output with a constant makes every module after it constant
    too, so the final output is the same constant wherever the chain was cut.

    MEASURED: ablating `1` and ablating `2` of a 20->64->32->3 net produced
    final logits differing by 1.19e-07, with zero variance across samples.
    """
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 32), nn.Linear(32, 3)
    ).eval()
    batch = torch.randn(64, 20)

    def ablated(module):
        mean = ca._mean_outputs(net, module, batch)
        handle = ca._replace_with(module, mean)
        try:
            with torch.no_grad():
                return net(batch).detach()
        finally:
            handle.remove()

    a, b = ablated(net[1]), ablated(net[2])
    assert float((a - b).abs().max()) < 1e-5
    assert float(a.std(dim=0).max()) < 1e-6, "every sample gets the same constant"


# ------------------------------------------------------------- the contract


class _Adapter:
    pass


def test_an_adapter_that_does_not_declare_its_task_is_refused():
    """KL is right for a classifier and meaningless for a regressor, and both
    still produce a plausible ordering — which is exactly why a default would
    be dangerous rather than convenient."""
    with pytest.raises(BadRequest, match="does not say what kind of model"):
        ca.read_task(_Adapter())


def test_an_unknown_task_names_the_ones_it_knows():
    mod = _Adapter()
    mod.TASK = "segmentation"
    with pytest.raises(BadRequest, match="classification, regression"):
        ca.read_task(mod)


def test_a_declared_task_is_read_case_insensitively():
    mod = _Adapter()
    mod.TASK = "  Classification "
    assert ca.read_task(mod) == "classification"


def test_an_adapter_with_no_sample_inputs_is_refused_with_the_shape_to_add():
    with pytest.raises(BadRequest, match="no sample_inputs"):
        ca.read_samples(_Adapter())


def test_too_few_samples_is_refused_because_the_mean_would_be_the_sample():
    """Mean ablation over one sample replaces a layer with itself, so every
    score would be zero — a clean-looking result from a measurement that did
    not happen."""
    mod = _Adapter()
    mod.sample_inputs = lambda: torch.randn(3, 20)
    with pytest.raises(BadRequest, match="at least 8"):
        ca.read_samples(mod)


def test_a_list_of_samples_is_stacked_into_a_batch():
    mod = _Adapter()
    mod.sample_inputs = lambda: [torch.randn(20) for _ in range(8)]
    assert tuple(ca.read_samples(mod).shape) == (8, 20)


def test_sample_inputs_raising_is_reported_as_the_adapters_own_error():
    mod = _Adapter()

    def boom():
        raise ValueError("my dataset path is wrong")

    mod.sample_inputs = boom
    with pytest.raises(BadRequest, match="my dataset path is wrong"):
        ca.read_samples(mod)


# ------------------------------------------------------------- the refusals


def test_integer_inputs_are_refused_for_occlusion():
    """The mean of two token ids is not a token."""
    torch.manual_seed(0)
    net = nn.Sequential(nn.Embedding(50, 8), nn.Flatten(), nn.Linear(64, 3)).eval()
    ids = torch.randint(0, 50, (16, 8))
    with pytest.raises(BadRequest, match="ablate the embedding layer instead"):
        ca.sweep_inputs(net, ids, task="classification")


def test_a_model_that_answers_identically_for_every_sample_is_refused():
    """A regression shift is reported in units of the model's own spread, and
    a model with no spread would be division by nothing.

    `nn.Sequential` and not a bare `nn.Linear`: `leaf_modules` walks
    `named_modules()` and skips the root, so a one-module model has no NAMED
    leaves and refuses earlier for an unrelated reason. Same behaviour as
    `custom.inspect`, and worth knowing — but it is not what this test is
    about.
    """
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(20, 3)).eval()
    with torch.no_grad():
        net[0].weight.zero_()
        net[0].bias.zero_()
    with pytest.raises(BadRequest, match="no spread"):
        ca.sweep_layers(net, torch.randn(16, 20), task="regression")


def test_a_model_with_no_leaf_modules_is_refused():
    class Bare(nn.Module):
        def forward(self, x):
            return x * 2

    with pytest.raises(BadRequest, match="no leaf modules"):
        ca.sweep_layers(Bare(), torch.randn(16, 20), task="classification")


def test_a_capped_sweep_says_what_it_did_not_measure(samples):
    torch.manual_seed(0)
    out = ca.sweep_inputs(
        OnlyFirstFive().eval(), samples, task="classification", max_sites=6
    )
    assert out.truncated == 14
    assert "not swept" in out.means()
    assert "not measured as zero" in out.means()


# ----------------------------------------------------------------- the shape


def test_a_regression_score_is_in_units_of_the_models_own_spread(samples):
    """A raw L2 shift is in whatever units the model happens to emit — a model
    predicting house prices moves by thousands and one predicting
    probabilities by hundredths, and neither number means anything without
    knowing which."""
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(20, 16), nn.ReLU(), nn.Linear(16, 1)).eval()
    out = ca.sweep_layers(net, samples, task="regression")
    assert "output spread" in out.unit
    assert all(s.effect >= 0.0 for s in out.sites)


def test_the_report_survives_json(samples):
    torch.manual_seed(0)
    out = ca.sweep_layers(TwoBranch().eval(), samples, task="classification")
    doc = json.loads(json.dumps(out.to_dict(), allow_nan=False))
    assert doc["kind"] == "layers"
    assert "means" in doc
    assert doc["sites"][0]["name"]


def test_the_summary_always_says_mean_ablation_is_off_distribution(samples):
    """A large effect can mean the layer matters or that the model has never
    seen an input like the one the ablation just built, and the reader cannot
    tell those apart from the number."""
    torch.manual_seed(0)
    out = ca.sweep_layers(TwoBranch().eval(), samples, task="classification")
    assert "OFF-DISTRIBUTION" in out.means()


def test_the_model_is_left_in_the_mode_it_arrived_in(samples):
    torch.manual_seed(0)
    net = TwoBranch()
    net.train()
    ca.sweep_layers(net, samples, task="classification")
    assert net.training is True


def test_the_hints_carry_real_line_breaks_and_not_escapes():
    """These are code samples the reader copies, so the newlines have to reach
    the message.

    Written with escapes they went through two rounds of tooling and arrived
    first as a literal backslash-n printed on screen, then — after a careless
    fix — as a Python line continuation, which swallows the break entirely and
    produced a hint reading "beside load():    TASK = ..." on one line. Both
    failures are invisible in the source and obvious in the output.
    """
    # Built from chr() rather than written as escapes, for exactly the reason
    # this test exists: an escape in a source file is one tooling round-trip
    # away from being something else, and this assertion has to survive that.
    newline = chr(10)
    backslash_n = chr(92) + "n"
    for hint in (ca.TASK_HINT, ca.SAMPLES_HINT):
        assert backslash_n not in hint, "a literal backslash-n reached the message"
        assert hint.count(newline) >= 4, "the code sample lost its line breaks"
        # The sample must survive as something copy-pasteable.
        assert newline + newline in hint
    assert newline + "    TASK = " in ca.TASK_HINT
    assert newline + "    def sample_inputs():" in ca.SAMPLES_HINT


def test_both_refusals_reach_the_route_as_422_with_the_guidance(tmp_path):
    """The refusals ARE the feature: an adapter that does not declare TASK
    gets no metric picked for it, and one with no sample_inputs() gets no mean
    invented for it. Neither is worth much if the message stops at the API."""
    pytest.importorskip("fastapi")
    import textwrap

    from fastapi.testclient import TestClient

    from modelmri import custom
    from modelmri.server import create_app

    custom.add_root(str(tmp_path))
    (tmp_path / "no_task.py").write_text(
        textwrap.dedent(
            """
            import torch
            from torch import nn
            def load(): return nn.Sequential(nn.Linear(20, 3))
            def sample_inputs(): return torch.randn(16, 20)
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "no_samples.py").write_text(
        textwrap.dedent(
            """
            import torch
            from torch import nn
            TASK = "classification"
            def load(): return nn.Sequential(nn.Linear(20, 3))
            """
        ),
        encoding="utf-8",
    )

    client = TestClient(create_app())
    for name, wanted in (
        ("no_task", "does not say what kind of model"),
        ("no_samples", "nothing to take a mean over"),
    ):
        loaded = client.post(
            "/api/custom/load", json={"path": str(tmp_path / f"{name}.py")}
        )
        assert loaded.status_code == 200, loaded.text
        out = client.post("/api/custom/ablate", json={"kind": "layers"})
        assert out.status_code == 422
        assert wanted in out.json()["error"]


def test_a_torchscript_model_is_refused_with_the_reason_it_cannot_be_swept():
    """TorchScript carries weights and no way to say what the model is FOR or
    what its real inputs look like, and this measurement needs both."""
    from modelmri.custom import CustomHandle

    handle = CustomHandle()
    handle.model = nn.Sequential(nn.Linear(4, 2))
    handle.module = None
    handle.status_.source = "torchscript"
    with pytest.raises(BadRequest, match="needs the adapter that built"):
        handle.ablate("layers")
