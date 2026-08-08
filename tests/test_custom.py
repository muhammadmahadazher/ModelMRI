"""Custom-model support, checked against real tensors rather than mocks.

Every assertion here comes from a network actually built and actually run. A
dead-ReLU test that mocks the activation proves nothing about whether the
dead-unit count is real.
"""

from __future__ import annotations

import textwrap

import pytest

torch = pytest.importorskip("torch")

from modelmri import custom  # noqa: E402


# --------------------------------------------------------------- fixtures


ADAPTER = """
import torch
from torch import nn


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def load():
    return Tiny()


def example_input():
    return torch.zeros(2, 8)


LABELS = ["cat", "dog", "bird"]
"""


@pytest.fixture()
def adapter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "tiny_adapter.py"
    p.write_text(textwrap.dedent(ADAPTER), encoding="utf-8")
    return p


# ------------------------------------------------------------ path safety


def test_refuses_a_path_outside_the_allowed_roots(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "elsewhere.py"
    outside.write_text("def load(): pass", encoding="utf-8")
    monkeypatch.chdir(work)
    monkeypatch.delenv("MODELMRI_MODELS_DIR", raising=False)

    with pytest.raises(custom.AdapterError) as err:
        custom.resolve_under_roots(outside)
    assert "outside" in str(err.value)
    assert "MODELMRI_MODELS_DIR" in str(err.value)


def test_models_dir_extends_the_allowed_roots(tmp_path, monkeypatch):
    work, other = tmp_path / "work", tmp_path / "other"
    work.mkdir()
    other.mkdir()
    target = other / "a.py"
    target.write_text("def load(): pass", encoding="utf-8")
    monkeypatch.chdir(work)
    monkeypatch.setenv("MODELMRI_MODELS_DIR", str(other))

    assert custom.resolve_under_roots(target) == target.resolve()


def test_missing_file_says_so(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(custom.AdapterError, match="does not exist"):
        custom.resolve_under_roots(tmp_path / "nope.py")


# --------------------------------------------------------------- adapters


def test_loads_a_model_from_an_adapter(adapter):
    h = custom.CustomHandle()
    st = h.load(adapter)
    assert st.loaded and st.source == "adapter"
    assert st.name == "Tiny"
    # 8*16+16 + 16*3+3 = 144 + 51 = 195
    assert st.n_params == 195
    assert st.labels == ["cat", "dog", "bird"]
    assert st.input_shape == [2, 8]
    assert st.input_origin == "adapter"


def test_adapter_without_load_explains_the_protocol(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "bad.py"
    p.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(custom.AdapterError) as err:
        custom.CustomHandle().load(p)
    assert "no load() function" in str(err.value)


def test_adapter_returning_a_state_dict_is_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "sd.py"
    p.write_text("def load():\n    return {'w': 1}\n", encoding="utf-8")
    with pytest.raises(custom.AdapterError) as err:
        custom.CustomHandle().load(p)
    assert "not a torch.nn.Module" in str(err.value)


def test_an_adapter_that_raises_names_its_own_exception(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "boom.py"
    p.write_text(
        "def load():\n    raise FileNotFoundError('checkpoint.pt missing')\n",
        encoding="utf-8",
    )
    with pytest.raises(custom.AdapterError) as err:
        custom.CustomHandle().load(p)
    assert "FileNotFoundError" in str(err.value)
    assert "checkpoint.pt missing" in str(err.value)


# ------------------------------------------------------------ torchscript


def test_a_state_dict_checkpoint_is_refused_with_the_reason(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "weights.pth"
    torch.save({"fc.weight": torch.zeros(3, 3), "fc.bias": torch.zeros(3)}, p)

    with pytest.raises(custom.AdapterError) as err:
        custom.CustomHandle().load(p)
    msg = str(err.value)
    assert "state_dict" in msg
    assert "weights without an architecture" in msg
    assert "adapter" in msg


def test_torchscript_loads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scripted = torch.jit.script(torch.nn.Sequential(torch.nn.Linear(4, 2)))
    p = tmp_path / "m.pt"
    torch.jit.save(scripted, p)

    st = custom.CustomHandle().load(p)
    assert st.loaded and st.source == "torchscript"


# ------------------------------------------------------ the layer map itself


def test_the_layer_map_reports_every_leaf_in_execution_order(adapter):
    h = custom.CustomHandle()
    h.load(adapter)
    out = h.run()

    names = [row["name"] for row in out["layers"]]
    assert names == ["fc1", "act", "fc2"]
    assert [row["kind"] for row in out["layers"]] == ["Linear", "ReLU", "Linear"]
    assert out["layers"][0]["out_shape"] == [2, 16]
    assert out["layers"][2]["out_shape"] == [2, 3]
    assert out["output_shape"] == [2, 3]
    assert out["layers"][0]["n_params"] == 144
    assert out["layers"][1]["n_params"] == 0
    assert out["layers"][1]["is_activation"] is True


def test_a_dead_relu_is_counted(tmp_path, monkeypatch):
    """A layer whose bias guarantees negative pre-activations is 100% dead."""
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "dead.py"
    p.write_text(
        textwrap.dedent(
            """
            import torch
            from torch import nn

            def load():
                m = nn.Sequential(nn.Linear(4, 6), nn.ReLU())
                with torch.no_grad():
                    m[0].weight.zero_()
                    m[0].bias.fill_(-5.0)   # nothing can ever fire
                return m

            def example_input():
                return torch.randn(3, 4)
            """
        ),
        encoding="utf-8",
    )
    out = custom.CustomHandle().load(p) and None
    h = custom.CustomHandle()
    h.load(p)
    layers = h.run()["layers"]

    relu = next(r for r in layers if r["kind"] == "ReLU")
    assert relu["pct_zero"] == 100.0
    assert relu["max"] == 0.0
    assert out is None


def test_a_live_relu_is_not_reported_dead(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "live.py"
    p.write_text(
        textwrap.dedent(
            """
            import torch
            from torch import nn

            def load():
                m = nn.Sequential(nn.Linear(4, 64), nn.ReLU())
                with torch.no_grad():
                    m[0].weight.normal_(0, 1)
                    m[0].bias.zero_()
                return m

            def example_input():
                return torch.randn(32, 4)
            """
        ),
        encoding="utf-8",
    )
    h = custom.CustomHandle()
    h.load(p)
    relu = next(r for r in h.run()["layers"] if r["kind"] == "ReLU")
    # A symmetric input through a zero-bias layer kills about half.
    assert 20.0 < relu["pct_zero"] < 80.0
    assert relu["max"] > 0


def test_saturation_is_only_reported_for_bounded_activations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "sat.py"
    p.write_text(
        textwrap.dedent(
            """
            import torch
            from torch import nn

            def load():
                m = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 2), nn.ReLU())
                with torch.no_grad():
                    m[0].weight.normal_(0, 50)   # drive tanh to its rails
                    m[0].bias.zero_()
                return m

            def example_input():
                return torch.randn(16, 4)
            """
        ),
        encoding="utf-8",
    )
    h = custom.CustomHandle()
    h.load(p)
    layers = h.run()["layers"]

    tanh = next(r for r in layers if r["kind"] == "Tanh")
    relu = next(r for r in layers if r["kind"] == "ReLU")
    assert tanh["pct_saturated"] is not None and tanh["pct_saturated"] > 90
    # "Saturated" is meaningless for an unbounded activation; we don't invent it.
    assert relu["pct_saturated"] is None


def test_nonfinite_values_are_counted_and_do_not_poison_the_row(tmp_path, monkeypatch):
    """One nan must not turn mean/std/min/max into nan and hide the origin."""
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "nan.py"
    p.write_text(
        textwrap.dedent(
            """
            import torch
            from torch import nn

            class Poison(nn.Module):
                def forward(self, x):
                    y = x.clone()
                    y[0, 0] = float("nan")
                    y[0, 1] = float("inf")
                    return y

            def load():
                return nn.Sequential(nn.Linear(4, 8), Poison(), nn.ReLU())

            def example_input():
                return torch.randn(2, 4)
            """
        ),
        encoding="utf-8",
    )
    h = custom.CustomHandle()
    h.load(p)
    rows = h.run()["layers"]

    poison = next(r for r in rows if r["kind"] == "Poison")
    assert poison["n_nonfinite"] == 2
    assert poison["mean"] is not None
    assert poison["mean"] == poison["mean"]  # not nan
    linear = next(r for r in rows if r["kind"] == "Linear")
    assert linear["n_nonfinite"] == 0


# ------------------------------------------------------------- input shape


def test_the_suggested_shape_comes_from_the_first_linear():
    from torch import nn

    shape, why = custom.suggest_input(nn.Sequential(nn.Linear(11, 3), nn.ReLU()))
    assert shape == [1, 11]
    assert "11" in why


def test_the_suggested_shape_for_a_conv_says_which_part_is_a_guess():
    from torch import nn

    shape, why = custom.suggest_input(nn.Sequential(nn.Conv2d(3, 8, 3)))
    assert shape == [1, 3, 32, 32]
    assert "guess" in why


def test_a_model_with_nothing_to_infer_from_refuses_rather_than_guessing():
    from torch import nn

    shape, why = custom.suggest_input(nn.Sequential(nn.ReLU(), nn.Sigmoid()))
    assert shape is None
    assert "give the input shape yourself" in why


def test_running_without_any_input_refuses(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "noex.py"
    p.write_text(
        "from torch import nn\ndef load():\n    return nn.Sequential(nn.Linear(4, 2))\n",
        encoding="utf-8",
    )
    h = custom.CustomHandle()
    h.load(p)
    with pytest.raises(custom.AdapterError, match="no input to run"):
        h.run(shape=None)


def test_a_user_shape_overrides_the_adapter_example(adapter):
    h = custom.CustomHandle()
    h.load(adapter)
    out = h.run(shape=[5, 8])
    assert out["input_shape"] == [5, 8]
    assert out["layers"][0]["out_shape"] == [5, 16]
    assert h.status().input_origin == "user"


def test_a_wrong_shape_says_it_is_probably_the_shape(adapter):
    h = custom.CustomHandle()
    h.load(adapter)
    with pytest.raises(custom.AdapterError) as err:
        h.run(shape=[2, 999])
    assert "forward pass raised" in str(err.value)
    assert "input shape is wrong" in str(err.value)


def test_absurd_shapes_are_refused_before_allocating(adapter):
    h = custom.CustomHandle()
    h.load(adapter)
    with pytest.raises(custom.AdapterError, match="too large"):
        h.run(shape=[100000, 10000])
    with pytest.raises(custom.AdapterError, match="non-positive"):
        h.run(shape=[0, 8])


def test_an_embedding_model_gets_integer_input(tmp_path, monkeypatch):
    """Feeding an Embedding floats raises a type error that reads like our bug."""
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "emb.py"
    p.write_text(
        textwrap.dedent(
            """
            from torch import nn

            def load():
                return nn.Sequential(nn.Embedding(50, 8), nn.ReLU())
            """
        ),
        encoding="utf-8",
    )
    h = custom.CustomHandle()
    st = h.load(p)
    assert st.input_shape == [1, 16]
    rows = h.run(shape=st.input_shape)["layers"]
    assert rows[0]["out_shape"] == [1, 16, 8]


# ------------------------------------------------------------------- state


def test_training_mode_is_restored_after_inspection(adapter):
    h = custom.CustomHandle()
    h.load(adapter)
    h.model.train()
    h.run()
    assert h.model.training is True, "inspection left the model in eval mode"


def test_hooks_are_removed_even_when_the_forward_pass_fails(adapter):
    h = custom.CustomHandle()
    h.load(adapter)
    with pytest.raises(custom.AdapterError):
        h.run(shape=[2, 999])

    # Assert the invariant directly. Checking the next run's row count instead
    # looks equivalent and isn't: a leaked hook closes over the *previous*
    # rows list, so it appends out of sight and the next run still reports
    # three layers while the model accumulates a hook per failed attempt.
    leaked = [
        (name, len(mod._forward_hooks), len(mod._forward_pre_hooks))
        for name, mod in custom.leaf_modules(h.model)
        if mod._forward_hooks or mod._forward_pre_hooks
    ]
    assert leaked == [], f"hooks survived a failed forward pass: {leaked}"


def test_unload_clears_everything(adapter):
    h = custom.CustomHandle()
    h.load(adapter)
    h.run()
    st = h.unload()
    assert st.loaded is False and h.model is None and h.rows == []
    with pytest.raises(custom.AdapterError, match="no custom model is loaded"):
        h.run()


# --------------------------------------------------------------- discovery


def test_discovery_finds_adapters_without_importing_them(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MODELMRI_MODELS_DIR", raising=False)
    (tmp_path / "my_mri_adapter.py").write_text(
        "def load():\n    ...\ndef example_input():\n    ...\n", encoding="utf-8"
    )
    (tmp_path / "explodes.py").write_text(
        "def load():\n    ...\nraise SystemExit('imported!')\n", encoding="utf-8"
    )
    (tmp_path / "unrelated.py").write_text("print('hi')\n", encoding="utf-8")
    skipped = tmp_path / "node_modules"
    skipped.mkdir()
    (skipped / "dep.py").write_text("def load(): ...\n", encoding="utf-8")

    found = custom.find_adapters()
    names = [f["name"] for f in found]
    # It survived a file whose import would have killed the process.
    assert "my_mri_adapter.py" in names
    assert "explodes.py" in names
    assert "unrelated.py" not in names
    assert "dep.py" not in names
    # The one that advertises itself sorts first.
    assert names[0] == "my_mri_adapter.py"
    assert found[0]["has_example"] is True


def test_a_load_method_is_not_an_adapter(tmp_path, monkeypatch):
    """`def load(self, ...)` is a method on somebody's class, not an adapter.

    A plain substring search matched it, which offered ModelMRI's own saes.py
    and vla.py in the picker as models the user had trained.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MODELMRI_MODELS_DIR", raising=False)
    (tmp_path / "library.py").write_text(
        "class Runtime:\n    def load(self, hf_id):\n        ...\n", encoding="utf-8"
    )
    (tmp_path / "real_adapter.py").write_text(
        "def load():\n    ...\n", encoding="utf-8"
    )

    names = [f["name"] for f in custom.find_adapters()]
    assert names == ["real_adapter.py"]


def test_a_test_suite_is_not_scanned_for_adapters(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MODELMRI_MODELS_DIR", raising=False)
    suite = tmp_path / "tests"
    suite.mkdir()
    (suite / "helpers.py").write_text("def load():\n    ...\n", encoding="utf-8")
    (tmp_path / "test_thing.py").write_text("def load():\n    ...\n", encoding="utf-8")
    (tmp_path / "mine.py").write_text("def load():\n    ...\n", encoding="utf-8")

    assert [f["name"] for f in custom.find_adapters()] == ["mine.py"]


# ------------------------------------------------------- saturation, correctly


def test_saturation_is_measured_against_the_real_bound_not_the_data():
    """The threshold used the tensor's own max, which inverted the answer.

    9,000 sigmoid units pinned at 0 have gradient s(1-s) ~ 0 — textbook
    saturation — but `abs(0) >= 0.99 * span` is false, so they were counted as
    healthy while the 1,000 units sitting harmlessly at 0.5 were counted as
    saturated. Reported 10% when the truth was 90%.
    """
    pinned_low = torch.cat([torch.full((9000,), 1e-9), torch.full((1000,), 0.5)])
    got = custom.tensor_stats(pinned_low, "Sigmoid")["pct_saturated"]
    assert got == pytest.approx(90.0, abs=0.5), (
        f"a sigmoid pinned at its LOW rail read as {got}% saturated"
    )


def test_a_healthy_bounded_activation_is_not_reported_saturated():
    mid = torch.randn(5000) * 0.005 + 0.5
    assert custom.tensor_stats(mid, "Sigmoid")["pct_saturated"] == pytest.approx(0.0)
    spread = torch.tanh(torch.randn(5000) * 0.5)
    assert custom.tensor_stats(spread, "Tanh")["pct_saturated"] < 2.0


def test_both_rails_count_as_saturated():
    rails = torch.cat([torch.full((500,), -0.999), torch.full((500,), 0.999)])
    assert custom.tensor_stats(rails, "Tanh")["pct_saturated"] == pytest.approx(100.0)


def test_softmax_gets_no_saturation_figure_at_all():
    """A softmax is saturated when it is PEAKED, not when elements near a bound.

    Per-element saturation over a distribution is not a meaningful quantity,
    and reporting one made a maximum-entropy uniform softmax — the least
    saturated distribution there is — read as 100% saturated.
    """
    uniform = torch.full((1000,), 1 / 1000)
    assert custom.tensor_stats(uniform, "Softmax").get("pct_saturated") is None
