# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Custom-model support, checked against real tensors rather than mocks.

Every assertion here comes from a network actually built and actually run. A
dead-ReLU test that mocks the activation proves nothing about whether the
dead-unit count is real.
"""

from __future__ import annotations

import math
import sys
import textwrap
import threading
import time

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
        "from torch import nn\n\n\ndef load():\n    return nn.Linear(4, 2)\n",
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


# --------------- a second request while the first one is still in flight


class _Gate:
    """A file handshake the adapter under test takes part in.

    Timing IS the test here, and a `sleep` is the wrong instrument for it: one
    long enough on a laptop is not long enough on a loaded CI box, and a test
    that only sometimes interleaves is a test that only sometimes catches the
    bug. The adapter announces that it is inside and then blocks until this
    side lets it through, so the ordering is exact.

    It blocks with a deadline, and this side waits with one: a gate nobody
    opens has to fail its test rather than hang the suite.
    """

    def __init__(self, folder, name: str) -> None:
        self.inside = folder / f"{name}.inside"
        self.go = folder / f"{name}.go"

    @property
    def source(self) -> str:
        """A `_gate()` the generated adapter calls from its top level."""
        return textwrap.dedent(
            f"""
            import os as _os
            import time as _time

            def _gate():
                open({str(self.inside)!r}, "w").close()
                _deadline = _time.monotonic() + 30
                while not _os.path.exists({str(self.go)!r}):
                    if _time.monotonic() > _deadline:
                        raise TimeoutError("the test never opened this gate")
                    _time.sleep(0.005)
            """
        )

    def arrived(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.inside.exists():
                return True
            time.sleep(0.005)
        return False

    def release(self) -> None:
        self.go.write_text("", encoding="utf-8")


def _joined(threads, timeout: float = 60.0) -> None:
    for t in threads:
        t.join(timeout=timeout)
    alive = [t.name for t in threads if t.is_alive()]
    assert not alive, f"threads never finished: {alive}"


def test_a_run_landing_after_an_unload_does_not_write_onto_the_empty_status(tmp_path):
    """`run()` releases the lock for the forward pass and re-takes it to store
    the result — and it stored onto `self.status_`, whatever that pointed at
    by then rather than the model the numbers actually describe.

    MEASURED 3/3 through the routes, with a 1.5s forward pass and
    `/api/custom/unload` 0.5s in, `GET /api/custom` answered:

        {"loaded": false, "path": null, "name": null, "input_shape": [3, 8],
         "input_origin": "user", "input_reason": "the shape you entered"}

    Nothing is loaded, and the departed model's input shape sits there with
    "the shape you entered" attached to it. `handle.rows` and `handle.meta`
    came back populated AFTER the unload had emptied them — the thing
    `test_unload_clears_everything` above asserts must not happen, which
    passed only because nothing was ever in flight while it ran.
    """
    gate = _Gate(tmp_path, "unload_midrun")
    (tmp_path / "gated_adapter.py").write_text(
        gate.source
        + textwrap.dedent(
            """
            from torch import nn


            class Gated(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(8, 3)

                def forward(self, x):
                    _gate()          # in flight, and holding still
                    return self.fc(x)


            def load():
                return Gated()
            """
        ),
        encoding="utf-8",
    )
    custom.add_root(str(tmp_path))

    h = custom.CustomHandle()
    h.load(tmp_path / "gated_adapter.py")
    out: dict = {}
    t = threading.Thread(target=lambda: out.update(run=h.run(shape=[3, 8])))
    t.start()
    try:
        assert gate.arrived(), "the forward pass never started"
        st = h.unload()
    finally:
        gate.release()
    _joined([t])

    assert st.loaded is False
    assert st.input_shape is None, "an unloaded model reported an input shape"
    assert st.input_origin == "" and st.input_reason == ""
    assert h.status().input_shape is None, "the run wrote onto the empty status"
    assert h.rows == [] and h.meta == {}, "the run repopulated an unloaded handle"

    # The caller still gets its own answer. The forward pass really did run,
    # and the return value is a statement about that run, not about what is
    # loaded now — dropping it would be inventing a second failure.
    assert out["run"]["input_shape"] == [3, 8]
    assert [r["name"] for r in out["run"]["layers"]] == ["fc"]


def test_a_run_landing_after_a_second_load_does_not_reshape_the_new_model(tmp_path):
    """The same write, aimed at a model that IS loaded — someone else's.

    MEASURED 3/3: a 1.5s forward at shape [3, 8], a different adapter loaded
    0.5s in, and the new model — whose own inferred input is [1, 4] — reported
    `input_shape: [3, 8]`, `input_origin: "user"`, "the shape you entered",
    for a shape nobody entered for it. Running at the shape the panel then
    showed came back 422: "mat1 and mat2 shapes cannot be multiplied (3x8 and
    4x2)", a refusal about arithmetic the reader never asked for.
    """
    gate = _Gate(tmp_path, "swap_midrun")
    (tmp_path / "gated_adapter.py").write_text(
        gate.source
        + textwrap.dedent(
            """
            from torch import nn


            class Gated(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(8, 3)

                def forward(self, x):
                    _gate()
                    return self.fc(x)


            def load():
                return Gated()
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "other_adapter.py").write_text(
        textwrap.dedent(
            """
            from torch import nn


            class Other(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(4, 2)

                def forward(self, x):
                    return self.fc(x)


            def load():
                return Other()
            """
        ),
        encoding="utf-8",
    )
    custom.add_root(str(tmp_path))

    h = custom.CustomHandle()
    h.load(tmp_path / "gated_adapter.py")
    t = threading.Thread(target=lambda: h.run(shape=[3, 8]))
    t.start()
    try:
        assert gate.arrived(), "the forward pass never started"
        swapped = h.load(tmp_path / "other_adapter.py")
    finally:
        gate.release()
    _joined([t])

    st = h.status()
    assert st.name == "Other"
    assert st.input_shape == swapped.input_shape == [1, 4]
    assert st.input_origin == "inferred", "a shape nobody entered, called 'user'"
    assert h.rows == [] and h.meta == {}, "the previous model's map survived a load"

    # And the shape the panel shows is one this model can actually take.
    assert h.run(shape=st.input_shape)["input_shape"] == [1, 4]


def test_one_adapter_import_does_not_break_a_concurrent_one(tmp_path):
    """`_import_adapter` puts the adapter's folder on `sys.path` and takes it
    off again, and the two halves used to race across threads.

    MEASURED through the routes, two adapters in ONE folder loaded together,
    each importing a different sibling module beside them:

        a_fast.py -> 200 in 0.31s
        b_slow.py -> 422 in 1.37s  "b_slow.py raised while being imported:
                                    ModuleNotFoundError: No module named
                                    'sib_alpha'"
        CONTROL  b alone           -> 200 in 1.27s
        CONTROL  different folders -> both 200

    The second load saw the entry already there, set `added = False`, and the
    FIRST load's `finally` removed the folder out from under it mid-import. So
    the refusal told the reader their file was broken — about a module sitting
    right beside it, which the two controls show is fine — and
    `_import_adapter` has no way to tell that apart from a genuinely broken
    adapter. `/api/custom/scan` lists adapters per folder, so two adapters in
    one directory is the ordinary case, and one browser tab is only safe
    because the panel disables its own button.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from modelmri.server import create_app

    (tmp_path / "sib_alpha.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "sib_beta.py").write_text("VALUE = 2\n", encoding="utf-8")
    first, second = _Gate(tmp_path, "first"), _Gate(tmp_path, "second")
    tail = "\nfrom torch import nn\n\n\ndef load():\n    return nn.Linear(4, 2)\n"
    (tmp_path / "a_fast.py").write_text(
        first.source + "\n_gate()\nimport sib_beta" + tail, encoding="utf-8"
    )
    (tmp_path / "b_slow.py").write_text(
        second.source + "\n_gate()\nimport sib_alpha" + tail, encoding="utf-8"
    )
    custom.add_root(str(tmp_path))

    client = TestClient(create_app())
    out: dict = {}

    def load(name: str) -> None:
        r = client.post("/api/custom/load", json={"path": str(tmp_path / name)})
        out[name] = (r.status_code, (r.json().get("error") or ""))

    threads = [
        threading.Thread(target=load, args=(n,)) for n in ("a_fast.py", "b_slow.py")
    ]
    try:
        threads[0].start()
        assert first.arrived(), "the first import never started"
        threads[1].start()
        # Bounded, and its outcome is deliberately NOT asserted, because the
        # two builds differ here and that is the whole point: unlocked, the
        # second import is already inside by now with `added = False`, which
        # is the state the bug needs; locked, it is parked at the lock and
        # cannot arrive until the first one leaves.
        second.arrived(timeout=2.0)
        # Let the first one all the way OUT before the second one moves, so
        # its `finally` has certainly run. Releasing both together left the
        # removal racing the second import and the bug went unobserved: this
        # test passed against the unfixed module until the join went in.
        first.release()
        _joined(threads[:1])
        second.release()
        _joined(threads)
    finally:
        first.release()
        second.release()
        for name in ("sib_alpha", "sib_beta"):
            sys.modules.pop(name, None)
        client.close()

    for name in ("a_fast.py", "b_slow.py"):
        code, err = out[name]
        assert code == 200, f"{name}: {code} {err}"


def test_only_one_adapter_folder_is_on_sys_path_at_a_time(tmp_path):
    """The quieter half of the same bug, and why this is a lock rather than a
    reference count on the path entry.

    While ANY adapter is importing, its folder sits at `sys.path[0]` for every
    import in the whole process — MEASURED: a module that raised
    ModuleNotFoundError before the load and again after it imported fine
    during it. That window cannot be closed; CPython resolves imports through
    a process-global `sys.path` and has no per-thread equivalent. What can be
    bounded is how much of it is open at once, and that is what separates the
    two candidate fixes: reference-counting repairs the removal race and
    leaves the rest, MEASURED at a peak of 3 of 3 user folders on `sys.path`
    during three concurrent loads, against 1 of 3 under this lock.
    """
    folders, gates = [], []
    for name in ("one", "two"):
        folder = tmp_path / name
        folder.mkdir()
        gate = _Gate(folder, name)
        (folder / "slow.py").write_text(
            gate.source + "\n_gate()\nfrom torch import nn\n\n\ndef load():\n"
            "    return nn.Linear(4, 2)\n",
            encoding="utf-8",
        )
        custom.add_root(str(folder))
        folders.append(folder)
        gates.append(gate)

    handles = [custom.CustomHandle(), custom.CustomHandle()]
    threads = [
        threading.Thread(target=h.load, args=(f / "slow.py",))
        for h, f in zip(handles, folders, strict=True)
    ]
    try:
        threads[0].start()
        assert gates[0].arrived(), "the first import never started"
        threads[1].start()
        # Bounded and unasserted for the same reason as the test above: the
        # second import is either inside or parked at the lock, and this is
        # the window in which both would be visible if it were inside.
        gates[1].arrived(timeout=1.0)
        # `_import_adapter` inserts `str(path.parent)` off an already-resolved
        # path, so comparing the same string is comparing what it wrote.
        on_path = [f for f in folders if str(f.resolve()) in sys.path]
    finally:
        for gate in gates:
            gate.release()
        _joined(threads)

    assert len(on_path) == 1, (
        f"{len(on_path)} adapter folders shadowed the process at once: {on_path}"
    )
    # And nothing is left behind once both are done.
    assert [f for f in folders if str(f.resolve()) in sys.path] == []


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


# ------------------ a top prediction that is actually a number


def test_a_partly_nonfinite_output_does_not_name_the_nan_as_the_answer():
    """The guard only fired when EVERY value was unusable, so a partly-NaN
    output fell through — and torch ranks NaN as the largest thing there is.

    MEASURED on [0.1, nan, 0.9, 0.3, inf]: `argmax` returns 1 and `topk`
    returns [nan, inf, 0.9]. So "your model's top prediction is class 1" named
    the slot with no number in it, on the panel whose entire job is telling a
    small-model author what their network answered — at 2am, when the loss is
    nan and they are trying to find out which layer did it.
    """
    from modelmri.custom import _summarise_output

    out = _summarise_output(torch.tensor([[0.1, float("nan"), 0.9, 0.3, float("inf")]]))

    assert out["argmax"] == 2, "the NaN slot was named as the prediction"
    assert out["top_index"] == [2, 3, 0]
    assert out["top_value"] == [0.9, 0.3, 0.1]
    # `math.isfinite`, not `v == v`: the NaN idiom reads as a tautology,
    # which is what CodeQL flags, and this covers inf in the same call.
    assert all(math.isfinite(v) for v in out["top_value"])


def test_the_count_of_unusable_outputs_travels_with_the_answer():
    """A network that is half NaN is a finding, not a reason to say nothing.
    The count rides beside the prediction rather than replacing it."""
    from modelmri.custom import _summarise_output

    out = _summarise_output(torch.tensor([[0.1, float("nan"), 0.9, float("nan")]]))
    assert out["n_nonfinite"] == 2
    assert out["n_out"] == 4
    assert out["argmax"] == 2


def test_a_healthy_output_reports_no_unusable_slots():
    """0 rather than absent, so the panel can state it instead of inferring
    it from a short list."""
    from modelmri.custom import _summarise_output

    out = _summarise_output(torch.tensor([[0.1, 0.5, 0.9]]))
    assert out["n_nonfinite"] == 0
    assert out["argmax"] == 2
    assert out["top_index"] == [2, 1, 0]


def test_an_entirely_nonfinite_output_still_says_so():
    """The existing behaviour, which was right and must stay."""
    from modelmri.custom import _summarise_output

    out = _summarise_output(torch.tensor([[float("nan"), float("-inf")]]))
    assert out["nonfinite"] is True
    assert out["n_nonfinite"] == 2
    assert "argmax" not in out


# ------------- a scan that answers about your files, not about their address


def _under(tmp_path, *parts):
    root = tmp_path.joinpath(*parts)
    root.mkdir(parents=True)
    return root


@pytest.mark.parametrize("ancestor", ["venv", "site-packages", "node_modules", ".git"])
def test_a_checkpoint_under_a_skipped_ancestor_is_still_found(tmp_path, ancestor):
    """`find_adapters` documents this fix in its own words — "`path.parts`
    includes every ancestor above the root, so a repo that happens to live
    under a directory named `build`, `dist`, `node_modules` or `venv` had
    EVERY candidate skipped and the scan silently found nothing" — and
    `find_torchscript` was left checking the absolute parts.

    So the checkpoint scanner returned an empty list for anyone whose models
    sit under such a path: an answer about where they keep their files,
    printed as an answer about what they have.
    """
    root = _under(tmp_path, ancestor, "my-models")
    (root / "epoch3.pt").write_bytes(b"x" * 100)

    custom.add_root(str(root))
    found = custom.find_torchscript()

    assert [f["name"] for f in found] == ["epoch3.pt"]


@pytest.mark.parametrize("ancestor", ["venv", "site-packages", "node_modules"])
def test_an_adapter_under_a_skipped_ancestor_is_still_found(tmp_path, ancestor):
    """The sibling, which was already correct. Both are pinned so the next
    fix cannot land in one and miss the other again."""
    root = _under(tmp_path, ancestor, "my-models")
    (root / "adapter.py").write_text("def load():\n    return None\n", encoding="utf-8")

    custom.add_root(str(root))
    found = custom.find_adapters()

    assert any(f["name"] == "adapter.py" for f in found)


def test_the_skip_list_still_applies_inside_the_scan_root(tmp_path):
    """Relative, not absent: a `__pycache__` BELOW the root is still skipped,
    which is what the list is for."""
    root = _under(tmp_path, "models")
    junk = root / "__pycache__"
    junk.mkdir()
    (junk / "cached.pt").write_bytes(b"x" * 100)
    (root / "real.pt").write_bytes(b"x" * 100)

    custom.add_root(str(root))
    names = [f["name"] for f in custom.find_torchscript()]

    assert "real.pt" in names
    assert "cached.pt" not in names


def test_the_candidate_walk_says_what_it_left_out(tmp_path, monkeypatch):
    """MEASURED: 45 adapter-shaped `.py` files and 45 `.pt` files in one root
    -> GET /api/custom/candidates returned 40 and 40, and the payload's keys
    were `['adapters', 'roots', 'torchscript']`. Nothing said anything was
    dropped, so five of the reader's own models were absent from the one panel
    that exists to list what they have.

    Both walks `return`ed at the limit, so like `scan_dir` before them they
    could not have reported a cap even if asked — they never learned what was
    past it. Counting is free; the expensive part is reading each file.
    """
    for i in range(45):
        (tmp_path / f"adapter_{i:02d}.py").write_text(
            "import torch\ndef load():\n    return torch.nn.Linear(4, 4)\n",
            encoding="utf-8",
        )
        (tmp_path / f"model_{i:02d}.pt").write_bytes(b"\0" * 32)

    monkeypatch.setattr(custom, "allowed_roots", lambda: [tmp_path])

    adapters = custom.find_adapters(tmp_path)
    assert len(adapters) == 40
    assert adapters.n_total == 45
    assert adapters.truncated is True

    scripts = custom.find_torchscript()
    assert len(scripts) == 40
    assert scripts.n_total == 45
    assert scripts.truncated is True

    # Still lists, so the CLI and the route keep indexing them unchanged.
    assert isinstance(adapters, list) and isinstance(scripts, list)


def test_a_walk_under_the_limit_is_not_reported_as_capped(tmp_path, monkeypatch):
    """The flag has to mean something: it must be false when nothing was
    dropped, or a panel that always warns is a panel nobody reads."""
    for i in range(3):
        (tmp_path / f"adapter_{i}.py").write_text(
            "import torch\ndef load():\n    return torch.nn.Linear(4, 4)\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(custom, "allowed_roots", lambda: [tmp_path])

    adapters = custom.find_adapters(tmp_path)
    assert len(adapters) == adapters.n_total == 3
    assert adapters.truncated is False


def test_two_candidate_walks_that_disagree_about_the_total_are_not_equal():
    """CodeQL flagged `Candidates` for adding `n_total` without overriding
    `__eq__`, and it is right: the class exists so the total travels, and
    inheriting `list.__eq__` made "40 of 40" compare equal to "40 of 45"."""
    complete = custom.Candidates([{"name": "a"}], n_total=1)
    capped = custom.Candidates([{"name": "a"}], n_total=45)
    assert complete != capped
    assert complete == custom.Candidates([{"name": "a"}], n_total=1)

    # A plain list carries no claim about the walk, so the rows decide.
    assert complete == [{"name": "a"}]


def test_the_total_counts_rows_not_paths(tmp_path, monkeypatch):
    """The total added to stop a fabricated count was itself fabricated.

    MEASURED with 45 adapter-shaped `.py` files and 45 `.pt` files in a folder
    that is BOTH the working directory and MODELMRI_MODELS_DIR — which is the
    ordinary case, because the allowed roots overlap by design:

        adapters 40 of 90, checkpoints 40 of 90

    Ninety, for forty-five files. The counter incremented on every path the
    glob yielded: before the skip-directory test, before the template and
    `test_` exclusions, before the `seen` dedupe that exists precisely because
    the roots overlap, and before the module-level `def load` test that
    decides whether a `.py` is a candidate at all.

    "40 of 90" is worse than the plain "40" it replaced — a specific,
    confident, wrong number that sends a reader looking for fifty models which
    do not exist. `n_total` has to mean "rows this walk WOULD have listed".
    """
    for i in range(45):
        (tmp_path / f"adapter_{i:02d}.py").write_text(
            "import torch\ndef load():\n    return torch.nn.Linear(4, 4)\n",
            encoding="utf-8",
        )
        (tmp_path / f"model_{i:02d}.pt").write_bytes(b"\0" * 32)

    # The same directory twice, which is what the real roots do.
    monkeypatch.setattr(custom, "allowed_roots", lambda: [tmp_path, tmp_path])

    adapters = custom.find_adapters(tmp_path)
    assert len(adapters) == 40
    assert adapters.n_total == 45, "the same file through two roots counted twice"

    scripts = custom.find_torchscript()
    assert len(scripts) == 40
    assert scripts.n_total == 45

    # And a non-candidate `.py` must not inflate it either.
    (tmp_path / "notes.py").write_text("# just a note\n", encoding="utf-8")
    (tmp_path / "test_thing.py").write_text(
        "import torch\ndef load():\n    return torch.nn.Linear(4, 4)\n",
        encoding="utf-8",
    )
    assert custom.find_adapters(tmp_path).n_total == 45
