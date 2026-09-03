# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Two robot checkpoints over identical frames, and what the comparison refuses.

rollout-doctor and TRI STEP both treat the policy as opaque, so neither can
say WHERE a finetune changed it. That is the whole value here, and it is also
what makes the output easy to over-read: a per-layer table invites being taken
as a verdict on which checkpoint is better, and there is no version of this
measurement that says which is better.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from modelmri import checkpoints as ck  # noqa: E402
from modelmri.errors import BadRequest  # noqa: E402

CONFIG = {
    "image_size": 224,
    "patch_size": 16,
    "num_hidden_layers": 6,
    "hidden_size": 16,
}


class Tower(nn.Module):
    """Six blocks. `drift_from` perturbs everything at or past that block."""

    def __init__(self, drift_from=None):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Linear(16, 16) for _ in range(6))
        self.drift_from = drift_from

    def forward(self, pixel_values, output_hidden_states=False):
        # `pixel_values` is a real [1, 3, S, S] image now, because the reader
        # hands back RGB and `vla.prepare_frame` normalises it — the same
        # transform the attention and occlusion paths use. This used to
        # reshape a flat [1, 64] vector, which only worked because the fake
        # reader implemented `frame_tensor`, a method no real reader has.
        n, c, h, w = pixel_values.shape
        flat = pixel_values.reshape(n, -1)
        # 4 patches x hidden 16, from whatever the image size is.
        x = flat[:, : 4 * 16].reshape(n, 4, 16)
        hidden = [x]
        for i, block in enumerate(self.blocks):
            x = x + torch.tanh(block(x))
            if self.drift_from is not None and i >= self.drift_from:
                x = x + 0.8 * torch.randn_like(x)
            hidden.append(x)
        return type("O", (), {"hidden_states": hidden})()


class Ep:
    def __init__(self, index, length):
        self.index, self.length = index, length


class FakeReader:
    repo_id = "lerobot/pusht"
    camera = "observation.images.top"

    def __init__(self, episodes=4, length=50):
        self._eps = [Ep(i, length) for i in range(episodes)]

    def episodes(self):
        return self._eps

    def raw_frame(self, episode, t):
        # `raw_frame`, because that is what the real readers offer. This
        # double used to implement `frame_tensor` -- a method that exists on
        # neither `LeRobotV3Reader` nor `Hdf5Reader` -- so the tests exercised
        # an API nobody had written and the feature was dead in every real
        # run while passing here.
        #
        # Deterministic per frame, so BOTH sides see identical input, which is
        # the entire premise of the comparison.
        import numpy as np

        gen = np.random.default_rng(episode * 1000 + t)
        return (gen.random((32, 32, 3)) * 255).astype("uint8")


def _loader(models, configs=None, cameras=None):
    def load_side(spec):
        return (
            models[spec].eval(),
            "cpu",
            (configs or {}).get(spec, CONFIG),
            (cameras or {}).get(spec, ["observation.images.top"]),
            lambda: None,
        )

    return load_side


@pytest.fixture
def pair():
    torch.manual_seed(1)
    a = Tower()
    b = Tower(drift_from=3)
    b.load_state_dict(a.state_dict())
    twin = Tower()
    twin.load_state_dict(a.state_dict())
    return a, b, twin


# ------------------------------------------------------------ what it finds


def test_the_drift_is_found_where_it_was_planted(pair):
    """MEASURED: drift planted at block 3 shows first at hidden index 4 —
    `hidden_states[i + 1]` is the output of block `i`."""
    a, b, _ = pair
    out = ck.compare(
        _loader({"base": a, "tuned": b}),
        "base",
        "tuned",
        FakeReader(),
        frame_stride=25,
    )
    assert out.first_divergent.layer == 4
    for row in out.layers[:4]:
        assert row.cka == pytest.approx(1.0, abs=1e-4)


def test_the_first_divergence_is_not_the_lowest_alignment(pair):
    """Once two towers come apart they stay apart and the gap compounds, so
    the lowest CKA is almost always the last layer and almost never the
    answer somebody is asking for."""
    a, b, _ = pair
    out = ck.compare(
        _loader({"base": a, "tuned": b}),
        "base",
        "tuned",
        FakeReader(),
        frame_stride=25,
    )
    assert out.most_divergent.layer > out.first_divergent.layer
    assert "almost never the answer" in out.means()


def test_identical_checkpoints_never_come_apart(pair):
    a, _, twin = pair
    out = ck.compare(
        _loader({"base": a, "copy": twin}),
        "base",
        "copy",
        FakeReader(),
        frame_stride=25,
    )
    assert all(r.cka == pytest.approx(1.0, abs=1e-6) for r in out.layers)
    assert out.first_divergent is None
    assert "never falls between layers" in out.means()


def test_both_sides_see_identical_frames(pair):
    """The premise of the whole comparison. If the two runs saw different
    frames, every distance would be about the frames."""
    a, b, _ = pair
    reader = FakeReader()
    seen: list = []

    original = reader.raw_frame
    reader.raw_frame = lambda e, t: seen.append((e, t)) or original(e, t)
    ck.compare(
        _loader({"base": a, "tuned": b}), "base", "tuned", reader, frame_stride=25
    )
    half = len(seen) // 2
    assert seen[:half] == seen[half:], "the two sides were shown different frames"


# ---------------------------------------------------- the compatibility gate


@pytest.mark.parametrize(
    "field,value",
    [
        ("image_size", 384),
        ("patch_size", 14),
        ("num_hidden_layers", 12),
        ("hidden_size", 32),
    ],
)
def test_a_mismatched_field_is_refused_by_name(field, value):
    """ "The checkpoints are incompatible" sends the reader to diff two configs
    by hand. Naming the field and both values does not."""
    with pytest.raises(BadRequest, match=f"`{field}`"):
        ck.check_compatible(CONFIG, {**CONFIG, field: value}, "base", "tuned")


def test_the_refusal_carries_both_values():
    with pytest.raises(BadRequest) as caught:
        ck.check_compatible(CONFIG, {**CONFIG, "image_size": 384}, "base", "tuned")
    message = str(caught.value)
    assert "224 in base" in message and "384 in tuned" in message
    assert "resized differently" in message


def test_a_missing_field_is_refused_rather_than_assumed():
    missing = {k: v for k, v in CONFIG.items() if k != "hidden_size"}
    with pytest.raises(BadRequest, match="does not state its `hidden_size`"):
        ck.check_compatible(CONFIG, missing, "base", "tuned")


def test_different_cameras_read_as_a_different_question_not_a_config_bug():
    """Two policies trained on different camera sets were never asked the same
    question, which is a different failure from a configuration mismatch."""
    with pytest.raises(BadRequest, match="never asked the same question"):
        ck.check_cameras(["top"], ["top", "wrist"], "base", "tuned")


def test_matching_cameras_in_any_order_pass():
    ck.check_cameras(["top", "wrist"], ["wrist", "top"], "base", "tuned")


def test_the_gate_fires_before_the_second_side_runs_its_frames(pair):
    """A mismatched pair should cost one load, not two full sweeps."""
    a, b, _ = pair
    frames_run = {"n": 0}
    reader = FakeReader()
    original = reader.raw_frame

    def counting(e, t):
        frames_run["n"] += 1
        return original(e, t)

    reader.raw_frame = counting
    with pytest.raises(BadRequest, match="`hidden_size`"):
        ck.compare(
            _loader(
                {"base": a, "tuned": b},
                configs={"base": CONFIG, "tuned": {**CONFIG, "hidden_size": 32}},
            ),
            "base",
            "tuned",
            reader,
            frame_stride=25,
        )
    # The first side ran its frames; the second must not have.
    assert frames_run["n"] == 8


def test_the_same_checkpoint_on_both_sides_is_refused(pair):
    a, _, _ = pair
    with pytest.raises(BadRequest, match="zero by construction"):
        ck.compare(_loader({"base": a}), "base", "base", FakeReader())


# ------------------------------------------------------------ the distances


def test_cka_is_invariant_to_rotation():
    """Two towers can encode the same thing in different bases, and a raw
    distance would call that a difference."""
    torch.manual_seed(0)
    x = torch.randn(20, 16)
    q, _ = torch.linalg.qr(torch.randn(16, 16))
    assert ck.cka(x, x @ q) == pytest.approx(1.0, abs=1e-6)


def test_cka_is_invariant_to_scale():
    torch.manual_seed(0)
    x = torch.randn(20, 16)
    assert ck.cka(x, x * 37.0) == pytest.approx(1.0, abs=1e-6)


def test_cka_works_across_different_widths():
    """Which is why this and not a plain correlation: `d` need not match."""
    torch.manual_seed(0)
    x = torch.randn(20, 16)
    y = torch.randn(20, 32)
    value = ck.cka(x, y)
    assert 0.0 <= value <= 1.0


def test_cosine_is_undefined_across_different_widths_rather_than_zero():
    """Not an error and not zero: CKA handles different widths and this cannot,
    so the caller gets a signal that this half is simply not defined here."""
    import math as _math

    torch.manual_seed(0)
    value = ck.mean_cosine(torch.randn(8, 16), torch.randn(8, 32))
    assert _math.isnan(value)


def test_a_constant_tower_is_refused_rather_than_scored_zero():
    """Not a similarity of zero — there is no direction to align with."""
    with pytest.raises(BadRequest, match="nothing for the other to be aligned"):
        ck.cka(torch.ones(8, 16), torch.randn(8, 16))


def test_one_frame_is_refused_because_centring_leaves_nothing():
    with pytest.raises(BadRequest, match="centring leaves nothing"):
        ck.cka(torch.randn(1, 16), torch.randn(1, 16))


def test_mismatched_frame_counts_are_refused():
    with pytest.raises(BadRequest, match="same frames in the same order"):
        ck.cka(torch.randn(8, 16), torch.randn(7, 16))


# --------------------------------------------------------------- the plan


def test_too_many_frames_is_refused_rather_than_trimmed():
    """A comparison over a silently trimmed frame set is not the comparison
    you asked for."""
    with pytest.raises(BadRequest, match="raise the stride"):
        ck.plan(FakeReader(episodes=100, length=500), frame_stride=1, max_frames=100)


def test_the_refusal_says_it_costs_a_pass_on_both_sides():
    with pytest.raises(BadRequest) as caught:
        ck.plan(FakeReader(episodes=100, length=500), frame_stride=1, max_frames=100)
    assert "BOTH sides" in str(caught.value)


def test_a_zero_stride_is_refused():
    with pytest.raises(BadRequest, match="at least 1"):
        ck.plan(FakeReader(), frame_stride=0)


def test_an_empty_dataset_is_refused():
    class Empty(FakeReader):
        def episodes(self):
            return []

    with pytest.raises(BadRequest, match="no episodes"):
        ck.plan(Empty())


# -------------------------------------------------- what it must not imply


def test_it_never_implies_a_winner(pair):
    """A checkpoint that diverges more might be the one that learned the
    task."""
    a, b, _ = pair
    out = ck.compare(
        _loader({"base": a, "tuned": b}),
        "base",
        "tuned",
        FakeReader(),
        frame_stride=25,
    )
    means = out.means()
    assert "NOT WHICH IS BETTER" in means
    for verdict in ("better checkpoint", "worse", "improved", "degraded", "winner"):
        assert verdict not in means.lower()


def test_the_absent_behaviour_half_is_named(pair):
    """Predicted-action distance needs the action expert. The report says
    which half ran rather than leaving a reader to assume both did."""
    a, b, _ = pair
    out = ck.compare(
        _loader({"base": a, "tuned": b}),
        "base",
        "tuned",
        FakeReader(),
        frame_stride=25,
    )
    assert out.ran_perception is True
    assert out.ran_behaviour is False
    assert "THE BEHAVIOUR HALF DID NOT" in out.means()
    assert "says nothing about what either would DO" in out.means()


def test_an_absent_reason_does_not_leave_a_dangling_colon():
    """A colon with nothing after it reads as a truncated message rather than
    as a half that deliberately did not run."""
    bare = ck.Comparison(checkpoint_a="a", checkpoint_b="b", dataset="d", camera="c")
    assert "DID NOT. So this compares" in bare.means()
    assert ":  " not in bare.means()


def test_each_side_is_released_even_when_a_capture_raises(pair):
    """A capture that raises must still give the memory back, or the second
    side has nowhere to load into and the real error is buried under an
    out-of-memory."""
    a, _, _ = pair
    released: list = []

    class Exploding(nn.Module):
        def forward(self, *args, **kwargs):
            raise RuntimeError("boom")

    models = {"base": a, "tuned": Exploding()}

    def loader(spec):
        return (
            models[spec].eval(),
            "cpu",
            CONFIG,
            ["observation.images.top"],
            lambda: released.append(spec),
        )

    with pytest.raises(RuntimeError, match="boom"):
        ck.compare(loader, "base", "tuned", FakeReader(), frame_stride=25)
    assert "tuned" in released


def test_the_report_survives_json(pair):
    a, b, _ = pair
    out = ck.compare(
        _loader({"base": a, "tuned": b}),
        "base",
        "tuned",
        FakeReader(),
        frame_stride=25,
    )
    doc = json.loads(json.dumps(out.to_dict(), allow_nan=False))
    assert doc["checkpoint_a"] == "base"
    assert "means" in doc
    assert len(doc["layers"]) == 7


# ------------------ a comparison that actually reaches a frame


class _Tower(torch.nn.Module):
    def __init__(self, drift: float = 0.0):
        super().__init__()
        self.drift = drift

    def forward(self, pixel_values, output_hidden_states=False):
        n = pixel_values.shape[0]
        base = pixel_values.mean(dim=(1, 2, 3)).reshape(n, 1, 1)
        return type(
            "O",
            (),
            {
                "hidden_states": [
                    (base + i * 0.1 + (self.drift if i >= 3 else 0.0)).expand(n, 4, 8)
                    for i in range(6)
                ]
            },
        )()


class _Reader:
    repo_id = "ds"
    camera = "cam"

    def episodes(self):
        return [type("E", (), {"index": i, "length": 50})() for i in range(2)]

    def raw_frame(self, episode, t):
        import numpy as np

        return (np.ones((32, 32, 3)) * ((episode * 50 + t) % 255)).astype("uint8")


_CFG = {"image_size": 16, "patch_size": 8, "num_hidden_layers": 6, "hidden_size": 8}


def _load_side(spec):
    return _Tower(0.0 if spec == "a" else 0.4), "cpu", _CFG, ["cam"], (lambda: None)


def test_a_comparison_reaches_the_frames_at_all():
    """`reader.frame_tensor(...)` does not exist — not on `LeRobotV3Reader`,
    not on `Hdf5Reader`, not anywhere in this package.

    It raised AttributeError on the FIRST frame, so the whole feature was
    dead: two checkpoints loaded, both released, nothing measured. A method
    name nobody implements is the same defect class as the SDK-shape drift
    `modelmri-record` exists to catch, sitting in this repo.
    """
    out = ck.compare(_load_side, "a", "b", _Reader(), frame_stride=25)

    assert out.n_frames > 0, "no frame was ever read"
    assert len(out.layers) == 6
    assert out.ran_perception is True


def test_the_frames_go_through_the_one_shared_normalisation():
    """Two normalisations would put the two towers' embeddings on different
    scales while the report claims they saw identical frames. `vla.prepare_frame`
    is the transform the attention and occlusion paths already use."""
    import inspect

    from modelmri import vla

    source = inspect.getsource(ck.compare)
    code = "\n".join(
        line.split("#", 1)[0] for line in source.splitlines() if line.strip()
    )
    assert "vla.prepare_frame" in code
    assert "frame_tensor" not in code
    assert callable(vla.prepare_frame)


def test_every_reader_method_the_comparison_calls_exists_on_both_readers():
    """The structural version. `frame_tensor` was called for as long as it took
    somebody to run the feature, because nothing checked the readers offer what
    this asks them for."""
    import inspect
    import re

    from modelmri import hdf5_data, vla_data

    called = set(re.findall(r"\breader\.([a-z_]+)\s*\(", inspect.getsource(ck)))
    assert called, "the pattern found no reader calls and would pass vacuously"

    for reader in (vla_data.LeRobotV3Reader, hdf5_data.Hdf5Reader):
        for name in sorted(called):
            assert hasattr(reader, name), (
                f"{reader.__name__} has no {name}(), which checkpoints.py calls"
            )
