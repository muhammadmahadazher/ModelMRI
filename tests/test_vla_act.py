# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""ACT in the perception half: a 500 that became a refusal, and a ResNet opened
without anybody pretending it has attention.

TWO THINGS ARE UNDER TEST HERE AND THEY ARE NOT THE SAME SIZE.

The first is a bug. `_vision_config` ended in a bare
`AutoConfig.from_pretrained(policy_snap)` with no except arm, and a draccus
lerobot config.json has `type: "act"` where transformers looks for
`model_type`. MEASURED against a synthetic ACT snapshot before the fix:

    ValueError: Unrecognized model in <hub>\\models--lerobot--act_fake\\
    snapshots\\deadbeef. Should have a `model_type` key in its config.json.

`except BadRequest` in server.py does not catch a plain ValueError, so POST
/api/vla/load answered HTTP 500 "Something inside ModelMRI failed rather than
refusing" — for a checkpoint that is perfectly well-formed and that the policy
sidecar can already run. `discover_vision_prefix` had a good 409 with an
actionable sentence one frame away, unreachable because the config lookup runs
first.

The second is the feature: ACT sees through a torchvision ResNet under
`model.backbone.`, and this opens it. The tests that matter most are not the
ones proving it loads — they are the ones proving it does NOT grow an
attention map on the way. A convolutional activation is exactly the right
shape to be reshaped into [heads, G, G] and painted into the panel where every
other attention map goes, and it would be read as attention by everyone who
saw it.

WHAT THESE FIXTURES ARE, HONESTLY. A real `lerobot/act_*` checkpoint is not
downloaded here and none is on this machine. The snapshots below are built
from torchvision's own `resnet18` — the same public builder lerobot calls —
with the classifier head dropped and every tensor renamed under
`model.backbone.`, beside a draccus-shaped config.json. That exercises the
config arm, the norm-layer decision, the tensor load, the measured grid, every
refusal, and a real occlusion sweep. It does NOT prove that a published ACT
checkpoint spells its keys exactly this way; only a real checkpoint settles
that, and the report accompanying this file says so.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

from safetensors.torch import save_file  # noqa: E402

from modelmri import vla, vla_occlude  # noqa: E402
from modelmri.errors import BadRequest, Refusal  # noqa: E402

# Small enough that a resnet18 forward pass is milliseconds, large enough that
# the stack's total stride still leaves rows in the feature map. Every test
# that loads weights passes this explicitly, so no test is measuring the
# 480x480 the config declares.
TEST_EDGE = 64


def _act_config(**over) -> dict:
    """A draccus-shaped ACT config: `type`, never `model_type`."""
    cfg = {
        "type": "act",
        "n_obs_steps": 1,
        "chunk_size": 100,
        "vision_backbone": "resnet18",
        "pretrained_backbone_weights": "ResNet18_Weights.IMAGENET1K_V1",
        "replace_final_stride_with_dilation": False,
        "dim_model": 512,
        "n_heads": 8,
        "input_features": {
            "observation.images.top": {"type": "VISUAL", "shape": [3, 480, 640]},
            "observation.state": {"type": "STATE", "shape": [14]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [14]}},
    }
    cfg.update(over)
    return {k: v for k, v in cfg.items() if v is not None}


def _backbone_state(frozen: bool = True) -> dict:
    """A resnet18 state dict shaped the way lerobot's ACT holds one.

    `FrozenBatchNorm2d` by default because that is what lerobot builds ACT
    with, and its buffers are the reason the norm layer has to be decided from
    the keys rather than assumed — see `_build_conv_tower`.

    Random values rather than the builder's initialisation: a backbone whose
    weights are all one thing returns one embedding for every frame, and
    `vla_occlude.scale_from` would correctly refuse the sweep for having no
    spread to measure against. The randomness is seeded, so a shift reported
    below is the same number on the next run.
    """
    kwargs: dict = {"weights": None}
    if frozen:
        from torchvision.ops.misc import FrozenBatchNorm2d

        kwargs["norm_layer"] = FrozenBatchNorm2d
    torch.manual_seed(11)
    net = torchvision.models.resnet18(**kwargs)
    state = {}
    for key, value in net.state_dict().items():
        if key.startswith("fc."):
            # lerobot wraps the backbone in an IntermediateLayerGetter that
            # stops at layer4, so a real checkpoint carries no classifier.
            continue
        if not value.dtype.is_floating_point:
            state[key] = value.clone().contiguous()
        elif key.endswith("running_var"):
            # A variance is positive, and a negative one makes the batch-norm
            # output NaN — which would be a fixture bug dressed as a finding.
            state[key] = (torch.randn_like(value) * 0.1).abs().add(1.0).contiguous()
        else:
            state[key] = (torch.randn_like(value) * 0.1).contiguous()
    return state


def _snapshot(root: Path, name: str, config: dict, tensors: dict) -> Path:
    """One cached repo on disk, laid out the way `vla._snapshot` looks for it."""
    snap = root / "hub" / f"models--lerobot--{name}" / "snapshots" / "cafe0001"
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "config.json").write_text(json.dumps(config), "utf-8")
    save_file(tensors, str(snap / "model.safetensors"))
    return snap


def _act_repo(root: Path, name: str = "act_synth", *, frozen: bool = True, **over):
    """A cached ACT-shaped policy, returning (repo_id, its backbone tensors)."""
    body = _backbone_state(frozen)
    tensors = {f"model.backbone.{k}": v for k, v in body.items()}
    # The rest of the policy, so the prefix discovery has something to lose to.
    tensors["model.encoder.layers.0.linear1.weight"] = torch.zeros(8, 8)
    tensors["model.action_head.weight"] = torch.zeros(14, 512)
    tensors["normalize_inputs.buffer_observation_state.mean"] = torch.zeros(14)
    _snapshot(root, name, _act_config(**over), tensors)
    return f"lerobot/{name}", body


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    """One ACT policy, loaded once, shared by every test that only reads it.

    Module-scoped because building and loading a resnet18 is the slow part of
    this file and none of the readers mutate the handle — `analyse` refuses
    before it caches anything, which is the property under test.
    """
    root = tmp_path_factory.mktemp("act_hub")
    repo, body = _act_repo(root)
    handle = vla.VLAHandle()
    status = handle.load(repo, root, image_size=TEST_EDGE)
    return handle, status, body, repo


# ------------------------------------------------- the bug: a 500 became a 409


def test_a_draccus_config_refuses_instead_of_reaching_the_500_arm(tmp_path):
    """THE REGRESSION. A lerobot config.json names its policy in `type` and has
    no `model_type` at all, so `AutoConfig.from_pretrained` raised a bare
    ValueError that server.py's `except Exception` turned into HTTP 500
    "Something inside ModelMRI failed rather than refusing".

    MEASURED before the fix, driving `VLAHandle.load` the way /api/vla/load
    drives it: `ValueError: Unrecognized model in <snapshot path>. Should have
    a `model_type` key in its config.json.` — a stack-trace sentence, about a
    checkpoint that is fine, carrying the snapshot path on this machine.

    `Refusal` is a RuntimeError, so this also pins that no ValueError escapes:
    a ValueError here would reach `except BadRequest` at 422 or the 500 arm,
    and both would be lies about whose fault it is.
    """
    # A draccus policy that names no encoder anywhere — the case that has to
    # end in words rather than a traceback.
    _snapshot(
        tmp_path,
        "diffusion_synth",
        {"type": "diffusion", "horizon": 16, "n_obs_steps": 2},
        {"diffusion.unet.down.0.weight": torch.zeros(2, 2)},
    )
    with pytest.raises(Refusal) as caught:
        vla.VLAHandle().load("lerobot/diffusion_synth", tmp_path)

    err = caught.value
    assert not isinstance(err, ValueError), "a ValueError here is a 422 or a 500"
    assert err.sentence, "a Refusal without a sentence is a 409 with nothing in it"
    # The sentence has to name what it found and a next step. The sidecar is
    # the next step that actually exists for this checkpoint: `modelmri policy
    # start` loads it through lerobot's own registry and does not care that
    # transformers cannot read the config.
    assert "modelmri policy start" in err.sentence
    assert "diffusion" in err.sentence
    assert "horizon" in err.sentence and "n_obs_steps" in err.sentence
    # A published 409 does not put a path from this machine in front of the
    # reader — which is exactly what the ValueError's own text did.
    assert str(tmp_path) not in err.sentence


def test_the_refusal_reports_the_true_key_count_when_it_truncates(tmp_path):
    """EVERY CAP IS REPORTED. The reader is scanning that list for the key they
    expected to be honoured, so a list silently cut at twelve is the one thing
    this sentence must not be."""
    wide = {"type": "act", **{f"z_key_{i:02d}": i for i in range(30)}}
    _snapshot(tmp_path, "wide_synth", wide, {"body.0.weight": torch.zeros(2, 2)})
    with pytest.raises(Refusal) as caught:
        vla._vision_config(
            tmp_path / "hub" / "models--lerobot--wide_synth" / "snapshots" / "cafe0001",
            tmp_path,
        )
    assert f"{vla.CONFIG_KEYS_SHOWN} of {len(wide)} keys" in caught.value.sentence


def test_a_config_transformers_can_read_but_has_no_vision_in_it(tmp_path):
    """The other arm into the same sentence, and it must not claim the config
    has no `model_type` when it plainly does — a reader who went and added one
    would be sent in a circle."""
    _snapshot(
        tmp_path,
        "text_only",
        {"model_type": "gpt2", "n_layer": 2, "n_head": 2, "n_embd": 8},
        {"h.0.attn.c_attn.weight": torch.zeros(2, 2)},
    )
    snap = tmp_path / "hub" / "models--lerobot--text_only" / "snapshots" / "cafe0001"
    with pytest.raises(Refusal) as caught:
        vla._vision_config(snap, tmp_path)
    assert "`gpt2` config" in caught.value.sentence
    assert "no `model_type`" not in caught.value.sentence


# ------------------------------------------------------- finding the backbone


def test_backbone_is_the_last_hint_so_a_named_tower_still_wins():
    """ORDER IS LOAD-BEARING in VISION_HINTS. The scan takes the first HINT
    that matches a key, not the leftmost match inside it, so "backbone" ahead
    of "visual" would collapse a Qwen-VL-style `backbone.visual.` checkpoint
    to the prefix `backbone.` — which also covers its language model. Measured
    both ways on this key list: 4 tensors either way, but the prefix flips."""
    assert vla.VISION_HINTS[-1] == "backbone"
    keys = [f"backbone.visual.encoder.layers.{i}.attn.q_proj.weight" for i in range(4)]
    keys += ["backbone.text.layers.0.mlp.weight", "action_head.weight"]
    prefix, found = vla.discover_vision_prefix(keys)
    assert (prefix, found) == ("backbone.visual.", 4)


def test_an_act_layout_is_found_by_the_new_hint():
    """Nothing in an ACT checkpoint contains the word "vision" in a tensor
    path, which is why it found zero tensors before."""
    keys = [f"model.backbone.layer{i}.0.conv1.weight" for i in range(1, 5)]
    keys += ["model.backbone.conv1.weight", "model.encoder.layers.0.linear1.weight"]
    prefix, found = vla.discover_vision_prefix(keys)
    assert (prefix, found) == ("model.backbone.", 5)


# ------------------------------------------------------------ what it loaded


def test_the_tower_holds_the_checkpoints_own_tensors(loaded):
    """Not "it loaded" — the same numbers. A backbone built and then left at
    its own random initialisation would pass every structural check in this
    file and report occlusion scores from weights no robot ever used."""
    handle, _, body, _ = loaded
    got = handle.model.body.state_dict()
    assert set(got) == set(body), "the module and the checkpoint disagree on keys"
    assert all(torch.equal(got[k].cpu(), body[k]) for k in body)
    assert len(body) == 100  # resnet18 with FrozenBatchNorm2d, classifier dropped


@pytest.mark.parametrize("frozen", [True, False])
def test_the_norm_layer_is_read_off_the_keys_rather_than_assumed(tmp_path, frozen):
    """lerobot builds ACT's ResNet with FrozenBatchNorm2d, whose buffers do not
    include `num_batches_tracked`; a stock BatchNorm2d has one per norm.
    MEASURED on torchvision 0.26: resnet18 is 102 state-dict keys frozen and
    122 unfrozen, the 20 extra being exactly those counters.

    Assume either one and the other checkpoint reports 20 missing tensors and
    is refused as "a different architecture", which it is not."""
    name = f"act_{'frozen' if frozen else 'plain'}"
    repo, body = _act_repo(tmp_path, name, frozen=frozen)
    status = vla.VLAHandle().load(repo, tmp_path, image_size=TEST_EDGE)
    assert status.loaded
    counters = [k for k in body if k.endswith("num_batches_tracked")]
    assert len(counters) == (0 if frozen else 20)


def test_a_backbone_name_torchvision_does_not_have_is_refused(tmp_path):
    """A string out of a config file reaching `getattr` on a module would hand
    back `os` as happily as `resnet18`. The refusal names the registry it was
    checked against rather than substituting something similar."""
    repo, _ = _act_repo(tmp_path, "act_bogus", vision_backbone="resnet17")
    with pytest.raises(Refusal) as caught:
        vla.VLAHandle().load(repo, tmp_path, image_size=TEST_EDGE)
    assert "resnet17" in caught.value.sentence
    assert "resnet18" in caught.value.sentence  # the near names it does have


# ------------------------------------------------------------- the input size


def test_a_config_that_declares_no_camera_shape_is_refused_with_the_fix(tmp_path):
    """A CNN has no native input size — 224 is a fact about ImageNet, not about
    this policy — so there is no default to fall back on, and a wrong square
    moves every occlusion block somewhere it was not."""
    repo, _ = _act_repo(tmp_path, "act_sizeless", input_features=None)
    with pytest.raises(Refusal) as caught:
        vla.VLAHandle().load(repo, tmp_path)
    assert "image_size" in caught.value.sentence
    assert "resnet18" in caught.value.sentence


def test_the_declared_camera_shape_is_read_when_nothing_overrides_it(tmp_path):
    """The config says 3x480x640; the square fed is the smaller edge, and the
    sentence says what it was squared from rather than doing it silently."""
    repo, _ = _act_repo(tmp_path, "act_declared")
    status = vla.VLAHandle().load(repo, tmp_path)
    assert status.image_size == 480
    assert "3x480x640" in status.reason
    assert "480x480" in status.reason


@pytest.mark.parametrize("bad", [0, -64, 16, 1.5, True])
def test_a_nonsense_image_size_is_the_callers_mistake_not_a_refusal(tmp_path, bad):
    """422, not 409: the parameter they just sent is wrong, and `True` is in
    this list because `isinstance(True, int)` is True and a bool arriving here
    would otherwise be fed to the tower as a 1-pixel square."""
    repo, _ = _act_repo(tmp_path, f"act_bad_{str(bad).replace('.', '_')}")
    with pytest.raises(BadRequest):
        vla.VLAHandle().load(repo, tmp_path, image_size=bad)


def test_a_shape_with_a_bool_in_it_does_not_contribute_a_one():
    """`isinstance(True, int)` is True, so the bool guard goes before the int
    check. A config that wrote `shape: [3, true, 640]` must not silently make
    the policy's input size 1 pixel."""
    spec = vla._conv_backbone_spec(
        {
            "vision_backbone": "resnet18",
            "input_features": {
                "observation.images.top": {"type": "VISUAL", "shape": [3, True, 640]}
            },
        }
    )
    assert spec is not None
    assert spec.image_size != 1


# ---------------------------------------------------- the honest part: no heads


def test_the_status_says_no_heads_rather_than_zero(loaded):
    """THE WHOLE POINT. A ResNet does not have zero attention heads — it has no
    such thing as an attention head, and a 0 in this field is a measurement
    somebody took. `grid` and `patch_size` stay real, because a feature map
    genuinely is a rectangle of positions over the frame and the occlusion
    sweep plans blocks on it."""
    _, status, _, repo = loaded
    assert status.loaded is True
    assert status.n_heads is None
    assert status.n_layers is None
    assert status.n_prefix_tokens is None
    # `mode` names WHICH HALF is open, not what the eyes are made of. A
    # separate value here would make every `mode === "perception"` check false
    # for ACT and hide the panel that does work on it.
    assert status.mode == "perception"
    assert status.grid == [TEST_EDGE // 32, TEST_EDGE // 32]
    assert status.patch_size == 32
    assert status.image_size == TEST_EDGE
    assert status.repo == repo
    # A reader of the payload alone must be able to tell why the two nulls are
    # null without reading this file.
    assert "no attention" in status.reason
    assert "occlusion" in status.reason


def test_the_grid_is_measured_from_the_stack_not_recited(tmp_path):
    """NOT `image_size // 32`. A ResNet's output grid is the product of its own
    strides and `replace_final_stride_with_dilation` changes it, so the only
    honest source is the shape the module returns for the size it is given.
    MEASURED on torchvision 0.26 resnet18: 64 -> 2x2, 224 -> 7x7, 480 -> 15x15
    — and 480 is not a multiple of 15 times anything the config mentions."""
    repo, _ = _act_repo(tmp_path, "act_grid")
    for edge, expected in ((64, 2), (224, 7), (480, 15)):
        status = vla.VLAHandle().load(repo, tmp_path, image_size=edge)
        assert status.grid == [expected, expected], edge
        assert status.patch_size == edge // expected, edge


def test_analyse_refuses_and_names_the_measurement_that_does_work(loaded):
    """A convolutional activation is exactly the right shape to be reshaped
    into [heads, G, G] and painted where every other attention map in this
    panel goes — where it would be read as attention. The refusal has to
    arrive before the forward pass and has to name occlusion, because a
    refusal that names the thing that DOES work is worth twice one that does
    not."""
    handle, _, _, repo = loaded
    frame = torch.zeros(TEST_EDGE, TEST_EDGE, 3, dtype=torch.uint8).numpy()
    with pytest.raises(Refusal) as caught:
        handle.analyse(frame, key=(0, 0))
    sentence = caught.value.sentence
    assert "no attention" in sentence
    assert "/api/vla/occlude" in sentence
    assert "resnet18" in sentence
    assert repo in sentence
    # Nothing was cached, so nothing downstream can find a stale map to draw.
    assert handle._attn == []
    assert handle._attn_key is None


def test_every_attention_door_says_the_same_thing(loaded):
    """Three routes reach the attention cache — analyse, attention, and the
    meta the panel polls — and a reader who tries them in a different order
    must not get three different stories about one policy. `attention/meta`
    in particular used to answer "analyse a frame first", which for this
    policy is an instruction that cannot succeed."""
    handle, _, _, repo = loaded
    expected = handle.model.no_attention_sentence(repo)
    with pytest.raises(Refusal) as from_analyse:
        handle.analyse(
            torch.zeros(TEST_EDGE, TEST_EDGE, 3, dtype=torch.uint8).numpy(), (0, 0)
        )
    with pytest.raises(Refusal) as from_attention:
        handle.attention(0, -1)
    meta = handle.attention_meta()
    assert from_analyse.value.sentence == expected
    assert from_attention.value.sentence == expected
    assert meta == {"available": False, "reason": expected}


def test_the_tower_itself_refuses_output_attentions(loaded):
    """Belt and braces, and not decoration: `vla_sweep` and anything else that
    holds the module can call it directly, and this class is the last place
    that can still say no before a tensor comes back that looks like
    attention."""
    handle, _, _, _ = loaded
    frame = torch.zeros(1, 3, TEST_EDGE, TEST_EDGE, device=handle.status_.device)
    with pytest.raises(Refusal):
        handle.model(pixel_values=frame, output_attentions=True)


# ------------------------------------------------ the part that genuinely works


def test_occlusion_really_runs_on_a_convolutional_backbone(loaded):
    """The claim in the refusal, checked rather than asserted in prose.
    `vla_occlude` needs a pooled embedding and nothing else, so the causal map
    is a real measurement on ACT — and this is what makes the refusal above an
    honest trade rather than a consolation."""
    handle, status, _, _ = loaded
    rng = torch.Generator().manual_seed(5)
    frames = [
        (torch.rand(48, 60, 3, generator=rng) * 255).to(torch.uint8).numpy()
        for _ in range(vla_occlude.SCALE_FRAMES)
    ]
    out = handle.occlude(
        frames[0], frames[1:], baseline="midpoint", stride=1, key=(0, 0)
    )

    assert out["grid"] == status.grid
    assert out["n_blocks"] == len(vla_occlude.plan(status.grid, 1))
    assert all(b["shift"] == b["shift"] for b in out["blocks"]), "a NaN shift"
    assert out["scale"] > 0
    # No attention exists, so there is no rank agreement to report — and None
    # is the honest answer rather than a 0.0 correlation nobody measured.
    assert out["attention_agreement"] is None
    assert out["compared_layer"] is None
    # The sentence still says what it always said about scope.
    assert "action" in out["means"]


def test_the_cost_estimate_answers_for_a_conv_grid(loaded):
    """`/api/vla/occlude/cost` reads `status.grid`, which is why the grid had
    to stay a real measurement instead of being nulled out beside the heads."""
    handle, status, _, _ = loaded
    cost = handle.occlusion_cost(stride=1)
    assert cost["grid"] == status.grid
    assert cost["blocks"] == status.grid[0] * status.grid[1]
