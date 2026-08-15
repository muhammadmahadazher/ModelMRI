"""A VLA viewer, not a SmolVLA viewer.

Three things used to pin this panel to one policy: the tensor prefix
(`model.vlm_with_expert.vlm.model.vision_model.`), the repo its vision config
came from, and the module class. Any other checkpoint found zero tensors and
was told its "layout is not supported" — true only because nothing had
looked.

These tests are about the looking. They use synthetic key lists rather than
real checkpoints on purpose: the point is that discovery works on layouts
nobody has downloaded here.
"""

from __future__ import annotations

import pytest

from modelmri import vla


def test_it_finds_smolvlas_prefix_without_being_told_it():
    """The exact string that used to be hardcoded, now derived."""
    keys = [
        "model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.bias",
        "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn.k_proj.weight",
        "model.vlm_with_expert.lm_expert.layers.0.mlp.down_proj.weight",
        "normalize_inputs.buffer_observation_state.mean",
    ]
    prefix, n = vla.discover_vision_prefix(keys)
    assert prefix == "model.vlm_with_expert.vlm.model.vision_model."
    assert n == 2


@pytest.mark.parametrize(
    "prefix",
    [
        "vision_tower.",  # LLaVA-style
        "model.vision_encoder.",
        "policy.image_encoder.",
        "backbone.visual.",  # Qwen-VL-style
    ],
)
def test_it_finds_the_other_conventions_too(prefix):
    """Published policies do not agree on a name. That is the whole problem."""
    keys = [f"{prefix}encoder.layers.{i}.attn.q_proj.weight" for i in range(4)]
    keys += ["action_head.mlp.0.weight", "state_proj.bias"]
    found, n = vla.discover_vision_prefix(keys)
    assert found == prefix
    assert n == 4


def test_the_busiest_candidate_wins():
    """One stray `visual_proj` must not outvote a real tower."""
    keys = [f"model.vision_model.encoder.layers.{i}.weight" for i in range(20)]
    keys += ["head.visual.proj.weight"]
    prefix, n = vla.discover_vision_prefix(keys)
    assert prefix == "model.vision_model."
    assert n == 20


def test_a_hint_inside_a_word_is_not_a_path_segment():
    """`supervision_model` contains `vision_model` and is not one."""
    keys = ["net.supervision_model.layer.weight", "trunk.weight"]
    with pytest.raises(RuntimeError, match="No vision tower"):
        vla.discover_vision_prefix(keys)


def test_a_checkpoint_with_no_tower_is_told_what_it_does_have():
    """ "Unsupported" is a verdict. The names present are a report — and the
    thing that lets someone say "it is called X in mine"."""
    keys = ["actor.mlp.0.weight", "critic.mlp.0.weight", "obs_norm.mean"]
    with pytest.raises(RuntimeError) as err:
        vla.discover_vision_prefix(keys)
    message = str(err.value)
    assert "actor" in message and "critic" in message
    # And it names what it looked for, so the gap is obvious.
    assert "vision_model" in message


def test_the_default_policy_is_a_default_not_a_requirement():
    """The loader takes a repo. If this signature ever loses it, the panel is
    back to one model."""
    import inspect

    sig = inspect.signature(vla.VLAHandle.load)
    assert "repo" in sig.parameters
    assert sig.parameters["repo"].default == vla.DEFAULT_VLA_REPO


def test_no_module_class_is_named_in_the_loader():
    """The vision module is built from the checkpoint's config by AutoModel.
    Naming a class here is what made every other architecture unreachable."""
    import inspect

    source = inspect.getsource(vla.VLAHandle.load)
    # Comments stripped first. The comment explaining WHY the class is no
    # longer named contains its name, and a check that cannot tell an
    # explanation from the thing it explains forbids documenting the fix.
    code = "\n".join(
        line.split("#", 1)[0] for line in source.splitlines() if line.strip()
    )
    assert "AutoModel.from_config" in code
    assert "SmolVLMVisionTransformer" not in code


def test_a_base_install_is_told_which_extra_encodes_a_frame(monkeypatch):
    """Pillow is in the vla-lite extra; `encode_png` is on three code paths.

    A base install used to get `ModuleNotFoundError: No module named 'PIL'`,
    which names a module the user never asked for and no way to fix it. This
    also caught a real CI hole: torch is a BASE dependency, so the VLA tests
    run everywhere, and they reached this line on a runner where nothing had
    pulled Pillow in.
    """
    import sys

    import numpy as np

    from modelmri.errors import Refusal
    from modelmri.vla_data import encode_png

    for name in [n for n in sys.modules if n == "PIL" or n.startswith("PIL.")]:
        monkeypatch.delitem(sys.modules, name)
    # A None entry is what makes `import PIL` raise, so the import inside
    # encode_png fails the same way it does where Pillow was never installed.
    monkeypatch.setitem(sys.modules, "PIL", None)

    with pytest.raises(Refusal) as caught:
        encode_png(np.zeros((4, 4, 3), dtype="uint8"))
    assert "modelmri[vla-lite]" in str(caught.value)


def test_raw_frame_bounds_the_timestep_like_every_other_accessor():
    """LeRobot v3.0 concatenates many episodes into ONE mp4, so an episode is
    a span inside a file. A `t` past the end resolves to `from_ts + t/fps`,
    lands inside the NEXT episode and decodes a real frame from it — no
    error, a picture on screen.

    `frame()` checks this and `Hdf5Reader.raw_frame` checks this; the one
    method that did not is the one the analysis and occlusion paths call, so
    the causal map, the attention comparison and the shared .mri would all be
    of another episode's frame labelled with this one.
    """
    import inspect

    from modelmri import vla_data

    source = inspect.getsource(vla_data.LeRobotV3Reader.raw_frame)
    assert "match.length" in source
    assert "t must be in" in source


def test_every_frame_accessor_bounds_its_timestep():
    """Structural, because three of the four had this and one did not."""
    import inspect

    from modelmri import hdf5_data, vla_data

    for owner, name in (
        (vla_data.LeRobotV3Reader, "frame"),
        (vla_data.LeRobotV3Reader, "raw_frame"),
        (hdf5_data.Hdf5Reader, "raw_frame"),
    ):
        source = inspect.getsource(getattr(owner, name))
        assert "t must be in" in source, f"{owner.__name__}.{name} bounds nothing"
