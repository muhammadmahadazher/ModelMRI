# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""What a LoRA touches, read off the adapter itself.

Written against the module's own promises rather than its implementation:

  a norm is a MAGNITUDE, never an effect  -> the response says so, always
  the alpha/rank scale is APPLIED         -> and flagged when it cannot be
  relative size needs the base weights    -> None, never approximated
  a cap is reported                       -> a 700-module adapter is not a
                                             small one with a short list

Nothing here downloads. The adapters are written to disk with real tensors,
because the thing under test reads a safetensors header and multiplies two
matrices — an adapter stubbed at the file layer would test the stub.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

from modelmri import adapter_diff  # noqa: E402
from modelmri.errors import BadRequest  # noqa: E402


def _write(tmp_path, tensors, *, name="pytorch_lora_weights.safetensors", config=None):
    path = tmp_path / name
    safetensors.save_file(tensors, str(path))
    if config is not None:
        (tmp_path / "adapter_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
    return path


def _pair(prefix, rank=4, dim=8, scale=1.0):
    """One LoRA pair, in diffusers spelling."""
    return {
        f"{prefix}.lora_A.weight": torch.full((rank, dim), scale),
        f"{prefix}.lora_B.weight": torch.full((dim, rank), scale),
    }


def test_it_names_the_component_and_the_role_it_reaches(tmp_path):
    """The question people actually have: did it touch the part that reads the
    prompt? Cross-attention is where words enter a diffusion model."""
    t = {}
    t.update(_pair("unet.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k"))
    t.update(_pair("text_encoder.layers.3.self_attn.q_proj"))
    got = adapter_diff.read(_write(tmp_path, t))

    assert got.modules_total == 2
    assert "UNet" in got.components and "text encoder" in got.components
    assert "cross-attention" in got.roles
    assert "self-attention" in got.roles


def test_the_alpha_scale_is_applied_not_assumed(tmp_path):
    """`ΔW = B @ A * (alpha / rank)`. An adapter with alpha below rank is
    weaker than its raw product, and most set it lower."""
    t = _pair("unet.attn2.to_q", rank=4, dim=8)
    unscaled = adapter_diff.read(_write(tmp_path, dict(t)))

    scaled = adapter_diff.read(
        _write(tmp_path, dict(t), config={"lora_alpha": 2, "r": 4})
    )

    assert unscaled.all_scaled is False
    assert scaled.all_scaled is True
    assert scaled.top[0].scale == pytest.approx(0.5)
    # Same tensors, half the alpha/rank: exactly half the reported magnitude.
    assert scaled.top[0].delta_norm == pytest.approx(
        unscaled.top[0].delta_norm * 0.5, rel=1e-5
    )


def test_an_unscalable_module_is_flagged_rather_than_assumed(tmp_path):
    """Assuming alpha equals rank silently inflates every adapter that set it
    lower. The norm is reported unscaled and SAID to be."""
    got = adapter_diff.read(_write(tmp_path, _pair("unet.attn1.to_v")))

    assert got.all_scaled is False
    assert any("UNSCALED" in n for n in got.notes)
    assert "UNSCALED" in got.means()


def test_relative_size_is_none_without_the_base_weights(tmp_path):
    """The ratio's whole point is its denominator. Without the base model
    there is no denominator, and an approximation would be a ratio against
    something nobody chose."""
    got = adapter_diff.read(_write(tmp_path, _pair("unet.attn2.to_k")))

    assert all(m.relative is None for m in got.top)
    assert "absolute norms only" in got.means()


def test_every_response_says_a_norm_is_not_an_effect(tmp_path):
    """The one sentence that must never be dropped: a big delta in a layer the
    sampler barely uses can matter less than a small one it leans on."""
    got = adapter_diff.read(_write(tmp_path, _pair("unet.attn2.to_k")))

    assert "not its effect" in got.means()


def test_a_truncated_list_says_how_many_it_left_out(tmp_path):
    """A cap nobody is told about reads as "this adapter is small"."""
    t = {}
    for i in range(12):
        t.update(_pair(f"unet.block{i}.attn2.to_k"))

    got = adapter_diff.read(_write(tmp_path, t), top=3)

    assert got.modules_total == 12
    assert got.modules_listed == 3
    assert "the rest are in the group totals" in got.means()
    # And the groups still account for everything.
    assert sum(g.modules for g in got.groups) == 12


def test_a_merged_checkpoint_is_refused_by_what_it_is(tmp_path):
    """A full fine-tune is a legitimate file this cannot decompose, and the
    remedy is different from "the file is broken"."""
    merged = {"unet.attn2.to_k.weight": torch.zeros((8, 8))}

    with pytest.raises(adapter_diff.NotAnAdapter) as err:
        adapter_diff.read(_write(tmp_path, merged))

    assert "no LoRA pairs" in err.value.sentence


def test_a_directory_with_no_adapter_says_what_it_looked_for(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()

    with pytest.raises(BadRequest) as err:
        adapter_diff.read(empty)

    assert "adapter_model.safetensors" in err.value.sentence


def test_kohya_spelling_is_read_too(tmp_path):
    """diffusers, PEFT and kohya spell the two halves differently, and a
    reader that knows one silently reports "no LoRA" for the others."""
    t = {
        "lora_unet_down_attn2_to_k.lora_down.weight": torch.ones((4, 8)),
        "lora_unet_down_attn2_to_k.lora_up.weight": torch.ones((8, 4)),
    }
    got = adapter_diff.read(_write(tmp_path, t))

    assert got.modules_total == 1
    assert got.components == ["UNet"]


# ---------------------------------------------------------------------------
# Both of these were found by running the reader against a REAL published
# adapter (latent-consistency/lcm-lora-sdxl) after the synthetic tests above
# were already green. Synthetic pairs prove the arithmetic; only a real file
# exercises the shapes and the spellings the ecosystem actually ships.
# ---------------------------------------------------------------------------


def test_a_convolutional_lora_composes(tmp_path):
    """A conv pair is `down (rank, in, kh, kw)` and `up (out, rank, 1, 1)`.

    Reshaping `up` from its LAST axis turned (1280, 64, 1, 1) into (81920, 1),
    and every conv module in the real adapter was dropped with a "do not
    compose" note. The note was honest; the arithmetic was not. Both halves
    flatten from axis 0.
    """
    t = {
        "unet.down_blocks.0.resnets.0.conv1.lora_A.weight": torch.ones((4, 320, 3, 3)),
        "unet.down_blocks.0.resnets.0.conv1.lora_B.weight": torch.ones((640, 4, 1, 1)),
    }
    got = adapter_diff.read(_write(tmp_path, t))

    assert got.modules_total == 1, got.notes
    assert not got.notes or all("do not compose" not in n for n in got.notes)
    assert got.top[0].role == "convolution"
    assert got.top[0].delta_norm > 0


def test_feed_forward_is_recognised_in_the_underscore_spelling(tmp_path):
    """`\bff\b` cannot match `..._blocks_9_ff_net_0_proj`: `_` is a word
    character, so there is no boundary. Every feed-forward module in the real
    adapter fell through to "other", which then reported as its LARGEST group
    — 162 modules and the top five deltas, all mislabelled."""
    t = _pair("lora_unet_up_blocks_0_attentions_1_transformer_blocks_9_ff_net_0_proj")

    got = adapter_diff.read(_write(tmp_path, t))

    assert got.top[0].role == "feed-forward"
    assert "other" not in got.roles
