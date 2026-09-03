# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""The cached attention map must belong to the frame being occluded.

`VLA._attn` holds whatever was analysed LAST. A cross-episode sweep overwrites
it wholesale (`vla_sweep` analyses under its own key), and the panel's own
frame slider moves independently of it.

Without a key check, occluding frame B while the cache holds frame A ranks B's
causal map against A's attention and reports the Spearman as B's — the headline
number of the whole feature, computed across two different pictures, with
nothing on screen saying so. Comparing the two maps is only a comparison when
they share a frame.

The same hole existed in `share_payload`, where it would have put one frame's
picture and another frame's attention in the same .mri.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from modelmri import vla  # noqa: E402


class OneRegionTower(nn.Module):
    """Reads one quadrant of the image and ignores the rest."""

    def forward(self, pixel_values, output_attentions=False):
        region = pixel_values[..., 32:64, 32:64]
        v = region.mean(dim=(1, 2, 3))
        hidden = torch.stack([v * (i + 1) * 0.01 for i in range(64)], dim=1)
        return type(
            "O", (), {"last_hidden_state": hidden.unsqueeze(-1).expand(-1, -1, 16)}
        )()


@pytest.fixture
def handle():
    """A perception handle with a cache belonging to episode 3, timestep 40."""
    h = vla.VLAHandle()
    h.model = OneRegionTower().eval()
    h.status_ = vla.VLAStatus(
        loaded=True,
        mode="perception",
        device="cpu",
        n_layers=2,
        n_heads=2,
        grid=[8, 8],
        image_size=64,
        patch_size=8,
    )
    # Two layers of [heads, 8, 8] attention, as `analyse` would leave them.
    torch.manual_seed(0)
    h._attn = [torch.rand(2, 8, 8) for _ in range(2)]
    # (episode, timestep, CAMERA). `raw_frame` reads through the reader's
    # process-wide current camera, so the same (episode, timestep) names a
    # different picture on a multi-camera dataset -- and this fixture stands
    # for a frame analysed while the reader was on `_Reader.camera`.
    h._attn_key = (3, 40, "top")
    return h


def _frames(n=4):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(n)]


# ------------------------------------------------------------------ occlude


def test_another_frames_attention_is_not_ranked_against_this_ones(handle):
    """The cache holds episode 3 t 40. Occluding episode 0 t 0 must NOT report
    an agreement, because there is nothing to agree with."""
    frames = _frames()
    out = handle.occlude(frames[0], frames, stride=4, key=(0, 0))
    assert out["attention_agreement"] is None
    assert out["compared_layer"] is None
    assert all(b["attention"] is None for b in out["blocks"])
    assert "No attention map was supplied" in out["means"]


def test_the_matching_frame_is_compared_normally(handle):
    """The guard must not disable the feature — the same key still compares."""
    frames = _frames()
    out = handle.occlude(frames[0], frames, stride=4, key=(3, 40, "top"))
    assert out["attention_agreement"] is not None
    assert out["compared_layer"] == 1, "layer -1 means the last layer"
    assert "against layer 1" in out["means"]


def test_a_caller_that_names_no_frame_still_gets_the_comparison(handle):
    """`key=None` means 'I am not telling you which frame this is' — the old
    behaviour, kept so nothing outside the server route changes silently."""
    frames = _frames()
    out = handle.occlude(frames[0], frames, stride=4)
    assert out["attention_agreement"] is not None


def test_an_empty_cache_is_not_mistaken_for_a_mismatch(handle):
    handle._attn = []
    handle._attn_key = None
    frames = _frames()
    out = handle.occlude(frames[0], frames, stride=4, key=(0, 0))
    assert out["attention_agreement"] is None


def test_a_layer_outside_the_tower_is_refused_not_silently_dropped(handle):
    """It used to fall through to 'no attention map for this frame, run the
    policy on it first' — which sends somebody to do a thing that cannot help,
    for a mistake that is in their layer number."""
    from modelmri.errors import BadRequest

    frames = _frames()
    with pytest.raises(BadRequest, match=r"layer must be in \[0,2\)"):
        handle.occlude(frames[0], frames, stride=4, layer=99, key=(3, 40, "top"))


# ------------------------------------------------------------------- share


class _Reader:
    camera = "top"
    repo_id = "test/dataset"

    def raw_frame(self, episode, t):
        rng = np.random.default_rng(episode * 1000 + t)
        return rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)

    def episodes(self):
        return []

    def summary(self):
        return {}


def test_a_shared_finding_does_not_carry_another_frames_attention(handle):
    """One frame's picture beside another frame's attention, in a file whose
    whole purpose is that somebody else can trust what is in it."""
    payload = vla.share_payload(handle, _Reader(), episode=0, timestep=0, layer=-1)
    assert "attention" not in payload or not payload["attention"]


def test_a_shared_finding_carries_the_attention_of_its_own_frame(handle):
    payload = vla.share_payload(handle, _Reader(), episode=3, timestep=40, layer=-1)
    assert payload["attention"], "the matching frame's maps must still ship"


class _OtherCamera(_Reader):
    """The same dataset, read through a different lens."""

    camera = "wrist"


def test_a_shared_finding_does_not_carry_another_cameras_attention(handle):
    """THE HALF THE KEY WAS MISSING. Episode and timestep matched exactly and
    the maps shipped anyway, because the key said nothing about WHICH CAMERA
    produced them -- so a wrist-tower attention grid could ride in a file
    beside an overhead frame, with `provenance.camera` naming only one of
    them and nothing saying the pair disagreed.

    Same episode, same timestep, different camera: not this frame's maps.
    """
    payload = vla.share_payload(
        handle, _OtherCamera(), episode=3, timestep=40, layer=-1
    )
    assert "attention" not in payload or not payload["attention"]
