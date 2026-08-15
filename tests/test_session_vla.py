"""Robot findings inside a `.mri`: what travels, and what is refused.

There is no portable, no-account artifact for robot-policy internals anywhere
— Foxglove archived its open-source Studio, Rerun's `.rrd` carries what the
robot recorded rather than what the network computed, and HF Spaces need an
upload and an account. So this section is the whole point of the feature, and
it is also the one most exposed: a `.mri` is designed to arrive from a
stranger, and this one reaches a browser as an <img> src and two nested loops.

The roadmap named two blocking items for it. Both have their own tests here:
a frame that does not state its resolution, and the same untrusted-input
treatment `_patch` got.
"""

from __future__ import annotations

import gzip
import json

import pytest

from modelmri import session
from modelmri.errors import BadRequest


def _provenance(**over) -> dict:
    out = {
        "policy": "lerobot/smolvla_base",
        "dataset": "lerobot/pusht",
        "camera": "observation.images.top",
        "revision": "abc123def456",
        "episode": 5,
        "timestep": 12,
    }
    out.update(over)
    return out


def _vla(**over) -> dict:
    out = {
        "provenance": _provenance(),
        "frame": "data:image/png;base64,AAAA",
        "frame_size": [96, 96],
        "attention": [[[0.1, 0.2], [0.3, 0.4]]],
        "occlusion": {
            "baseline": "episode_mean",
            "grid": [2, 2],
            "stride": 1,
            "attention_agreement": -0.12,
            "blocks": [
                {
                    "row": 0, "col": 0, "shift": 0.4,
                    "control_max": 0.1, "clears_control": True, "control_draws": 8,
                },
                {
                    "row": 0, "col": 1, "shift": 0.1,
                    "control_max": None, "clears_control": None, "control_draws": 0,
                },
            ],
        },
    }
    out.update(over)
    return out


def _build(**over) -> bytes:
    args = dict(
        model_id="smolvla",
        device="cpu",
        dtype="float32",
        n_params=1,
        tokens=["a"],
        prompt="a",
        generation="",
        attention={},
        n_layers=1,
        n_heads=1,
    )
    args.update(over)
    return session.build(**args)


# ------------------------------------------------------------- round trip


def test_robot_findings_survive_the_round_trip():
    parsed = session.parse(_build(vla=_vla()))
    assert parsed.has_vla()
    assert parsed.vla["provenance"]["episode"] == 5
    assert parsed.vla["occlusion"]["blocks"][0]["shift"] == pytest.approx(0.4)


def test_a_session_without_findings_carries_no_empty_section():
    raw = json.loads(gzip.decompress(_build()).decode("utf-8"))
    assert "vla" not in raw
    assert session.parse(_build()).has_vla() is False


def test_an_uncontrolled_block_keeps_none_rather_than_zero():
    """0.0 would read as "a random occlusion here did nothing", which nothing
    measured for a block nobody controlled."""
    parsed = session.parse(_build(vla=_vla()))
    block = parsed.vla["occlusion"]["blocks"][1]
    assert block["control_max"] is None
    assert block["clears_control"] is None


def test_a_negative_agreement_survives():
    """The two maps disagreeing IS the finding, and a negative Spearman is the
    strongest form of it."""
    parsed = session.parse(_build(vla=_vla()))
    assert parsed.vla["occlusion"]["attention_agreement"] == pytest.approx(-0.12)


def test_the_layer_the_agreement_came_from_survives_the_round_trip():
    """A shared .mri carrying a Spearman without the layer it was measured
    against is not a reproducible claim — the number changes with the layer."""
    occlusion = _vla()["occlusion"] | {"compared_layer": 11, "compared_head": -1}
    parsed = session.parse(_build(vla=_vla(occlusion=occlusion)))
    assert parsed.vla["occlusion"]["compared_layer"] == 11
    assert parsed.vla["occlusion"]["compared_head"] == -1


def test_no_layer_is_stored_as_none_rather_than_left_absent():
    """A reader finding no key cannot tell 'not compared' from 'layer 0'."""
    parsed = session.parse(_build(vla=_vla()))
    assert parsed.vla["occlusion"]["compared_layer"] is None


def test_layer_zero_survives_and_is_not_read_as_absent():
    """The falsy-integer trap: layer 0 is a real layer."""
    occlusion = _vla()["occlusion"] | {"compared_layer": 0, "compared_head": 0}
    parsed = session.parse(_build(vla=_vla(occlusion=occlusion)))
    assert parsed.vla["occlusion"]["compared_layer"] == 0
    assert parsed.vla["occlusion"]["compared_head"] == 0


# ------------------------------------------- blocking item 1: the frame size


def test_a_frame_that_does_not_state_its_resolution_is_refused():
    """A causal map is drawn OVER the frame. A frame silently shrunk to fit a
    byte budget puts every block in the wrong place, and the picture is wrong
    in a way that looks exactly like a finding."""
    with pytest.raises(BadRequest, match="does not state its own resolution"):
        session._vla({"vla": _vla(frame_size=None)})


def test_a_downsampled_frame_must_say_so():
    """`False` is the positive claim "this is the resolution the policy saw".
    A missing key would let a downsampled frame pass as an original."""
    plain = session._vla({"vla": _vla()})
    assert plain["frame_downsampled"] is False
    assert "frame_note" not in plain

    shrunk = session._vla({"vla": _vla(frame_downsampled=True)})
    assert shrunk["frame_downsampled"] is True
    assert "downsampled" in shrunk["frame_note"]


def test_an_oversized_frame_is_refused_with_both_numbers():
    big = "data:image/png;base64," + "A" * (session.MAX_VLA_FRAME_BYTES + 1)
    with pytest.raises(BadRequest, match="above the"):
        session._vla({"vla": _vla(frame=big)})


def test_a_frame_that_is_a_link_rather_than_the_image_is_refused():
    """A `.mri` never carries a path or a URL — the frame travels inside it or
    not at all, or the file is not portable."""
    with pytest.raises(BadRequest, match="image data URL"):
        session._vla({"vla": _vla(frame="https://example.com/frame.png")})


def test_an_occlusion_map_over_an_unsized_frame_is_refused():
    doc = _vla()
    del doc["frame_size"]
    with pytest.raises(BadRequest, match="resolution"):
        session._vla({"vla": doc})


# -------------------------------- blocking item 2: untrusted grids


def test_a_ragged_attention_grid_is_refused():
    """A ragged grid renders as a heat map with a torn edge and nothing on
    screen says the numbers are wrong."""
    with pytest.raises(BadRequest, match="ragged"):
        session._vla({"vla": _vla(attention=[[[0.1, 0.2], [0.3]]])})


def test_a_non_finite_cell_is_refused():
    """A NaN quantises every cell to zero: a smooth, plausible, entirely blank
    map. `_quantise` records the same lesson."""
    with pytest.raises(BadRequest, match="not a finite number"):
        session._vla({"vla": _vla(attention=[[[0.1, float("nan")], [0.3, 0.4]]])})


def test_a_string_where_a_number_belongs_is_refused():
    with pytest.raises(BadRequest, match="not a finite number"):
        session._vla({"vla": _vla(attention=[[["0.1", 0.2], [0.3, 0.4]]])})


def test_an_absurd_grid_is_refused_before_a_browser_lays_it_out():
    huge = [[0.0] * 2 for _ in range(session.MAX_VLA_GRID + 1)]
    with pytest.raises(BadRequest, match="above the"):
        session._vla({"vla": _vla(attention=[huge])})


def test_a_grid_that_is_not_a_grid_is_refused():
    with pytest.raises(BadRequest, match="not a grid"):
        session._vla({"vla": _vla(attention=[42])})


def test_a_block_with_no_shift_is_refused():
    doc = _vla()
    doc["occlusion"]["blocks"] = [{"row": 0, "col": 0}]
    with pytest.raises(BadRequest, match="carries no shift"):
        session._vla({"vla": doc})


# -------------------------------------------------------- the provenance


@pytest.mark.parametrize("field", ["policy", "dataset", "camera", "revision"])
def test_every_provenance_string_is_required(field):
    """A heat map without the policy, dataset, episode, timestep and camera
    that produced it is a picture of nothing in particular."""
    with pytest.raises(BadRequest, match=f"which {field}"):
        session._vla({"vla": _vla(provenance=_provenance(**{field: ""}))})


@pytest.mark.parametrize("field", ["episode", "timestep"])
def test_every_provenance_index_is_required(field):
    with pytest.raises(BadRequest, match=f"which {field}"):
        session._vla({"vla": _vla(provenance=_provenance(**{field: None}))})


def test_a_section_with_no_provenance_at_all_is_refused():
    doc = _vla()
    del doc["provenance"]
    with pytest.raises(BadRequest, match="carries no provenance"):
        session._vla({"vla": doc})


def test_an_occlusion_map_without_its_fill_baseline_is_refused():
    """Occlusion is out of distribution and the two baselines do not agree, so
    a map without its fill is not reproducible."""
    doc = _vla()
    doc["occlusion"]["baseline"] = ""
    with pytest.raises(BadRequest, match="which fill baseline"):
        session._vla({"vla": doc})


# ------------------------------------------ the writer is not laxer


def test_the_writer_refuses_the_same_shape_the_reader_refuses():
    """A writer laxer than the reader is how you build files nobody can open,
    and it means a section missing its provenance is refused at WRITE time
    rather than reaching somebody else's viewer."""
    doc = _vla()
    del doc["provenance"]
    doc["provenance"] = {"policy": "x"}  # incomplete
    with pytest.raises(BadRequest, match="which dataset"):
        _build(vla=doc)


def test_a_section_that_is_not_fields_is_refused():
    with pytest.raises(BadRequest, match="not a set of fields"):
        session._vla({"vla": [1, 2, 3]})


def test_the_section_survives_json_with_no_non_finite_values():
    blob = _build(vla=_vla())
    raw = json.loads(gzip.decompress(blob).decode("utf-8"), parse_constant=_reject)
    assert raw["vla"]["provenance"]["camera"].startswith("observation.images")


def _reject(name):
    raise AssertionError(f"a non-finite {name} reached the file")
