"""An image run inside a `.mri`: what travels, and what is refused.

A6 on the roadmap, and the last unbuilt item in Theme A. Every other result
this tool produces could be sent to somebody — a generation, a patching trace,
a head ranking, a robot episode — and the one that is a PICTURE could not. So
an image finding was the only kind that had to be screenshot to be shared, and
a screenshot carries no provenance, no seed, no scheduler and no statement of
what was shrunk on the way out.

The section is held to `_vla`'s standard because it is exposed the same way: a
`.mri` is designed to arrive from a stranger, and this one reaches a browser as
a run of <img> srcs and a heat map drawn over them.

Three rules are specific to it and each has its own test below:

  every frame states its size    and whether it was shrunk, and from what. A
                                 map drawn over a silently resized picture is
                                 wrong in the way that looks like a finding.
  a revision may be EMPTY        but never absent. "" is the claim "this
                                 checkpoint published none"; a missing field
                                 cannot be told from nobody having looked.
  a section needs a measurement  provenance and a prompt describe a run. They
                                 are not one.
"""

from __future__ import annotations

import gzip
import json

import pytest

from modelmri import session
from modelmri.errors import BadRequest

PNG = "data:image/png;base64,AAAA"


def _provenance(**over) -> dict:
    out = {
        "repo": "stabilityai/sd-turbo",
        "family": "diffusion",
        "architecture": "UNet2DConditionModel",
        "revision": "abc123",
        "kind": "denoising",
    }
    out.update(over)
    return out


def _frame(**over) -> dict:
    out = {
        "step": 0,
        "timestep": 999.0,
        "png": PNG,
        "size": [64, 64],
        "downsampled": False,
        "latent_rms": 1.25,
    }
    out.update(over)
    return out


def _image(**over) -> dict:
    out = {
        "provenance": _provenance(),
        "prompt": "an astronaut riding a horse",
        "seed": 7,
        "scheduler": "EulerAncestralDiscreteScheduler",
        "frames": [_frame(), _frame(step=4, timestep=500.0)],
        "steps_requested": 20,
        "steps_run": 20,
        "decoded_steps": [0, 4],
        "skipped_steps": [1, 2, 3],
        "steps_never_reached": [],
        "attention": {
            "tokens": ["an", "astronaut", "riding", "a", "horse"],
            "steps": [
                {
                    "step": 0,
                    "timestep": 999.0,
                    "per_token": [0.1, 0.5, 0.2, 0.05, 0.15],
                    "blocks": 16,
                }
            ],
            "padding_from": 5,
            "conditioning_width": 77,
            "columns_unlabelled": 0,
            "steps_requested": 20,
            "steps_measured": 1,
            "resolutions": [8, 16, 32],
            "means": "one step of twenty",
        },
        "means": "two decoded frames of a twenty-step run",
    }
    out.update(over)
    return out


def _build(**over) -> bytes:
    args = dict(
        model_id="sd-turbo",
        device="cpu",
        dtype="float16",
        n_params=None,
        tokens=[],
        prompt="",
        generation="",
        attention={},
        n_layers=0,
        n_heads=0,
    )
    args.update(over)
    return session.build(**args)


# ------------------------------------------------------------- round trip


def test_an_image_run_survives_the_round_trip():
    parsed = session.parse(_build(image=_image()))
    assert parsed.has_image()
    assert parsed.image["provenance"]["repo"] == "stabilityai/sd-turbo"
    assert len(parsed.image["frames"]) == 2
    assert parsed.image["frames"][1]["step"] == 4
    assert parsed.image["attention"]["steps"][0]["per_token"][1] == pytest.approx(0.5)


def test_a_session_without_an_image_run_carries_no_empty_section():
    """Additive, like every section after `patch`. A file written before this
    existed has no `image` key, and a reader that invented an empty one would
    make "no image run" indistinguishable from "an image run with nothing in
    it"."""
    raw = json.loads(gzip.decompress(_build()).decode("utf-8"))
    assert "image" not in raw
    assert session.parse(_build()).has_image() is False


def test_the_format_version_does_not_move_for_an_additive_section():
    """The whole reason these sections are additive: an older reader has to
    ignore the key rather than refuse the file."""
    plain = json.loads(gzip.decompress(_build()).decode("utf-8"))
    withimg = json.loads(gzip.decompress(_build(image=_image())).decode("utf-8"))
    assert plain["format_version"] == withimg["format_version"]


# ---------------------------------------------------- unknown is not zero


def test_an_unseeded_run_keeps_none_rather_than_zero():
    """THE DISTINCTION THIS SECTION EXISTS ON. A run with no fixed seed is not
    a run with seed 0: rerun it and the trajectory differs, and nothing
    downstream compares. A 0 written here would be a promise of repeatability
    that the file cannot keep."""
    parsed = session.parse(_build(image=_image(seed=None)))
    assert parsed.image["seed"] is None


def test_seed_zero_is_a_real_seed_and_survives_as_one():
    """The other half. Refusing to write 0 would be as wrong as inventing it."""
    parsed = session.parse(_build(image=_image(seed=0)))
    assert parsed.image["seed"] == 0


def test_a_frame_with_no_latent_rms_keeps_none():
    parsed = session.parse(_build(image=_image(frames=[_frame(latent_rms=None)])))
    assert parsed.image["frames"][0]["latent_rms"] is None


# ------------------------------------------- the picture states its size


def test_a_frame_that_does_not_state_its_resolution_is_refused():
    """A cross-attention map is drawn over the picture. A frame that has been
    shrunk without saying so puts every cell in the wrong place, and the
    picture is then wrong in the way that looks exactly like a finding."""
    with pytest.raises(BadRequest) as caught:
        session.parse(_build(image=_image(frames=[_frame(size=None)])))
    assert "resolution" in caught.value.sentence


def test_downsampled_without_the_original_size_is_refused():
    """ "Shrunk from an unknown size" is not a resolution anybody can put a map
    back onto."""
    with pytest.raises(BadRequest) as caught:
        session.parse(_build(image=_image(frames=[_frame(downsampled=True)])))
    assert "from what" in caught.value.sentence


def test_a_downsampled_frame_that_says_from_what_is_kept():
    parsed = session.parse(
        _build(image=_image(frames=[_frame(downsampled=True, decoded_size=[512, 512])]))
    )
    frame = parsed.image["frames"][0]
    assert frame["downsampled"] is True
    assert frame["decoded_size"] == [512, 512]


def test_a_frame_that_is_not_a_data_url_is_refused():
    """A `.mri` never carries a path or a link. The picture travels inside it
    or the frame does not travel at all."""
    with pytest.raises(BadRequest) as caught:
        session.parse(_build(image=_image(frames=[_frame(png="/tmp/step0.png")])))
    assert "data URL" in caught.value.sentence


def test_a_frame_with_no_step_is_refused():
    """A strip whose frames have no steps reads as consecutive, and these are
    usually a sample of a much longer run."""
    with pytest.raises(BadRequest) as caught:
        session.parse(_build(image=_image(frames=[_frame(step=None)])))
    assert "denoising step" in caught.value.sentence


# --------------------------------------------------------- the provenance


@pytest.mark.parametrize("field", ["repo", "family", "kind"])
def test_provenance_names_every_field_it_is_missing(field):
    with pytest.raises(BadRequest) as caught:
        session.parse(_build(image=_image(provenance=_provenance(**{field: ""}))))
    assert field in caught.value.sentence


def test_an_empty_architecture_is_a_claim_and_is_accepted():
    """`architecture` is not in the loop above, and this is why. A transformers
    `config.json` that carries no `architectures` gives `imaging` nothing to
    report, so "" is the true answer -- the checkpoint published none.

    Refusing it refused the SHARE, at the reader, after the measurement had
    already been made: `image_share` writes what the status carries, and
    `session.build` validates the writer's own output before writing a byte.
    So a readout of such a checkpoint could be measured and never sent."""
    parsed = session.parse(
        _build(image=_image(provenance=_provenance(architecture="")))
    )
    assert parsed.image["provenance"]["architecture"] == ""


def test_a_missing_architecture_is_refused():
    """The other half, exactly as for `revision`: "" is a claim and absence is
    silence, and the two cannot be told apart once they are folded together."""
    prov = _provenance()
    del prov["architecture"]
    with pytest.raises(BadRequest) as caught:
        session.parse(_build(image=_image(provenance=prov)))
    assert "architecture" in caught.value.sentence


def test_an_empty_revision_is_a_claim_and_is_accepted():
    """A checkpoint out of a local folder published no revision, and saying so
    is true. Refusing it would make the feature unusable for exactly the
    people most likely to need it."""
    parsed = session.parse(_build(image=_image(provenance=_provenance(revision=""))))
    assert parsed.image["provenance"]["revision"] == ""


def test_a_missing_revision_is_refused():
    """The other half, and the reason the empty one is allowed: absence cannot
    be told apart from nobody having looked."""
    prov = _provenance()
    del prov["revision"]
    with pytest.raises(BadRequest) as caught:
        session.parse(_build(image=_image(provenance=prov)))
    assert "revision" in caught.value.sentence


def test_a_section_with_no_provenance_at_all_is_refused():
    img = _image()
    del img["provenance"]
    # `build` drops a section with no provenance rather than writing one it
    # will refuse to read, so this goes in through the reader.
    with pytest.raises(BadRequest):
        session._image({"image": img})


# ---------------------------------------------- a section needs a finding


def test_provenance_alone_is_not_an_image_run():
    """Provenance and a prompt describe a run. They are not one, and a file
    claiming an image run nobody can look at is worse than no section."""
    with pytest.raises(BadRequest) as caught:
        session._image({"image": {"provenance": _provenance(), "prompt": "a horse"}})
    assert "no measurement" in caught.value.sentence


def test_cross_attention_alone_is_a_complete_run():
    """Capturing the maps never decodes a frame, which is most of the cost. A
    file with maps and no pictures is a real and common share."""
    img = _image()
    del img["frames"]
    parsed = session.parse(_build(image=img))
    assert parsed.has_image()
    assert "frames" not in parsed.image


def test_a_readout_alone_is_a_complete_run():
    parsed = session.parse(
        _build(
            image=_image(
                frames=[_frame()],
                attention=None,
                readout={
                    "kind": "detection",
                    "rows": [
                        {
                            "label": "horse",
                            "score": 0.92,
                            "index": 17,
                            "query": 3,
                            "box_xyxy": [1.0, 2.0, 30.0, 40.0],
                        }
                    ],
                    "means": "one box over threshold",
                },
            )
        )
    )
    row = parsed.image["readout"]["rows"][0]
    assert row["label"] == "horse"
    assert row["box_xyxy"] == [1.0, 2.0, 30.0, 40.0]


def test_a_readout_that_does_not_say_what_its_scores_are_is_refused():
    """A classifier's probability and a detector's score are different
    quantities that both render as a number between 0 and 1. A readout with no
    kind invites a reader to compare two things that do not compare."""
    with pytest.raises(BadRequest) as caught:
        session._image(
            {
                "image": {
                    **_image(),
                    "readout": {"rows": [{"label": "horse", "score": 0.9}]},
                }
            }
        )
    assert "kind" in caught.value.sentence


def test_a_readout_row_with_no_finite_score_is_refused():
    """It would render as a blank bar in a chart of measured ones."""
    with pytest.raises(BadRequest) as caught:
        session._image(
            {
                "image": {
                    **_image(),
                    "readout": {
                        "kind": "classification",
                        "rows": [{"label": "horse", "score": float("nan")}],
                    },
                }
            }
        )
    assert "finite score" in caught.value.sentence


def test_a_row_with_no_box_keeps_none_rather_than_a_zero_rectangle():
    """A box at the origin with no width is a drawable rectangle, and it would
    be drawn."""
    parsed = session.parse(
        _build(
            image=_image(
                readout={
                    "kind": "classification",
                    "rows": [{"label": "horse", "score": 0.9}],
                    "means": "",
                }
            )
        )
    )
    assert parsed.image["readout"]["rows"][0]["box_xyxy"] is None


# ------------------------------------------------------ untrusted input


def test_a_nan_in_a_map_is_refused_rather_than_rendered_blank():
    """A NaN quantises the whole map to a smooth, plausible, entirely blank
    picture. `_vla` records the same lesson about the same failure."""
    img = _image()
    img["attention"]["steps"][0]["per_token"] = [0.1, float("nan"), 0.2, 0.0, 0.1]
    with pytest.raises(BadRequest) as caught:
        session._image({"image": img})
    assert "finite number" in caught.value.sentence


def test_an_attention_run_with_no_steps_is_refused():
    """Early steps decide layout and late ones decide texture, so a run with no
    steps is not a smaller answer — it is no answer."""
    img = _image()
    img["attention"]["steps"] = []
    with pytest.raises(BadRequest) as caught:
        session._image({"image": img})
    assert "no steps" in caught.value.sentence


def test_absurd_counts_are_refused_by_name():
    """Every cap says what it is and what was claimed, so a reader of the
    refusal knows whether their file is broken or merely large."""
    img = _image()
    img["frames"] = [_frame(step=i) for i in range(session.MAX_IMAGE_FRAMES + 1)]
    with pytest.raises(BadRequest) as caught:
        session._image({"image": img})
    assert str(session.MAX_IMAGE_FRAMES) in caught.value.sentence


def test_a_frame_past_the_byte_cap_is_refused():
    img = _image()
    img["frames"] = [_frame(png=PNG + "A" * session.MAX_IMAGE_FRAME_BYTES)]
    with pytest.raises(BadRequest) as caught:
        session._image({"image": img})
    assert "bytes" in caught.value.sentence


def test_a_section_that_is_not_fields_is_refused():
    with pytest.raises(BadRequest):
        session._image({"image": [1, 2, 3]})


# ------------------------------------------- unknown counts keep their None


@pytest.mark.parametrize("written", [None, "five", -3, 2.0, True])
def test_an_unmeasured_padding_boundary_stays_unknown(written):
    """THE WORST NUMBER THIS FORMAT COULD INVENT.

    `padding_from` is an INDEX, not a count. Every other number beside it --
    `conditioning_width`, `columns_unlabelled`, `steps_measured` -- degrades
    safely at 0, because a panel shown 0 of them shows nothing and every one
    is gated on `> 0`. This one at 0 is an ASSERTION: the padding starts at
    column zero, so every measured column is `<pad>` and none of them is your
    prompt.

    That is the exact conclusion the field exists to prevent, and collapsing an
    absent claim into 0 made the replay panel state it in prose over a run
    whose boundary was never measured.

    Kept apart from a real 0 deliberately: `image_attention` reads 0 as "there
    is no padding" (`if self.padding_from and ...`) and so does the live panel.
    A file that says 0 is saying that; a file that says nothing is saying
    nothing."""
    raw = _image()
    if written is None:
        raw["attention"].pop("padding_from", None)
    else:
        raw["attention"]["padding_from"] = written
    parsed = session.parse(_build(image=raw))
    assert parsed.image["attention"]["padding_from"] is None


def test_a_measured_padding_boundary_survives_including_zero():
    """The other half. 0 here is a real claim -- "the padding starts at column
    zero" -- and this reader has no standing to second-guess it."""
    for written in (0, 5):
        raw = _image()
        raw["attention"]["padding_from"] = written
        parsed = session.parse(_build(image=raw))
        assert parsed.image["attention"]["padding_from"] == written


@pytest.mark.parametrize("field", ["steps_requested", "steps_run"])
@pytest.mark.parametrize("written", [None, "twenty", -1, True])
def test_an_unstated_run_length_stays_unknown(field, written):
    """ "3 of 0 step(s) decoded" is a sentence about a run that cannot have
    happened: nothing decodes three frames of a zero-step run. A file that
    never said how long the run was is not a file claiming it was zero."""
    raw = _image(frames=[_frame()])
    if written is None:
        raw.pop(field, None)
    else:
        raw[field] = written
    parsed = session.parse(_build(image=raw))
    assert parsed.image[field] is None


def test_a_stated_run_length_survives():
    raw = _image(frames=[_frame()])
    raw["steps_requested"] = 50
    raw["steps_run"] = 50
    parsed = session.parse(_build(image=raw))
    assert parsed.image["steps_requested"] == 50
    assert parsed.image["steps_run"] == 50
