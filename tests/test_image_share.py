"""Shaping an image run for a `.mri`, and the one property that matters.

THE PROPERTY: everything `image_share` writes must be something
`session._image` will read. Those are two modules with two different jobs —
one runs on data this process just measured, the other on a file that arrived
from a stranger — and the reader is deliberately strict. A writer that emits
something the reader refuses is a share button that produces a file nobody can
open, and the failure lands on the RECIPIENT, who has no way to fix it.

So almost every test here ends by putting the payload through the real reader
rather than by inspecting a dict. Asserting on the shape would pass happily
while the two drifted apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from modelmri import image_share, session
from modelmri.errors import BadRequest

PNG = "data:image/png;base64,AAAA"


@dataclass
class FakeStatus:
    repo: str = "PixArt-alpha/PixArt-XL-2-512x512"
    family: str = "diffusion"
    architecture: str = "PixArtTransformer2DModel"
    revision: str = "main"
    device: str = "cuda:0"
    dtype: str = "float16"


@dataclass
class FakeFrame:
    """The fields `image_steps.Frame.to_dict` publishes, and only those."""

    step: int = 0
    timestep: float | None = 999.0
    png: str | None = PNG
    width: int | None = 64
    height: int | None = 64
    decoded_width: int | None = 64
    decoded_height: int | None = 64
    latent_rms: float | None = 1.5

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "timestep": self.timestep,
            "png": self.png,
            "width": self.width,
            "height": self.height,
            "decoded_width": self.decoded_width,
            "decoded_height": self.decoded_height,
            "downsampled": (self.width, self.height)
            != (self.decoded_width, self.decoded_height),
            "latent_rms": self.latent_rms,
        }


@dataclass
class FakeStrip:
    prompt: str = "an astronaut riding a horse"
    seed: int | None = 7
    scheduler: str = "DPMSolverMultistepScheduler"
    frames: list = field(default_factory=lambda: [FakeFrame(), FakeFrame(step=4)])
    steps_requested: int = 20
    steps_run: int = 20
    decoded_steps: list = field(default_factory=lambda: [0, 4])
    skipped_steps: list = field(default_factory=list)
    steps_never_reached: list = field(default_factory=list)


@dataclass
class FakeAttention:
    seed: int | None = 7

    def to_dict(self) -> dict:
        return {
            "tokens": ["an", "astronaut"],
            "steps": [
                {"step": 0, "timestep": 999.0, "per_token": [0.4, 0.6], "blocks": 28}
            ],
            "padding_from": 2,
            "conditioning_width": 120,
            "columns_unlabelled": 0,
            "steps_requested": 20,
            "steps_measured": 1,
            "resolutions": [16, 32],
            "means": "one step of twenty",
        }


def _reads(payload: dict) -> dict:
    """Through the real reader, which is the whole point of these tests."""
    return session._image({"image": payload})


# ------------------------------------------------- the writer feeds the reader


def test_a_filmstrip_share_is_readable():
    got = _reads(image_share.from_filmstrip(FakeStatus(), FakeStrip()))
    assert got["provenance"]["family"] == "diffusion"
    assert got["prompt"] == "an astronaut riding a horse"
    assert len(got["frames"]) == 2
    assert got["frames"][0]["size"] == [64, 64]


def test_a_filmstrip_with_its_cross_attention_is_readable():
    got = _reads(
        image_share.from_filmstrip(FakeStatus(), FakeStrip(), attention=FakeAttention())
    )
    assert got["attention"]["conditioning_width"] == 120
    assert got["attention"]["steps"][0]["per_token"] == [0.4, 0.6]


def test_a_bare_cross_attention_share_is_readable():
    """Capturing the maps never decodes a frame, which is most of the cost, so
    a file with maps and no pictures is a real and common share."""
    got = _reads(image_share.from_attention(FakeStatus(), FakeAttention()))
    assert "frames" not in got
    assert got["attention"]["steps_measured"] == 1
    assert "does not decode the picture" in got["means"]


# --------------------------------------------- what the writer refuses to ship


def test_a_frame_with_no_bytes_is_dropped_rather_than_shipped_as_a_hole():
    """`image_steps` writes `None` here precisely so a decode that produced
    nothing can be told from one that never ran, and a `.mri` reader has no way
    to render that difference — so the frame is left out and the count says so.
    """
    strip = FakeStrip(frames=[FakeFrame(), FakeFrame(step=4, png=None)])
    got = _reads(image_share.from_filmstrip(FakeStatus(), strip))
    assert len(got["frames"]) == 1
    assert "1 frame(s) were left out" in got["means"]


def test_a_frame_with_no_stated_size_is_dropped_and_reported():
    """The reader refuses one, so shipping it would be a share button that
    makes a file the recipient cannot open."""
    strip = FakeStrip(frames=[FakeFrame(), FakeFrame(step=4, width=None)])
    got = _reads(image_share.from_filmstrip(FakeStatus(), strip))
    assert len(got["frames"]) == 1
    assert "no stated resolution" in got["means"]


def test_a_downsampled_frame_carries_the_size_it_came_from():
    strip = FakeStrip(
        frames=[FakeFrame(width=64, height=64, decoded_width=512, decoded_height=512)]
    )
    got = _reads(image_share.from_filmstrip(FakeStatus(), strip))
    assert got["frames"][0]["downsampled"] is True
    assert got["frames"][0]["decoded_size"] == [512, 512]
    assert "shrunk to fit the file" in got["means"]


def test_a_downsampled_frame_with_no_original_size_drops_the_claim():
    """The reader refuses "shrunk from an unknown size" and it is right to.
    Rather than ship a frame it will reject, the writer drops the CLAIM and
    sends the frame at the size it actually is."""
    strip = FakeStrip(
        frames=[FakeFrame(width=64, height=64, decoded_width=None, decoded_height=None)]
    )
    got = _reads(image_share.from_filmstrip(FakeStatus(), strip))
    assert got["frames"][0]["downsampled"] is False
    assert got["frames"][0]["size"] == [64, 64]


# ------------------------------------------------------- unknown is not zero


def test_an_unseeded_run_says_so_and_keeps_none():
    got = _reads(image_share.from_filmstrip(FakeStatus(), FakeStrip(seed=None)))
    assert got["seed"] is None
    assert "NO SEED WAS FIXED" in got["means"]


def test_seed_zero_is_a_seed_and_the_sentence_agrees():
    got = _reads(image_share.from_filmstrip(FakeStatus(), FakeStrip(seed=0)))
    assert got["seed"] == 0
    assert "Seed 0, so the run repeats" in got["means"]


def test_a_choice_and_a_gap_are_reported_separately():
    """One is "we sampled the run" and the other is "the pipeline's callback
    never fired". A strip that folded them together would read as eight of
    fifty either way."""
    strip = FakeStrip(skipped_steps=[1, 2], steps_never_reached=[19])
    got = _reads(image_share.from_filmstrip(FakeStatus(), strip))
    assert got["skipped_steps"] == [1, 2]
    assert got["steps_never_reached"] == [19]
    assert "a choice, not a gap" in got["means"]
    assert "a gap, not a choice" in got["means"]


# ------------------------------------------------------------ the readout arm


@dataclass
class FakePrediction:
    task: str = "detection"
    width: int = 640
    height: int = 480
    boxes: list = field(default_factory=list)
    classes_top: list = field(default_factory=list)
    segments: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "width": self.width,
            "height": self.height,
            "boxes": list(self.boxes),
            "classes_top": list(self.classes_top),
            "segments": list(self.segments),
            "means": "read off the head",
        }


def test_a_detection_readout_is_readable_and_says_what_its_scores_are():
    pred = FakePrediction(
        boxes=[
            {
                "query": 3,
                "index": 17,
                "label": "horse",
                "score": 0.92,
                "box_xyxy": [1.0, 2.0, 30.0, 40.0],
            }
        ]
    )
    got = _reads(image_share.from_readout(FakeStatus(family="detection"), pred))
    assert got["readout"]["kind"] == "detection"
    assert got["readout"]["rows"][0]["box_xyxy"] == [1.0, 2.0, 30.0, 40.0]
    assert "do not sum to one" in got["means"]


def test_a_classification_readout_reads_the_other_column():
    """`classes_top` carries `probability`, `boxes` carries `score`. Reading
    the wrong key produced an empty readout and no error at all — which is a
    share button that silently sends nothing."""
    pred = FakePrediction(
        task="classification",
        classes_top=[{"index": 5, "label": "zebra", "probability": 0.71}],
    )
    got = _reads(image_share.from_readout(FakeStatus(), pred))
    assert got["readout"]["rows"][0]["label"] == "zebra"
    assert got["readout"]["rows"][0]["score"] == pytest.approx(0.71)
    assert "probabilities over the class list" in got["means"]


def test_a_segmentation_readout_uses_the_fraction_and_names_it_as_one():
    """A third quantity again: a share of the MAP's cells, which is neither a
    probability nor a confidence."""
    pred = FakePrediction(
        task="semantic_segmentation",
        segments=[{"index": 2, "label": "sky", "fraction": 0.34, "cells": 120}],
    )
    got = _reads(image_share.from_readout(FakeStatus(), pred))
    assert got["readout"]["rows"][0]["score"] == pytest.approx(0.34)
    assert "shares of the map's cells" in got["means"]


def test_the_kind_comes_off_the_prediction_and_cannot_be_mislabelled():
    """`image_cv` decides the task from what the model RETURNED. Taking it as
    an argument would let a caller file a detector's output as a classifier's,
    which is exactly the confusion the field exists to prevent."""
    import inspect

    assert "kind" not in inspect.signature(image_share.from_readout).parameters


def test_a_readout_carries_the_picture_its_boxes_are_drawn_on():
    pred = FakePrediction(boxes=[{"index": 1, "label": "cat", "score": 0.5}])
    got = _reads(image_share.from_readout(FakeStatus(), pred, picture=PNG))
    assert got["frames"][0]["size"] == [640, 480]


def test_a_picture_with_no_stated_size_is_left_out_and_the_reason_given():
    """Boxes drawn over a picture of the wrong resolution land nowhere, so no
    picture beats a wrong one — and the file says which happened."""
    pred = FakePrediction(
        width=0, height=0, boxes=[{"index": 1, "label": "cat", "score": 0.5}]
    )
    got = _reads(image_share.from_readout(FakeStatus(), pred, picture=PNG))
    assert "frames" not in got
    assert "did not state the resolution" in got["means"]


def test_a_row_with_no_finite_score_is_dropped_and_counted():
    pred = FakePrediction(
        boxes=[
            {"index": 1, "label": "cat", "score": 0.5},
            {"index": 2, "label": "dog", "score": None},
        ]
    )
    got = _reads(image_share.from_readout(FakeStatus(), pred, picture=PNG))
    assert len(got["readout"]["rows"]) == 1
    assert "1 row(s) were left out" in got["means"]


def test_an_unlabelled_class_gets_its_index_not_a_borrowed_name():
    """A checkpoint that publishes no `id2label` gets indices. A wrong class
    name reads as the model's answer, which is worse than a number nobody can
    interpret."""
    pred = FakePrediction(
        task="classification", classes_top=[{"index": 281, "probability": 0.4}]
    )
    got = _reads(image_share.from_readout(FakeStatus(), pred))
    assert got["readout"]["rows"][0]["label"] == "#281"


# ------------------------------------------------------------ the leak rule


def test_a_local_folder_is_shared_by_name_and_never_by_path(tmp_path):
    """`runtime.export_session` learned this the hard way: `repo` is a Hub id
    for a Hub model and an ABSOLUTE PATH for one loaded out of a local folder,
    and a `.mri` is the one artefact here designed to leave the machine.
    Publishing the raw id shipped somebody's home directory to whoever they
    sent the file to."""
    folder = tmp_path / "my-secret-project" / "sd-turbo"
    folder.mkdir(parents=True)
    got = _reads(image_share.from_filmstrip(FakeStatus(repo=str(folder)), FakeStrip()))
    assert got["provenance"]["repo"] == "sd-turbo"
    assert str(tmp_path) not in got["provenance"]["repo"]


def test_a_hub_id_survives_intact():
    """The other half — a Hub id has slashes in it and is not a path."""
    got = _reads(image_share.from_filmstrip(FakeStatus(), FakeStrip()))
    assert got["provenance"]["repo"] == "PixArt-alpha/PixArt-XL-2-512x512"


def test_a_checkpoint_with_no_revision_writes_the_empty_claim_not_a_gap():
    """ "" is "this checkpoint published none" and the reader accepts it; a
    MISSING field is refused, because it cannot be told from nobody having
    looked. The writer must therefore always emit the key."""
    got = image_share.from_filmstrip(FakeStatus(revision=""), FakeStrip())
    assert got["provenance"]["revision"] == ""
    assert _reads(got)["provenance"]["revision"] == ""


def test_a_status_missing_a_field_entirely_still_produces_a_readable_file():
    """`getattr` defaults are not laziness here: `ImageStatus` has no
    `revision` field at all, so a share built from one must still emit the key
    rather than raise or omit it."""

    @dataclass
    class Thin:
        repo: str = "x/y"
        family: str = "diffusion"
        architecture: str = "UNet"

    got = _reads(image_share.from_filmstrip(Thin(), FakeStrip()))
    assert got["provenance"]["revision"] == ""


def test_a_status_with_no_family_is_refused_by_the_reader_rather_than_shipped():
    """The one case the writer cannot paper over: a section that does not say
    what kind of model drew the picture is not reproducible, and the reader
    names the missing field."""

    @dataclass
    class NoFamily:
        repo: str = "x/y"
        family: str = ""
        architecture: str = "UNet"

    with pytest.raises(BadRequest) as caught:
        _reads(image_share.from_filmstrip(NoFamily(), FakeStrip()))
    assert "family" in caught.value.sentence
