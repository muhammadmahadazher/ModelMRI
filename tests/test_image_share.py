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

import pathlib
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


API_KEY = "sk-ant-api03-" + "A" * 88

#: What CLIP's tokenizer does to that key: no piece matches any credential
#: pattern on its own, and `"".join` puts it back together exactly.
SPLIT = ["a", " key", ":", " sk", "-ant", "-api", "03", "-" + "A" * 88]


@dataclass
class KeyAttention:
    """A cross-attention run whose prompt was a credential."""

    seed: int | None = 7

    def to_dict(self) -> dict:
        return {
            "tokens": list(SPLIT),
            "steps": [
                {
                    "step": 0,
                    "timestep": 999.0,
                    "per_token": [1.0 / len(SPLIT)] * len(SPLIT),
                    "blocks": 28,
                }
            ],
            "padding_from": len(SPLIT),
            "conditioning_width": 120,
            "columns_unlabelled": 0,
            "steps_requested": 20,
            "steps_measured": 1,
            "resolutions": [16, 32],
            "means": "one step of twenty",
        }


def test_a_credential_in_an_image_prompt_does_not_reach_the_file():
    """`session.build` sends a text run's prompt through the recorder's
    patterns before a byte is written. A prompt is a prompt whether it
    conditions a language model or a denoiser."""
    got = _reads(
        image_share.from_filmstrip(FakeStatus(), FakeStrip(prompt=f"a key: {API_KEY}"))
    )
    assert API_KEY not in got["prompt"]
    assert "[redacted:api-key]" in got["prompt"]


def test_the_cross_attention_strip_is_where_the_image_prompt_actually_lives():
    """THE TRAP. `from_attention` writes `"prompt": ""` -- it never captured
    one -- so scanning the prompt field and stopping there looks like
    redaction and does nothing. The words are in the token strip, cut into
    pieces by the tokenizer, and `"".join` spells the key."""
    got = _reads(image_share.from_attention(FakeStatus(), KeyAttention()))
    assert got["prompt"] == ""
    assert API_KEY not in "".join(got["attention"]["tokens"])


def test_the_image_strip_keeps_one_entry_per_column():
    """Each token labels a cross-attention column and has one `per_token`
    weight beside it. A shorter strip after redacting would put every weight
    under somebody else's word."""
    got = _reads(image_share.from_attention(FakeStatus(), KeyAttention()))
    tokens = got["attention"]["tokens"]
    assert len(tokens) == len(SPLIT)
    assert len(got["attention"]["steps"][0]["per_token"]) == len(tokens)


def test_the_file_says_a_credential_was_replaced():
    """Applied and REPORTED. A file that quietly says something other than
    what was typed is one whose reader cannot tell a redaction from a
    measurement."""
    got = _reads(image_share.from_attention(FakeStatus(), KeyAttention()))
    assert "1 credential-shaped value(s) were replaced" in got["means"]
    assert "1x api-key" in got["means"]


def test_an_image_share_with_no_credential_says_nothing_about_one():
    """The common case must not acquire a sentence about secrets."""
    got = _reads(image_share.from_filmstrip(FakeStatus(), FakeStrip()))
    assert got["prompt"] == "an astronaut riding a horse"
    assert "credential-shaped" not in got["means"]


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
    #: `image_cv.Prediction.means` opens with the model NAME, so this is where
    #: a local checkpoint's absolute path enters the shared prose.
    means_text: str = "read off the head"

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "width": self.width,
            "height": self.height,
            "boxes": list(self.boxes),
            "classes_top": list(self.classes_top),
            "segments": list(self.segments),
            "means": self.means_text,
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
    got = _reads(
        image_share.from_readout(
            FakeStatus(), pred, picture=PNG, picture_size=(640, 480)
        )
    )
    assert got["frames"][0]["size"] == [640, 480]


def test_the_frame_states_the_pictures_own_size_and_not_the_tensors():
    """THE TWO NUMBERS ARE DIFFERENT AND THE FILE CARRIES THE RIGHT ONE.

    A classifier is shown a 224x224 tensor and the bytes carried beside its
    answer are the upload -- 4000x3000, whatever the photograph was. This wrote
    the TENSOR's shape into the frame's `size` and `downsampled: False` beside
    it, which is a false statement about the file's own contents: `size` means
    "the resolution of these bytes", it is what a map gets drawn onto, and the
    replay panel acts on it by squashing the photograph into a square.

    The tensor's shape is a real fact and is stated as one, in prose, because
    the box coordinates are in that space and a reader scaling them to the
    picture without knowing would put every rectangle in the wrong place."""
    pred = FakePrediction(
        width=224,
        height=224,
        boxes=[{"index": 1, "label": "cat", "score": 0.5, "box_xyxy": [1, 2, 3, 4]}],
    )
    got = _reads(
        image_share.from_readout(
            FakeStatus(), pred, picture=PNG, picture_size=(4000, 3000)
        )
    )
    assert got["frames"][0]["size"] == [4000, 3000]
    assert got["frames"][0]["downsampled"] is False
    assert "224x224 tensor" in got["means"]
    assert "scale them before drawing" in got["means"]


def test_a_picture_whose_own_size_was_not_measured_is_left_out():
    """No picture beats a wrong one, and the file says which happened.

    The size is measured by the CALLER, off the decoded header, because the
    caller is the only one holding the decoded picture. Absent, it is not
    guessed from the tensor -- that guess is the bug the test above pins."""
    pred = FakePrediction(boxes=[{"index": 1, "label": "cat", "score": 0.5}])
    got = _reads(
        image_share.from_readout(FakeStatus(), pred, picture=PNG, picture_size=None)
    )
    assert "frames" not in got
    assert "own resolution could not be read" in got["means"]


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


# --------------------------------- the writer never emits what the reader refuses


def test_a_frame_too_large_for_a_mri_is_dropped_and_the_remedy_given():
    """`image_input` accepts a 32 MB upload and a `.mri` frame is capped at
    4 MB, so a high-resolution run produced a payload the reader refuses and
    the share button answered 422 on the very run somebody wanted to send.

    Dropped and REPORTED, with a remedy that is actually actionable: the frame
    is still on the machine that made it."""
    big = PNG + "A" * session.MAX_IMAGE_FRAME_BYTES
    strip = FakeStrip(frames=[FakeFrame(), FakeFrame(step=4, png=big)])
    got = _reads(image_share.from_filmstrip(FakeStatus(), strip))
    assert len(got["frames"]) == 1
    assert "larger than a `.mri` carries" in got["means"]
    assert "decode fewer steps" in got["means"]


def test_the_two_reasons_a_frame_is_missing_are_counted_apart():
    """ "There was nothing to carry" and "it did not fit" send a reader to two
    different places, so they never share a number."""
    big = PNG + "A" * session.MAX_IMAGE_FRAME_BYTES
    strip = FakeStrip(
        frames=[FakeFrame(), FakeFrame(step=2, png=None), FakeFrame(step=4, png=big)]
    )
    got = _reads(image_share.from_filmstrip(FakeStatus(), strip))
    assert "1 frame(s) were left out of this file because they" in got["means"]
    assert "1 frame(s) were left out because they are larger" in got["means"]


def test_the_total_byte_budget_stops_a_strip_before_the_reader_does():
    """The per-frame cap is not the only one. Enough medium frames add up past
    what the reader will accept in total, and the writer has to stop first."""
    each = PNG + "A" * (session.MAX_IMAGE_FRAME_BYTES - 1_000)
    n = (session.MAX_IMAGE_BYTES_TOTAL // len(each)) + 3
    strip = FakeStrip(frames=[FakeFrame(step=i, png=each) for i in range(n)])
    got = _reads(image_share.from_filmstrip(FakeStatus(), strip))
    total = sum(len(f["png"]) for f in got["frames"])
    assert total <= session.MAX_IMAGE_BYTES_TOTAL
    assert "larger than a `.mri` carries" in got["means"]


def test_an_oversized_readout_picture_is_dropped_and_the_readout_survives():
    """The rows are the measurement; the picture is what the boxes sit on. A
    photograph too big to travel must not take the readout down with it."""
    big = PNG + "A" * session.MAX_IMAGE_FRAME_BYTES
    pred = FakePrediction(boxes=[{"index": 1, "label": "cat", "score": 0.5}])
    got = _reads(image_share.from_readout(FakeStatus(), pred, picture=big))
    assert "frames" not in got
    assert len(got["readout"]["rows"]) == 1
    assert "above the" in got["means"] and "a `.mri` frame holds" in got["means"]


def test_the_writer_reads_the_reader_s_caps_rather_than_restating_them(
    monkeypatch,
):
    """THE DRIFT THIS GUARDS. Two constants with the same value in two files
    stay equal until somebody tunes one. Lowering the READER's cap has to
    change what the WRITER emits, or the next tuning ships a share button that
    makes files nobody can open."""
    monkeypatch.setattr(session, "MAX_IMAGE_FRAME_BYTES", len(PNG) + 10)
    strip = FakeStrip(frames=[FakeFrame(png=PNG + "A" * 500)])
    got = image_share.from_filmstrip(FakeStatus(), strip)
    assert got["frames"] == []
    assert "larger than a `.mri` carries" in got["means"]


# ------------------------------------- the machine's path never leaves with it


LOCAL = str(pathlib.Path(__file__).resolve().parent)


def test_a_local_checkpoint_path_is_scrubbed_from_the_readout_prose():
    """`_shared_name` protects the provenance FIELD, and for a while that was
    read as the whole job. It is not.

    `image_cv.Prediction.means` opens with the model NAME and the name it is
    handed is `status.repo` -- a Hub id for a Hub model and an ABSOLUTE PATH
    for one loaded out of a local folder. So the field said `sd-turbo` while
    the paragraph under it carried the whole path, in the one artefact in this
    project designed to leave the machine.
    """
    pred = FakePrediction(boxes=[{"index": 1, "label": "cat", "score": 0.5}])
    pred.means_text = f"{LOCAL} is a detector, read from the SHAPE of its output."
    got = _reads(image_share.from_readout(FakeStatus(repo=LOCAL), pred))
    assert LOCAL not in got["means"]
    assert LOCAL not in got["readout"]["means"]
    assert LOCAL not in got["provenance"]["repo"]
    # Replaced, not deleted: the sentence is about which checkpoint was
    # measured and is worth less with the name cut out of it.
    assert pathlib.Path(LOCAL).name in got["readout"]["means"]


def test_a_local_checkpoint_path_is_scrubbed_from_the_attention_prose():
    """The same leak through the other arm. `AttentionRun.means` names
    `self.model`, and `capture` is handed `status.repo` -- with a comment
    already warning that a drive letter must not ship.
    """

    class Local(FakeAttention):
        def to_dict(self) -> dict:
            out = FakeAttention.to_dict(self)
            out["means"] = f"Cross-attention from 1 denoising steps of {LOCAL}."
            return out

    got = _reads(image_share.from_attention(FakeStatus(repo=LOCAL), Local()))
    assert LOCAL not in got["attention"]["means"]
    assert pathlib.Path(LOCAL).name in got["attention"]["means"]


def test_a_hub_id_is_left_exactly_as_it_is():
    """The other half: a Hub id IS the name, so nothing needs saying and
    nothing is rewritten."""
    hub = "PixArt-alpha/PixArt-XL-2-512x512"
    said = f"{hub} is a detector."
    assert image_share._no_local_path(said, hub) == said


# -------------------------------- the prose counts the prompt, not the padding


def test_the_attention_sentence_counts_prompt_tokens_not_padded_columns():
    """`_tokenize` pads to the tokenizer's `model_max_length` -- 77 for CLIP --
    so the token list is the PADDED width. Counting it announced
    "cross-attention over 77 prompt token(s)" for a two-word prompt, which is
    the `<pad>` confusion the boundary exists to prevent, restated in prose.
    """

    class Padded(FakeAttention):
        def to_dict(self) -> dict:
            out = FakeAttention.to_dict(self)
            out["tokens"] = ["an", "astronaut", "<pad>", "<pad>", "<pad>"]
            out["steps"] = [
                {
                    "step": 0,
                    "timestep": 999.0,
                    "per_token": [0.4, 0.3, 0.1, 0.1, 0.1],
                    "blocks": 28,
                }
            ]
            out["padding_from"] = 2
            return out

    got = _reads(image_share.from_attention(FakeStatus(), Padded()))
    assert "2 prompt token(s)" in got["means"]
    assert "5 prompt token(s)" not in got["means"]
    # And the padded tail is REPORTED rather than dropped: it carries real
    # attention mass, which is the finding.
    assert "3 past the prompt are padding" in got["means"]


def test_an_unmeasured_boundary_makes_the_sentence_say_columns_not_words():
    """Reported, never guessed. A run whose boundary was not measured is not a
    run whose prompt happens to be exactly as long as the padding."""

    class NoBoundary(FakeAttention):
        def to_dict(self) -> dict:
            out = FakeAttention.to_dict(self)
            out.pop("padding_from")
            return out

    got = _reads(image_share.from_attention(FakeStatus(), NoBoundary()))
    assert got["attention"]["padding_from"] is None
    assert "conditioning column(s)" in got["means"]
    assert "was not measured" in got["means"]


# ------------------------------------------------- never emit what it refuses


def test_a_non_finite_score_is_dropped_rather_than_shipped_and_refused():
    """A head that produced NaN publishes a `float`: it passes a type check,
    and the reader refuses it for not being finite. So the share button
    answered 422 on a readout it had just made -- the exact failure this
    module exists to prevent."""
    pred = FakePrediction(
        boxes=[
            {"index": 1, "label": "cat", "score": 0.5},
            {"index": 2, "label": "dog", "score": float("nan")},
            {"index": 3, "label": "cow", "score": float("inf")},
        ]
    )
    got = _reads(image_share.from_readout(FakeStatus(), pred))
    assert [r["label"] for r in got["readout"]["rows"]] == ["cat"]
    # Reported, never only applied.
    assert "2 row(s) were left out" in got["means"]


def test_a_non_finite_box_is_dropped_rather_than_drawn_at_nan():
    pred = FakePrediction(
        boxes=[
            {
                "index": 1,
                "label": "cat",
                "score": 0.5,
                "box_xyxy": [1.0, float("nan"), 3.0, 4.0],
            }
        ]
    )
    got = _reads(image_share.from_readout(FakeStatus(), pred))
    assert got["readout"]["rows"][0]["box_xyxy"] is None


def test_a_readout_with_no_rows_is_refused_by_the_writer_with_a_next_step():
    """A detector that found nothing above the cut is a REAL answer on screen
    and an unshareable one: the file would carry a heading and nothing under
    it, and the reader says so by quoting the file format at somebody who
    asked for a download. Named here instead, with what to do about it."""
    empty = image_share.from_readout(FakeStatus(), FakePrediction(boxes=[]))
    why = image_share.refusal(empty)
    assert "no scored rows" in why
    assert "Lower the threshold" in why
    # And the reader would indeed have refused it, which is why this exists.
    with pytest.raises(BadRequest):
        _reads(empty)


def test_a_run_with_measurements_is_not_refused():
    made = image_share.from_filmstrip(FakeStatus(), FakeStrip())
    assert image_share.refusal(made) == ""
    assert "no image run to share" in image_share.refusal({})


def test_the_run_environment_is_recorded_and_never_reaches_the_file():
    """`/api/image/share` stamped the file's device and dtype off the LIVE
    handle, so loading a second checkpoint between the run and the share said
    the old model ran on the new one's hardware. Recorded at capture time
    instead -- and stripped on the way out, because `session.build` rebuilds
    the section from the fields the reader knows."""
    out = image_share.from_filmstrip(FakeStatus(), FakeStrip())
    assert out["_env"] == {"device": "cuda:0", "dtype": "float16"}
    assert "_env" not in _reads(out)


def test_the_padding_boundary_keeps_its_unknown_through_the_writer():
    """A writer that read an absent boundary as 0 emitted the claim that every
    measured column is padding."""

    class NoBoundary(FakeAttention):
        def to_dict(self) -> dict:
            out = FakeAttention.to_dict(self)
            out["padding_from"] = None
            return out

    got = _reads(image_share.from_attention(FakeStatus(), NoBoundary()))
    assert got["attention"]["padding_from"] is None
