"""Write a ModelMRI measurement into a timeline a roboticist already opens.

## The decision this must not overturn

`modelmri/session.py` and `POST /api/vla/share` in `modelmri/server.py` carry
the same sentence, and it is the reason the `.mri` format exists at all:

    There is no portable, no-account artifact for robot-policy internals
    anywhere: Foxglove archived its open-source Studio, Rerun's `.rrd`
    carries what the robot recorded rather than what the network computed,
    and HF Spaces need an upload and an account.

Read as a veto, that would forbid this module. Read as what it says, it
forbids something narrower and more useful: it forbids pretending an MCAP file
or a `.rrd` is where the internals live. `.rrd` carries *what the robot
recorded*; `.mri` carries *what the network computed*. Those are two jobs, and
this module does not merge them. It goes one way only — a strip of numbers
this tool measured, out into the viewer the reader already has open, with the
`.mri` named in the file as the place the rest of the finding stayed.

So the governing rule is the one `modelmri/circuit.py:8-11` states for the
reverse direction, run outwards:

    A rendered graph this tool did not compute must never be mistakable for
    one it did

Nothing ModelMRI writes into somebody else's container may be mistakable for
something that container's own tools measured. Provenance is a required field
of every file this module produces, not a caption a viewer may forget to draw:
`Provenance` has no defaults for the dataset or the tool, and `Timeline`
refuses to exist without one.

## What travels, and what stays behind

A timeline container holds *series over time*. It does not hold a 32x32
attention grid per layer, an occlusion map with its control band, or a set of
knockout bars — those are the `.mri` sections, and squeezing them through a
scalar channel would produce a plot that looks like a measurement and answers
no question anybody asked.

Which means every file this module writes is missing most of the finding, and
THAT is the failure mode worth engineering against. A reader who opens this in
Foxglove, sees four topics, and concludes those four were all ModelMRI had is
reading the file correctly and drawing the wrong conclusion. So the omissions
are not a docs problem: `Timeline.omitted` is written into the container as an
MCAP metadata record named `modelmri:not_exported`, in sentences, beside the
provenance. `omissions_for()` builds that list and names, specifically:

  * the frames BETWEEN the sampled ones. A sweep at stride 25 measured four
    frames of a hundred; the other ninety-six are absent, not zero. A Foxglove
    plot draws a gap either way, which is exactly why the stride has to be
    stated rather than inferred from the point spacing.
  * the EPISODES that were never opened. `vla_sweep.Sweep.means()` carries
    both halves of its own caveat — "THE STRIDE IS 25 FRAMES (and every 5
    episode)" — and a file that names the frame gap and not the episode gap
    tells a reader who counts its topics that the dataset has that many
    episodes. Same mistake, one level up.
  * the frames that FAILED to decode. `vla_sweep.Sweep.n_failed` counts them
    and they are absent from `rows` — a frame that would not decode is not a
    frame with a low score.
  * the coverage fraction ITSELF when the sweep published no frame total.
    Unknown does not collapse into silence: the file says the share is not
    stated rather than leaving the sentence out for the reader to fill in.
  * the grids, the maps and the bars, by name.
  * the weights. This file is a record of what was measured, not a way to
    measure it again.

## Unit and resolution ride on every sample

`modelmri/vla_actions.py:124` refuses to draw a policy's actions over a
dataset's when either side publishes no normalisation statistics — "two
unlabelled axes that happen to be the same length is precisely the case that
looks comparable and is not". A number crossing into a foreign viewer is that
case with the labels even further away, so:

  * a `Track` with a blank `unit` is REFUSED. There is no default unit.
  * a `Track` with a blank `resolution` is REFUSED. `RESOLUTION_UNSTATED` is
    the way to say "the measurement that produced this did not publish one" —
    a sentence, written into the file, rather than an omission. Unknown does
    not collapse into silence any more than it collapses into zero.
  * both strings are repeated in EVERY message body, not only in the channel
    metadata. Redundant on purpose: Foxglove's plot panel shows a channel's
    metadata when you go looking, and shows the message body when you hover a
    point, and the person about to misread the number is hovering. The cost of
    that redundancy is measured rather than assumed — `ExportPlan` reports it
    as `unit_overhead_bytes`.

    MEASURED, and re-measurable: `tests/test_robot_export.py::
    test_the_measured_overhead_in_the_module_docstring_is_reproducible` holds
    the exact inputs — a real `vla_sweep.Sweep`, two episodes of four frames
    at stride 25, unit "nats over the patch grid" (24 chars) and the stated
    resolution "entropy over a 16-patch grid, so a value is bounded by ln(16)
    = 2.7726 nats" (75 chars). On those: 1,000 bytes of a 1,709-byte payload,
    58.5%, which is 125 bytes per sample. The figure moves with the length of
    those two strings, which is why they are written down beside it — an
    earlier revision of this docstring published 920 of 1,630 (56.4%) with no
    record of the resolution string it was measured on, so nobody could check
    it. The share falls as the unit strings get shorter relative to the sample
    count and it is never free; the alternative is a bare float in a foreign
    tool, which is the thing this module exists to not produce.

## The clock

MCAP timestamps messages in nanoseconds, so writing one requires a clock, and
a dataset that publishes no frame rate does not have one. Defaulting to 30 fps
would be inventing a duration the reader would then measure off the axis.

`clock_for(fps)` therefore returns one of two clocks and says which in the
file: a real `seconds` clock when the dataset published its fps, or a
`frame-index` clock that renders one frame per second and carries the sentence
"this dataset published no frame rate, so nothing here knows how long a frame
took; do not read a duration off this axis". Both are written to the
`modelmri:clock` metadata record.

## What it costs before you spend it

`plan()` takes no dependency, touches no disk and returns the whole shape of
the file: channels, messages, the EXACT payload byte count, and an estimate of
the file on disk with its basis stated. Following `modelmri/budget.py`'s rule
that an estimate is labelled with where it came from, `estimated_file_bytes`
is payload + MCAP's fixed 31-byte message header + a 1 kB envelope, and it is
an UNCOMPRESSED, UNCHUNKED figure — the reference writer chunks with zstd by
default, so a real file is usually smaller. `write_mcap()` returns both the
estimate and the measured size so the estimate can be checked against the
thing it estimated, which is the only way it ever gets better.

The plan also carries `max_messages` and `over_cap`, because every cap this
module enforces is reported where the decision is taken: a plan that
described 600,000 messages without saying the write refuses above 500,000
would be describing a file that never exists, to a caller deciding whether to
spend the disk on it.

## Neither writer is installed here

Checked on this machine, 2026-08-24, against `C:/venvs/modelmri`:

    python -c "import mcap"        ModuleNotFoundError: No module named 'mcap'
    python -c "import rerun"       ModuleNotFoundError: No module named 'rerun'

So nothing here adds a dependency. `write_mcap()` is guarded exactly the way
`vla_data.encode_png` guards Pillow — an `ImportError` becomes a `Refusal`
naming the package and the command that installs it — and `plan()` works
regardless, because the plan is the useful half when the writer is missing:
it tells the reader what they would get before they install anything.

`write_rrd()` refuses unconditionally, and not only because `rerun` is absent.
Rerun's logging API is version-tied — `ROADMAP.md:378` already records that
"Rerun could not load v3.0 at all until a patch this year" — and this module
has never been run against an installed `rerun-sdk`. Emitting a `.rrd` from
code nobody has executed would be publishing an artifact whose correctness is
a guess, in a file format whose reader is pinned to the version that wrote it.
MCAP is the priority for the same reason in reverse: it is a documented open
container (https://mcap.dev), and `mcap_records()` builds the whole record
sequence as data, so the part of the writer this project owns is testable with
no writer installed at all.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import fmt
from .errors import BadRequest, Refusal

# Every topic this module writes is namespaced, so a reader who has merged
# this file with a recording from the robot can tell in the topic list which
# side each series came from. That is the same claim `circuit.py` makes with
# its banner, made where a Foxglove user actually looks.
# Seconds to wait for `rerun analytics config`. A privacy check that hangs
# is a export that hangs, and the answer on timeout is 'could not tell',
# which refuses.
ANALYTICS_TIMEOUT_S = 10

# Seconds to wait for Rerun's batching pipeline to reach the file. The SDK
# defaults to effectively forever; a writer that hangs is worse than one
# that raises, and the raise names the file.
RRD_FLUSH_TIMEOUT_S = 60.0

TOPIC_PREFIX = "modelmri/"

# MCAP's profile field names a well-known message-schema convention ("ros1",
# "ros2"). ModelMRI's messages are its own JSON, so the honest value is the
# empty string — claiming a profile would tell a reader's tooling to interpret
# these bodies under a schema convention they do not follow.
PROFILE = ""
MESSAGE_ENCODING = "json"
SCHEMA_ENCODING = "jsonschema"

# What a `Track` says when the measurement behind it published no resolution.
# Not `""` and not omitted: this string is written into the file, because a
# reader who cannot find a resolution field cannot tell "nobody stated one"
# from "the exporter dropped it".
RESOLUTION_UNSTATED = (
    "the measurement that produced this number published no resolution, so "
    "nothing here knows how many of these digits are real"
)

# Bytes of MCAP record framing per message, summed from the spec's Message
# record: opcode 1 + record length 8 + channel_id 2 + sequence 4 + log_time 8
# + publish_time 8 = 31, before the payload. Read from the format definition
# rather than measured, because there is no writer on this machine to measure
# it with — and it is stated here so the estimate below can be checked by
# anyone who has one.
MCAP_MESSAGE_HEADER_BYTES = 31

# The fixed cost of a file with no messages in it: magic bytes at both ends,
# the Header and Footer records, DataEnd, and a summary section holding one
# statistics record and the channel/schema indexes. Rounded UP to a round
# kilobyte and labelled an envelope rather than a measurement, for the same
# reason as above.
MCAP_ENVELOPE_BYTES = 1024

# Above this, the export is REFUSED with the count rather than truncated. A
# timeline missing its tail looks exactly like a timeline, which is the
# argument `vla_sweep.MAX_FRAMES` already makes about a ranking — and the
# consequence here is worse, because the reader is in a different application
# by then and has nothing of ours to read the caveat off.
MAX_MESSAGES = 500_000

# Which formats this module knows the shape of. "rrd" is planned and refused,
# not absent: a caller asking what a Rerun export would contain gets the full
# answer, and only the writing is declined.
FORMATS = ("mcap", "rrd")


# ------------------------------------------------------------------ the clock


@dataclass(frozen=True)
class Clock:
    """How a frame index becomes a timestamp, and whether that is a duration.

    Frozen for `receipts.Receipt`'s reason: this describes a decision already
    taken about a file already written, and a clock that can be edited after
    the fact is one that can be made to describe a recording nobody made.
    """

    #: "seconds" when the dataset published a frame rate, "frame-index" when
    #: it did not. Never a third value — `clock_for` is the only constructor.
    kind: str
    #: `None` for "this dataset published no frame rate", never 0.0. A zero
    #: fps would divide, and a default of 30 would invent the duration this
    #: whole class exists to avoid inventing.
    fps: float | None
    sentence: str

    def stamp(self, timestep: int) -> int:
        """Nanoseconds for a frame index, under this clock."""
        if self.kind == "seconds" and self.fps:
            return int(round(timestep / self.fps * 1e9))
        # One frame per second. Any mapping from an unclocked index to a time
        # axis is a fiction; this one is at least a legible fiction, and
        # `self.sentence` travels into the file to say so.
        return int(timestep) * 1_000_000_000

    def to_dict(self) -> dict:
        return {"kind": self.kind, "fps": self.fps, "sentence": self.sentence}


def clock_for(fps: float | None) -> Clock:
    """A clock from a published frame rate, or an honest substitute.

    `None` is the only way to say "this dataset published no frame rate" —
    a non-finite or non-positive fps is a `BadRequest`, because it is a value
    somebody supplied and got wrong, not an absence.

    The `bool` check runs BEFORE the numeric one because `isinstance(True, int)`
    is True: `float(True)` is 1.0, and a `True` that slipped out of a config
    parser would otherwise be published in the file as a measured rate of one
    frame per second, with the seconds axis claiming a duration nobody timed.
    """
    if isinstance(fps, bool):
        raise BadRequest(
            f"a frame rate of {fps!r} is a boolean, not a rate. `float({fps!r})` "
            f"would be {float(fps)}, which this file would then publish as a "
            f"measured frame rate and draw a seconds axis from. Pass the "
            f"dataset's fps, or pass None."
        )
    if fps is None:
        return Clock(
            kind="frame-index",
            fps=None,
            sentence=(
                "The time axis is FRAME INDEX, rendered one frame per second. "
                "This dataset published no frame rate, so nothing here knows "
                "how long a frame took — do not read a duration off this axis."
            ),
        )
    try:
        value = float(fps)
    except (TypeError, ValueError) as err:
        raise BadRequest(
            f"a frame rate of {fps!r} is not a number, so nothing here can "
            f"build a time axis from it. Pass the dataset's fps, or pass None "
            f"to get a frame-index axis that says in the file that no "
            f"duration is claimed."
        ) from err
    if not math.isfinite(value) or value <= 0:
        raise BadRequest(
            f"a frame rate of {fps!r} is not a clock. Pass the dataset's fps, "
            f"or pass None to get a frame-index axis that says in the file "
            f"that no duration is claimed."
        )
    return Clock(
        kind="seconds",
        fps=value,
        sentence=(
            f"The time axis is seconds, from this dataset's published rate of "
            f"{fmt.measured(value, 3)} frames per second. A sample sits at "
            f"frame_index / fps; the frames between samples were never "
            f"measured, so the gaps are absence and not zero."
        ),
    )


# ---------------------------------------------------------------- provenance


@dataclass(frozen=True)
class Provenance:
    """What produced these numbers, travelling inside somebody else's format.

    Every field is required at the call site. There are no defaults for the
    dataset or the tool because a default is how provenance goes missing: a
    file written with an empty `dataset` looks identical to one written before
    anybody thought about it, and by the time it is open in Foxglove there is
    nobody to ask.
    """

    tool: str
    tool_version: str
    dataset: str
    camera: str
    #: `None` for "no policy was resident when these numbers were measured",
    #: mirroring `vla_sweep.Sweep.policy`. Never `""` — a sweep read back out
    #: of sqlite months later has only this field to say what ran.
    policy: str | None
    #: The resolved commit, when the local cache could say. `None` is "the
    #: cache could not resolve one", which is different from the repo id and
    #: is written into the file as such.
    policy_revision: str | None
    #: The code that produced the numbers, e.g. "modelmri.vla_sweep.run".
    measured_by: str
    taken_at: str
    #: Where the rest of the finding stayed. The whole argument of this module
    #: is that the internals live in a `.mri`; naming the file here is what
    #: makes that recoverable from inside Foxglove.
    mri_pointer: str = ""

    def __post_init__(self) -> None:
        for name in ("tool", "tool_version", "dataset", "measured_by", "taken_at"):
            if not str(getattr(self, name) or "").strip():
                raise Refusal(
                    f"this export has no {name}, and a measurement written "
                    f"into another tool's format without one becomes "
                    f"indistinguishable from that tool's own. Fill it in, or "
                    f"do not write the file."
                )

    def to_metadata(self) -> dict[str, str]:
        """Flat string pairs, which is all an MCAP metadata record holds.

        Absent facts get a sentence rather than an empty value, so a reader
        scanning the metadata panel sees the state instead of a blank they
        will read as an oversight.
        """
        return {
            "tool": self.tool,
            "tool_version": self.tool_version,
            "dataset": self.dataset,
            "camera": self.camera or "not recorded",
            "policy": self.policy or "no policy was resident for this measurement",
            "policy_revision": (
                self.policy_revision
                or "the local cache could not resolve a commit for this policy"
            ),
            "measured_by": self.measured_by,
            "taken_at": self.taken_at,
            "internals_are_not_in_this_file": (
                self.mri_pointer
                or "this file carries series over time only; ModelMRI keeps the "
                "attention grids and causal maps in a .mri, and none was named "
                "when this file was written"
            ),
            "not_measured_by_this_viewer": (
                "Every number in this file was measured by ModelMRI, outside "
                "the application you are reading it in. Nothing here was "
                "computed by Foxglove, Rerun, or the robot."
            ),
        }


def now_stamp() -> str:
    """UTC, seconds resolution — the same shape `receipts.stamp` writes."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# -------------------------------------------------------------- the timeline


def _refuse_bool(value, what: str, because: str) -> None:
    """Refuse a `bool` where a number belongs, before any numeric check.

    `isinstance(True, int)` is True and `float(True)` is 1.0, so every bound
    check, every finiteness check and every `int()` downstream accepts a bool
    silently and publishes it as a measured 1. This runs first, everywhere a
    number crosses into the file.
    """
    if isinstance(value, bool):
        raise BadRequest(
            f"{what} is {value!r}, which is a boolean and not a measurement. "
            f"Python would read it as {int(value)} — {because}. A bool here is "
            f"a flag that reached a numeric field, so the number that belongs "
            f"there was never written down."
        )


@dataclass(frozen=True)
class Sample:
    """One measured value at one frame index."""

    timestep: int
    value: float


@dataclass(frozen=True)
class Track:
    """One measured quantity over one episode, with its unit and resolution.

    `frame_stride` is not decoration. A plot of four points over a hundred
    frames is the same picture whether ninety-six frames were skipped or
    ninety-six frames failed, and the reader is in another application with
    no way to ask. So the stride rides on the channel and in every message.
    """

    metric: str
    unit: str
    resolution: str
    episode: int
    samples: tuple[Sample, ...]
    frame_stride: int
    #: Frames in the episode, when the source published it. `None` means the
    #: source did not say — never 0, which would read as an empty episode.
    span: int | None = None
    note: str = ""

    @property
    def topic(self) -> str:
        return f"{TOPIC_PREFIX}{self.metric}/episode_{self.episode}"

    def channel_metadata(self) -> dict[str, str]:
        out = {
            "metric": self.metric,
            "unit": self.unit,
            "resolution": self.resolution,
            "episode": str(self.episode),
            "frame_stride": str(self.frame_stride),
            "samples": str(len(self.samples)),
            "episode_length_frames": (
                str(self.span)
                if self.span is not None
                else "not published by the source of this measurement"
            ),
            "sampling": (
                f"one frame in {self.frame_stride}. The frames between these "
                f"points were never measured and are ABSENT from this file, "
                f"not zero."
            ),
        }
        if self.note:
            out["note"] = self.note
        return out


@dataclass(frozen=True)
class Frame:
    """A camera frame, and whether it is the size the policy actually saw.

    `downsampled` exists because `vla.share_payload` shrinks a large frame and
    says so — "a silently shrunk frame under a causal map is a wrong picture".
    The same hazard survives the trip into Foxglove, where the image panel
    offers a pixel readout, so the note travels with the image.
    """

    png: bytes
    #: `None` for "the source did not publish a frame size", never 0. A 0x0
    #: in the channel metadata reads as a measured size, and a reader
    #: comparing it against the policy's input resolution would conclude the
    #: frame was empty rather than that nobody wrote the dimensions down.
    width: int | None
    height: int | None
    episode: int
    timestep: int
    downsampled: bool = False
    note: str = ""

    @property
    def topic(self) -> str:
        return f"{TOPIC_PREFIX}camera/episode_{self.episode}"


@dataclass(frozen=True)
class Timeline:
    """Everything one exported file will contain, before any writer exists.

    Constructed and validated in one step, so an invalid timeline cannot be
    held: a track with no unit, a non-finite value, or two tracks fighting
    over one topic are all refused here rather than at the writer, where the
    caller has already opened a file handle and half the work is on disk.

    Three more are refused here for the same reason, each of them a shape that
    used to produce a file:

      * a track with NO SAMPLES. It registers a channel and writes nothing to
        it, and an empty topic reads as a metric that was measured and came
        out empty. It is also the only way `n_messages` could reach zero.
      * a track whose topic collides with the CAMERA frame's. The image
        channel and the measurement channel share one topic namespace, so a
        metric called "camera" lands on `Frame.topic`; the writer keys its
        channels by topic and one would overwrite the other.
      * a bool or a negative index anywhere a number belongs. `log_time` is an
        unsigned field, and `isinstance(True, int)` is True.
    """

    provenance: Provenance
    clock: Clock
    tracks: tuple[Track, ...]
    frame: Frame | None = None
    #: Sentences naming what this file does NOT contain. Written into the
    #: container, never merely returned.
    omitted: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.tracks and self.frame is None:
            raise Refusal(
                "there is nothing to export: no track carries a sample and no "
                "camera frame was attached. An empty timeline in Foxglove "
                "reads as a policy that produced no measurements, which is a "
                "different claim from one nobody took."
            )
        seen: dict[str, str] = {}
        for track in self.tracks:
            _refuse_bool(
                track.episode,
                f"the episode of track {track.metric!r}",
                "it becomes part of the topic name and of every message body",
            )
            _refuse_bool(
                track.frame_stride,
                f"the frame stride of track {track.metric!r}",
                "it is the file's whole explanation of the gaps between its points",
            )
            if track.episode < 0:
                raise BadRequest(
                    f"the track {track.metric!r} reports episode "
                    f"{track.episode}, and there is no episode before the "
                    f"first one. modelmri/session.py refuses a negative "
                    f"episode for the same data; a topic named "
                    f"{track.topic!r} would carry the sign into the reader's "
                    f"topic list."
                )
            if not track.samples:
                raise Refusal(
                    f"the track {track.metric!r} on episode {track.episode} "
                    f"carries no samples, so it would register a channel and "
                    f"write nothing to it. An empty topic in Foxglove reads "
                    f"as a metric that was measured and came out empty, which "
                    f"is a different claim from one nobody took. Leave the "
                    f"track out, or give it the samples it stands for."
                )
            if not str(track.unit or "").strip():
                raise Refusal(
                    f"the track {track.metric!r} on episode {track.episode} "
                    f"has no unit. modelmri/vla_actions.py refuses to draw a "
                    f"policy's actions over a dataset's for exactly this "
                    f"reason — an unlabelled axis in a foreign viewer is the "
                    f"case that looks comparable and is not. State the unit, "
                    f"or do not export the track."
                )
            if not str(track.resolution or "").strip():
                raise Refusal(
                    f"the track {track.metric!r} on episode {track.episode} "
                    f"has no stated resolution. If the measurement behind it "
                    f"published none, pass robot_export.RESOLUTION_UNSTATED — "
                    f"that sentence is written into the file, and a reader "
                    f"who finds no resolution field cannot tell an absent one "
                    f"from a dropped one."
                )
            if track.frame_stride < 1:
                raise BadRequest(
                    f"the track {track.metric!r} on episode {track.episode} "
                    f"reports a frame stride of {track.frame_stride}, which "
                    f"cannot describe any sampling. It is what the file says "
                    f"about the gaps between its points."
                )
            if track.topic in seen:
                raise Refusal(
                    f"two tracks both want the topic {track.topic!r}. MCAP "
                    f"permits it and every viewer draws them on one plot with "
                    f"no way to tell which point came from which measurement."
                )
            seen[track.topic] = f"the measurements of {track.metric!r}"
            for sample in track.samples:
                where = f"{track.metric} at episode {track.episode}"
                # Bools first, both times: `isinstance(True, int)` is True, so
                # every check below would pass a `True` through and the file
                # would publish `"value":true` against a schema that declares
                # `value` a number.
                _refuse_bool(
                    sample.value,
                    f"the value of {where}",
                    "the message schema declares `value` a number, and JSON "
                    "renders a bool as `true`",
                )
                _refuse_bool(
                    sample.timestep,
                    f"the timestep of {where}",
                    "it is the frame index the clock turns into a timestamp",
                )
                if not math.isfinite(sample.value):
                    raise Refusal(
                        f"{where}, timestep "
                        f"{sample.timestep} is {sample.value!r}. A viewer "
                        f"draws that as a gap, which is what an unmeasured "
                        f"frame also looks like — leave it out and let the "
                        f"stride explain the gap, or fix the measurement."
                    )
                if sample.timestep < 0:
                    raise BadRequest(
                        f"{where} has timestep {sample.timestep}. MCAP's "
                        f"log_time is an unsigned 64-bit field and this clock "
                        f"would hand it {self.clock.stamp(sample.timestep):,}; "
                        f"modelmri/session.py refuses a negative timestep for "
                        f"the same data rather than wrapping it."
                    )
        if self.frame is not None:
            if not self.frame.png:
                raise Refusal(
                    "a camera frame was attached with no PNG bytes in it. An "
                    "empty image panel in Foxglove reads as a camera that "
                    "recorded nothing."
                )
            _refuse_bool(
                self.frame.episode,
                "the episode of the camera frame",
                "it becomes part of the topic name",
            )
            _refuse_bool(
                self.frame.timestep,
                "the timestep of the camera frame",
                "it is the frame index the clock turns into a timestamp",
            )
            if self.frame.episode < 0 or self.frame.timestep < 0:
                raise BadRequest(
                    f"the camera frame reports episode {self.frame.episode} "
                    f"at timestep {self.frame.timestep}, and MCAP's log_time "
                    f"is an unsigned field. A negative index here is a "
                    f"damaged source record, not a frame before the start."
                )
            # The duplicate-topic guard above is built from tracks only, and
            # the image channel shares the topic namespace with them: a track
            # whose metric is "camera" lands on exactly `Frame.topic`. Both
            # channels would then register on one topic, `channel_ids` is
            # keyed by topic, and the image channel would overwrite the
            # measurement channel — every measurement written under
            # `foxglove.CompressedImage`, the measurement channel empty.
            if self.frame.topic in seen:
                raise Refusal(
                    f"the camera frame and {seen[self.frame.topic]} both want "
                    f"the topic {self.frame.topic!r}. They carry different "
                    f"schemas, so one channel would overwrite the other and "
                    f"every measurement would be declared an image. Rename "
                    f"the metric — {TOPIC_PREFIX}camera/* belongs to the "
                    f"frame."
                )

    @property
    def n_messages(self) -> int:
        return sum(len(t.samples) for t in self.tracks) + (
            1 if self.frame is not None else 0
        )

    @property
    def n_channels(self) -> int:
        return len(self.tracks) + (1 if self.frame is not None else 0)


# --------------------------------------------------- building one from a sweep

#: The fields `timeline_from_sweep` reads off whatever it is handed. Checked
#: by name so a caller passing the wrong object gets a sentence naming the
#: field, rather than an AttributeError that reaches the browser as a 500.
_SWEEP_FIELDS = (
    "metric",
    "unit",
    "dataset",
    "policy",
    "camera",
    "frame_stride",
    "episode_stride",
    "rows",
    "n_failed",
    "n_frames",
    "frames_total",
)


_ROW_MISSING = object()


def _row_field(row, name: str, index: int):
    """One field off one sweep row, or a refusal naming the row and the field.

    There is no default here on purpose. `getattr(row, "value", 0.0)` reads
    beautifully and publishes a measurement nobody took: `Sweep.to_dict` uses
    `asdict` and `vla_sweep.stored()` hands back a list of DICTS, so a sweep
    round-tripped through sqlite or JSON has rows that answer no attribute at
    all — and the whole file would come out zeros stamped at timestep 0, which
    is exactly the "ABSENT rather than scored zero" claim this module is for,
    inverted. Mappings are read by key, everything else by attribute, and a
    row that can supply neither stops the export.
    """
    if isinstance(row, Mapping):
        value = row.get(name, _ROW_MISSING)
        how = "a mapping with no such key"
    else:
        value = getattr(row, name, _ROW_MISSING)
        how = f"a {type(row).__name__} with no such attribute"
    if value is _ROW_MISSING or value is None:
        raise BadRequest(
            f"row {index} of this sweep has no {name}: it is {how}. Every "
            f"sample this module writes carries a measured value at a measured "
            f"frame index, and a row that cannot supply one is not a zero at "
            f"timestep 0 — it is a row nothing here can read. "
            f"`modelmri.vla_sweep.stored()` returns dicts and "
            f"`Sweep.to_dict()` uses `asdict`, so check which shape the rows "
            f"arrived in."
        )
    return value


def _row_int(row, name: str, index: int) -> int:
    value = _row_field(row, name, index)
    _refuse_bool(value, f"the {name} of row {index}", "it indexes the timeline")
    try:
        return int(value)
    except (TypeError, ValueError) as err:
        raise BadRequest(
            f"row {index} of this sweep reports {name}={value!r}, which is "
            f"not a whole number. An index this module cannot read is not an "
            f"index it may guess."
        ) from err


def _row_float(row, name: str, index: int) -> float:
    value = _row_field(row, name, index)
    _refuse_bool(
        value,
        f"the {name} of row {index}",
        "it is the measurement itself, and the schema declares it a number",
    )
    try:
        return float(value)
    except (TypeError, ValueError) as err:
        raise BadRequest(
            f"row {index} of this sweep reports {name}={value!r}, which is "
            f"not a number. A measurement this module cannot read is not a "
            f"measurement it may replace with 0.0."
        ) from err


def _sweep_int(sweep, name: str) -> int | None:
    """A whole number off a sweep, or `None` for "the sweep did not say".

    `int(getattr(sweep, name, 0) or 0)` is the same fabrication as the row
    default one level up: a sweep that published no `frames_total` would get a
    coverage sentence computed against zero. `None` travels instead, and every
    caller below writes a sentence for it.
    """
    value = getattr(sweep, name, None)
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def timeline_from_sweep(
    sweep,
    *,
    fps: float | None,
    tool_version: str,
    resolution: str = RESOLUTION_UNSTATED,
    policy_revision: str | None = None,
    mri_pointer: str = "",
    episode_lengths: dict[int, int] | None = None,
    taken_at: str | None = None,
) -> Timeline:
    """A cross-episode sweep, reshaped into one channel per episode.

    A `vla_sweep.Sweep` is already a timeline in everything but its container:
    rows of (episode, timestep, value) with a stated unit and a stated stride.
    That is why it is the first thing wired up here — nothing has to be
    re-measured, and `THE SWEEP ADDS ITERATION, NOT MEASUREMENT` continues to
    hold across the export.

    `resolution` has no silent default: it defaults to the sentence saying
    none was published, which is written into the file. Pass the real one when
    the caller knows it — an entropy in nats over a 16-patch grid is bounded
    by ln(16), and a reader who knows the ceiling reads the number differently.
    """
    missing = [name for name in _SWEEP_FIELDS if not hasattr(sweep, name)]
    if missing:
        raise BadRequest(
            f"this does not look like a sweep: it has no "
            f"{', '.join(missing)}. `timeline_from_sweep` reads the fields "
            f"`modelmri.vla_sweep.Sweep` publishes."
        )

    stride = _sweep_int(sweep, "frame_stride")
    if stride is None:
        raise BadRequest(
            f"this sweep's frame_stride is {sweep.frame_stride!r}, which is "
            f"not a whole number of frames. The stride is what the exported "
            f"file says about the gaps between its points, so it cannot be "
            f"defaulted to 1 — that would describe a sampling nobody ran."
        )

    lengths = episode_lengths or {}
    by_episode: dict[int, list[Sample]] = {}
    for index, row in enumerate(sweep.rows):
        episode = _row_int(row, "episode", index)
        by_episode.setdefault(episode, []).append(
            Sample(
                timestep=_row_int(row, "timestep", index),
                value=_row_float(row, "value", index),
            )
        )

    tracks = []
    for episode in sorted(by_episode):
        # Sorted by time. `vla_sweep.run` ranks its rows by VALUE, which is
        # the right order for a table and produces a zig-zag in a plot: a
        # viewer joining points in arrival order draws a line that crosses
        # itself and means nothing.
        samples = tuple(sorted(by_episode[episode], key=lambda s: s.timestep))
        tracks.append(
            Track(
                metric=str(sweep.metric),
                unit=str(sweep.unit),
                resolution=resolution,
                episode=episode,
                samples=samples,
                frame_stride=stride,
                span=lengths.get(episode),
            )
        )

    return Timeline(
        provenance=Provenance(
            tool="ModelMRI",
            tool_version=tool_version,
            dataset=str(sweep.dataset),
            camera=str(sweep.camera or ""),
            policy=sweep.policy or None,
            policy_revision=policy_revision,
            measured_by="modelmri.vla_sweep.run",
            taken_at=taken_at or now_stamp(),
            mri_pointer=mri_pointer,
        ),
        clock=clock_for(fps),
        tracks=tuple(tracks),
        omitted=tuple(omissions_for(sweep=sweep, frame=None)),
    )


def frame_from_mri(section: dict) -> Frame:
    """Lift the camera frame out of a `.mri`'s `vla` section.

    The bridge that keeps the two formats in their own lanes: the `.mri` holds
    the attention grids and the causal map, and the timeline file carries the
    picture plus a pointer back. `session._vla` already refused the section
    without its provenance, so episode and timestep are present or the file
    would not have parsed.

    Refuses anything that is not a PNG data URL rather than guessing an
    encoding — a viewer handed JPEG bytes labelled `png` renders nothing and
    reports no error.
    """
    prov = section.get("provenance") or {}
    url = section.get("frame") or ""
    prefix = "data:image/png;base64,"
    if not isinstance(url, str) or not url.startswith(prefix):
        raise Refusal(
            "this .mri section carries no PNG camera frame. "
            "`vla_data.encode_png` writes a `data:image/png;base64,` URL and "
            "this field starts otherwise, so nothing here knows what encoding "
            "the bytes are in or what a viewer would do with them."
        )
    try:
        png = base64.b64decode(url[len(prefix) :], validate=True)
    except (binascii.Error, ValueError) as err:
        raise Refusal(
            "this .mri's camera frame is not valid base64, so there is no "
            "image to export. The file is damaged or was not written by "
            "ModelMRI."
        ) from err
    width, height = _frame_size(section.get("frame_size"))
    return Frame(
        png=png,
        width=width,
        height=height,
        episode=_mri_index(prov, "episode"),
        timestep=_mri_index(prov, "timestep"),
        downsampled=bool(section.get("frame_downsampled")),
        note=str(section.get("frame_note") or ""),
    )


def _frame_size(size) -> tuple[int | None, int | None]:
    """The frame's pixel size, or `(None, None)` for "the section did not say".

    A missing `frame_size` used to become `0 x 0` in the channel metadata,
    which a reader compares against the policy's input resolution and reads as
    a measured size. Zero is not a size anything was stored at.
    """
    if size is None:
        return (None, None)
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise Refusal(
            f"this .mri section reports frame_size={size!r}, which is not a "
            f"[width, height] pair. `vla_data` writes two whole numbers of "
            f"pixels; anything else is a damaged section, and a size guessed "
            f"here is written into the file as a measured one."
        )
    out: list[int] = []
    for name, raw in zip(("width", "height"), size, strict=True):
        _refuse_bool(raw, f"the frame {name}", "it is a count of pixels")
        try:
            pixels = int(raw)
        except (TypeError, ValueError) as err:
            raise Refusal(
                f"this .mri section reports a frame {name} of {raw!r}, which "
                f"is not a number of pixels. The section is damaged or was "
                f"not written by ModelMRI."
            ) from err
        if pixels <= 0:
            raise Refusal(
                f"this .mri section reports a frame {name} of {pixels}. No "
                f"frame was stored at that size, so this is a damaged "
                f"section — and written into the channel metadata it would "
                f"read as a measured dimension."
            )
        out.append(pixels)
    return (out[0], out[1])


def _mri_index(prov: dict, name: str) -> int:
    """`episode` / `timestep` off a `.mri` provenance block, or a refusal.

    `session._vla` refused the section without its provenance, so these are
    present in a file this project wrote. A file that reaches here without
    them is not one whose indices may be defaulted to 0 — a frame stamped
    episode 0, timestep 0 lands on a real topic beside real measurements.
    """
    value = prov.get(name)
    _refuse_bool(value, f"the camera frame's {name}", "it indexes the timeline")
    if value is None:
        raise Refusal(
            f"this .mri section's camera frame has no {name} in its "
            f"provenance, so nothing here knows which frame it is. Exported "
            f"as {name} 0 it would sit on a topic beside measurements taken "
            f"somewhere else entirely."
        )
    try:
        return int(value)
    except (TypeError, ValueError) as err:
        raise Refusal(
            f"this .mri section's camera frame reports {name}={value!r}, "
            f"which is not a frame index. The section is damaged or was not "
            f"written by ModelMRI."
        ) from err


def omissions_for(*, sweep=None, frame: Frame | None = None) -> list[str]:
    """What this file will not contain, in sentences, for writing into it.

    The list is deliberately written from the reader's position rather than
    the writer's: each entry says what they will not find and what it would
    be wrong to conclude from not finding it.
    """
    # Each multi-line sentence is PARENTHESISED. Adjacent string literals in a
    # list concatenate silently, and the overwhelmingly common cause of that
    # is a forgotten comma — so `py/implicit-string-concatenation-in-list`
    # reads this shape as a probable four-item list that lost one. Here the
    # joining is deliberate, and parentheses are how you say so: they turn a
    # pattern that is usually a bug into one that cannot be.
    out = [
        (
            "The per-layer attention grids are NOT in this file. ModelMRI "
            "measured them; a timeline channel holds a number per instant, "
            "not a 32x32 map per layer. They are in the .mri."
        ),
        (
            "The causal occlusion map and its control band are NOT in this "
            "file, for the same reason. A scalar summary of a causal map is "
            "not the map."
        ),
        "The input-stream knockout bars are NOT in this file.",
        (
            "The model weights and the dataset are NOT in this file. It is a "
            "record of what was measured, not a way to measure it again."
        ),
    ]
    if frame is None:
        out.append(
            "No camera frame is in this file. The numbers were measured on "
            "frames that exist; none of them travelled."
        )
    else:
        if frame.downsampled:
            out.append(
                "The camera frame in this file was DOWNSAMPLED before it was "
                "stored. " + (frame.note or "Do not measure pixel coordinates off it.")
            )
    if sweep is not None:
        stride = _sweep_int(sweep, "frame_stride")
        n_frames = _sweep_int(sweep, "n_frames")
        total = _sweep_int(sweep, "frames_total")
        episode_stride = _sweep_int(sweep, "episode_stride")
        if stride is not None and stride > 1:
            out.append(
                f"Only one frame in {stride} was measured. The other "
                f"{stride - 1} of every {stride} are ABSENT from this file, "
                f"not zero — a gap in these plots is a frame nobody looked at."
            )
        if episode_stride is not None and episode_stride > 1:
            # `vla_sweep.Sweep.means()` carries this caveat — "THE STRIDE IS
            # 25 FRAMES (and every 5 episode)" — and a file that names the
            # frame gap but not the episode gap tells a reader who counts the
            # topics that the dataset has that many episodes.
            out.append(
                f"Only one episode in {episode_stride} was opened. The other "
                f"{episode_stride - 1} of every {episode_stride} episodes "
                f"were never measured and are ABSENT from this file — the "
                f"episodes you see here are not all the episodes there are."
            )
        if total is not None and total > 0 and n_frames is not None:
            out.append(
                f"This covers {n_frames:,} of {total:,} frames "
                f"({100.0 * n_frames / total:.1f}%). The extreme value in "
                f"this file is the extreme of what was SAMPLED, which is not "
                f"the same claim as the extreme of the dataset."
            )
        else:
            # Unknown does not collapse into silence any more than it
            # collapses into zero: with no total there is no coverage
            # fraction, and leaving the sentence out lets the reader supply
            # their own.
            out.append(
                "What share of the dataset this covers is NOT stated: the "
                "sweep behind this file published no frame total, so nothing "
                "here knows how much was left unmeasured."
            )
        n_failed = _sweep_int(sweep, "n_failed")
        if n_failed:
            out.append(
                f"{n_failed:,} frame(s) could not be measured and are ABSENT "
                f"rather than scored zero. A frame that failed to decode is "
                f"not a frame with a low score, and in a plot the two are the "
                f"same gap."
            )
    return out


# ----------------------------------------------------- the record sequence


@dataclass(frozen=True)
class Record:
    """One instruction for a timeline writer: metadata, schema, channel, message.

    The whole file is built as a list of these BEFORE any writer is imported,
    which is what makes the part this project owns testable on a machine with
    no `mcap` installed. `write_mcap` is then a replay loop with no decisions
    left in it.
    """

    kind: str
    name: str = ""
    topic: str = ""
    schema: str = ""
    encoding: str = ""
    data: bytes = b""
    metadata: dict = field(default_factory=dict)
    log_time_ns: int = 0

    @property
    def nbytes(self) -> int:
        return len(self.data)


#: The message body every measurement channel carries. Written into the file
#: as a JSON Schema so a viewer can type the fields — and so `unit` and
#: `resolution` are declared REQUIRED, which is the machine-readable form of
#: this module's central rule.
MEASUREMENT_SCHEMA = {
    "type": "object",
    "title": "modelmri.Measurement",
    "description": (
        "One number ModelMRI measured, with the unit and resolution it was "
        "measured in. Not computed by the application displaying it."
    ),
    "properties": {
        "value": {"type": "number"},
        "unit": {"type": "string"},
        "resolution": {"type": "string"},
        "metric": {"type": "string"},
        "episode": {"type": "integer"},
        "timestep": {"type": "integer"},
        "frame_stride": {"type": "integer"},
    },
    "required": ["value", "unit", "resolution", "metric", "episode", "timestep"],
}

#: Foxglove renders `foxglove.CompressedImage` natively, so the frame arrives
#: as a picture rather than as a base64 blob in a raw-message panel. The
#: schema name is theirs; the bytes and the caveat are ours.
IMAGE_SCHEMA_NAME = "foxglove.CompressedImage"
IMAGE_SCHEMA = {
    "type": "object",
    "title": IMAGE_SCHEMA_NAME,
    "properties": {
        "timestamp": {
            "type": "object",
            "properties": {
                "sec": {"type": "integer"},
                "nsec": {"type": "integer"},
            },
        },
        "frame_id": {"type": "string"},
        "data": {"type": "string", "contentEncoding": "base64"},
        "format": {"type": "string"},
    },
    "required": ["timestamp", "frame_id", "data", "format"],
}


def _pixels(value: int | None) -> str:
    if value is None:
        return "not published by the .mri section this frame came from"
    return str(value)


def _json_bytes(payload: dict) -> bytes:
    # `separators` because every byte here is multiplied by the message count,
    # and `sort_keys` because a byte estimate that moves with dict ordering is
    # not an estimate anybody can check twice.
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def mcap_records(timeline: Timeline) -> list[Record]:
    """The complete record sequence for one MCAP file, as data.

    Order matters and is the writer's contract: metadata first so a reader
    scanning the head of the file meets the provenance and the omissions
    before any number, then schemas, then channels, then messages in time
    order per channel.
    """
    records: list[Record] = [
        Record(
            kind="metadata",
            name="modelmri:provenance",
            metadata=timeline.provenance.to_metadata(),
        ),
        Record(
            kind="metadata",
            name="modelmri:clock",
            metadata={
                "kind": timeline.clock.kind,
                "fps": (
                    fmt.measured(timeline.clock.fps, 3)
                    if timeline.clock.fps is not None
                    else "not published by this dataset"
                ),
                "sentence": timeline.clock.sentence,
            },
        ),
    ]
    if timeline.omitted:
        # Numbered rather than joined into one paragraph: Foxglove's metadata
        # panel is a key/value table, and a single 900-character value renders
        # as one unreadable cell.
        records.append(
            Record(
                kind="metadata",
                name="modelmri:not_exported",
                metadata={
                    f"{index:02d}": sentence
                    for index, sentence in enumerate(timeline.omitted, start=1)
                },
            )
        )

    if timeline.tracks:
        records.append(
            Record(
                kind="schema",
                name="modelmri.Measurement",
                encoding=SCHEMA_ENCODING,
                data=_json_bytes(MEASUREMENT_SCHEMA),
            )
        )
    if timeline.frame is not None:
        records.append(
            Record(
                kind="schema",
                name=IMAGE_SCHEMA_NAME,
                encoding=SCHEMA_ENCODING,
                data=_json_bytes(IMAGE_SCHEMA),
            )
        )

    for track in timeline.tracks:
        records.append(
            Record(
                kind="channel",
                topic=track.topic,
                schema="modelmri.Measurement",
                encoding=MESSAGE_ENCODING,
                metadata=track.channel_metadata(),
            )
        )
    if timeline.frame is not None:
        frame = timeline.frame
        records.append(
            Record(
                kind="channel",
                topic=frame.topic,
                schema=IMAGE_SCHEMA_NAME,
                encoding=MESSAGE_ENCODING,
                metadata={
                    # `None` gets the sentence, not a 0. A "0" here reads as a
                    # measured dimension to anybody comparing it against the
                    # resolution the policy saw.
                    "width": _pixels(frame.width),
                    "height": _pixels(frame.height),
                    "downsampled": "true" if frame.downsampled else "false",
                    "note": frame.note
                    or (
                        "stored at the resolution the policy saw"
                        if not frame.downsampled
                        else "this frame was resized before storage"
                    ),
                    "measured_by": "not measured — this is the input, not a result",
                },
            )
        )

    for track in timeline.tracks:
        for sample in track.samples:
            records.append(
                Record(
                    kind="message",
                    topic=track.topic,
                    log_time_ns=timeline.clock.stamp(sample.timestep),
                    data=_json_bytes(
                        {
                            "value": sample.value,
                            "unit": track.unit,
                            "resolution": track.resolution,
                            "metric": track.metric,
                            "episode": track.episode,
                            "timestep": sample.timestep,
                            "frame_stride": track.frame_stride,
                        }
                    ),
                )
            )
    if timeline.frame is not None:
        frame = timeline.frame
        stamp = timeline.clock.stamp(frame.timestep)
        records.append(
            Record(
                kind="message",
                topic=frame.topic,
                log_time_ns=stamp,
                data=_json_bytes(
                    {
                        "timestamp": {
                            "sec": stamp // 1_000_000_000,
                            "nsec": stamp % 1_000_000_000,
                        },
                        "frame_id": frame.topic,
                        "data": base64.b64encode(frame.png).decode("ascii"),
                        "format": "png",
                    }
                ),
            )
        )
    return records


def unit_overhead_bytes(timeline: Timeline) -> int:
    """What repeating unit and resolution in every message body costs.

    Measured, not assumed: the same bodies are re-serialised without the two
    strings and the difference is returned. It is reported in the plan because
    the redundancy is a deliberate trade — see the module docstring — and a
    deliberate trade with an unmeasured cost is a preference.
    """
    total = 0
    for track in timeline.tracks:
        lean = _json_bytes(
            {
                "value": 0.0,
                "metric": track.metric,
                "episode": track.episode,
                "timestep": 0,
                "frame_stride": track.frame_stride,
            }
        )
        full = _json_bytes(
            {
                "value": 0.0,
                "unit": track.unit,
                "resolution": track.resolution,
                "metric": track.metric,
                "episode": track.episode,
                "timestep": 0,
                "frame_stride": track.frame_stride,
            }
        )
        total += (len(full) - len(lean)) * len(track.samples)
    return total


def mean_message_bytes(payload: int, count: int) -> float | None:
    """The mean payload of `count` messages, or `None` when there are none.

    A separate function so the rule is reachable and testable on its own.
    `Timeline` now refuses an export with no messages in it — an empty track
    and an empty timeline are both stopped at construction — so `plan()` never
    reaches `count == 0` from a valid timeline. The branch stays and is tested
    here rather than deleted, because the rule it holds is the module's: a
    mean of nothing is not 0.0 bytes. 0.0 would be a message size somebody
    could quote, computed from no messages.
    """
    if count <= 0:
        return None
    return round(payload / count, 1)


# ------------------------------------------------------------------ the plan


@dataclass(frozen=True)
class ExportPlan:
    """The whole shape of the file, before a writer exists to write it.

    Answers "how large will this be, and what will be in it" without importing
    anything optional, which is the useful half on a machine that has neither
    writer installed — the reader finds out what they would get before they
    install anything. `budget.py`'s rule applies: `payload_bytes` is exact and
    `estimated_file_bytes` carries the basis it was computed from.
    """

    container: str
    writer_package: str
    writer_available: bool
    writer_reason: str
    n_channels: int
    n_messages: int
    n_metadata_records: int
    #: EXACT. The sum of the message payloads this module hands the writer.
    payload_bytes: int
    estimated_file_bytes: int
    estimated_basis: str
    #: `None` when there is nothing to average, never 0.0.
    mean_message_bytes: float | None
    unit_overhead_bytes: int
    tracks_without_resolution: int
    #: The cap `write_mcap` enforces, and whether this timeline is over it.
    #: Reported here because the plan is documented as the whole shape of the
    #: file: a plan that describes 600,000 messages and does not mention that
    #: the write will refuse has answered "what will be in it" with something
    #: that will never exist.
    max_messages: int
    over_cap: bool
    omitted: tuple[str, ...]
    clock: dict
    provenance: dict

    def to_dict(self) -> dict:
        out = {
            "container": self.container,
            "writer_package": self.writer_package,
            "writer_available": self.writer_available,
            "writer_reason": self.writer_reason,
            "n_channels": self.n_channels,
            "n_messages": self.n_messages,
            "n_metadata_records": self.n_metadata_records,
            "payload_bytes": self.payload_bytes,
            "estimated_file_bytes": self.estimated_file_bytes,
            "estimated_basis": self.estimated_basis,
            "mean_message_bytes": self.mean_message_bytes,
            "unit_overhead_bytes": self.unit_overhead_bytes,
            "tracks_without_resolution": self.tracks_without_resolution,
            "max_messages": self.max_messages,
            "over_cap": self.over_cap,
            "omitted": list(self.omitted),
            "clock": self.clock,
            "provenance": self.provenance,
        }
        out["means"] = self.sentence()
        return out

    def sentence(self) -> str:
        parts = [
            f"{self.n_messages:,} message(s) across {self.n_channels} "
            f"channel(s), {fmt.bytes_si(self.payload_bytes)} of payload "
            f"exactly. On disk, about {fmt.bytes_si(self.estimated_file_bytes)} "
            f"— {self.estimated_basis}"
        ]
        if self.unit_overhead_bytes:
            share = (
                100.0 * self.unit_overhead_bytes / self.payload_bytes
                if self.payload_bytes
                else 0.0
            )
            parts.append(
                f"{fmt.bytes_si(self.unit_overhead_bytes)} of that "
                f"({share:.1f}%) is the unit and resolution repeated in every "
                f"message, which is what stops a number being read as a bare "
                f"float in a foreign viewer."
            )
        if self.tracks_without_resolution:
            parts.append(
                f"{self.tracks_without_resolution} track(s) carry no stated "
                f"resolution — the file says so per track rather than leaving "
                f"the field out."
            )
        if self.over_cap:
            parts.append(
                f"NOTHING WILL BE WRITTEN: {self.n_messages:,} messages is "
                f"over the {self.max_messages:,}-message cap, and the write "
                f"refuses rather than truncating — a timeline missing its "
                f"tail looks exactly like a timeline. Raise the sweep's "
                f"stride."
            )
        if not self.writer_available:
            parts.append(f"NOTHING WILL BE WRITTEN: {self.writer_reason}")
        if self.omitted:
            parts.append(
                f"{len(self.omitted)} statement(s) about what is NOT in this "
                f"file travel inside it."
            )
        return " ".join(parts)


def rerun_cli() -> Path | None:
    """The `rerun` binary that ships INSIDE the installed wheel, or None.

    Located from `rerun.__file__` rather than from PATH or a literal path: the
    wheel bundles its own CLI beside the Python package, and that is the one
    whose version matches the SDK doing the writing. A `rerun` on PATH could be
    a different build with a different analytics config, which would make the
    check below answer a question about the wrong program.
    """
    try:
        import rerun
    except ImportError:
        return None
    root = Path(rerun.__file__).resolve().parent.parent / "rerun_cli"
    for name in ("rerun.exe", "rerun"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def rerun_analytics() -> tuple[bool | None, str]:
    """Whether rerun's usage analytics are on, asked of rerun itself.

    Returns `(enabled, detail)`. **`None` means could not determine, and it is
    a distinct answer from False.** Treating "we could not ask" as "it is off"
    would be the `?? 0` bug pointed at a privacy promise: the caller would get
    silence where it needed a no.

    Asked by running `rerun analytics config`, which prints its own JSON, so
    the config path is never hardcoded here. It moves between platforms and it
    is rerun's to move.
    """
    cli = rerun_cli()
    if cli is None:
        return None, "the rerun CLI that ships with the wheel was not found"
    try:
        done = subprocess.run(
            [str(cli), "analytics", "config"],
            capture_output=True,
            text=True,
            timeout=ANALYTICS_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as err:
        return None, (
            f"`rerun analytics config` could not be run ({type(err).__name__})"
        )
    if done.returncode != 0:
        return None, f"`rerun analytics config` exited {done.returncode}"
    # The command prints JSON on stdout, but a first run prints a welcome
    # banner too, so the object is found rather than assumed to be the whole
    # of stdout.
    text = done.stdout
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None, "`rerun analytics config` printed no JSON object"
    try:
        config = json.loads(text[start : end + 1])
    except json.JSONDecodeError as err:
        return None, (
            f"`rerun analytics config` printed unreadable JSON ({type(err).__name__})"
        )
    value = config.get("analytics_enabled")
    if not isinstance(value, bool):
        return None, "`rerun analytics config` did not report analytics_enabled"
    return value, f"reported by {cli.name} in {config.get('config_file_path', '?')}"


def writer_available(container: str) -> tuple[tuple[bool, str], str]:
    """Whether this machine can write `format`, and which package would.

    Returns `((available, reason), package)`. `reason` is `""` when available,
    following `vla_data.frames_readable`'s rule that the caller gets one field
    to test rather than two that can disagree.
    """
    if container == "mcap":
        try:
            import mcap  # noqa: F401
            from mcap.writer import Writer  # noqa: F401
        except ImportError as err:
            return (
                False,
                (
                    f"Writing MCAP needs the `mcap` package, which is not "
                    f"installed on this machine ({err.name or 'mcap'} is "
                    f"missing). Install it with `pip install mcap`. It is not "
                    f"a ModelMRI dependency: this export exists for people who "
                    f"already run Foxglove, and everyone else should not pay "
                    f"for it. The plan beside this sentence is real and was "
                    f"computed without it."
                ),
            ), "mcap"
        return (True, ""), "mcap"

    if container == "rrd":
        try:
            import rerun  # noqa: F401
        except ImportError:
            return (
                False,
                (
                    "Writing .rrd needs the `rerun-sdk` package, which is not "
                    "installed on this machine. Install it with `pip install "
                    "rerun-sdk`. It is not a ModelMRI dependency: this export "
                    "exists for people who already run Rerun, and everyone "
                    "else should not pay 134 MB for it. The plan beside this "
                    "sentence is real and was computed without it."
                ),
            ), "rerun-sdk"

        # THE REFUSAL THAT REPLACED "we have never run this".
        #
        # We have now. rerun-sdk 0.36.3 writes a file `rerun rrd verify` loads
        # without error, and the round trip is a test. What is left is not a
        # doubt about correctness, it is a promise this project made:
        # ModelMRI has no telemetry and says so on its front page. rerun ships
        # analytics ENABLED BY DEFAULT — measured on this machine, a first run
        # created a persistent analytics id under
        # AppData/Roaming/rerun/config/analytics.json and said so on stderr.
        #
        # Writing an .rrd through a library that phones home would make that
        # promise false for anyone who used this button, and they would have no
        # way to know. So the refusal is conditional and it names the exact
        # command that clears it, rather than being a blanket no.
        enabled, detail = rerun_analytics()
        if enabled is not False:
            unknown = enabled is None
            return (
                False,
                (
                    (
                        "Cannot tell whether rerun's usage analytics are on "
                        f"({detail}), and an unknown here is not a no. "
                        if unknown
                        else f"rerun's usage analytics are ENABLED ({detail}). "
                    )
                    + "ModelMRI has no telemetry and says so on its front "
                    "page; writing this file through a library that reports "
                    "usage would make that false for you without your "
                    "knowing. Run `rerun analytics disable` once and this "
                    "export works — it is rerun's own command and it is "
                    "machine-wide. Or write the MCAP instead, which needs no "
                    "such thing: it is a documented open container and the "
                    "plan beside this sentence describes exactly what it "
                    "would contain."
                ),
            ), "rerun-sdk"

        return (True, ""), "rerun-sdk"

    raise BadRequest(
        f"unknown export container {container!r} — expected one of {list(FORMATS)}"
    )


def plan(timeline: Timeline, *, container: str = "mcap") -> ExportPlan:
    """What the file will contain and roughly weigh. Touches no disk."""
    (available, reason), package = writer_available(container)
    records = mcap_records(timeline)
    messages = [r for r in records if r.kind == "message"]
    payload = sum(r.nbytes for r in messages)
    metadata_records = [r for r in records if r.kind == "metadata"]
    estimated = (
        payload + MCAP_MESSAGE_HEADER_BYTES * len(messages) + MCAP_ENVELOPE_BYTES
    )
    return ExportPlan(
        container=container,
        writer_package=package,
        writer_available=available,
        writer_reason=reason,
        n_channels=timeline.n_channels,
        n_messages=len(messages),
        n_metadata_records=len(metadata_records),
        payload_bytes=payload,
        estimated_file_bytes=estimated,
        estimated_basis=(
            f"payload plus MCAP's {MCAP_MESSAGE_HEADER_BYTES}-byte message "
            f"header and a {MCAP_ENVELOPE_BYTES}-byte envelope for the header, "
            f"footer and summary index. UNCOMPRESSED and unchunked: the "
            f"reference writer chunks with zstd by default, so a real file is "
            f"usually smaller. This is a figure to compare against a disk "
            f"budget, not a size to promise anybody."
        ),
        mean_message_bytes=mean_message_bytes(payload, len(messages)),
        unit_overhead_bytes=unit_overhead_bytes(timeline),
        tracks_without_resolution=sum(
            1 for t in timeline.tracks if t.resolution == RESOLUTION_UNSTATED
        ),
        max_messages=MAX_MESSAGES,
        over_cap=len(messages) > MAX_MESSAGES,
        omitted=tuple(timeline.omitted),
        clock=timeline.clock.to_dict(),
        provenance=timeline.provenance.to_metadata(),
    )


# ---------------------------------------------------------------- the writers


def write_mcap(timeline: Timeline, path: str | Path) -> dict:
    """Write the timeline as MCAP, or refuse and name the install command.

    The replay loop below holds no decisions — every one of them was taken in
    `mcap_records`, which is testable with no writer installed. What is NOT
    verified here is the bytes: this project has no `mcap` on any machine it
    has run on, so the record sequence is checked and the container is not.
    The receipt reports the measured size beside the estimate for exactly that
    reason: the first person to run this with `mcap` installed finds out
    immediately whether the estimate was any good.
    """
    (available, reason), _ = writer_available("mcap")
    if not available:
        raise Refusal(reason)

    shape = plan(timeline, container="mcap")
    if shape.n_messages > MAX_MESSAGES:
        raise Refusal(
            f"that is {shape.n_messages:,} messages and the cap is "
            f"{MAX_MESSAGES:,}. Raise the sweep's stride rather than having "
            f"the export cut short: a timeline missing its tail looks exactly "
            f"like a timeline, and by the time it is open in Foxglove there "
            f"is nothing of ours left to say so."
        )

    from mcap.writer import Writer

    from . import __version__

    records = mcap_records(timeline)
    destination = Path(path)
    schema_ids: dict[str, int] = {}
    channel_ids: dict[str, int] = {}
    sequence: dict[str, int] = {}

    with open(destination, "wb") as stream:
        writer = Writer(stream)
        writer.start(profile=PROFILE, library=f"modelmri {__version__}")
        for record in records:
            if record.kind == "metadata":
                writer.add_metadata(record.name, record.metadata)
            elif record.kind == "schema":
                schema_ids[record.name] = writer.register_schema(
                    name=record.name,
                    encoding=record.encoding,
                    data=record.data,
                )
            elif record.kind == "channel":
                channel_ids[record.topic] = writer.register_channel(
                    topic=record.topic,
                    message_encoding=record.encoding,
                    schema_id=schema_ids[record.schema],
                    metadata=record.metadata,
                )
            elif record.kind == "message":
                index = sequence.get(record.topic, 0)
                sequence[record.topic] = index + 1
                writer.add_message(
                    channel_id=channel_ids[record.topic],
                    log_time=record.log_time_ns,
                    publish_time=record.log_time_ns,
                    sequence=index,
                    data=record.data,
                )
        writer.finish()

    written = destination.stat().st_size
    if written == 0:
        # A zero-byte file that nothing raised on is the failure this whole
        # project is about: the export "succeeded", the panel says so, and
        # Foxglove opens nothing. Named here rather than handed back as a
        # receipt with a 0 in it — and the empty file goes with it, because a
        # refusal that says "nothing was written" while leaving an .mcap on
        # disk hands the reader the artifact it just disowned.
        removed = True
        try:
            destination.unlink()
        except OSError:
            removed = False
        raise Refusal(
            f"the MCAP writer produced a zero-byte file at {destination.name}. "
            f"Nothing was written and nothing raised, so the installed `mcap` "
            f"is not behaving the way this module drives it — check its "
            f"version before trusting any file it wrote. "
            + (
                "The empty file has been removed."
                if removed
                else f"The empty file is still at {destination} and could not "
                f"be removed — delete it before anyone opens it."
            )
        )

    return {
        "path": str(destination),
        "container": "mcap",
        "bytes_written": written,
        "bytes_estimated": shape.estimated_file_bytes,
        # The estimate divided by what actually happened. Reported rather than
        # asserted: an estimate nobody checks is a number that never improves.
        "estimate_over_actual": round(shape.estimated_file_bytes / written, 3),
        "n_messages": shape.n_messages,
        "n_channels": shape.n_channels,
        "n_metadata_records": shape.n_metadata_records,
        "plan": shape.to_dict(),
        "means": (
            f"{shape.n_messages:,} measurement(s) written to "
            f"{destination.name} ({fmt.bytes_si(written)}). Every one carries "
            f"its unit and resolution, and the file states what ModelMRI "
            f"measured that did NOT travel into it."
        ),
    }


def rrd_entity(track: Track) -> str:
    """Where one track's series lands in the Rerun entity tree.

    Namespaced under `TOPIC_PREFIX` for the same reason every MCAP topic is:
    somebody who merges this with a recording from the robot has to be able to
    tell, in the entity list, which side each series came from.
    """
    return f"{TOPIC_PREFIX}episode_{track.episode}/{track.metric}"


def _rrd_provenance(timeline: Timeline, version: str) -> str:
    """The provenance block, as markdown Rerun will render.

    The same content the MCAP writer puts in metadata records. Markdown rather
    than JSON because Rerun renders a TextDocument, and the person who opens
    the file is who this is for.
    """
    p = timeline.provenance
    policy = f"- **policy** - {p.policy or 'not recorded'}"
    if p.policy_revision:
        policy += f" @ {p.policy_revision}"
    lines = [
        "# ModelMRI export",
        "",
        f"- **tool** - {p.tool} {p.tool_version}",
        f"- **written by** - rerun-sdk {version} (an .rrd is read by the "
        "version that wrote it)",
        f"- **dataset** - {p.dataset}",
        f"- **camera** - {p.camera}",
        policy,
        f"- **measured by** - {p.measured_by}",
        f"- **taken at** - {p.taken_at}",
        f"- **clock** - {timeline.clock.sentence}",
    ]
    if p.mri_pointer:
        lines.append(f"- **.mri** - {p.mri_pointer}")
    if timeline.omitted:
        lines += ["", "## Not in this file", ""]
        lines += [f"- {sentence}" for sentence in timeline.omitted]
    return chr(10).join(lines)


def write_rrd(timeline: Timeline, path: str | Path) -> dict:
    """Write the timeline as a Rerun `.rrd`, or refuse and name the fix.

    This function refused unconditionally until 2026-08-27, on the grounds that
    "nothing here has ever been run against an installed rerun-sdk, so emitting
    one would be publishing a file whose correctness is a guess." That was true
    and it is no longer: rerun-sdk 0.36.3 writes a file that `rerun rrd verify`
    loads without error, and the test beside this runs that round trip rather
    than asserting the bytes look plausible.

    What did NOT go away is the version tie, so it is PUBLISHED instead of
    argued about. An `.rrd` is read by the Rerun version that wrote it, and the
    API moves under it: `rr.set_time_sequence`, which nearly every example
    still uses, does not exist in 0.36.3 - the call below is
    `rr.set_time(..., sequence=...)`. So the version goes in the receipt AND
    into the file, because a reader who cannot open it needs to know which
    version to install and the receipt will be long gone.

    The static `SeriesLines` per track is not decoration. With no name the
    viewer labels a series by its entity path, so the UNIT - the thing that
    makes the number mean anything - would appear nowhere on the plot.
    """
    (available, reason), _ = writer_available("rrd")
    if not available:
        raise Refusal(reason)

    import rerun as rr

    shape = plan(timeline, container="rrd")
    if shape.n_messages > MAX_MESSAGES:
        raise Refusal(
            f"that is {shape.n_messages:,} messages and the cap is "
            f"{MAX_MESSAGES:,}. Raise the sweep's stride rather than having "
            f"the export cut short: a timeline missing its tail looks exactly "
            f"like a timeline, and by the time it is open in Rerun there is "
            f"nothing of ours left to say so."
        )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    version = getattr(rr, "__version__", "unknown")

    # A recording id derived from the run rather than a fresh uuid per call, so
    # two exports of the same measurement are the same recording to a reader
    # merging them.
    # THE FILE IS NOT WHOLE UNTIL THE STREAM CLOSES, AND `save()` DOES NOT
    # CLOSE IT.
    #
    # `save()` only attaches the sink. The batching pipeline drains on its own
    # thread and the footer is written when the stream is dropped, so a size
    # read straight after `save()` is the size of a half-written file. Measured
    # here, both before the fix: the receipt published 7,427 bytes for a file
    # that settled at 14,555, and `flush()` alone still only reached 14,558 of
    # it. A caller who copied the path the moment `write_rrd` returned would
    # have copied a truncated recording, and `rerun rrd verify` on it is the
    # only thing that would have said so.
    #
    # The context manager closes it. Every measurement below is taken AFTER.
    with rr.RecordingStream(
        "modelmri",
        recording_id=f"{timeline.provenance.dataset}-{timeline.clock.kind}",
    ) as recording:
        recording.save(destination)

        for track in timeline.tracks:
            entity = rrd_entity(track)
            rr.log(
                entity,
                rr.SeriesLines(names=[f"{track.metric} ({track.unit})"]),
                static=True,
                recording=recording,
            )
            for sample in track.samples:
                rr.set_time(
                    "timestep", sequence=int(sample.timestep), recording=recording
                )
                rr.log(entity, rr.Scalars(float(sample.value)), recording=recording)

        if timeline.frame is not None:
            frame = timeline.frame
            rr.set_time("timestep", sequence=int(frame.timestep), recording=recording)
            rr.log(
                f"{TOPIC_PREFIX}episode_{frame.episode}/camera",
                rr.EncodedImage(contents=frame.png, media_type="image/png"),
                recording=recording,
            )

        # WHAT THE FILE SAYS ABOUT ITSELF. `omitted` names what ModelMRI
        # measured that did NOT travel into this file, and it is written INTO
        # the recording rather than only returned: a receipt in a terminal is
        # not attached to the artifact somebody opens a week later.
        rr.log(
            f"{TOPIC_PREFIX}provenance",
            rr.TextDocument(
                _rrd_provenance(timeline, version), media_type="text/markdown"
            ),
            static=True,
            recording=recording,
        )
        recording.flush(timeout_sec=RRD_FLUSH_TIMEOUT_S)

    written = destination.stat().st_size

    return {
        "path": str(destination),
        "container": "rrd",
        "bytes_written": written,
        # The estimate is MCAP's, and this says so rather than quietly
        # comparing a Rerun file against a plan for a different container.
        "bytes_estimated": shape.estimated_file_bytes,
        "estimate_basis": (
            "the MCAP plan. Rerun chunks and compresses on its own terms, so "
            "this ratio describes the payload rather than the container."
        ),
        "estimate_over_actual": round(shape.estimated_file_bytes / written, 3),
        "n_messages": shape.n_messages,
        "n_channels": shape.n_channels,
        "n_metadata_records": shape.n_metadata_records,
        "writer_version": version,
        "plan": shape.to_dict(),
        "means": (
            f"{shape.n_messages:,} measurement(s) written to "
            f"{destination.name} ({fmt.bytes_si(written)}) by rerun-sdk "
            f"{version}. An .rrd is read by the version that wrote it, so that "
            f"version is in the file as well as in this sentence. Every series "
            f"carries its unit in its name, and the file states what ModelMRI "
            f"measured that did NOT travel into it."
        ),
    }


def write(timeline: Timeline, path: str | Path, *, container: str = "mcap") -> dict:
    """One entry point for the route. An unknown container is a 422, not a 500."""
    if container == "mcap":
        return write_mcap(timeline, path)
    if container == "rrd":
        return write_rrd(timeline, path)
    raise BadRequest(
        f"unknown export container {container!r} — expected one of {list(FORMATS)}"
    )
