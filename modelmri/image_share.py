"""An image run, shaped for a `.mri`.

A6, and the last unbuilt item in Theme A. Every other measurement this tool
makes could be sent to somebody — a generation, a patching trace, a head
ranking, a robot episode — and the one that is a PICTURE could not. So an
image finding was the only kind that had to be screenshot to be shared, and a
screenshot carries no provenance, no seed, no scheduler and no statement of
what was shrunk on the way out.

This module is only the shaping. `session._image` is the validator and it is
deliberately a separate thing: this side runs on data this process just
measured, and that side runs on a file that arrived from a stranger. Writing
one function for both would mean the checks that matter for the second are
paid for by the first, and — much worse — that a change made for the writer's
convenience silently loosens the reader.

WHAT IS AND IS NOT CARRIED

  carried   the strip, each frame with the step it came from, its emitted
            resolution, whether it was downsampled and from what, and the
            latent RMS it was decoded at; the cross-attention per step with
            the padding boundary and the conditioning width; a detector or
            classifier readout with the KIND of number its scores are.
  not       the pipeline, the latents, or anything that would need this
            machine to open. A `.mri` is read with nothing installed.

THE SEED IS THE POINT OF `None`. A run with no fixed seed is not a run with
seed 0: rerun it and the trajectory differs, and every number downstream stops
comparing. It travels as `None` and the sentence says so, rather than being
filled in with a plausible integer at any layer.
"""

from __future__ import annotations

from typing import Any

# What a readout's `score` column actually IS. A classifier publishes a
# probability over classes and a detector publishes a per-query confidence;
# both render as a number between 0 and 1 and they do not compare. The kind
# travels with the rows so a reader is never invited to compare them.
CLASSIFICATION = "classification"
DETECTION = "detection"


def _reader():
    """The reader module, for its caps.

    Imported lazily and by REFERENCE rather than copied as numbers: the whole
    hazard this guards is the writer and the reader drifting apart, and two
    constants with the same value in two files drift the moment one is tuned.
    Lazy because `session` imports are not free and this module is imported at
    server start.
    """
    from . import session

    return session


def _size(width: Any, height: Any) -> list[int] | None:
    if (
        isinstance(width, int)
        and not isinstance(width, bool)
        and isinstance(height, int)
        and not isinstance(height, bool)
        and width > 0
        and height > 0
    ):
        return [width, height]
    return None


def _provenance(status, kind: str) -> dict:
    """Which checkpoint drew this, in the four fields the reader requires.

    `revision` is read off the status when it carries one and is otherwise the
    empty string, which is a CLAIM — "this checkpoint published none" — and the
    one `session._image` accepts. What it refuses is the field being absent,
    because that cannot be told apart from nobody having looked.
    """
    return {
        "repo": _shared_name(getattr(status, "repo", "") or ""),
        "family": getattr(status, "family", "") or "",
        "architecture": getattr(status, "architecture", "") or "",
        "revision": str(getattr(status, "revision", "") or ""),
        "kind": kind,
    }


def _shared_name(repo: str) -> str:
    """The name, never the path.

    `runtime.export_session` learned this the hard way and the comment there is
    the record: `hf_id` is a Hub id for a Hub model and an ABSOLUTE PATH for one
    loaded out of a local folder, and a `.mri` is the one artefact in this
    project designed to leave the machine. Publishing the raw id shipped
    `C:\\Users\\<their real name>\\...` to whoever the file was sent to.
    """
    from pathlib import Path

    if not repo:
        return ""
    try:
        if Path(repo).exists():
            return Path(repo).name
    except OSError:
        # A malformed path is not a reason to fail a share. The name is
        # metadata, and an unreadable one is better dropped than leaked.
        return Path(repo).name or ""
    return repo


def from_filmstrip(status, strip, *, attention=None) -> dict:
    """A denoising run: the strip, and the cross-attention over it when there is one.

    The two travel together rather than as two shares because they are one run
    — the map is drawn OVER the frames, and a reader given the map alone has a
    heat field with nothing under it.
    """
    frames = []
    budget = 0
    oversized = 0
    for frame in getattr(strip, "frames", []) or []:
        row = frame.to_dict()
        png = row.get("png")
        if not png:
            # A frame with no bytes is dropped rather than carried as a hole.
            # `image_steps` writes `None` here precisely so a decode that
            # produced nothing can be told from one that never ran, and a
            # `.mri` reader has no way to render the difference.
            continue
        emitted = _size(row.get("width"), row.get("height"))
        if emitted is None:
            # The reader refuses a frame with no stated resolution, so a frame
            # that cannot state one is left out here rather than made up.
            continue
        # THE READER'S OWN CAPS, imported rather than restated, so the two
        # cannot drift into a writer that emits files its reader refuses.
        # Without this a high-resolution strip produced a payload over the
        # bound and the share button answered 422 — a button that fails on
        # exactly the run somebody most wanted to send.
        if len(png) > _reader().MAX_IMAGE_FRAME_BYTES or (
            budget + len(png) > _reader().MAX_IMAGE_BYTES_TOTAL
        ):
            oversized += 1
            continue
        budget += len(png)
        keep: dict = {
            "step": row.get("step"),
            "timestep": row.get("timestep"),
            "png": png,
            "size": emitted,
            "downsampled": bool(row.get("downsampled")),
            "latent_rms": row.get("latent_rms"),
        }
        if keep["downsampled"]:
            decoded = _size(row.get("decoded_width"), row.get("decoded_height"))
            if decoded is None:
                # "Shrunk from an unknown size" is not a resolution anybody can
                # put a map back onto, and the reader says so. Rather than ship
                # a frame it will refuse, this drops the claim: the frame goes
                # out at the size it actually is.
                keep["downsampled"] = False
            else:
                keep["decoded_size"] = decoded
        frames.append(keep)

    out: dict = {
        "provenance": _provenance(status, "denoising"),
        "prompt": getattr(strip, "prompt", "") or "",
        "seed": getattr(strip, "seed", None),
        "scheduler": getattr(strip, "scheduler", "") or "",
        "frames": frames,
        "steps_requested": int(getattr(strip, "steps_requested", 0) or 0),
        "steps_run": int(getattr(strip, "steps_run", 0) or 0),
        "decoded_steps": list(getattr(strip, "decoded_steps", []) or []),
        # A choice and a gap, kept apart. One is "we sampled the run", the
        # other is "the pipeline's callback never fired" — and a strip that
        # folded them together would read as eight of fifty either way.
        "skipped_steps": list(getattr(strip, "skipped_steps", []) or []),
        "steps_never_reached": list(getattr(strip, "steps_never_reached", []) or []),
    }
    if attention is not None:
        out["attention"] = _attention(attention)
    out["means"] = _means(out, strip, oversized)
    return out


def _attention(run) -> dict:
    """The cross-attention run, in the reader's shape."""
    row = run.to_dict() if hasattr(run, "to_dict") else dict(run)
    return {
        "tokens": list(row.get("tokens") or []),
        "steps": [
            {
                "step": s.get("step"),
                "timestep": s.get("timestep"),
                "per_token": list(s.get("per_token") or []),
                "blocks": s.get("blocks"),
            }
            for s in (row.get("steps") or [])
        ],
        # Carried, never recomputed downstream: both are claims the measurement
        # made. `padding_from` is why a reader is not told the model is
        # fascinated by `<pad>`, and `columns_unlabelled` is a cap on what can
        # be SHOWN rather than on what was measured.
        "padding_from": int(row.get("padding_from") or 0),
        "conditioning_width": int(row.get("conditioning_width") or 0),
        "columns_unlabelled": int(row.get("columns_unlabelled") or 0),
        "steps_requested": int(row.get("steps_requested") or 0),
        "steps_measured": int(row.get("steps_measured") or 0),
        "resolutions": list(row.get("resolutions") or []),
        "means": str(row.get("means") or ""),
    }


def from_attention(status, run) -> dict:
    """A cross-attention run with no strip beside it.

    Legitimate on its own: `/api/image/attention` captures the maps without
    decoding a single frame, which is most of the cost. The file then carries a
    map over a picture it does not have, and `means` says exactly that rather
    than letting a reader assume the frames were lost.
    """
    out: dict = {
        "provenance": _provenance(status, "cross-attention"),
        "prompt": "",
        "seed": getattr(run, "seed", None),
        "scheduler": "",
        "attention": _attention(run),
    }
    out["means"] = (
        f"Cross-attention over {len(out['attention']['tokens'])} prompt token(s) "
        f"across {out['attention']['steps_measured']} denoising step(s), with no "
        f"decoded frames beside it — capturing the maps does not decode the "
        f"picture, and this file carries what was measured rather than a frame "
        f"nobody rendered. " + _seed_sentence(out["seed"])
    )
    return out


def from_readout(status, prediction, *, picture: str = "") -> dict:
    """What a classifier, a detector or a segmenter said about one picture.

    THE KIND IS READ OFF THE PREDICTION, NOT PASSED IN. `image_cv` decides the
    task from what the model actually RETURNED, and taking it as an argument
    here would let a caller label a detector's output as a classifier's --
    which is precisely the confusion the field exists to prevent.

    And it exists because the three arms publish three different quantities
    that all render as a number between 0 and 1:

      classification   a probability over the class list
      detection        a per-query confidence, which is not a probability over
                       anything and does not sum to one
      segmentation     the FRACTION OF THE MAP a label claims, which is a
                       share of cells rather than a confidence at all

    `session._image` refuses a readout that does not say which, so the three
    can never be read side by side as though they compared.
    """
    payload = (
        prediction.to_dict() if hasattr(prediction, "to_dict") else dict(prediction)
    )
    kind = str(payload.get("task") or "")
    if kind == DETECTION:
        source, score_key = payload.get("boxes"), "score"
    elif kind == CLASSIFICATION:
        source, score_key = payload.get("classes_top"), "probability"
    else:
        # Every segmentation task -- per-pixel, mask-query and promptable.
        # `fraction` is the share of the MAP's cells, not of the image's
        # pixels: the map is coarser, and `image_cv` is explicit that the
        # fraction is the honest one either way.
        source, score_key = payload.get("segments"), "fraction"

    rows = []
    for row in source or []:
        if not isinstance(row, dict):
            continue
        score = row.get(score_key)
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            # The reader refuses a row with no finite score because it would
            # render as a blank bar among measured ones. Dropped here rather
            # than shipped for the reader to reject -- and counted, below, so
            # the drop is reported instead of only applied.
            continue
        keep = {
            # A checkpoint that published no `id2label` gets indices AND a
            # note saying so, which `image_cv` already carries in `means`.
            # `#17` is a worse label than "horse" and a far better one than a
            # borrowed ImageNet name that would read as the model's answer.
            "label": str(row.get("label") or "") or f"#{row.get('index')}",
            "score": float(score),
            "index": row.get("index"),
            "query": row.get("query"),
        }
        box = row.get("box_xyxy")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            keep["box_xyxy"] = [float(v) for v in box]
        rows.append(keep)
    dropped = len(source or []) - len(rows)

    out: dict = {
        "provenance": _provenance(status, kind or "readout"),
        "prompt": "",
        # A readout is one forward pass over one picture: there is no sampling
        # and so no seed. `None` here means "no seed applies", which is the
        # same word this format uses for "no seed was fixed" and the right one
        # for both -- neither is a run you can reproduce by setting 0.
        "seed": None,
        "scheduler": "",
        "readout": {
            "kind": kind or "readout",
            "rows": rows,
            "means": str(payload.get("means") or ""),
        },
    }
    too_big = False
    if picture:
        size = _size(payload.get("width"), payload.get("height"))
        # `image_input` accepts a 32 MB upload and a `.mri` frame is capped at
        # 4 MB, so a large photograph produced a payload the reader refuses
        # and the share button answered 422 on a readout it had just made.
        # Dropped and REPORTED here instead — the file is still worth having
        # without the picture, and the sentence says which half is missing.
        too_big = len(picture) > _reader().MAX_IMAGE_FRAME_BYTES
        if size is not None and not too_big:
            # The picture the reader supplied, carried so the boxes have
            # something to be drawn on -- at the size the MODEL was shown,
            # which is what the box coordinates are in.
            out["frames"] = [
                {
                    "step": 0,
                    "timestep": None,
                    "png": picture,
                    "size": size,
                    "downsampled": False,
                    "latent_rms": None,
                }
            ]
        # No `else`. A picture whose size is unknown is left out rather than
        # carried at a guessed one: boxes drawn over a picture of the wrong
        # resolution land nowhere, and the reader refuses a frame with no
        # stated size for exactly that reason.

    what = {
        DETECTION: "box",
        CLASSIFICATION: "label",
    }.get(kind, "region")
    quantity = {
        DETECTION: (
            "per-query detector confidences, which are not probabilities over "
            "anything and do not sum to one"
        ),
        CLASSIFICATION: "probabilities over the class list",
    }.get(kind, "shares of the map's cells rather than confidences")
    said = (
        f"{len(rows)} {what}(s) from a {kind or 'vision'} readout. The scores "
        f"are {quantity}, which is why the kind travels with them."
    )
    if dropped > 0:
        # Reported, never only applied.
        said += (
            f" {dropped} row(s) were left out of this file because they carried "
            f"no finite score."
        )
    if "frames" not in out and picture:
        # TWO REASONS, TWO SENTENCES, because they have two remedies: one is
        # "this checkpoint did not publish the geometry" and the other is
        # "your photograph is bigger than a shareable file", which the reader
        # can act on by sending a smaller one.
        said += (
            f" The picture is not carried: it is "
            f"{len(picture):,} bytes, above the "
            f"{_reader().MAX_IMAGE_FRAME_BYTES:,} a `.mri` frame holds — the "
            f"readout above is unaffected, and a smaller image would travel "
            f"with it."
            if too_big
            else " The picture is not carried: it did not state the resolution "
            "the model was shown it at, and boxes drawn over the wrong one "
            "land nowhere."
        )
    out["means"] = said
    return out


def _seed_sentence(seed) -> str:
    if seed is None:
        return (
            "NO SEED WAS FIXED, so this trajectory is not repeatable: running "
            "the same prompt again gives a different one. That is why the seed "
            "is absent here rather than written down as 0."
        )
    return f"Seed {seed}, so the run repeats."


def _means(out: dict, strip, oversized: int = 0) -> str:
    frames = out.get("frames") or []
    requested = out.get("steps_requested") or 0
    run = out.get("steps_run") or 0
    # `oversized` is counted separately and subtracted here, so the two
    # reasons a frame can be missing never get folded into one number: one is
    # "there was nothing to carry", the other is "it did not fit", and they
    # send a reader to two different places.
    dropped = len(getattr(strip, "frames", []) or []) - len(frames) - oversized
    parts = [
        f"{len(frames)} decoded frame(s) of a {run or requested}-step run",
    ]
    if out.get("skipped_steps"):
        parts.append(
            f"{len(out['skipped_steps'])} step(s) ran and were not decoded — a "
            f"choice, not a gap"
        )
    if out.get("steps_never_reached"):
        parts.append(
            f"{len(out['steps_never_reached'])} step(s) were selected and never "
            f"arrived — a gap, not a choice"
        )
    if dropped > 0:
        # Reported, never only applied.
        parts.append(
            f"{dropped} frame(s) were left out of this file because they "
            f"carried no bytes or no stated resolution"
        )
    if oversized > 0:
        # A DIFFERENT SENTENCE from the one above, with a different remedy: a
        # frame that did not fit is still on the machine that made it, and
        # decoding fewer or smaller frames gets it into the file.
        parts.append(
            f"{oversized} frame(s) were left out because they are larger than "
            f"a `.mri` carries — decode fewer steps, or a smaller "
            f"`frame_pixels`, to fit them in"
        )
    if any(f.get("downsampled") for f in frames):
        parts.append(
            "some frames were shrunk to fit the file and each says so, and from "
            "what — a map drawn over a silently resized picture is wrong in the "
            "way that looks like a finding"
        )
    return ". ".join(parts) + ". " + _seed_sentence(out.get("seed"))
