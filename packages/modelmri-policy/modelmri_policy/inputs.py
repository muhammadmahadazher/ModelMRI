"""Turning a request into the tensors a policy expects, or refusing to.

Every function here is about the same question: is what arrived actually what
the policy consumes? A VLA takes a fixed set of cameras at a fixed resolution,
a state vector of a fixed width, and a task string. Hand it something else and
most of them will not crash — they will broadcast, pad, or silently take the
first channel, and return an action chunk that looks exactly like an answer.

That is the failure this module exists to prevent. A zero-filled action chunk
reads as a policy deciding to hold still; an action computed from a black
image reads as a policy that has decided to do nothing much. Neither is
distinguishable downstream from a real measurement, so neither is allowed to
be produced.

No lerobot here on purpose. This is PNG bytes, numpy shapes and dict keys —
the half that can be tested without a six-gigabyte environment, and the half
whose rules do not change when lerobot's API does.
"""

from __future__ import annotations

import base64
import binascii
import io


class InputError(ValueError):
    """What arrived is not what the policy consumes, and the message says how."""


# A frame arrives as base64 PNG. This is the ceiling on ONE of them, decoded:
# a 1024x1024 RGB frame is 3 MB, so this leaves room for a large camera while
# still refusing something that is not a frame at all.
MAX_FRAME_BYTES = 32 * 1024 * 1024

# The most cameras a single request may carry. Real robots have two or three;
# the ceiling is here so a malformed request cannot ask this process to decode
# a thousand images before anything notices.
MAX_CAMERAS = 8


def decode_frame(payload: object, *, camera: str):
    """One base64 PNG to an HWC uint8 array, or a refusal naming the camera.

    Returns a numpy array. The camera name is in every message because a
    request carries several and "could not decode the image" sends somebody
    looking at the wrong one.
    """
    import numpy as np

    if not isinstance(payload, str) or not payload:
        raise InputError(
            f"camera {camera!r} carried {type(payload).__name__} rather than a "
            f"base64 image, so there is nothing here to look at"
        )
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise InputError(
            f"camera {camera!r} is not valid base64, so no image could be read from it"
        ) from None
    # This branch IS reachable, and only on some interpreters — which is why
    # it is here rather than removed as dead.
    #
    # It was removed once, on the reasoning that the only input decoding to
    # nothing is the empty string and the falsy check above already caught it.
    # True on Python 3.11 and later, where `b64decode("=", validate=True)`
    # raises. On 3.10 the same call RETURNS `b""`, so `"="` reached PIL and
    # came back as `UnidentifiedImageError` — a refusal, but one that blames
    # the image instead of the encoding, on exactly one of the four Pythons
    # this project supports. The 3.10 CI cell caught it.
    #
    # The lesson is the wider one: "this guard cannot fire" is a claim about
    # every interpreter, and it was checked on one.
    if not raw:
        raise InputError(
            f"camera {camera!r} decoded to zero bytes, so there is no image "
            f"data in it at all"
        )
    if len(raw) > MAX_FRAME_BYTES:
        raise InputError(
            f"camera {camera!r} decoded to {len(raw) / 1e6:,.1f} MB, past the "
            f"{MAX_FRAME_BYTES / 1e6:,.0f} MB a single frame may be"
        )

    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - depends on the venv
        raise InputError(
            "this sidecar's environment has no Pillow, so it cannot decode a "
            "frame. Rebuild it with `modelmri policy install --force`."
        ) from None

    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.load()
            # RGB, always, and converted rather than assumed. A PNG with an
            # alpha channel or a palette would otherwise reach the policy as
            # 4 or 1 channels and be silently reinterpreted.
            array = np.asarray(img.convert("RGB"), dtype=np.uint8)
    except Exception as err:
        raise InputError(
            f"camera {camera!r} could not be decoded as an image ({type(err).__name__})"
        ) from None

    if array.ndim != 3 or array.shape[2] != 3:
        raise InputError(
            f"camera {camera!r} decoded to shape {tuple(array.shape)}, which is "
            f"not a colour image"
        )
    return array


def read_frames(body: dict, *, expected: list[str] | None = None) -> dict:
    """Every camera in the request, decoded, with the set itself checked.

    `expected` is the camera list the POLICY was trained with, when it is
    known. A missing camera is refused rather than zero-filled: a VLA trained
    on a wrist view and a top-down view, given only the top-down, does not
    produce a degraded answer — it produces a confident answer to a different
    question.
    """
    frames = body.get("frames")
    if not isinstance(frames, dict) or not frames:
        raise InputError(
            "the request carried no frames, and a vision-language-action "
            "policy cannot act without seeing anything"
        )
    if len(frames) > MAX_CAMERAS:
        raise InputError(
            f"the request carried {len(frames)} cameras, past the "
            f"{MAX_CAMERAS} this accepts"
        )

    out = {name: decode_frame(value, camera=name) for name, value in frames.items()}

    # `is not None`, NOT truthiness. `[]` and `None` are different answers and
    # collapsing them was a real hole: a policy whose config declares no visual
    # features would have had EVERY camera check skipped, so any set of frames
    # under any names reached the forward pass unvalidated. `None` means
    # nobody knows what this policy takes; `[]` means it takes none, and
    # sending it a camera is then as wrong as omitting one.
    if expected is not None:
        missing = [name for name in expected if name not in out]
        extra = [name for name in out if name not in expected]
        if missing:
            raise InputError(
                f"this policy was trained with {len(expected)} camera(s) "
                f"({', '.join(expected)}) and the request is missing "
                f"{', '.join(missing)}. Substituting a blank frame would give "
                f"a confident answer to a different question, so this refuses "
                f"instead."
            )
        if not expected:
            raise InputError(
                f"this policy declares no camera inputs at all, and the "
                f"request carried {', '.join(sorted(out))}. Feeding an image "
                f"to a policy that does not consume one is not a degraded "
                f"measurement, it is a measurement of something else."
            )
        if extra:
            raise InputError(
                f"the request carried camera(s) {', '.join(extra)} that this "
                f"policy does not consume. Feeding them in under another "
                f"name would be guessing at which view is which."
            )
    return out


def read_state(body: dict, *, width: int | None = None):
    """The proprioceptive state vector, checked against the width the policy wants.

    Width matters more than it looks. Joint vectors are ordered, and a
    six-joint arm's reading fed to a seven-joint policy does not fail — it
    shifts every joint by one and returns a plausible chunk for a robot that
    does not exist.
    """
    import numpy as np

    state = body.get("state")
    if state is None:
        # `is not None`, not truthiness: `width=0` means the config published
        # a zero-wide state, which is a strange checkpoint but a stated fact,
        # and `width=None` means nothing was published. Treating 0 as "no
        # requirement" would skip the check on exactly the checkpoint whose
        # shapes are least trustworthy.
        if width is not None:
            raise InputError(
                f"this policy consumes a {width}-wide state vector and the "
                f"request carried none"
            )
        return None
    if not isinstance(state, (list, tuple)):
        raise InputError(
            f"state arrived as {type(state).__name__} rather than a list of numbers"
        )
    for value in state:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InputError(
                f"state contains {type(value).__name__} where a number was expected"
            )
    array = np.asarray(state, dtype=np.float32)
    if not np.isfinite(array).all():
        raise InputError(
            "state contains NaN or infinity, which no policy can act on and "
            "which would propagate silently through the whole chunk"
        )
    if width is not None and array.shape[0] != width:
        raise InputError(
            f"this policy consumes a {width}-wide state vector and the request "
            f"carried {array.shape[0]}. Joint vectors are ORDERED, so a "
            f"mismatched width does not degrade the answer, it shifts every "
            f"joint and returns a chunk for a robot that does not exist."
        )
    return array


def read_instruction(body: dict) -> str:
    """The task string. Empty is allowed and is not the same as absent.

    ROADMAP #50's instruction-swap test deliberately runs a policy with NO
    instruction as one of its arms, and the panel must label that arm "no
    instruction" rather than "the instruction did not matter". So an empty
    string travels as an empty string; it is a condition, not a missing field.
    """
    task = body.get("instruction", "")
    if task is None:
        return ""
    if not isinstance(task, str):
        raise InputError(
            f"instruction arrived as {type(task).__name__} rather than text"
        )
    return task


def read_seed(body: dict) -> int | None:
    """The sampler seed, or None for "do not touch the RNG".

    None and 0 are different requests and must stay so. 0 is a seed; None
    means the caller wants whatever the process's RNG already is, which is the
    honest thing to return when nobody asked for reproducibility.
    """
    seed = body.get("seed")
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise InputError(
            f"seed arrived as {type(seed).__name__} rather than a whole number"
        )
    if not 0 <= seed < 2**31:
        raise InputError(f"seed {seed} is outside the range a sampler accepts")
    return seed
