"""One picture, turned into the exact tensor a checkpoint was trained on.

`vision_attr` deliberately does no image loading — "what the model is shown is
exactly what you built" — because a module that quietly resized things would
be deciding what the map is about. This is the piece that builds it, and it is
separate for the same reason: every choice here changes the numbers, so every
choice here is stated.

## The image travels inside the request, never as a path

A data URL, the same rule `session.py` states for a robot frame: "a `.mri`
never carries a path or a link — the frame travels inside it or not at all".
A path in a request body is a path on the SERVER's disk, which is somebody
else's machine as often as it is yours, and a browser cannot produce one for a
file the user picked anyway.

## The value range is READ from the model, not inferred from the picture

This is the part worth the module. `vision_attr` fills its occluder in the
image's own value space, and needs to know what that space is. Absent a stated
range it infers one from the image's own extremes and says that it did — which
is honest but weak, because one photograph's darkest pixel is a lower bound on
the model's input range and not the range itself. A picture of a bright sky
never reaches the bottom of it, so "grey" lands somewhere that is not the
midpoint and the fill is a different colour than the one asked for.

The processor knows the real answer. A checkpoint normalising with mean `m`
and standard deviation `s` over a [0, 1] rescale maps the darkest possible
pixel to `(0 - m) / s` and the brightest to `(1 - m) / s`, per channel. Those
are the true attainable extremes, so they are computed and passed rather than
left to be guessed — and where the processor does not publish them, `None` is
returned so `vision_attr` falls back to inferring AND SAYING SO, rather than
this module inventing a range that would look exactly as authoritative.
"""

from __future__ import annotations

import base64
import binascii
import re

from .errors import BadRequest, Refusal

# What a data URL may weigh. Generous enough for a photograph off a phone and
# bounded because it arrives from a request body: base64 is 4/3 of the bytes
# it encodes, so this is roughly a 24 MB image.
MAX_IMAGE_BYTES = 32 * 1024 * 1024

# The largest picture that will be decoded. A decompression bomb is a few
# kilobytes of PNG that expands to gigabytes of pixels — the byte bound above
# does not catch it, because the bound is on the COMPRESSED size.
MAX_PIXELS = 64_000_000

_DATA_URL = re.compile(r"^data:image/([a-zA-Z0-9.+-]+);base64,(.*)$", re.DOTALL)


class BadImage(BadRequest):
    """The picture could not be read, and the message says which part failed."""


def decode(data_url: str):
    """A `data:image/...;base64,...` string as a PIL image in RGB.

    Every refusal names which stage failed — the prefix, the base64, or the
    bytes underneath — because "could not read your image" sends somebody to
    re-export a file whose only problem was how it was pasted.
    """
    if not isinstance(data_url, str) or not data_url:
        raise BadImage(
            "no image was sent. This measures what a model looked at in ONE "
            "picture, so there is no default image worth substituting."
        )
    if len(data_url) > MAX_IMAGE_BYTES:
        raise BadImage(
            f"that image is {len(data_url):,} bytes, above the "
            f"{MAX_IMAGE_BYTES:,} this reads. The bound is on what arrives in "
            f"the request rather than on what the model sees — the checkpoint "
            f"resizes to its own input size anyway, so a smaller upload costs "
            f"the measurement nothing."
        )

    match = _DATA_URL.match(data_url.strip())
    if match is None:
        raise BadImage(
            "that is not an image data URL. It has to start with "
            "`data:image/<type>;base64,` — a path or a link is not read here, "
            "because a path in a request names a file on the server's disk "
            "rather than on yours."
        )

    try:
        # `validate=True` so stray characters are a refusal rather than being
        # silently dropped into a different, shorter image.
        raw = base64.b64decode(match.group(2), validate=True)
    except (binascii.Error, ValueError):
        raise BadImage(
            "the base64 in that data URL will not decode. The prefix was "
            "right, so this is the payload rather than the format."
        ) from None
    if not raw:
        raise BadImage("that data URL carries an empty image.")

    try:
        from PIL import Image
    except ImportError:
        raise Refusal(
            "reading an image needs the `pillow` package, which is not "
            "installed. `pip install 'modelmri[image]'` adds it."
        ) from None

    import io

    try:
        picture = Image.open(io.BytesIO(raw))
        # Pillow is lazy: `open` reads the header only, so a truncated or
        # hostile file fails HERE rather than three functions later inside a
        # sweep that has already been priced and started.
        picture.load()
    except Exception as err:
        # The class, never the text. Pillow's messages carry file paths.
        raise BadImage(
            f"those bytes are not an image this can decode "
            f"({type(err).__name__}). The base64 decoded cleanly, so the "
            f"payload arrived intact and it is the image format that is the "
            f"problem."
        ) from None

    width, height = picture.size
    if width * height > MAX_PIXELS:
        raise BadImage(
            f"that image is {width:,}x{height:,}, which is "
            f"{width * height:,} pixels and above the {MAX_PIXELS:,} this "
            f"decodes. A few kilobytes of compressed file can expand to "
            f"gigabytes of pixels, so the limit is on the decoded size rather "
            f"than the transferred one."
        )

    # RGB, always. A palette or greyscale image reaching a three-channel model
    # is a shape error deep inside a forward pass; a four-channel RGBA one
    # silently makes the alpha a colour the model has opinions about.
    return picture.convert("RGB")


def value_range_of(processor) -> tuple | None:
    """The extremes a preprocessed pixel can actually reach, or `None`.

    `None` is the honest answer when the processor does not publish enough to
    compute it — `vision_attr` then infers a range from the image's own
    extremes and SAYS it inferred it, which is a weaker claim clearly labelled
    as one. Returning a plausible `(0.0, 1.0)` here instead would be the same
    weak claim wearing the authority of a stated one.
    """
    mean = _floats(getattr(processor, "image_mean", None))
    std = _floats(getattr(processor, "image_std", None))
    if not getattr(processor, "do_normalize", True):
        # Rescaled but not normalised: the range is whatever the rescale
        # produced, and `rescale_factor` is what says so.
        if not getattr(processor, "do_rescale", True):
            return None
        factor = getattr(processor, "rescale_factor", None)
        if not isinstance(factor, (int, float)) or isinstance(factor, bool):
            return None
        # 8-bit input, which is what a decoded PIL image is.
        return (0.0, float(255 * factor))
    if not mean or not std:
        return None
    if len(mean) != len(std):
        # A processor whose two lists disagree is one this should not do
        # arithmetic on. Unknown, said as unknown.
        return None
    if any(s == 0 for s in std):
        return None

    # Per channel, the darkest and brightest a [0, 1] pixel can become. The
    # widest pair across channels is the range the fill has to sit inside:
    # a per-channel range would put "grey" at a different value in each
    # channel, which is a colour cast rather than a neutral occluder.
    lows = [(0.0 - m) / s for m, s in zip(mean, std, strict=True)]
    highs = [(1.0 - m) / s for m, s in zip(mean, std, strict=True)]
    lo, hi = min(lows), max(highs)
    if not (hi > lo):
        return None
    return (float(lo), float(hi))


def _floats(value) -> list:
    """A processor attribute as a list of floats, or `[]` if it is not one.

    `image_mean` is a list on most processors and a bare float on a few, and
    a numpy array on others. Anything this cannot read as numbers becomes an
    empty list, which `value_range_of` reports as unknown rather than
    guessing around.
    """
    if value is None:
        return []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    try:
        out = [float(v) for v in value]
    except (TypeError, ValueError):
        return []
    return out


def to_tensor(picture, processor, *, device: str = "", dtype=None):
    """The picture as `[1, C, H, W]`, prepared exactly as the model expects.

    The processor does the resize, the rescale and the normalisation, because
    those three are what the checkpoint was trained with and doing any of them
    here would be doing them differently.

    Returned in FLOAT32 whatever the model's own dtype is. `vision_attr` fills
    occluders by arithmetic on this tensor and compares logits at six decimal
    places; in float16 the fill value itself is rounded and the last three of
    those decimals are quantisation. The model is cast back to its own dtype
    by the forward pass, so this costs a conversion and not a re-load.
    """
    import torch

    try:
        prepared = processor(images=picture, return_tensors="pt")
    except Exception as err:
        raise BadImage(
            f"this checkpoint's own preprocessor could not read that image "
            f"({type(err).__name__}). The picture decoded, so the image is "
            f"intact — it is the preparation the model expects that failed."
        ) from None

    values = prepared.get("pixel_values") if hasattr(prepared, "get") else None
    if values is None:
        raise BadImage(
            "this checkpoint's preprocessor returned no `pixel_values`, so "
            "there is no tensor to occlude. A processor that produces "
            "something else entirely needs a different measurement than this "
            "one."
        )
    if not isinstance(values, torch.Tensor):
        raise BadImage(
            f"this checkpoint's preprocessor returned `pixel_values` as "
            f"{type(values).__name__} rather than a tensor."
        )

    # Some video-capable processors emit [batch, frames, C, H, W]. One frame
    # is one image; more than one is a question this measurement does not
    # answer, and picking the first would be picking silently.
    if values.ndim == 5:
        if int(values.shape[1]) != 1:
            raise BadImage(
                f"this checkpoint's preprocessor produced "
                f"{int(values.shape[1])} frames from one picture. Attribution "
                f"is about a single image, and choosing one of them here would "
                f"be choosing what the map is about by accident."
            )
        values = values[:, 0]

    values = values.to(torch.float32)
    if device:
        values = values.to(device)
    return values
