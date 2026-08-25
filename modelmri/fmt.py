"""Printing a measured number without printing it as zero.

The Python half of what `frontend/src/measured.ts` does for the browser, and
it exists because the two disagreed on screen. MEASURED, in the image-model
panel after the knockout:

    row:   astronaut · 3.0e-5      (the panel, through `measured()`)
    means: 'astronaut' moved it furthest (0.0000).

One quantity, two formatters, and the sentence under the row contradicted it.
`image_attention.py` wrote that line with `{...:,.4f}`, which floors every
distance below 0.00005 — so a word that moved the image reported as having
moved it by nothing, inside the sentence naming it as the word that moved it
furthest.

The rule is the project's own, stated in `measured.ts`: a number that rounds
to zero without being zero is a fabricated measurement, and it does not matter
that the fabrication happened in a format string. An exact zero still prints
as zero, because that one IS the measurement.

WHERE THIS BELONGS AND WHERE IT DOES NOT. It is for quantities that can
legitimately be tiny and still mean something — a KL, a distance, a residual,
a per-step movement. It is NOT for durations, byte counts, percentages built
for display, or a CKA that lives near 1: those have their own units and their
own reasons, and `progress._si_bytes` already covers the byte case.
"""

from __future__ import annotations

import math


def measured(value: float, decimals: int = 4) -> str:
    """`value` at `decimals` places, escaping rather than rounding to zero.

    >>> measured(0.0)
    '0.0000'
    >>> measured(3.0e-5)
    '3.0e-05'
    >>> measured(0.4271)
    '0.4271'
    """
    # `math.isfinite`, which is false for NaN and for both infinities — one
    # call for what this actually means. It was `value != value or value in
    # (inf, -inf)`: the first half is the classic NaN idiom, correct and
    # obscure enough that CodeQL flagged it as a comparison of identical
    # values, and the second half was a second check for what the first was
    # already reaching toward.
    if not math.isfinite(value):
        return "not a number"
    if value == 0:
        return f"{value:.{decimals}f}"
    # The magnitude below which fixed places would render zero.
    if abs(value) < 0.5 * 10**-decimals:
        return f"{value:.1e}"
    return f"{value:,.{decimals}f}"


def measured_value(value: float, decimals: int = 4) -> float:
    """The number `measured()` would print, as a number.

    For the JSON field beside the sentence. `round(x, decimals)` is the
    obvious thing to put there and it has the same defect the string form was
    written to fix, one layer down: a measured gap of 4.8e-07 stores as
    `-0.0`, which reads as "these add up exactly" — a claim the arithmetic did
    not make — and carries a negative zero into the payload for good measure.

    Kept beside `measured` on purpose. A field and the sentence naming it must
    be the same quantity to the same precision, and this module exists because
    they once were not.

    >>> measured_value(0.0)
    0.0
    >>> measured_value(4.768372e-07, 6)
    4.8e-07
    >>> measured_value(-4.768372e-07, 6)
    -4.8e-07
    >>> measured_value(0.12580681, 6)
    0.125807
    """
    if not math.isfinite(value):
        return value
    if value == 0:
        # `0.0`, never `-0.0`. An exact zero IS the measurement, and its sign
        # is an artefact of which subtraction produced it.
        return 0.0
    if abs(value) < 0.5 * 10**-decimals:
        # The same one significant figure `measured` escapes to, parsed back.
        # At this magnitude that is all the precision there is: the quantity
        # is a difference of two floats and is at its own last bit.
        return float(f"{value:.1e}")
    return round(value, decimals)


def bytes_si(n: float) -> str:
    """A byte count in the unit that keeps its significant digits.

    `f"{n / 1e9:,.1f} GB"` alone is what turned a real 4 MB pipeline into
    "0.0 GB of weights" — measured on
    `hf-internal-testing/tiny-stable-diffusion-torch`, in the sentence that
    had just been fixed for the `bytes_resident == 0` case and not for this
    one. Zero and "rounds to zero" are different failures and both produce the
    same wrong words.

    The same rule and the same shape as `progress._si_bytes`, which was
    written first for the download meter; this is its home now that more than
    one place needs it.
    """
    if n >= 1e9:
        return f"{n / 1e9:,.1f} GB"
    if n >= 1e6:
        return f"{n / 1e6:,.0f} MB"
    if n >= 1e3:
        return f"{n / 1e3:,.0f} kB"
    return f"{max(0, int(n)):,} bytes"
