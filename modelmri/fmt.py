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


def measured(value: float, decimals: int = 4) -> str:
    """`value` at `decimals` places, escaping rather than rounding to zero.

    >>> measured(0.0)
    '0.0000'
    >>> measured(3.0e-5)
    '3.0e-05'
    >>> measured(0.4271)
    '0.4271'
    """
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        return "not a number"
    if value == 0:
        return f"{value:.{decimals}f}"
    # The magnitude below which fixed places would render zero.
    if abs(value) < 0.5 * 10**-decimals:
        return f"{value:.1e}"
    return f"{value:,.{decimals}f}"
