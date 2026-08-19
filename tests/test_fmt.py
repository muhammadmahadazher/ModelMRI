"""One rule for printing a measured number, and it lives in one place.

The Python half of what `frontend/src/measured.ts` does for the browser, and
it exists because the two disagreed on screen: the knockout panel drew a row
reading `astronaut · 3.0e-5` with the server's own sentence under it saying
`'astronaut' moved it furthest (0.0000)`.
"""

from __future__ import annotations

import pytest

from modelmri import fmt


@pytest.mark.parametrize(
    "value,expected",
    [
        # An exact zero IS the measurement and keeps its fixed places.
        (0.0, "0.0000"),
        (0, "0.0000"),
        # Ordinary values are untouched.
        (0.4271, "0.4271"),
        (1.5, "1.5000"),
        # Nonzero but below what four places can show: escapes rather than
        # rounding away. This is the case that produced "0.0000" beside a row
        # reading 3.0e-5.
        (3.0e-5, "3.0e-05"),
        (-2.5e-6, "-2.5e-06"),
    ],
)
def test_a_measurement_never_rounds_away_to_nothing(value, expected):
    assert fmt.measured(value) == expected


def test_the_decimals_are_the_callers_choice():
    assert fmt.measured(0.5, 2) == "0.50"
    assert fmt.measured(0.001, 2) == "1.0e-03"


def test_a_non_number_says_so_rather_than_printing_one():
    assert fmt.measured(float("nan")) == "not a number"
    assert fmt.measured(float("inf")) == "not a number"


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, "0 bytes"),
        (512, "512 bytes"),
        (8_803_344, "9 MB"),
        (692_645_451, "693 MB"),
        (3_338_069_029, "3.3 GB"),
    ],
)
def test_a_byte_count_keeps_its_significant_digits(n, expected):
    """MEASURED: `hf-internal-testing/tiny-stable-diffusion-torch` is 8.8 MB
    and the status sentence reported "0.0 GB of weights" — in the very line
    that had just been fixed for the `bytes_resident == 0` case and not for
    this one. Zero and "rounds to zero" are different failures that produce
    the same wrong words."""
    assert fmt.bytes_si(n) == expected


def test_the_disk_refusal_names_a_size_it_can_see():
    """`{gb:,.1f} GB` printed a measured 40 MB of free disk as "0.0 GB free" —
    the same token this module uses elsewhere for "could not measure".

    A refusal saying you have nothing free when you have 40 MB, inside the
    sentence explaining why a download stopped, sends somebody to clear a disk
    that is not the problem. The same floor hit the other side: a 4 MB repo
    asked for "0.0 GB".
    """
    from modelmri.capacity import _human

    assert _human(0.004) == "4 MB"
    assert _human(0.04) == "40 MB"
    assert _human(0.999) == "999 MB"
    # A gigabyte and up keeps the GB form: this column is read by comparing
    # its rows, and "4.0 GB" against "1.2 TB" is the comparison.
    assert _human(1.0) == "1.0 GB"
    assert _human(4.0) == "4.0 GB"
    # The TB arm is this function's own — `bytes_si` stops at GB.
    assert _human(1200.0) == "1.2 TB"
