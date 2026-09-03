# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

"""Every image family the server can name belongs to a section, or is exempt.

`ImagePanel` decides which of its two sections owns a loaded checkpoint by
FAMILY, not by capability — the comment above `OWNED_FAMILIES` explains why: a
class-conditioned DiT supports none of the four diffusion measurements, and
routing by capability made it belong to no section at all, so the panel showed
its resting sketch with 3.3 GB of that model resident. "I loaded it and the
panel went blank" is worse than a refusal.

That constant then misspelled two of the four families it names. The server
emits `detection` and `segmentation`; the panel listed "detector" and
"segmenter", which appear nowhere in `imaging.py`. So a detector reintroduced
the exact failure the constant was written to prevent.

A list of names re-typed on the other side of a wire is the defect. This is the
cheap half of the fix: the names are still typed twice, but they can no longer
disagree without a test going red.
"""

from __future__ import annotations

import re
from pathlib import Path

from modelmri import imaging

PANEL = Path(__file__).resolve().parents[1] / "frontend" / "src" / "ImagePanel.tsx"

#: Families no section claims, on purpose.
#:
#: `clip` is an image-text embedding model: neither section measures one, and
#: inventing a home for it would offer controls that cannot run. `unknown` is
#: the family for an architecture `imaging.detect` could not name, and routing
#: a guess into a section is the failure the UNKNOWN member exists to avoid.
DELIBERATELY_HOMELESS = {imaging.CLIP, imaging.UNKNOWN}


def _owned_families() -> dict[str, list[str]]:
    """`OWNED_FAMILIES` as the panel actually declares it."""
    src = PANEL.read_text("utf-8", errors="replace")
    block = re.search(
        r"const OWNED_FAMILIES: Record<ImageKind, readonly string\[\]> = \{(.*?)\};",
        src,
        re.S,
    )
    assert block, "OWNED_FAMILIES is not where this test expects it"
    out: dict[str, list[str]] = {}
    for kind, names in re.findall(r"(\w+):\s*\[([^\]]*)\]", block.group(1)):
        out[kind] = re.findall(r'"([^"]+)"', names)
    assert out, "parsed no sections out of OWNED_FAMILIES"
    return out


def _server_families() -> set[str]:
    """Every family constant `imaging` defines, read off the module.

    Enumerated rather than listed here, so a family added to `imaging.py`
    without a section fails this test instead of silently going homeless.
    """
    return {
        value
        for name, value in vars(imaging).items()
        if name.isupper() and isinstance(value, str) and value in _known_labels()
    }


def _known_labels() -> set[str]:
    """The families that have a prose label, which is every real one."""
    return set(imaging._FAMILY_LABEL)


def test_the_panel_names_families_the_server_actually_emits():
    """The two that were wrong: "detector" and "segmenter"."""
    claimed = {n for names in _owned_families().values() for n in names}
    unknown = claimed - _server_families()
    assert not unknown, (
        f"ImagePanel claims families the server never emits: {sorted(unknown)}. "
        f"The server's names are {sorted(_server_families())}."
    )


def test_every_family_belongs_to_one_section_or_is_deliberately_homeless():
    """A family nobody claims is a checkpoint that loads to a blank panel."""
    owned = _owned_families()
    claimed = {n for names in owned.values() for n in names}
    homeless = _server_families() - claimed - DELIBERATELY_HOMELESS
    assert not homeless, (
        f"these families load to no section, so the panel shows its resting "
        f"sketch with the model resident: {sorted(homeless)}"
    )


def test_no_family_is_claimed_by_both_sections():
    """Two sections offering the same checkpoint is two homes for one model,
    and the controls under them measure different things."""
    owned = _owned_families()
    both = set(owned.get("diffusion", [])) & set(owned.get("vision", []))
    assert not both, f"claimed by both sections: {sorted(both)}"


def test_the_exempt_list_still_names_real_families():
    """An exemption for a family that no longer exists is dead weight that
    would silently excuse a future family of the same name."""
    stale = DELIBERATELY_HOMELESS - _server_families()
    assert not stale, f"exempt but no longer a family: {sorted(stale)}"
