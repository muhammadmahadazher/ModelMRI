"""The README sells half the product if its pictures are one theme.

That is the roadmap's own complaint about D1 — "GIFs in BOTH themes, the
current ones are dark only" — and it was fixed for three images and left for
the rest, which is worse than not starting: a reader on a light GitHub sees
two animations and then a run of dark screenshots, and the dark ones look
like a different tool.

GitHub's mechanism is a URL fragment: `#gh-dark-mode-only` and
`#gh-light-mode-only` on two `<img>` tags, one of which the viewer's theme
hides. It has no fallback and no warning — an image with neither fragment
shows in BOTH themes, which is the state every unconverted screenshot is in.

Three things are checked, and each one had a real instance when this was
written:

  a pair that disagrees   `picker.gif#gh-dark-mode-only` sat beside
                          `light/hero.png#gh-light-mode-only`. Same slot, two
                          different pictures: dark readers were shown the
                          model picker animating and light readers a static
                          hero, under alt text describing the picker.
  a themeless image       `docs/media/patching.gif` was published with no
                          fragment at all, while `docs/media/light/patching
                          .gif` sat on disk unused.
  a missing file          an `<img>` pointing at a path that is not there
                          renders as a broken-image icon on the busiest page
                          this project has.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
MEDIA = ROOT / "docs" / "media"

#: Images that are deliberately one picture for both themes, with the reason.
#: Listed rather than inferred: "this one does not need a light version" is a
#: judgement, and an unlisted themeless image is the bug above.
THEME_NEUTRAL: dict[str, str] = {}

_IMG = re.compile(r'<img\s+src="([^"]+)"[^>]*?alt="([^"]*)"[^>]*>')


@pytest.fixture(scope="module")
def images() -> list[tuple[str, str, str]]:
    """Every local `<img>` in the README as (path, fragment, alt)."""
    text = README.read_text("utf-8")
    out = []
    for src, alt in _IMG.findall(text):
        if not src.startswith("docs/media"):
            continue  # a badge, served from elsewhere
        path, _, fragment = src.partition("#")
        out.append((path, fragment, alt))
    return out


def test_the_readme_has_media_at_all(images):
    """A coverage floor: a regex that stopped matching would pass everything
    below by checking nothing."""
    assert len(images) >= 6, f"only {len(images)} local images found — parser broken?"


def test_every_image_the_readme_points_at_is_on_disk(images):
    missing = sorted({p for p, _, _ in images if not (ROOT / p).is_file()})
    assert not missing, "broken image(s) in the README: " + ", ".join(missing)


def test_every_screenshot_is_published_for_both_themes(images):
    """One picture with no fragment shows in BOTH, which is the unconverted
    state — and it is invisible to anyone reviewing in the theme it was shot
    in."""
    themeless = sorted(
        {p for p, frag, _ in images if not frag and p not in THEME_NEUTRAL}
    )
    assert not themeless, (
        "published to both themes as one picture: "
        + ", ".join(themeless)
        + " — add a `#gh-dark-mode-only` / `#gh-light-mode-only` pair, or list "
        "it in THEME_NEUTRAL with the reason"
    )


def test_a_dark_image_and_its_light_twin_are_the_same_picture(images):
    """Same slot, same subject. `picker.gif` against `light/hero.png` was two
    different pictures under one alt text, so what a reader saw depended on
    their theme."""
    dark = {p: alt for p, frag, alt in images if frag == "gh-dark-mode-only"}
    light = {p: alt for p, frag, alt in images if frag == "gh-light-mode-only"}
    assert dark and light, "no themed pairs at all — the fragments were dropped"

    problems = []
    for path, alt in dark.items():
        twin = MEDIA / "light" / Path(path).name
        want = str(twin.relative_to(ROOT)).replace("\\", "/")
        if want not in light:
            problems.append(
                f"{path} has no light twin of the same name (looked for {want}; "
                f"light images present: {sorted(light)})"
            )
        elif light[want] != alt:
            problems.append(
                f"{path} and {want} carry different alt text, so they are not "
                f"the same picture: {alt!r} vs {light[want]!r}"
            )
    assert not problems, "\n  ".join([""] + problems)


def test_no_light_image_is_orphaned(images):
    """The mirror of the above: a light image with no dark counterpart shows
    for half the readers and nothing for the other half."""
    dark = {Path(p).name for p, frag, _ in images if frag == "gh-dark-mode-only"}
    orphans = sorted(
        p
        for p, frag, _ in images
        if frag == "gh-light-mode-only" and Path(p).name not in dark
    )
    assert not orphans, "light image(s) with no dark counterpart: " + ", ".join(orphans)
