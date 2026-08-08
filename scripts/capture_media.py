"""Capture the README's screenshots and GIFs from a live ModelMRI.

Screen-recorded demos rot: the UI moves, the recording doesn't, and a year
later the README is showing a product that no longer exists. This drives a
real browser against a real server and regenerates the media, so it can be
re-run on any release and the pictures are never a lie.

    modelmri serve &
    python scripts/capture_media.py --out docs/media

Needs playwright (`pip install playwright && playwright install chromium`) and
ffmpeg on PATH for the GIF encode. Falls back to stills if ffmpeg is missing,
and says so rather than silently shipping nothing.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

URL = "http://127.0.0.1:5900/"
PROMPT = "The Eiffel Tower is located in the city of"
VIEWPORT = {"width": 1280, "height": 820}


def encode_gif(
    webm: Path,
    gif: Path,
    *,
    start: float = 0.0,
    dur: float | None = None,
    fps: int = 10,
    width: int = 800,
) -> bool:
    """webm -> gif via a generated palette. One pass looks like 1998.

    `start`/`dur` trim to the part worth watching. Without them the clip is
    mostly the model loading, which is ten seconds of a still image and the
    difference between a 6 MB GIF and a 1 MB one.
    """
    ff = shutil.which("ffmpeg")
    if not ff:
        return False
    palette = webm.with_suffix(".png")
    vf = f"fps={fps},scale={width}:-1:flags=lanczos"
    trim = ["-ss", f"{start:.2f}"] + (["-t", f"{dur:.2f}"] if dur else [])
    subprocess.run(
        [
            ff,
            "-y",
            *trim,
            "-i",
            str(webm),
            "-vf",
            f"{vf},palettegen=stats_mode=diff",
            str(palette),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            ff,
            "-y",
            *trim,
            "-i",
            str(webm),
            "-i",
            str(palette),
            "-lavfi",
            f"{vf} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3",
            str(gif),
        ],
        check=True,
        capture_output=True,
    )
    palette.unlink(missing_ok=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/media", help="where to write media")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--theme", default="dark", choices=["light", "dark"])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "_raw"
    raw.mkdir(exist_ok=True)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "playwright missing: pip install playwright && playwright install chromium",
            file=sys.stderr,
        )
        return 2

    made: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()

        def page(record: bool):
            ctx = browser.new_context(
                viewport=VIEWPORT,
                device_scale_factor=2,  # retina stills; GIFs downscale from it
                record_video_dir=str(raw) if record else None,
                record_video_size=VIEWPORT if record else None,
                color_scheme=args.theme,
            )
            pg = ctx.new_page()
            pg.goto(args.url, wait_until="networkidle")
            pg.evaluate(
                "t => { document.documentElement.dataset.theme = t;"
                " document.documentElement.style.colorScheme = t; }",
                args.theme,
            )
            return ctx, pg

        def generate(pg) -> None:
            pg.fill("textarea", PROMPT)
            pg.get_by_role("button", name="Generate", exact=True).click()
            pg.wait_for_selector(".attn-scroll .tok", timeout=180_000)
            pg.wait_for_timeout(1200)

        # ---------------------------------------------------- 1. attention
        ctx, pg = page(record=True)
        t0 = time.monotonic()
        generate(pg)
        pg.locator(".panel.attn").scroll_into_view_if_needed()
        sweep_at = time.monotonic() - t0
        pg.wait_for_timeout(600)
        chips = pg.locator(".attn-scroll .tok")
        n = min(chips.count(), 26)
        for i in range(6, n, 2):  # sweep the strip, arcs following
            chips.nth(i).hover()
            pg.wait_for_timeout(160)
        chips.nth(min(14, n - 1)).click()  # pin one
        pg.wait_for_timeout(1400)
        pg.locator(".panel.attn").screenshot(path=str(out / "attention.png"))
        video = pg.video.path() if pg.video else None
        ctx.close()
        if video and encode_gif(
            Path(video), out / "attention.gif", start=max(0.0, sweep_at - 0.4), dur=6.0
        ):
            made.append("attention.gif")
        made.append("attention.png")

        # ---------------------------------------------------- 2. hero
        ctx, pg = page(record=False)
        pg.wait_for_timeout(1500)
        pg.screenshot(
            path=str(out / "hero.png"),
            clip={"x": 0, "y": 0, "width": VIEWPORT["width"], "height": 600},
        )
        ctx.close()
        made.append("hero.png")

        # ---------------------------------------------------- 3. picker
        ctx, pg = page(record=True)
        t0 = time.monotonic()
        pg.click(".model-btn")
        pg.wait_for_selector(".model-row", timeout=30_000)
        pg.wait_for_timeout(1600)
        pg.locator(".sheet").screenshot(path=str(out / "picker.png"))
        pg.keyboard.press("Escape")
        pg.wait_for_timeout(700)
        video = pg.video.path() if pg.video else None
        ctx.close()
        if video and encode_gif(
            Path(video),
            out / "picker.gif",
            start=max(0.0, (time.monotonic() - t0) - 4.5),
            dur=4.0,
        ):
            made.append("picker.gif")
        made.append("picker.png")

        browser.close()

    shutil.rmtree(raw, ignore_errors=True)
    for name in made:
        f = out / name
        print(f"  {name:18} {f.stat().st_size / 1024:7.1f} KiB")
    if not any(m.endswith(".gif") for m in made):
        print(
            "\nno GIFs: ffmpeg not found on PATH (stills were still written)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
