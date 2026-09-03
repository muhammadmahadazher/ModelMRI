# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Muhammad Mahad Azher

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
import base64
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
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
    ap.add_argument(
        "--only",
        default="",
        help=(
            "comma-separated steps: attention, patching, hero, picker, share, "
            "viewer. Default is all of them."
        ),
    )
    ap.add_argument(
        "--viewer-url",
        default="",
        help="a served viewer bundle; captures the zero-install reader too",
    )
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

    wanted = {s.strip() for s in args.only.split(",") if s.strip()} or None
    known = {"attention", "patching", "hero", "picker", "share", "viewer"}
    if wanted and not wanted <= known:
        print(f"unknown step(s): {sorted(wanted - known)}", file=sys.stderr)
        return 2

    made: list[str] = []
    failed: list[str] = []

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
            # "load", not "networkidle". The page polls -- pull progress,
            # telemetry, the accelerator readout -- so the network never goes
            # idle and this timed out at 30 s against a perfectly healthy
            # server. Every capture below already waits on the SELECTOR it
            # needs, which is the real readiness signal anyway.
            pg.goto(args.url, wait_until="load")
            pg.wait_for_selector(".panel", timeout=60_000)
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

        # ONE SLOW STEP MUST NOT COST THE FIVE AFTER IT.
        #
        # MEASURED: a light-theme run died on `wait_for_selector(".patch-grid
        # td")` after its 180 s, and took `hero`, `picker`, `share` and
        # `viewer` down with it — none of which depend on the patching trace
        # in any way. The four files that were missing from
        # `docs/media/light/` had been missing for exactly that reason, so the
        # README shipped dark screenshots to light readers.
        #
        # A step that raises is now reported BY NAME and the run continues,
        # and `--only` re-runs the ones that failed without redoing the ones
        # that did not. A capture that half-succeeds and says which half is
        # far more useful than one that has to be perfect or nothing.
        @contextmanager
        def step(name: str):
            if wanted is not None and name not in wanted:
                print(f"  {name:18} skipped (--only)")
                yield False
                return
            try:
                yield True
            # Broad on purpose: every failure mode here is somebody else's
            # — a timeout, a selector the UI moved, a codec — and the
            # point is to name it and keep going rather than classify it.
            except Exception as err:
                failed.append(name)
                print(
                    f"  {name:18} FAILED: {type(err).__name__}: {err}",
                    file=sys.stderr,
                )

        with step("attention") as go:
            if go:
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
                    Path(video),
                    out / "attention.gif",
                    start=max(0.0, sweep_at - 0.4),
                    dur=6.0,
                ):
                    made.append("attention.gif")
                made.append("attention.png")

        with step("patching") as go:
            if go:
                # ------------------------------------------- 1b. patching
                # The grid is the headline of this release and the README had no
                # picture of it. Recorded rather than stilled because the thing worth
                # showing is the press: rows settling in as each block is pulled, then
                # the three tabs disagreeing about where the fact lives.
                ctx, pg = page(record=True)
                t0 = time.monotonic()
                patch = pg.locator(".panel.patch")
                if patch.count():
                    patch.scroll_into_view_if_needed()
                    pg.wait_for_timeout(400)
                    start_at = time.monotonic() - t0
                    pg.locator(".panel.patch button", has_text="Trace it").click()
                    # The trace is hundreds of forward passes; wait for the grid,
                    # not for a fixed number of seconds, or a slower machine
                    # records a spinner.
                    #
                    # TEN MINUTES, and the number is measured rather than
                    # generous-looking. `POST /api/patch` on Qwen3-1.7B with the
                    # panel's own default prompts took 3m40s on the machine this
                    # was written on — a laptop 4060 with 3.8 GB free. The
                    # previous 180_000 was under that, so the capture had been
                    # timing out on a trace that was working perfectly, and
                    # taking every step after it down with it. A bigger model or
                    # a busier card is the case the headroom is for.
                    pg.wait_for_selector(".patch-grid td", timeout=600_000)
                    pg.wait_for_timeout(1200)
                    patch.screenshot(path=str(out / "patching.png"))
                    for label in ("attention", "MLP", "residual stream"):
                        tab = pg.locator(".patch-tabs button", has_text=label)
                        if tab.count():
                            tab.first.click()
                            pg.wait_for_timeout(900)
                    video = pg.video.path() if pg.video else None
                    ctx.close()
                    if video and encode_gif(
                        Path(video), out / "patching.gif", start=start_at, dur=9.0
                    ):
                        made.append("patching.gif")
                    made.append("patching.png")
                else:
                    ctx.close()
                    print(
                        "  patching panel not mounted — load a model first",
                        file=sys.stderr,
                    )

        with step("hero") as go:
            if go:
                # ---------------------------------------------------- 2. hero
                ctx, pg = page(record=False)
                pg.wait_for_timeout(1500)
                pg.screenshot(
                    path=str(out / "hero.png"),
                    clip={"x": 0, "y": 0, "width": VIEWPORT["width"], "height": 600},
                )
                ctx.close()
                made.append("hero.png")

        with step("picker") as go:
            if go:
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

        with step("share") as go:
            if go:
                # ------------------------------------------- 4. sharing a session
                # The whole point of `.mri`: an analysis leaves the machine, the
                # model does not. Export from the live app, then open the exact
                # bytes back — the same round trip a reader would make.
                ctx, pg = page(record=False)
                generate(pg)
                pg.locator(".panel.attn").scroll_into_view_if_needed()
                pg.get_by_role("button", name="Share this view").click()
                pg.wait_for_timeout(300)
                # `.share-row`, not `.panel.attn .row`. That panel grew three
                # more control rows — anchors, gradients, the head dials — and
                # the old selector went from one match to three, so the run
                # died in strict mode. That is the media rotting as the UI
                # moves, which is the exact failure this script exists to
                # prevent; `.share-row` is the control's own class and moves
                # with it. Same for the note field: `.share-note` is a shared
                # input style now, not a unique element.
                row = pg.locator(".panel.attn .share-row")
                row.locator(".share-note").fill("L8 H3 copies the subject token")
                pg.wait_for_timeout(500)
                row.screenshot(path=str(out / "share.png"))
                made.append("share.png")

                blob = pg.evaluate(
                    """async () => {
                      const r = await fetch('/api/session/export?layer=8&head=3&note='
                        + encodeURIComponent('L8 H3 copies the subject token'));
                      const buf = new Uint8Array(await r.arrayBuffer());
                      let s = ''; for (const b of buf) s += String.fromCharCode(b);
                      return btoa(s);
                    }"""
                )
                (out / "_session.mri").write_bytes(base64.b64decode(blob))

                pg.evaluate(
                    """async (b64) => {
                      const bin = atob(b64);
                      const buf = new Uint8Array(bin.length);
                      for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
                      const file = new File([buf], 'shared.mri');
                      const input = document.querySelector('input[type=file][accept=".mri"]');
                      const dt = new DataTransfer(); dt.items.add(file);
                      input.files = dt.files;
                      input.dispatchEvent(new Event('change', {bubbles: true}));
                      await new Promise(r => setTimeout(r, 1200));
                    }""",
                    blob,
                )
                pg.wait_for_selector(".panel.replay", timeout=15_000)
                pg.locator(".panel.replay").scroll_into_view_if_needed()
                pg.wait_for_timeout(600)
                pg.locator(".panel.replay").screenshot(path=str(out / "session.png"))
                made.append("session.png")
                ctx.close()

        with step("viewer") as go:
            if go:
                # THE `.mri` IS AN ARTEFACT, NOT A VARIABLE. `share` exports
                # one and writes it here; this step opens it. They used to
                # share a local `blob`, which was invisible while the two
                # always ran together and became `UnboundLocalError` the
                # moment `--only viewer` was possible. Reading the file back
                # is also the more honest test: it is the exact bytes a
                # recipient would be sent.
                mri = out / "_session.mri"
                if not mri.is_file():
                    raise FileNotFoundError(
                        f"{mri} is not there — run the `share` step first "
                        f"(it exports the .mri this opens), or pass "
                        f"--only share,viewer"
                    )
                shared = base64.b64encode(mri.read_bytes()).decode("ascii")
                # ---------------------------------------- 5. the zero-install viewer
                # Only when a viewer is being served (see --viewer-url), because it
                # is a separate build. Same .mri as above, so the two pictures are
                # demonstrably the same analysis on two very different machines.
                if args.viewer_url:
                    ctx = browser.new_context(
                        viewport=VIEWPORT,
                        device_scale_factor=2,
                        color_scheme=args.theme,
                    )
                    pg = ctx.new_page()
                    pg.goto(args.viewer_url, wait_until="load")
                    pg.evaluate(
                        "t => { document.documentElement.dataset.theme = t;"
                        " document.documentElement.style.colorScheme = t; }",
                        args.theme,
                    )
                    pg.wait_for_timeout(900)
                    pg.screenshot(
                        path=str(out / "viewer-empty.png"),
                        clip={
                            "x": 0,
                            "y": 0,
                            "width": VIEWPORT["width"],
                            "height": 700,
                        },
                    )
                    made.append("viewer-empty.png")

                    pg.evaluate(
                        """async (b64) => {
                          const bin = atob(b64);
                          const buf = new Uint8Array(bin.length);
                          for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
                          const file = new File([buf], 'shared.mri');
                          const input = document.querySelector('input[type=file][accept=".mri"]');
                          const dt = new DataTransfer(); dt.items.add(file);
                          input.files = dt.files;
                          input.dispatchEvent(new Event('change', {bubbles: true}));
                          await new Promise(r => setTimeout(r, 1400));
                        }""",
                        shared,
                    )
                    pg.wait_for_selector(".panel.attn", timeout=20_000)
                    pg.wait_for_timeout(900)
                    chips = pg.locator(".attn-scroll .tok")
                    if chips.count() > 12:
                        chips.nth(11).click()
                        pg.wait_for_timeout(900)
                    pg.screenshot(path=str(out / "viewer.png"), full_page=False)
                    made.append("viewer.png")
                    ctx.close()

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
