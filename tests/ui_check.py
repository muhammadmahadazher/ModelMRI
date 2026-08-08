"""Browser checks for things only a browser can see.

    modelmri serve
    uv run python tests/ui_check.py [url]

tests/e2e_check.py drives the HTTP API; this drives the page. The distinction
matters — the bug this file was written for lived entirely in a mount effect,
so every backend test passed while the app opened a dataset and decoded video
the moment you loaded it.

Needs Playwright:  uv run playwright install chromium
"""

from __future__ import annotations

import asyncio
import sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5900/"

# Windows consoles default to cp1252, and this script prints the UI's own text
# back at you — one theme-toggle glyph was enough to kill the run with a
# UnicodeEncodeError that looked like a test failure.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Endpoints that cost real memory or real time. None of these may be requested
# before the user asks for something. Measured on a cold server: /api/vla/episodes
# imports pyarrow and opens the dataset, /api/vla/frame imports pyav and decodes
# a video frame — together 396 MB of RSS and about 4.4 seconds.
EXPENSIVE = (
    "/api/vla/episodes",
    "/api/vla/frame",
    "/api/vla/load",
    "/api/vla/analyse",
    "/api/model/load",
    "/api/sae/load",
    "/api/model/prompt",
    # Walks the filesystem, imports a user's Python, runs a forward pass.
    "/api/custom/candidates",
    "/api/custom/load",
    "/api/custom/run",
)

# Every panel that starts inert, and the button it must offer instead. Keyed
# by the panel's own heading class so a second resting panel can't satisfy
# the first one's assertion — which is exactly what happened when the custom
# panel landed above the robot panel and inherited its check.
RESTING = {
    ".h-custom": "Find models here",
    ".h-vla": "Open dataset",
}

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    return ok


async def main() -> int:
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(viewport={"width": 1440, "height": 1000})

        api: list[str] = []
        errors: list[str] = []
        page.on(
            "request",
            lambda r: (
                api.append(r.url.split("?")[0].split("//", 1)[-1].split("/", 1)[1])
                if "/api/" in r.url
                else None
            ),
        )
        page.on(
            "console",
            lambda m: errors.append(m.text) if m.type == "error" else None,
        )

        await page.goto(BASE, wait_until="networkidle")
        await page.wait_for_timeout(2500)

        print("\nnothing loads until asked")
        greedy = [p for p in api if any(e.lstrip("/") in p for e in EXPENSIVE)]
        check(
            "no expensive call fires on page load",
            not greedy,
            f"fired: {greedy}" if greedy else f"{len(api)} cheap status calls only",
        )
        check("console is clean on load", not errors, "; ".join(errors[:3]))

        print("\nthe page is usable in its resting state")
        text = await page.inner_text("body")
        # Ask the page, not the URL. The demo bundle is routinely served from
        # a plain localhost port during verification, and a URL sniff called
        # that a real server and asserted a resting state the demo never has.
        demo = await page.query_selector(".demo-banner") is not None
        if not demo:
            check("model pill says nothing is loaded", "no model loaded" in text)
        for heading, expected in RESTING.items():
            button = await page.query_selector(f".panel:has({heading}) .resting button")
            if not check(f"{heading} panel starts inert", button is not None):
                continue
            label = (await button.inner_text()).strip()
            check(
                f"{heading} button says what it will do",
                expected in label,
                f"{label!r} (wanted {expected!r})",
            )

        print("\nevery button looks like a button")
        # Tailwind's preflight leaves <button> transparent, borderless AND
        # `cursor: default`, so one the author forgot to give a class renders
        # as bare text. It stays clickable, which is why it passes every
        # functional test and fails the only thing that matters.
        #
        # Deliberately chrome-less controls exist here too — trace rows,
        # candidate rows, the theme segments — and all of them set
        # `cursor: pointer`. That is the discriminator, and it was measured by
        # injecting the bug rather than reasoned about: the injected control
        # below must be flagged, or this check is not checking anything.
        result = await page.evaluate(
            """() => {
              const cv = document.createElement('canvas');
              cv.width = cv.height = 1;
              const cx = cv.getContext('2d', { willReadFrequently: true });
              const alpha = (c) => {
                cx.clearRect(0, 0, 1, 1);
                cx.fillStyle = c;
                cx.fillRect(0, 0, 1, 1);
                return cx.getImageData(0, 0, 1, 1).data[3];
              };
              const bare = (b) => {
                const s = getComputedStyle(b);
                if (s.cursor === 'pointer') return false;   // deliberate
                const filled = alpha(s.backgroundColor) > 8;
                const bordered =
                  parseFloat(s.borderTopWidth) > 0 && alpha(s.borderTopColor) > 8;
                const outlined =
                  s.outlineStyle !== 'none' && parseFloat(s.outlineWidth) > 0;
                const underlined = s.textDecorationLine.includes('underline');
                return !filled && !bordered && !outlined && !underlined;
              };

              const canary = document.createElement('button');
              canary.textContent = 'canary';
              document.body.appendChild(canary);
              const sensitive = bare(canary);
              canary.remove();

              const found = [];
              for (const b of document.querySelectorAll('button')) {
                const r = b.getBoundingClientRect();
                if (!r.width || !r.height) continue;
                if (bare(b)) found.push((b.innerText || '(icon)').trim().slice(0, 30));
              }
              return { sensitive, found };
            }"""
        )
        check(
            "the check can still detect an unstyled button",
            result["sensitive"],
            ""
            if result["sensitive"]
            else "an injected classless button was NOT flagged - this check is blind",
        )
        check(
            "no button renders as bare text",
            not result["found"],
            f"unstyled: {result['found']}",
        )

        print("\nthe page does not scroll sideways")
        for width in (1440, 768, 375):
            await page.set_viewport_size({"width": width, "height": 900})
            await page.wait_for_timeout(250)
            overflow = await page.evaluate(
                "() => document.documentElement.scrollWidth"
                " - document.documentElement.clientWidth"
            )
            check(
                f"no horizontal overflow at {width}px", overflow <= 0, f"{overflow}px"
            )

        await browser.close()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
